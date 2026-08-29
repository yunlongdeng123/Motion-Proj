"""Model the all-horizon event through a density for the maximum future cost."""

from __future__ import annotations
import argparse,json,math,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _mixture_nll,_predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload


class JointMaxCostDensity(nn.Module):
 def __init__(self,input_count,components,hidden):
  super().__init__();layers=[];width=input_count
  for h in hidden:layers.extend((nn.Linear(width,int(h)),nn.SiLU()));width=int(h)
  layers.append(nn.Linear(width,3*components));self.net=nn.Sequential(*layers);self.components=components
 def forward(self,x):
  logits,means,raw_scales=self.net(x).chunk(3,1);return logits,means,.05+functional.softplus(raw_scales)


@torch.no_grad()
def _max_cdf(model,features,budgets):
 logits,means,scales=model(features);threshold=torch.from_numpy(np.log1p(budgets).astype(np.float32)).cuda();standardized=(threshold[None,:,None]-means[:,None])/scales[:,None];component=.5*(1+torch.erf(standardized/math.sqrt(2)));return torch.sum(functional.softmax(logits,1)[:,None]*component,2).cpu().numpy()


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True
 ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 marginal_density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);raw=np.concatenate((scores,clearances),1);split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;mean=raw[train].mean(0);scale=raw[train].std(0).clip(1e-6);features=torch.from_numpy(((raw-mean)/scale).astype(np.float32)).cuda();target=torch.from_numpy(np.log1p(costs.max(1)).astype(np.float32)).cuda();train_idx=torch.from_numpy(np.flatnonzero(train)).cuda();m=c['model'];model=JointMaxCostDensity(8,int(m['component_count']),m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];logits,means,scales=model(features[idx]);loss=_mixture_nll(logits,means,scales,target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P210 joint max-cost density step={step+1} nll={last:.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);dev_idx=torch.from_numpy(np.flatnonzero(dev)).cuda();candidate=_max_cdf(model.eval(),features[dev_idx],budgets);norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(marginal_density,scores[dev,i],np.full(dev.sum(),h[i],np.float32),clearances[dev,i],budgets,norms) for i in range(4)],1);baseline=_joint_probabilities(copula,features.new_tensor(((raw[dev]-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)),torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c['evaluation']['monte_carlo_samples']),int(c['seed']));truth=np.all(costs[dev,:,None]<=budgets[None,None,:],1);cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((baseline-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(baseline.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'integrated_Brier_strictly_better_than_P199':cb<pb,'calibration_error_noninferior_to_P199':cr>=float(c['decision']['minimum_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'feature_mean':mean,'feature_scale':scale,'component_count':m['component_count'],'hidden_dimensions':m['hidden_dimensions'],'horizons_seconds':h},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int(train.sum()),'final_log_max_cost_nll':last},'development':{'trajectory_count':int(dev.sum()),'max_density_integrated_brier':cb,'P199_integrated_brier':pb,'Brier_reduction_vs_P199':br,'max_density_mean_absolute_reliability_error':ce,'P199_mean_absolute_reliability_error':pe,'calibration_error_reduction_vs_P199':cr},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
