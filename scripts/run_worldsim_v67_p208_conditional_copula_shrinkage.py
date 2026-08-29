"""Conditionally shrink frozen P199 dependence toward the independence copula."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload,_variable_cdf


class ConditionalShrinkage(nn.Module):
 def __init__(self,initial_logit):
  super().__init__();self.linear=nn.Linear(8,1);nn.init.zeros_(self.linear.weight);nn.init.constant_(self.linear.bias,float(initial_logit))
 def forward(self,x):return torch.sigmoid(self.linear(x)).squeeze(1)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats()
 ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);raw=np.concatenate((scores,clearances),1);features=torch.from_numpy(((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();norms=tuple(fd['norms']);clip=float(c['evaluation']['probability_clip']);u=np.stack([_variable_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],costs[:,i],norms) for i in range(4)],1);normal=torch.distributions.Normal(torch.tensor(0.),torch.tensor(1.));z=normal.icdf(torch.from_numpy(np.clip(u,clip,1-clip))).float().cuda();split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;indices=torch.from_numpy(np.flatnonzero(train)).cuda();t=c['training'];gate=ConditionalShrinkage(t['initial_P199_logit']).cuda();opt=torch.optim.AdamW(gate.parameters(),lr=float(t['learning_rate']),weight_decay=float(t['weight_decay']));last=0.
 for step in range(int(t['steps'])):
  idx=indices[torch.randint(len(indices),(int(t['batch_size']),),device='cuda')]
  with torch.no_grad():
   chol=copula.correlation_cholesky(features[idx]);target=z[idx];solved=torch.cholesky_solve(target[:,:,None],chol).squeeze(2);log_copula=-.5*(2*torch.log(torch.diagonal(chol,dim1=1,dim2=2)).sum(1)+(target*solved).sum(1)-target.square().sum(1))
  weight=gate(features[idx]).clamp(1e-5,1-1e-5);loss=-torch.logaddexp(torch.log(weight)+log_copula,torch.log1p(-weight)).mean();opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P208 conditional shrinkage step={step+1} mixture_nll={last:.6f} mean_weight={float(weight.mean()):.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);dev_idx=torch.from_numpy(np.flatnonzero(dev)).cuda();marginal_tensor=torch.from_numpy(marginal[dev].astype(np.float32)).cuda();p199_prob=_joint_probabilities(copula,features[dev_idx],marginal_tensor,int(c['evaluation']['monte_carlo_samples']),int(c['seed']));independent=np.prod(marginal[dev],axis=1);weights=gate(features[dev_idx]).detach().cpu().numpy();candidate=weights[:,None]*p199_prob+(1-weights[:,None])*independent;truth=np.all(costs[dev,:,None]<=budgets[None,None,:],1);cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((p199_prob-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(p199_prob.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'integrated_Brier_strictly_better_than_P199':cb<pb,'calibration_error_noninferior_to_P199':cr>=float(c['decision']['minimum_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':gate.state_dict(),'feature_mean':p199['feature_mean'],'feature_scale':p199['feature_scale'],'components':['P199','independence']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'final_mixture_copula_nll':last},'development':{'shrinkage_integrated_brier':cb,'P199_integrated_brier':pb,'Brier_reduction_vs_P199':br,'shrinkage_mean_absolute_reliability_error':ce,'P199_mean_absolute_reliability_error':pe,'calibration_error_reduction_vs_P199':cr,'mean_P199_weight':float(weights.mean()),'minimum_P199_weight':float(weights.min()),'maximum_P199_weight':float(weights.max())},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
