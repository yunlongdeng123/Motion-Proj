"""Ablate P199 instance conditioning with a trained global Gaussian copula."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload,_variable_cdf


class ConstantCopula(nn.Module):
 def __init__(self):super().__init__();self.raw=nn.Parameter(torch.zeros(10))
 def correlation_cholesky(self,features):
  lower=torch.zeros((4,4),device=features.device);cursor=0
  for row in range(4):
   for col in range(row+1):
    value=self.raw[cursor];lower[row,col]=functional.softplus(value)+.2 if row==col else torch.tanh(value);cursor+=1
  cov=lower@lower.T;scale=torch.sqrt(torch.diagonal(cov).clamp_min(1e-8));corr=cov/(scale[:,None]*scale[None,:]);chol=torch.linalg.cholesky(corr+1e-4*torch.eye(4,device=features.device));return chol[None].expand(len(features),-1,-1)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats()
 ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');conditional=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();conditional.load_state_dict(p199['model_state_dict']);conditional.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
 h=np.asarray(c['horizons_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);norms=tuple(fd['norms']);clip=float(c['evaluation']['probability_clip']);u=np.stack([_variable_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],costs[:,i],norms) for i in range(4)],1);normal=torch.distributions.Normal(torch.tensor(0.),torch.tensor(1.));z=normal.icdf(torch.from_numpy(np.clip(u,clip,1-clip))).float().cuda();split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;indices=torch.from_numpy(np.flatnonzero(train)).cuda();dummy=torch.ones((len(scores),1),device='cuda');model=ConstantCopula().cuda();opt=torch.optim.Adam(model.parameters(),lr=float(c['training']['learning_rate']));last=0.
 for step in range(int(c['training']['steps'])):
  idx=indices[torch.randint(len(indices),(int(c['training']['batch_size']),),device='cuda')];chol=model.correlation_cholesky(dummy[idx]);target=z[idx];solved=torch.cholesky_solve(target[:,:,None],chol).squeeze(2);loss=.5*(2*torch.log(torch.diagonal(chol,dim1=1,dim2=2)).sum(1)+(target*solved).sum(1)).mean();opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P206 constant copula step={step+1} nll={last:.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);dev_idx=torch.from_numpy(np.flatnonzero(dev)).cuda();mc=int(c['evaluation']['monte_carlo_samples']);candidate=_joint_probabilities(model,dummy[dev_idx],torch.from_numpy(marginal[dev].astype(np.float32)).cuda(),mc,int(c['seed']));features=torch.from_numpy(((np.concatenate((scores,clearances),1)-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();control=_joint_probabilities(conditional,features[dev_idx],torch.from_numpy(marginal[dev].astype(np.float32)).cuda(),mc,int(c['seed']));truth=np.all(costs[dev,:,None]<=budgets[None,None,:],1);cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((control-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(control.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'integrated_Brier_strictly_better_than_P199':cb<pb,'minimum_calibration_error_reduction_vs_P199':cr>=float(c['decision']['minimum_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'base_marginal':c['frozen_p182']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'final_copula_nll_without_marginal_constant':last},'development':{'constant_integrated_brier':cb,'P199_integrated_brier':pb,'Brier_reduction_vs_P199':br,'constant_mean_absolute_reliability_error':ce,'P199_mean_absolute_reliability_error':pe,'calibration_error_reduction_vs_P199':cr},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
