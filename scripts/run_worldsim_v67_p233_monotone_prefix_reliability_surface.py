"""Compile four horizon-prefix reliability curves into a two-axis monotone surface."""

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
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration


class MonotonePrefixSurface(nn.Module):
 def __init__(self,widths):
  super().__init__();layers=[];width=36
  for hidden in widths:layers.extend((nn.Linear(width,int(hidden)),nn.SiLU()));width=int(hidden)
  layers.append(nn.Linear(width,29));self.network=nn.Sequential(*layers)
 def forward(self,feature):
  raw=self.network(feature);base=functional.softmax(raw[:,:8],1).cumsum(1)[:,:7];transition=raw[:,8:].reshape(-1,3,7);start=transition[:,:,:1];increments=functional.softplus(transition[:,:,1:]);logit=start+torch.cat((torch.zeros_like(start),torch.cumsum(increments,2)),2);retention=torch.sigmoid(logit);curves=[base]
  for index in range(3):curves.append(curves[-1]*retention[:,index])
  return torch.stack(curves,1)


def _dataset(path,members,ensemble,density,fd,p199,copula,calibrator,h,budgets,floor,samples,seed,ignored):
 with np.load(path,allow_pickle=False) as loaded:arrays={name:loaded[name] for name in loaded.files}
 scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ensemble,floor),h);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,tuple(fd['norms'])) for i in range(4)],1);raw=np.concatenate((scores,clearances),1);base=((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32);teachers=[];teacher_seconds=0.
 for prefix in range(4):
  local=marginal.copy();local[:,prefix+1:,:]=ignored;before=time.monotonic();joint=_joint_probabilities(copula,torch.from_numpy(base).cuda(),torch.from_numpy(local.astype(np.float32)).cuda(),samples,seed);teacher_seconds+=time.monotonic()-before
  if prefix==3:joint=calibrator(torch.from_numpy(joint).cuda()).detach().cpu().numpy()
  teachers.append(joint)
 feature=np.concatenate((base,marginal.reshape(len(base),-1)),1).astype(np.float32);truth=np.stack([np.all(costs[:,:prefix+1,None]<=budgets[None,None,:],1) for prefix in range(4)],1).astype(np.float32);return feature,np.stack(teachers,1).astype(np.float32),truth,scenes,teacher_seconds


def _evaluate(model,feature,teacher,truth,index=None):
 if index is not None:feature=feature[index];teacher=teacher[index];truth=truth[index]
 torch.cuda.synchronize();before=time.monotonic();candidate=model(torch.from_numpy(feature).cuda()).detach();torch.cuda.synchronize();forward=time.monotonic()-before;candidate=candidate.cpu().numpy();cb=float(np.mean((candidate-truth)**2));tb=float(np.mean((teacher-truth)**2));cc=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));tc=float(np.mean(np.abs(teacher.mean(0)-truth.mean(0))));return {'trajectory_count':int(len(feature)),'surface_teacher_probability_MAE':float(np.mean(np.abs(candidate-teacher))),'final_curve_teacher_probability_MAE':float(np.mean(np.abs(candidate[:,-1]-teacher[:,-1]))),'student_surface_integrated_brier':cb,'teacher_surface_integrated_brier':tb,'relative_surface_Brier_degradation_vs_teacher':(cb-tb)/tb,'student_surface_calibration_error':cc,'teacher_surface_calibration_error':tc,'absolute_surface_calibration_error_increase_vs_teacher':cc-tc,'budget_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'horizon_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h,budgets,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,st,sy,scenes,ss=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);f183,t183,y183,_,s183=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);f201,t201,y201,_,s201=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();target=torch.from_numpy(st).cuda();m=c['student'];model=MonotonePrefixSurface(m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];loss=functional.mse_loss(model(x[idx]),target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P233 monotone prefix surface step={step+1} mse={last:.7f}',flush=True)
 source=_evaluate(model,sf,st,sy,dev);r183=_evaluate(model,f183,t183,y183);r201=_evaluate(model,f201,t201,y201);decision=c['decision'];checks={'P201_surface_teacher_fidelity':r201['surface_teacher_probability_MAE']<=float(decision['maximum_P201_surface_teacher_probability_MAE']),'P201_final_curve_teacher_fidelity':r201['final_curve_teacher_probability_MAE']<=float(decision['maximum_P201_final_curve_teacher_probability_MAE']),'P201_surface_quality_noninferior':r201['relative_surface_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_P201_surface_Brier_degradation']) and r201['absolute_surface_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_surface_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'input_dimension':36,'surface_shape':[4,7]},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_teacher_surface_mse':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
