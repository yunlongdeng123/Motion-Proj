"""Distill P203(P199) Monte Carlo curves into a monotone single-pass student."""

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


class MonotoneCurveStudent(nn.Module):
 def __init__(self,widths):
  super().__init__();layers=[];width=36
  for h in widths:layers.extend((nn.Linear(width,int(h)),nn.SiLU()));width=int(h)
  layers.append(nn.Linear(width,8));self.network=nn.Sequential(*layers)
 def forward(self,x):return functional.softmax(self.network(x),1).cumsum(1)[:,:7]


def _dataset(path,members,ens,density,fd,p199,copula,calibrator,h,budgets,floor,samples,seed):
 with np.load(path,allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,floor),h);norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);raw=np.concatenate((scores,clearances),1);base=((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32);before=time.monotonic();joint=_joint_probabilities(copula,torch.from_numpy(base).cuda(),torch.from_numpy(marginal.astype(np.float32)).cuda(),samples,seed);teacher_seconds=time.monotonic()-before;teacher=calibrator(torch.from_numpy(joint).cuda()).detach().cpu().numpy();feature=np.concatenate((base,marginal.reshape(len(base),-1)),1).astype(np.float32);truth=np.all(costs[:,:,None]<=budgets[None,None,:],1).astype(np.float32);return feature,teacher,truth,scenes,teacher_seconds


def _evaluate(student,feature,teacher,truth,index=None):
 if index is not None:feature=feature[index];teacher=teacher[index];truth=truth[index]
 torch.cuda.synchronize();before=time.monotonic();candidate=student(torch.from_numpy(feature).cuda()).detach();torch.cuda.synchronize();student_seconds=time.monotonic()-before;candidate=candidate.cpu().numpy();cb=float(np.mean((candidate-truth)**2));tb=float(np.mean((teacher-truth)**2));cc=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));tc=float(np.mean(np.abs(teacher.mean(0)-truth.mean(0))));return {'trajectory_count':int(len(feature)),'student_teacher_probability_MAE':float(np.mean(np.abs(candidate-teacher))),'student_integrated_brier':cb,'teacher_integrated_brier':tb,'relative_Brier_degradation_vs_teacher':(cb-tb)/tb,'student_calibration_error':cc,'teacher_calibration_error':tc,'absolute_calibration_error_increase_vs_teacher':cc-tc,'student_forward_seconds':student_seconds}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);floor=float(c['boundary_state_cost']['clearance_floor_m']);samples=int(c['teacher']['monte_carlo_samples']);seed=int(c['seed']);common=(members,ens,density,fd,p199,copula,calibrator,h,budgets,floor,samples,seed);sf,st,sy,scenes,ss=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);f183,t183,y183,_,s183=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);f201,t201,y201,_,s201=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();target=torch.from_numpy(st).cuda();m=c['student'];student=MonotoneCurveStudent(m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];loss=functional.mse_loss(student(x[idx]),target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P227 monotone curve distillation step={step+1} mse={last:.7f}',flush=True)
 source=_evaluate(student,sf,st,sy,dev);r183=_evaluate(student,f183,t183,y183);r201=_evaluate(student,f201,t201,y201);decision=c['decision'];checks={'P201_teacher_probability_fidelity':r201['student_teacher_probability_MAE']<=float(decision['maximum_P201_teacher_probability_MAE']),'P201_probability_quality_noninferior':r201['relative_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_P201_Brier_degradation']) and r201['absolute_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'input_dimension':36,'output_budget_count':7},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_teacher_probability_mse':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
