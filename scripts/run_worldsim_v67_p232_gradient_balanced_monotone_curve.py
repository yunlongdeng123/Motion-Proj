"""Balance teacher and outcome gradients without a loss-weight sweep."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p227_monotone_reliability_curve_distillation import MonotoneCurveStudent,_dataset,_evaluate


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h,budgets,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']));sf,st,sy,scenes,ss=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);f183,t183,y183,_,s183=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);f201,t201,y201,_,s201=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();teacher=torch.from_numpy(st).cuda();truth=torch.from_numpy(sy).cuda();m=c['student'];student=MonotoneCurveStudent(m['hidden_dimensions']).cuda();parameters=list(student.parameters());opt=torch.optim.AdamW(parameters,lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last_teacher=last_truth=0.;conflicts=0
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];prediction=student(x[idx]);teacher_loss=functional.mse_loss(prediction,teacher[idx]);truth_loss=functional.mse_loss(prediction,truth[idx]);teacher_gradient=torch.autograd.grad(teacher_loss,parameters,retain_graph=True);truth_gradient=torch.autograd.grad(truth_loss,parameters);dot=sum((left*right).sum() for left,right in zip(teacher_gradient,truth_gradient));teacher_norm=sum((value*value).sum() for value in teacher_gradient).clamp_min(1e-20);adjusted=list(truth_gradient)
  if dot<0:
   adjusted=[right-dot/teacher_norm*left for left,right in zip(teacher_gradient,truth_gradient)];conflicts+=1
  task_norm=sum((value*value).sum() for value in adjusted).clamp_min(1e-20);scale=torch.sqrt(teacher_norm/task_norm);opt.zero_grad(set_to_none=True)
  for parameter,left,right in zip(parameters,teacher_gradient,adjusted):parameter.grad=left+scale*right
  opt.step();last_teacher=float(teacher_loss.detach());last_truth=float(truth_loss.detach())
  if step%500==0:print(f'P232 gradient-balanced step={step+1} teacher={last_teacher:.7f} truth={last_truth:.7f} conflicts={conflicts}',flush=True)
 source=_evaluate(student,sf,st,sy,dev);r183=_evaluate(student,f183,t183,y183);r201=_evaluate(student,f201,t201,y201);decision=c['decision'];checks={'P201_teacher_probability_fidelity':r201['student_teacher_probability_MAE']<=float(decision['maximum_P201_teacher_probability_MAE']),'P201_integrated_Brier_strictly_better_than_teacher':r201['student_integrated_brier']<r201['teacher_integrated_brier'],'P201_calibration_noninferior':r201['absolute_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'input_dimension':36,'output_budget_count':7},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_teacher_loss':last_teacher,'final_truth_loss':last_truth,'conflicting_steps':conflicts,'steps':int(m['steps'])},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
