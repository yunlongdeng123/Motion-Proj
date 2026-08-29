"""Distill the prefix surface into a continuously queryable budget CDF."""

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
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset


class ContinuousBudgetPrefixSurface(nn.Module):
 def __init__(self,widths,components):
  super().__init__();layers=[];width=36
  for hidden in widths:layers.extend((nn.Linear(width,int(hidden)),nn.SiLU()));width=int(hidden)
  layers.append(nn.Linear(width,3*components+6));self.network=nn.Sequential(*layers);self.components=int(components)
 def forward(self,feature,normalized_log_budget):
  raw=self.network(feature);k=self.components;weight=functional.softmax(raw[:,:k],1);location=raw[:,k:2*k];scale=functional.softplus(raw[:,2*k:3*k])+.05;z=normalized_log_budget.reshape(-1,1);base=torch.sum(weight*torch.sigmoid((z-location)/scale),1);transition=raw[:,3*k:].reshape(-1,3,2);retention=torch.sigmoid(transition[:,:,0]+functional.softplus(transition[:,:,1])*z);curves=[base]
  for index in range(3):curves.append(curves[-1]*retention[:,index])
  return torch.stack(curves,1)


def _paired_dataset(path,common,training_budgets,heldout_budgets):
 feature,teacher,truth,scenes,train_seconds=_dataset(path,*common[:8],training_budgets,*common[8:]);heldout_feature,heldout_teacher,heldout_truth,heldout_scenes,heldout_seconds=_dataset(path,*common[:8],heldout_budgets,*common[8:])
 if not np.array_equal(scenes,heldout_scenes):raise RuntimeError('P238 training/heldout budget rows misaligned')
 return feature,teacher,truth,heldout_teacher,heldout_truth,scenes,train_seconds+heldout_seconds


def _evaluate(model,feature,teacher,truth,budgets,mean,scale,index=None):
 if index is not None:feature=feature[index];teacher=teacher[index];truth=truth[index]
 tensor=torch.from_numpy(feature).cuda();outputs=[];torch.cuda.synchronize();before=time.monotonic()
 for value in budgets:
  z=torch.full((len(feature),),(np.log(float(value))-mean)/scale,device='cuda');outputs.append(model(tensor,z).detach().cpu().numpy())
 torch.cuda.synchronize();forward=time.monotonic()-before;candidate=np.stack(outputs,2);cb=float(np.mean((candidate-truth)**2));tb=float(np.mean((teacher-truth)**2));cc=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));tc=float(np.mean(np.abs(teacher.mean(0)-truth.mean(0))));return {'trajectory_count':int(len(feature)),'heldout_budget_count':int(len(budgets)),'surface_teacher_probability_MAE':float(np.mean(np.abs(candidate-teacher))),'final_curve_teacher_probability_MAE':float(np.mean(np.abs(candidate[:,-1]-teacher[:,-1]))),'student_surface_integrated_brier':cb,'teacher_surface_integrated_brier':tb,'relative_surface_Brier_degradation_vs_teacher':(cb-tb)/tb,'student_surface_calibration_error':cc,'teacher_surface_calibration_error':tc,'absolute_surface_calibration_error_increase_vs_teacher':cc-tc,'budget_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'horizon_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);training_budgets=np.asarray(c['training_budgets'],np.float32);heldout_budgets=np.sqrt(training_budgets[:-1]*training_budgets[1:]);common=(members,ens,density,fd,p199,copula,calibrator,h,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,st,sy,sht,shy,scenes,ss=_paired_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],common,training_budgets,heldout_budgets);f183,t183,y183,ht183,hy183,_,s183=_paired_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],common,training_budgets,heldout_budgets);f201,t201,y201,ht201,hy201,_,s201=_paired_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],common,training_budgets,heldout_budgets);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();target=torch.from_numpy(st).cuda();log_budget=np.log(training_budgets);budget_mean=float(log_budget.mean());budget_scale=float(log_budget.std());budget_z=torch.from_numpy(((log_budget-budget_mean)/budget_scale).astype(np.float32)).cuda();m=c['student'];model=ContinuousBudgetPrefixSurface(m['hidden_dimensions'],m['logistic_component_count']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];budget_idx=torch.randint(len(training_budgets),(len(idx),),device='cuda');prediction=model(x[idx],budget_z[budget_idx]);truth=target[idx,:,budget_idx];loss=functional.mse_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P238 continuous budget surface step={step+1} mse={last:.7f}',flush=True)
 source=_evaluate(model,sf,sht,shy,heldout_budgets,budget_mean,budget_scale,dev);r183=_evaluate(model,f183,ht183,hy183,heldout_budgets,budget_mean,budget_scale);r201=_evaluate(model,f201,ht201,hy201,heldout_budgets,budget_mean,budget_scale);decision=c['decision'];checks={'P201_heldout_surface_teacher_fidelity':r201['surface_teacher_probability_MAE']<=float(decision['maximum_P201_heldout_surface_teacher_probability_MAE']),'P201_heldout_final_curve_teacher_fidelity':r201['final_curve_teacher_probability_MAE']<=float(decision['maximum_P201_heldout_final_curve_teacher_probability_MAE']),'P201_heldout_surface_quality_noninferior':r201['relative_surface_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_P201_heldout_surface_Brier_degradation']) and r201['absolute_surface_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_heldout_surface_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'logistic_component_count':m['logistic_component_count'],'input_dimension':36,'budget_log_mean':budget_mean,'budget_log_scale':budget_scale,'training_budgets':training_budgets},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'budgets':[float(v) for v in training_budgets],'final_teacher_surface_mse':last},'heldout_budgets':[float(v) for v in heldout_budgets],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
