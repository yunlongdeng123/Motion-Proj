"""Distill the integrated monotone surface with probability-space L1 imitation."""

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


class IntegratedMonotonePrefixSurface(nn.Module):
 def __init__(self,context_width,rate_width,quadrature_points):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(36,context_width),nn.SiLU(),nn.Linear(context_width,context_width),nn.SiLU());self.intercept=nn.Linear(context_width,4);self.rate=nn.Sequential(nn.Linear(context_width+1,rate_width),nn.SiLU(),nn.Linear(rate_width,4));nodes,weights=np.polynomial.legendre.leggauss(int(quadrature_points));self.register_buffer('quadrature_nodes',torch.tensor(nodes,dtype=torch.float32));self.register_buffer('quadrature_weights',torch.tensor(weights,dtype=torch.float32))
 def forward(self,feature,normalized_log_budget):
  context=self.encoder(feature);z=normalized_log_budget.reshape(-1,1);span=.5*(z+1);locations=span*self.quadrature_nodes.reshape(1,-1)+.5*(z-1);expanded=context[:,None,:].expand(-1,len(self.quadrature_nodes),-1);rate=functional.softplus(self.rate(torch.cat((expanded,locations[:,:,None]),2)));integral=span*(rate*self.quadrature_weights.reshape(1,-1,1)).sum(1);units=torch.sigmoid(self.intercept(context)+integral);return torch.cumprod(units,1)


def _paired_dataset(path,common,feature_budgets,training_budgets,heldout_budgets):
 feature,_,_,scenes,feature_seconds=_dataset(path,*common[:8],feature_budgets,*common[8:]);_,teacher,truth,training_scenes,train_seconds=_dataset(path,*common[:8],training_budgets,*common[8:]);_,heldout_teacher,heldout_truth,heldout_scenes,heldout_seconds=_dataset(path,*common[:8],heldout_budgets,*common[8:])
 if not np.array_equal(scenes,training_scenes) or not np.array_equal(scenes,heldout_scenes):raise RuntimeError('P242 feature/training/heldout rows misaligned')
 return feature,teacher,truth,heldout_teacher,heldout_truth,scenes,feature_seconds+train_seconds+heldout_seconds


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
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['heldout_anchor_budgets'],np.float32);heldout_budgets=np.sqrt(anchors[:-1]*anchors[1:]);domain=c['training_budget_domain'];log_min=float(np.log(domain[0]));log_max=float(np.log(domain[1]));count=int(c['training_budget_count']);fractions=(np.arange(count,dtype=np.float32)+float(c['training_budget_offset']))/count;training_budgets=np.exp(log_min+fractions*(log_max-log_min)).astype(np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,st,sy,sht,shy,scenes,ss=_paired_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],common,anchors,training_budgets,heldout_budgets);f183,t183,y183,ht183,hy183,_,s183=_paired_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],common,anchors,training_budgets,heldout_budgets);f201,t201,y201,ht201,hy201,_,s201=_paired_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],common,anchors,training_budgets,heldout_budgets);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();target=torch.from_numpy(st).cuda();budget_mean=.5*(log_min+log_max);budget_scale=.5*(log_max-log_min);budget_z=torch.from_numpy(((np.log(training_budgets)-budget_mean)/budget_scale).astype(np.float32)).cuda();m=c['student'];model=IntegratedMonotonePrefixSurface(int(m['context_width']),int(m['rate_width']),int(m['quadrature_points'])).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];budget_idx=torch.randint(len(training_budgets),(len(idx),),device='cuda');prediction=model(x[idx],budget_z[budget_idx]);truth=target[idx,:,budget_idx];loss=functional.l1_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P242 L1 integrated monotone surface step={step+1} mae={last:.7f}',flush=True)
 source=_evaluate(model,sf,sht,shy,heldout_budgets,budget_mean,budget_scale,dev);r183=_evaluate(model,f183,ht183,hy183,heldout_budgets,budget_mean,budget_scale);r201=_evaluate(model,f201,ht201,hy201,heldout_budgets,budget_mean,budget_scale);decision=c['decision'];checks={'P201_heldout_surface_teacher_fidelity':r201['surface_teacher_probability_MAE']<=float(decision['maximum_P201_heldout_surface_teacher_probability_MAE']),'P201_heldout_final_curve_teacher_fidelity':r201['final_curve_teacher_probability_MAE']<=float(decision['maximum_P201_heldout_final_curve_teacher_probability_MAE']),'P201_heldout_surface_quality_noninferior':r201['relative_surface_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_P201_heldout_surface_Brier_degradation']) and r201['absolute_surface_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_heldout_surface_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'context_width':m['context_width'],'rate_width':m['rate_width'],'quadrature_points':m['quadrature_points'],'input_dimension':36,'budget_log_mean':budget_mean,'budget_log_scale':budget_scale,'training_budgets':training_budgets},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'budget_count':int(len(training_budgets)),'final_teacher_surface_mae':last},'heldout_budgets':[float(v) for v in heldout_budgets],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
