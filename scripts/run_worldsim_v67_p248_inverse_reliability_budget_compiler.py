"""Compile frozen P246 probabilities into non-crossing reliability-budget quantiles."""

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
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface


class InverseReliabilityBudgetCompiler(nn.Module):
 def __init__(self,context_width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.encoder=nn.Sequential(nn.Linear(36,context_width),nn.SiLU(),nn.Linear(context_width,context_width),nn.SiLU());self.intercept=nn.Linear(context_width,4);self.rate_knots=nn.Linear(context_width,4*self.knot_count)
 def forward(self,feature,reliability_level):
  context=self.encoder(feature);rates=functional.softplus(self.rate_knots(context).reshape(-1,4,self.knot_count));width=2./(self.knot_count-1);areas=.5*(rates[:,:,:-1]+rates[:,:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:,:1]),torch.cumsum(areas,2)),2);position=(2*reliability_level.reshape(-1,1)).clamp(0,2)/width;index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;gather=index[:,None,:].expand(-1,4,-1);r0=torch.gather(rates,2,gather).squeeze(2);r1=torch.gather(rates,2,gather+1).squeeze(2);base=torch.gather(cumulative,2,gather).squeeze(2);integral=base+width*(r0*fraction+.5*(r1-r0)*fraction.square());latent=self.intercept(context)+integral;first=latent[:,:1];increments=functional.softplus(latent[:,1:]);return torch.cat((first,first+torch.cumsum(increments,1)),1).clamp(-1,1)


def _inverse_teacher(model,feature,levels,chunk_size):
 outputs=[]
 with torch.no_grad():
  for start in range(0,len(feature),chunk_size):
   x=torch.from_numpy(feature[start:start+chunk_size]).cuda();count=len(x);expanded=x[:,None,:].expand(-1,len(levels),-1).reshape(-1,x.shape[1]);alpha=torch.tensor(levels,device='cuda').reshape(1,-1).expand(count,-1).reshape(-1,1);low=torch.full((len(expanded),4),-1.,device='cuda');high=torch.full_like(low,1.);low_probability=model(expanded,low[:,0]);high_probability=model(expanded,high[:,0])
   for _ in range(24):
    middle=.5*(low+high);columns=[]
    for horizon in range(4):columns.append(model(expanded,middle[:,horizon])[:,horizon])
    probability=torch.stack(columns,1);below=probability<alpha;low=torch.where(below,middle,low);high=torch.where(below,high,middle)
   target=.5*(low+high);target=torch.where(low_probability>=alpha,torch.full_like(target,-1),target);target=torch.where(high_probability<alpha,torch.ones_like(target),target);outputs.append(target.reshape(count,len(levels),4).permute(0,2,1).cpu().numpy())
 return np.concatenate(outputs)


def _probability_at_horizon_budgets(model,feature,budget_surface,chunk_size):
 n,h,l=budget_surface.shape;outputs=[]
 with torch.no_grad():
  for start in range(0,n,chunk_size):
   x=torch.from_numpy(feature[start:start+chunk_size]).cuda();z=torch.from_numpy(budget_surface[start:start+chunk_size]).cuda();rows=[]
   for horizon in range(h):
    columns=[]
    for level in range(l):columns.append(model(x,z[:,horizon,level])[:,horizon])
    rows.append(torch.stack(columns,1))
   outputs.append(torch.stack(rows,1).cpu().numpy())
 return np.concatenate(outputs)


def _evaluate(student,teacher_model,feature,teacher_budget,levels,chunk_size):
 x=torch.from_numpy(feature).cuda();predictions=[];torch.cuda.synchronize();before=time.monotonic()
 with torch.no_grad():
  for level in levels:predictions.append(student(x,torch.full((len(x),),float(level),device='cuda')).cpu().numpy())
 torch.cuda.synchronize();forward=time.monotonic()-before;candidate=np.stack(predictions,2);teacher_probability=_probability_at_horizon_budgets(teacher_model,feature,teacher_budget,chunk_size);candidate_probability=_probability_at_horizon_budgets(teacher_model,feature,candidate,chunk_size);return {'trajectory_count':int(len(feature)),'reliability_level_count':int(len(levels)),'inverse_normalized_log_budget_MAE':float(np.mean(np.abs(candidate-teacher_budget))),'final_horizon_inverse_normalized_log_budget_MAE':float(np.mean(np.abs(candidate[:,-1]-teacher_budget[:,-1]))),'reconstructed_teacher_probability_MAE':float(np.mean(np.abs(candidate_probability-teacher_probability))),'lower_censored_target_fraction':float(np.mean(teacher_budget<=-1+1e-6)),'upper_censored_target_fraction':float(np.mean(teacher_budget>=1-1e-6)),'reliability_level_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'horizon_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)<-1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,scenes,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,_,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,_,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);frozen=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(frozen['context_width']),int(frozen['rate_knot_count'])).cuda();teacher.load_state_dict(frozen['model_state_dict']);teacher.eval();training_levels=np.asarray(c['training_reliability_levels'],np.float32);heldout_levels=np.asarray(c['heldout_reliability_levels'],np.float32);chunk=int(c['teacher_inverse_chunk_size']);source_target=_inverse_teacher(teacher,sf,training_levels,chunk);target183=_inverse_teacher(teacher,f183,heldout_levels,chunk);target201=_inverse_teacher(teacher,f201,heldout_levels,chunk);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout_target=_inverse_teacher(teacher,sf[dev],heldout_levels,chunk);x=torch.from_numpy(sf).cuda();target=torch.from_numpy(source_target).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=InverseReliabilityBudgetCompiler(int(m['context_width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];level_index=torch.randint(len(training_levels),(len(idx),),device='cuda');prediction=student(x[idx],torch.from_numpy(training_levels).cuda()[level_index]);truth=target[idx,:,level_index];loss=functional.l1_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P248 inverse budget compiler step={step+1} mae={last:.7f}',flush=True)
 source=_evaluate(student,teacher,sf[dev],source_heldout_target,heldout_levels,chunk);r183=_evaluate(student,teacher,f183,target183,heldout_levels,chunk);r201=_evaluate(student,teacher,f201,target201,heldout_levels,chunk);decision=c['decision'];checks={'P201_inverse_budget_fidelity':r201['inverse_normalized_log_budget_MAE']<=float(decision['maximum_P201_inverse_normalized_log_budget_MAE']),'P201_reconstructed_probability_fidelity':r201['reconstructed_teacher_probability_MAE']<=float(decision['maximum_P201_reconstructed_teacher_probability_MAE'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'context_width':m['context_width'],'rate_knot_count':m['rate_knot_count'],'input_dimension':36,'budget_log_mean':frozen['budget_log_mean'],'budget_log_scale':frozen['budget_log_scale'],'training_reliability_levels':training_levels,'base_model':c['frozen_p246']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'reliability_level_count':int(len(training_levels)),'final_normalized_log_budget_mae':last},'heldout_reliability_levels':[float(v) for v in heldout_levels],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
