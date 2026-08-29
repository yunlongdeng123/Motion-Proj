"""Compile set-level dual prices for attainable fixed-budget allocation."""

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
from scripts.run_worldsim_v67_p254_shadow_price_budget_policy import ShadowPriceBudgetPolicy


class GroupBudgetDualCompiler(nn.Module):
 def __init__(self,width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.encoder=nn.Sequential(nn.Linear(72,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.intercept=nn.Linear(width,1);self.rate_knots=nn.Linear(width,self.knot_count)
 def forward(self,summary,normalized_fraction):
  context=self.encoder(summary);rates=functional.softplus(self.rate_knots(context));width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((normalized_fraction+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);integral=base+width*(r0*fraction+.5*(r1-r0)*fraction.square());return torch.tanh(self.intercept(context).squeeze(1)-integral)


def _groups(feature,scenes,size):
 groups=[];labels=[]
 for scene in np.unique(scenes):
  index=np.flatnonzero(scenes==scene)
  for start in range(0,len(index)-size+1,size):groups.append(feature[index[start:start+size]]);labels.append(int(scene))
 return np.stack(groups).astype(np.float32),np.asarray(labels,np.int64)


def _allocate(policy,groups,price_z,chunk_size):
 g,f=price_z.shape;size=groups.shape[1];features=np.broadcast_to(groups[:,None,:,:],(g,f,size,36)).reshape(-1,36);prices=np.broadcast_to(price_z[:,:,None],(g,f,size)).reshape(-1);outputs=[]
 with torch.no_grad():
  for start in range(0,len(features),chunk_size):outputs.append(policy(torch.from_numpy(features[start:start+chunk_size]).cuda(),torch.from_numpy(prices[start:start+chunk_size]).cuda()).cpu().numpy())
 return np.concatenate(outputs).reshape(g,f,size)


def _target_prices(policy,groups,fractions,steps,chunk_size):
 endpoints=np.tile(np.asarray([[-1.,1.]],np.float32),(len(groups),1));allocation=_allocate(policy,groups,endpoints,chunk_size);high_cost=np.mean((allocation[:,0]+1)/2,1);low_cost=np.mean((allocation[:,1]+1)/2,1);desired=low_cost[:,None]+fractions[None,:]*(high_cost-low_cost)[:,None];low=np.full_like(desired,-1);high=np.ones_like(desired)
 for _ in range(steps):
  middle=.5*(low+high);cost=np.mean((_allocate(policy,groups,middle,chunk_size)+1)/2,2);above=cost>desired;low=np.where(above,middle,low);high=np.where(above,high,middle)
 return .5*(low+high),low_cost,high_cost


def _reward(teacher,groups,budget_z,chunk_size):
 features=np.broadcast_to(groups[:,None,:,:],budget_z.shape+(36,)).reshape(-1,36);budgets=budget_z.reshape(-1);outputs=[]
 with torch.no_grad():
  for start in range(0,len(features),chunk_size):outputs.append(teacher(torch.from_numpy(features[start:start+chunk_size]).cuda(),torch.from_numpy(budgets[start:start+chunk_size]).cuda()).mean(1).cpu().numpy())
 return np.concatenate(outputs).reshape(budget_z.shape+(groups.shape[1],)).mean(2)


def _evaluate(student,policy,teacher,groups,target_price,low_cost,high_cost,fractions,price_mean,price_scale,chunk_size):
 summary=np.concatenate((groups.mean(1),groups.std(1)),1).astype(np.float32);x=torch.from_numpy(summary).cuda();candidate=[];torch.cuda.synchronize();before=time.monotonic()
 with torch.no_grad():
  for value in fractions:candidate.append(student(x,torch.full((len(x),),float(2*value-1),device='cuda')).cpu().numpy())
 torch.cuda.synchronize();forward=time.monotonic()-before;candidate=np.stack(candidate,1);candidate_budget=_allocate(policy,groups,candidate.astype(np.float32),chunk_size);target_budget=_allocate(policy,groups,target_price.astype(np.float32),chunk_size);candidate_cost=np.mean((candidate_budget+1)/2,2);target_cost=np.mean((target_budget+1)/2,2);attained=(candidate_cost-low_cost[:,None])/(high_cost-low_cost)[:,None].clip(min=1e-6);candidate_reward=_reward(teacher,groups,candidate_budget.astype(np.float32),chunk_size);target_reward=_reward(teacher,groups,target_budget.astype(np.float32),chunk_size);physical_price=np.exp(price_mean+price_scale*target_price);regret=(target_reward-physical_price*target_cost)-(candidate_reward-physical_price*candidate_cost);return {'group_count':int(len(groups)),'group_size':int(groups.shape[1]),'fraction_count':int(len(fractions)),'normalized_log_price_MAE':float(np.mean(np.abs(candidate-target_price))),'attained_budget_fraction_MAE':float(np.mean(np.abs(attained-fractions[None,:]))),'mean_frozen_Lagrangian_utility_regret':float(np.mean(regret)),'fraction_price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,ss,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,s183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,s201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);size=int(c['group_size']);sg,sgs=_groups(sf,ss,size);g183,_=_groups(f183,s183,size);g201,_=_groups(f201,s201,size);forward=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(forward['context_width']),int(forward['rate_knot_count'])).cuda();teacher.load_state_dict(forward['model_state_dict']);teacher.eval();artifact=torch.load(a.runs_root/c['frozen_p254']['run']/c['frozen_p254']['artifact'],map_location='cuda');policy=ShadowPriceBudgetPolicy(int(artifact['width']),int(artifact['rate_knot_count'])).cuda();policy.load_state_dict(artifact['model_state_dict']);policy.eval();[parameter.requires_grad_(False) for parameter in policy.parameters()];training_fractions=np.asarray(c['training_attainable_budget_fractions'],np.float32);heldout_fractions=np.asarray(c['heldout_attainable_budget_fractions'],np.float32);chunk=int(c['inference_chunk_size']);source_target,source_low,source_high=_target_prices(policy,sg,training_fractions,int(c['teacher_bisection_steps']),chunk);target183,low183,high183=_target_prices(policy,g183,heldout_fractions,int(c['teacher_bisection_steps']),chunk);target201,low201,high201=_target_prices(policy,g201,heldout_fractions,int(c['teacher_bisection_steps']),chunk);dev=sgs%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout,source_dev_low,source_dev_high=_target_prices(policy,sg[dev],heldout_fractions,int(c['teacher_bisection_steps']),chunk);summary=np.concatenate((sg.mean(1),sg.std(1)),1).astype(np.float32);x=torch.from_numpy(summary).cuda();target=torch.from_numpy(source_target).cuda();fraction_z=torch.from_numpy(2*training_fractions-1).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=GroupBudgetDualCompiler(int(m['width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];fraction_idx=torch.randint(len(training_fractions),(len(idx),),device='cuda');prediction=student(x[idx],fraction_z[fraction_idx]);truth=target[idx,fraction_idx];loss=functional.l1_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P256 group dual step={step+1} price_mae={last:.7f}',flush=True)
 price_mean=float(artifact['shadow_price_log_mean']);price_scale=float(artifact['shadow_price_log_scale']);source=_evaluate(student,policy,teacher,sg[dev],source_heldout,source_dev_low,source_dev_high,heldout_fractions,price_mean,price_scale,chunk);r183=_evaluate(student,policy,teacher,g183,target183,low183,high183,heldout_fractions,price_mean,price_scale,chunk);r201=_evaluate(student,policy,teacher,g201,target201,low201,high201,heldout_fractions,price_mean,price_scale,chunk);decision=c['decision'];checks={'P201_budget_constraint_fidelity':r201['attained_budget_fraction_MAE']<=float(decision['maximum_P201_attained_budget_fraction_MAE']),'P201_frozen_utility_regret':r201['mean_frozen_Lagrangian_utility_regret']<=float(decision['maximum_P201_mean_frozen_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'rate_knot_count':m['rate_knot_count'],'input_dimension':72,'base_model':c['frozen_p254']},d/c['model_artifact']);summary_out={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'group_count':int((~dev).sum()),'group_size':size,'fraction_count':int(len(training_fractions)),'final_normalized_log_price_mae':last},'heldout_attainable_budget_fractions':[float(v) for v in heldout_fractions],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary_out,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary_out},indent=2))


if __name__=='__main__':main()
