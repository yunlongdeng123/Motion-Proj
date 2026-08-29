"""Compile fixed-group dual prices for reliability-floor regularized allocation."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface
from scripts.run_worldsim_v67_p256_group_budget_dual_compiler import _groups
from scripts.run_worldsim_v67_p268_reliability_floor_lagrangian_policy import ReliabilityFloorLagrangianPolicy


class ReliabilityFloorGroupDual(nn.Module):
 def __init__(self,width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.encoder=nn.Sequential(nn.Linear(74,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.intercept=nn.Linear(width,1);self.rate_knots=nn.Linear(width,self.knot_count)
 def forward(self,summary,normalized_fraction,alpha,normalized_floor):
  context=self.encoder(torch.cat((summary,alpha[:,None],normalized_floor[:,None]),1));rates=F.softplus(self.rate_knots(context));width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((normalized_fraction+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);integral=base+width*(r0*fraction+.5*(r1-r0)*fraction.square());return torch.tanh(self.intercept(context).squeeze(1)-integral)


def _allocate(policy,groups,price_z,alphas,floor_z,chunk):
 g,a,l,f=price_z.shape;size=groups.shape[1];features=np.broadcast_to(groups[:,None,None,None,:,:],(g,a,l,f,size,36)).reshape(-1,36);prices=np.broadcast_to(price_z[:,:,:,:,None],(g,a,l,f,size)).reshape(-1);av=np.broadcast_to(alphas[None,:,None,None,None],(g,a,l,f,size)).reshape(-1);fv=np.broadcast_to(floor_z[None,None,:,None,None],(g,a,l,f,size)).reshape(-1);outputs=[]
 with torch.no_grad():
  for start in range(0,len(features),chunk):outputs.append(policy(torch.from_numpy(features[start:start+chunk]).cuda(),torch.from_numpy(prices[start:start+chunk]).cuda(),torch.from_numpy(av[start:start+chunk]).cuda(),torch.from_numpy(fv[start:start+chunk]).cuda()).cpu().numpy())
 return np.concatenate(outputs).reshape(g,a,l,f,size)


def _target_prices(policy,groups,alphas,floor_z,fractions,steps,chunk):
 endpoints=np.broadcast_to(np.asarray([-1.,1.],np.float32)[None,None,None,:],(len(groups),len(alphas),len(floor_z),2));allocation=_allocate(policy,groups,endpoints,alphas,floor_z,chunk);high_cost=np.mean((allocation[:,:,:,0]+1)/2,3);low_cost=np.mean((allocation[:,:,:,1]+1)/2,3);desired=low_cost[:,:,:,None]+fractions[None,None,None,:]*(high_cost-low_cost)[:,:,:,None];low=np.full_like(desired,-1);high=np.ones_like(desired)
 for _ in range(steps):
  middle=.5*(low+high);cost=np.mean((_allocate(policy,groups,middle,alphas,floor_z,chunk)+1)/2,4);above=cost>desired;low=np.where(above,middle,low);high=np.where(above,high,middle)
 return .5*(low+high),low_cost,high_cost


def _alpha_value(probability,alpha,epsilon):
 shifted=probability+epsilon
 if abs(float(alpha)-1)<1e-7:return np.log(shifted).mean(-1)
 power=1-float(alpha);return ((np.power(shifted,power)-1)/power).mean(-1)


def _utility(teacher,groups,budget_z,alphas,floors,epsilon,penalty,chunk):
 g,a,l,f,size=budget_z.shape;features=np.broadcast_to(groups[:,None,None,None,:,:],(g,a,l,f,size,36)).reshape(-1,36);budgets=budget_z.reshape(-1);outputs=[]
 with torch.no_grad():
  for start in range(0,len(features),chunk):outputs.append(teacher(torch.from_numpy(features[start:start+chunk]).cuda(),torch.from_numpy(budgets[start:start+chunk]).cuda()).cpu().numpy())
 probability=np.concatenate(outputs).reshape(g,a,l,f,size,4);return np.stack([np.stack([(_alpha_value(probability[:,ai,li],alpha,epsilon)-penalty*np.maximum(float(floor)-probability[:,ai,li,:,:,-1],0)).mean(-1) for li,floor in enumerate(floors)],1) for ai,alpha in enumerate(alphas)],1)


def _evaluate(student,policy,teacher,groups,target,low,high,alphas,floors,floor_z,fractions,mean,scale,epsilon,penalty,chunk):
 summary=np.concatenate((groups.mean(1),groups.std(1)),1).astype(np.float32);x=torch.from_numpy(summary).cuda();candidate=[];torch.cuda.synchronize();started=time.monotonic()
 with torch.no_grad():
  for alpha in alphas:
   by_floor=[]
   for value in floor_z:by_floor.append(np.stack([student(x,torch.full((len(x),),float(2*fraction-1),device='cuda'),torch.full((len(x),),float(alpha),device='cuda'),torch.full((len(x),),float(value),device='cuda')).cpu().numpy() for fraction in fractions],1))
   candidate.append(np.stack(by_floor,1))
 torch.cuda.synchronize();forward=time.monotonic()-started;candidate=np.stack(candidate,1);candidate_budget=_allocate(policy,groups,candidate.astype(np.float32),alphas,floor_z,chunk);target_budget=_allocate(policy,groups,target.astype(np.float32),alphas,floor_z,chunk);candidate_cost=np.mean((candidate_budget+1)/2,4);target_cost=np.mean((target_budget+1)/2,4);attained=(candidate_cost-low[:,:,:,None])/(high-low)[:,:,:,None].clip(min=1e-6);candidate_value=_utility(teacher,groups,candidate_budget.astype(np.float32),alphas,floors,epsilon,penalty,chunk);target_value=_utility(teacher,groups,target_budget.astype(np.float32),alphas,floors,epsilon,penalty,chunk);price=np.exp(mean+scale*target);regret=(target_value-price*target_cost)-(candidate_value-price*candidate_cost);return {'group_count':int(len(groups)),'group_size':int(groups.shape[1]),'alpha_count':int(len(alphas)),'floor_count':int(len(floors)),'fraction_count':int(len(fractions)),'normalized_log_price_MAE':float(np.mean(np.abs(candidate-target))),'attained_budget_fraction_MAE':float(np.mean(np.abs(attained-fractions[None,None,None,:]))),'mean_frozen_floor_Lagrangian_group_utility_regret':float(np.mean(regret)),'fraction_price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=3)>1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,ss,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,s183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,s201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);size=int(c['group_size']);sg,sgs=_groups(sf,ss,size);g183,_=_groups(f183,s183,size);g201,_=_groups(f201,s201,size);forward=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(forward['context_width']),int(forward['rate_knot_count'])).cuda();teacher.load_state_dict(forward['model_state_dict']);teacher.eval();artifact=torch.load(a.runs_root/c['frozen_p268']['run']/c['frozen_p268']['artifact'],map_location='cuda');policy=ReliabilityFloorLagrangianPolicy(int(artifact['width']),int(artifact['rate_knot_count'])).cuda();policy.load_state_dict(artifact['model_state_dict']);policy.eval();alphas=np.asarray(c['training_alpha_fairness'],np.float32);ha=np.asarray(c['heldout_alpha_fairness'],np.float32);floors=np.asarray(c['training_final_reliability_floors'],np.float32);hf=np.asarray(c['heldout_final_reliability_floors'],np.float32);domain=np.asarray(artifact['floor_domain'],np.float32);floor_z=(2*(floors-domain[0])/(domain[1]-domain[0])-1).astype(np.float32);heldout_floor_z=(2*(hf-domain[0])/(domain[1]-domain[0])-1).astype(np.float32);fractions=np.asarray(c['training_attainable_budget_fractions'],np.float32);heldout_fractions=np.asarray(c['heldout_attainable_budget_fractions'],np.float32);chunk=int(c['inference_chunk_size']);steps=int(c['teacher_bisection_steps']);source_target,_,_=_target_prices(policy,sg,alphas,floor_z,fractions,steps,chunk);target183,low183,high183=_target_prices(policy,g183,ha,heldout_floor_z,heldout_fractions,steps,chunk);target201,low201,high201=_target_prices(policy,g201,ha,heldout_floor_z,heldout_fractions,steps,chunk);dev=sgs%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout,source_low,source_high=_target_prices(policy,sg[dev],ha,heldout_floor_z,heldout_fractions,steps,chunk);summary=np.concatenate((sg.mean(1),sg.std(1)),1).astype(np.float32);x=torch.from_numpy(summary).cuda();target=torch.from_numpy(source_target).cuda();fz=torch.from_numpy(2*fractions-1).cuda();at=torch.from_numpy(alphas).cuda();ft=torch.from_numpy(floor_z).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=ReliabilityFloorGroupDual(int(m['width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');li=torch.randint(len(floors),(len(idx),),device='cuda');fi=torch.randint(len(fractions),(len(idx),),device='cuda');prediction=student(x[idx],fz[fi],at[ai],ft[li]);loss=F.l1_loss(prediction,target[idx,ai,li,fi]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P269 floor group dual step={step+1} price_mae={last:.7f}',flush=True)
 mean=float(artifact['shadow_price_log_mean']);scale=float(artifact['shadow_price_log_scale']);epsilon=float(artifact['alpha_utility_epsilon']);penalty=float(artifact['shortfall_penalty']);source=_evaluate(student,policy,teacher,sg[dev],source_heldout,source_low,source_high,ha,hf,heldout_floor_z,heldout_fractions,mean,scale,epsilon,penalty,chunk);r183=_evaluate(student,policy,teacher,g183,target183,low183,high183,ha,hf,heldout_floor_z,heldout_fractions,mean,scale,epsilon,penalty,chunk);r201=_evaluate(student,policy,teacher,g201,target201,low201,high201,ha,hf,heldout_floor_z,heldout_fractions,mean,scale,epsilon,penalty,chunk);decision=c['decision'];checks={'P201_budget_constraint_fidelity':r201['attained_budget_fraction_MAE']<=float(decision['maximum_P201_attained_budget_fraction_MAE']),'P201_frozen_floor_Lagrangian_group_regret':r201['mean_frozen_floor_Lagrangian_group_utility_regret']<=float(decision['maximum_P201_mean_frozen_floor_Lagrangian_group_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'rate_knot_count':m['rate_knot_count'],'input_dimension':74,'base_model':c['frozen_p268']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'group_count':int((~dev).sum()),'group_size':size,'alpha_count':int(len(alphas)),'floor_count':int(len(floors)),'fraction_count':int(len(fractions)),'final_normalized_log_price_mae':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
