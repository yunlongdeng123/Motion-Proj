"""Compile attainable group budgets into composite-risk shadow prices."""

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
from scripts.run_worldsim_v67_p256_group_budget_dual_compiler import _groups
from scripts.run_worldsim_v67_p275_epistemic_lcb_surface import EpistemicLCBSurface
from scripts.run_worldsim_v67_p276_epistemic_lcb_lagrangian_policy import _alpha_utility
from scripts.run_worldsim_v67_p279_epistemic_tail_cvar_allocator import EpistemicTailCVaRAllocator,_infer_probability


class EpistemicTailCVaRGroupDual(nn.Module):
 def __init__(self,width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.encoder=nn.Sequential(nn.Linear(76,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.intercept=nn.Linear(width,1);self.rate_knots=nn.Linear(width,self.knot_count)
 def forward(self,summary,fraction,alpha,beta,floor,tail_mass):
  context=self.encoder(torch.cat((summary,alpha[:,None],beta[:,None],floor[:,None],tail_mass[:,None]),1));rates=F.softplus(self.rate_knots(context));width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((fraction+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);part=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);return torch.tanh(self.intercept(context).squeeze(1)-base-width*(r0*part+.5*(r1-r0)*part.square()))


def _allocate(policy,groups,price_z,alphas,beta_z,floor_z,tail_masses,chunk):
 g,a,e,l,q,f=price_z.shape;s=groups.shape[1];features=np.broadcast_to(groups[:,None,None,None,None,None,:,:],(g,a,e,l,q,f,s,36)).reshape(-1,s,36);prices=price_z.reshape(-1);av=np.broadcast_to(alphas[None,:,None,None,None,None],price_z.shape).reshape(-1);bv=np.broadcast_to(beta_z[None,None,:,None,None,None],price_z.shape).reshape(-1);fv=np.broadcast_to(floor_z[None,None,None,:,None,None],price_z.shape).reshape(-1);qv=np.broadcast_to(tail_masses[None,None,None,None,:,None],price_z.shape).reshape(-1);out=[]
 with torch.no_grad():
  for start in range(0,len(features),chunk):out.append(policy(torch.from_numpy(features[start:start+chunk]).cuda(),torch.from_numpy(prices[start:start+chunk]).cuda(),torch.from_numpy(av[start:start+chunk]).cuda(),torch.from_numpy(bv[start:start+chunk]).cuda(),torch.from_numpy(fv[start:start+chunk]).cuda(),torch.from_numpy(qv[start:start+chunk]).cuda()).cpu().numpy())
 return np.concatenate(out).reshape(g,a,e,l,q,f,s)


def _target_prices(policy,groups,alphas,beta_z,floor_z,tail_masses,fractions,steps,chunk):
 endpoints=np.broadcast_to(np.asarray([-1.,1.],np.float32)[None,None,None,None,None,:],(len(groups),len(alphas),len(beta_z),len(floor_z),len(tail_masses),2));allocation=_allocate(policy,groups,endpoints,alphas,beta_z,floor_z,tail_masses,chunk);high=np.mean((allocation[:,:,:,:,:,0]+1)/2,5);low=np.mean((allocation[:,:,:,:,:,1]+1)/2,5);desired=low[:,:,:,:,:,None]+fractions[None,None,None,None,None,:]*(high-low)[:,:,:,:,:,None];left=np.full_like(desired,-1);right=np.ones_like(desired)
 for _ in range(steps):
  middle=.5*(left+right);cost=np.mean((_allocate(policy,groups,middle,alphas,beta_z,floor_z,tail_masses,chunk)+1)/2,6);above=cost>desired;left=np.where(above,middle,left);right=np.where(above,right,middle)
 return .5*(left+right),low,high


def _risk(teacher,groups,budget_z,alphas,beta_z,floors,tail_masses,epsilon,penalty,chunk):
 probability=_infer_probability(teacher,groups,budget_z,beta_z,chunk);by_alpha=[]
 for ai,alpha in enumerate(alphas):
  by_beta=[]
  for ei in range(len(beta_z)):
   by_floor=[]
   for li,floor in enumerate(floors):
    by_tail=[]
    for qi,tail_mass in enumerate(tail_masses):
     p=probability[:,ai,ei,li,qi];utility=_alpha_utility(p,float(alpha),epsilon).mean(2);shortfall=np.maximum(float(floor)-p[:,:,:,-1],0);k=max(1,int(np.ceil(float(tail_mass)*shortfall.shape[2])));cvar=np.sort(shortfall,axis=2)[:,:,-k:].mean(2);by_tail.append(utility-float(penalty)*cvar)
    by_floor.append(np.stack(by_tail,1))
   by_beta.append(np.stack(by_floor,1))
  by_alpha.append(np.stack(by_beta,1))
 return np.stack(by_alpha,1)


def _predict(student,summary,alphas,beta_z,floor_z,tail_masses,fractions):
 x=torch.from_numpy(summary).cuda();by_alpha=[]
 with torch.no_grad():
  for alpha in alphas:
   by_beta=[]
   for beta in beta_z:
    by_floor=[]
    for floor in floor_z:
     by_tail=[]
     for tail in tail_masses:by_tail.append(np.stack([student(x,torch.full((len(x),),float(2*fraction-1),device='cuda'),torch.full((len(x),),float(alpha),device='cuda'),torch.full((len(x),),float(beta),device='cuda'),torch.full((len(x),),float(floor),device='cuda'),torch.full((len(x),),float(tail),device='cuda')).cpu().numpy() for fraction in fractions],1))
     by_floor.append(np.stack(by_tail,1))
    by_beta.append(np.stack(by_floor,1))
   by_alpha.append(np.stack(by_beta,1))
 return np.stack(by_alpha,1)


def _evaluate(student,policy,teacher,groups,target,low,high,alphas,beta_z,floors,floor_z,tail_masses,fractions,mean,scale,epsilon,penalty,chunk):
 summary=np.concatenate((groups.mean(1),groups.std(1)),1).astype(np.float32);torch.cuda.synchronize();started=time.monotonic();candidate=_predict(student,summary,alphas,beta_z,floor_z,tail_masses,fractions);torch.cuda.synchronize();forward=time.monotonic()-started;candidate_budget=_allocate(policy,groups,candidate.astype(np.float32),alphas,beta_z,floor_z,tail_masses,chunk);target_budget=_allocate(policy,groups,target.astype(np.float32),alphas,beta_z,floor_z,tail_masses,chunk);candidate_cost=np.mean((candidate_budget+1)/2,6);target_cost=np.mean((target_budget+1)/2,6);attained=(candidate_cost-low[:,:,:,:,:,None])/(high-low)[:,:,:,:,:,None].clip(min=1e-6);candidate_value=_risk(teacher,groups,candidate_budget.astype(np.float32),alphas,beta_z,floors,tail_masses,epsilon,penalty,chunk);target_value=_risk(teacher,groups,target_budget.astype(np.float32),alphas,beta_z,floors,tail_masses,epsilon,penalty,chunk);price=np.exp(mean+scale*target);regret=(target_value-price*target_cost)-(candidate_value-price*candidate_cost);return {'group_count':int(len(groups)),'group_size':int(groups.shape[1]),'normalized_log_price_MAE':float(np.mean(np.abs(candidate-target))),'attained_budget_fraction_MAE':float(np.mean(np.abs(attained-fractions[None,None,None,None,None,:]))),'mean_frozen_composite_Lagrangian_group_utility_regret':float(np.mean(regret)),'fraction_price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=5)>1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,ss,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f201,_,_,s201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);size=int(c['group_size']);sg,sgs=_groups(sf,ss,size);g201,_=_groups(f201,s201,size);surface=torch.load(a.runs_root/c['frozen_p275']['run']/c['frozen_p275']['artifact'],map_location='cuda');teacher=EpistemicLCBSurface(int(surface['context_width']),int(surface['budget_rate_knot_count']),int(surface['beta_rate_knot_count'])).cuda();teacher.load_state_dict(surface['model_state_dict']);teacher.eval();artifact=torch.load(a.runs_root/c['frozen_p279']['run']/c['frozen_p279']['artifact'],map_location='cuda');policy=EpistemicTailCVaRAllocator(int(artifact['element_width']),int(artifact['context_width']),int(artifact['rate_knot_count'])).cuda();policy.load_state_dict(artifact['model_state_dict']);policy.eval();alphas=np.asarray(c['training_alpha_fairness'],np.float32);ha=np.asarray(c['heldout_alpha_fairness'],np.float32);betas=np.asarray(c['training_epistemic_betas'],np.float32);hb=np.asarray(c['heldout_epistemic_betas'],np.float32);bd=np.asarray(artifact['epistemic_beta_domain'],np.float32);beta_z=(2*(betas-bd[0])/(bd[1]-bd[0])-1).astype(np.float32);heldout_beta_z=(2*(hb-bd[0])/(bd[1]-bd[0])-1).astype(np.float32);floors=np.asarray(c['training_final_reliability_floors'],np.float32);hf=np.asarray(c['heldout_final_reliability_floors'],np.float32);fdomain=np.asarray(artifact['floor_domain'],np.float32);floor_z=(2*(floors-fdomain[0])/(fdomain[1]-fdomain[0])-1).astype(np.float32);heldout_floor_z=(2*(hf-fdomain[0])/(fdomain[1]-fdomain[0])-1).astype(np.float32);tails=np.asarray(c['training_tail_masses'],np.float32);ht=np.asarray(c['heldout_tail_masses'],np.float32);fractions=np.asarray(c['training_attainable_budget_fractions'],np.float32);heldout_fractions=np.asarray(c['heldout_attainable_budget_fractions'],np.float32);chunk=int(c['inference_chunk_size']);steps=int(c['teacher_bisection_steps']);source_target,_,_=_target_prices(policy,sg,alphas,beta_z,floor_z,tails,fractions,steps,chunk);dev=sgs%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout,source_low,source_high=_target_prices(policy,sg[dev],ha,heldout_beta_z,heldout_floor_z,ht,heldout_fractions,steps,chunk);target201,low201,high201=_target_prices(policy,g201,ha,heldout_beta_z,heldout_floor_z,ht,heldout_fractions,steps,chunk);summary=np.concatenate((sg.mean(1),sg.std(1)),1).astype(np.float32);x=torch.from_numpy(summary).cuda();target=torch.from_numpy(source_target).cuda();fz=torch.from_numpy(2*fractions-1).cuda();at=torch.from_numpy(alphas).cuda();bt=torch.from_numpy(beta_z).cuda();ft=torch.from_numpy(floor_z).cuda();qt=torch.from_numpy(tails).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=EpistemicTailCVaRGroupDual(int(m['width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');ei=torch.randint(len(betas),(len(idx),),device='cuda');li=torch.randint(len(floors),(len(idx),),device='cuda');qi=torch.randint(len(tails),(len(idx),),device='cuda');fi=torch.randint(len(fractions),(len(idx),),device='cuda');prediction=student(x[idx],fz[fi],at[ai],bt[ei],ft[li],qt[qi]);loss=F.l1_loss(prediction,target[idx,ai,ei,li,qi,fi]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P280 composite group dual step={step+1} price_mae={last:.7f}',flush=True)
 mean=float(artifact['shadow_price_log_mean']);scale=float(artifact['shadow_price_log_scale']);epsilon=float(artifact['alpha_utility_epsilon']);penalty=float(artifact['tail_CVaR_shortfall_penalty']);source=_evaluate(student,policy,teacher,sg[dev],source_heldout,source_low,source_high,ha,heldout_beta_z,hf,heldout_floor_z,ht,heldout_fractions,mean,scale,epsilon,penalty,chunk);r201=_evaluate(student,policy,teacher,g201,target201,low201,high201,ha,heldout_beta_z,hf,heldout_floor_z,ht,heldout_fractions,mean,scale,epsilon,penalty,chunk);decision=c['decision'];checks={'P201_budget_constraint_fidelity':r201['attained_budget_fraction_MAE']<=float(decision['maximum_P201_attained_budget_fraction_MAE']),'P201_composite_group_regret':r201['mean_frozen_composite_Lagrangian_group_utility_regret']<=float(decision['maximum_P201_mean_frozen_composite_Lagrangian_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'rate_knot_count':m['rate_knot_count'],'base_model':c['frozen_p279']},d/c['model_artifact']);summary_out={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'group_count':int((~dev).sum()),'group_size':size,'final_normalized_log_price_mae':last},'source_development':source,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary_out,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary_out},indent=2))


if __name__=='__main__':main()
