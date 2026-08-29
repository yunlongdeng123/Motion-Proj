"""Compile attainable budgets into composite-risk prices for variable sets."""

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
from scripts.run_worldsim_v67_p279_epistemic_tail_cvar_allocator import EpistemicTailCVaRAllocator
from scripts.run_worldsim_v67_p280_epistemic_tail_cvar_group_dual import _allocate,_target_prices,_risk


class VariableSetCompositeDual(nn.Module):
 def __init__(self,element_width,context_width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.element=nn.Sequential(nn.Linear(36,element_width),nn.SiLU(),nn.Linear(element_width,element_width),nn.SiLU());self.context=nn.Sequential(nn.Linear(3*element_width+5,context_width),nn.SiLU(),nn.Linear(context_width,context_width),nn.SiLU());self.intercept=nn.Linear(context_width,1);self.rate_knots=nn.Linear(context_width,self.knot_count)
 def forward(self,groups,fraction,alpha,beta,floor,tail_mass):
  encoded=self.element(groups);mean=encoded.mean(1);std=torch.sqrt(encoded.var(1,unbiased=False)+1e-6);maximum=encoded.amax(1);size=torch.full_like(alpha,float(groups.shape[1])).log2()/7.;context=self.context(torch.cat((mean,std,maximum,size[:,None],alpha[:,None],beta[:,None],floor[:,None],tail_mass[:,None]),1));rates=F.softplus(self.rate_knots(context));width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((fraction+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);part=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);return torch.tanh(self.intercept(context).squeeze(1)-base-width*(r0*part+.5*(r1-r0)*part.square()))


def _predict(student,groups,alphas,beta_z,floor_z,tails,fractions):
 x=torch.from_numpy(groups).cuda();by_alpha=[]
 with torch.no_grad():
  for alpha in alphas:
   by_beta=[]
   for beta in beta_z:
    by_floor=[]
    for floor in floor_z:
     by_tail=[]
     for tail in tails:by_tail.append(np.stack([student(x,torch.full((len(x),),float(2*fraction-1),device='cuda'),torch.full((len(x),),float(alpha),device='cuda'),torch.full((len(x),),float(beta),device='cuda'),torch.full((len(x),),float(floor),device='cuda'),torch.full((len(x),),float(tail),device='cuda')).cpu().numpy() for fraction in fractions],1))
     by_floor.append(np.stack(by_tail,1))
    by_beta.append(np.stack(by_floor,1))
   by_alpha.append(np.stack(by_beta,1))
 return np.stack(by_alpha,1)


def _evaluate(student,policy,teacher,groups,target,low,high,alphas,beta_z,floors,floor_z,tails,fractions,mean,scale,epsilon,penalty,chunk):
 torch.cuda.synchronize();started=time.monotonic();candidate=_predict(student,groups,alphas,beta_z,floor_z,tails,fractions);torch.cuda.synchronize();forward=time.monotonic()-started;cb=_allocate(policy,groups,candidate.astype(np.float32),alphas,beta_z,floor_z,tails,chunk);tb=_allocate(policy,groups,target.astype(np.float32),alphas,beta_z,floor_z,tails,chunk);cc=np.mean((cb+1)/2,6);tc=np.mean((tb+1)/2,6);attained=(cc-low[:,:,:,:,:,None])/(high-low)[:,:,:,:,:,None].clip(min=1e-6);cv=_risk(teacher,groups,cb.astype(np.float32),alphas,beta_z,floors,tails,epsilon,penalty,chunk);tv=_risk(teacher,groups,tb.astype(np.float32),alphas,beta_z,floors,tails,epsilon,penalty,chunk);price=np.exp(mean+scale*target);regret=(tv-price*tc)-(cv-price*cc);return {'group_count':int(len(groups)),'group_size':int(groups.shape[1]),'normalized_log_price_MAE':float(np.mean(np.abs(candidate-target))),'attained_budget_fraction_MAE':float(np.mean(np.abs(attained-fractions[None,None,None,None,None,:]))),'mean_frozen_composite_Lagrangian_group_utility_regret':float(np.mean(regret)),'fraction_price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=5)>1e-7)),'student_forward_seconds':forward}


def _aggregate(rows):
 keys=('normalized_log_price_MAE','attained_budget_fraction_MAE','mean_frozen_composite_Lagrangian_group_utility_regret');return {'evaluated_group_sizes':[int(k) for k in rows],**{key:float(np.mean([v[key] for v in rows.values()])) for key in keys},'fraction_price_monotonicity_violations':int(sum(v['fraction_price_monotonicity_violations'] for v in rows.values())),'by_group_size':rows}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,ss,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f201,_,_,s201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);surface=torch.load(a.runs_root/c['frozen_p275']['run']/c['frozen_p275']['artifact'],map_location='cuda');teacher=EpistemicLCBSurface(int(surface['context_width']),int(surface['budget_rate_knot_count']),int(surface['beta_rate_knot_count'])).cuda();teacher.load_state_dict(surface['model_state_dict']);teacher.eval();artifact=torch.load(a.runs_root/c['frozen_p281']['run']/c['frozen_p281']['artifact'],map_location='cuda');policy=EpistemicTailCVaRAllocator(int(artifact['element_width']),int(artifact['context_width']),int(artifact['rate_knot_count'])).cuda();policy.load_state_dict(artifact['model_state_dict']);policy.eval();alphas=np.asarray(c['training_alpha_fairness'],np.float32);ha=np.asarray(c['heldout_alpha_fairness'],np.float32);betas=np.asarray(c['training_epistemic_betas'],np.float32);hb=np.asarray(c['heldout_epistemic_betas'],np.float32);bd=np.asarray(artifact['epistemic_beta_domain'],np.float32);beta_z=(2*(betas-bd[0])/(bd[1]-bd[0])-1).astype(np.float32);hbeta_z=(2*(hb-bd[0])/(bd[1]-bd[0])-1).astype(np.float32);floors=np.asarray(c['training_final_reliability_floors'],np.float32);hf=np.asarray(c['heldout_final_reliability_floors'],np.float32);fdomain=np.asarray(artifact['floor_domain'],np.float32);floor_z=(2*(floors-fdomain[0])/(fdomain[1]-fdomain[0])-1).astype(np.float32);hfloor_z=(2*(hf-fdomain[0])/(fdomain[1]-fdomain[0])-1).astype(np.float32);tails=np.asarray(c['training_tail_masses'],np.float32);ht=np.asarray(c['heldout_tail_masses'],np.float32);fractions=np.asarray(c['training_attainable_budget_fractions'],np.float32);hfractions=np.asarray(c['heldout_attainable_budget_fractions'],np.float32);chunk=int(c['inference_chunk_size']);steps=int(c['teacher_bisection_steps']);train={}
 for size in map(int,c['training_group_sizes']):
  groups,scenes=_groups(sf,ss,size);target,_,_=_target_prices(policy,groups,alphas,beta_z,floor_z,tails,fractions,steps,chunk);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train[size]=(torch.from_numpy(groups).cuda(),torch.from_numpy(target).cuda(),torch.from_numpy(np.flatnonzero(~dev)).cuda())
 fz=torch.from_numpy(2*fractions-1).cuda();at=torch.from_numpy(alphas).cuda();bt=torch.from_numpy(beta_z).cuda();ft=torch.from_numpy(floor_z).cuda();qt=torch.from_numpy(tails).cuda();m=c['student'];student=VariableSetCompositeDual(int(m['element_width']),int(m['context_width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));sizes=list(train);last=0.
 for step in range(int(m['steps'])):
  size=sizes[step%len(sizes)];x,target,index=train[size];idx=index[torch.randint(len(index),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');ei=torch.randint(len(betas),(len(idx),),device='cuda');li=torch.randint(len(floors),(len(idx),),device='cuda');qi=torch.randint(len(tails),(len(idx),),device='cuda');fi=torch.randint(len(fractions),(len(idx),),device='cuda');prediction=student(x[idx],fz[fi],at[ai],bt[ei],ft[li],qt[qi]);loss=F.l1_loss(prediction,target[idx,ai,ei,li,qi,fi]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P282 variable composite dual step={step+1} size={size} price_mae={last:.7f}',flush=True)
 mean=float(artifact['shadow_price_log_mean']);scale=float(artifact['shadow_price_log_scale']);epsilon=float(artifact['alpha_utility_epsilon']);penalty=float(artifact['tail_CVaR_shortfall_penalty']);source={};r201={}
 for size in map(int,c['heldout_group_sizes']):
  groups,scenes=_groups(sf,ss,size);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);target,low,high=_target_prices(policy,groups[dev],ha,hbeta_z,hfloor_z,ht,hfractions,steps,chunk);source[str(size)]=_evaluate(student,policy,teacher,groups[dev],target,low,high,ha,hbeta_z,hf,hfloor_z,ht,hfractions,mean,scale,epsilon,penalty,chunk);groups,_=_groups(f201,s201,size);target,low,high=_target_prices(policy,groups,ha,hbeta_z,hfloor_z,ht,hfractions,steps,chunk);r201[str(size)]=_evaluate(student,policy,teacher,groups,target,low,high,ha,hbeta_z,hf,hfloor_z,ht,hfractions,mean,scale,epsilon,penalty,chunk)
 source=_aggregate(source);r201=_aggregate(r201);decision=c['decision'];checks={'P201_variable_set_budget_constraint_fidelity':r201['attained_budget_fraction_MAE']<=float(decision['maximum_P201_attained_budget_fraction_MAE']),'P201_variable_set_composite_regret':r201['mean_frozen_composite_Lagrangian_group_utility_regret']<=float(decision['maximum_P201_mean_frozen_composite_Lagrangian_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'element_width':m['element_width'],'context_width':m['context_width'],'rate_knot_count':m['rate_knot_count'],'base_model':c['frozen_p281']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'group_sizes':sizes,'final_normalized_log_price_mae':last},'heldout_group_sizes':list(map(int,c['heldout_group_sizes'])),'source_development':source,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
