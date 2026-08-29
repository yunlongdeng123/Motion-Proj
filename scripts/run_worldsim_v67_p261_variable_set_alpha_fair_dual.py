"""Train a permutation-invariant alpha-fair dual compiler across group sizes."""

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
from scripts.run_worldsim_v67_p259_alpha_fair_shadow_price_policy import AlphaFairShadowPricePolicy
from scripts.run_worldsim_v67_p260_alpha_fair_group_dual import _allocate,_target_prices,_utility


class VariableSetAlphaFairDual(nn.Module):
 def __init__(self,element_width,context_width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.element=nn.Sequential(nn.Linear(36,element_width),nn.SiLU(),nn.Linear(element_width,element_width),nn.SiLU());self.context=nn.Sequential(nn.Linear(3*element_width+2,context_width),nn.SiLU(),nn.Linear(context_width,context_width),nn.SiLU());self.intercept=nn.Linear(context_width,1);self.rate_knots=nn.Linear(context_width,self.knot_count)
 def forward(self,groups,mask,normalized_fraction,alpha):
  encoded=self.element(groups);weight=mask[:,:,None];count=weight.sum(1).clamp(min=1);mean=(encoded*weight).sum(1)/count;variance=((encoded-mean[:,None]).square()*weight).sum(1)/count;maximum=encoded.masked_fill(~mask[:,:,None],-torch.inf).amax(1);size_feature=torch.log2(count.squeeze(1))/7.;context=self.context(torch.cat((mean,torch.sqrt(variance+1e-6),maximum,size_feature[:,None],alpha[:,None]),1));rates=F.softplus(self.rate_knots(context));width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((normalized_fraction+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);integral=base+width*(r0*fraction+.5*(r1-r0)*fraction.square());return torch.tanh(self.intercept(context).squeeze(1)-integral)


def _pad(group_sets):
 maximum=max(x.shape[1] for x in group_sets);total=sum(len(x) for x in group_sets);out=np.zeros((total,maximum,36),np.float32);mask=np.zeros((total,maximum),bool);sizes=[];offset=0
 for groups in group_sets:
  n,size=len(groups),groups.shape[1];out[offset:offset+n,:size]=groups;mask[offset:offset+n,:size]=True;sizes.extend([size]*n);offset+=n
 return out,mask,np.asarray(sizes,np.int64)


def _predict(student,groups,mask,alphas,fractions,batch):
 result=[]
 with torch.no_grad():
  for alpha in alphas:
   columns=[]
   for fraction in fractions:
    values=[]
    for start in range(0,len(groups),batch):
     stop=start+batch;g=torch.from_numpy(groups[start:stop]).cuda();m=torch.from_numpy(mask[start:stop]).cuda();values.append(student(g,m,torch.full((len(g),),float(2*fraction-1),device='cuda'),torch.full((len(g),),float(alpha),device='cuda')).cpu().numpy())
    columns.append(np.concatenate(values))
   result.append(np.stack(columns,1))
 return np.stack(result,1)


def _evaluate_size(student,policy,teacher,groups,target,low,high,alphas,fractions,price_mean,price_scale,epsilon,chunk,batch):
 padded,mask,_=_pad([groups]);torch.cuda.synchronize();started=time.monotonic();candidate=_predict(student,padded,mask,alphas,fractions,batch);torch.cuda.synchronize();forward=time.monotonic()-started;candidate_budget=_allocate(policy,groups,candidate.astype(np.float32),alphas,chunk);target_budget=_allocate(policy,groups,target.astype(np.float32),alphas,chunk);candidate_cost=np.mean((candidate_budget+1)/2,3);target_cost=np.mean((target_budget+1)/2,3);attained=(candidate_cost-low[:,:,None])/(high-low)[:,:,None].clip(min=1e-6);candidate_reward=_utility(teacher,groups,candidate_budget.astype(np.float32),alphas,epsilon,chunk);target_reward=_utility(teacher,groups,target_budget.astype(np.float32),alphas,epsilon,chunk);price=np.exp(price_mean+price_scale*target);regret=(target_reward-price*target_cost)-(candidate_reward-price*candidate_cost);return {'group_count':int(len(groups)),'group_size':int(groups.shape[1]),'normalized_log_price_MAE':float(np.mean(np.abs(candidate-target))),'attained_budget_fraction_MAE':float(np.mean(np.abs(attained-fractions[None,None,:]))),'mean_frozen_alpha_fair_Lagrangian_utility_regret':float(np.mean(regret)),'fraction_price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)>1e-7)),'student_forward_seconds':forward}


def _aggregate(rows):
 weights=np.asarray([r['group_count'] for r in rows],np.float64);total=float(weights.sum());keys=['normalized_log_price_MAE','attained_budget_fraction_MAE','mean_frozen_alpha_fair_Lagrangian_utility_regret'];result={k:float(sum(w*r[k] for w,r in zip(weights,rows))/total) for k in keys};result.update({'group_count':int(total),'heldout_group_sizes':[r['group_size'] for r in rows],'fraction_price_monotonicity_violations':int(sum(r['fraction_price_monotonicity_violations'] for r in rows)),'per_group_size':rows});return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,ss,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,s183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,s201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);forward=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(forward['context_width']),int(forward['rate_knot_count'])).cuda();teacher.load_state_dict(forward['model_state_dict']);teacher.eval();artifact=torch.load(a.runs_root/c['frozen_p259']['run']/c['frozen_p259']['artifact'],map_location='cuda');policy=AlphaFairShadowPricePolicy(int(artifact['width']),int(artifact['rate_knot_count'])).cuda();policy.load_state_dict(artifact['model_state_dict']);policy.eval();alphas=np.asarray(c['training_alpha_fairness'],np.float32);heldout_alphas=np.asarray(c['heldout_alpha_fairness'],np.float32);fractions=np.asarray(c['training_attainable_budget_fractions'],np.float32);heldout_fractions=np.asarray(c['heldout_attainable_budget_fractions'],np.float32);chunk=int(c['inference_chunk_size']);steps=int(c['teacher_bisection_steps']);train_sizes=[int(v) for v in c['training_group_sizes']];heldout_sizes=[int(v) for v in c['heldout_group_sizes']];source_sets=[];scene_sets=[];target_sets=[]
 for size in train_sizes:
  groups,scenes=_groups(sf,ss,size);target,_,_=_target_prices(policy,groups,alphas,fractions,steps,chunk);source_sets.append(groups);scene_sets.append(scenes);target_sets.append(target)
 padded,mask,size_labels=_pad(source_sets);scene_labels=np.concatenate(scene_sets);targets=np.concatenate(target_sets);development=scene_labels%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);m=c['student'];student=VariableSetAlphaFairDual(int(m['element_width']),int(m['context_width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));pg=torch.from_numpy(padded).cuda();pm=torch.from_numpy(mask).cuda();pt=torch.from_numpy(targets).cuda();fz=torch.from_numpy(2*fractions-1).cuda();at=torch.from_numpy(alphas).cuda();train_idx=torch.from_numpy(np.flatnonzero(~development)).cuda();last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');fi=torch.randint(len(fractions),(len(idx),),device='cuda');prediction=student(pg[idx],pm[idx],fz[fi],at[ai]);loss=F.l1_loss(prediction,pt[idx,ai,fi]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P261 variable-set dual step={step+1} price_mae={last:.7f}',flush=True)
 mean=float(artifact['shadow_price_log_mean']);scale=float(artifact['shadow_price_log_scale']);epsilon=float(artifact['alpha_utility_epsilon']);batch=int(c['student_inference_batch_size']);collections=[]
 for role,feature,scenes in [('source_development',sf,ss),('P183_consumed_development',f183,s183),('P201_post_hoc_development',f201,s201)]:
  rows=[]
  for size in heldout_sizes:
   groups,labels=_groups(feature,scenes,size)
   if role=='source_development':groups=groups[labels%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder'])]
   target,low,high=_target_prices(policy,groups,heldout_alphas,heldout_fractions,steps,chunk);rows.append(_evaluate_size(student,policy,teacher,groups,target,low,high,heldout_alphas,heldout_fractions,mean,scale,epsilon,chunk,batch))
  collections.append(_aggregate(rows))
 source,r183,r201=collections;decision=c['decision'];checks={'P201_variable_size_budget_constraint_fidelity':r201['attained_budget_fraction_MAE']<=float(decision['maximum_P201_attained_budget_fraction_MAE']),'P201_variable_size_alpha_fair_utility_regret':r201['mean_frozen_alpha_fair_Lagrangian_utility_regret']<=float(decision['maximum_P201_mean_frozen_alpha_fair_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'element_width':m['element_width'],'context_width':m['context_width'],'rate_knot_count':m['rate_knot_count'],'training_group_sizes':train_sizes,'base_model':c['frozen_p259']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'group_count':int((~development).sum()),'group_sizes':train_sizes,'final_normalized_log_price_mae':last},'heldout_group_sizes':heldout_sizes,'heldout_alpha_fairness':[float(v) for v in heldout_alphas],'heldout_attainable_budget_fractions':[float(v) for v in heldout_fractions],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
