"""Train a permutation-equivariant allocator for group-tail reliability CVaR."""

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


class TailCVaREquivariantAllocator(nn.Module):
 def __init__(self,element_width,context_width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.element=nn.Sequential(nn.Linear(36,element_width),nn.SiLU(),nn.Linear(element_width,element_width),nn.SiLU());self.context=nn.Sequential(nn.Linear(4*element_width+2,context_width),nn.SiLU(),nn.Linear(context_width,context_width),nn.SiLU());self.intercept=nn.Linear(context_width,1);self.price_rates=nn.Linear(context_width,self.knot_count);self.floor_rates=nn.Linear(context_width,self.knot_count)
 def _integral(self,rates,value):
  rates=F.softplus(rates);width=2./(self.knot_count-1);areas=.5*(rates[:,:,:-1]+rates[:,:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:,:1]),torch.cumsum(areas,2)),2);position=((value+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;r0=torch.gather(rates,2,index[:,:,None]).squeeze(2);r1=torch.gather(rates,2,(index+1)[:,:,None]).squeeze(2);base=torch.gather(cumulative,2,index[:,:,None]).squeeze(2);return base+width*(r0*fraction+.5*(r1-r0)*fraction.square())
 def forward(self,groups,normalized_log_price,alpha,normalized_floor,tail_mass):
  encoded=self.element(groups);mean=encoded.mean(1);std=torch.sqrt(encoded.var(1,unbiased=False)+1e-6);maximum=encoded.amax(1);pooled=torch.cat((mean,std,maximum),1);size=groups.shape[1];context=self.context(torch.cat((encoded,pooled[:,None].expand(-1,size,-1),alpha[:,None,None].expand(-1,size,1),tail_mass[:,None,None].expand(-1,size,1)),2));price=normalized_log_price[:,None].expand(-1,size);floor=normalized_floor[:,None].expand(-1,size);return torch.tanh(self.intercept(context).squeeze(2)-self._integral(self.price_rates(context),price)+self._integral(self.floor_rates(context),floor))


def _grid_probability(model,groups,grid_z,chunk):
 flat=groups.reshape(-1,36);outputs=[]
 with torch.no_grad():
  for start in range(0,len(flat),chunk):
   x=torch.from_numpy(flat[start:start+chunk]).cuda();outputs.append(torch.stack([model(x,torch.full((len(x),),float(value),device='cuda')) for value in grid_z],1).cpu().numpy())
 return np.concatenate(outputs).reshape(len(groups),groups.shape[1],len(grid_z),4)


def _alpha_utility(probability,alpha,epsilon):
 shifted=probability+epsilon
 if abs(float(alpha)-1)<1e-7:return np.log(shifted).mean(-1)
 power=1-float(alpha);return ((np.power(shifted,power)-1)/power).mean(-1)


def _teacher_targets(probability,alphas,floors,tail_masses,prices,grid_z,eta_grid,epsilon,penalty):
 prob=torch.from_numpy(probability).cuda();cost=torch.from_numpy(((grid_z+1)/2).astype(np.float32)).cuda();etas=torch.from_numpy(eta_grid).cuda();targets=np.empty((len(probability),len(alphas),len(floors),len(tail_masses),len(prices),probability.shape[1]),np.float32)
 for ai,alpha in enumerate(alphas):
  utility=torch.from_numpy(_alpha_utility(probability,float(alpha),epsilon)).cuda()
  for li,floor in enumerate(floors):
   shortfall=(float(floor)-prob[:,:,:,-1]).clamp(min=0);hinge=(shortfall[:,None]-etas[None,:,None,None]).clamp(min=0)
   for qi,tail_mass in enumerate(tail_masses):
    penalty_hinge=float(penalty)/float(tail_mass)*hinge
    for pi,price in enumerate(prices):
     objective=utility[:,None]-float(price)*cost[None,None,None,:]-penalty_hinge;best=objective.argmax(3);chosen=torch.gather(objective,3,best[:,:,:,None]).squeeze(3).mean(2)-float(penalty)*etas[None,:];eta_index=chosen.argmax(1);member_index=best[torch.arange(len(best),device='cuda'),eta_index];targets[:,ai,li,qi,pi]=grid_z[member_index.cpu().numpy()]
 return targets


def _infer_probability(model,groups,budget_z,chunk):
 features=np.broadcast_to(groups[:,None,None,None,None,:,:],budget_z.shape+(36,)).reshape(-1,36);budgets=budget_z.reshape(-1);outputs=[]
 with torch.no_grad():
  for start in range(0,len(features),chunk):outputs.append(model(torch.from_numpy(features[start:start+chunk]).cuda(),torch.from_numpy(budgets[start:start+chunk]).cuda()).cpu().numpy())
 return np.concatenate(outputs).reshape(budget_z.shape+(4,))


def _values(probability,budget_z,alphas,floors,tail_masses,prices,epsilon,penalty):
 result=[];cvars=[]
 for ai,alpha in enumerate(alphas):
  by_floor=[];by_floor_cvar=[]
  for li,floor in enumerate(floors):
   by_tail=[];by_tail_cvar=[]
   for qi,tail_mass in enumerate(tail_masses):
    p=probability[:,ai,li,qi];utility=_alpha_utility(p,alpha,epsilon).mean(2);shortfall=np.maximum(float(floor)-p[:,:,:,-1],0);k=max(1,int(np.ceil(float(tail_mass)*shortfall.shape[2])));cvar=np.sort(shortfall,axis=2)[:,:,-k:].mean(2);cost=((budget_z[:,ai,li,qi]+1)/2).mean(2);by_tail.append(utility-prices[None,:]*cost-float(penalty)*cvar);by_tail_cvar.append(cvar)
   by_floor.append(np.stack(by_tail,1));by_floor_cvar.append(np.stack(by_tail_cvar,1))
  result.append(np.stack(by_floor,1));cvars.append(np.stack(by_floor_cvar,1))
 return np.stack(result,1),np.stack(cvars,1)


def _evaluate(student,teacher,groups,target,alphas,floors,floor_z,tail_masses,prices,price_z,epsilon,penalty,chunk):
 x=torch.from_numpy(groups).cuda();candidate=[];torch.cuda.synchronize();started=time.monotonic()
 with torch.no_grad():
  for alpha in alphas:
   by_floor=[]
   for value in floor_z:
    by_tail=[]
    for tail_mass in tail_masses:by_tail.append(np.stack([student(x,torch.full((len(x),),float(price),device='cuda'),torch.full((len(x),),float(alpha),device='cuda'),torch.full((len(x),),float(value),device='cuda'),torch.full((len(x),),float(tail_mass),device='cuda')).cpu().numpy() for price in price_z],1))
    by_floor.append(np.stack(by_tail,1))
   candidate.append(np.stack(by_floor,1))
 torch.cuda.synchronize();forward=time.monotonic()-started;candidate=np.stack(candidate,1);candidate_probability=_infer_probability(teacher,groups,candidate.astype(np.float32),chunk);target_probability=_infer_probability(teacher,groups,target.astype(np.float32),chunk);candidate_value,candidate_cvar=_values(candidate_probability,candidate,alphas,floors,tail_masses,prices,epsilon,penalty);target_value,target_cvar=_values(target_probability,target,alphas,floors,tail_masses,prices,epsilon,penalty);return {'group_count':int(len(groups)),'group_size':int(groups.shape[1]),'alpha_count':int(len(alphas)),'floor_count':int(len(floors)),'tail_mass_count':int(len(tail_masses)),'price_count':int(len(prices)),'normalized_log_budget_MAE':float(np.mean(np.abs(candidate-target))),'mean_frozen_tail_CVaR_Lagrangian_utility_regret':float(np.mean(target_value-candidate_value)),'mean_candidate_tail_CVaR_shortfall':float(np.mean(candidate_cvar)),'mean_teacher_tail_CVaR_shortfall':float(np.mean(target_cvar)),'price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=4)>1e-7)),'floor_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,ss,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,s183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,s201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);size=int(c['group_size']);sg,sgs=_groups(sf,ss,size);g183,_=_groups(f183,s183,size);g201,_=_groups(f201,s201,size);frozen=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(frozen['context_width']),int(frozen['rate_knot_count'])).cuda();teacher.load_state_dict(frozen['model_state_dict']);teacher.eval();domain=np.asarray(c['shadow_price_domain'],np.float32);prices=np.exp(np.linspace(np.log(domain[0]),np.log(domain[1]),int(c['training_price_count']),dtype=np.float32));heldout_prices=np.sqrt(prices[:-1]*prices[1:]);mean=.5*(np.log(domain[0])+np.log(domain[1]));scale=.5*(np.log(domain[1])-np.log(domain[0]));price_z=((np.log(prices)-mean)/scale).astype(np.float32);heldout_price_z=((np.log(heldout_prices)-mean)/scale).astype(np.float32);alphas=np.asarray(c['training_alpha_fairness'],np.float32);ha=np.asarray(c['heldout_alpha_fairness'],np.float32);floors=np.asarray(c['training_final_reliability_floors'],np.float32);hf=np.asarray(c['heldout_final_reliability_floors'],np.float32);floor_domain=np.asarray(c['final_reliability_floor_domain'],np.float32);floor_z=(2*(floors-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);heldout_floor_z=(2*(hf-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);tail_masses=np.asarray(c['training_tail_masses'],np.float32);heldout_tail_masses=np.asarray(c['heldout_tail_masses'],np.float32);grid_z=np.linspace(-1,1,int(c['teacher_budget_grid_count']),dtype=np.float32);eta_grid=np.linspace(0,float(floor_domain[1]),int(c['teacher_eta_grid_count']),dtype=np.float32);epsilon=float(c['alpha_utility_epsilon']);penalty=float(c['tail_CVaR_shortfall_penalty']);chunk=int(c['teacher_chunk_size']);source_probability=_grid_probability(teacher,sg,grid_z,chunk);source_target=_teacher_targets(source_probability,alphas,floors,tail_masses,prices,grid_z,eta_grid,epsilon,penalty);dev=sgs%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout_target=_teacher_targets(source_probability[dev],ha,hf,heldout_tail_masses,heldout_prices,grid_z,eta_grid,epsilon,penalty);target183=_teacher_targets(_grid_probability(teacher,g183,grid_z,chunk),ha,hf,heldout_tail_masses,heldout_prices,grid_z,eta_grid,epsilon,penalty);target201=_teacher_targets(_grid_probability(teacher,g201,grid_z,chunk),ha,hf,heldout_tail_masses,heldout_prices,grid_z,eta_grid,epsilon,penalty);x=torch.from_numpy(sg).cuda();target=torch.from_numpy(source_target).cuda();pz=torch.from_numpy(price_z).cuda();at=torch.from_numpy(alphas).cuda();ft=torch.from_numpy(floor_z).cuda();qt=torch.from_numpy(tail_masses).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=TailCVaREquivariantAllocator(int(m['element_width']),int(m['context_width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');li=torch.randint(len(floors),(len(idx),),device='cuda');qi=torch.randint(len(tail_masses),(len(idx),),device='cuda');pi=torch.randint(len(prices),(len(idx),),device='cuda');prediction=student(x[idx],pz[pi],at[ai],ft[li],qt[qi]);truth=target[idx,ai,li,qi,pi];loss=F.l1_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P270 tail-CVaR allocator step={step+1} budget_mae={last:.7f}',flush=True)
 source=_evaluate(student,teacher,sg[dev],source_heldout_target,ha,hf,heldout_floor_z,heldout_tail_masses,heldout_prices,heldout_price_z,epsilon,penalty,chunk);r183=_evaluate(student,teacher,g183,target183,ha,hf,heldout_floor_z,heldout_tail_masses,heldout_prices,heldout_price_z,epsilon,penalty,chunk);r201=_evaluate(student,teacher,g201,target201,ha,hf,heldout_floor_z,heldout_tail_masses,heldout_prices,heldout_price_z,epsilon,penalty,chunk);decision=c['decision'];checks={'P201_budget_allocation_fidelity':r201['normalized_log_budget_MAE']<=float(decision['maximum_P201_normalized_log_budget_MAE']),'P201_frozen_tail_CVaR_Lagrangian_regret':r201['mean_frozen_tail_CVaR_Lagrangian_utility_regret']<=float(decision['maximum_P201_mean_frozen_tail_CVaR_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'element_width':m['element_width'],'context_width':m['context_width'],'rate_knot_count':m['rate_knot_count'],'input_dimension_per_member':36,'shadow_price_log_mean':mean,'shadow_price_log_scale':scale,'floor_domain':floor_domain,'alpha_utility_epsilon':epsilon,'tail_CVaR_shortfall_penalty':penalty,'base_model':c['frozen_p246']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'group_count':int((~dev).sum()),'group_size':size,'alpha_count':int(len(alphas)),'floor_count':int(len(floors)),'tail_mass_count':int(len(tail_masses)),'price_count':int(len(prices)),'final_normalized_log_budget_mae':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
