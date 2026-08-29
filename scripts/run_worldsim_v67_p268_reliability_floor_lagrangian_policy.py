"""Train a bi-monotone shadow-price policy with a soft final-reliability floor."""

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


class ReliabilityFloorLagrangianPolicy(nn.Module):
 def __init__(self,width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.encoder=nn.Sequential(nn.Linear(37,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.intercept=nn.Linear(width,1);self.price_rates=nn.Linear(width,self.knot_count);self.floor_rates=nn.Linear(width,self.knot_count)
 def _integral(self,rates,value):
  rates=F.softplus(rates);width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((value+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);return base+width*(r0*fraction+.5*(r1-r0)*fraction.square())
 def forward(self,feature,normalized_log_price,alpha,normalized_floor):
  context=self.encoder(torch.cat((feature,alpha[:,None]),1));return torch.tanh(self.intercept(context).squeeze(1)-self._integral(self.price_rates(context),normalized_log_price)+self._integral(self.floor_rates(context),normalized_floor))


def _alpha_utility(probability,alpha,epsilon):
 shifted=probability+epsilon
 if abs(float(alpha)-1)<1e-7:return np.log(shifted).mean(-1)
 power=1-float(alpha);return ((np.power(shifted,power)-1)/power).mean(-1)


def _grid_probability(model,feature,grid_z,chunk):
 outputs=[]
 with torch.no_grad():
  for start in range(0,len(feature),chunk):
   x=torch.from_numpy(feature[start:start+chunk]).cuda();outputs.append(torch.stack([model(x,torch.full((len(x),),float(value),device='cuda')) for value in grid_z],1).cpu().numpy())
 return np.concatenate(outputs)


def _optimal_budget(model,feature,alphas,floors,prices,grid_z,epsilon,penalty,chunk):
 probability=_grid_probability(model,feature,grid_z,chunk);cost=(grid_z+1)/2;by_alpha=[]
 for alpha in alphas:
  utility=_alpha_utility(probability,alpha,epsilon);by_floor=[]
  for floor in floors:
   shortfall=np.maximum(float(floor)-probability[:,:,-1],0);objective=utility[:,:,None]-prices[None,None,:]*cost[None,:,None]-float(penalty)*shortfall[:,:,None];by_floor.append(grid_z[np.argmax(objective,axis=1)])
  by_alpha.append(np.stack(by_floor,1))
 return np.stack(by_alpha,1)


def _evaluate(student,teacher,feature,target,alphas,floors,floor_z,prices,price_z,epsilon,penalty):
 x=torch.from_numpy(feature).cuda();candidate=[];candidate_value=[];target_value=[];candidate_shortfall=[];target_shortfall=[];torch.cuda.synchronize();started=time.monotonic()
 with torch.no_grad():
  for ai,alpha in enumerate(alphas):
   by_floor=[];by_cvalue=[];by_tvalue=[];by_cs=[];by_ts=[]
   for fi,floor in enumerate(floors):
    row=[];crow=[];trow=[];csrow=[];tsrow=[];fv=torch.full((len(x),),float(floor_z[fi]),device='cuda')
    for pi,value in enumerate(price_z):
     av=torch.full((len(x),),float(alpha),device='cuda');prediction=student(x,torch.full((len(x),),float(value),device='cuda'),av,fv);truth=torch.from_numpy(target[:,ai,fi,pi]).cuda();cp=teacher(x,prediction).cpu().numpy();tp=teacher(x,truth).cpu().numpy();cs=np.maximum(float(floor)-cp[:,-1],0);ts=np.maximum(float(floor)-tp[:,-1],0);row.append(prediction.cpu().numpy());crow.append(_alpha_utility(cp,alpha,epsilon)-prices[pi]*(prediction.cpu().numpy()+1)/2-penalty*cs);trow.append(_alpha_utility(tp,alpha,epsilon)-prices[pi]*(target[:,ai,fi,pi]+1)/2-penalty*ts);csrow.append(cs);tsrow.append(ts)
    by_floor.append(np.stack(row,1));by_cvalue.append(np.stack(crow,1));by_tvalue.append(np.stack(trow,1));by_cs.append(np.stack(csrow,1));by_ts.append(np.stack(tsrow,1))
   candidate.append(np.stack(by_floor,1));candidate_value.append(np.stack(by_cvalue,1));target_value.append(np.stack(by_tvalue,1));candidate_shortfall.append(np.stack(by_cs,1));target_shortfall.append(np.stack(by_ts,1))
 torch.cuda.synchronize();forward=time.monotonic()-started;candidate=np.stack(candidate,1);candidate_value=np.stack(candidate_value,1);target_value=np.stack(target_value,1);candidate_shortfall=np.stack(candidate_shortfall,1);target_shortfall=np.stack(target_shortfall,1);return {'trajectory_count':int(len(feature)),'alpha_count':int(len(alphas)),'floor_count':int(len(floors)),'price_count':int(len(prices)),'normalized_log_budget_MAE':float(np.mean(np.abs(candidate-target))),'mean_frozen_floor_Lagrangian_utility_regret':float(np.mean(target_value-candidate_value)),'mean_candidate_final_reliability_shortfall':float(np.mean(candidate_shortfall)),'mean_teacher_final_reliability_shortfall':float(np.mean(target_shortfall)),'price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=3)>1e-7)),'floor_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,scenes,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,_,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,_,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);frozen=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(frozen['context_width']),int(frozen['rate_knot_count'])).cuda();teacher.load_state_dict(frozen['model_state_dict']);teacher.eval();domain=np.asarray(c['shadow_price_domain'],np.float32);training_prices=np.exp(np.linspace(np.log(domain[0]),np.log(domain[1]),int(c['training_price_count']),dtype=np.float32));heldout_prices=np.sqrt(training_prices[:-1]*training_prices[1:]);mean=.5*(np.log(domain[0])+np.log(domain[1]));scale=.5*(np.log(domain[1])-np.log(domain[0]));training_price_z=((np.log(training_prices)-mean)/scale).astype(np.float32);heldout_price_z=((np.log(heldout_prices)-mean)/scale).astype(np.float32);alphas=np.asarray(c['training_alpha_fairness'],np.float32);ha=np.asarray(c['heldout_alpha_fairness'],np.float32);floors=np.asarray(c['training_final_reliability_floors'],np.float32);hf=np.asarray(c['heldout_final_reliability_floors'],np.float32);floor_domain=np.asarray(c['final_reliability_floor_domain'],np.float32);floor_z=(2*(floors-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);heldout_floor_z=(2*(hf-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);grid_z=np.linspace(-1,1,int(c['teacher_budget_grid_count']),dtype=np.float32);epsilon=float(c['alpha_utility_epsilon']);penalty=float(c['final_reliability_shortfall_penalty']);chunk=int(c['teacher_chunk_size']);source_target=_optimal_budget(teacher,sf,alphas,floors,training_prices,grid_z,epsilon,penalty,chunk);target183=_optimal_budget(teacher,f183,ha,hf,heldout_prices,grid_z,epsilon,penalty,chunk);target201=_optimal_budget(teacher,f201,ha,hf,heldout_prices,grid_z,epsilon,penalty,chunk);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout=_optimal_budget(teacher,sf[dev],ha,hf,heldout_prices,grid_z,epsilon,penalty,chunk);x=torch.from_numpy(sf).cuda();target=torch.from_numpy(source_target).cuda();pz=torch.from_numpy(training_price_z).cuda();at=torch.from_numpy(alphas).cuda();ft=torch.from_numpy(floor_z).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=ReliabilityFloorLagrangianPolicy(int(m['width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');fi=torch.randint(len(floors),(len(idx),),device='cuda');pi=torch.randint(len(training_prices),(len(idx),),device='cuda');prediction=student(x[idx],pz[pi],at[ai],ft[fi]);loss=F.l1_loss(prediction,target[idx,ai,fi,pi]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P268 reliability-floor policy step={step+1} budget_mae={last:.7f}',flush=True)
 source=_evaluate(student,teacher,sf[dev],source_heldout,ha,hf,heldout_floor_z,heldout_prices,heldout_price_z,epsilon,penalty);r183=_evaluate(student,teacher,f183,target183,ha,hf,heldout_floor_z,heldout_prices,heldout_price_z,epsilon,penalty);r201=_evaluate(student,teacher,f201,target201,ha,hf,heldout_floor_z,heldout_prices,heldout_price_z,epsilon,penalty);decision=c['decision'];checks={'P201_budget_fidelity':r201['normalized_log_budget_MAE']<=float(decision['maximum_P201_normalized_log_budget_MAE']),'P201_frozen_floor_Lagrangian_utility_regret':r201['mean_frozen_floor_Lagrangian_utility_regret']<=float(decision['maximum_P201_mean_frozen_floor_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'rate_knot_count':m['rate_knot_count'],'input_dimension':39,'shadow_price_log_mean':mean,'shadow_price_log_scale':scale,'floor_domain':floor_domain,'alpha_utility_epsilon':epsilon,'shortfall_penalty':penalty,'base_model':c['frozen_p246']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'alpha_count':int(len(alphas)),'floor_count':int(len(floors)),'price_count':int(len(training_prices)),'final_normalized_log_budget_mae':last},'heldout_alpha_fairness':[float(v) for v in ha],'heldout_final_reliability_floors':[float(v) for v in hf],'heldout_shadow_prices':[float(v) for v in heldout_prices],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
