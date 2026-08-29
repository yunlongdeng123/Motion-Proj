"""Train a shadow-price policy over the epistemic-aversion LCB surface."""

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
from scripts.run_worldsim_v67_p275_epistemic_lcb_surface import EpistemicLCBSurface


class EpistemicLCBLagrangianPolicy(nn.Module):
 def __init__(self,width,knot_count):
  super().__init__();self.knot_count=int(knot_count);self.encoder=nn.Sequential(nn.Linear(38,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.intercept=nn.Linear(width,1);self.price_rates=nn.Linear(width,self.knot_count);self.floor_rates=nn.Linear(width,self.knot_count)
 def _integral(self,rates,value):
  rates=F.softplus(rates);width=2./(self.knot_count-1);areas=.5*(rates[:,:-1]+rates[:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:1]),torch.cumsum(areas,1)),1);position=((value+1)/width).clamp(0,self.knot_count-1);index=torch.floor(position).long().clamp(max=self.knot_count-2);fraction=position-index;r0=torch.gather(rates,1,index[:,None]).squeeze(1);r1=torch.gather(rates,1,(index+1)[:,None]).squeeze(1);base=torch.gather(cumulative,1,index[:,None]).squeeze(1);return base+width*(r0*fraction+.5*(r1-r0)*fraction.square())
 def forward(self,feature,normalized_log_price,alpha,normalized_beta,normalized_floor):
  context=self.encoder(torch.cat((feature,alpha[:,None],normalized_beta[:,None]),1));return torch.tanh(self.intercept(context).squeeze(1)-self._integral(self.price_rates(context),normalized_log_price)+self._integral(self.floor_rates(context),normalized_floor))


def _alpha_utility(probability,alpha,epsilon):
 shifted=probability+epsilon
 if abs(float(alpha)-1)<1e-7:return np.log(shifted).mean(-1)
 power=1-float(alpha);return ((np.power(shifted,power)-1)/power).mean(-1)


def _grid_probability(model,feature,grid_z,beta_z,chunk):
 outputs=[]
 with torch.no_grad():
  for start in range(0,len(feature),chunk):
   x=torch.from_numpy(feature[start:start+chunk]).cuda();by_beta=[]
   for beta in beta_z:by_beta.append(torch.stack([model(x,torch.full((len(x),),float(value),device='cuda'),torch.full((len(x),),float(beta),device='cuda')) for value in grid_z],1))
   outputs.append(torch.stack(by_beta,1).cpu().numpy())
 return np.concatenate(outputs)


def _optimal_budget(model,feature,alphas,betas,beta_z,floors,prices,grid_z,epsilon,penalty,chunk):
 probability=_grid_probability(model,feature,grid_z,beta_z,chunk);cost=(grid_z+1)/2;by_alpha=[]
 for alpha in alphas:
  by_beta=[]
  for ei,_ in enumerate(betas):
   utility=_alpha_utility(probability[:,ei],alpha,epsilon);by_floor=[]
   for floor in floors:
    shortfall=np.maximum(float(floor)-probability[:,ei,:,-1],0);objective=utility[:,:,None]-prices[None,None,:]*cost[None,:,None]-float(penalty)*shortfall[:,:,None];by_floor.append(grid_z[np.argmax(objective,axis=1)])
   by_beta.append(np.stack(by_floor,1))
  by_alpha.append(np.stack(by_beta,1))
 return np.stack(by_alpha,1)


def _evaluate(student,teacher,feature,target,alphas,betas,beta_z,floors,floor_z,prices,price_z,epsilon,penalty):
 x=torch.from_numpy(feature).cuda();candidate=[];cvalue=[];tvalue=[];cshort=[];tshort=[];torch.cuda.synchronize();started=time.monotonic()
 with torch.no_grad():
  for ai,alpha in enumerate(alphas):
   by_beta=[];bcv=[];btv=[];bcs=[];bts=[]
   for ei,_ in enumerate(betas):
    by_floor=[];fcv=[];ftv=[];fcs=[];fts=[];bv=torch.full((len(x),),float(beta_z[ei]),device='cuda');av=torch.full((len(x),),float(alpha),device='cuda')
    for li,floor in enumerate(floors):
     row=[];cr=[];tr=[];csr=[];tsr=[];fv=torch.full((len(x),),float(floor_z[li]),device='cuda')
     for pi,price in enumerate(price_z):
      prediction=student(x,torch.full((len(x),),float(price),device='cuda'),av,bv,fv);truth=torch.from_numpy(target[:,ai,ei,li,pi]).cuda();cp=teacher(x,prediction,bv).cpu().numpy();tp=teacher(x,truth,bv).cpu().numpy();cs=np.maximum(float(floor)-cp[:,-1],0);ts=np.maximum(float(floor)-tp[:,-1],0);pb=prediction.cpu().numpy();row.append(pb);cr.append(_alpha_utility(cp,alpha,epsilon)-prices[pi]*(pb+1)/2-penalty*cs);tr.append(_alpha_utility(tp,alpha,epsilon)-prices[pi]*(target[:,ai,ei,li,pi]+1)/2-penalty*ts);csr.append(cs);tsr.append(ts)
     by_floor.append(np.stack(row,1));fcv.append(np.stack(cr,1));ftv.append(np.stack(tr,1));fcs.append(np.stack(csr,1));fts.append(np.stack(tsr,1))
    by_beta.append(np.stack(by_floor,1));bcv.append(np.stack(fcv,1));btv.append(np.stack(ftv,1));bcs.append(np.stack(fcs,1));bts.append(np.stack(fts,1))
   candidate.append(np.stack(by_beta,1));cvalue.append(np.stack(bcv,1));tvalue.append(np.stack(btv,1));cshort.append(np.stack(bcs,1));tshort.append(np.stack(bts,1))
 torch.cuda.synchronize();forward=time.monotonic()-started;candidate=np.stack(candidate,1);cvalue=np.stack(cvalue,1);tvalue=np.stack(tvalue,1);cshort=np.stack(cshort,1);tshort=np.stack(tshort,1);return {'trajectory_count':int(len(feature)),'normalized_log_budget_MAE':float(np.mean(np.abs(candidate-target))),'mean_frozen_LCB_floor_Lagrangian_utility_regret':float(np.mean(tvalue-cvalue)),'mean_candidate_LCB_final_shortfall':float(np.mean(cshort)),'mean_teacher_LCB_final_shortfall':float(np.mean(tshort)),'price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=4)>1e-7)),'floor_monotonicity_violations':int(np.sum(np.diff(candidate,axis=3)<-1e-7)),'mean_budget_change_low_to_high_beta':float(np.mean(candidate[:,:,-1]-candidate[:,:,0])),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,scenes,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f201,_,_,_,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);fresh,_,_,_,_=_dataset(a.runs_root/c['fresh_rows']['run']/c['fresh_rows']['artifact'],*common,anchors,*tail);artifact=torch.load(a.runs_root/c['frozen_p275']['run']/c['frozen_p275']['artifact'],map_location='cuda');teacher=EpistemicLCBSurface(int(artifact['context_width']),int(artifact['budget_rate_knot_count']),int(artifact['beta_rate_knot_count'])).cuda();teacher.load_state_dict(artifact['model_state_dict']);teacher.eval();domain=np.asarray(c['shadow_price_domain'],np.float32);prices=np.exp(np.linspace(np.log(domain[0]),np.log(domain[1]),int(c['training_price_count']),dtype=np.float32));heldout_prices=np.sqrt(prices[:-1]*prices[1:]);mean=.5*(np.log(domain[0])+np.log(domain[1]));scale=.5*(np.log(domain[1])-np.log(domain[0]));price_z=((np.log(prices)-mean)/scale).astype(np.float32);heldout_price_z=((np.log(heldout_prices)-mean)/scale).astype(np.float32);alphas=np.asarray(c['training_alpha_fairness'],np.float32);ha=np.asarray(c['heldout_alpha_fairness'],np.float32);betas=np.asarray(c['training_epistemic_betas'],np.float32);hb=np.asarray(c['heldout_epistemic_betas'],np.float32);beta_domain=np.asarray(artifact['epistemic_beta_domain'],np.float32);beta_z=(2*(betas-beta_domain[0])/(beta_domain[1]-beta_domain[0])-1).astype(np.float32);heldout_beta_z=(2*(hb-beta_domain[0])/(beta_domain[1]-beta_domain[0])-1).astype(np.float32);floors=np.asarray(c['training_final_reliability_floors'],np.float32);hf=np.asarray(c['heldout_final_reliability_floors'],np.float32);floor_domain=np.asarray(c['final_reliability_floor_domain'],np.float32);floor_z=(2*(floors-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);heldout_floor_z=(2*(hf-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);grid_z=np.linspace(-1,1,int(c['teacher_budget_grid_count']),dtype=np.float32);epsilon=float(c['alpha_utility_epsilon']);penalty=float(c['final_reliability_shortfall_penalty']);chunk=int(c['teacher_chunk_size']);source_target=_optimal_budget(teacher,sf,alphas,betas,beta_z,floors,prices,grid_z,epsilon,penalty,chunk);target201=_optimal_budget(teacher,f201,ha,hb,heldout_beta_z,hf,heldout_prices,grid_z,epsilon,penalty,chunk);fresh_target=_optimal_budget(teacher,fresh,ha,hb,heldout_beta_z,hf,heldout_prices,grid_z,epsilon,penalty,chunk);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout=_optimal_budget(teacher,sf[dev],ha,hb,heldout_beta_z,hf,heldout_prices,grid_z,epsilon,penalty,chunk);x=torch.from_numpy(sf).cuda();target=torch.from_numpy(source_target).cuda();pz=torch.from_numpy(price_z).cuda();at=torch.from_numpy(alphas).cuda();bt=torch.from_numpy(beta_z).cuda();ft=torch.from_numpy(floor_z).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=EpistemicLCBLagrangianPolicy(int(m['width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];ai=torch.randint(len(alphas),(len(idx),),device='cuda');ei=torch.randint(len(betas),(len(idx),),device='cuda');li=torch.randint(len(floors),(len(idx),),device='cuda');pi=torch.randint(len(prices),(len(idx),),device='cuda');prediction=student(x[idx],pz[pi],at[ai],bt[ei],ft[li]);loss=F.l1_loss(prediction,target[idx,ai,ei,li,pi]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P276 epistemic LCB policy step={step+1} budget_mae={last:.7f}',flush=True)
 source=_evaluate(student,teacher,sf[dev],source_heldout,ha,hb,heldout_beta_z,hf,heldout_floor_z,heldout_prices,heldout_price_z,epsilon,penalty);r201=_evaluate(student,teacher,f201,target201,ha,hb,heldout_beta_z,hf,heldout_floor_z,heldout_prices,heldout_price_z,epsilon,penalty);fresh_result=_evaluate(student,teacher,fresh,fresh_target,ha,hb,heldout_beta_z,hf,heldout_floor_z,heldout_prices,heldout_price_z,epsilon,penalty);decision=c['decision'];checks={'P201_budget_fidelity':r201['normalized_log_budget_MAE']<=float(decision['maximum_P201_normalized_log_budget_MAE']),'P201_frozen_LCB_floor_Lagrangian_regret':r201['mean_frozen_LCB_floor_Lagrangian_utility_regret']<=float(decision['maximum_P201_mean_frozen_LCB_floor_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'rate_knot_count':m['rate_knot_count'],'shadow_price_log_mean':mean,'shadow_price_log_scale':scale,'floor_domain':floor_domain,'epistemic_beta_domain':beta_domain,'alpha_utility_epsilon':epsilon,'shortfall_penalty':penalty,'base_model':c['frozen_p275']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_normalized_log_budget_mae':last},'source_development':source,'P201_post_hoc_development':r201,'P243_consumed_descriptive':fresh_result,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
