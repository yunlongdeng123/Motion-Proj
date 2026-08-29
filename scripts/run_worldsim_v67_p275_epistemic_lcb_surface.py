"""Distill an epistemic-aversion conditioned lower-confidence reliability surface."""

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
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface,_paired_dataset


class EpistemicLCBSurface(nn.Module):
 def __init__(self,context_width,budget_knot_count,beta_knot_count):
  super().__init__();self.budget_knot_count=int(budget_knot_count);self.beta_knot_count=int(beta_knot_count);self.encoder=nn.Sequential(nn.Linear(36,context_width),nn.SiLU(),nn.Linear(context_width,context_width),nn.SiLU());self.intercept=nn.Linear(context_width,4);self.budget_rates=nn.Linear(context_width,4*self.budget_knot_count);self.beta_rates=nn.Linear(context_width,4*self.beta_knot_count)
 def _integral(self,rates,value,knot_count):
  rates=F.softplus(rates.reshape(-1,4,knot_count));width=2./(knot_count-1);areas=.5*(rates[:,:,:-1]+rates[:,:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:,:1]),torch.cumsum(areas,2)),2);position=((value.reshape(-1,1)+1)/width).clamp(0,knot_count-1);index=torch.floor(position).long().clamp(max=knot_count-2);fraction=position-index;gather=index[:,None,:].expand(-1,4,-1);r0=torch.gather(rates,2,gather).squeeze(2);r1=torch.gather(rates,2,gather+1).squeeze(2);base=torch.gather(cumulative,2,gather).squeeze(2);return base+width*(r0*fraction+.5*(r1-r0)*fraction.square())
 def forward(self,feature,normalized_log_budget,normalized_beta):
  context=self.encoder(feature);units=torch.sigmoid(self.intercept(context)+self._integral(self.budget_rates(context),normalized_log_budget,self.budget_knot_count)-self._integral(self.beta_rates(context),normalized_beta,self.beta_knot_count));return torch.cumprod(units,1)


def _ensemble(models,feature,budgets,mean,scale,chunk):
 outputs=[]
 with torch.no_grad():
  for budget in budgets:
   by_member=[]
   for start in range(0,len(feature),chunk):
    x=torch.from_numpy(feature[start:start+chunk]).cuda();z=torch.full((len(x),),(np.log(float(budget))-mean)/scale,device='cuda');by_member.append(torch.stack([model(x,z) for model in models]).cpu().numpy())
   outputs.append(np.concatenate(by_member,1))
 prediction=np.stack(outputs,3);return prediction.mean(0).astype(np.float32),prediction.std(0).astype(np.float32)


def _evaluate(model,feature,ensemble_mean,ensemble_std,truth,budgets,betas,budget_mean,budget_scale,beta_domain,chunk):
 candidate=[]
 with torch.no_grad():
  for beta in betas:
   by_budget=[]
   for budget in budgets:
    values=[]
    for start in range(0,len(feature),chunk):
     x=torch.from_numpy(feature[start:start+chunk]).cuda();bz=torch.full((len(x),),(np.log(float(budget))-budget_mean)/budget_scale,device='cuda');ez=torch.full((len(x),),2*(float(beta)-beta_domain[0])/(beta_domain[1]-beta_domain[0])-1,device='cuda');values.append(model(x,bz,ez).cpu().numpy())
    by_budget.append(np.concatenate(values))
   candidate.append(np.stack(by_budget,2))
 candidate=np.stack(candidate,1);target=np.stack([np.clip(ensemble_mean-float(beta)*ensemble_std,0,1) for beta in betas],1);return {'trajectory_count':int(len(feature)),'beta_count':int(len(betas)),'budget_count':int(len(budgets)),'surface_LCB_teacher_probability_MAE':float(np.mean(np.abs(candidate-target))),'final_curve_LCB_teacher_probability_MAE':float(np.mean(np.abs(candidate[:,:,-1]-target[:,:,-1]))),'mean_probability_drop_beta_min_to_max':float(np.mean(candidate[:,0]-candidate[:,-1])),'mean_LCB_overprediction_vs_realized_frequency':float(np.mean(candidate-truth[:,None])),'budget_monotonicity_violations':int(np.sum(np.diff(candidate,axis=3)<-1e-7)),'horizon_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)>1e-7)),'beta_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7))}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();seed=int(c['seed']);torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);feature_anchors=np.asarray(c['feature_anchor_budgets'],np.float32);heldout_anchors=np.asarray(c['heldout_anchor_budgets'],np.float32);heldout_budgets=np.sqrt(heldout_anchors[:-1]*heldout_anchors[1:]);domain=c['training_budget_domain'];log_min=float(np.log(domain[0]));log_max=float(np.log(domain[1]));count=int(c['training_budget_count']);fractions=(np.arange(count,dtype=np.float32)+float(c['training_budget_offset']))/count;training_budgets=np.exp(log_min+fractions*(log_max-log_min)).astype(np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),seed,float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,_,_,scenes,_=_paired_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],common,feature_anchors,training_budgets,heldout_budgets);f201,_,_,_,hy201,_,_=_paired_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],common,feature_anchors,training_budgets,heldout_budgets);fresh,_,_,_,fresh_truth,_,_=_paired_dataset(a.runs_root/c['fresh_rows']['run']/c['fresh_rows']['artifact'],common,feature_anchors,training_budgets,heldout_budgets)
 artifact=torch.load(a.runs_root/c['frozen_p274']['run']/c['frozen_p274']['artifact'],map_location='cuda');models=[]
 for state in artifact['member_state_dicts']:
  member=MonotoneRateSplineSurface(int(artifact['context_width']),int(artifact['rate_knot_count'])).cuda();member.load_state_dict(state);models.append(member.eval())
 budget_mean=float(artifact['budget_log_mean']);budget_scale=float(artifact['budget_log_scale']);chunk=int(c['inference_chunk_size']);source_mean,source_std=_ensemble(models,sf,training_budgets,budget_mean,budget_scale,chunk);p201_mean,p201_std=_ensemble(models,f201,heldout_budgets,budget_mean,budget_scale,chunk);fresh_mean,fresh_std=_ensemble(models,fresh,heldout_budgets,budget_mean,budget_scale,chunk);betas=np.asarray(c['training_epistemic_betas'],np.float32);heldout_betas=np.asarray(c['heldout_epistemic_betas'],np.float32);beta_domain=np.asarray(c['epistemic_beta_domain'],np.float32);target=np.stack([np.clip(source_mean-float(beta)*source_std,0,1) for beta in betas],1).astype(np.float32);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();y=torch.from_numpy(target).cuda();budget_z=torch.from_numpy(((np.log(training_budgets)-budget_mean)/budget_scale).astype(np.float32)).cuda();beta_z=torch.from_numpy((2*(betas-beta_domain[0])/(beta_domain[1]-beta_domain[0])-1).astype(np.float32)).cuda();m=c['student'];model=EpistemicLCBSurface(int(m['context_width']),int(m['budget_rate_knot_count']),int(m['beta_rate_knot_count'])).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];bi=torch.randint(len(training_budgets),(len(idx),),device='cuda');ei=torch.randint(len(betas),(len(idx),),device='cuda');loss=F.l1_loss(model(x[idx],budget_z[bi],beta_z[ei]),y[idx,ei,:,bi]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();last=float(loss.detach())
  if step%500==0:print(f'P275 epistemic LCB surface step={step+1} mae={last:.7f}',flush=True)
 p201=_evaluate(model,f201,p201_mean,p201_std,hy201,heldout_budgets,heldout_betas,budget_mean,budget_scale,beta_domain,chunk);fresh_result=_evaluate(model,fresh,fresh_mean,fresh_std,fresh_truth,heldout_budgets,heldout_betas,budget_mean,budget_scale,beta_domain,chunk);decision=c['decision'];checks={'P201_LCB_surface_fidelity':p201['surface_LCB_teacher_probability_MAE']<=float(decision['maximum_P201_surface_LCB_teacher_probability_MAE']),'P201_LCB_final_curve_fidelity':p201['final_curve_LCB_teacher_probability_MAE']<=float(decision['maximum_P201_final_curve_LCB_teacher_probability_MAE'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'context_width':m['context_width'],'budget_rate_knot_count':m['budget_rate_knot_count'],'beta_rate_knot_count':m['beta_rate_knot_count'],'budget_log_mean':budget_mean,'budget_log_scale':budget_scale,'epistemic_beta_domain':beta_domain,'base_model':c['frozen_p274']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_LCB_teacher_MAE':last},'P201_post_hoc_development':p201,'P243_consumed_descriptive':fresh_result,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
