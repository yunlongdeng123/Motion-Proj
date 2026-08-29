"""Train a proportional-fair shadow-price policy over frozen P246 reliability."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface
from scripts.run_worldsim_v67_p254_shadow_price_budget_policy import ShadowPriceBudgetPolicy


def _grid_reward(model,feature,grid_z,epsilon,chunk_size):
 outputs=[]
 with torch.no_grad():
  for start in range(0,len(feature),chunk_size):
   x=torch.from_numpy(feature[start:start+chunk_size]).cuda();columns=[torch.log(model(x,torch.full((len(x),),float(value),device='cuda'))+epsilon).mean(1) for value in grid_z];outputs.append(torch.stack(columns,1).cpu().numpy())
 return np.concatenate(outputs)


def _optimal_budget(model,feature,prices,grid_z,epsilon,chunk_size):
 reward=_grid_reward(model,feature,grid_z,epsilon,chunk_size);utility=reward[:,:,None]-prices[None,None,:]*(grid_z[None,:,None]+1)/2;return grid_z[np.argmax(utility,axis=1)]


def _evaluate(student,teacher,feature,target,prices,price_z,epsilon):
 x=torch.from_numpy(feature).cuda();candidate=[];candidate_reward=[];target_reward=[];torch.cuda.synchronize();before=time.monotonic()
 with torch.no_grad():
  for column,value in enumerate(price_z):
   prediction=student(x,torch.full((len(x),),float(value),device='cuda'));candidate.append(prediction.cpu().numpy());candidate_reward.append(torch.log(teacher(x,prediction)+epsilon).mean(1).cpu().numpy());truth=torch.from_numpy(target[:,column]).cuda();target_reward.append(torch.log(teacher(x,truth)+epsilon).mean(1).cpu().numpy())
 torch.cuda.synchronize();forward=time.monotonic()-before;candidate=np.stack(candidate,1);candidate_reward=np.stack(candidate_reward,1);target_reward=np.stack(target_reward,1);candidate_utility=candidate_reward-prices[None,:]*(candidate+1)/2;target_utility=target_reward-prices[None,:]*(target+1)/2;return {'trajectory_count':int(len(feature)),'price_count':int(len(prices)),'normalized_log_budget_MAE':float(np.mean(np.abs(candidate-target))),'mean_frozen_log_utility_regret':float(np.mean(target_utility-candidate_utility)),'price_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,scenes,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,_,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,_,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);frozen=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(frozen['context_width']),int(frozen['rate_knot_count'])).cuda();teacher.load_state_dict(frozen['model_state_dict']);teacher.eval();price_domain=np.asarray(c['shadow_price_domain'],np.float32);training_prices=np.exp(np.linspace(np.log(price_domain[0]),np.log(price_domain[1]),int(c['training_price_count']),dtype=np.float32));heldout_prices=np.sqrt(training_prices[:-1]*training_prices[1:]);price_mean=.5*(np.log(price_domain[0])+np.log(price_domain[1]));price_scale=.5*(np.log(price_domain[1])-np.log(price_domain[0]));training_price_z=((np.log(training_prices)-price_mean)/price_scale).astype(np.float32);heldout_price_z=((np.log(heldout_prices)-price_mean)/price_scale).astype(np.float32);grid_z=np.linspace(-1,1,int(c['teacher_budget_grid_count']),dtype=np.float32);chunk=int(c['teacher_chunk_size']);epsilon=float(c['log_utility_epsilon']);source_target=_optimal_budget(teacher,sf,training_prices,grid_z,epsilon,chunk);target183=_optimal_budget(teacher,f183,heldout_prices,grid_z,epsilon,chunk);target201=_optimal_budget(teacher,f201,heldout_prices,grid_z,epsilon,chunk);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout_target=_optimal_budget(teacher,sf[dev],heldout_prices,grid_z,epsilon,chunk);x=torch.from_numpy(sf).cuda();target=torch.from_numpy(source_target).cuda();z=torch.from_numpy(training_price_z).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=ShadowPriceBudgetPolicy(int(m['width']),int(m['rate_knot_count'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];price_idx=torch.randint(len(training_prices),(len(idx),),device='cuda');prediction=student(x[idx],z[price_idx]);truth=target[idx,price_idx];loss=torch.nn.functional.l1_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P257 log-utility price policy step={step+1} budget_mae={last:.7f}',flush=True)
 source=_evaluate(student,teacher,sf[dev],source_heldout_target,heldout_prices,heldout_price_z,epsilon);r183=_evaluate(student,teacher,f183,target183,heldout_prices,heldout_price_z,epsilon);r201=_evaluate(student,teacher,f201,target201,heldout_prices,heldout_price_z,epsilon);decision=c['decision'];checks={'P201_budget_fidelity':r201['normalized_log_budget_MAE']<=float(decision['maximum_P201_normalized_log_budget_MAE']),'P201_frozen_log_utility_regret':r201['mean_frozen_log_utility_regret']<=float(decision['maximum_P201_mean_frozen_log_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'rate_knot_count':m['rate_knot_count'],'input_dimension':36,'shadow_price_log_mean':price_mean,'shadow_price_log_scale':price_scale,'log_utility_epsilon':epsilon,'base_model':c['frozen_p246']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'price_count':int(len(training_prices)),'final_normalized_log_budget_mae':last},'heldout_shadow_prices':[float(v) for v in heldout_prices],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
