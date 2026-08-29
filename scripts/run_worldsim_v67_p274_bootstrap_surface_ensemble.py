"""Train a scene-bootstrap ensemble of monotone reliability surfaces."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface,_paired_dataset


def _evaluate(models,feature,teacher,truth,budgets,mean,scale):
 x=torch.from_numpy(feature).cuda();means=[];stds=[];torch.cuda.synchronize();started=time.monotonic()
 with torch.no_grad():
  for budget in budgets:
   z=torch.full((len(x),),(np.log(float(budget))-mean)/scale,device='cuda');prediction=torch.stack([model(x,z) for model in models]);means.append(prediction.mean(0).cpu().numpy());stds.append(prediction.std(0,unbiased=False).cpu().numpy())
 torch.cuda.synchronize();forward=time.monotonic()-started;candidate=np.stack(means,2);epistemic=np.stack(stds,2);row_uncertainty=epistemic.mean((1,2));row_error=np.abs(candidate-teacher).mean((1,2));rho=float(spearmanr(row_uncertainty,row_error).statistic);cut=np.quantile(row_uncertainty,.75);lift=float(row_error[row_uncertainty>=cut].mean()/max(row_error.mean(),1e-9));cb=float(np.mean((candidate-truth)**2));tb=float(np.mean((teacher-truth)**2));return {'trajectory_count':int(len(feature)),'surface_teacher_probability_MAE':float(np.mean(np.abs(candidate-teacher))),'final_curve_teacher_probability_MAE':float(np.mean(np.abs(candidate[:,-1]-teacher[:,-1]))),'ensemble_mean_surface_integrated_brier':cb,'teacher_surface_integrated_brier':tb,'relative_surface_Brier_degradation_vs_teacher':(cb-tb)/tb,'mean_epistemic_probability_std':float(epistemic.mean()),'final_epistemic_probability_std':float(epistemic[:,-1].mean()),'epistemic_teacher_error_spearman':rho,'top_uncertainty_quartile_teacher_error_lift':lift,'budget_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'horizon_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7)),'ensemble_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();seed=int(c['seed']);torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);feature_anchors=np.asarray(c['feature_anchor_budgets'],np.float32);heldout_anchors=np.asarray(c['heldout_anchor_budgets'],np.float32);heldout_budgets=np.sqrt(heldout_anchors[:-1]*heldout_anchors[1:]);domain=c['training_budget_domain'];log_min=float(np.log(domain[0]));log_max=float(np.log(domain[1]));count=int(c['training_budget_count']);fractions=(np.arange(count,dtype=np.float32)+float(c['training_budget_offset']))/count;training_budgets=np.exp(log_min+fractions*(log_max-log_min)).astype(np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),seed,float(c['teacher']['ignored_future_marginal_probability']));sf,st,_,_,_,scenes,_=_paired_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],common,feature_anchors,training_budgets,heldout_budgets);f201,_,_,ht201,hy201,_,_=_paired_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],common,feature_anchors,training_budgets,heldout_budgets);fresh,_,_,fresh_teacher,fresh_truth,_,_=_paired_dataset(a.runs_root/c['fresh_rows']['run']/c['fresh_rows']['artifact'],common,feature_anchors,training_budgets,heldout_budgets)
 dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_scenes=np.unique(scenes[~dev]);rng=np.random.default_rng(seed);m=c['ensemble'];models=[];pools=[]
 for member_index in range(int(m['member_count'])):
  torch.manual_seed(seed+1009*(member_index+1));models.append(MonotoneRateSplineSurface(int(m['context_width']),int(m['rate_knot_count'])).cuda());sampled=rng.choice(train_scenes,size=len(train_scenes),replace=True);pools.append(np.concatenate([np.flatnonzero(scenes==scene) for scene in sampled]))
 optimizer=torch.optim.AdamW([parameter for model in models for parameter in model.parameters()],lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));x=torch.from_numpy(sf).cuda();target=torch.from_numpy(st).cuda();budget_z=torch.from_numpy(((np.log(training_budgets)-.5*(log_min+log_max))/(.5*(log_max-log_min))).astype(np.float32)).cuda();last=[]
 for step in range(int(m['steps'])):
  optimizer.zero_grad(set_to_none=True);losses=[]
  for model,pool in zip(models,pools):
   idx=torch.from_numpy(rng.choice(pool,size=int(m['batch_size']),replace=True)).cuda();bi=torch.randint(len(training_budgets),(len(idx),),device='cuda');loss=F.l1_loss(model(x[idx],budget_z[bi]),target[idx,:,bi]);(loss/len(models)).backward();losses.append(float(loss.detach()))
  optimizer.step();last=losses
  if step%500==0:print(f'P274 bootstrap surface ensemble step={step+1} mean_mae={np.mean(last):.7f} member_std={np.std(last):.7f}',flush=True)
 budget_mean=.5*(log_min+log_max);budget_scale=.5*(log_max-log_min);p201=_evaluate(models,f201,ht201,hy201,heldout_budgets,budget_mean,budget_scale);fresh_result=_evaluate(models,fresh,fresh_teacher,fresh_truth,heldout_budgets,budget_mean,budget_scale);decision=c['decision'];checks={'P201_ensemble_mean_final_curve_fidelity':p201['final_curve_teacher_probability_MAE']<=float(decision['maximum_P201_final_curve_teacher_probability_MAE']),'P201_epistemic_disagreement_tracks_teacher_error':p201['epistemic_teacher_error_spearman']>=float(decision['minimum_P201_epistemic_teacher_error_spearman'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'member_state_dicts':[model.state_dict() for model in models],'member_count':len(models),'context_width':m['context_width'],'rate_knot_count':m['rate_knot_count'],'budget_log_mean':budget_mean,'budget_log_scale':budget_scale,'training_budgets':training_budgets},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'member_count':len(models),'scene_bootstrap_unique_counts':[int(len(np.unique(scenes[pool]))) for pool in pools],'final_member_teacher_MAEs':last},'P201_post_hoc_development':p201,'P243_consumed_fresh_descriptive':fresh_result,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
