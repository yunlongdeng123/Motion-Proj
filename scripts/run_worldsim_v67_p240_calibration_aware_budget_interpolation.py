"""Interpolate P233 full-horizon knots before the frozen P203 calibration map."""

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
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import MonotonePrefixSurface
from scripts.run_worldsim_v67_p238_continuous_budget_prefix_surface import _paired_dataset


def _inverse_calibration(calibrator, target):
 lo=torch.full_like(target,1e-5);hi=torch.full_like(target,1-1e-5)
 for _ in range(32):
  mid=.5*(lo+hi);below=calibrator(mid)<target;lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
 return .5*(lo+hi)


def _evaluate(model,calibrator,feature,teacher,truth,index=None):
 if index is not None:feature=feature[index];teacher=teacher[index];truth=truth[index]
 torch.cuda.synchronize();before=time.monotonic()
 with torch.no_grad():
  knots=model(torch.from_numpy(feature).cuda());candidate=.5*(knots[:,:,:-1]+knots[:,:,1:]);raw_full=_inverse_calibration(calibrator,knots[:,-1]);candidate[:,-1]=calibrator(.5*(raw_full[:,:-1]+raw_full[:,1:]));candidate=candidate.cpu().numpy()
 torch.cuda.synchronize();forward=time.monotonic()-before;cb=float(np.mean((candidate-truth)**2));tb=float(np.mean((teacher-truth)**2));cc=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));tc=float(np.mean(np.abs(teacher.mean(0)-truth.mean(0))));return {'trajectory_count':int(len(feature)),'heldout_budget_count':6,'surface_teacher_probability_MAE':float(np.mean(np.abs(candidate-teacher))),'final_curve_teacher_probability_MAE':float(np.mean(np.abs(candidate[:,-1]-teacher[:,-1]))),'student_surface_integrated_brier':cb,'teacher_surface_integrated_brier':tb,'relative_surface_Brier_degradation_vs_teacher':(cb-tb)/tb,'student_surface_calibration_error':cc,'teacher_surface_calibration_error':tc,'absolute_surface_calibration_error_increase_vs_teacher':cc-tc,'budget_monotonicity_violations':int(np.sum(np.diff(candidate,axis=2)<-1e-7)),'horizon_monotonicity_violations':int(np.sum(np.diff(candidate,axis=1)>1e-7)),'student_forward_and_interpolation_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);training=np.asarray(c['training_budgets'],np.float32);heldout=np.sqrt(training[:-1]*training[1:]);common=(members,ens,density,fd,p199,copula,calibrator,h,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,sht,shy,scenes,ss=_paired_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],common,training,heldout);f183,_,_,ht183,hy183,_,s183=_paired_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],common,training,heldout);f201,_,_,ht201,hy201,_,s201=_paired_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],common,training,heldout);artifact=torch.load(a.runs_root/c['frozen_p233']['run']/c['frozen_p233']['artifact'],map_location='cuda');model=MonotonePrefixSurface(artifact['hidden_dimensions']).cuda();model.load_state_dict(artifact['model_state_dict']);model.eval();dev=scenes%5==1;source=_evaluate(model,calibrator,sf,sht,shy,dev);r183=_evaluate(model,calibrator,f183,ht183,hy183);r201=_evaluate(model,calibrator,f201,ht201,hy201);decision=c['decision'];checks={'P201_heldout_surface_teacher_fidelity':r201['surface_teacher_probability_MAE']<=float(decision['maximum_P201_heldout_surface_teacher_probability_MAE']),'P201_heldout_final_curve_teacher_fidelity':r201['final_curve_teacher_probability_MAE']<=float(decision['maximum_P201_heldout_final_curve_teacher_probability_MAE']),'P201_heldout_surface_quality_noninferior':r201['relative_surface_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_P201_heldout_surface_Brier_degradation']) and r201['absolute_surface_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_heldout_surface_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'heldout_budgets':[float(v) for v in heldout],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
