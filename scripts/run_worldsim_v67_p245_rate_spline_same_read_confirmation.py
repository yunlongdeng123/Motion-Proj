"""Evaluate frozen P244 as a prospective secondary on P243's first fresh read."""

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
from scripts.run_worldsim_v67_p241_integrated_monotone_budget_surface import _evaluate
from scripts.run_worldsim_v67_p244_monotone_rate_spline_surface import MonotoneRateSplineSurface


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();rows=a.runs_root/c['p243_rows']['run']/c['p243_rows']['artifact'];deadline=time.monotonic()+float(c['rows_wait_timeout_seconds'])
 while not rows.is_file():
  if time.monotonic()>=deadline:raise TimeoutError(f'P243 rows not ready: {rows}')
  time.sleep(5)
 torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['anchor_budgets'],np.float32);heldout=np.sqrt(anchors[:-1]*anchors[1:]);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));feature,_,_,scenes,anchor_seconds=_dataset(rows,*common,anchors,*tail);_,teacher,truth,heldout_scenes,teacher_seconds=_dataset(rows,*common,heldout,*tail)
 if not np.array_equal(scenes,heldout_scenes):raise RuntimeError('P245 anchor/heldout rows misaligned')
 artifact=torch.load(a.runs_root/c['frozen_p244']['run']/c['frozen_p244']['artifact'],map_location='cuda');model=MonotoneRateSplineSurface(int(artifact['context_width']),int(artifact['rate_knot_count'])).cuda();model.load_state_dict(artifact['model_state_dict']);model.eval();result=_evaluate(model,feature,teacher,truth,heldout,float(artifact['budget_log_mean']),float(artifact['budget_log_scale']));decision=c['decision'];checks={'fresh_surface_teacher_fidelity':result['surface_teacher_probability_MAE']<=float(decision['maximum_surface_teacher_probability_MAE']),'fresh_final_curve_teacher_fidelity':result['final_curve_teacher_probability_MAE']<=float(decision['maximum_final_curve_teacher_probability_MAE']),'fresh_surface_quality_noninferior':result['relative_surface_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_surface_Brier_degradation']) and result['absolute_surface_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_surface_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'scene_indices':sorted(int(v) for v in np.unique(scenes)),'heldout_budgets':[float(v) for v in heldout],'fresh_secondary':result,'teacher_MC_seconds':anchor_seconds+teacher_seconds,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
