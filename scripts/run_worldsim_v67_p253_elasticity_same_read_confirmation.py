"""Evaluate frozen P252 elasticity on P243's first fresh read."""

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
from scripts.run_worldsim_v67_p252_marginal_reliability_elasticity import MarginalReliabilityElasticity,_teacher_elasticity,_evaluate


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();rows=a.runs_root/c['p243_rows']['run']/c['p243_rows']['artifact'];student_artifact=a.runs_root/c['frozen_p252']['run']/c['frozen_p252']['artifact'];deadline=time.monotonic()+float(c['inputs_wait_timeout_seconds'])
 while not rows.is_file() or not student_artifact.is_file():
  if time.monotonic()>=deadline:raise TimeoutError(f'P243 rows or P252 artifact not ready: {rows}, {student_artifact}')
  time.sleep(5)
 torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));feature,_,_,scenes,_=_dataset(rows,*common,anchors,*tail);frozen=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(frozen['context_width']),int(frozen['rate_knot_count'])).cuda();teacher.load_state_dict(frozen['model_state_dict']);teacher.eval();heldout_anchors=np.asarray(c['heldout_anchor_budgets'],np.float32);budgets=np.sqrt(heldout_anchors[:-1]*heldout_anchors[1:]);z=((np.log(budgets)-float(frozen['budget_log_mean']))/float(frozen['budget_log_scale'])).astype(np.float32);target=_teacher_elasticity(teacher,feature,z,int(c['teacher_chunk_size']));artifact=torch.load(student_artifact,map_location='cuda');student=MarginalReliabilityElasticity(int(artifact['width'])).cuda();student.load_state_dict(artifact['model_state_dict']);student.eval();result=_evaluate(student,feature,target,z);decision=c['decision'];checks={'fresh_elasticity_fidelity':result['elasticity_MAE']<=float(decision['maximum_elasticity_MAE']),'fresh_marginal_value_ranking':result['mean_within_query_Spearman']>=float(decision['minimum_mean_within_query_Spearman'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'scene_indices':sorted(int(v) for v in np.unique(scenes)),'heldout_budgets':[float(v) for v in budgets],'fresh_secondary':result,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
