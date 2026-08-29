"""Confirm the frozen variable-set tail-CVaR compiler on P243 fresh rows."""

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
from scripts.run_worldsim_v67_p256_group_budget_dual_compiler import _groups
from scripts.run_worldsim_v67_p270_tail_cvar_equivariant_allocator import TailCVaREquivariantAllocator
from scripts.run_worldsim_v67_p272_variable_set_tail_cvar_dual import VariableSetTailCVaRDual,_target_prices,_evaluate_size,_aggregate


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));feature,_,_,scenes,_=_dataset(a.runs_root/c['fresh_rows']['run']/c['fresh_rows']['artifact'],*common,anchors,*tail)
 forward=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(forward['context_width']),int(forward['rate_knot_count'])).cuda();teacher.load_state_dict(forward['model_state_dict']);teacher.eval();primal_artifact=torch.load(a.runs_root/c['frozen_p271']['run']/c['frozen_p271']['artifact'],map_location='cuda');policy=TailCVaREquivariantAllocator(int(primal_artifact['element_width']),int(primal_artifact['context_width']),int(primal_artifact['rate_knot_count'])).cuda();policy.load_state_dict(primal_artifact['model_state_dict']);policy.eval();dual_artifact=torch.load(a.runs_root/c['frozen_p272']['run']/c['frozen_p272']['artifact'],map_location='cuda');student=VariableSetTailCVaRDual(int(dual_artifact['element_width']),int(dual_artifact['context_width']),int(dual_artifact['rate_knot_count'])).cuda();student.load_state_dict(dual_artifact['model_state_dict']);student.eval()
 alphas=np.asarray(c['alpha_fairness'],np.float32);floors=np.asarray(c['final_reliability_floors'],np.float32);domain=np.asarray(primal_artifact['floor_domain'],np.float32);floor_z=(2*(floors-domain[0])/(domain[1]-domain[0])-1).astype(np.float32);tail_masses=np.asarray(c['tail_masses'],np.float32);fractions=np.asarray(c['attainable_budget_fractions'],np.float32);alloc_batch=int(c['allocator_inference_batch_size']);chunk=int(c['teacher_inference_chunk_size']);steps=int(c['teacher_bisection_steps']);predict_batch=int(c['student_inference_batch_size']);rows=[]
 for size in map(int,c['group_sizes']):
  groups,_=_groups(feature,scenes,size);target,low,high=_target_prices(policy,groups,alphas,floor_z,tail_masses,fractions,steps,alloc_batch);rows.append(_evaluate_size(student,policy,teacher,groups,target,low,high,alphas,floors,floor_z,tail_masses,fractions,float(primal_artifact['shadow_price_log_mean']),float(primal_artifact['shadow_price_log_scale']),float(primal_artifact['alpha_utility_epsilon']),float(primal_artifact['tail_CVaR_shortfall_penalty']),alloc_batch,chunk,predict_batch))
 result=_aggregate(rows);decision=c['decision'];checks={'fresh_attained_budget_fraction_fidelity':result['attained_budget_fraction_MAE']<=float(decision['maximum_attained_budget_fraction_MAE']),'fresh_tail_CVaR_Lagrangian_regret':result['mean_frozen_tail_CVaR_Lagrangian_utility_regret']<=float(decision['maximum_mean_frozen_tail_CVaR_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'fresh_scene_count':int(len(np.unique(scenes))),'fresh_trajectory_count':int(len(feature)),'fresh_confirmation':result,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
