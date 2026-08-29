"""Confirm the frozen epistemic-LCB allocator on six newly processed scenes."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import yaml
from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p275_epistemic_lcb_surface import EpistemicLCBSurface
from scripts.run_worldsim_v67_p276_epistemic_lcb_lagrangian_policy import EpistemicLCBLagrangianPolicy,_optimal_budget,_evaluate


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();data=c['evaluation_data'];metadata=Path(data['metadata_root'])/'v1.0-trainval';scene_meta=json.loads((metadata/'scene.json').read_text());index={row['name']:i for i,row in enumerate(scene_meta)};pending={name:Path(data['processed_root'])/f'{index[name]:03d}' for name in data['scene_names']};deadline=time.monotonic()+float(data['readiness_timeout_seconds'])
 while pending:
  for name in [name for name,path in pending.items() if (path/'instances'/'instances_info.json').is_file() and (path/'lidar_pose').is_dir()]:pending.pop(name)
  if pending:
   if time.monotonic()>=deadline:raise TimeoutError(f'P277 scenes not ready: {sorted(pending)}')
   time.sleep(5)
 parts=[]
 for name in data['scene_names']:
  part=materialize_actor_query_rows([Path(data['processed_root'])/f'{index[name]:03d}'],data['horizons_seconds'],data);parts.append(part);print(json.dumps({'materialized':name,'row_count':int(len(part['features']))}),flush=True)
 arrays={name:np.concatenate([part[name] for part in parts]) for name in parts[0]};partial=d/'P277_FRESH_EPISTEMIC_ROWS.partial.npz';np.savez_compressed(partial,**arrays);rows=d/c['model_artifact'];partial.replace(rows);torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));feature,_,_,scenes,_=_dataset(rows,*common,anchors,*tail);surface_artifact=torch.load(a.runs_root/c['frozen_p275']['run']/c['frozen_p275']['artifact'],map_location='cuda');teacher=EpistemicLCBSurface(int(surface_artifact['context_width']),int(surface_artifact['budget_rate_knot_count']),int(surface_artifact['beta_rate_knot_count'])).cuda();teacher.load_state_dict(surface_artifact['model_state_dict']);teacher.eval();policy_artifact=torch.load(a.runs_root/c['frozen_p276']['run']/c['frozen_p276']['artifact'],map_location='cuda');student=EpistemicLCBLagrangianPolicy(int(policy_artifact['width']),int(policy_artifact['rate_knot_count'])).cuda();student.load_state_dict(policy_artifact['model_state_dict']);student.eval();alphas=np.asarray(c['alpha_fairness'],np.float32);betas=np.asarray(c['epistemic_betas'],np.float32);beta_domain=np.asarray(policy_artifact['epistemic_beta_domain'],np.float32);beta_z=(2*(betas-beta_domain[0])/(beta_domain[1]-beta_domain[0])-1).astype(np.float32);floors=np.asarray(c['final_reliability_floors'],np.float32);floor_domain=np.asarray(policy_artifact['floor_domain'],np.float32);floor_z=(2*(floors-floor_domain[0])/(floor_domain[1]-floor_domain[0])-1).astype(np.float32);prices=np.asarray(c['shadow_prices'],np.float32);mean=float(policy_artifact['shadow_price_log_mean']);scale=float(policy_artifact['shadow_price_log_scale']);price_z=((np.log(prices)-mean)/scale).astype(np.float32);grid_z=np.linspace(-1,1,int(c['teacher_budget_grid_count']),dtype=np.float32);epsilon=float(policy_artifact['alpha_utility_epsilon']);penalty=float(policy_artifact['shortfall_penalty']);target=_optimal_budget(teacher,feature,alphas,betas,beta_z,floors,prices,grid_z,epsilon,penalty,int(c['teacher_chunk_size']));result=_evaluate(student,teacher,feature,target,alphas,betas,beta_z,floors,floor_z,prices,price_z,epsilon,penalty);decision=c['decision'];checks={'fresh_budget_fidelity':result['normalized_log_budget_MAE']<=float(decision['maximum_normalized_log_budget_MAE']),'fresh_frozen_LCB_floor_Lagrangian_regret':result['mean_frozen_LCB_floor_Lagrangian_utility_regret']<=float(decision['maximum_mean_frozen_LCB_floor_Lagrangian_utility_regret'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'scenes':data['scene_names'],'scene_indices':sorted(int(v) for v in np.unique(scenes)),'fresh_confirmation':result,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
