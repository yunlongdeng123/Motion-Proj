"""Confirm frozen P227 monotone curve distillation on unused validation scenes."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p227_monotone_reliability_curve_distillation import MonotoneCurveStudent


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.cuda.reset_peak_memory_stats();data=c['evaluation_data'];metadata=Path(data['metadata_root'])/'v1.0-trainval';scenes_meta=json.loads((metadata/'scene.json').read_text());index={x['name']:i for i,x in enumerate(scenes_meta)};pending={n:Path(data['processed_root'])/f'{index[n]:03d}' for n in data['scene_names']};deadline=time.monotonic()+float(data['readiness_timeout_seconds'])
 while pending:
  ready=[n for n,s in pending.items() if (s/'instances'/'instances_info.json').is_file() and (s/'lidar_pose').is_dir()]
  for n in ready:pending.pop(n)
  if pending:
   if time.monotonic()>=deadline:raise TimeoutError(f'P228 scenes not ready: {sorted(pending)}')
   time.sleep(5)
 parts=[]
 for name in data['scene_names']:
  path=Path(data['processed_root'])/f'{index[name]:03d}';part=materialize_actor_query_rows([path],data['horizons_seconds'],data);parts.append(part);print(json.dumps({'materialized':name,'row_count':int(len(part['features']))}),flush=True)
 arrays={name:np.concatenate([part[name] for part in parts]) for name in parts[0]};partial=d/'P228_MONOTONE_DISTILLATION_ROWS.partial.npz';np.savez_compressed(partial,**arrays);partial.replace(d/c['model_artifact']);ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();p227=torch.load(a.runs_root/c['frozen_p227']['run']/c['frozen_p227']['artifact'],map_location='cuda');student=MonotoneCurveStudent(p227['hidden_dimensions']).cuda();student.load_state_dict(p227['model_state_dict']);student.eval();h=np.asarray(data['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);scores,clearances,costs,_=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,tuple(fd['norms'])) for i in range(4)],1);raw=np.concatenate((scores,clearances),1);base=((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32);feature=np.concatenate((base,marginal.reshape(len(base),-1)),1).astype(np.float32);truth=np.all(costs[:,:,None]<=budgets[None,None,:],1).astype(np.float32)
 torch.cuda.synchronize();before=time.monotonic();joint=_joint_probabilities(copula,torch.from_numpy(base).cuda(),torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c['teacher']['monte_carlo_samples']),int(c['seed']));teacher=calibrator(torch.from_numpy(joint).cuda()).detach();torch.cuda.synchronize();teacher_seconds=time.monotonic()-before;torch.cuda.synchronize();before=time.monotonic();candidate=student(torch.from_numpy(feature).cuda()).detach();torch.cuda.synchronize();student_seconds=time.monotonic()-before;teacher=teacher.cpu().numpy();candidate=candidate.cpu().numpy();sb=float(np.mean((candidate-truth)**2));tb=float(np.mean((teacher-truth)**2));sc=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));tc=float(np.mean(np.abs(teacher.mean(0)-truth.mean(0))));mae=float(np.mean(np.abs(candidate-teacher)));bd=(sb-tb)/tb;ci=sc-tc;decision=c['decision'];checks={'teacher_probability_fidelity':mae<=float(decision['maximum_teacher_probability_MAE']),'Brier_noninferior':bd<=float(decision['maximum_relative_brier_degradation']),'calibration_noninferior':ci<=float(decision['maximum_absolute_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'scenes':data['scene_names'],'trajectory_count':int(len(feature)),'student_teacher_probability_MAE':mae,'student_integrated_brier':sb,'teacher_integrated_brier':tb,'relative_Brier_degradation_vs_teacher':bd,'student_calibration_error':sc,'teacher_calibration_error':tc,'absolute_calibration_error_increase_vs_teacher':ci,'student_forward_seconds':student_seconds,'teacher_MC_forward_seconds':teacher_seconds,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
