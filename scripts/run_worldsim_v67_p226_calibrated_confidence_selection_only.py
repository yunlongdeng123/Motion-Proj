"""Isolate P203's effect on trajectory selection while scoring raw P199 probabilities."""

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
from scripts.run_worldsim_v67_p223_pairwise_selective_authority_recovery import _dataset


def _evaluate(calibrator,prob,truth,coverage,index=None):
 if index is not None:prob=prob[index];truth=truth[index]
 q=calibrator(torch.from_numpy(prob).cuda()).detach().cpu().numpy();k=int(round(len(prob)*coverage));candidate=np.argsort(np.mean(q*(1-q),1),kind='stable')[:k];control=np.argsort(np.mean(prob*(1-prob),1),kind='stable')[:k];cb=float(np.mean((prob[candidate]-truth[candidate])**2));bb=float(np.mean((prob[control]-truth[control])**2));cc=float(np.mean(np.abs(prob[candidate].mean(0)-truth[candidate].mean(0))));bc=float(np.mean(np.abs(prob[control].mean(0)-truth[control].mean(0))));return {'trajectory_count':int(len(prob)),'calibrated_confidence_selection_raw_P199_brier':cb,'raw_confidence_selection_raw_P199_brier':bb,'Brier_reduction_vs_raw_selection':(bb-cb)/bb,'calibrated_confidence_selection_raw_P199_calibration_error':cc,'raw_confidence_selection_raw_P199_calibration_error':bc,'calibration_error_reduction_vs_raw_selection':(bc-cc)/max(bc,1e-12),'selection_overlap_fraction':float(len(np.intersect1d(candidate,control))/k)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);floor=float(c['boundary_state_cost']['clearance_floor_m']);samples=int(c['evaluation']['monte_carlo_samples']);seed=int(c['seed']);common=(members,ens,density,fd,p199,copula,h,budgets,floor,samples,seed);_,sp,st,scenes=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);_,p183,t183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);_,p201,t201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);coverage=float(c['evaluation']['authority_coverage']);source=_evaluate(calibrator,sp,st,coverage,dev);r183=_evaluate(calibrator,p183,t183,coverage);r201=_evaluate(calibrator,p201,t201,coverage);checks={'Brier_improves_on_both_P183_and_P201':r183['Brier_reduction_vs_raw_selection']>0 and r201['Brier_reduction_vs_raw_selection']>0,'calibration_noninferior_on_both_P183_and_P201':r183['calibration_error_reduction_vs_raw_selection']>=0 and r201['calibration_error_reduction_vs_raw_selection']>=0};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
