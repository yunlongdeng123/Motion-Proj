"""Evaluate frozen P196 density pool on already-consumed P183 compact rows."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p196_monotone_horizon_density_router import MonotoneHorizonRouter, _load_density


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/"worldsim_v67"/c["task_id"]/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/"resolved.yaml").write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with np.load(a.runs_root/c["frozen_rows"]["run"]/c["frozen_rows"]["artifact"],allow_pickle=False) as z: arrays={n:z[n] for n in z.files}
    ens=torch.load(a.runs_root/c["frozen_p126"]["run"]/c["frozen_p126"]["artifact"],map_location="cuda");members=[]
    for state in ens["member_state_dicts"]:
        m=DirectionalActorGaussian(20,ens["hidden_dimensions"]).cuda();m.load_state_dict(state);members.append(m.eval())
    p182,f182=_load_density(a.runs_root/c["frozen_p182"]["run"]/c["frozen_p182"]["artifact"]);p192,f192=_load_density(a.runs_root/c["frozen_p192"]["run"]/c["frozen_p192"]["artifact"])
    frozen_router=torch.load(a.runs_root/c["frozen_p196"]["run"]/c["frozen_p196"]["artifact"],map_location="cuda");router=MonotoneHorizonRouter().cuda();router.load_state_dict(frozen_router["model_state_dict"]);router.eval()
    score,scenes=_ensemble_trajectory_score(arrays,members,np.asarray(ens["feature_mean"],dtype=np.float32),np.asarray(ens["feature_scale"],dtype=np.float32),np.asarray(ens["target_mean"],dtype=np.float32),np.asarray(ens["target_scale"],dtype=np.float32));cost,cost_scenes=_continuous_cost(arrays,float(c["boundary_state_cost"]["clearance_floor_m"]));assert np.array_equal(scenes,cost_scenes)
    horizon=_trajectory_horizon(arrays);clearance=_trajectory_clearance(arrays);budgets=np.asarray(c["reliability_budgets"],dtype=np.float32);target=cost[:,None]<=budgets[None]
    base=_predict_cdf(p182,score,horizon,clearance,budgets,tuple(f182["norms"]));other=_predict_cdf(p192,score,horizon,clearance,budgets,tuple(f192["norms"]));lo,hi=frozen_router["minimum_horizon_seconds"],frozen_router["maximum_horizon_seconds"]
    with torch.no_grad(): weight=router(torch.from_numpy(np.clip((horizon-lo)/(hi-lo),0,1).astype(np.float32)).cuda()).cpu().numpy()
    candidate=(1-weight[:,None])*base+weight[:,None]*other;results={}
    for value in c["horizons_seconds"]:
        mask=np.isclose(horizon,float(value));bb=float(np.mean((base[mask]-target[mask])**2));cb=float(np.mean((candidate[mask]-target[mask])**2));be=float(np.mean(np.abs(base[mask].mean(0)-target[mask].mean(0))));ce=float(np.mean(np.abs(candidate[mask].mean(0)-target[mask].mean(0))))
        results[str(value)]={"trajectory_count":int(mask.sum()),"p182_integrated_brier":bb,"routed_integrated_brier":cb,"brier_reduction_vs_p182":(bb-cb)/bb,"p182_mean_absolute_reliability_error":be,"routed_mean_absolute_reliability_error":ce,"calibration_error_reduction_vs_p182":(be-ce)/max(be,1e-12),"mean_p192_weight":float(weight[mask].mean())}
    br=[r["brier_reduction_vs_p182"] for r in results.values()];cr=[r["calibration_error_reduction_vs_p182"] for r in results.values()];macro={"mean_brier_reduction_vs_p182":float(np.mean(br)),"mean_calibration_error_reduction_vs_p182":float(np.mean(cr))};checks={"every_horizon_brier_noninferior_vs_p182":bool(min(br)>=0),"minimum_mean_calibration_error_reduction_vs_p182":macro["mean_calibration_error_reduction_vs_p182"]>=float(c["decision"]["minimum_mean_calibration_error_reduction_vs_p182"])};verdict=c["verdict_on_pass"] if all(checks.values()) else c["verdict_on_failure"]
    summary={"schema_version":c["output_schema_version"],"task_id":c["task_id"],"hypothesis_id":c["hypothesis_id"],"status":"done","verdict":verdict,"role":c["role"],"per_horizon":results,"macro_metrics":macro,"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":c["claim_boundary"]};(d/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(d/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n");print(json.dumps({"run_dir":str(d),**summary},indent=2))


if __name__=="__main__":main()
