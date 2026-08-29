"""Materialize a fresh cohort and confirm frozen P199 joint-horizon reliability."""

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


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/"worldsim_v67"/c["task_id"]/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/"resolved.yaml").write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.cuda.reset_peak_memory_stats();data=c["evaluation_data"]
    metadata=Path(data["metadata_root"])/"v1.0-trainval";scenes_meta=json.loads((metadata/"scene.json").read_text());index={x["name"]:i for i,x in enumerate(scenes_meta)};pending={n:Path(data["processed_root"])/f'{index[n]:03d}' for n in data["scene_names"]};deadline=time.monotonic()+float(data["readiness_timeout_seconds"])
    while pending:
        ready=[n for n,s in pending.items() if (s/"instances"/"instances_info.json").is_file() and (s/"lidar_pose").is_dir()]
        for n in ready:pending.pop(n)
        if pending:
            if time.monotonic()>=deadline:raise TimeoutError(f'P201 scenes not ready: {sorted(pending)}')
            time.sleep(5)
    scene_paths=[Path(data["processed_root"])/f'{index[n]:03d}' for n in data["scene_names"]];parts=[]
    for name,path in zip(data["scene_names"],scene_paths):
        part=materialize_actor_query_rows([path],data["horizons_seconds"],data);parts.append(part);print(json.dumps({"materialized":name,"row_count":int(len(part["features"]))}),flush=True)
    arrays={name:np.concatenate([part[name] for part in parts]) for name in parts[0]};partial=d/"P201_JOINT_HORIZON_ROWS.partial.npz";np.savez_compressed(partial,**arrays);partial.replace(d/"P201_JOINT_HORIZON_ROWS.npz")
    ens=torch.load(a.runs_root/c["frozen_p126"]["run"]/c["frozen_p126"]["artifact"],map_location="cuda");members=[]
    for state in ens["member_state_dicts"]:
        member=DirectionalActorGaussian(20,ens["hidden_dimensions"]).cuda();member.load_state_dict(state);members.append(member.eval())
    density,frozen_density=_load_density(a.runs_root/c["frozen_p182"]["run"]/c["frozen_p182"]["artifact"]);artifact=torch.load(a.runs_root/c["frozen_p199"]["run"]/c["frozen_p199"]["artifact"],map_location="cuda");model=JointHorizonCopula(8,artifact["hidden_dimensions"],4).cuda();model.load_state_dict(artifact["model_state_dict"]);model.eval();horizons=np.asarray(c["evaluation_data"]["horizons_seconds"],dtype=np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c["boundary_state_cost"]["clearance_floor_m"])),horizons)
    raw=np.concatenate((scores,clearances),axis=1);features=torch.from_numpy(((raw-np.asarray(artifact["feature_mean"]))/np.asarray(artifact["feature_scale"])).astype(np.float32)).cuda();budgets=np.asarray(c["reliability_budgets"],dtype=np.float32);norms=tuple(frozen_density["norms"]);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),horizons[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],axis=1);candidate=_joint_probabilities(model,features,torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c["evaluation"]["monte_carlo_samples"]),int(c["evaluation"]["seed"]));independent=np.prod(marginal,axis=1);target=np.all(costs[:,:,None]<=budgets[None,None,:],axis=1)
    cb=float(np.mean((candidate-target)**2));bb=float(np.mean((independent-target)**2));ce=float(np.mean(np.abs(candidate.mean(0)-target.mean(0))));be=float(np.mean(np.abs(independent.mean(0)-target.mean(0))));br=(bb-cb)/bb;cr=(be-ce)/max(be,1e-12);checks={"joint_integrated_brier_strictly_better_than_independent_marginals":cb<bb,"minimum_joint_calibration_error_reduction":cr>=float(c["decision"]["minimum_joint_calibration_error_reduction"])};verdict=c["verdict_on_pass"] if all(checks.values()) else c["verdict_on_failure"]
    summary={"schema_version":c["output_schema_version"],"task_id":c["task_id"],"hypothesis_id":c["hypothesis_id"],"status":"done","verdict":verdict,"role":c["role"],"scenes":c["evaluation_data"]["scene_names"],"joint_trajectory_count":int(len(scores)),"joint_integrated_brier":cb,"independent_marginal_product_integrated_brier":bb,"joint_brier_reduction":br,"joint_mean_absolute_reliability_error":ce,"independent_mean_absolute_reliability_error":be,"joint_calibration_error_reduction":cr,"per_budget":[{"budget":float(budgets[i]),"empirical_joint_reliability":float(target[:,i].mean()),"copula_predicted_joint_reliability":float(candidate[:,i].mean()),"independent_predicted_joint_reliability":float(independent[:,i].mean())} for i in range(len(budgets))],"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":c["claim_boundary"]};(d/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(d/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n");print(json.dumps({"run_dir":str(d),**summary},indent=2))


if __name__=="__main__":main()
