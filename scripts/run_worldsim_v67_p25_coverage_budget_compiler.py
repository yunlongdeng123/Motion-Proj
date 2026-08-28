"""Train and confirm coverage-constrained fixed-total action allocation."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import BoundedCaseOffset, FEATURE_NAMES, adaptive_fixed_total_selection, case_offset_dataset, coverage_constrained_selection, score_case_offset, train_case_offset
from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, score_listwise_compiler
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None: path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _load_p20(config, runs_root):
    a=torch.load(runs_root/config["inputs"]["action_compiler_run"]/config["inputs"]["action_compiler_artifact"],map_location="cuda",weights_only=False)
    m=BoundedListwiseCompiler(len(a["feature_names"]),list(a["hidden_dimensions"]),float(a["maximum_residual_cost"])).cuda();m.load_state_dict(a["state_dict"])
    return m.eval(),np.asarray(a["mean"],dtype=np.float32),np.asarray(a["scale"],dtype=np.float32)


def _load_offset(runs_root, run, artifact):
    a=torch.load(runs_root/run/artifact,map_location="cuda",weights_only=False)
    m=BoundedCaseOffset(len(a["feature_names"]),int(a["hidden_dimension"]),float(a["maximum_case_offset"])).cuda();m.load_state_dict(a["state_dict"])
    return m.eval(),np.asarray(a["mean"],dtype=np.float32),np.asarray(a["scale"],dtype=np.float32)


def run(config_path: Path, runs_root: Path, run_id: str):
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"));run_dir=runs_root/"worldsim_v67"/config["task_id"]/run_id;run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/"resolved.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8");_write_json(run_dir/"status.json",{"status":"running","started_at_utc":datetime.now(timezone.utc).isoformat()})
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();p20,p20_mean,p20_scale=_load_p20(config,runs_root)
    train=_combine([Path(p) for p in config["inputs"]["train_action_caches"]]);train_scores=score_listwise_compiler(p20,train,p20_mean,p20_scale)
    fraction=float(config["compiler"]["fixed_selected_fraction"]);train_cases=case_offset_dataset(train,train_scores,fraction)
    model,mean,scale,training=train_case_offset(train_cases,config["model"],int(config["seed"]))
    torch.save({"state_dict":model.state_dict(),"feature_names":list(FEATURE_NAMES),"hidden_dimension":int(config["model"]["hidden_dimension"]),"maximum_case_offset":float(config["model"]["maximum_case_offset"]),"mean":mean,"scale":scale,"frozen_action_compiler_run":config["inputs"]["action_compiler_run"]},run_dir/"COVERAGE_BUDGET_COMPILER.pt")
    p24,p24_mean,p24_scale=_load_offset(runs_root,config["inputs"]["adaptive_budget_run"],config["inputs"]["adaptive_budget_artifact"])
    _write_json(run_dir/"model_frozen.json",{"p20_ranking_frozen":True,"p25_offset_frozen_before_confirmation_materialization":True,"p24_baseline_frozen":True,"development_domain_count":9,"train_case_count":int(len(train_cases["case_index"]))})
    cache=Path(config["confirmation_materialization"]["cache_path"]);mat={"cache_path":str(cache),"cache_reused":cache.is_file()}
    if not cache.is_file():mat.update(materialize_quantiles(config["confirmation_materialization"]["data"],runs_root,cache))
    selection=_load(cache);scores=score_listwise_compiler(p20,selection,p20_mean,p20_scale);cases=case_offset_dataset(selection,scores,fraction)
    offsets=score_case_offset(model,cases,mean,scale);selective=coverage_constrained_selection(selection,scores,offsets,fraction,int(config["compiler"]["maximum_actions_per_case"]),float(config["compiler"]["minimum_case_coverage"]))
    p24_offsets=score_case_offset(p24,cases,p24_mean,p24_scale);p24_result=adaptive_fixed_total_selection(selection,scores,p24_offsets,fraction,int(config["compiler"]["p24_maximum_actions_per_case"]))
    fixed=_within_case_selection(np.asarray(selection["target_cost"],dtype=np.float32),scores,np.asarray(selection["case_index"]),np.asarray(selection["scene_index"]),fraction)
    improvement={"coverage_budget_delta_over_fixed_p20":float(selective["relative_cost_reduction"]-fixed["relative_cost_reduction"]),"coverage_budget_delta_over_p24":float(selective["relative_cost_reduction"]-p24_result["relative_cost_reduction"])}
    g=config["gates"];gates={"exact_fixed_total_action_budget":selective["selected_action_count"]==selective["fixed_total_action_budget"],"minimum_case_coverage":selective["case_coverage"]>=float(g["minimum_case_coverage"]),"minimum_cost_reduction":selective["relative_cost_reduction"]>=float(g["minimum_cost_reduction"]),"minimum_reduction_delta_over_fixed_p20":improvement["coverage_budget_delta_over_fixed_p20"]>=float(g["minimum_reduction_delta_over_fixed_p20"]),"minimum_nonincreasing_scene_support":selective["scene_nonincreasing_count"]>=int(g["minimum_nonincreasing_scene_support"])}
    verdict="supported_coverage_constrained_fixed_total_budget" if all(gates.values()) else "rejected_coverage_constrained_fixed_total_budget"
    summary={"schema_version":"worldsim_v67.p25_coverage_budget_summary.v1","task_id":config["task_id"],"hypothesis_id":config["hypothesis_id"],"status":"done","verdict":verdict,"role":config["role"],"claim_boundary":config["claim_boundary"],"training":training,"confirmation_materialization":mat,"coverage_budget":selective,"p24_adaptive_baseline":p24_result,"fixed_p20":fixed,"selection_improvement":improvement,"gate_results":gates,"failure_ledger_delta":"pending_result","resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/(1024**3),"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),"wall_seconds":time.monotonic()-started}}
    _write_json(run_dir/"summary.json",summary);_write_json(run_dir/"status.json",{"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()});return {"run_dir":str(run_dir),"verdict":verdict,"gate_results":gates}


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();print(json.dumps(run(a.config.resolve(),a.runs_root.resolve(),a.run_id),indent=2))


if __name__=="__main__":main()
