"""Train and confirm coverage-constrained fixed-total action allocation."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import BUDGET_CONDITIONED_FEATURE_NAMES, BUDGET_HORIZON_CONDITIONED_FEATURE_NAMES, HORIZON_CONDITIONED_FEATURE_NAMES, BoundedCaseOffset, FEATURE_NAMES, adaptive_fixed_total_selection, budget_conditioned_case_offset_dataset, budget_horizon_conditioned_case_offset_dataset, case_offset_dataset, coverage_constrained_selection, group_coverage_constrained_selection, horizon_conditioned_case_offset_dataset, nested_group_budget_selection, score_case_offset, train_case_offset
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
    fraction=float(config["compiler"]["fixed_selected_fraction"])
    training_fractions=config["model"].get("training_selected_fractions")
    training_horizons=config["model"].get("training_horizon_seconds_by_domain")
    if training_horizons and training_fractions:
        train_cases=budget_horizon_conditioned_case_offset_dataset(train,train_scores,[float(value) for value in training_fractions],[float(value) for value in training_horizons])
        feature_names=BUDGET_HORIZON_CONDITIONED_FEATURE_NAMES
    elif training_horizons:
        train_cases=horizon_conditioned_case_offset_dataset(train,train_scores,fraction,[float(value) for value in training_horizons])
        feature_names=HORIZON_CONDITIONED_FEATURE_NAMES
    elif training_fractions:
        train_cases=budget_conditioned_case_offset_dataset(train,train_scores,[float(value) for value in training_fractions])
        feature_names=BUDGET_CONDITIONED_FEATURE_NAMES
    else:
        train_cases=case_offset_dataset(train,train_scores,fraction)
        feature_names=FEATURE_NAMES
    model,mean,scale,training=train_case_offset(train_cases,config["model"],int(config["seed"]))
    artifact_name=str(config.get("artifact_name","COVERAGE_BUDGET_COMPILER.pt"))
    torch.save({"state_dict":model.state_dict(),"feature_names":list(feature_names),"hidden_dimension":int(config["model"]["hidden_dimension"]),"maximum_case_offset":float(config["model"]["maximum_case_offset"]),"mean":mean,"scale":scale,"frozen_action_compiler_run":config["inputs"]["action_compiler_run"]},run_dir/artifact_name)
    has_p24="adaptive_budget_run" in config["inputs"]
    if has_p24:
        p24,p24_mean,p24_scale=_load_offset(runs_root,config["inputs"]["adaptive_budget_run"],config["inputs"]["adaptive_budget_artifact"])
    _write_json(run_dir/"model_frozen.json",{"p20_ranking_frozen":True,"allocation_offset_frozen_before_confirmation_materialization":True,"p24_baseline_frozen":has_p24,"training_selected_fractions":training_fractions,"training_horizon_seconds_by_domain":training_horizons,"development_domain_count":int(len(np.unique(train_cases["domain_index"]))),"train_case_count":int(len(train_cases["case_index"]))})
    cache=Path(config["confirmation_materialization"]["cache_path"]);mat={"cache_path":str(cache),"cache_reused":cache.is_file()}
    if not cache.is_file():mat.update(materialize_quantiles(config["confirmation_materialization"]["data"],runs_root,cache))
    selection=_load(cache);scores=score_listwise_compiler(p20,selection,p20_mean,p20_scale)
    if "nested_evaluation_fractions" in config["compiler"]:
        low_fraction,high_fraction=[float(value) for value in config["compiler"]["nested_evaluation_fractions"]]
        low_cases=budget_conditioned_case_offset_dataset(selection,scores,[low_fraction]);high_cases=budget_conditioned_case_offset_dataset(selection,scores,[high_fraction])
        low_offsets=score_case_offset(model,low_cases,mean,scale);high_offsets=score_case_offset(model,high_cases,mean,scale)
        scene_to_group={int(key):int(value) for key,value in config["compiler"]["scene_to_group"].items()}
        nested=nested_group_budget_selection(selection,scores,low_offsets,high_offsets,low_fraction,high_fraction,int(config["compiler"]["maximum_actions_per_case"]),float(config["compiler"]["minimum_case_coverage"]),scene_to_group,float(config["compiler"]["minimum_group_case_coverage"]))
        low,high=nested["low_budget"],nested["high_budget"]
        fixed_low=_within_case_selection(np.asarray(selection["target_cost"],dtype=np.float32),scores,np.asarray(selection["case_index"]),np.asarray(selection["scene_index"]),low_fraction)
        fixed_high=_within_case_selection(np.asarray(selection["target_cost"],dtype=np.float32),scores,np.asarray(selection["case_index"]),np.asarray(selection["scene_index"]),high_fraction)
        improvement={"low_delta_over_fixed_p20":float(low["relative_cost_reduction"]-fixed_low["relative_cost_reduction"]),"high_delta_over_fixed_p20":float(high["relative_cost_reduction"]-fixed_high["relative_cost_reduction"])}
        g=config["gates"];gates={"exact_both_total_budgets":low["selected_action_count"]==low["fixed_total_action_budget"] and high["selected_action_count"]==high["fixed_total_action_budget"],"low_subset_of_high":nested["low_subset_of_high"],"minimum_group_case_coverage_both":low["minimum_group_case_coverage"]>=float(g["minimum_group_case_coverage"]) and high["minimum_group_case_coverage"]>=float(g["minimum_group_case_coverage"]),"minimum_low_cost_reduction":low["relative_cost_reduction"]>=float(g["minimum_low_cost_reduction"]),"minimum_high_cost_reduction":high["relative_cost_reduction"]>=float(g["minimum_high_cost_reduction"]),"minimum_reduction_delta_over_fixed_p20_both":improvement["low_delta_over_fixed_p20"]>=float(g["minimum_low_delta_over_fixed_p20"]) and improvement["high_delta_over_fixed_p20"]>=float(g["minimum_high_delta_over_fixed_p20"]),"minimum_nonincreasing_scene_support_both":low["scene_nonincreasing_count"]>=int(g["minimum_nonincreasing_scene_support"]) and high["scene_nonincreasing_count"]>=int(g["minimum_nonincreasing_scene_support"])}
        verdict=config.get("verdict_on_pass","supported_nested_budget_authority") if all(gates.values()) else config.get("verdict_on_failure","rejected_nested_budget_authority")
        summary={"schema_version":config["output_schema_version"],"task_id":config["task_id"],"hypothesis_id":config["hypothesis_id"],"status":"done","verdict":verdict,"role":config["role"],"claim_boundary":config["claim_boundary"],"training":training,"confirmation_materialization":mat,"nested_budget":nested,"fixed_p20":{"low_budget":fixed_low,"high_budget":fixed_high},"selection_improvement":improvement,"gate_results":gates,"failure_ledger_delta":"pending_result","resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/(1024**3),"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),"wall_seconds":time.monotonic()-started}}
        _write_json(run_dir/"summary.json",summary);_write_json(run_dir/"status.json",{"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()});return {"run_dir":str(run_dir),"verdict":verdict,"gate_results":gates}
    if training_horizons and training_fractions:
        cases=budget_horizon_conditioned_case_offset_dataset(selection,scores,[fraction],[float(config["model"]["confirmation_horizon_seconds"])])
    elif training_horizons:
        cases=horizon_conditioned_case_offset_dataset(selection,scores,fraction,[float(config["model"]["confirmation_horizon_seconds"])])
    else:
        cases=budget_conditioned_case_offset_dataset(selection,scores,[fraction]) if training_fractions else case_offset_dataset(selection,scores,fraction)
    offsets=score_case_offset(model,cases,mean,scale)
    if "scene_to_group" in config["compiler"]:
        scene_to_group={int(key):int(value) for key,value in config["compiler"]["scene_to_group"].items()}
        selective=group_coverage_constrained_selection(selection,scores,offsets,fraction,int(config["compiler"]["maximum_actions_per_case"]),float(config["compiler"]["minimum_case_coverage"]),scene_to_group,float(config["compiler"]["minimum_group_case_coverage"]))
    else:
        selective=coverage_constrained_selection(selection,scores,offsets,fraction,int(config["compiler"]["maximum_actions_per_case"]),float(config["compiler"]["minimum_case_coverage"]))
    p24_result=None
    if has_p24:
        p24_cases=case_offset_dataset(selection,scores,fraction)
        p24_offsets=score_case_offset(p24,p24_cases,p24_mean,p24_scale);p24_result=adaptive_fixed_total_selection(selection,scores,p24_offsets,fraction,int(config["compiler"]["p24_maximum_actions_per_case"]))
    fixed=_within_case_selection(np.asarray(selection["target_cost"],dtype=np.float32),scores,np.asarray(selection["case_index"]),np.asarray(selection["scene_index"]),fraction)
    improvement={"coverage_budget_delta_over_fixed_p20":float(selective["relative_cost_reduction"]-fixed["relative_cost_reduction"])}
    if p24_result is not None: improvement["coverage_budget_delta_over_p24"]=float(selective["relative_cost_reduction"]-p24_result["relative_cost_reduction"])
    g=config["gates"];gates={"exact_fixed_total_action_budget":selective["selected_action_count"]==selective["fixed_total_action_budget"],"minimum_case_coverage":selective["case_coverage"]>=float(g["minimum_case_coverage"]),"minimum_cost_reduction":selective["relative_cost_reduction"]>=float(g["minimum_cost_reduction"]),"minimum_reduction_delta_over_fixed_p20":improvement["coverage_budget_delta_over_fixed_p20"]>=float(g["minimum_reduction_delta_over_fixed_p20"]),"minimum_nonincreasing_scene_support":selective["scene_nonincreasing_count"]>=int(g["minimum_nonincreasing_scene_support"])}
    if "minimum_group_case_coverage" in g:
        gates["minimum_group_case_coverage"]=selective["minimum_group_case_coverage"]>=float(g["minimum_group_case_coverage"])
    verdict=config.get("verdict_on_pass","supported_coverage_constrained_fixed_total_budget") if all(gates.values()) else config.get("verdict_on_failure","rejected_coverage_constrained_fixed_total_budget")
    summary={"schema_version":config.get("output_schema_version","worldsim_v67.p25_coverage_budget_summary.v1"),"task_id":config["task_id"],"hypothesis_id":config["hypothesis_id"],"status":"done","verdict":verdict,"role":config["role"],"claim_boundary":config["claim_boundary"],"training":training,"confirmation_materialization":mat,"coverage_budget":selective,"fixed_p20":fixed,"selection_improvement":improvement,"gate_results":gates,"failure_ledger_delta":"pending_result","resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/(1024**3),"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),"wall_seconds":time.monotonic()-started}}
    if p24_result is not None: summary["p24_adaptive_baseline"]=p24_result
    _write_json(run_dir/"summary.json",summary);_write_json(run_dir/"status.json",{"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()});return {"run_dir":str(run_dir),"verdict":verdict,"gate_results":gates}


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();print(json.dumps(run(a.config.resolve(),a.runs_root.resolve(),a.run_id),indent=2))


if __name__=="__main__":main()
