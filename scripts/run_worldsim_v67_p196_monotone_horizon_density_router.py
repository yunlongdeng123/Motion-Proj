"""Fit a two-scalar monotone horizon router over frozen P182/P192 density experts."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import LogCostMixtureDensity, _predict_cdf


class MonotoneHorizonRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def forward(self, normalized_horizon: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(functional.softplus(self.raw_slope) * normalized_horizon + self.intercept)


def _load_density(path: Path) -> tuple[LogCostMixtureDensity, dict]:
    frozen = torch.load(path, map_location="cuda")
    model = LogCostMixtureDensity(int(frozen["component_count"]), list(frozen["hidden_dimensions"])).cuda()
    model.load_state_dict(frozen["model_state_dict"])
    return model.eval(), frozen


@torch.no_grad()
def _log_probability(
    model: LogCostMixtureDensity, score: np.ndarray, horizon: np.ndarray, clearance: np.ndarray,
    cost: np.ndarray, norms: tuple[float, ...],
) -> torch.Tensor:
    condition = np.stack(((score-norms[0])/norms[1], (horizon-norms[2])/norms[3], (clearance-norms[4])/norms[5]), axis=1).astype(np.float32)
    x = torch.from_numpy(condition).cuda()
    target = torch.from_numpy(np.log1p(cost).astype(np.float32)).cuda()
    outputs = []
    for start in range(0, len(x), 131072):
        logits, means, scales = model(x[start:start+131072])
        standardized = (target[start:start+131072, None] - means) / scales
        component = -0.5*standardized.square() - torch.log(scales) - 0.5*math.log(2*math.pi)
        outputs.append(torch.logsumexp(functional.log_softmax(logits, dim=1)+component, dim=1))
    return torch.cat(outputs)


def _prepare(arrays: dict[str, np.ndarray], actor_models: list, actor_ensemble: dict, floor: float):
    score, scenes = _ensemble_trajectory_score(arrays, actor_models, np.asarray(actor_ensemble["feature_mean"],dtype=np.float32), np.asarray(actor_ensemble["feature_scale"],dtype=np.float32), np.asarray(actor_ensemble["target_mean"],dtype=np.float32), np.asarray(actor_ensemble["target_scale"],dtype=np.float32))
    cost, cost_scenes = _continuous_cost(arrays, floor)
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P196 trajectory grouping is not aligned")
    return score, _trajectory_horizon(arrays), _trajectory_clearance(arrays), cost


def _evaluate(arrays, actor_models, actor_ensemble, p182, f182, p192, f192, router, config):
    score, horizon, clearance, cost = _prepare(arrays, actor_models, actor_ensemble, float(config["boundary_state_cost"]["clearance_floor_m"]))
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    a = _predict_cdf(p182, score, horizon, clearance, budgets, tuple(f182["norms"]))
    b = _predict_cdf(p192, score, horizon, clearance, budgets, tuple(f192["norms"]))
    lo, hi = float(config["router"]["minimum_horizon_seconds"]), float(config["router"]["maximum_horizon_seconds"])
    normalized = np.clip((horizon-lo)/(hi-lo), 0.0, 1.0).astype(np.float32)
    with torch.no_grad():
        weight = router(torch.from_numpy(normalized).cuda()).cpu().numpy()
    predicted = (1-weight[:,None])*a + weight[:,None]*b
    target = cost[:,None] <= budgets[None]
    brier = float(np.mean(np.square(predicted-target)))
    control_brier = float(np.mean(np.square(a-target)))
    calibration = float(np.mean(np.abs(predicted.mean(axis=0)-target.mean(axis=0))))
    control_calibration = float(np.mean(np.abs(a.mean(axis=0)-target.mean(axis=0))))
    return {
        "trajectory_count": int(len(cost)), "router_integrated_brier": brier,
        "p182_integrated_brier": control_brier,
        "brier_change_vs_p182": (brier-control_brier)/max(control_brier,1e-12),
        "router_mean_absolute_reliability_error": calibration,
        "p182_mean_absolute_reliability_error": control_calibration,
        "calibration_error_reduction_vs_p182": (control_calibration-calibration)/max(control_calibration,1e-12),
        "mean_p192_weight": float(np.mean(weight)),
    }


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--runs-root",type=Path,required=True); parser.add_argument("--run-id",required=True); args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text()); run_dir=args.runs_root/"worldsim_v67"/config["task_id"]/args.run_id; run_dir.mkdir(parents=True,exist_ok=False); (run_dir/"resolved.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    started=time.monotonic(); torch.manual_seed(int(config["seed"])); torch.cuda.reset_peak_memory_stats()
    actor_ensemble=torch.load(args.runs_root/config["frozen_p126"]["run"]/config["frozen_p126"]["artifact"],map_location="cuda")
    actor_models=[]
    for state in actor_ensemble["member_state_dicts"]:
        model=DirectionalActorGaussian(20,actor_ensemble["hidden_dimensions"]).cuda(); model.load_state_dict(state); actor_models.append(model.eval())
    p182,f182=_load_density(args.runs_root/config["frozen_p182"]["run"]/config["frozen_p182"]["artifact"])
    p192,f192=_load_density(args.runs_root/config["frozen_p192"]["run"]/config["frozen_p192"]["artifact"])
    with np.load(args.runs_root/config["source_rows"]["run"]/config["source_rows"]["artifact"],allow_pickle=False) as loaded: source={n:loaded[n] for n in loaded.files}
    score,horizon,clearance,cost=_prepare(source,actor_models,actor_ensemble,float(config["boundary_state_cost"]["clearance_floor_m"]))
    log_a=_log_probability(p182,score,horizon,clearance,cost,tuple(f182["norms"])); log_b=_log_probability(p192,score,horizon,clearance,cost,tuple(f192["norms"]))
    lo,hi=float(config["router"]["minimum_horizon_seconds"]),float(config["router"]["maximum_horizon_seconds"])
    normalized=torch.from_numpy(np.clip((horizon-lo)/(hi-lo),0,1).astype(np.float32)).cuda()
    router=MonotoneHorizonRouter().cuda(); optimizer=torch.optim.Adam(router.parameters(),lr=float(config["router"]["learning_rate"])); batch=int(config["router"]["batch_size"])
    for step in range(int(config["router"]["training_steps"])):
        idx=torch.randint(len(normalized),(batch,),device="cuda"); weight=router(normalized[idx]); mixture=torch.logaddexp(torch.log1p(-weight)+log_a[idx],torch.log(weight)+log_b[idx]); loss=-mixture.mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step%500==0: print(f"P196 router step={step+1} nll={float(loss):.6f}",flush=True)
    evaluations={}
    for cohort in config["decision_cohorts"]:
        with np.load(args.runs_root/cohort["run"]/cohort["artifact"],allow_pickle=False) as loaded: arrays={n:loaded[n] for n in loaded.files}
        evaluations[cohort["name"]]=_evaluate(arrays,actor_models,actor_ensemble,p182,f182,p192,f192,router,config)
    calibration=[r["calibration_error_reduction_vs_p182"] for r in evaluations.values()]
    checks={"brier_noninferior_to_p182_every_cohort":all(r["router_integrated_brier"]<=r["p182_integrated_brier"] for r in evaluations.values()),"minimum_mean_calibration_error_reduction_vs_p182":float(np.mean(calibration))>=float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"])}
    verdict=config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    artifact={"model_state_dict":router.state_dict(),"minimum_horizon_seconds":lo,"maximum_horizon_seconds":hi,"p182":config["frozen_p182"],"p192":config["frozen_p192"]}; torch.save(artifact,run_dir/config["model_artifact"])
    with torch.no_grad(): endpoint=router(torch.tensor([0.,1.],device="cuda")).cpu().tolist()
    summary={"schema_version":config["output_schema_version"],"task_id":config["task_id"],"hypothesis_id":config["hypothesis_id"],"status":"done","verdict":verdict,"role":config["role"],"router":{"p192_weight_at_min_horizon":endpoint[0],"p192_weight_at_max_horizon":endpoint[1],"positive_slope":float(functional.softplus(router.raw_slope).detach())},"consumed_development_evaluations":evaluations,"mean_calibration_error_reduction_vs_p182":float(np.mean(calibration)),"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":config["claim_boundary"]}
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (run_dir/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n"); print(json.dumps({"run_dir":str(run_dir),**summary},indent=2),flush=True)


if __name__ == "__main__": main()
