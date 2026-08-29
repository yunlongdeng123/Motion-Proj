"""Learn joint multi-horizon dependence over frozen P182 cost-density marginals."""

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


class JointHorizonCopula(nn.Module):
    def __init__(self, input_count: int, hidden_dimensions: list[int], dimension: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_count
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        output = nn.Linear(width, dimension * (dimension + 1) // 2)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)
        self.dimension = dimension

    def correlation_cholesky(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features)
        batch = len(features)
        lower = torch.zeros((batch, self.dimension, self.dimension), device=features.device)
        cursor = 0
        for row in range(self.dimension):
            for column in range(row + 1):
                value = raw[:, cursor]
                lower[:, row, column] = functional.softplus(value) + 0.2 if row == column else torch.tanh(value)
                cursor += 1
        covariance = lower @ lower.transpose(1, 2)
        scale = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2).clamp_min(1e-8))
        correlation = covariance / (scale[:, :, None] * scale[:, None, :])
        eye = torch.eye(self.dimension, device=features.device)[None]
        return torch.linalg.cholesky(correlation + 1e-4 * eye)


def _load_density(path: Path) -> tuple[LogCostMixtureDensity, dict]:
    frozen = torch.load(path, map_location="cuda")
    model = LogCostMixtureDensity(int(frozen["component_count"]), list(frozen["hidden_dimensions"])).cuda()
    model.load_state_dict(frozen["model_state_dict"])
    return model.eval(), frozen


def _identities(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.unique(np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1), axis=0)


def _trajectory_payload(arrays: dict[str, np.ndarray], members: list, ensemble: dict, floor: float):
    score, scenes = _ensemble_trajectory_score(
        arrays, members, np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    cost, cost_scenes = _continuous_cost(arrays, floor)
    horizon, clearance, identities = _trajectory_horizon(arrays), _trajectory_clearance(arrays), _identities(arrays)
    if not (len(identities) == len(score) == len(cost)) or not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P199 trajectory payload is not aligned")
    return identities, score, horizon, clearance, cost


def _align(payload, horizons: np.ndarray):
    identities, score, horizon, clearance, cost = payload
    table = {
        (int(row[0]), int(row[2]), int(row[3])): (float(score[i]), float(clearance[i]), float(cost[i]))
        for i, row in enumerate(identities)
    }
    keys_by_horizon = []
    tables = []
    for value in horizons:
        mask = np.isclose(horizon, float(value))
        local = {
            (int(row[0]), int(row[2]), int(row[3])): (float(score[i]), float(clearance[i]), float(cost[i]))
            for i, row in enumerate(identities) if mask[i]
        }
        tables.append(local)
        keys_by_horizon.append(set(local))
    keys = sorted(set.intersection(*keys_by_horizon))
    scores = np.asarray([[tables[j][key][0] for j in range(len(horizons))] for key in keys], dtype=np.float32)
    clearances = np.asarray([[tables[j][key][1] for j in range(len(horizons))] for key in keys], dtype=np.float32)
    costs = np.asarray([[tables[j][key][2] for j in range(len(horizons))] for key in keys], dtype=np.float32)
    scenes = np.asarray([key[0] for key in keys], dtype=np.int32)
    return scores, clearances, costs, scenes


@torch.no_grad()
def _variable_cdf(model, score, horizon, clearance, threshold, norms):
    condition = np.stack(((score-norms[0])/norms[1], (horizon-norms[2])/norms[3], (clearance-norms[4])/norms[5]), axis=1).astype(np.float32)
    outputs = []
    for start in range(0, len(score), 131072):
        x = torch.from_numpy(condition[start:start+131072]).cuda()
        y = torch.from_numpy(np.log1p(threshold[start:start+131072]).astype(np.float32)).cuda()
        logits, means, scales = model(x)
        standardized = (y[:, None] - means) / scales
        component = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        outputs.append(torch.sum(functional.softmax(logits, dim=1) * component, dim=1).cpu().numpy())
    return np.concatenate(outputs)


def _joint_probabilities(model, features, marginal, samples, seed):
    model.eval()
    normal = torch.distributions.Normal(torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda"))
    outputs = []
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for start in range(0, len(features), 512):
        x = features[start:start+512]
        probability = marginal[start:start+512]
        chol = model.correlation_cholesky(x)
        threshold = normal.icdf(probability.clamp(1e-4, 1.0-1e-4))
        noise = torch.randn((len(x), samples, 4), generator=generator, device="cuda")
        draws = torch.einsum("bij,bsj->bsi", chol, noise)
        joint = (draws[:, :, :, None] <= threshold[:, None, :, :]).all(dim=2).float().mean(dim=1)
        outputs.append(joint.cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--runs-root", type=Path, required=True); parser.add_argument("--run-id", required=True); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()); run_dir = args.runs_root/"worldsim_v67"/config["task_id"]/args.run_id; run_dir.mkdir(parents=True, exist_ok=False); (run_dir/"resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.monotonic(); torch.manual_seed(int(config["seed"])); torch.cuda.reset_peak_memory_stats()
    ensemble = torch.load(args.runs_root/config["frozen_p126"]["run"]/config["frozen_p126"]["artifact"], map_location="cuda")
    members = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda(); member.load_state_dict(state); members.append(member.eval())
    density, frozen = _load_density(args.runs_root/config["frozen_p182"]["run"]/config["frozen_p182"]["artifact"]); norms = tuple(frozen["norms"])
    with np.load(args.runs_root/config["source_rows"]["run"]/config["source_rows"]["artifact"], allow_pickle=False) as loaded: arrays = {name: loaded[name] for name in loaded.files}
    horizons = np.asarray(config["horizons_seconds"], dtype=np.float32); scores, clearances, costs, scenes = _align(_trajectory_payload(arrays, members, ensemble, float(config["boundary_state_cost"]["clearance_floor_m"])), horizons)
    marginal_at_cost = np.stack([_variable_cdf(density, scores[:,i], np.full(len(scores), horizons[i], np.float32), clearances[:,i], costs[:,i], norms) for i in range(4)], axis=1)
    clip = float(config["evaluation"]["probability_clip"]); normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0)); z = normal.icdf(torch.from_numpy(np.clip(marginal_at_cost, clip, 1.0-clip))).numpy().astype(np.float32)
    raw_features = np.concatenate((scores, clearances), axis=1); split = config["split"]; development = scenes % int(split["development_scene_modulus"]) == int(split["development_scene_remainder"]); training = ~development
    mean = raw_features[training].mean(axis=0); scale = raw_features[training].std(axis=0).clip(1e-6); features = torch.from_numpy(((raw_features-mean)/scale).astype(np.float32)).cuda(); z_tensor = torch.from_numpy(z).cuda(); train_indices = torch.from_numpy(np.flatnonzero(training)).cuda()
    model_config = config["model"]; model = JointHorizonCopula(8, model_config["hidden_dimensions"], 4).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])); batch = int(model_config["batch_size"]); final_nll = 0.0
    for step in range(int(model_config["training_steps"])):
        index = train_indices[torch.randint(len(train_indices), (batch,), device="cuda")]; chol = model.correlation_cholesky(features[index]); target = z_tensor[index]; solved = torch.cholesky_solve(target[:, :, None], chol).squeeze(2); loss = 0.5 * (2.0*torch.log(torch.diagonal(chol,dim1=1,dim2=2)).sum(dim=1) + (target*solved).sum(dim=1)).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step(); final_nll=float(loss.detach())
        if step % 500 == 0: print(f"P199 copula step={step+1} nll={final_nll:.6f}", flush=True)
    dev_scores, dev_clearances, dev_costs = scores[development], clearances[development], costs[development]; budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    marginal_budget = np.stack([_predict_cdf(density, dev_scores[:,i], np.full(len(dev_scores),horizons[i],np.float32), dev_clearances[:,i], budgets, norms) for i in range(4)], axis=1)
    marginal_tensor = torch.from_numpy(marginal_budget.astype(np.float32)).cuda(); development_indices = torch.from_numpy(np.flatnonzero(development)).cuda(); candidate = _joint_probabilities(model, features[development_indices], marginal_tensor, int(config["evaluation"]["monte_carlo_samples"]), int(config["seed"])); independent = np.prod(marginal_budget, axis=1); target = np.all(dev_costs[:,:,None] <= budgets[None,None,:], axis=1)
    candidate_brier=float(np.mean((candidate-target)**2)); independent_brier=float(np.mean((independent-target)**2)); candidate_error=float(np.mean(np.abs(candidate.mean(axis=0)-target.mean(axis=0)))); independent_error=float(np.mean(np.abs(independent.mean(axis=0)-target.mean(axis=0)))); brier_reduction=(independent_brier-candidate_brier)/independent_brier; calibration_reduction=(independent_error-candidate_error)/max(independent_error,1e-12)
    checks={"joint_integrated_brier_strictly_better_than_independent_marginals":candidate_brier<independent_brier,"minimum_joint_calibration_error_reduction":calibration_reduction>=float(config["decision"]["minimum_joint_calibration_error_reduction"])}; verdict=config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    artifact={"model_state_dict":model.state_dict(),"feature_mean":mean,"feature_scale":scale,"horizons_seconds":horizons,"hidden_dimensions":model_config["hidden_dimensions"],"marginal_density":config["frozen_p182"]}; torch.save(artifact,run_dir/config["model_artifact"])
    summary={"schema_version":config["output_schema_version"],"task_id":config["task_id"],"hypothesis_id":config["hypothesis_id"],"status":"done","verdict":verdict,"role":config["role"],"alignment":{"complete_joint_trajectory_count":int(len(scores)),"training_count":int(training.sum()),"development_count":int(development.sum()),"training_scene_count":int(len(np.unique(scenes[training]))),"development_scene_count":int(len(np.unique(scenes[development])))},"training":{"final_copula_nll_without_marginal_constant":final_nll},"development":{"joint_integrated_brier":candidate_brier,"independent_marginal_product_integrated_brier":independent_brier,"joint_brier_reduction":brier_reduction,"joint_mean_absolute_reliability_error":candidate_error,"independent_mean_absolute_reliability_error":independent_error,"joint_calibration_error_reduction":calibration_reduction,"per_budget":[{"budget":float(budgets[i]),"empirical_joint_reliability":float(target[:,i].mean()),"copula_predicted_joint_reliability":float(candidate[:,i].mean()),"independent_predicted_joint_reliability":float(independent[:,i].mean())} for i in range(len(budgets))]},"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":config["claim_boundary"]}; (run_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(run_dir/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n");print(json.dumps({"run_dir":str(run_dir),**summary},indent=2),flush=True)


if __name__ == "__main__": main()
