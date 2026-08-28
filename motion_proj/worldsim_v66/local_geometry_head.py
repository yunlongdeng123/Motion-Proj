"""低容量 instance-evidence Actor local geometry head。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


FEATURE_NAMES = (
    "q0_mean",
    "q0_p90",
    "log_boundary_count",
    "log_sensor_hit_count",
    "log_current_envelope_count",
    "log_swept_envelope_count",
    "hit_current_density",
    "current_swept_ratio",
)


class LocalGeometryHead(torch.nn.Module):
    """固定两层小MLP；输出只服务local geometry排序。"""

    def __init__(self, input_dimension: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        dimensions = [int(input_dimension), *(int(value) for value in hidden_dimensions), 1]
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(torch.nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1)


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        current = max(float(row["current_envelope_count"]), 1.0)
        swept = max(float(row["swept_envelope_count"]), 1.0)
        values.append(
            [
                float(row["q0_mean"]),
                float(row["q0_p90"]),
                np.log1p(float(row["boundary_count"])),
                np.log1p(float(row["sensor_hit_count"])),
                np.log1p(current),
                np.log1p(swept),
                float(row["sensor_hit_count"]) / current,
                current / swept,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([bool(row["local_geometry_conflict"]) for row in rows], dtype=bool)


def score_head(
    model: LocalGeometryHead,
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    normalized = (np.asarray(values, dtype=np.float32) - mean) / scale
    with torch.inference_mode():
        logits = model(torch.from_numpy(normalized).cuda())
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def train_head(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    model_config: Mapping[str, Any],
    seed: int,
) -> tuple[LocalGeometryHead, np.ndarray, np.ndarray, dict[str, float]]:
    torch.manual_seed(int(seed))
    mean = np.asarray(train_values, dtype=np.float32).mean(axis=0)
    scale = np.maximum(np.asarray(train_values, dtype=np.float32).std(axis=0), 1e-6)
    normalized = (np.asarray(train_values, dtype=np.float32) - mean) / scale
    x = torch.from_numpy(normalized).cuda()
    y = torch.from_numpy(np.asarray(train_labels, dtype=np.float32)).cuda()
    model = LocalGeometryHead(
        train_values.shape[1], model_config["hidden_dimensions"]
    ).cuda()
    negative = int(np.count_nonzero(~train_labels))
    positive = int(np.count_nonzero(train_labels))
    if min(negative, positive) == 0:
        raise RuntimeError("P3L training requires both local-conflict classes")
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], device="cuda")
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = float("nan")
    model.train()
    for _ in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    return model, mean, scale, {
        "final_weighted_bce": final_loss,
        "positive_count": positive,
        "negative_count": negative,
    }


def ranking_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(target, scores)),
        "auprc": float(average_precision_score(target, scores)),
    }


def scene_support(
    rows: Sequence[Mapping[str, Any]], target: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    scene_rows = []
    for scene in sorted({str(row["scene"]) for row in rows}):
        members = np.asarray([str(row["scene"]) == scene for row in rows])
        if np.unique(target[members]).size != 2:
            scene_rows.append({"scene": scene, "evaluable": False})
            continue
        auroc = float(roc_auc_score(target[members], scores[members]))
        scene_rows.append(
            {
                "scene": scene,
                "evaluable": True,
                "row_count": int(np.count_nonzero(members)),
                "auroc": auroc,
                "above_chance": bool(auroc > 0.5),
            }
        )
    return {
        "evaluable_scene_count": sum(bool(row["evaluable"]) for row in scene_rows),
        "above_chance_scene_count": sum(
            bool(row.get("above_chance", False)) for row in scene_rows
        ),
        "scene_rows": scene_rows,
    }
