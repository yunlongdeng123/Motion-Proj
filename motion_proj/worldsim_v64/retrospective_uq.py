"""在冻结 V6.3 原生 sidecar 上诊断 feature-level uncertainty。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from motion_proj.worldsim_v61.occupancy import FREE, UNKNOWN


@dataclass(frozen=True)
class PointChunk:
    """一个方法可见的曲面点块。"""

    features: np.ndarray
    logits: np.ndarray
    hidden_free: np.ndarray


def _unit_dirs(root: Path, scene: str) -> list[Path]:
    scene_root = root / "units" / scene
    if not scene_root.is_dir():
        raise FileNotFoundError(f"missing scene units: {scene_root}")
    units = sorted(path for path in scene_root.iterdir() if path.is_dir())
    if not units:
        raise RuntimeError(f"scene has no units: {scene}")
    return units


def _native_unit_dir(
    native_root: Path,
    scene: str,
    unit_name: str,
    partition_by_scene: Mapping[str, str] | None,
) -> Path:
    partition = (
        str(partition_by_scene.get(scene, "development"))
        if partition_by_scene is not None
        else "development"
    )
    return native_root / "units" / partition / scene / unit_name


def _eligible_indices(surface_unit: Path) -> np.ndarray:
    with np.load(surface_unit / "SURFACE_POINTS.npz", allow_pickle=False) as arrays:
        method = np.asarray(arrays["method_state"], dtype=np.uint8)
        contradiction = np.asarray(arrays["method_contradiction"], dtype=bool)
        native_valid = np.asarray(arrays["native_valid"], dtype=bool)
    return np.flatnonzero((method == UNKNOWN) & ~contradiction & native_valid)


def _load_points(
    surface_unit: Path,
    native_unit: Path,
    selected: np.ndarray,
) -> PointChunk:
    with np.load(surface_unit / "SURFACE_POINTS.npz", allow_pickle=False) as arrays:
        native_indices = np.asarray(arrays["native_indices"][selected], dtype=np.int64)
        method = np.asarray(arrays["method_state"][selected], dtype=np.uint8)
        target = np.asarray(arrays["target_state"][selected], dtype=np.uint8)
        contradiction = np.asarray(arrays["method_contradiction"][selected], dtype=bool)
    logits_grid = np.load(native_unit / "NATIVE_LOGITS.npy", mmap_mode="r")
    bev_grid = np.load(native_unit / "BEV_LATENT.npy", mmap_mode="r")
    logits = np.asarray(
        logits_grid[
            native_indices[:, 0],
            native_indices[:, 1],
            native_indices[:, 2],
        ],
        dtype=np.float32,
    )
    bev = np.asarray(
        bev_grid[native_indices[:, 0], native_indices[:, 1]],
        dtype=np.float32,
    )
    features = np.concatenate((logits, bev), axis=1)
    hidden_free = (method == UNKNOWN) & (target == FREE) & ~contradiction
    return PointChunk(features=features, logits=logits, hidden_free=hidden_free)


def iter_scene_chunks(
    surface_root: Path,
    native_root: Path,
    scene: str,
    *,
    chunk_size: int,
    native_partition_by_scene: Mapping[str, str] | None = None,
) -> Iterator[PointChunk]:
    """顺序读取一个 scene 的全部方法可见曲面点。"""

    for surface_unit in _unit_dirs(surface_root, scene):
        native_unit = _native_unit_dir(
            native_root, scene, surface_unit.name, native_partition_by_scene
        )
        if not native_unit.is_dir():
            raise FileNotFoundError(f"missing native unit: {native_unit}")
        eligible = _eligible_indices(surface_unit)
        for start in range(0, eligible.size, int(chunk_size)):
            yield _load_points(
                surface_unit,
                native_unit,
                eligible[start : start + int(chunk_size)],
            )


def sample_training_points(
    surface_root: Path,
    native_root: Path,
    scenes: Sequence[str],
    *,
    maximum_points_per_scene: int,
    seed: int,
    native_partition_by_scene: Mapping[str, str] | None = None,
) -> PointChunk:
    """按 scene 等额抽取 GMM 拟合点，selection target 不参与拟合。"""

    rng = np.random.default_rng(int(seed))
    sampled: list[PointChunk] = []
    for scene in scenes:
        units = _unit_dirs(surface_root, scene)
        per_unit = max(1, int(np.ceil(maximum_points_per_scene / len(units))))
        scene_parts: list[PointChunk] = []
        for surface_unit in units:
            eligible = _eligible_indices(surface_unit)
            if eligible.size > per_unit:
                eligible = rng.choice(eligible, size=per_unit, replace=False)
            native_unit = _native_unit_dir(
                native_root, scene, surface_unit.name, native_partition_by_scene
            )
            scene_parts.append(_load_points(surface_unit, native_unit, eligible))
        features = np.concatenate([part.features for part in scene_parts], axis=0)
        logits = np.concatenate([part.logits for part in scene_parts], axis=0)
        hidden_free = np.concatenate([part.hidden_free for part in scene_parts], axis=0)
        if features.shape[0] > maximum_points_per_scene:
            keep = rng.choice(
                features.shape[0], size=int(maximum_points_per_scene), replace=False
            )
            features = features[keep]
            logits = logits[keep]
            hidden_free = hidden_free[keep]
        sampled.append(
            PointChunk(features=features, logits=logits, hidden_free=hidden_free)
        )
    return PointChunk(
        features=np.concatenate([part.features for part in sampled], axis=0),
        logits=np.concatenate([part.logits for part in sampled], axis=0),
        hidden_free=np.concatenate([part.hidden_free for part in sampled], axis=0),
    )


def _geometry_class(logits: np.ndarray) -> np.ndarray:
    return (np.asarray(logits).argmax(axis=1) != 0).astype(np.uint8)


def _softmax_uncertainty(logits: np.ndarray) -> dict[str, np.ndarray]:
    shifted = np.asarray(logits, dtype=np.float32)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)), axis=1
    ) / np.log(probabilities.shape[1])
    top2 = np.partition(probabilities, -2, axis=1)[:, -2:]
    margin = top2.max(axis=1) - top2.min(axis=1)
    return {
        "u0_max_probability": 1.0 - probabilities.max(axis=1),
        "u0_entropy": entropy,
        "u0_inverse_margin": 1.0 - margin,
    }


class NativeFeatureDensityUQ:
    """OCCUQ 风格的 geometry-conditioned diagonal GMM。"""

    def __init__(
        self,
        *,
        pca_dimension: int,
        component_count: int,
        seed: int,
    ) -> None:
        self.pca_dimension = int(pca_dimension)
        self.component_count = int(component_count)
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components=self.pca_dimension,
            svd_solver="randomized",
            random_state=self.seed,
        )
        self.models: dict[int, GaussianMixture] = {}
        self.location_scale: dict[int, tuple[float, float]] = {}

    def fit(self, features: np.ndarray, logits: np.ndarray) -> "NativeFeatureDensityUQ":
        standardized = self.scaler.fit_transform(np.asarray(features, dtype=np.float32))
        projected = self.pca.fit_transform(standardized).astype(np.float32)
        geometry = _geometry_class(logits)
        for group in (0, 1):
            selected = projected[geometry == group]
            if selected.shape[0] < self.component_count * 20:
                raise RuntimeError(
                    f"geometry group {group} has too few points: {selected.shape[0]}"
                )
            model = GaussianMixture(
                n_components=self.component_count,
                covariance_type="diag",
                reg_covar=1e-5,
                max_iter=100,
                n_init=1,
                random_state=self.seed,
            )
            model.fit(selected)
            raw = -model.score_samples(selected)
            median = float(np.median(raw))
            scale = float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25))
            self.models[group] = model
            self.location_scale[group] = (median, max(scale, 1e-6))
        return self

    def score(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        standardized = self.scaler.transform(np.asarray(features, dtype=np.float32))
        projected = self.pca.transform(standardized).astype(np.float32)
        geometry = _geometry_class(logits)
        result = np.empty(projected.shape[0], dtype=np.float32)
        for group, model in self.models.items():
            selected = geometry == group
            if not np.any(selected):
                continue
            median, scale = self.location_scale[group]
            result[selected] = (
                -model.score_samples(projected[selected]) - median
            ) / scale
        return result


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "auroc": None,
            "auprc": None,
            "fpr_at_95_tpr": None,
        }
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": float(fpr[eligible[0]]) if eligible.size else 1.0,
    }


def _risk_coverage(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(np.asarray(scores, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.float64)[order]
    rows: list[dict[str, float]] = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(np.floor(labels.size * coverage)))
        rows.append(
            {
                "coverage": float(coverage),
                "hidden_free_risk": float(labels[:count].mean()),
            }
        )
    return rows


def evaluate_scene(
    model: NativeFeatureDensityUQ,
    surface_root: Path,
    native_root: Path,
    scene: str,
    *,
    chunk_size: int,
    native_partition_by_scene: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """在一个完整 scene denominator 上计算 U0/U2。"""

    label_parts: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {
        "u0_max_probability": [],
        "u0_entropy": [],
        "u0_inverse_margin": [],
        "u2_feature_density": [],
    }
    for chunk in iter_scene_chunks(
        surface_root,
        native_root,
        scene,
        chunk_size=chunk_size,
        native_partition_by_scene=native_partition_by_scene,
    ):
        label_parts.append(chunk.hidden_free)
        for name, values in _softmax_uncertainty(chunk.logits).items():
            score_parts[name].append(np.asarray(values, dtype=np.float32))
        score_parts["u2_feature_density"].append(
            model.score(chunk.features, chunk.logits)
        )
    labels = np.concatenate(label_parts)
    scores = {name: np.concatenate(parts) for name, parts in score_parts.items()}
    metrics = {
        "scene": scene,
        "point_count": int(labels.size),
        "hidden_free_count": int(labels.sum()),
        "hidden_free_prevalence": float(labels.mean()),
        "scores": {
            name: {
                **_binary_metrics(labels, values),
                "risk_coverage": _risk_coverage(labels, values),
            }
            for name, values in scores.items()
        },
    }
    return metrics, {"labels": labels, **scores}


def pooled_metrics(
    scene_arrays: Sequence[dict[str, np.ndarray]],
) -> dict[str, object]:
    """合并 scene denominator，但保留逐 scene 结果作为主要诊断。"""

    labels = np.concatenate([row["labels"] for row in scene_arrays])
    names = [name for name in scene_arrays[0] if name != "labels"]
    return {
        "point_count": int(labels.size),
        "hidden_free_count": int(labels.sum()),
        "hidden_free_prevalence": float(labels.mean()),
        "scores": {
            name: {
                **_binary_metrics(
                    labels, np.concatenate([row[name] for row in scene_arrays])
                ),
                "risk_coverage": _risk_coverage(
                    labels, np.concatenate([row[name] for row in scene_arrays])
                ),
            }
            for name in names
        },
    }
