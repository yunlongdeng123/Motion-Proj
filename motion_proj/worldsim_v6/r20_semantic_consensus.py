"""WorldSim V6 R20 双架构 semantic consensus 正式实验。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

TASK_ID = "WS-V6-R20-SEMANTIC-CONSENSUS-01"


class R20ExperimentError(RuntimeError):
    """R20 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R20ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _load_semantic_model(root: Path) -> torch.nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=19,
        aux_loss=True,
    )
    # 冻结权重的 auxiliary head 保留 torchvision 默认 21 类输出。
    model.aux_classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)
    checkpoint = torch.load(
        root / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    return model.eval().cuda()


@torch.inference_mode()
def _predict_semantic(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    logits = model(((tensor - mean) / std).cuda())["out"]
    return logits.argmax(dim=1)[0].to(torch.int16).cpu().numpy()


@torch.inference_mode()
def _predict_segformer(
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    image: np.ndarray,
) -> np.ndarray:
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda(non_blocking=False)
    logits = model(pixel_values=pixel_values).logits
    resized = functional.interpolate(
        logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    return resized.argmax(dim=1)[0].cpu().numpy().astype(np.int16)


def _masked_iou(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    union = mask & (first | second)
    if not np.any(union):
        return 1.0
    return float(np.count_nonzero(mask & first & second) / np.count_nonzero(union))


def run_experiment(
    repo_root: Path,
    config_path: Path,
    run_root: Path,
    deeplab_root: Path,
    segformer_root: Path,
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R20ExperimentError("正式 R20 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R20ExperimentError("R20 task_id 漂移")
    source_run = _resolve_runs_uri(config["sources"]["r16_run"])
    source_files = {
        source_run / "MANIFEST.json": config["sources"]["r16_manifest_sha256"],
        source_run / "R16_GATE.json": config["sources"]["r16_gate_sha256"],
        source_run / "CASES.jsonl": config["sources"]["r16_cases_sha256"],
        source_run
        / "verifier_worker/PER_CASE_ARMS.jsonl": config["sources"][
            "r16_per_case_arms_sha256"
        ],
        deeplab_root / config["models"]["deeplab"]["model_file"]: config["models"][
            "deeplab"
        ]["model_sha256"],
        segformer_root / config["models"]["segformer"]["model_file"]: config[
            "models"
        ]["segformer"]["model_sha256"],
        segformer_root / "config.json": config["models"]["segformer"][
            "config_sha256"
        ],
        segformer_root / "preprocessor_config.json": config["models"]["segformer"][
            "preprocessor_sha256"
        ],
    }
    for path, expected in source_files.items():
        if _sha256(path) != expected:
            raise R20ExperimentError(f"冻结输入漂移：{path}")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R20ExperimentError("R20 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__semantic-consensus-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cases = {row["case_id"]: row for row in _read_jsonl(source_run / "CASES.jsonl")}
        source_rows = _read_jsonl(source_run / "verifier_worker/PER_CASE_ARMS.jsonl")
        evidence_rows = [row for row in source_rows if row["P3"]["semantic_evidence"]]
        expected_count = int(config["cohort"]["expected_semantic_evidence_case_count"])
        if len(evidence_rows) != expected_count:
            raise R20ExperimentError("R20 semantic evidence 分母漂移")

        deeplab = _load_semantic_model(deeplab_root)
        segformer_processor = SegformerImageProcessor.from_pretrained(
            segformer_root, local_files_only=True
        )
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            segformer_root, local_files_only=True
        ).cuda().eval()
        dynamic_ids = set(int(value) for value in config["consensus"]["dynamic_class_ids"])
        minimum_iou = float(config["consensus"]["minimum_dynamic_iou"])
        proposal_directory = str(config["sources"]["proposal_directory"])

        rows: list[dict[str, Any]] = []
        for source_row in evidence_rows:
            case_id = source_row["case_id"]
            case = cases[case_id]
            proposal_path = source_run / proposal_directory / f"{case_id}__repeat1.npy"
            if _sha256(proposal_path) != source_row["proposal_sha256"]:
                raise R20ExperimentError(f"proposal 漂移：{case_id}")
            verifier_input = source_run / "verifier_inputs" / f"{case_id}.npz"
            if _sha256(verifier_input) != case["verifier_input_sha256"]:
                raise R20ExperimentError(f"verifier input 漂移：{case_id}")
            proposal = np.load(proposal_path, allow_pickle=False).astype(np.uint8)
            with np.load(verifier_input, allow_pickle=False) as archive:
                mask = np.asarray(archive["mask"], dtype=bool)

            deeplab_labels = _predict_semantic(deeplab, proposal)
            segformer_labels = _predict_segformer(
                segformer_processor, segformer, proposal
            )
            deeplab_dynamic = np.isin(deeplab_labels, list(dynamic_ids))
            segformer_dynamic = np.isin(segformer_labels, list(dynamic_ids))
            consensus_iou = _masked_iou(deeplab_dynamic, segformer_dynamic, mask)
            accepted = consensus_iou >= minimum_iou
            truth_safe = bool(source_row["P3"]["truth_safe"])
            rows.append(
                {
                    "schema_version": "worldsim_v6.r20_semantic_consensus.v1",
                    "case_id": case_id,
                    "hole_type": source_row["hole_type"],
                    "proposal_sha256": source_row["proposal_sha256"],
                    "consensus_dynamic_iou": consensus_iou,
                    "deeplab_dynamic_pixels_in_mask": int(
                        np.count_nonzero(mask & deeplab_dynamic)
                    ),
                    "segformer_dynamic_pixels_in_mask": int(
                        np.count_nonzero(mask & segformer_dynamic)
                    ),
                    "decision": "ACCEPT" if accepted else "REJECT",
                    "truth_safe": truth_safe,
                    "false_safe": bool(accepted and not truth_safe),
                    "original_p3_decision": source_row["P3"]["decision"],
                    "original_p3_false_safe": bool(source_row["P3"]["false_safe"]),
                }
            )
        _write_jsonl(run_dir / "SEMANTIC_CONSENSUS.jsonl", rows)

        accepted_rows = [row for row in rows if row["decision"] == "ACCEPT"]
        false_safe_count = sum(row["false_safe"] for row in accepted_rows)
        false_safe_rate = (
            0.0 if not accepted_rows else float(false_safe_count / len(accepted_rows))
        )
        accept_coverage = float(len(accepted_rows) / expected_count)
        p0_false_safe_count = sum(not row["truth_safe"] for row in rows)
        p0_false_safe_rate = float(p0_false_safe_count / expected_count)
        false_safe_reduction = float(p0_false_safe_rate - false_safe_rate)
        original_accepts = [row for row in rows if row["original_p3_decision"] == "ACCEPT"]
        original_false_safe_rate = float(
            sum(row["original_p3_false_safe"] for row in original_accepts)
            / len(original_accepts)
        )
        wall_seconds = time.monotonic() - started
        gate_cfg = config["gate"]
        checks = {
            "semantic_evidence_count_exact": len(rows) == expected_count,
            "minimum_accept_coverage": accept_coverage
            >= float(gate_cfg["minimum_accept_coverage"]),
            "maximum_false_safe_rate": false_safe_rate
            <= float(gate_cfg["maximum_false_safe_rate"]),
            "minimum_false_safe_reduction_vs_p0": false_safe_reduction
            >= float(gate_cfg["minimum_false_safe_reduction_vs_p0"]),
            "decision_uses_model_consensus_only": True,
            "target_dynamic_not_read_by_decision": True,
            "architectures_distinct": config["models"]["deeplab"]["architecture"]
            != config["models"]["segformer"]["architecture"],
            "source_immutable": all(_sha256(path) == expected for path, expected in source_files.items()),
            "gpu_within_budget": float(torch.cuda.max_memory_reserved() / (1024**2))
            <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
            "bake_not_started": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r20_gate.v1",
            "checks": checks,
            "decision": "semantic_consensus_arm_eligible"
            if checks["passed"]
            else "reject_or_pivot_semantic_consensus",
        }
        _write_json(run_dir / "R20_GATE.json", gate)
        metrics = {
            "schema_version": "worldsim_v6.r20_metrics.v1",
            "case_count": expected_count,
            "accept_count": len(accepted_rows),
            "reject_count": expected_count - len(accepted_rows),
            "accept_coverage": accept_coverage,
            "false_safe_count": false_safe_count,
            "false_safe_rate": false_safe_rate,
            "p0_false_safe_count": p0_false_safe_count,
            "p0_false_safe_rate": p0_false_safe_rate,
            "false_safe_reduction_vs_p0": false_safe_reduction,
            "original_p3_accept_count": len(original_accepts),
            "original_p3_false_safe_rate": original_false_safe_rate,
        }
        _write_json(run_dir / "METRICS.json", metrics)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r20_resource_audit.v1",
                "gpu": config["resources"]["gpu"],
                "peak_gpu_memory_mib": float(
                    torch.cuda.max_memory_reserved() / (1024**2)
                ),
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r20_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "semantic_consensus_eligible": bool(checks["passed"]),
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
            "bake_started": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "SEMANTIC_CONSENSUS.jsonl",
            "R20_GATE.json",
            "METRICS.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r20_manifest.v1",
                "files": {
                    name: {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r20_semantic_consensus_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--deeplab-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/worldsim_v6/r9_semantic_deeplab_cityscapes"
        ),
    )
    parser.add_argument(
        "--segformer-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/worldsim_v6/r20_semantic_segformer_cityscapes"
        ),
    )
    args = parser.parse_args()
    run_experiment(
        args.repo_root,
        args.config,
        args.run_root,
        args.deeplab_root,
        args.segformer_root,
    )
    return 0
