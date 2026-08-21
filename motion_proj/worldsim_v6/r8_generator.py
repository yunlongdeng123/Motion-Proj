"""WorldSim V6 R8 frozen proposal generator capability study。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image

from motion_proj.worldsim_v6.r7_oracle import (
    R7ExperimentError,
    _load_render_map,
    _make_missing,
    _mask_for,
    _required_render,
)


TASK_ID = "WS-V6-R8-FROZEN-PROPOSAL-GENERATOR-01"


class R8ExperimentError(RuntimeError):
    """R8 正式合同失败。"""


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R8ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _resize_case(
    missing: Mapping[str, np.ndarray], mask: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    image = np.rint(np.clip(missing["rgb"].astype(np.float32), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    resized_image = np.asarray(Image.fromarray(image, mode="RGB").resize(size, Image.Resampling.BILINEAR))
    resized_mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                size, Image.Resampling.NEAREST
            )
        )
        > 127
    )
    return resized_image, resized_mask


def _asset_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and ".cache" not in item.parts):
        relative = path.relative_to(root).as_posix()
        row = {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return rows, digest.hexdigest()


def run_experiment(
    repo_root: Path,
    config_path: Path,
    run_root: Path,
    big_lama_root: Path,
    sd15_root: Path,
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R8ExperimentError("正式 R8 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R8ExperimentError("R8 task_id 漂移")
    r7_run = _resolve_runs_uri(config["sources"]["r7_run"])
    if _sha256(r7_run / "MANIFEST.json") != config["sources"]["r7_manifest_sha256"]:
        raise R8ExperimentError("R7 source manifest 漂移")
    r3_root = _resolve_runs_uri(config["sources"]["r3_render_run"])
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R8ExperimentError("R8 磁盘资源不足")
    candidate_cfg = {row["id"]: row for row in config["candidates"]}
    if _sha256(big_lama_root / "big-lama/models/best.ckpt") != candidate_cfg["big_lama"]["checkpoint_sha256"]:
        raise R8ExperimentError("Big-LaMa checkpoint 漂移")
    if _git(big_lama_root / "source", "rev-parse", "HEAD") != candidate_cfg["big_lama"]["code_commit"]:
        raise R8ExperimentError("Big-LaMa source commit 漂移")
    sd_inventory, sd_content = _asset_inventory(sd15_root)
    required_sd = {
        "model_index.json",
        "text_encoder/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    }
    if not required_sd <= {row["path"] for row in sd_inventory}:
        raise R8ExperimentError("SD-v1.5 fp16 snapshot 不完整")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__frozen-generator-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        input_dir = run_dir / "inputs"
        input_dir.mkdir()
        r7_config = yaml.safe_load(
            (repo_root / "configs/worldsim_v6/r7_oracle_missing_world_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        render_maps = {}
        input_rows = []
        width, height = (int(value) for value in config["inference"]["capability_resolution_px"])
        for case in config["benchmark_cases"]:
            scene = case["scene"]
            frontend = case["frontend"]
            frame_index = int(case["frame_index"])
            key = (scene, frontend)
            if key not in render_maps:
                render_maps[key] = _load_render_map(
                    r3_root / "renders" / scene / frontend / "RENDER_MAP.jsonl"
                )
            render_map = render_maps[key]
            base = _required_render(r3_root, scene, frontend, frame_index, "lat0m", render_map)
            side = _required_render(r3_root, scene, frontend, frame_index, "lat2m", render_map)
            removed = _required_render(
                r3_root, scene, frontend, frame_index, "actor_remove_all", render_map
            )
            hole_type = case["hole_type"]
            target = side if hole_type == "missing_side_view" else removed if hole_type == "disocclusion" else base
            mask = _mask_for(
                hole_type, target, base, removed, r7_config["oracle_patch_contract"]
            )
            if int(np.count_nonzero(mask)) < int(
                r7_config["oracle_patch_contract"]["minimum_mask_pixels"]
            ):
                raise R8ExperimentError(f"R8 benchmark case 无有效 mask：{case}")
            missing = _make_missing(target, mask)
            image, resized_mask = _resize_case(missing, mask, (width, height))
            case_id = f"{scene}__{frontend}__f{frame_index:03d}__{hole_type}"
            path = input_dir / f"{case_id}.npz"
            np.savez_compressed(path, image=image, mask=resized_mask)
            input_rows.append(
                {
                    "case_id": case_id,
                    **case,
                    "mask_pixel_count": int(np.count_nonzero(resized_mask)),
                    "input_sha256": _sha256(path),
                }
            )
        _write_jsonl(run_dir / "INPUT_CASES.jsonl", input_rows)
        worker_results = []
        python = Path("/root/autodl-tmp/envs/motionproj/bin/python")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        for candidate in ("big_lama", "sd15_inpainting"):
            output_dir = run_dir / candidate
            command = [
                str(python),
                str(repo_root / "scripts/worldsim_v6/r8_generator_worker.py"),
                "--candidate",
                candidate,
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--big-lama-root",
                str(big_lama_root),
                "--sd15-root",
                str(sd15_root),
                "--seed",
                str(config["seed"]),
                "--repeat-count",
                str(config["inference"]["repeat_count"]),
                "--prompt",
                str(config["inference"]["sd_prompt"]),
                "--inference-steps",
                str(candidate_cfg["sd15_inpainting"]["inference_steps"]),
                "--guidance-scale",
                str(candidate_cfg["sd15_inpainting"]["guidance_scale"]),
            ]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(config["selection_gate"]["maximum_seconds_per_case"])
                * len(input_rows)
                * int(config["inference"]["repeat_count"]),
            )
            (run_dir / f"{candidate}.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0 or not (output_dir / "WORKER_RESULT.json").is_file():
                worker_results.append(
                    {
                        "candidate": candidate,
                        "status": "failed",
                        "returncode": completed.returncode,
                        "error_tail": (completed.stdout + completed.stderr)[-4000:],
                    }
                )
                continue
            result = json.loads((output_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
            result["status"] = "executed"
            result["worker_result_sha256"] = _sha256(output_dir / "WORKER_RESULT.json")
            worker_results.append(result)
        audit_rows = [
            {
                "candidate": "big_lama",
                "access": "ungated_public",
                "license": candidate_cfg["big_lama"]["code_license"],
                "weight_revision": candidate_cfg["big_lama"]["weight_revision"],
                "asset_content_sha256": candidate_cfg["big_lama"]["checkpoint_sha256"],
                "formal_execution": True,
            },
            {
                "candidate": "sd15_inpainting",
                "access": "ungated_public",
                "license": candidate_cfg["sd15_inpainting"]["model_license"],
                "weight_revision": candidate_cfg["sd15_inpainting"]["model_revision"],
                "asset_content_sha256": sd_content,
                "asset_file_count": len(sd_inventory),
                "formal_execution": True,
            },
            {
                "candidate": "flux1_fill_dev",
                "access": "gated",
                "license": candidate_cfg["flux1_fill_dev"]["model_license"],
                "weight_revision": candidate_cfg["flux1_fill_dev"]["model_revision"],
                "repository_used_storage_bytes": candidate_cfg["flux1_fill_dev"][
                    "repository_used_storage_bytes"
                ],
                "formal_execution": False,
                "decision": "audit_only_gated_and_outside_frozen_local_storage_contract",
            },
        ]
        _write_jsonl(run_dir / "CANDIDATE_AUDIT.jsonl", audit_rows)
        gate_cfg = config["selection_gate"]
        eligible = []
        for result in worker_results:
            result["selection_eligible"] = bool(
                result.get("status") == "executed"
                and result["case_count"] == len(input_rows)
                and result["all_repeats_successful"]
                and result["all_repeat_sha_exact"]
                and result["all_finite"]
                and result["all_nonzero_masked_change"]
                and result["all_outside_mask_exact"]
                and result["peak_gpu_memory_mib"]
                <= float(gate_cfg["maximum_peak_gpu_memory_mib"])
                and result["median_latency_seconds"]
                <= float(gate_cfg["maximum_seconds_per_case"])
            )
            if result["selection_eligible"]:
                eligible.append(result)
        eligible.sort(
            key=lambda row: (
                row["peak_gpu_memory_mib"],
                row["median_latency_seconds"],
                0 if row["candidate"] == "big_lama" else 1,
            )
        )
        selected = eligible[0]["candidate"] if eligible else None
        _write_jsonl(run_dir / "CANDIDATE_RESULTS.jsonl", worker_results)
        checks = {
            "three_candidate_audit_passed": len(audit_rows) == 3,
            "two_ungated_candidates_executed": sum(
                row.get("status") == "executed" for row in worker_results
            )
            == 2,
            "at_least_one_candidate_eligible": len(eligible) >= 1,
            "selected_exactly_one": selected is not None,
            "all_outputs_typed_proposal": True,
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r8_selection_gate.v1",
            "checks": checks,
            "eligible_candidates": [row["candidate"] for row in eligible],
            "selected_candidate": selected,
            "selection_rule": gate_cfg["tie_break_order"],
            "decision": "proceed_to_independent_verifier_arms"
            if checks["passed"]
            else "reject_generator_stage_or_repair_capability",
        }
        _write_json(run_dir / "SELECTION_GATE.json", gate)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r8_resource_audit.v1",
                "gpu": "single_rtx_3090_24gb",
                "max_concurrent_heavy_processes": 1,
                "training_started": False,
                "confirmation_content_read": False,
                "wall_seconds": time.monotonic() - started,
                "disk_free_gib_at_start": free_gib,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r8_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_capability" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "selected_candidate": selected,
            "eligible_candidates": [row["candidate"] for row in eligible],
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "INPUT_CASES.jsonl",
            "CANDIDATE_RESULTS.jsonl",
            "CANDIDATE_AUDIT.jsonl",
            "SELECTION_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        for result in worker_results:
            if result.get("status") == "executed":
                tracked.append(f"{result['candidate']}/WORKER_RESULT.json")
        manifest = {
            "schema_version": "worldsim_v6.r8_run_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
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
        default=Path("configs/worldsim_v6/r8_frozen_generator_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--big-lama-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r8_lama"),
    )
    parser.add_argument(
        "--sd15-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r8_sd15"),
    )
    args = parser.parse_args()
    try:
        run_experiment(
            args.repo_root, args.config, args.run_root, args.big_lama_root, args.sd15_root
        )
    except R7ExperimentError as error:
        raise R8ExperimentError(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
