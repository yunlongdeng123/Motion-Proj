#!/usr/bin/env python
"""Validate and aggregate the three-scene WorldSim V3 A0 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


EXPECTED_SCENES = ("scene-0230", "scene-0242", "scene-0255")
EXPECTED_BOUNDARY_STATUS = {
    "scene-0230": "done",
    "scene-0242": "ABSTAIN",
    "scene-0255": "done",
}


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_terminal_done(run_dir: Path) -> dict[str, object]:
    path = run_dir / "terminal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "done":
        raise RuntimeError(f"run is not terminal done: {run_dir}")
    return payload


def parse_scene_record(value: str) -> tuple[str, Path, Path]:
    try:
        scene, paths = value.split("=", 1)
        source, actor = paths.split(",", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "scene record must be SCENE=SOURCE_RUN_DIR,ACTOR_RUN_DIR"
        ) from error
    return scene, Path(source), Path(actor)


def metric(region: dict[str, object], name: str):
    return region.get(name) if region.get("status") == "done" else None


def validate_scene_record(
    scene: str, source_run_dir: Path, actor_run_dir: Path
) -> dict[str, object]:
    require_terminal_done(source_run_dir)
    require_terminal_done(actor_run_dir)
    source = json.loads((source_run_dir / "summary.json").read_text(encoding="utf-8"))
    actor_wrapper = json.loads(
        (actor_run_dir / "summary.json").read_text(encoding="utf-8")
    )
    actor = actor_wrapper["actor_metrics"]
    if source.get("status") != "done" or actor_wrapper.get("status") != "done":
        raise RuntimeError(f"non-done summary for {scene}")
    if source.get("scene_name") != scene or actor.get("scene_name") != scene:
        raise RuntimeError(f"scene binding mismatch for {scene}")
    if actor.get("checkpoint_sha256_before") != source["checkpoint"]["sha256"]:
        raise RuntimeError(f"actor diagnostic checkpoint mismatch for {scene}")
    if actor.get("checkpoint_sha256_after") != source["checkpoint"]["sha256"]:
        raise RuntimeError(f"checkpoint changed during actor diagnostic for {scene}")
    split = actor["heldout_split"]
    if split.get("formal_full_split") is not True or split.get("max_images_per_role") is not None:
        raise RuntimeError(f"actor diagnostic is not a formal full-split run for {scene}")
    roles = actor["roles"]
    if roles["high-support"].get("status") != "done":
        raise RuntimeError(f"high-support actor diagnostic is missing for {scene}")
    expected_boundary = EXPECTED_BOUNDARY_STATUS[scene]
    if roles["boundary-support"].get("status") != expected_boundary:
        raise RuntimeError(
            f"boundary status mismatch for {scene}: "
            f"expected {expected_boundary}, got {roles['boundary-support'].get('status')}"
        )

    heldout = source["heldout_metrics"]
    checkpoint = source["checkpoint"]
    high = roles["high-support"]
    high_region = high["actor_region"]
    high_boundary = high["boundary_band"]
    boundary = roles["boundary-support"]
    boundary_region = boundary.get("actor_region", {})
    boundary_band = boundary.get("boundary_band", {})
    total_gaussians = int(checkpoint["background_gaussians"]) + int(
        checkpoint["rigid_gaussians"]
    )
    return {
        "scene": scene,
        "scene_index": source["scene_index"],
        "source_run_dir": str(source_run_dir),
        "actor_run_dir": str(actor_run_dir),
        "checkpoint_sha256": checkpoint["sha256"],
        "background_gaussians": checkpoint["background_gaussians"],
        "rigid_gaussians": checkpoint["rigid_gaussians"],
        "total_gaussians": total_gaussians,
        "global_psnr": heldout["image_metrics/test/psnr"],
        "global_ssim": heldout["image_metrics/test/ssim"],
        "global_lpips": heldout["image_metrics/test/lpips"],
        "dynamic_masked_psnr": heldout["image_metrics/test/masked_psnr"],
        "dynamic_masked_ssim": heldout["image_metrics/test/masked_ssim"],
        "vehicle_psnr": heldout["image_metrics/test/vehicle_psnr"],
        "vehicle_ssim": heldout["image_metrics/test/vehicle_ssim"],
        "high_actor_token": high["actor"]["instance_token"],
        "high_actor_gaussians": high["actor"]["gaussian_count"],
        "high_visible_images": high["visible_effect_image_count"],
        "high_effect_pixel_coverage": high["effect_pixel_coverage"],
        "high_actor_psnr": high_region["psnr"],
        "high_actor_ssim": high_region["ssim"],
        "high_actor_lpips_tight_crop": high_region[
            "masked_lpips_alex_tight_crop_256px"
        ],
        "high_boundary_psnr": high_boundary["psnr"],
        "high_boundary_ssim": high_boundary["ssim"],
        "high_boundary_lpips_tight_crop": high_boundary[
            "masked_lpips_alex_tight_crop_256px"
        ],
        "boundary_status": boundary["status"],
        "boundary_actor_token": (
            boundary.get("actor") or {}
        ).get("instance_token"),
        "boundary_actor_gaussians": (
            boundary.get("actor") or {}
        ).get("gaussian_count"),
        "boundary_visible_images": boundary.get("visible_effect_image_count"),
        "boundary_effect_pixel_coverage": boundary.get("effect_pixel_coverage"),
        "boundary_actor_psnr": metric(boundary_region, "psnr"),
        "boundary_actor_ssim": metric(boundary_region, "ssim"),
        "boundary_actor_lpips_tight_crop": metric(
            boundary_region, "masked_lpips_alex_tight_crop_256px"
        ),
        "boundary_band_psnr": metric(boundary_band, "psnr"),
        "boundary_band_ssim": metric(boundary_band, "ssim"),
        "boundary_band_lpips_tight_crop": metric(
            boundary_band, "masked_lpips_alex_tight_crop_256px"
        ),
        "train_seconds": source["train_resources"]["duration_seconds"],
        "train_peak_gpu_mib": source["train_resources"][
            "peak_gpu_memory_mib_sampled"
        ],
        "eval_seconds": source["eval_resources"]["duration_seconds"],
        "eval_peak_gpu_mib": source["eval_resources"][
            "peak_gpu_memory_mib_sampled"
        ],
        "actor_eval_seconds": actor_wrapper["resources"]["duration_seconds"],
        "actor_eval_peak_gpu_mib": actor_wrapper["resources"][
            "peak_gpu_memory_mib_sampled"
        ],
    }


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "# WorldSim V3 A0 three-scene baseline",
        "",
        "All actor masks are model-counterfactual diagnostics, not ground-truth segmentation.",
        "",
        "| scene | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | high boundary PSNR / SSIM | boundary actor / band PSNR | bg / rigid GS | train s / peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        boundary_value = (
            f"{row['boundary_actor_psnr']:.2f} / {row['boundary_band_psnr']:.2f}"
            if row["boundary_status"] == "done"
            else "ABSTAIN"
        )
        lines.append(
            "| {scene} | {global_psnr:.2f} / {global_ssim:.3f} / {global_lpips:.3f} "
            "| {high_actor_psnr:.2f} / {high_actor_ssim:.3f} / {high_actor_lpips_tight_crop:.3f} "
            "| {high_boundary_psnr:.2f} / {high_boundary_ssim:.3f} "
            "| {boundary_value} | {background_gaussians:,} / {rigid_gaussians:,} "
            "| {train_seconds:.1f} / {train_peak_gpu_mib:,} |".format(
                boundary_value=boundary_value, **row
            )
        )
    lines.extend(
        [
            "",
            "Primary A0 interpretation:",
            "",
            "- Global reconstruction quality does not predict selected dynamic-actor quality across scenes.",
            "- Actor boundary quality is a measurable weakness in scene-0230 and the scene-0255 boundary actor has especially low region SSIM.",
            "- Cross-scene Gaussian counts are descriptive only; causal conclusions require within-scene A1/A2 ablations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--scene-record", action="append", type=parse_scene_record, required=True
    )
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    records = {scene: (source, actor) for scene, source, actor in args.scene_record}
    if tuple(sorted(records)) != EXPECTED_SCENES:
        raise RuntimeError(
            f"expected exactly {EXPECTED_SCENES}, got {tuple(sorted(records))}"
        )

    for name in ("artifacts", "source_snapshot"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "failure": None},
    )
    source_file = args.project_root / "scripts/finalize_worldsim_v3_a0.py"
    shutil.copy2(source_file, args.run_dir / "source_snapshot" / source_file.name)
    manifest = {
        "schema_version": 1,
        "task_id": "WS-V3-A0-NATIVE-BASELINE-01",
        "component": "three-scene A0 finalizer",
        "project_commit": command_output(
            "git", "rev-parse", "HEAD", cwd=args.project_root
        ),
        "project_status": command_output(
            "git", "status", "--short", cwd=args.project_root
        ).splitlines(),
        "records": {
            scene: {"source_run_dir": str(source), "actor_run_dir": str(actor)}
            for scene, (source, actor) in records.items()
        },
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    try:
        rows = [
            validate_scene_record(scene, *records[scene]) for scene in EXPECTED_SCENES
        ]
        matrix_json = args.run_dir / "artifacts" / "a0_matrix.json"
        atomic_json(matrix_json, {"status": "done", "rows": rows})
        matrix_csv = args.run_dir / "artifacts" / "a0_matrix.csv"
        with matrix_csv.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = args.run_dir / "artifacts" / "a0_report.md"
        report.write_text(markdown_table(rows), encoding="utf-8")
        summary = {
            "status": "done",
            "scene_count": len(rows),
            "scenes": list(EXPECTED_SCENES),
            "matrix_json": str(matrix_json),
            "matrix_csv": str(matrix_csv),
            "report": str(report),
            "rows": rows,
        }
        atomic_json(args.run_dir / "summary.json", summary)
        atomic_json(
            args.run_dir / "terminal.json",
            {"status": "done", "failure": None},
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as error:
        atomic_json(
            args.run_dir / "terminal.json",
            {
                "status": "blocked",
                "failure": {
                    "code": "A0_FINALIZATION_FAILED",
                    "detail": f"{type(error).__name__}: {error}",
                },
            },
        )
        raise


if __name__ == "__main__":
    main()
