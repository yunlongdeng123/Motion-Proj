"""把 V7/V7.1 的真实 Gaussian 渲染正结果接回 V7.2 EAS 所有权接口。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _montage(run_root: Path, keys: list[str], output: Path) -> None:
    columns = ("target", "original", "initial", "final")
    examples = {
        (key, column): Image.open(run_root / f"{key}_{column}.png").convert("RGB")
        for key in keys
        for column in columns
    }
    width, height = examples[(keys[0], columns[0])].size
    header = 34
    canvas = Image.new("RGB", (width * len(columns), (height + header) * len(keys)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, key in enumerate(keys):
        y = row * (height + header)
        for column, name in enumerate(columns):
            x = column * width
            canvas.paste(examples[(key, name)], (x, y + header))
            draw.text((x + 8, y + 9), f"{key} | {name}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "phase": "audit"})
    started = time.monotonic()
    m23_root = Path(config["inheritance"]["m23_run"])
    m27_root = Path(config["inheritance"]["m27_run"])
    e3_root = Path(config["current"]["anchored_appearance_run"])
    e4_root = Path(config["current"]["se3_run"])
    m23 = _read_json(m23_root / "summary.json")
    m27 = _read_json(m27_root / "summary.json")
    current = _read_json(e3_root / "summary.json")
    e4 = _read_json(e4_root / "summary.json")
    actor_rows = [
        json.loads(line)
        for line in (m23_root / "ACTOR_APPEARANCE_ROWS.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    actor_token = str(m27["actor_token"])
    association = [row for row in actor_rows if str(row["actor_token"]) == actor_token]
    if len(association) != 1:
        raise RuntimeError(f"M27 Actor 在 M23 ownership 中匹配数为 {len(association)}")
    heldout = [row for row in m27["rows"] if row["split"] == "heldout"]
    keys = [f"f{int(row['frame']):03d}_c{int(row['camera'])}" for row in heldout]
    sizes = set()
    hashes = {}
    inside_change_sum = outside_change_sum = 0.0
    inside_count = outside_count = 0
    artifact_rows = []
    for key, row in zip(keys, heldout):
        paths = {
            name: m27_root / f"{key}_{name}.png"
            for name in ("target", "original", "initial", "final", "footprint")
        }
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"incomplete render artifacts for {key}")
        images = {
            name: np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            for name, path in paths.items()
            if name != "footprint"
        }
        mask_image = Image.open(paths["footprint"]).convert("L")
        mask = np.asarray(mask_image) > 0
        sizes.add(Image.open(paths["target"]).size)
        change = np.abs(images["final"] - images["initial"]).mean(axis=2)
        inside_change_sum += float(change[mask].sum())
        outside_change_sum += float(change[~mask].sum())
        inside_count += int(mask.sum())
        outside_count += int((~mask).sum())
        artifact_rows.append(
            {
                "view": key,
                "footprint_pixels": int(mask.sum()),
                "actor_psnr_delta_db": float(row["final_minus_initial_actor_psnr_db"]),
                "full_frame_psnr_db": float(row["final_full_psnr_db"]),
                "inside_appearance_change_mae": float(change[mask].mean()),
                "outside_appearance_change_mae": float(change[~mask].mean()),
            }
        )
        hashes.update({f"{key}_{name}": _sha256(path) for name, path in paths.items()})
    inside_change = inside_change_sum / inside_count
    outside_change = outside_change_sum / outside_count
    spill_ratio = outside_change / max(inside_change, 1.0e-12)
    montage_path = run_dir / "E3_RENDER_BRIDGE.png"
    _montage(m27_root, keys, montage_path)
    deltas = np.asarray([row["actor_psnr_delta_db"] for row in artifact_rows])
    decisions = {
        "full_frame_render_artifacts_verified": sizes == {tuple(config["decision"]["required_frame_size_wh"])},
        "physical_parent_identity_matches_render_actor": True,
        "all_physical_parents_have_appearance_attributes": int(association[0]["assigned_attribute_count"])
        == int(association[0]["physical_gaussian_count"]),
        "all_heldout_views_improve_actor_psnr": bool(np.all(deltas > 0.0)),
        "appearance_change_is_footprint_concentrated": spill_ratio
        <= float(config["decision"]["maximum_outside_to_inside_change_ratio"]),
        "current_v72_rgb_does_not_update_physical_state": int(current["physical_parameter_update_count"]) == 0,
        "shared_se3_state_is_numerically_equivariant": float(e4["metrics"]["maximum_se3_commutation_error_m"])
        <= float(config["decision"]["maximum_se3_error_m"]),
    }
    summary = {
        "schema_version": "worldsim_v72.e3_render_bridge.v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "run_uri": f"run://worldsim_v72/{config['task_id']}/{run_id}",
        "status": "done",
        "verdict": "inherited_rendering_and_eas_ownership_bridge_supported"
        if all(decisions.values())
        else "inherited_rendering_and_eas_ownership_bridge_not_supported",
        "scene_name": m27["scene_name"],
        "actor_token": actor_token,
        "ownership": {
            "physical_parent_count": int(association[0]["physical_gaussian_count"]),
            "assigned_parent_count": int(association[0]["assigned_attribute_count"]),
            "hierarchical_visual_primitive_count": int(m27["render_carrier_gaussian_count"]),
            "physical_geometry_trainable_from_rgb": False,
            "trajectory_trainable_from_rgb": False,
        },
        "rendering": {
            "heldout_view_count": len(artifact_rows),
            "frame_size_wh": list(next(iter(sizes))),
            "pooled_actor_psnr_initial_db": float(m27["aggregate"]["heldout_initial_actor_psnr_db"]),
            "pooled_actor_psnr_final_db": float(m27["aggregate"]["heldout_final_actor_psnr_db"]),
            "pooled_actor_psnr_delta_db": float(m27["aggregate"]["heldout_final_actor_psnr_db"] - m27["aggregate"]["heldout_initial_actor_psnr_db"]),
            "streetgs_reference_psnr_db": float(m27["aggregate"]["heldout_original_actor_psnr_db"]),
            "remaining_reference_gap_db": float(m27["aggregate"]["heldout_original_minus_final_gap_db"]),
            "inside_appearance_change_mae": inside_change,
            "outside_appearance_change_mae": outside_change,
            "outside_to_inside_change_ratio": spill_ratio,
            "views": artifact_rows,
            "montage": str(montage_path),
            "artifact_sha256": hashes,
        },
        "current_v72_boundary": {
            "candidate_rgb_claim": current["interpretation"],
            "vggt_development_rgb_psnr": current["results"]["vggt"]["development"]["rgb_psnr"],
            "pi3x_development_rgb_psnr": current["results"]["pi3x"]["development"]["rgb_psnr"],
            "visibility_gain_claimed": False,
            "inherited_render_views_previously_exposed": True,
            "full_render_gain_is_inherited_development_evidence": True,
        },
        "se3": {
            "max_commutation_error_m": e4["metrics"]["maximum_se3_commutation_error_m"],
            "max_actor_frame_recovery_error_m": e4["metrics"]["maximum_actor_frame_recovery_error_m"],
            "appearance_to_surface_error": e4["metrics"]["maximum_appearance_to_surface_error_m"],
        },
        "decisions": decisions,
        "source_test_read": False,
        "external_test_read": False,
        "resources": {
            "device": "cpu",
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "wall_seconds": time.monotonic() - started,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"status": "done", "summary": "summary.json", "source_test_read": False, "external_test_read": False})
    _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

