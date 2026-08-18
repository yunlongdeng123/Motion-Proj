#!/usr/bin/env python3
"""Independently replay and audit the r046 Trace3D capability probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.audit_worldsim_v51_f0b_three_view_association_parity import _load_json, _load_jsonl, _load_yaml
from scripts.audit_worldsim_v51_f0c_upstream_batch_association_repeatability import _manifest_inventory
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"
EXPECTED_RUN_NAME = "20260818T230000Z__m1-stage-g-g0a-trace3d-capability-s20260814-r046"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _projection_matrix(torch: Any, image: dict[str, Any], device: str) -> Any:
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = 1.0 / float(image["tan_fovx"])
    matrix[1, 1] = 1.0 / float(image["tan_fovy"])
    matrix[3, 2] = 1.0
    matrix[2, 2] = float(image["zfar"]) / (float(image["zfar"]) - float(image["znear"]))
    matrix[2, 3] = -(float(image["zfar"]) * float(image["znear"])) / (float(image["zfar"]) - float(image["znear"]))
    return matrix.transpose(0, 1).contiguous()


def _independent_probe(config: dict[str, Any], package_root: Path) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package_root))
    torch = importlib.import_module("torch")
    raster = importlib.import_module("diff_id_rasterization")
    module_path = Path(raster.__file__).resolve()
    try:
        module_path.relative_to(package_root.resolve())
    except ValueError as exc:
        raise ProtocolError("r046 audit imported non-frozen package") from exc
    spec = config["synthetic_probe"]
    image = spec["image"]
    device = spec["device"]
    height, width = int(image["height"]), int(image["width"])
    view = torch.eye(4, dtype=torch.float32, device=device)
    projection = _projection_matrix(torch, image, device)
    operator = raster.GaussianRasterizer(raster_settings=raster.GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=float(image["tan_fovx"]), tanfovy=float(image["tan_fovy"]),
        bg=torch.zeros(3, dtype=torch.float32, device=device), scale_modifier=1.0, viewmatrix=view, projmatrix=projection,
        sh_degree=0, campos=torch.zeros(3, dtype=torch.float32, device=device), prefiltered=False, debug=False, include_feature=False,
    ))
    means3d = torch.tensor(spec["gaussian"]["means3d"], dtype=torch.float32, device=device)
    means2d = torch.zeros_like(means3d)
    opacity = torch.tensor(spec["gaussian"]["opacity"], dtype=torch.float32, device=device)
    scales = torch.tensor(spec["gaussian"]["scales_2d"], dtype=torch.float32, device=device)
    rotations = torch.tensor(spec["gaussian"]["rotations_wxyz"], dtype=torch.float32, device=device)
    inputs = [means3d, means2d, opacity, scales, rotations, view, projection]
    frozen_inputs = [tensor.clone() for tensor in inputs]

    def traced(label: int, alpha: bool) -> Any:
        weights = torch.zeros((1, 2), dtype=torch.float32, device=device)
        mask = torch.full((height, width), label, dtype=torch.int32, device=device)
        operator.trace(means3D=means3d, means2D=means2d, opacities=opacity, weights=weights, id_masks=mask,
                       num_class=1, alpha_w=alpha, scales=scales, rotations=rotations)
        torch.cuda.synchronize(0)
        return weights

    background = traced(0, False)
    foreground = traced(1, False)
    repeat = traced(1, False)
    alpha = traced(1, True)
    checks = {
        "background_exact": background.cpu().tolist() == [[1.0, 0.0]],
        "foreground_exact": foreground.cpu().tolist() == [[0.0, 1.0]],
        "repeat_bitwise": bool(torch.equal(foreground, repeat)),
        "alpha_finite_positive_bounded": bool(torch.isfinite(alpha).all().item() and 0 < alpha[0, 1].item() <= foreground[0, 1].item()),
        "inputs_immutable": all(torch.equal(before, after) for before, after in zip(frozen_inputs, inputs)),
    }
    if not all(checks.values()):
        raise ProtocolError(f"r046 independent probe failed: {checks}")
    return {"checks": checks, "background": background.cpu().tolist(), "foreground": foreground.cpu().tolist(), "alpha": alpha.cpu().tolist()}


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    if run_dir.name != EXPECTED_RUN_NAME:
        raise ProtocolError("r046 run identity drift")
    config = _load_yaml(config_path)
    conclusion = config["decision"]["pass_conclusion"]
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if status.get("task_id") != TASK_ID or status.get("status") != "done" or status.get("conclusion") != conclusion:
        raise ProtocolError("r046 terminal drift")
    source_commit = status["source_commit"]
    source_tree = _git(PROJECT, "show", "-s", "--format=%T", source_commit)
    resolved_path = run_dir / "resolved_config.yaml"
    committed = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"])
    if resolved_path.read_bytes() != committed:
        raise ProtocolError("r046 resolved config drift")
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r046 event drift")

    summary_path = run_dir / "summary.json"
    build_path = run_dir / "artifacts/build_report.json"
    probe_path = run_dir / "artifacts/probe.json"
    resources_path = run_dir / "artifacts/resources.json"
    summary, build, probe, resources = (_load_json(path) for path in (summary_path, build_path, probe_path, resources_path))
    if (
        summary.get("status") != "done" or summary.get("conclusion") != conclusion
        or summary.get("source_commit") != source_commit or summary.get("source_tree") != source_tree
        or summary.get("build") != build or summary.get("probe") != probe or summary.get("resources") != resources
        or not all(summary.get("resource_checks", {}).values()) or not all(probe.get("checks", {}).values())
    ):
        raise ProtocolError("r046 summary/build/probe/resource drift")
    false_fields = (
        "network_access", "official_source_mutation", "submodules_initialized", "model_download", "real_checkpoint_read",
        "camera_metadata_read", "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation",
        "real_worldsim_adapter_supported", "quality_supported",
    )
    if any(summary.get(field) is not False for field in false_fields):
        raise ProtocolError("r046 no-data/no-quality declaration drift")

    source = config["official_source"]
    repo = Path(source["repository"])
    ls_tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", source["commit"], source["subdirectory"]])
    if (
        _git(repo, "rev-parse", "HEAD") != source["commit"] or _git(repo, "rev-parse", "HEAD^{tree}") != source["tree"]
        or _git(repo, "status", "--porcelain") or len(ls_tree.splitlines()) != int(source["tracked_entry_count"])
        or hashlib.sha256(ls_tree).hexdigest() != source["git_ls_tree_sha256"]
    ):
        raise ProtocolError("r046 official source audit drift")

    publish_target = Path(config["build"]["publish_target"])
    inventory = _tree_inventory(publish_target)
    if build.get("publish_target") != str(publish_target) or build.get("publish_inventory") != inventory or build.get("source_patch_applied") is not False:
        raise ProtocolError("r046 published build inventory drift")
    wheel = run_dir / build["wheel"]["path"]
    build_log = run_dir / build["build_log"]["path"]
    for path, evidence in ((wheel, build["wheel"]), (build_log, build["build_log"])):
        if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != evidence["sha256"]:
            raise ProtocolError(f"r046 build artifact drift: {path}")
    extension = publish_target / probe["extension"]["path"]
    if extension.stat().st_size != int(probe["extension"]["bytes"]) or sha256_file(extension) != probe["extension"]["sha256"]:
        raise ProtocolError("r046 extension identity drift")
    official_init = repo / source["subdirectory"] / "diff_id_rasterization/__init__.py"
    installed_init = publish_target / "diff_id_rasterization/__init__.py"
    if installed_init.read_bytes() != official_init.read_bytes():
        raise ProtocolError("r046 installed Python wrapper differs from official source")
    replay = _independent_probe(config, publish_target)

    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    run_inventory = _manifest_inventory(run_dir)
    if manifest.get("task_id") != TASK_ID or manifest.get("status") != "done" or manifest.get("inventory") != run_inventory:
        raise ProtocolError("r046 manifest drift")
    return {
        "schema_version": "worldsim_v51_stage_g_g0a_r046_audit_v1", "task_id": TASK_ID, "status": "pass",
        "conclusion": conclusion, "run_dir": str(run_dir), "source_commit": source_commit, "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "build_report": {"bytes": build_path.stat().st_size, "sha256": sha256_file(build_path)},
        "probe": {"bytes": probe_path.stat().st_size, "sha256": sha256_file(probe_path)},
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "entry_count": len(run_inventory), "logical_bytes": sum(int(row["bytes"]) for row in run_inventory)},
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "official_source": {"commit": source["commit"], "tree": source["tree"], "tracked_entry_count": len(ls_tree.splitlines()), "clean": True, "source_patch_applied": False},
        "wheel": build["wheel"], "extension": probe["extension"], "published_inventory_entries": len(inventory),
        "independent_probe": replay, "resources": resources, "resource_checks": summary["resource_checks"],
        "real_checkpoint_read": False, "camera_metadata_read": False, "image_pixels_read": False, "mask_pixels_read": False,
        "quality_metrics_read": False, "training": False, "gaussian_mutation": False,
        "failure_ledger_delta": "none", "next_action": config["decision"]["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0a_trace3d_reverse_tracing_capability_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.config.resolve(), args.run_dir.resolve())
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
