"""Prepare inputs on CPU; explicitly enabled CUDA renders unknown Gaussian colors.

The only counterfactual is white SH color for same-frame R2 unknown row IDs.
Geometry, opacity, temporal weights, masks, row order and native environment
are retained. This is neither Gaussian deletion nor R2's local white background.

CPU preparation (the default) never imports torch or opens SSH:
  python render_unknown_white.py --scene-dir <scene1> --output render_inputs
Later, after GPU execution has been authorized separately:
  python render_unknown_white.py --enable-gpu --input-contract <render_inputs/input_contract.json> --output <fresh-output>
The existing remote resource controller must also permit execution. No job is
enqueued and no resource-control authorization is edited by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
import time
from pathlib import Path


CONTEXT = "14705303724557273004_3105_000_3125_000"
SCHEMA = "v43-native-unknown-white-v1"
INPUT_SCHEMA = "v43-native-unknown-white-input-v1"
RAWS = list(range(77, 126))
MOTION_RAWS = list(range(86, 117))
STATIC_ROWS, ACTOR_ROWS = 918850, 6241
C0 = 0.28209479177387814
GROUP_CASES = {
    "v43_actor_10000": "reconstruction",
    "v43_motion_no_edit": "no_edit",
    "v43_motion_delete_target": "delete_target",
    "v43_motion_translate_lateral_1m": "translate_lateral_1m",
    "v43_motion_turn_left15": "turn_left15",
    "v43_shift_3m": "translate_lateral_3m",
}
EVIDENCE_NEEDED = {
    "v43_actor_1000": "Actor1K has stable seed row identities but no checkpoint-matched R2 actor alpha*T attribution.",
    "v43_actor_5000": "Actor5K has stable seed row identities but no checkpoint-matched R2 actor alpha*T attribution.",
    "v43_main_noedit": "V1 native original dynamic namespace has 1011350 rows; Actor10K labels do not apply.",
}
DEFAULT_CODE = Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/code")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assignment(path):
    source = Path(path).read_text(encoding="utf-8-sig")
    return json.JSONDecoder().raw_decode(source[source.index("=") + 1:])[0]


def relative_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Path escapes its bundle: " + str(relative))
    return path


def require_render_authorization(context=CONTEXT, campaign=DEFAULT_CODE.parent):
    """Read-only rendering scope; never unlock training or any old queue."""
    campaign = Path(campaign)
    require(context == CONTEXT, "This renderer is authorized for scene1 only")
    require(not (campaign / "USER_STOP").exists(), "User stop requested")
    auth = read(campaign / "authorization.json")
    require(auth.get("allow_unknown_white_render") is True, "Unknown-white rendering is not authorized")
    require(auth.get("unknown_white_render_contexts") == [CONTEXT], "Unknown-white context scope differs")
    require(auth.get("gpu_execution_hold") is False, "GPU execution is held")
    require(auth.get("quota_stop_enabled") is False, "Quota stop is enabled")
    require(auth.get("allow_v43_training") is False, "Training must remain disabled during this read-only render")
    require(auth.get("old_queue_resumed") is False, "Old queues must remain held")
    require(shutil.disk_usage(str(campaign)).free > 25 * 1024 ** 3, "Preserve 25 GiB space reserve and recovery states")


def classification(radius=5, fraction=0.2):
    require(radius in (0, 3, 5, 8), "Unsupported boundary radius")
    require(math.isfinite(fraction) and 0 <= fraction <= 1, "Invalid boundary fraction")
    return {
        "radius": radius, "fraction": fraction,
        "actor_rule": "unknown = NOT(present AND 14 <= class_id <= 19 AND NOT(boundary_fraction[radius] >= fraction)); radius 0 disables boundary only",
        "static_rule": "unknown = exact same-frame R2 final_removed_ids; all other static rows keep native color",
        "source": "frozen no-edit Actor10K alpha*T semantics/contour plus inherited R2 same-frame static IDs",
        "edited_visibility": "source-view labels carried with the same rows; native renderer recomputes edited-view contributions",
        "known_actor_color": "native RGB/SH, never semantic green",
        "white_operation": "SH DC=(1-0.5)/C0; higher SH coefficients=0 on unknown rows only",
        "environment": "native environment unchanged; no local scope clipping",
        "labels_for_training": False,
    }


def actor_partition(frame, radii, radius=5, fraction=0.2):
    """Exactly r2_ownership_runtime.js actorPartition, using JS-like doubles.

    In particular do not cast fraction to float16 or reapply purity/mass gates
    to rounded fields. The saved class_id already contains the original gates.
    """
    data = frame["actor"]
    require(len(data["present"]) == len(data["class_id"]) == ACTOR_ROWS, "Actor namespace mismatch")
    require(len(data["boundary_fraction"]) == ACTOR_ROWS * len(radii), "Boundary layout mismatch")
    index = radii.index(radius) if radius in radii else -1
    unknown = []
    counts = dict(green=0, unknown=0, boundary=0, base=0, missing=0)
    for row, base in enumerate(data["class_id"]):
        missing = not data["present"][row]
        vehicle = 14 <= base <= 19
        band = index >= 0 and float(data["boundary_fraction"][row * len(radii) + index]) >= fraction
        green = not missing and vehicle and not band
        if not green:
            unknown.append(row)
        counts["green" if green else "unknown"] += 1
        counts["boundary"] += int(band)
        counts["base"] += int(not missing and not vehicle)
        counts["missing"] += int(missing)
    return unknown, counts


def validate_evidence(evidence, inherited):
    require(evidence["schema"] == "r2-intact-ownership-evidence-v1", "Wrong ownership schema")
    require(evidence["context"] == inherited["context"] == CONTEXT, "Context mismatch")
    require(evidence["raws"] == inherited["raws"] == RAWS, "Require exact scene1 all49 coverage")
    require(inherited["status"] == "complete" and inherited["validated"], "R2 has not passed its audit")
    require(evidence["branch_sizes"] == dict(static=STATIC_ROWS, actor=ACTOR_ROWS), "Row namespace mismatch")
    require(evidence["radii"] == [3, 5, 8], "Unexpected boundary columns")
    for raw in RAWS:
        frame = evidence["frames"][str(raw)]
        static = frame["static"]
        for key in ("existing_removed_ids", "new_unknown_ids", "final_removed_ids"):
            ids = static[key]
            require(ids == sorted(set(ids)) and all(0 <= i < STATIC_ROWS for i in ids), f"raw{raw}: invalid {key}")
        require(static["final_removed_ids"] == sorted(set(static["existing_removed_ids"]) | set(static["new_unknown_ids"])), f"raw{raw}: R2 union differs")
        require(len(static["final_removed_ids"]) == inherited["frames"][str(raw)]["removed_rows"], f"raw{raw}: R2 count differs")
        unknown, counts = actor_partition(frame, evidence["radii"])
        require(counts["green"] == frame["counts"]["actor_core"], f"raw{raw}: default role count differs")
        if "default_role" in frame["actor"]:
            require(unknown == [i for i, role in enumerate(frame["actor"]["default_role"]) if not role], f"raw{raw}: runtime and saved roles differ")


def prepare(scene, output, radius=5, fraction=0.2):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    require(not output.exists(), "No-clobber: choose a fresh preparation directory")
    evidence_path = scene / "r2_ownership_data.js"
    inherited_path = scene / "r2_inherited_manifest.json"
    evidence, inherited = assignment(evidence_path), read(inherited_path)
    validate_evidence(evidence, inherited)
    groups = {g["id"]: g for g in read(scene / "v43_groups_data.json")["groups"]}
    poses = read(scene / "assets/v43/10000/poses_world.json")
    actor_metric = read(scene / "evidence/v43_actor_10000_metrics.json")
    frame_receipts = {raw: read(scene / inherited["frames"][str(raw)]["receipt"]) for raw in RAWS}
    first = frame_receipts[RAWS[0]]
    for raw, rec in frame_receipts.items():
        require(rec["raw"] == raw and rec["context"] == CONTEXT and rec["camera_index"] == raw - 77, "Frame identity mismatch")
        require(rec["checkpoint_ref"] == first["checkpoint_ref"] and rec["row_namespace"] == evidence["static_namespace"], "Evidence checkpoint mismatch")
        require(rec["static_checkpoint_ref"] == first["static_checkpoint_ref"], "Static checkpoint mismatch")
    output.mkdir(parents=True)
    source_assets = {}
    for name, source in [("ownership", evidence_path), ("inherited_manifest", inherited_path),
                         ("actor_checkpoint_receipt", scene / "evidence/v43_actor_10000_metrics.json")]:
        relative = "source_assets/" + source.name
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        digest = sha(source)
        require(digest == sha(target), "Input copy failed: " + name)
        source_assets[name] = dict(path=relative, sha256=digest)
    manifest = dict(schema=SCHEMA, context=CONTEXT, status="prepared", gpu_executed=False,
                    training_updates=0, classification=classification(radius, fraction), groups={},
                    required_group_frame_count=204, verified_group_frame_count=0)
    contract = dict(schema=INPUT_SCHEMA, context=CONTEXT, status="prepared", gpu_executed=False,
                    classification=manifest["classification"], source_assets=source_assets,
                    actor_checkpoint_ref=first["checkpoint_ref"], actor_checkpoint_sha256=actor_metric["checkpoint_sha256"],
                    static_checkpoint_ref=first["static_checkpoint_ref"], static_namespace=evidence["static_namespace"],
                    actor_namespace=evidence["actor_namespace"], branch_sizes=evidence["branch_sizes"], groups={})
    for gid, case in GROUP_CASES.items():
        expected_raws = RAWS if case == "reconstruction" else MOTION_RAWS
        group = groups[gid]
        require(sorted(map(int, group["frames"])) == expected_raws, "Unexpected source frames: " + gid)
        metric = read(scene / group["summary"])
        diagnostics = metric.get("diagnostics", metric.get("native_summary", {}))
        requests = {r["raw"]: r for r in diagnostics.get("records", []) if r["case"] == case}
        frames, pending = {}, {}
        for raw in expected_raws:
            rec = frame_receipts[raw]
            rgb_path = group["frames"][str(raw)]["channels"]["rgb"]
            source = relative_file(scene, rgb_path)
            relative = f"baselines/{gid}/{raw:03d}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha(source)
            require(digest == sha(target), "Baseline copy failed")
            requested = requests.get(raw)
            require(case == "reconstruction" or requested is not None, f"Missing physical request: {gid}/{raw}")
            frames[str(raw)] = dict(raw=raw, camera_index=rec["camera_index"], frame_id=rec["frame_id"],
                                    model_time=rec["model_time"], original_actor_pose_world=poses[str(raw)],
                                    source_rgb_path=rgb_path, source_rgb_sha256=digest, baseline_file=relative,
                                    requested_rotation_world=requested["requested_rotation_world"] if requested else None,
                                    requested_center_world=requested["requested_center_world"] if requested else None,
                                    evidence_sha256=evidence["frames"][str(raw)]["source_sha256"]["r2_evidence"])
            pending[str(raw)] = dict(group_id=gid, raw=raw, radius=radius, fraction=fraction,
                                     status="prepared", reason="Native CUDA recolor has not been executed", images={})
        contract["groups"][gid] = dict(case=case, frames=frames)
        manifest["groups"][gid] = dict(status="prepared", frames=pending)
    for gid, reason in EVIDENCE_NEEDED.items():
        manifest["groups"][gid] = dict(status="evidence_needed", reason=reason, frames={})
    contract["unsupported_groups"] = EVIDENCE_NEEDED
    write(output / "input_contract.json", contract)
    write(output / "render_manifest.prepared.json", manifest)
    return manifest


def load_bundle(contract_path):
    contract_path = Path(contract_path).resolve()
    contract, bundle = read(contract_path), contract_path.parent
    require(contract["schema"] == INPUT_SCHEMA and contract["context"] == CONTEXT, "Wrong input contract")
    require(contract["classification"] == classification(contract["classification"]["radius"], contract["classification"]["fraction"]), "Classification contract changed")
    for entry in contract["source_assets"].values():
        require(sha(relative_file(bundle, entry["path"])) == entry["sha256"], "Source evidence copy changed")
    evidence = assignment(relative_file(bundle, contract["source_assets"]["ownership"]["path"]))
    inherited = read(relative_file(bundle, contract["source_assets"]["inherited_manifest"]["path"]))
    validate_evidence(evidence, inherited)
    require(contract["static_namespace"] == evidence["static_namespace"], "Static namespace changed")
    for group in contract["groups"].values():
        for entry in group["frames"].values():
            require(sha(relative_file(bundle, entry["baseline_file"])) == entry["source_rgb_sha256"], "Source RGB changed")
    return contract, evidence, bundle


def execute(args):
    # All validation above this boundary is CPU and standard-library only.
    require(args.enable_gpu, "CUDA rendering needs the explicit --enable-gpu flag")
    contract, evidence, bundle = load_bundle(args.input_contract)
    output = Path(args.output).resolve()
    require(not output.exists(), "No-clobber: choose a fresh render output directory")
    selected = args.groups or list(GROUP_CASES)
    require(len(selected) == len(set(selected)) and all(g in GROUP_CASES for g in selected), "Unsupported or duplicate group")
    raw_filter = set(args.raws) if args.raws else None
    if raw_filter:
        require(raw_filter <= set(RAWS), "Raw outside scene1")
    code = Path(args.code_dir).resolve()
    require((code / "train_actor.py").is_file(), "Missing existing native renderer code")
    sys.path.insert(0, str(code))
    # Dedicated render permission leaves the legacy training/queue holds intact.
    require_render_authorization(CONTEXT)
    module = importlib.import_module("train_actor")  # first CUDA-aware import
    # train_actor installs persistent_world/code, which owns gpu_slot.py.
    gpu_slot = importlib.import_module("gpu_slot")
    torch, np = module.torch, module.np
    from PIL import Image
    from utils.sh_utils import C0 as native_C0
    require(native_C0 == C0, "Unexpected native SH convention")
    require(torch.cuda.is_available(), "CUDA is unavailable; no output claimed")
    actor_checkpoint = Path(contract["actor_checkpoint_ref"])
    require(sha(actor_checkpoint) == contract["actor_checkpoint_sha256"], "Actor10K checkpoint bytes differ")
    output.mkdir(parents=True)
    radius, fraction = contract["classification"]["radius"], contract["classification"]["fraction"]
    manifest = dict(schema=SCHEMA, context=CONTEXT, status="running", gpu_executed=True,
                    training_updates=0, classification=contract["classification"], groups={},
                    required_group_frame_count=204, verified_group_frame_count=0,
                    input_contract_sha256=sha(args.input_contract),
                    evidence_sha256=contract["source_assets"]["ownership"]["sha256"],
                    actor_checkpoint_ref=str(actor_checkpoint), actor_checkpoint_sha256=contract["actor_checkpoint_sha256"],
                    static_namespace=contract["static_namespace"], actor_namespace=contract["actor_namespace"])
    for gid in GROUP_CASES:
        manifest["groups"][gid] = dict(status="pending", frames={})
    for gid, reason in EVIDENCE_NEEDED.items():
        manifest["groups"][gid] = dict(status="evidence_needed", reason=reason, frames={})
    write(output / "render_manifest.json", manifest)
    started = time.monotonic()
    try:
        with gpu_slot.exclusive(), torch.no_grad():
            require_render_authorization(CONTEXT)
            torch.set_num_threads(1)
            dataset = module.ROOT / "datasets" / CONTEXT
            frames = read(dataset / "manifest.json")["frames"]
            require([f["raw"] for f in frames] == RAWS, "Dataset frame order differs")
            actor = module.Actor(np.load(dataset / "seeds.npz"))
            payload = torch.load(str(actor_checkpoint), map_location="cuda")
            require(payload["step"] == 10000, "Wrong actor checkpoint step")
            actor.load_state_dict(payload["actor"])
            del payload
            actor_before = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            native = module.NativeMotion(Path(contract["static_checkpoint_ref"]), 30000)
            require(len(native.static._xyz) == STATIC_ROWS and len(actor.xyz) == ACTOR_ROWS, "Loaded row namespace differs")
            env_before = {k: v.detach().cpu().clone() for k, v in native.env.state_dict().items()}

            def unchanged():
                return native.unchanged() and all(torch.equal(v.detach().cpu(), actor_before[k]) for k, v in actor.state_dict().items()) and all(torch.equal(v.detach().cpu(), env_before[k]) for k, v in native.env.state_dict().items())

            def rgb_bytes(value):
                return np.round(np.clip(module.cpu(value).transpose(1, 2, 0), 0, 1) * 255).astype("uint8")

            for gid in selected:
                group = contract["groups"][gid]
                for raw_string, expected in group["frames"].items():
                    raw = int(raw_string)
                    if raw_filter and raw not in raw_filter:
                        continue
                    manifest["active_attempt"] = dict(group_id=gid, raw=raw)
                    require_render_authorization(CONTEXT)
                    i, case = expected["camera_index"], group["case"]
                    frame = frames[i]
                    require(frame["raw"] == raw and frame["model_time"] == expected["model_time"], "Dataset frame identity differs")
                    native_frame_id = native.records[i]["img_name"]
                    require(native_frame_id.split("/")[-1] == expected["frame_id"], "Native camera frame differs")
                    pose = np.eye(4)
                    pose[:3, :3], pose[:3, 3] = frame["actor_rotation_world"], frame["actor_center_world"]
                    require(np.allclose(pose, expected["original_actor_pose_world"], rtol=0, atol=1e-10), "Actor source pose differs")
                    edit = None
                    if case not in ("reconstruction", "no_edit", "delete_target"):
                        edit = dict(rotation=expected["requested_rotation_world"], center=expected["requested_center_world"])
                    row = dict(camera_index=i, frame_id=native_frame_id, camera_model_time=frame["model_time"])
                    camera = native.camera(row, {})
                    static = module.prepare_rasterization(camera, native.static, native.args)
                    require(len(static["means3D"]) == STATIC_ROWS and static["features"].shape[1] == 2, "Static raster row convention differs")
                    static["means2D"] = torch.zeros_like(static["means3D"])
                    static["features"] = torch.cat((static["features"], torch.zeros_like(static["features"][:, :1])), 1)
                    kwargs = static
                    actor_visible = case != "delete_target"
                    if actor_visible:
                        kwargs = module.helpers.merge_rows(static, actor.kwargs(native, camera, frame, edit))
                    require(kwargs["colors_precomp"] is None and kwargs["shs"] is not None, "Require native SH path")
                    actor_unknown, counts = actor_partition(evidence["frames"][raw_string], evidence["radii"], radius, fraction)
                    static_unknown = evidence["frames"][raw_string]["static"]["final_removed_ids"]
                    selected_rows = static_unknown + ([STATIC_ROWS + r for r in actor_unknown] if actor_visible else [])
                    unknown_indices = torch.tensor(selected_rows, dtype=torch.long, device="cuda")
                    colored = dict(kwargs)
                    colored["shs"] = kwargs["shs"].clone()
                    colored["shs"][unknown_indices] = 0
                    colored["shs"][unknown_indices, 0, :] = (1.0 - 0.5) / C0
                    # Only a cloned SH tensor differs. All native row/order/mask,
                    # opacity, covariance, geometry and feature objects are shared.
                    require(all(colored[k] is kwargs[k] for k in kwargs if k != "shs"), "Non-color raster inputs changed")
                    unknown_mask = torch.zeros(len(kwargs["means3D"]), dtype=torch.bool, device="cuda")
                    unknown_mask[unknown_indices] = True
                    require(torch.equal(colored["shs"][~unknown_mask], kwargs["shs"][~unknown_mask]), "Known row colors changed")
                    versions = {k: v._version for k, v in kwargs.items() if torch.is_tensor(v)}
                    colored_versions = {k: v._version for k, v in colored.items() if torch.is_tensor(v)}
                    baseline = module.helpers.raster(camera, native.args, native.static.active_sh_degree, kwargs, native.bg)
                    white = module.helpers.raster(camera, native.args, native.static.active_sh_degree, colored, native.bg)
                    require(all(v._version == versions[k] for k, v in kwargs.items() if torch.is_tensor(v)), "Raster mutated its input tensors")
                    require(all(v._version == colored_versions[k] for k, v in colored.items() if torch.is_tensor(v)), "Raster mutated color-only input tensors")
                    checks = {key + "_identical": torch.equal(baseline[key], white[key]) for key in ("alpha", "radii", "depth_sum", "extra")}
                    require(all(checks.values()), "Color-only render changed native coverage/depth")
                    env = native.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
                    base_rgb = baseline["rgb"] + (1 - baseline["alpha"]) * env
                    white_rgb = white["rgb"] + (1 - white["alpha"]) * env
                    reference = native.render(row, dict(family="delete")) if not actor_visible else module.render(native, actor, frame, i, edit=edit)
                    reference_error = float((reference["rgb"] - base_rgb).abs().max())
                    require(reference_error <= 2e-6, "Adapter differs from existing native renderer")
                    for value in (base_rgb, white_rgb, baseline["alpha"], env):
                        require(bool(torch.isfinite(value).all()), "Nonfinite native output")
                    source = relative_file(bundle, expected["baseline_file"])
                    with Image.open(source) as image:
                        source_pixels = np.asarray(image.convert("RGB"))
                    pixels = rgb_bytes(base_rgb)
                    require(source_pixels.shape == pixels.shape and np.array_equal(source_pixels, pixels), "Native baseline differs from the existing saved PNG; preserve attempt and investigate")
                    require(unchanged(), "Frozen source model changed")
                    prefix = Path("assets/v43_unknown_white") / gid / f"{raw:03d}"
                    destination = output / prefix
                    destination.mkdir(parents=True)
                    Image.fromarray(pixels).save(destination / "baseline.png")
                    Image.fromarray(rgb_bytes(white_rgb)).save(destination / "unknownwhite.png")
                    shutil.copy2(source, destination / "source.png")
                    np.savez_compressed(destination / "native_float.npz", baseline_rgb=module.cpu(base_rgb),
                        unknownwhite_rgb=module.cpu(white_rgb), baseline_gaussian_rgb=module.cpu(baseline["rgb"]),
                        unknownwhite_gaussian_rgb=module.cpu(white["rgb"]), environment_rgb=module.cpu(env),
                        baseline_alpha=module.cpu(baseline["alpha"]), unknownwhite_alpha=module.cpu(white["alpha"]),
                        baseline_radii=module.cpu(baseline["radii"]), unknownwhite_radii=module.cpu(white["radii"]),
                        static_unknown_ids=np.asarray(static_unknown, dtype=np.int64),
                        actor_unknown_ids=np.asarray(actor_unknown if actor_visible else [], dtype=np.int64))
                    receipt = dict(schema=SCHEMA, status="verified", group_id=gid, raw=raw, context=CONTEXT,
                        radius=radius, fraction=fraction, case=case, classification=contract["classification"],
                        camera_index=i, frame_id=native_frame_id, model_time=frame["model_time"], image_size=[480, 320],
                        camera_world_view=module.cpu(camera.world_view_transform).tolist(),
                        camera_full_projection=module.cpu(camera.full_proj_transform).tolist(),
                        requested_edit=edit, original_actor_pose_world=pose.tolist(),
                        static_namespace=contract["static_namespace"], actor_namespace=contract["actor_namespace"],
                        actor_checkpoint_ref=str(actor_checkpoint), actor_checkpoint_sha256=contract["actor_checkpoint_sha256"],
                        source_rgb_path=expected["source_rgb_path"], source_rgb_sha256=expected["source_rgb_sha256"],
                        source_evidence_sha256=expected["evidence_sha256"], ownership_data_sha256=manifest["evidence_sha256"],
                        static_unknown_rows=len(static_unknown), actor_unknown_rows=len(actor_unknown) if actor_visible else 0,
                        actor_partition_counts=counts, actor_visible=actor_visible, active_input_rows=len(kwargs["means3D"]),
                        raster_mask_rows=int(kwargs["mask"].sum()), unknown_color=[1., 1., 1.],
                        checks=dict(**checks, baseline_saved_png_pixels_identical=True, non_color_inputs_identical=True,
                                    known_row_SH_identical=True, row_order_unchanged=True, native_environment_unchanged=True,
                                    frozen_model_unchanged=True, actual_native_reference_max_error=reference_error),
                        training_updates=0, gpu_executed=True, pixel_painting=False, row_deletion=False,
                        local_scope_clipping=False, physically_completed_background=False)
                    receipt["files"] = {name: dict(sha256=sha(destination / name), bytes=(destination / name).stat().st_size)
                                        for name in ("baseline.png", "unknownwhite.png", "source.png", "native_float.npz")}
                    write(destination / "receipt.json", receipt)
                    entry = dict(group_id=gid, raw=raw, status="verified", radius=radius, fraction=fraction,
                                 images={name: (prefix / (name + ".png")).as_posix() for name in ("baseline", "unknownwhite", "source")},
                                 receipt=(prefix / "receipt.json").as_posix(), floats=(prefix / "native_float.npz").as_posix(),
                                 receipt_sha256=sha(destination / "receipt.json"), checks=receipt["checks"])
                    manifest["groups"][gid]["frames"][raw_string] = entry
                    manifest["verified_group_frame_count"] += 1
                    manifest["groups"][gid]["status"] = "complete" if len(manifest["groups"][gid]["frames"]) == len(group["frames"]) else "partial"
                    write(output / "render_manifest.json", manifest)
                    print("UNKNOWN_WHITE_VERIFIED", gid, raw, flush=True)
                    del kwargs, colored, static, baseline, white, reference, base_rgb, white_rgb, unknown_mask
            require(unchanged() and sha(actor_checkpoint) == contract["actor_checkpoint_sha256"], "Final immutable state check failed")
        manifest["status"] = "complete" if manifest["verified_group_frame_count"] == 204 else "partial"
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["final_frozen_state_verified"] = True
        manifest.pop("active_attempt", None)
        write(output / "render_manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = str(error)
        write(output / "render_manifest.json", manifest)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="CPU preparation; also the default")
    mode.add_argument("--enable-gpu", action="store_true", help="Explicitly execute real native CUDA rendering")
    parser.add_argument("--scene-dir", type=Path, help="Local published scene1 directory for CPU preparation")
    parser.add_argument("--input-contract", type=Path, help="Prepared portable input_contract.json for CUDA execution")
    parser.add_argument("--output", type=Path, required=True, help="Fresh destination; never overwritten")
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--radius", type=int, default=5, help="CPU preparation classification parameter")
    parser.add_argument("--fraction", type=float, default=0.2, help="CPU preparation classification parameter")
    parser.add_argument("--groups", nargs="+", choices=list(GROUP_CASES), help="Optional later CUDA bounded replay")
    parser.add_argument("--raws", nargs="+", type=int, help="Optional later CUDA bounded replay; yields partial manifest")
    args = parser.parse_args()
    if args.enable_gpu:
        require(args.input_contract is not None and args.scene_dir is None, "GPU mode needs input-contract, not scene-dir")
        result = execute(args)
    else:
        require(args.scene_dir is not None and args.input_contract is None, "CPU preparation needs scene-dir")
        require(not args.groups and not args.raws, "CPU contract always covers the full supported 204 entries")
        result = prepare(args.scene_dir, args.output, args.radius, args.fraction)
    print(json.dumps({key: result[key] for key in ("schema", "status", "gpu_executed", "verified_group_frame_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
