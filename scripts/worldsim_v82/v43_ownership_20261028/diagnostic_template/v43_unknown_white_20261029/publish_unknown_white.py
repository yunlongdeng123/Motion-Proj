"""Validate a real CUDA delivery on CPU, then publish it to scene1.

Prepared manifests and synthetic test deliveries are never accepted. This
program performs no rendering, GPU initialization, SSH or job submission.
"""
from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import json
import math
import shutil
from pathlib import Path

import render_unknown_white as runner


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHECKS = ("alpha_identical", "radii_identical", "depth_sum_identical", "extra_identical",
          "baseline_saved_png_pixels_identical", "non_color_inputs_identical",
          "known_row_SH_identical", "row_order_unchanged", "native_environment_unchanged",
          "frozen_model_unchanged")
FILES = ("baseline.png", "unknownwhite.png", "source.png", "native_float.npz")


def validate(manifest_path, contract_path, root=ROOT):
    manifest_path, contract_path, root = Path(manifest_path).resolve(), Path(contract_path).resolve(), Path(root).resolve()
    manifest = runner.read(manifest_path)
    runner.require(manifest.get("schema") == runner.SCHEMA, "Unexpected render manifest schema")
    runner.require(manifest.get("status") in ("complete", "partial"), "Only real complete/partial CUDA deliveries may be published; prepared input is rejected")
    runner.require(manifest.get("gpu_executed") is True and manifest.get("final_frozen_state_verified") is True, "Missing real CUDA execution/final immutable-state verification")
    runner.require(manifest.get("training_updates") == 0, "Unexpected training updates")
    contract = runner.read(contract_path)
    runner.require(contract.get("schema") == runner.INPUT_SCHEMA, "Unexpected input contract schema")
    runner.require(manifest.get("context") == contract.get("context") == runner.CONTEXT, "Context mismatch")
    runner.require(manifest.get("input_contract_sha256") == runner.sha(contract_path), "Input contract changed since rendering")
    parameters = contract["classification"]
    runner.require(parameters == runner.classification(parameters["radius"], parameters["fraction"]), "Classification rule changed")
    runner.require(manifest.get("classification") == parameters, "Render/input classification mismatch")
    ownership = contract["source_assets"]["ownership"]
    ownership_path = runner.relative_file(contract_path.parent, ownership["path"])
    runner.require(runner.sha(ownership_path) == ownership["sha256"] == manifest.get("evidence_sha256"), "Ownership evidence changed")
    evidence = runner.assignment(ownership_path)
    inherited = runner.read(runner.relative_file(contract_path.parent, contract["source_assets"]["inherited_manifest"]["path"]))
    runner.validate_evidence(evidence, inherited)
    for field in ("actor_checkpoint_ref", "actor_checkpoint_sha256", "static_namespace", "actor_namespace"):
        runner.require(manifest.get(field) == contract[field], "Model/evidence binding differs: " + field)
    scene = root / "outputs/GS_Scene_Diagnostics_20261027" / runner.CONTEXT
    published_groups = {g["id"]: g for g in runner.read(scene / "v43_groups_data.json")["groups"]}
    required_ids = set(runner.GROUP_CASES) | set(runner.EVIDENCE_NEEDED)
    runner.require(set(manifest["groups"]) == required_ids, "Unexpected or missing groups")
    for gid in runner.EVIDENCE_NEEDED:
        group = manifest["groups"][gid]
        runner.require(group["status"] == "evidence_needed" and not group["frames"], "Unsupported checkpoint was given render results")

    # Imports are CPU libraries; rejection of prepared input happens above.
    import numpy as np
    from PIL import Image

    def png(path):
        with Image.open(path) as image:
            runner.require(image.format == "PNG" and image.mode == "RGB" and image.size == (480, 320), "Wrong image format/shape: " + str(path))
            return np.array(image)

    def encoded(rgb):
        return np.round(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).astype("uint8")

    assets, verified = [], 0
    for gid in runner.GROUP_CASES:
        group = manifest["groups"][gid]
        expected_group = contract["groups"][gid]
        actual_raws, expected_raws = set(group["frames"]), set(expected_group["frames"])
        runner.require(actual_raws <= expected_raws, "Unexpected frame in " + gid)
        expected_status = "complete" if actual_raws == expected_raws else "partial" if actual_raws else "pending"
        runner.require(group["status"] == expected_status, "Group status/count mismatch: " + gid)
        for raw_string, entry in group["frames"].items():
            raw, expected = int(raw_string), expected_group["frames"][raw_string]
            runner.require(entry.get("status") == "verified", "Frame was not verified")
            for field, value in (("group_id", gid), ("raw", raw), ("radius", parameters["radius"]), ("fraction", parameters["fraction"])):
                runner.require(entry.get(field) == value, "Entry identity/parameters mismatch: " + field)
            receipt_path = runner.relative_file(manifest_path.parent, entry["receipt"])
            runner.require(runner.sha(receipt_path) == entry["receipt_sha256"], "Receipt checksum differs")
            receipt = runner.read(receipt_path)
            for field in ("group_id", "raw", "radius", "fraction"):
                runner.require(receipt.get(field) == entry[field], "Receipt/entry mismatch: " + field)
            runner.require(receipt.get("schema") == runner.SCHEMA and receipt.get("status") == "verified", "Unverified receipt")
            runner.require(receipt.get("context") == runner.CONTEXT and receipt.get("classification") == parameters, "Receipt source/classification differs")
            runner.require(receipt.get("gpu_executed") is True and receipt.get("training_updates") == 0, "Receipt does not record read-only CUDA execution")
            runner.require(receipt.get("case") == expected_group["case"], "Physical case differs")
            for field in ("pixel_painting", "row_deletion", "local_scope_clipping"):
                runner.require(receipt.get(field) is False, "Forbidden render operation: " + field)
            checks = receipt["checks"]
            runner.require(entry.get("checks") == checks and all(checks.get(k) is True for k in CHECKS), "Missing or failed native invariant check")
            error = checks.get("actual_native_reference_max_error")
            runner.require(isinstance(error, (int, float)) and math.isfinite(error) and 0 <= error <= 2e-6, "Native reference comparison failed")
            for field in ("actor_checkpoint_ref", "actor_checkpoint_sha256", "static_namespace", "actor_namespace"):
                runner.require(receipt.get(field) == contract[field], "Receipt namespace/checkpoint differs: " + field)
            runner.require(receipt.get("ownership_data_sha256") == ownership["sha256"] and receipt.get("source_evidence_sha256") == expected["evidence_sha256"], "Receipt evidence binding differs")
            runner.require(receipt.get("camera_index") == expected["camera_index"] and receipt.get("model_time") == expected["model_time"], "Frame camera/time differs")
            runner.require(receipt["frame_id"].split("/")[-1] == expected["frame_id"] and receipt.get("image_size") == [480, 320], "Frame/image identity differs")
            runner.require(receipt.get("original_actor_pose_world") == expected["original_actor_pose_world"], "Source actor pose differs")
            edited = expected_group["case"] not in ("reconstruction", "no_edit", "delete_target")
            wanted_edit = dict(rotation=expected["requested_rotation_world"], center=expected["requested_center_world"]) if edited else None
            runner.require(receipt.get("requested_edit") == wanted_edit, "Physical edit differs")
            source_rgb = published_groups[gid]["frames"][raw_string]["channels"]["rgb"]
            runner.require(source_rgb == expected["source_rgb_path"] == receipt.get("source_rgb_path"), "Current published source RGB route changed")
            published_source = runner.relative_file(scene, source_rgb)
            runner.require(runner.sha(published_source) == expected["source_rgb_sha256"] == receipt.get("source_rgb_sha256"), "Current published source RGB differs")
            runner.require(set(receipt["files"]) == set(FILES), "Receipt file inventory differs")
            frame_files = {}
            for filename in FILES:
                path = receipt_path.parent / filename
                metadata = receipt["files"][filename]
                runner.require(path.is_file() and path.stat().st_size == metadata["bytes"] and runner.sha(path) == metadata["sha256"], "Missing/corrupt render asset: " + str(path))
                frame_files[filename] = path
            for channel in ("baseline", "unknownwhite", "source"):
                runner.require(runner.relative_file(manifest_path.parent, entry["images"][channel]) == frame_files[channel + ".png"], "Image/receipt path differs")
            runner.require(runner.relative_file(manifest_path.parent, entry["floats"]) == frame_files["native_float.npz"], "Float/receipt path differs")
            runner.require(receipt["files"]["source.png"]["sha256"] == expected["source_rgb_sha256"], "Preserved source PNG differs")
            baseline_png, white_png, source_png = (png(frame_files[n]) for n in FILES[:3])
            runner.require(np.array_equal(baseline_png, source_png), "Native baseline and preserved source pixels differ")
            actor_unknown, counts = runner.actor_partition(evidence["frames"][raw_string], evidence["radii"] , parameters["radius"], parameters["fraction"])
            static_unknown = evidence["frames"][raw_string]["static"]["final_removed_ids"]
            actor_visible = expected_group["case"] != "delete_target"
            actor_unknown = actor_unknown if actor_visible else []
            total = runner.STATIC_ROWS + (runner.ACTOR_ROWS if actor_visible else 0)
            runner.require(receipt.get("actor_visible") is actor_visible and receipt.get("active_input_rows") == total, "Rendered branch namespace differs")
            runner.require(receipt.get("static_unknown_rows") == len(static_unknown) and receipt.get("actor_unknown_rows") == len(actor_unknown) and receipt.get("actor_partition_counts") == counts, "Rendered unknown population differs")
            with np.load(frame_files["native_float.npz"], allow_pickle=False) as data:
                for key in ("baseline_rgb", "unknownwhite_rgb", "baseline_gaussian_rgb", "unknownwhite_gaussian_rgb", "environment_rgb"):
                    runner.require(data[key].shape == (3, 320, 480) and np.isfinite(data[key]).all(), "Invalid float RGB: " + key)
                for key in ("baseline_alpha", "unknownwhite_alpha"):
                    runner.require(data[key].shape == (1, 320, 480) and np.isfinite(data[key]).all(), "Invalid alpha array")
                runner.require(np.array_equal(data["baseline_alpha"], data["unknownwhite_alpha"]), "Saved float alpha differs")
                runner.require(data["baseline_radii"].shape == data["unknownwhite_radii"].shape == (total,) and np.array_equal(data["baseline_radii"], data["unknownwhite_radii"]), "Saved native radii differ")
                for key, ids in (("static_unknown_ids", static_unknown), ("actor_unknown_ids", actor_unknown)):
                    runner.require(np.issubdtype(data[key].dtype, np.integer) and np.array_equal(data[key], np.asarray(ids, dtype=np.int64)), "Unknown row IDs differ: " + key)
                runner.require(np.array_equal(encoded(data["baseline_rgb"]), baseline_png) and np.array_equal(encoded(data["unknownwhite_rgb"]), white_png), "PNG does not encode its saved native float RGB")
                for prefix in ("baseline", "unknownwhite"):
                    composite = data[prefix + "_gaussian_rgb"] + (1 - data[prefix + "_alpha"]) * data["environment_rgb"]
                    runner.require(float(np.max(np.abs(composite - data[prefix + "_rgb"]))) <= 1e-6, "Native environment composition differs")
            assets.append(dict(group_id=gid, raw=raw_string, receipt=receipt_path, files=frame_files))
            verified += 1
    runner.require(0 < verified == manifest.get("verified_group_frame_count") <= 204 and manifest.get("required_group_frame_count") == 204, "Manifest verified count differs")
    runner.require((manifest["status"] == "complete") == (verified == 204), "Overall completion status differs")
    return manifest, assets, scene, published_groups


def publish(manifest_path, contract_path, root=ROOT, validate_only=False):
    manifest_path = Path(manifest_path).resolve()
    manifest, assets, scene, groups = validate(manifest_path, contract_path, root)
    if validate_only:
        return dict(passed=True, published=False, frames=len(assets), status=manifest["status"])
    delivery_sha = runner.sha(manifest_path)
    prefix = Path("assets/v43_unknown_white") / delivery_sha[:16]
    published = copy.deepcopy(manifest)
    for item in assets:
        gid, raw = item["group_id"], item["raw"]
        target = scene / prefix / gid / f"{int(raw):03d}"
        target.mkdir(parents=True, exist_ok=True)
        entry = published["groups"][gid]["frames"][raw]
        receipt = runner.read(item["receipt"])
        for name, source in {**item["files"], "receipt.json": item["receipt"]}.items():
            expected_sha = entry["receipt_sha256"] if name == "receipt.json" else receipt["files"][name]["sha256"]
            destination = target / name
            if not destination.exists():
                shutil.copy2(source, destination)
            runner.require(runner.sha(destination) == expected_sha, "Published asset failed checksum; existing versioned files were not overwritten")
        entry["images"] = {k: (target / (k + ".png")).relative_to(scene).as_posix() for k in ("baseline", "unknownwhite", "source")}
        entry["receipt"] = (target / "receipt.json").relative_to(scene).as_posix()
        entry["floats"] = (target / "native_float.npz").relative_to(scene).as_posix()
        # The existing standalone viewer loads PNGs through path+'.js'. This
        # packages the verified PNG bytes; it does not synthesize image pixels.
        image_path = entry["images"]["unknownwhite"]
        uri = "data:image/png;base64," + base64.b64encode((target / "unknownwhite.png").read_bytes()).decode("ascii")
        (target / "unknownwhite.png.js").write_text("window.IMAGE_DATA[" + json.dumps(image_path) + "]=" + json.dumps(uri) + ";\n", encoding="utf8")
    for gid, group in published["groups"].items():
        group["title"] = groups[gid]["title"]
    published["publication"] = dict(validated_on_CPU=True, source_render_manifest_sha256=delivery_sha,
                                     versioned_asset_prefix=prefix.as_posix(), no_GPU_or_SSH_used_by_publisher=True)
    runner.write(scene / "unknown_white_manifest.json", published)
    spec = importlib.util.spec_from_file_location("unknown_white_installer", HERE / "install_unknown_white.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    installer.install(Path(root))
    installed = runner.read(scene / "unknown_white_manifest.json")
    runner.require(installed == published, "Installer did not preserve the verified native delivery")
    return dict(passed=True, published=True, frames=len(assets), status=published["status"],
                manifest=str(scene / "unknown_white_manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.render_manifest, args.input_contract, args.root, args.validate_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
