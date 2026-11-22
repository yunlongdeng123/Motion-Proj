"""Cold-process reload of a proved static/dynamic context-inactive sidecar.

Uses the original source_attribution.Audit CLI plus --cleanup-dir. It reconstructs
the original source-bound Runtime, reads the persisted sidecar, applies only its
context mask, and compares actual BUILD86/110 renders with saved final arrays.
No cleanup selection, optimizer step, road patch or source-asset write occurs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

BUILD = (86, 110)
PARITY_LIMIT = 5e-6
SHARED_KEYS = ("rgb", "alpha", "actor_alpha", "K_alpha")
ACTOR_KEYS = ("rgb", "alpha", "depth_sum_m")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def id_array(value, label):
    a = np.asarray(value)
    if a.ndim == 1 and not a.size:
        return np.empty(0, np.int64)
    if a.ndim != 1 or a.dtype.kind not in "iu" or (a < 0).any():
        raise ValueError(label + " must be a nonnegative integer vector")
    if len(np.unique(a)) != len(a):
        raise ValueError(label + " contains duplicate IDs")
    return a.astype(np.int64, copy=False)


def validate_sidecar(sidecar, binding, proof, ns, nd):
    """Pure CPU validation, including topology-derived mixed source branches."""
    required = {"native_global_row_ids", "opacity_multiplier", "source_branch", "global_per_camera_selection"}
    if not required.issubset(sidecar):
        raise ValueError("Sidecar lacks required inactive/mixed-branch fields")
    ids = id_array(sidecar["native_global_row_ids"], "sidecar IDs")
    if len(ids) and (np.diff(ids) <= 0).any():
        raise ValueError("Sidecar native IDs must retain strict global sort order")
    if ns <= 0 or nd < 0 or (ids >= ns + nd).any():
        raise ValueError("Sidecar reaches outside native checkpoints, including road/new a20")
    multiplier = sidecar["opacity_multiplier"]
    if multiplier.shape != ids.shape or multiplier.dtype.kind != "f" or not np.isfinite(multiplier).all() or not (multiplier == 0).all():
        raise ValueError("Only exact-zero inactive opacity multipliers are supported")
    branch = sidecar["source_branch"]
    expected_branch = (ids >= ns).astype(np.int8)
    if branch.shape != ids.shape or branch.dtype.kind not in "iu" or not np.array_equal(branch, expected_branch):
        raise ValueError("source_branch does not match exact static/dynamic global topology")
    if "source_row_id" in sidecar:
        source_row = sidecar["source_row_id"]
        expected_row = ids - expected_branch.astype(np.int64) * ns
        if source_row.shape != ids.shape or source_row.dtype.kind not in "iu" or not np.array_equal(source_row, expected_row):
            raise ValueError("source_row_id differs from branch-local checkpoint identity")
    per_camera = np.asarray(sidecar["global_per_camera_selection"])
    if per_camera.shape != () or per_camera.dtype != np.bool_ or bool(per_camera):
        raise ValueError("Camera-specific sidecar selection is forbidden")
    if not proof.get("complete") or not proof.get("full_two_view_protection_passed"):
        raise ValueError("Cleanup proof is incomplete or lacks passing whole-set guards")
    if proof.get("optimizer_updates") != 0 or binding.get("optimizer_updates") != 0:
        raise ValueError("Cold reload expects a zero-update context cleanup")
    if proof.get("heldout_frames_opened", []) != []:
        raise ValueError("Cleanup proof opened sealed frames")
    if binding.get("camera_specific_mask") is not False or binding.get("actor_and_road_values_unchanged") is not True:
        raise ValueError("Binding must freeze actor/road and use one global sidecar")
    if not proof.get("actor_and_global_active_unchanged") or not proof.get("road_asset_bytes_unchanged"):
        raise ValueError("Cleanup proof does not preserve actor and road")
    if not proof.get("source_files_unchanged") or not all(proof["source_files_unchanged"].values()) or proof.get("Adam_unchanged") is not True:
        raise ValueError("Cleanup source-file preservation proof failed")
    accepted = id_array(proof.get("accepted_global_row_ids", []), "proved accepted IDs")
    if not np.array_equal(np.sort(accepted), ids) or proof.get("accepted_context_rows") != len(ids):
        raise ValueError("Persisted sidecar differs from the completed accepted set")
    dynamic = ids[expected_branch == 1]
    proved_dynamic = id_array(proof.get("new_dynamic_global_row_ids", np.empty(0, np.int64)), "proved dynamic IDs")
    if not np.array_equal(np.sort(proved_dynamic), dynamic):
        raise ValueError("Every dynamic ID must be explicitly enumerated by the final proof")
    if len(dynamic):
        if proof.get("dynamic_source_branch") != 1 or not proof.get("checked_owner_masks_exact_zero"):
            raise ValueError("Mixed proof lacks explicit dynamic/retained-owner strata")
        if binding.get("whole_actor_cleanup") is not False or binding.get("other_actor_source_parameters_unchanged") is not True:
            raise ValueError("Dynamic cleanup must remain an individual-row sidecar")
        if binding.get("source_branch_convention") != {"0": "native static", "1": "native dynamic"}:
            raise ValueError("Mixed binding source-branch convention is missing")
    owner_strata = {}
    for row in proof.get("explicitly_checked_owner_visible_masks", []):
        raw, owner = int(row["raw"]), int(row["owner"])
        if raw not in BUILD or owner not in (8, 14):
            raise ValueError("Unexpected dynamic owner/frame stratum in proof")
        if row.get("exact_zero_change") is not True or row.get("visible_scene_RGB_change_max") != 0 or row.get("visible_owner_alpha_change_max") != 0:
            raise ValueError("Dynamic retained-owner visibility stratum is not exact-zero")
        owner_strata[(raw, owner)] = row
    return ids, expected_branch, owner_strata


def ref_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from ref_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from ref_records(child)


def compare(actual, expected, keys, tolerance):
    differences = {}
    for key in keys:
        if key not in actual or key not in expected:
            raise ValueError("Missing required comparison array: " + key)
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.shape != b.shape or a.dtype.kind != "f" or b.dtype.kind != "f":
            raise ValueError("Comparison shape/dtype mismatch: " + key)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError("Nonfinite comparison array: " + key)
        differences[key] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
    if any(value > tolerance for value in differences.values()):
        raise RuntimeError("Cold-render parity failed: " + json.dumps(differences))
    return differences


def cpu_self_check():
    ids = np.asarray([1, 11], np.int64)
    sidecar = dict(native_global_row_ids=ids, opacity_multiplier=np.zeros(2, np.float32),
                   source_branch=np.asarray([0, 1], np.int8), source_row_id=np.asarray([1, 1], np.int64),
                   global_per_camera_selection=np.asarray(False))
    binding = dict(optimizer_updates=0, camera_specific_mask=False, actor_and_road_values_unchanged=True,
                   whole_actor_cleanup=False, other_actor_source_parameters_unchanged=True,
                   source_branch_convention={"0": "native static", "1": "native dynamic"})
    proof = dict(complete=True, full_two_view_protection_passed=True, optimizer_updates=0,
                 actor_and_global_active_unchanged=True, road_asset_bytes_unchanged=True,
                 source_files_unchanged={"test": True}, Adam_unchanged=True, accepted_global_row_ids=ids.tolist(), accepted_context_rows=2,
                 new_dynamic_global_row_ids=[11], dynamic_source_branch=1, checked_owner_masks_exact_zero=True)
    validate_sidecar(sidecar, binding, proof, 10, 10)
    failures = 0
    for changes, ns, nd in [({"source_branch": np.zeros(2, np.int8)}, 10, 10),
                            ({"native_global_row_ids": np.asarray([1, 20])}, 10, 10),
                            ({"source_row_id": np.asarray([1, 11])}, 10, 10)]:
        try:
            validate_sidecar(dict(sidecar, **changes), binding, proof, ns, nd)
        except ValueError:
            failures += 1
    try:
        validate_sidecar(sidecar, binding, dict(proof, new_dynamic_global_row_ids=[]), 10, 10)
    except ValueError:
        failures += 1
    assert failures == 4
    a = {key: np.ones((2, 3), np.float32) for key in SHARED_KEYS}
    assert not any(compare(a, a, SHARED_KEYS, PARITY_LIMIT).values())
    return dict(passed=True, GPU_used=False, mixed_branch_validation=True,
                wrong_branch_road_range_local_ID_unproved_dynamic_rejected=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-self-check", action="store_true")
    for key in ("runtime-dir", "checkpoint", "actor", "init", "manifest", "source-manifest", "prior-contract",
                "context-plan", "path-map-file", "roi-file", "cleanup-dir", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--path-map", action="append", default=[])
    p.add_argument("--expected-actor-sha-prefix", default="ab4fa110")
    p.add_argument("--top-rows", type=int, default=64)
    return p.parse_args()


def execute(args):
    if args.cleanup_dir is None or args.output is None:
        raise ValueError("--cleanup-dir and fresh --output are required")
    cleanup = args.cleanup_dir.resolve()
    if not cleanup.is_dir() or args.output.exists():
        raise ValueError("Cleanup must exist and output must be fresh")
    if any(getattr(args, key) is None for key in ("runtime_dir", "checkpoint", "actor", "source_manifest")):
        raise ValueError("Original Audit parent arguments are required")
    binding_path, proof_path = cleanup / "scene_context_binding.json", cleanup / "proof.json"
    sidecar_path = cleanup / "context_inactive_sidecar.npz"
    binding, proof = read(binding_path), read(proof_path)
    watched = {}

    def watch(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError("Source SHA mismatch: " + str(path))
        previous = watched.setdefault(str(path), value)
        if previous != value:
            raise ValueError("Source changed while loading: " + str(path))
        return path

    for path in (binding_path, proof_path):
        watch(path)
    watch(__file__)
    for key, path in (("sidecar", sidecar_path), ("actor", args.actor), ("actor_checkpoint", args.checkpoint),
                      ("native_source_manifest", args.source_manifest)):
        if not isinstance(binding.get(key), dict) or not binding[key].get("sha256"):
            raise ValueError("Missing bound source: " + key)
        watch(path, binding[key]["sha256"])
    watch(sidecar_path, proof["context_sidecar"]["sha256"])
    watch(args.actor, proof["actor_asset"]["sha256"])
    expected_after = {}
    for raw in BUILD:
        path = watch(cleanup / "evaluation" / f"raw{raw:03d}_after.npz")
        expected_after[raw] = bank(path)
        if not set(SHARED_KEYS).issubset(expected_after[raw]):
            raise ValueError("Saved evaluation requires rgb/alpha/actor_alpha/K_alpha")

    # Importing Audit is CPU-safe; only constructing it enters the native CUDA path.
    import source_attribution as source
    watch(source.__file__)
    started = time.monotonic()
    audit = source.Audit(args)
    rt, t, core = audit.rt, audit.t, audit.core
    if tuple(core.BUILD) != BUILD:
        raise ValueError("Cold reload is restricted to BUILD86/110")
    road_path = rt.s.path(audit.i["native"]["native_renderer"]["road_root"]) / "road.npz"
    watch(road_path, binding["original_road"]["sha256"])
    alias = {binding[key]["path"]: Path(path).resolve() for key, path in
             (("sidecar", sidecar_path), ("actor", args.actor), ("actor_checkpoint", args.checkpoint),
              ("native_source_manifest", args.source_manifest), ("original_road", road_path))}
    for record in ref_records([binding, proof]):
        path = alias.get(record["path"])
        if path is None:
            path = rt.s.path(record["path"])
            if not path.is_absolute():
                path = cleanup / path
        watch(path, record["sha256"])
    # Every actual native checkpoint/ownership source is independently hash checked.
    for record in audit.i["native"]["native_renderer"]["native_source_bindings"]:
        watch(rt.s.get(record, "cold reload native parent"), record["sha256"])
    for path in audit.source_paths:
        watch(path, audit.source_hashes[str(path.resolve())])
    for record in rt.s.opened:
        watch(record["path"], record.get("sha256"))
    for key in ("init_path", "manifest_path", "native_path", "teacher_path", "measured_path"):
        if audit.i.get(key) is not None:
            watch(audit.i[key])

    # Reload persisted arrays here; no accepted list or result variable is reused.
    sidecar = bank(sidecar_path)
    ids, branch, strata = validate_sidecar(sidecar, binding, proof, audit.ns, audit.nd)
    if np.intersect1d(ids, audit.arr(rt.K_preserve)).size:
        raise ValueError("Sidecar intersects the complete native K preservation union")
    base_removed = np.asarray(rt.context_plan.get("remove_native_global_row_ids", []), np.int64)
    if np.intersect1d(ids, base_removed).size:
        raise ValueError("Runtime was not loaded from original pre-cleanup context")
    actor_before = {}
    seen = set()
    context_checks = []
    with t.no_grad():
        for raw in BUILD:
            car = rt.render(raw=raw, actor_only=True)
            actor_before[raw] = {key: audit.arr(car[key]) for key in ACTOR_KEYS}
            packet, owners_t, road_t, gids_t = rt.context[raw]
            gids, owners, road = audit.arr(gids_t), audit.arr(owners_t), audit.arr(road_t)
            selected = np.isin(gids, ids)
            rows = np.flatnonzero(selected)
            seen.update(gids[selected].tolist())
            if (road[selected] >= 0).any():
                raise ValueError("Loaded sidecar would mask a road row")
            dynamic = gids[selected] >= audit.ns
            if ((owners[selected][~dynamic] > 0) & (owners[selected][~dynamic] != 21)).any():
                raise ValueError("Static cleanup would mask a retained other actor")
            if dynamic.any():
                if not np.isin(owners[selected][dynamic], [8, 14]).all():
                    raise ValueError("Dynamic cleanup owner is outside explicit owner8/14 strata")
                for owner in np.unique(owners[selected][dynamic]):
                    if (raw, int(owner)) not in strata:
                        raise ValueError("Actual dynamic owner/frame lacks an explicit exact-zero visibility proof")
            context_checks.append(dict(raw=raw, selected_present_rows=len(rows),
                                       static_present_rows=int((~dynamic).sum()), dynamic_present_rows=int(dynamic.sum()),
                                       selected_owner_counts={str(int(k)): int(v) for k, v in zip(*np.unique(owners[selected], return_counts=True))},
                                       original_context_mask_active_rows=int(audit.arr(packet["mask"])[selected].sum())))
        if seen != set(ids.tolist()):
            raise ValueError("Some persisted IDs are absent from both original BUILD packets")
        for raw in BUILD:
            packet, owners, road, gids = rt.context[raw]
            inactive = t.isin(gids, t.as_tensor(ids, device=gids.device, dtype=gids.dtype))
            rt.context[raw] = (dict(packet, mask=packet["mask"] & ~inactive), owners, road, gids)

        shared_comparisons, actor_comparisons = [], []
        folder = args.output / "evaluation"
        folder.mkdir(parents=True, exist_ok=True)
        for raw in BUILD:
            rendered = rt.render(raw=raw)
            actual = {key: audit.arr(rendered[key]) for key in SHARED_KEYS}
            maximum = compare(actual, expected_after[raw], SHARED_KEYS, PARITY_LIMIT)
            shared_comparisons.append(dict(raw=raw, maximum=maximum, tolerance=PARITY_LIMIT))
            car = rt.render(raw=raw, actor_only=True)
            isolated = {key: audit.arr(car[key]) for key in ACTOR_KEYS}
            car_max = compare(isolated, actor_before[raw], ACTOR_KEYS, 0.)
            actor_comparisons.append(dict(raw=raw, maximum=car_max, tolerance=0.))
            np.savez_compressed(folder / f"raw{raw:03d}_reloaded.npz", **actual)

    actor_unchanged = audit.gpu.tree_equal(audit.gpu.cpu(rt.actor.state_dict()), audit.actor_before)
    Adam_unchanged = audit.gpu.tree_equal(audit.gpu.cpu(audit.opt.state_dict()), audit.Adam_before)
    if not actor_unchanged or not Adam_unchanged:
        raise RuntimeError("Cold reload changed actor state or Adam")
    frozen = rt.check_frozen()
    source_unchanged = {path: digest(path) == value for path, value in watched.items()}
    if not all(source_unchanged.values()):
        raise RuntimeError("An original source or cleanup evidence file changed during cold reload")
    result = dict(schema="a20.scene_residual_cleanup.cold_reload_proof.v1", complete=True, passed=True,
                  cleanup_dir=str(cleanup), sidecar=core.ref(sidecar_path), binding=core.ref(binding_path), original_proof=core.ref(proof_path),
                  script=core.ref(__file__), original_runtime=core.ref(args.runtime_dir / "joint_gpu.py"),
                  actor=core.ref(args.actor), actor_checkpoint=core.ref(args.checkpoint), original_road=core.ref(road_path),
                  native_source_manifest=core.ref(args.source_manifest), native_topology=dict(static_rows=audit.ns, dynamic_rows=audit.nd),
                  inactive_rows=len(ids), static_rows=int((branch == 0).sum()), dynamic_rows=int((branch == 1).sum()),
                  exact_source_branch_validated=True, loaded_sidecar_applied_to_original_Runtime=True,
                  reused_accepted_variable=False, no_road_or_new_a20_rows=True, context_checks=context_checks,
                  actual_shared_render_comparisons=shared_comparisons, actor_only_exact_zero_change=actor_comparisons,
                  actor_and_global_active_unchanged=actor_unchanged, Adam_unchanged=Adam_unchanged,
                  source_files_unchanged=source_unchanged, source_SHA256=watched, frozen_scene=frozen,
                  BUILD=list(BUILD), heldout_frames_opened=[], optimizer_updates=0, cleanup_decisions=0,
                  road_patch_activated=False, actual_native_CUDA=True, native_calls=rt.calls,
                  elapsed_seconds=time.monotonic() - started, GPU_keep_on=True, power_operations=0)
    core.write(args.output / "proof.json", result)
    core.write(args.output / "resource_ledger.json", dict(native_calls=rt.calls, optimizer_updates=0, cleanup_decisions=0,
                                                        cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(), GPU_keep_on=True, power_operations=0))
    compact = dict(rows=len(ids), static=result["static_rows"], dynamic=result["dynamic_rows"],
                   maximum_shared=max(v for r in shared_comparisons for v in r["maximum"].values()))
    print("COLD_RELOAD_CLEANUP_PASS", json.dumps(compact), flush=True)
    return result


if __name__ == "__main__":
    args = parse_args()
    preexisting_output = args.output is not None and args.output.exists()
    if args.cpu_self_check:
        print(json.dumps(cpu_self_check(), indent=2))
    else:
        try:
            execute(args)
        except Exception as exc:
            if args.output is not None and not preexisting_output and args.output.is_dir():
                (args.output / "failure.json").write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                                                                        optimizer_updates=0, GPU_keep_on=True, power_operations=0,
                                                                        PID=os.getpid()), indent=2) + "\n", encoding="utf-8")
            raise
