"""R8: continue the verified new field after its 2000-step checkpoint; total budget2800.

The objective.step function is the only mathematical optimizer/topology step.
Cache/reference creation, evaluation, readout and engineering replay are separate
from unique training updates. SIGTERM is handled at the next optimizer boundary.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import signal
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ("xyz", "log_scales", "raw_quaternion", "opacity_logits", "dc")
TRAIN = [106, 107, 108, 109, 110, 120, 137, 165]
GROUPS = [[106, 107, 108, 109, 110], [120], [137], [165]]
READOUT = list(range(106, 111)) + list(range(181, 186))
SCHEMA = "scene065.teacher_gaussian.phase_checkpoint.v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=path.as_posix(), sha256=sha(path))


def pinned(spec):
    path = Path(spec["path"])
    assert path.is_file() and sha(path) == spec["sha256"], str(path)
    return path.resolve()


def module(path, name):
    path = Path(path).resolve(); sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def preflight(a):
    for path in [a.config, a.runtime_contract, a.execution_contract, a.runtime_module, a.objective_module, a.lifecycle_module]:
        assert path.is_file(), str(path)
    cfg, rc, ex = read(a.config), read(a.runtime_contract), read(a.execution_contract)
    assert ex["authorized"] is True and ex["full_gaussian_reconstruction"] is True
    assert ex["unique_optimizer_budget"] == cfg["target_unique_updates"] == 2800
    assert ex["native30k_original_unedited"] and ex["green_and_orange_frozen"]
    assert ex["new_parameter_groups"] == list(NAMES) and ex["densification_enabled"]
    assert not ex["fixed_plane_RGB_only"] and not ex["permanent_A20_world_carve"]
    assert cfg["probe_unique_updates"] == 32 and cfg["views_per_optimizer_step"] == 2
    assert cfg["train_frames"] == TRAIN and cfg["view_groups"] == GROUPS
    assert set(cfg["learning_rates"]) == set(NAMES)
    assert set(cfg["fixed_eval_frames"]) == set(TRAIN)
    assert not set(TRAIN) & set(cfg["evaluation_frames"])
    assert set(READOUT) <= set(rc["frames"])
    assert rc["schema"] in ["scene065.teacher_gaussian.runtime.v1", "scene065.teacher_gaussian_reconstruction.runtime.v1"]
    teacher_path = a.teacher_targets or pinned(ex["teacher_targets"])
    teacher_path = Path(teacher_path).resolve()
    assert pin(teacher_path)["sha256"] == ex["teacher_targets"]["sha256"]
    binding = read(teacher_path)
    assert set(binding["training_frames"]) == set(TRAIN)
    for row in binding["frames"]:
        if int(row["raw"]) in cfg["evaluation_frames"]:
            assert not row.get("categories"), "Held-out RGB cannot enter the training cache"
    mask_root = a.mask_root or Path(ex["source_masks"]["path"]).parent
    mask_root = Path(mask_root).resolve()
    assert pin(mask_root / "manifest.json")["sha256"] == ex["source_masks"]["sha256"]
    sources = {}
    for name in ["native_adapter_code", "native30k_contract", "candidate_pool", "pose", "context_retirement"]:
        sources[name] = pin(pinned(rc[name]))
    seed_spec = rc.get("teacher_seed", rc.get("new_field"))
    assert seed_spec is not None
    sources["teacher_seed"] = pin(pinned(seed_spec))
    assert sources["teacher_seed"]["sha256"] == ex["automatic_seed"]["sha256"]
    assert sources["candidate_pool"]["sha256"] == ex["candidate_pool"]["sha256"]
    native = read(sources["native30k_contract"]["path"])
    assert native["iteration"] == 30000 and native["complete_unedited_native_30k"]
    binds = dict(configuration=pin(a.config), runtime_contract=pin(a.runtime_contract), execution_contract=pin(a.execution_contract),
        teacher_targets=pin(teacher_path), masks_manifest=pin(mask_root / "manifest.json"),
        runtime_code=pin(a.runtime_module), objective_code=pin(a.objective_module),
        lifecycle_code=pin(a.lifecycle_module), driver_code=pin(__file__), **sources)
    for name in ["source_teachers", "source_native"]:
        if name in ex:
            binds[name] = pin(pinned(ex[name]))
    return cfg, rc, ex, teacher_path, mask_root, binds


def seed_all(t, seed):
    random.seed(seed); np.random.seed(seed); t.manual_seed(seed)
    if t.cuda.is_initialized():
        t.cuda.manual_seed_all(seed)


def sync(rt):
    if rt.torch.cuda.is_initialized():
        rt.torch.cuda.synchronize()


def rows(rt):
    return len(rt.named_new_parameters()["xyz"])


def inference(rt, life, action):
    """Diagnostics cannot consume training RNG or change topology/residual state."""
    t = rt.torch
    before = life.cpu(rt.capture_model_state())
    rng = (random.getstate(), np.random.get_state(), t.get_rng_state().clone(),
           [x.clone() for x in t.cuda.get_rng_state_all()] if t.cuda.is_initialized() else [])
    try:
        with t.no_grad():
            result = action()
        assert life.equal(before, rt.capture_model_state()), "Inference changed model/densification state"
        rt.assert_fixed()
        return result
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); t.set_rng_state(rng[2])
        if rng[3]:
            t.cuda.set_rng_state_all(rng[3])


def readout(rt, path, logical_step, frames):
    from PIL import Image
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    records = []
    for raw in frames:
        folder = path / f"{raw:03}"; folder.mkdir()
        arrays = {k: v.detach().cpu().numpy() for k, v in rt.render(raw).items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(folder / "joint.npz", **arrays)
        Image.fromarray(np.rint(np.clip(arrays["rgb"].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)).save(folder / "joint.png")
        for key in ["alpha", "local_effective_alpha", "actor_alpha", "context_alpha"]:
            if key in arrays:
                Image.fromarray(np.rint(np.clip(np.squeeze(arrays[key]), 0, 1) * 255).astype(np.uint8)).save(folder / (key + ".png"))
        records.append(dict(raw=raw, actual_native_joint_render=True, pixel_domain="native shader sRGB", RGB_training=raw in TRAIN))
    write(path / "receipt.json", dict(schema="scene065.teacher_gaussian.actual_readout.v1", logical_step=logical_step,
        rows=records, train_sequence=list(range(106, 111)), heldout_render_only_sequence=list(range(181, 186)), point_count=rows(rt)))


def topology_counts(event, before, after):
    if event is None:
        assert before == after
        return None
    assert event["rows_before"] == before and event["rows_after"] == after
    for key in ["birth", "clone", "split", "prune"]:
        assert int(event[key]) == event[key] and event[key] >= 0
    assert after == before + event["birth"] + event["clone"] + event["split"] - event["prune"]
    return event


def checked_step(rt, optimizer, sampler, cache, cfg, objective, logical_step):
    before = rows(rt); sync(rt); started = time.perf_counter()
    record, event = objective.step(rt, optimizer, sampler, cache, cfg, logical_step)
    sync(rt); seconds = time.perf_counter() - started
    assert len(record["raws"]) == cfg["views_per_optimizer_step"] and set(record["raws"]) <= set(TRAIN)
    assert np.isfinite(record["logical_loss"])
    assert set(record["gradients_by_group"]) == set(record["parameter_delta_by_group"]) == set(NAMES)
    for name in NAMES:
        gradient, delta = record["gradients_by_group"][name], record["parameter_delta_by_group"][name]
        assert gradient["finite"] and np.isfinite(gradient["l2"])
        assert np.isfinite(delta["l2"]) and np.isfinite(delta["max_abs"])
    assert not getattr(rt, "_render_records", []), "Unconsumed screen gradients at update boundary"
    return record, topology_counts(event, before, rows(rt)), seconds


def instrument(optimizer, calls):
    if getattr(optimizer, "_r5_driver_observed", False):
        return
    original = optimizer.step
    def observed(*args, **kwargs):
        calls["attempted"] += 1
        result = original(*args, **kwargs)
        calls["completed"] += 1
        return result
    optimizer.step = observed
    optimizer._r5_driver_observed = True


def load_checkpoint(t, path):
    result = t.load(path, map_location="cpu", weights_only=False)
    assert result["schema"] == SCHEMA and result.get("recovery_safe", True)
    return result


def compare_replay(life, expected, actual, expected_record, record, expected_event, event, cfg):
    tolerance = 4 * np.finfo(np.float32).eps
    result = dict(full_state=life.equal(expected, actual, rtol=tolerance, atol=tolerance),
        views_exact=expected_record["raws"] == record["raws"], event_exact=life.equal(expected_event, event),
        loss_abs=abs(float(expected_record["logical_loss"]) - float(record["logical_loss"])),
        rtol=float(tolerance), atol=float(tolerance), loss_atol=1e-7)
    result["passed"] = bool(result["full_state"] and result["views_exact"] and result["event_exact"] and result["loss_abs"] <= 1e-7)
    assert result["passed"], result
    return result


def check_long_run(evidence, smoke_receipt):
    probe = read(Path(evidence) / "probe/receipt.json")
    projected = float(probe["projected_unique_training_seconds"])
    if projected < 3600:
        return
    assert smoke_receipt is not None, "Probe projects >=1h: complete approximately 10 minutes of real GPU smoke before formal training"
    receipt = read(smoke_receipt)
    assert float(receipt["active_training_seconds"]) >= 600
    assert receipt["real_GPU"] and receipt["finite_forward_backward"]
    assert receipt["checkpoint_save_pass"] and receipt["evaluation_pass"] and receipt["complete_restore_next_step_pass"] and receipt["next_stage_entry_pass"]
    assert receipt["bindings"] == probe["bindings"]


def run(a):
    cfg, rc, ex, target_path, mask_root, base_binds = preflight(a)
    output = a.output.resolve(); evidence = (a.evidence_output or a.output).resolve()
    cache_root = (a.cache_root or (HERE.parent / "fixed_training_cache_v1")).resolve()
    if a.validate_only:
        print(json.dumps(dict(CPU_preflight_pass=True, GPU_started=False, bindings=base_binds, cache_root=str(cache_root))))
        return
    assert not (output / a.phase).exists(), "Keep previous phase evidence; use a fresh --output for continuation"
    if a.phase == "probe":
        assert a.resume_checkpoint is None, "Probe restarts fresh; --resume-checkpoint continues an already verified formal train"
        assert not (output / "zero").exists()
    else:
        assert read(evidence / "probe/receipt.json")["complete"]
        check_long_run(evidence, a.long_smoke_receipt)
    import torch as t
    assert t.cuda.is_available(), "This driver requires the actual native GPU runtime"
    runtime_module = module(a.runtime_module, "_r5_full_native_runtime")
    objective = module(a.objective_module, "_r5_shared_training_objective")
    life = module(a.lifecycle_module, "_r5_full_gaussian_lifecycle")
    seed_all(t, cfg["seed"])
    output.mkdir(parents=True, exist_ok=True); stage = output / a.phase; stage.mkdir(exist_ok=False)
    origin = dict(phase=a.phase, pid=os.getpid(), host=socket.gethostname(), nonce=uuid.uuid4().hex, started_UTC=datetime.now(timezone.utc).isoformat())
    flag = {"stop": False}; previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: flag.__setitem__("stop", True))
    rt = optimizer = sampler = ledger = None
    records, curve = [], []
    logical = 0; phase_unique = 0; phase_replays = 0; active = 0.0; replay_seconds = 0.0
    cache_seconds = evaluation_seconds = readout_seconds = 0.0
    error = None; interrupted = False; replay = None; binds = base_binds; started = time.perf_counter()
    calls = dict(attempted=0, completed=0)

    def envelope(step, record=None, event=None):
        return dict(schema=SCHEMA, recovery_safe=True, complete_state=life.snapshot(rt, optimizer, sampler, step, binds),
            execution_ledger=ledger.state(), unique_training_seconds=active, origin=copy.deepcopy(origin),
            replay_verification=copy.deepcopy(replay), record=life.cpu(record), event=life.cpu(event),
            bindings=copy.deepcopy(binds))

    def save(path, payload):
        temporary = Path(path).with_suffix(".tmp"); t.save(payload, temporary); temporary.replace(path)

    def monitor(step):
        nonlocal evaluation_seconds
        tick = time.perf_counter()
        result = inference(rt, life, lambda: objective.evaluate(rt, cache, cfg["fixed_eval_frames"]))
        evaluation_seconds += time.perf_counter() - tick
        assert {int(r["raw"]) for r in result} == set(TRAIN)
        curve.append(dict(step=step, rows=result))

    def render(step, folder):
        nonlocal readout_seconds
        tick = time.perf_counter(); inference(rt, life, lambda: readout(rt, folder, step, cfg['readout_frames'])); readout_seconds += time.perf_counter() - tick

    try:
        rt = runtime_module.TeacherGaussianRuntime(a.runtime_contract, stage / "native")
        assert tuple(rt.named_new_parameters()) == NAMES
        tick = time.perf_counter()
        manifest_path = cache_root / "manifest.json"
        cache_built = not manifest_path.is_file()
        if not manifest_path.is_file():
            assert a.phase == "probe", "Formal continuation must reuse the original fixed cache"
            built = objective.build_cache(rt, target_path, mask_root, cache_root, cfg)
            assert built["complete"] and Path(built["manifest"]).resolve() == manifest_path
        cache = objective.load_cache(t, cache_root)
        cache_seconds = time.perf_counter() - tick
        cache_pin = pin(manifest_path)
        binds = dict(base_binds, cache_manifest=cache_pin)
        rt.frozen_assets["cache_manifest"] = cache_pin["sha256"]
        binds["frozen_assets"] = copy.deepcopy(rt.frozen_assets)
        write(stage / "bindings.json", binds)
        write(stage / "fixed_cache_receipt.json", dict(cache_manifest=cache_pin, built_in_this_process=cache_built))
        optimizer = rt.make_optimizer(cfg); instrument(optimizer, calls)
        sampler = life.BalancedSampler(TRAIN, cfg["seed"], GROUPS)
        ledger = life.OptimizerEventLedger(unique_budget=2800)
        if a.phase == "probe":
            zero = output / "zero"; zero.mkdir(exist_ok=False)
            initial = life.cpu(rt.capture_model_state())
            tick = time.perf_counter(); zero_rows = inference(rt, life, lambda: objective.evaluate(rt, cache, cfg["fixed_eval_frames"])); evaluation_seconds += time.perf_counter() - tick
            write(zero / "fixed_eval_curve.json", [dict(step=0, rows=zero_rows)])
            render(0, zero / "actual_step000000")
            assert life.equal(initial, rt.capture_model_state()); rt.assert_fixed()
            rt.export_field(zero / "new_field_initial.npz")
            save(zero / "checkpoint000000.pt", envelope(0))
            write(zero / "proof.json", dict(initial_field_unchanged=True, original_native30k_frozen=True,
                point_count=rows(rt), new_parameter_groups=list(NAMES), cache_manifest=cache_pin, optimizer_updates=0))
            write(zero / "receipt.json", dict(schema="scene065.teacher_gaussian.zero.v1", complete=True, bindings=binds,
                logical_step=0, unique_optimizer_updates=0, actual_optimizer_updates=0, point_count=rows(rt),
                actual_native_forward_calls=rt.calls, actual_native_backward_calls=rt.backward_calls, human_approval_changed=False))
            first, end = 1, 32
        elif a.resume_checkpoint is not None:
            source = load_checkpoint(t, a.resume_checkpoint); logical = int(source["complete_state"]["logical_step"])
            assert 32 <= logical < 2800 and source["replay_verification"]["passed"]
            assert source["origin"]["nonce"] != origin["nonce"] and source["origin"]["pid"] != origin["pid"]
            optimizer = life.restore(rt, optimizer, sampler, source["complete_state"], binds); instrument(optimizer, calls)
            ledger = life.OptimizerEventLedger(source["execution_ledger"], unique_budget=2800); active = float(source["unique_training_seconds"])
            replay = source["replay_verification"]; first, end = logical + 1, 2800
            write(stage / "resume.json", dict(source=pin(a.resume_checkpoint), restored_complete_state=True, next_unique_update=first))
        else:
            zero = read(evidence / "zero/receipt.json"); assert life.equal(zero["bindings"], binds)
            source = load_checkpoint(t, evidence / "probe/checkpoint31.pt")
            expected = load_checkpoint(t, evidence / "probe/expected32.pt")
            assert source["complete_state"]["logical_step"] == 31 and expected["complete_state"]["logical_step"] == 32
            assert source["origin"]["nonce"] != origin["nonce"] and source["origin"]["pid"] != origin["pid"]
            optimizer = life.restore(rt, optimizer, sampler, source["complete_state"], binds); instrument(optimizer, calls)
            record, event, replay_seconds = checked_step(rt, optimizer, sampler, cache, cfg, objective, 32)
            actual = life.snapshot(rt, optimizer, sampler, 32, binds)
            replay = compare_replay(life, expected["complete_state"], actual, expected["record"], record, expected["event"], event, cfg)
            replay.update(fresh_process=True, source=pin(evidence / "probe/checkpoint31.pt"), expected=pin(evidence / "probe/expected32.pt"))
            ledger = life.OptimizerEventLedger(expected["execution_ledger"], unique_budget=2800); ledger.record_update(32, replay=True, topology=event)
            logical = 32; phase_replays = 1; active = float(expected["unique_training_seconds"]); first, end = 33, 2800
            write(stage / "step32_replay.json", replay)
            save(stage / "checkpoint000032.pt", envelope(32, record, event))
        for step in range(first, end + 1):
            if flag["stop"]:
                interrupted = True; break
            record, event, seconds = checked_step(rt, optimizer, sampler, cache, cfg, objective, step)
            logical = step; phase_unique += 1; active += seconds; ledger.record_update(step, topology=event)
            records.append(dict(step=step, active_seconds=seconds, record=record, event=event))
            if step == 31 and a.phase == "probe":
                save(stage / "checkpoint31.pt", envelope(step, record, event))
            if step == 32 and a.phase == "probe":
                payload = envelope(step, record, event); save(stage / "expected32.pt", payload); save(stage / "checkpoint000032.pt", payload)
            stopped = flag["stop"]; is_end = step == end or stopped
            checkpoint = step % 100 == 0 or is_end
            if checkpoint and step != 32:
                save(stage / f"checkpoint{step:06d}.pt", envelope(step, record, event))
            if step % 100 == 0 or is_end:
                monitor(step)
            if step in cfg.get('readout_at', [32, 400, 1000, 2800]):
                render(step, stage / f"actual_step{step:06d}")
            if checkpoint:
                rt.export_field(stage / f"new_field_step{step:06d}.npz")
                write(stage / "losses.json", records); write(stage / "fixed_eval_curve.json", curve)
            if step in [1, 16, 32] or checkpoint:
                print(json.dumps(dict(phase=a.phase, unique_updates=logical, actual_updates=ledger.state()["actual_optimizer_updates"],
                    replay_updates=ledger.state()["engineering_replay_updates"], point_count=rows(rt), logical_loss=record["logical_loss"], topology=event)), flush=True)
            if stopped:
                interrupted = True; break
        interrupted = interrupted or flag["stop"]
        save(stage / "checkpoint_final.pt", envelope(logical))
        save(stage / "model_final.pt", dict(schema="scene065.teacher_gaussian.inference_asset.v1",
            model=life.cpu(rt.capture_model_state()), bindings=copy.deepcopy(binds),
            frozen_assets=copy.deepcopy(rt.frozen_assets), logical_step=logical))
        rt.export_field(stage / "new_field_final.npz"); rt.assert_fixed()
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        interrupted = interrupted or flag["stop"]
        write(stage / "losses.json", records); write(stage / "fixed_eval_curve.json", curve)
        state = ledger.state() if ledger is not None else None
        gradient_totals = {k: sum(int(r["record"]["gradients_by_group"][k].get("nonzero_count", 0)) for r in records) for k in NAMES}
        delta_totals = {k: sum(int(r["record"]["parameter_delta_by_group"][k].get("nonzero_count", 0)) for r in records) for k in NAMES}
        receipt = dict(schema="scene065.teacher_gaussian.phase.v1", phase=a.phase, complete=error is None and not interrupted,
            interrupted_at_optimizer_boundary=interrupted, error=error, origin=origin, bindings=binds,
            logical_step=logical, unique_optimizer_updates=logical, this_phase_unique_updates=phase_unique,
            this_phase_engineering_replay_updates=phase_replays, this_phase_actual_optimizer_updates=calls["completed"],
            optimizer_step_attempts=calls["attempted"], failed_or_uncommitted_optimizer_updates=max(0, calls["completed"] - phase_unique - phase_replays),
            execution_ledger=state, unique_training_seconds=active, engineering_replay_seconds=replay_seconds,
            projected_unique_training_seconds=active / logical * 2800 if logical else 0,
            cache_seconds=cache_seconds, evaluation_seconds=evaluation_seconds, readout_seconds=readout_seconds, wall_seconds=time.perf_counter() - started,
            final_point_count=rows(rt) if rt is not None else None,
            actual_native_forward_calls=rt.calls if rt is not None else 0, actual_native_backward_calls=rt.backward_calls if rt is not None else 0,
            gradient_nonzero_totals_by_group=gradient_totals, parameter_changed_value_totals_by_group=delta_totals,
            all_five_groups_have_actual_changes=all(v > 0 for v in delta_totals.values()),
            replay_verification=replay, old_r4_checkpoint_loaded=False, initial_from_engineering_step1=False,
            training_frames=TRAIN, heldout_render_only_frames=cfg["evaluation_frames"], human_approval_changed=False, final_visual_quality_pass=False)
        write(stage / "receipt.json", receipt); print(json.dumps(receipt, ensure_ascii=False), flush=True)
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "train"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--teacher-targets", "--targets", dest="teacher_targets", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--runtime-module", type=Path, default=HERE / "teacher_gaussian_runtime.py")
    parser.add_argument("--objective-module", type=Path, default=HERE / "training_objective.py")
    parser.add_argument("--lifecycle-module", type=Path, default=HERE / "full_gaussian_lifecycle.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--long-smoke-receipt", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())
