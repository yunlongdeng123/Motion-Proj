"""Atomic paper checkpoint bundles and explicit continuation state.

Component .pth files retain the upstream (capture, iteration) format. A complete
event is published only after all components, continuation state, and evaluation
files exist. This supports external CPU video/HTML consumers without training
subprocesses. Exact RNG/sampler restoration is tested; CUDA bitwise determinism
is not promised.
"""
import hashlib
import json
import math
import os
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(payload))


def camera_id(camera):
    return os.path.basename(os.path.dirname(camera.image_path)) + "/" + camera.image_name


def camera_record(camera):
    return {"uid": int(camera.colmap_id), "frame_id": camera_id(camera),
            "camera": os.path.basename(os.path.dirname(camera.image_path)),
            "image_name": camera.image_name, "timestamp": float(camera.timestamp),
            "source_image": os.path.abspath(camera.image_path),
            "width": int(camera.image_width), "height": int(camera.image_height)}


def paper_evaluation_configs(scene):
    """A fixed, chronological per-camera set at every requested paper node."""
    scale = scene.resolution_scales[scene.scale_index]
    order = lambda camera: (camera_id(camera).split("/")[0], float(camera.timestamp), camera.image_name)
    return ({"name": "test", "cameras": sorted(scene.getTestCameras(scale=scale), key=order)},
            {"name": "train", "cameras": sorted(scene.getTrainCameras(), key=order)})


def _config_dict(args):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else dict(args)


def capture_training_state(iteration, scene, viewpoint_stack, models, args, sgcm=None, ema=None):
    camera_orders = [{"scale": float(scale), "ids": [camera_id(c) for c in cameras]}
                     for scale, cameras in scene.train_cameras.items()]
    state = {"schema_version": 1, "iteration": int(iteration),
             "config": _config_dict(args), "scale_index": int(scene.scale_index),
             "camera_orders": camera_orders, "viewpoint_stack": list(viewpoint_stack or []),
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
             "model_runtime": {name: {"no_time_split": bool(model.no_time_split)}
                               for name, model in models.items() if model is not None and hasattr(model, "no_time_split")},
             "ema": {key: float(value) for key, value in (ema or {}).items()},
             "sgcm": None}
    if hasattr(scene, "hybrid_sampler"):
        state["hybrid_sampler"] = scene.hybrid_sampler.capture()
    if sgcm is not None:
        from dataclasses import asdict
        state["sgcm"] = {"config": asdict(sgcm.config), "metadata": sgcm.metadata,
                         "model_scale": sgcm.model_scale, "plane_model": sgcm.plane_model,
                         "model_to_anchor": sgcm.model_to_anchor, "regions_anchor": sgcm.regions_anchor}
    return state


def _validate_resume_configuration(saved_config, args):
    """Fail closed before restoring continuation state under a different protocol.

    The resume checkpoint selector and output directory are bookkeeping only.
    All other resolved fields must match, including evaluation cadence because
    evaluation may initialize components that consume random numbers.
    """
    current = _config_dict(args)
    if not isinstance(saved_config, dict) or not isinstance(current, dict):
        raise ValueError("Strict resume requires complete resolved configuration dictionaries")
    required = ("iterations", "position_lr_max_steps", "position_lr_init",
                "position_lr_final", "position_lr_delay_mult", "t_lr_init",
                "feature_lr", "opacity_lr", "scaling_lr", "scaling_t_lr",
                "velocity_lr", "rotation_lr", "envmap_lr")
    missing = [key for key in required if key not in saved_config or key not in current]
    if missing:
        raise ValueError("Strict resume requires saved and current schedule fields: " + ", ".join(missing))
    permitted = {"model_path", "start_checkpoint"}
    missing_value = object()
    changed = [key for key in sorted(set(saved_config) | set(current))
               if key not in permitted and not _same_config_value(saved_config.get(key, missing_value), current.get(key, missing_value))]
    if changed:
        details = "; ".join(f"{key}: {saved_config.get(key, '<missing>')!r} -> {current.get(key, '<missing>')!r}"
                            for key in changed)
        raise ValueError("Strict checkpoint resume rejected changed training configuration: " + details)


def _same_config_value(saved, current):
    """Types are part of the protocol: True/1 and 50000/50000.0 differ."""
    if type(saved) is not type(current):
        return False
    if isinstance(saved, dict):
        return set(saved) == set(current) and all(_same_config_value(saved[key], current[key]) for key in saved)
    if isinstance(saved, (list, tuple)):
        return len(saved) == len(current) and all(_same_config_value(a, b) for a, b in zip(saved, current))
    return saved == current


def _validate_continuation(state, iteration, scene, models, args, sgcm):
    required = {"schema_version", "iteration", "config", "scale_index", "camera_orders", "viewpoint_stack",
                "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "model_runtime", "ema", "sgcm"}
    if args.get("hybrid_method", False):
        required.add("hybrid_sampler")
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("Strict resume requires exactly the complete continuation schema")
    if type(state["schema_version"]) is not int or state["schema_version"] != 1 or type(state["iteration"]) is not int or state["iteration"] != iteration:
        raise ValueError("Training-state schema or iteration does not match component checkpoint")
    _validate_resume_configuration(state["config"], args)
    if args.get("hybrid_method", False):
        from utils.hybrid30k_runtime import CameraSampler
        probe = CameraSampler(args.seed, scene.getTrainCameras())
        probe.restore(state["hybrid_sampler"])
        if probe.steps != iteration:
            raise ValueError("hybrid采样步数与检查点不同")
    if type(state["scale_index"]) is not int or not 0 <= state["scale_index"] < len(scene.resolution_scales):
        raise ValueError("Saved camera scale is unavailable")
    orders = state["camera_orders"]
    if not isinstance(orders, list) or len(orders) != len(scene.train_cameras) or any(not isinstance(order, dict) or set(order) != {"scale", "ids"} for order in orders):
        raise ValueError("Complete camera order for every resolution scale is required")
    if len({order["scale"] for order in orders}) != len(orders) or {order["scale"] for order in orders} != set(scene.train_cameras):
        raise ValueError("Resolution scales changed across checkpoint resume")
    for order in orders:
        current = scene.train_cameras[order["scale"]]
        ids = order["ids"]
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Saved camera identities are malformed or duplicated")
        actual = [camera_id(camera) for camera in current]
        if len(set(actual)) != len(actual) or set(actual) != set(ids):
            raise ValueError("Training camera identities changed across checkpoint resume")
    stack = state["viewpoint_stack"]
    active_count = len(next(order["ids"] for order in orders if order["scale"] == scene.resolution_scales[state["scale_index"]]))
    if not isinstance(stack, list) or any(type(index) is not int or not 0 <= index < active_count for index in stack) or len(set(stack)) != len(stack):
        raise ValueError("Saved camera sampler indices are invalid")
    runtime = state["model_runtime"]
    expected_runtime = {name for name, model in models.items() if model is not None and hasattr(model, "no_time_split")}
    if not isinstance(runtime, dict) or set(runtime) != expected_runtime or any(not isinstance(value, dict) or set(value) != {"no_time_split"} or type(value["no_time_split"]) is not bool for value in runtime.values()):
        raise ValueError("Complete static/dynamic runtime flags are required")
    if not isinstance(state["ema"], dict) or not state["ema"] or any(type(value) not in (float, int) or not math.isfinite(value) for value in state["ema"].values()):
        raise ValueError("Finite saved EMA diagnostics are required")
    if (state["sgcm"] is None) != (sgcm is None):
        raise ValueError("SGCM enablement changed across checkpoint resume")
    if sgcm is not None:
        from dataclasses import asdict
        geometry = state["sgcm"]
        if not isinstance(geometry, dict) or set(geometry) != {"config", "metadata", "model_scale", "plane_model", "model_to_anchor", "regions_anchor"}:
            raise ValueError("Complete SGCM continuation geometry is required")
        if not _same_config_value(geometry["config"], asdict(sgcm.config)):
            raise ValueError("SGCM configuration changed across checkpoint resume")
        if not math.isfinite(geometry["model_scale"]) or geometry["model_scale"] <= 0:
            raise ValueError("Saved SGCM scale is invalid")
        for name in ("plane_model", "model_to_anchor", "regions_anchor"):
            if not np.isfinite(np.asarray(geometry[name], dtype=float)).all():
                raise ValueError("Saved SGCM geometry is nonfinite: " + name)
        if not isinstance(geometry["metadata"], dict) or not geometry["metadata"]:
            raise ValueError("Saved SGCM metadata is missing")
    random.Random().setstate(state["python_rng"])
    np.random.RandomState().set_state(state["numpy_rng"])
    if not isinstance(state["torch_rng"], torch.Tensor) or state["torch_rng"].dtype != torch.uint8 or state["torch_rng"].device.type != "cpu":
        raise ValueError("Saved Torch CPU RNG state is invalid")
    torch.Generator(device="cpu").set_state(state["torch_rng"])
    if not isinstance(state["cuda_rng"], list) or not state["cuda_rng"] or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.device.type != "cpu" or value.ndim != 1 for value in state["cuda_rng"]):
        raise ValueError("Complete saved CUDA RNG state is required")
    if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
        raise ValueError("CUDA device count changed across checkpoint resume")


def restore_training_state(path, iteration, scene, models, args, sgcm=None):
    """Call after all initialization, immediately before the training loop."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Strict resume requires a training-state sidecar with saved configuration, RNG, and camera sampler: " + str(path))
    state = torch.load(path, map_location="cpu", weights_only=False)
    _validate_continuation(state, iteration, scene, models, args, sgcm)
    if not 0 <= state["scale_index"] < len(scene.resolution_scales):
        raise ValueError("Saved camera scale is unavailable")
    if set(item["scale"] for item in state["camera_orders"]) != set(scene.train_cameras):
        raise ValueError("Resolution scales changed across checkpoint resume")
    for order in state["camera_orders"]:
        current = scene.train_cameras[order["scale"]]
        lookup = {camera_id(camera): camera for camera in current}
        if len(lookup) != len(current) or len(set(order["ids"])) != len(order["ids"]) or set(lookup) != set(order["ids"]):
            raise ValueError("Training camera identities changed across checkpoint resume")
        scene.train_cameras[order["scale"]] = [lookup[key] for key in order["ids"]]
    scene.scale_index = state["scale_index"]
    stack = state["viewpoint_stack"]
    if len(set(stack)) != len(stack) or any(index < 0 or index >= len(scene.getTrainCameras()) for index in stack):
        raise ValueError("Saved camera sampler indices are invalid")
    for name, attributes in state["model_runtime"].items():
        if models.get(name) is None:
            raise ValueError("A saved model branch is missing on resume: " + name)
        for key, value in attributes.items():
            setattr(models[name], key, value)
    if (state["sgcm"] is None) != (sgcm is None):
        raise ValueError("SGCM enablement changed across checkpoint resume")
    if sgcm is not None:
        from dataclasses import asdict
        if state["sgcm"]["config"] != asdict(sgcm.config):
            raise ValueError("SGCM configuration changed across checkpoint resume")
        for key in ("model_scale", "plane_model", "model_to_anchor", "regions_anchor", "metadata"):
            setattr(sgcm, key, state["sgcm"][key])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed across checkpoint resume")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if args.get("hybrid_method", False):
        if "hybrid_sampler" not in state or not hasattr(scene, "hybrid_sampler"):
            raise ValueError("hybrid完整采样状态缺失，拒绝旧检查点")
        scene.hybrid_sampler.restore(state["hybrid_sampler"])
    return list(stack), state.get("ema", {})


def save_components(model_path, iteration, models):
    """Write each component atomically, but never advertise a partial bundle."""
    root = Path(model_path).resolve()
    event_path = root / "checkpoint_events" / f"iteration_{iteration:06d}.json"
    if event_path.exists():
        raise FileExistsError("Refusing to overwrite a published checkpoint event: " + str(event_path))
    names = {"static": f"chkpnt{iteration}.pth", "dynamic": f"dynamic_chkpnt{iteration}.pth",
             "environment": f"env_light_chkpnt{iteration}.pth"}
    result = {}
    for branch, filename in names.items():
        model = models.get(branch)
        if model is not None:
            destination = root / filename
            _atomic_write(destination, lambda stream, model=model: torch.save((model.capture(), iteration), stream))
            result[branch] = {"path": str(destination), "bytes": destination.stat().st_size}
    if "static" not in result:
        raise ValueError("Static model is required for a checkpoint bundle")
    return result


def publish_checkpoint_event(model_path, iteration, components, training_state, evaluations, args, scene):
    """Continuation state is captured AFTER evaluation, including any RNG use."""
    root = Path(model_path).resolve()
    if training_state["iteration"] != iteration:
        raise ValueError("Continuation-state iteration differs from checkpoint")
    expected = {"static"}
    if args.including_dynamic:
        expected.add("dynamic")
    if args.env_map_res > 0:
        expected.add("environment")
    if set(components) != expected:
        raise ValueError("Checkpoint components do not match enabled model branches")
    if args.get("paper_eval_train_each_checkpoint", False):
        train_evaluations = [item for item in evaluations if item.get("split") == "train"]
        expected_cameras = {camera_id(camera) for camera in scene.getTrainCameras()}
        if len(train_evaluations) != 1:
            raise ValueError("A full training-camera evaluation is required before checkpoint publication")
        frames = train_evaluations[0]["frames"]
        actual_cameras = [frame["frame_id"] for frame in frames]
        if len(actual_cameras) != len(expected_cameras) or set(actual_cameras) != expected_cameras:
            raise ValueError("Training evaluation is missing or duplicating camera frames")
        if any(set(frame["channels"]) != {"gt", "composite", "static", "dynamic"} for frame in frames):
            raise ValueError("Training evaluation must contain all four image channels")
    for record in components.values():
        if not Path(record["path"]).is_file() or Path(record["path"]).stat().st_size != record["bytes"] or record["bytes"] == 0:
            raise ValueError("Missing or truncated checkpoint component")
    for evaluation in evaluations:
        for key in ("metrics", "paper_metrics"):
            if evaluation.get(key) and not Path(evaluation[key]).is_file():
                raise ValueError("Evaluation metrics missing: " + evaluation[key])
        for frame in evaluation["frames"]:
            for path in frame["channels"].values():
                if not Path(path).is_file():
                    raise ValueError("Evaluation frame missing: " + path)
    state_path = root / f"training_state{iteration}.pth"
    _atomic_write(state_path, lambda stream: torch.save(training_state, stream))
    components = dict(components, training_state={"path": str(state_path), "bytes": state_path.stat().st_size})
    config = _config_dict(args)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    event = {"schema_version": 1, "status": "complete", "iteration": int(iteration),
             "model_path": str(root), "source_path": os.path.abspath(args.source_path),
             "checkpoints": components, "checkpoint_paths": {
                 "static": components["static"]["path"],
                 "dynamic": components.get("dynamic", {}).get("path"),
                 "env": components.get("environment", {}).get("path")},
             "config": config, "config_sha256": config_hash,
             "camera_manifest": [camera_record(camera) for camera in scene.getTrainCameras()],
             "evaluations": evaluations, "parameters_evaluated": "after_optimizer_and_sgcm_same_as_checkpoint",
             "resume_state": {"rng": ["python", "numpy", "torch", "cuda_if_initialized"],
                              "camera_sampler": True, "scale": True, "optimizer": True,
                              "sgcm_geometry": training_state["sgcm"] is not None,
                              "bitwise_cuda_equivalence_claimed": False},
             "sgcm": training_state["sgcm"]["metadata"] if training_state["sgcm"] is not None else None}
    train_eval = next((evaluation for evaluation in evaluations if evaluation["split"] == "train"), None)
    event.update({key: train_eval[key] if train_eval else ( [] if key == "frames" else None)
                  for key in ("eval_dir", "metrics", "paper_metrics", "frames")})
    event_path = root / "checkpoint_events" / f"iteration_{iteration:06d}.json"
    if event_path.exists():
        raise FileExistsError("Checkpoint event already published")
    atomic_json(event_path, event)
    print("CHECKPOINT_EVENT " + str(event_path), flush=True)
    return event
