#!/usr/bin/env python3
"""Run the first declared paper-method pilot from step zero with an auditable 50k horizon."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(cli):
    from omegaconf import OmegaConf
    from PIL import Image
    repo, scene, output = cli.repository.resolve(), cli.scene.resolve(), cli.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse a nonempty run: {output}")
    if shutil.disk_usage(output.parent).free < 18 * 1024 ** 3:
        raise RuntimeError("Less than18GiB free for five full checkpoints and diagnostic media")
    args = OmegaConf.merge(OmegaConf.load(repo / "configs/base.yaml"),
                           OmegaConf.load(repo / cli.config),
                           OmegaConf.create({"source_path": str(scene), "model_path": str(output)}))
    expected = [10000, 20000, 30000, 40000, 50000]
    assert args.iterations == args.position_lr_max_steps == 50000
    for key in ("checkpoint_iterations", "test_iterations", "save_iterations"):
        assert list(args[key]) == expected, (key, args[key])
    assert args.paper_method and args.including_dynamic and args.paper_eval_train_each_checkpoint
    assert not args.start_checkpoint, "This pilot must start at step0 with a50k LR horizon"
    assert not args.eval, "All49 fixed reconstruction views are used, without an invented holdout split"
    frames = sorted((scene / "image/image_0").glob("*.png"))
    if not frames:
        frames = sorted((scene / "image/image_0").glob("*.jpg"))
    assert len(frames) == 49, f"Expected49 FRONT frames, got{len(frames)}"
    for frame in frames:
        with Image.open(frame) as image:
            dimensions = image.size
        for directory in ("sky/sky_0", "dynamic_masks/dynamic_mask_0"):
            mask = scene / directory / (frame.stem + ".png")
            assert mask.is_file(), mask
            with Image.open(mask) as image:
                assert image.size == dimensions, (mask, image.size, dimensions)
    assert (scene / "points3d.ply").is_file()
    assert list(scene.glob("points3d_dynamic_*.ply")), "Missing real actor LiDAR initialization"
    assert (scene / "dynamic_image/dynamic_image_0").is_dir()
    if cli.check_only:
        print(json.dumps({"status": "preflight_passed", "frames": len(frames), "iterations": 50000,
                          "checkpoint_iterations": expected, "scene": str(scene), "output": str(output)}))
        return repo, scene, output, args
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(args, output / "effective_config.yaml")
    inventory = []
    for path in sorted(scene.rglob("*")):
        if path.is_file() and not path.is_symlink():
            inventory.append({"path": str(path.relative_to(scene)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(output / "input_files_before_training.json", inventory)
    code_paths = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo).decode().split("\0")
    selected = sorted({name for name in code_paths if name and Path(name).suffix in (".py", ".yaml", ".yml", ".json", ".md", ".cu", ".cpp", ".h", ".sh")
                       and "__pycache__" not in Path(name).parts and ".codexbak" not in name and (repo / name).is_file()})
    with tarfile.open(output / "code_snapshot.tar.gz", "w:gz") as archive:
        for name in selected:
            archive.add(repo / name, arcname=name)
    (output / "source_changes.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo))
    provenance = {"started_at": now(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
                  "code_snapshot_sha256": sha256(output / "code_snapshot.tar.gz"), "config_sha256": sha256(output / "effective_config.yaml"),
                  "input_inventory_sha256": sha256(output / "input_files_before_training.json"),
                  "repository": str(repo), "scene": str(scene), "frames": [frame.name for frame in frames],
                  "total_iterations": 50000, "checkpoints": expected,
                  "protocol": "First49-frame FRONT scene, paper-method50k pilot; not the paper's four-scene mean.",
                  "deviations": ["Exact author frame windows and loss weights are undisclosed; selected0..48 and declared pilot weights.",
                                 "DEVA/SAM body masks and projected LiDAR actor separation approximate the paper preprocessing.",
                                 "Detect-AnyShadow and M-detector are not yet part of these inputs.",
                                 "RTX309024GB, scale4, seed0; paper usesA600048GB and omits resolution/split details."],
                  "comparison": "All checkpoint videos use the same49 views in source-frame order,10fps Waymo physical sampling; model frame_interval=.02 is retained separately."}
    atomic_json(output / "run_provenance.json", provenance)
    return repo, scene, output, args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="configs/paper_waymo_reconstruction.yaml")
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args()
    repo, scene, output, args = prepare(cli)
    if cli.check_only:
        return
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    spec = importlib.util.spec_from_file_location("paper50k_train", repo / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.cuda.set_device(torch.device("cuda:0"))
    module.seed_everything(args.seed)
    atomic_json(output / "runtime_environment.json", {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "seed": int(args.seed),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "dynamic_xyz_lr_initial": float(args.position_lr_init) * 10,
        "dynamic_xyz_lr_final": float(args.position_lr_final), "lr_horizon": int(args.iterations),
    })
    with (output / "pip_freeze.txt").open("w") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=handle)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    state = {"state": "initializing", "iteration": 0, "total_iterations": 50000, "started_at": now(),
             "source_path": str(scene), "model_path": str(output), "pid": os.getpid(),
             "scope": "Paper-method49-frame pilot with declared preprocessing differences; not a four-scene paper result."}

    def update(**values):
        state.update(values)
        state["updated_at"] = now()
        state["wall_seconds"] = time.perf_counter() - started
        atomic_json(output / "progress_status.json", state)

    update()
    original_eval = module.complete_eval

    def tracked_eval(*positional, **keyword):
        iteration, scheduled, diagnostics = positional[1], positional[1] in positional[2], positional[6]
        values = {key: float(value.item()) if torch.is_tensor(value) else float(value) for key, value in diagnostics.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Nonfinite training diagnostic at{iteration}: {values}")
        if iteration == 1 or iteration % 100 == 0 or scheduled:
            update(state="evaluating" if scheduled else "training", iteration=iteration,
                   loss=values.get("loss"), psnr=values.get("psnr"), total_points=values.get("total_points"),
                   cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
        result = original_eval(*positional, **keyword)
        if scheduled:
            update(state="training", iteration=iteration, last_evaluation_iteration=iteration)
        return result

    module.complete_eval = tracked_eval
    try:
        module.training(args)
        torch.cuda.synchronize()
        completed = []
        for iteration in args.checkpoint_iterations:
            event = output / "checkpoint_events" / f"iteration_{iteration:06d}.json"
            record = json.loads(event.read_text())
            assert record["status"] == "complete", (event, record.get("status"))
            for prefix in ("chkpnt", "dynamic_chkpnt", "env_light_chkpnt"):
                assert (output / f"{prefix}{iteration}.pth").stat().st_size > 1024
            completed.append(iteration)
        update(state="completed", iteration=50000, completed_checkpoints=completed,
               cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
        print("PAPER50K_COMPLETED " + json.dumps(state), flush=True)
    except BaseException as error:
        update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
