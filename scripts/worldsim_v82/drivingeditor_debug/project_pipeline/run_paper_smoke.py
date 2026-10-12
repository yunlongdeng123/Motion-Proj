#!/usr/bin/env python3
"""Exercise paper-method code numerically; this short run is not a paper result."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import threading
import subprocess
import traceback
import faulthandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--paper-config", required=True)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--including-dynamic", action="store_true")
    parser.add_argument("--resident-sampler", action="store_true",
                        help="Opt in to threaded nvidia-smi sampling; off for stable CUDA initialization")
    cli = parser.parse_args()
    repository = cli.repository.resolve()
    output = cli.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to reuse a nonempty training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    import torch
    from omegaconf import OmegaConf
    spec = importlib.util.spec_from_file_location("drivingeditor_official_train", repository / "train.py")
    training_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_module)
    args = OmegaConf.merge(OmegaConf.load("configs/base.yaml"),
                           OmegaConf.load("configs/waymo_reconstruction.yaml"),
                           OmegaConf.load(cli.paper_config),
                           OmegaConf.create({"source_path": str(cli.scene.resolve()),
                                             "model_path": str(output),
                                             "iterations": cli.iterations,
                                             "including_dynamic": cli.including_dynamic}))
    if cli.start_checkpoint:
        args.start_checkpoint = str(cli.start_checkpoint.resolve())
    # Reproduce the official train.py __main__ schedule setup.
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    if args.exhaust_test:
        args.test_iterations += [i for i in range(0, args.iterations, args.test_interval)]
    print(OmegaConf.to_yaml(args), flush=True)
    OmegaConf.save(args, output / "smoke_effective_config.yaml")
    torch.cuda.set_device(torch.device("cuda:0"))
    training_module.seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats()
    evaluations = []
    step_statistics = []
    original_evaluate = training_module.complete_eval

    def timed_evaluate(*positional, **keyword):
        iteration = positional[1]
        values = {key: float(value.item()) if torch.is_tensor(value) else float(value)
                  for key, value in positional[6].items()}
        step_statistics.append({"iteration": iteration, **values})
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Non-finite training diagnostic at iteration {iteration}: {values}")
        scheduled = iteration in positional[2]
        if scheduled:
            torch.cuda.synchronize()
        begin = time.perf_counter()
        result = original_evaluate(*positional, **keyword)
        if scheduled:
            torch.cuda.synchronize()
            evaluations.append({"iteration": iteration, "seconds": time.perf_counter() - begin})
        return result

    training_module.complete_eval = timed_evaluate
    restore_records = []
    original_static_restore = training_module.GaussianModel.restore

    def optimizer_steps(optimizer):
        return sorted({float(state["step"].item()) if torch.is_tensor(state["step"]) else float(state["step"])
                       for state in optimizer.state.values() if "step" in state})

    def audited_static_restore(model, model_args, training_args=None):
        result = original_static_restore(model, model_args, training_args)
        restore_records.append({"component": "static", "loaded_xyz_exact": bool(torch.equal(model._xyz, model_args[1])),
                                "points": len(model._xyz), "optimizer_steps_after_restore": optimizer_steps(model.optimizer)})
        return result

    original_env_restore = training_module.EnvLight.restore

    def audited_env_restore(model, model_args, training_args=None):
        result = original_env_restore(model, model_args, training_args)
        restore_records.append({"component": "environment", "loaded_base_exact": bool(torch.equal(model.base, model_args[0])),
                                "optimizer_binds_loaded_base": any(model.base is p for group in model.optimizer.param_groups for p in group["params"]),
                                "optimizer_steps_after_restore": optimizer_steps(model.optimizer)})
        return result

    training_module.GaussianModel.restore = audited_static_restore
    training_module.EnvLight.restore = audited_env_restore
    original_dynamic_restore = training_module.Dynamic_GaussianModel.restore

    def audited_dynamic_restore(model, model_args, training_args=None):
        result = original_dynamic_restore(model, model_args, training_args)
        restore_records.append({"component": "dynamic", "loaded_xyz_exact": bool(torch.equal(model._xyz, model_args[1])),
                                "points": len(model._xyz), "optimizer_steps_after_restore": optimizer_steps(model.optimizer)})
        return result

    training_module.Dynamic_GaussianModel.restore = audited_dynamic_restore
    resident_samples = []
    stop_sampler = threading.Event()

    def sample_resident_memory():
        while not stop_sampler.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                for line in result.stdout.splitlines():
                    pid, memory = [value.strip() for value in line.split(",", 1)]
                    if int(pid) == os.getpid():
                        resident_samples.append({"monotonic_seconds": time.perf_counter(), "mib": float(memory)})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            stop_sampler.wait(0.25)

    sampler = threading.Thread(target=sample_resident_memory, daemon=True)
    if cli.resident_sampler:
        sampler.start()
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    begin = time.perf_counter()
    success = False
    try:
        training_module.training(args)
        torch.cuda.synchronize()
        success = True
    finally:
        elapsed = time.perf_counter() - begin
        faulthandler.cancel_dump_traceback_later()
        stop_sampler.set()
        if cli.resident_sampler:
            sampler.join(timeout=6)
        eval_seconds = sum(e["seconds"] for e in evaluations)
        telemetry = {"success": success, "diagnostic_only": True, "iterations": cli.iterations,
                     "paper_config": cli.paper_config,
                     "scope": "Method integration smoke; shortened LR horizon and demo inputs, not Table I.",
                     "source_path": str(cli.scene.resolve()), "model_path": str(output),
                     "start_checkpoint": args.start_checkpoint,
                     "wall_seconds_including_setup_and_eval": elapsed,
                     "evaluation_seconds": eval_seconds,
                     "training_setup_and_saving_seconds_excluding_eval": elapsed - eval_seconds,
                     "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                     "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
                     "nvidia_smi_sampled_process_peak_mib": max((s["mib"] for s in resident_samples), default=None),
                     "nvidia_smi_sample_count": len(resident_samples),
                     "nvidia_smi_sampling_interval_seconds": 0.25,
                     "evaluations": evaluations,
                     "restore_checks": restore_records,
                     "steps_executed": len(step_statistics),
                     "first_step": step_statistics[0] if step_statistics else None,
                     "last_step": step_statistics[-1] if step_statistics else None,
                     "all_step_statistics_finite": all(math.isfinite(value) for row in step_statistics for value in row.values()),
                     "deviations_from_official_reconstruction_config": {"iterations": cli.iterations,
                         "including_dynamic": cli.including_dynamic, "scene": cli.scene.name}}
        (output / "smoke_telemetry.json").write_text(json.dumps(telemetry, indent=2))
        (output / "smoke_step_statistics.json").write_text(json.dumps(step_statistics, indent=2))
        print("SMOKE_TELEMETRY " + json.dumps(telemetry), flush=True)
    if not success:
        raise RuntimeError("Training did not complete")


if __name__ == "__main__":
    main()
