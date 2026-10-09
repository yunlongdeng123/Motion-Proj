"""有界 P0 队列：等待输入，真实优化、断点恢复、独立场景推理。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_state(path: Path, **values) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "time": time.time(), **values},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--svd", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=6)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    state = run / "controller_state.json"
    if state.exists():
        previous = json.loads(state.read_text())
        pid = previous.get("pid", -1)
        if previous.get("phase") not in ("complete", "failed", "input_timeout") and Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"已有 P0 控制器运行，PID={pid}")
    model_files = [args.svd / sub / filename for sub, filename in (
        ("vae", "diffusion_pytorch_model.fp16.safetensors"),
        ("unet", "diffusion_pytorch_model.fp16.safetensors"),
        ("image_encoder", "model.fp16.safetensors"))]
    started = time.monotonic()
    try:
        while True:
            missing = [str(path) for path in model_files if not path.is_file()]
            prep_path = run / "data" / "preparation_status.json"
            ready = False
            if prep_path.exists():
                prep = json.loads(prep_path.read_text())
                ready = len(prep.get("ready", [])) == len(prep.get("selected", [])) == 8
            if not missing and ready:
                break
            write_state(state, phase="waiting_inputs", missing_weights=missing,
                        data_ready=ready, elapsed_seconds=time.monotonic() - started)
            if time.monotonic() - started > args.wait_hours * 3600:
                write_state(state, phase="input_timeout", missing_weights=missing, data_ready=ready)
                return
            time.sleep(30)
        # 准备程序可能渐进写清单；训练前冻结副本，恢复时不改变样本池。
        fixed = run / "fixed_inputs"
        fixed.mkdir(exist_ok=True)
        for name in ("train.jsonl", "val.jsonl", "train_metadata.json", "val_metadata.json"):
            (fixed / name).write_bytes((run / "data" / name).read_bytes())
        train_manifest, val_manifest = fixed / "train.jsonl", fixed / "val.jsonl"
        train_output = run / "train"
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")

        def execute(phase: str, command: list[str]) -> None:
            write_state(state, phase=phase, command=command)
            with (run / f"{phase}.log").open("a", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        base = [sys.executable, "-u", "-m", "motion_proj.worldsim_v81.train",
                "--manifest", str(train_manifest), "--output-dir", str(train_output),
                "--svd", str(args.svd), "--seed", "8101", "--amp", "bf16"]
        first, second = train_output / "checkpoint-000001.pt", train_output / "checkpoint-000002.pt"
        if not first.exists():
            execute("train_step1", base + ["--max-steps", "1"])
        if not second.exists():
            execute("resume_step2", base + ["--max-steps", "2", "--resume", str(first)])
        execute("infer_val", [sys.executable, "-u", "-m", "motion_proj.worldsim_v81.infer",
                              "--manifest", str(val_manifest), "--checkpoint", str(second),
                              "--svd", str(args.svd), "--output-dir", str(run / "inference"),
                              "--steps", "25", "--seed", "8101"])
        execute("build_review", [sys.executable, "scripts/worldsim_v81/build_p0_review.py",
                                 "--run-dir", str(run)])
        write_state(state, phase="complete", checkpoint=str(second),
                    elapsed_seconds=time.monotonic() - started,
                    protocol="nuScenes P0 / all-frame propagation / Euler smoke; not P1 paper reproduction")
    except Exception as error:
        write_state(state, phase="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
