"""顺序执行冻结审计推理与视频导出；不自动关机，等待审核、保存和进程检查。"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    lock = open(root / "controller.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = Path(__file__).resolve().parent
    state_path = root / "controller_state.json"
    state = {"pid": os.getpid(), "state": "running", "started_unix": time.time(),
             "shutdown_authorized_after_review_and_save": True, "human_verdict": None}

    def save():
        state["updated_unix"] = time.time()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temp.replace(state_path)

    stages = [
        ("drive", "/root/autodl-tmp/envs/driveeditor/bin/python", "drive_batch.py", ["--root", str(root)]),
        ("encode", "/root/autodl-tmp/envs/motionproj/bin/python", "encode_review.py", ["--root", str(root)]),
        ("html", "/root/autodl-tmp/envs/motionproj/bin/python", "build_audit_html.py",
         ["--report", str(root / "review/review_manifest.json"), "--output", str(root / "review/index.html")]),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for stage, python, script, options in stages:
        state["stage"] = stage
        save()
        with open(root / f"{stage}_batch.log", "a", buffering=1) as log:
            process = subprocess.Popen([python, str(scripts / script), *options],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            state["child_pid"] = process.pid
            save()
            code = process.wait()
        if code:
            state.update(state="failed_engineering", returncode=code)
            save()
            raise SystemExit(code)
    state.update(state="ready_for_assistant_one_frame_review_and_delivery", child_pid=None)
    save()


if __name__ == "__main__":
    main()
