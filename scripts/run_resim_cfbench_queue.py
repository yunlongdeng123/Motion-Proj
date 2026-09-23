"""单卡串行ReSim固定case；等待OmniDreams退出，只跑6个原生ego任务。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import av
from add_cfbench_original_videos import make_original


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")


def trim_video(source, output):
    count = 0
    with av.open(str(source)) as reader, av.open(str(output), "w") as writer:
        stream = writer.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 896, 512, "yuv420p"
        stream.options = {"crf": "18"}
        for i, frame in enumerate(reader.decode(video=0)):
            count += 1
            if 4 <= i <= 27:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")):
                    writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    assert count == 49, count
    with av.open(str(output)) as reader:
        assert sum(1 for _ in reader.decode(video=0)) == 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-omnidreams", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.inputs / "index.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"task_id": "WS-V75-DOWNSTREAM-FULL-01", "run_id": "20260923-r1", "model_id": "resim",
             "status": "waiting_for_omnidreams", "requested": 24, "supported": 6, "unsupported": 18,
             "completed_pairs": 0, "completed_branches": 0, "human_verdict": None,
             "failure_ledger_refs": [], "failure_ledger_delta": "none", "cases": index["cases"]}
    save(args.output / "queue-result.json", state)
    gate = args.wait_for_omnidreams / "generate-result.json"
    while not gate.exists():
        time.sleep(10)
    assert json.loads(gate.read_text())["status"] == "complete", "OmniDreams失败，不自行启动后续任务"
    state["status"] = "running"
    save(args.output / "queue-result.json", state)
    root = Path("/root/autodl-tmp/external/worldsim_v75_downstream_bench/ReSim")
    env = os.environ.copy()
    env.update(WORLD_SIZE="1", RANK="0", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    began = time.monotonic()
    try:
        for row in state["cases"]:
            if row["status"] == "unsupported":
                continue
            out = args.output / row["case_id"]
            out.mkdir()
            save(out / "input.json", row)
            make_original(row["case"], row.get("camera_index", 0), out)
            for branch in ["factual", "counterfactual"]:
                cfg = Path(row["configs"][branch])
                before = set((root / "outputs").glob(f"{cfg.stem}-*/*/Sample*.mp4"))
                cmd = ["/root/autodl-tmp/envs/resim/bin/python", "-u", str(Path(__file__).with_name("run_resim_lowmem.py")),
                       "--vae-chunk-frames", "17", "--resim-root", str(root), "--config", str(cfg), "--result", str(out / f"{branch}-result.json")]
                with (out / f"{branch}.log").open("w") as log:
                    run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                assert run.returncode == 0, f"ReSim失败: {row['case_id']} {branch}; 看原始日志"
                result = json.loads((out / f"{branch}-result.json").read_text())
                assert result["status"] == "complete"
                after = set((root / "outputs").glob(f"{cfg.stem}-*/*/Sample*.mp4"))
                created = sorted(after-before)
                assert len(created) == 1, [str(p) for p in created]
                shutil.copy2(created[0], out / f"{branch}-native49.mp4")
                trim_video(created[0], out / f"{branch}.mp4")
                state["completed_branches"] += 1
                state["active_case"] = row["case_id"]
                save(args.output / "queue-result.json", state)
                print(json.dumps({"case_id": row["case_id"], "branch": branch, "status": "complete"}), flush=True)
            row["status"] = "generation_complete"
            state["completed_pairs"] += 1
            save(out / "result.json", {"status": "generation_complete", "case_id": row["case_id"], "frames_per_branch": 24,
                "native_frames_per_branch": 49, "seed": 42, "input_variant": "history_only_padded_future_chunk17_offload", "human_verdict": None})
            save(args.output / "queue-result.json", state)
        state["status"] = "supported_generation_complete"
    except BaseException as exc:
        state.update(status="failed_stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        state["wall_s"] = time.monotonic()-began
        save(args.output / "queue-result.json", state)


if __name__ == "__main__":
    main()
