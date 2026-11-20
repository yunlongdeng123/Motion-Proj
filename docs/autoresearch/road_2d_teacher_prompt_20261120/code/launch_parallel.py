"""四进程执行同一固定60矩阵，记录总显存与完成状态；用户要求保持实例开机。"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

import infer_next_round as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    smoke = c.read(args.smoke_receipt)
    c.need(smoke["complete"] and smoke["mode"] == "smoke" and
           smoke["completed_diffusion_calls"] == smoke["attempted_diffusion_calls"] == 4 and
           smoke["saved_receipt_and_outputs_restored"] and smoke["adapter_unchanged"] and
           smoke["checkpoint_unchanged"], "真实smoke未通过")
    c.need(all(j["historical_replay_exact"] and all(j["historical_replay_exact"].values())
               for j in smoke["jobs"] if j["historical_replay_expected"]), "smoke历史门失败")
    c.need(smoke["manifest"] == c.pin(args.manifest), "smoke与正式计划不同")
    c.need(not args.out.exists(), "保留旧正式现场")
    args.out.mkdir(parents=True)
    receipt = dict(schema="road_2d_teacher_parallel_execution.v1", complete=False,
        started_UTC=datetime.now(timezone.utc).isoformat(), manifest=c.pin(args.manifest),
        smoke_receipt=c.pin(args.smoke_receipt), smoke_diffusion_calls=4,
        formal_diffusion_budget=60, worker_count=4, shutdown_authorized=False,
        power_policy="keep_instance_running_explicit_user_instruction", resource_samples=[], workers=[])
    processes, logs = [], []
    started = time.perf_counter()
    try:
        env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")
        for index in range(4):
            logfile = (args.out / f"worker{index}.log").open("w")
            logs.append(logfile)
            command = [sys.executable, "-u", str(Path(__file__).with_name("smoke_and_workers.py")),
                "worker", "--manifest", str(args.manifest), "--out", str(args.out / f"worker{index}"),
                "--worker-index", str(index), "--worker-count", "4", "--execute-gpu"]
            proc = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, env=env)
            processes.append(proc)
            receipt["workers"].append(dict(index=index, pid=proc.pid, command=command))
        last = None
        while True:
            codes = [proc.poll() for proc in processes]
            counts = []
            for index in range(4):
                p = args.out / f"worker{index}/receipt.json"
                row = c.read(p) if p.exists() else {}
                counts.append(row.get("completed_diffusion_calls", 0))
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            receipt["resource_samples"].append(dict(seconds=time.perf_counter()-started,
                nvidia_smi=result.stdout.strip(), worker_completed=counts, return_codes=codes))
            receipt.update(worker_completed=counts, return_codes=codes, wall_seconds=time.perf_counter()-started)
            c.write(args.out / "parallel_receipt.json", receipt)
            if counts != last:
                print(c.json.dumps(dict(worker_completed=counts, total=sum(counts), GPU=result.stdout.strip())), flush=True)
                last = counts
            c.need(not any(code not in (None, 0) for code in codes), "推理分片失败；保留日志并停止本控制器的其他分片")
            if all(code == 0 for code in codes):
                break
            time.sleep(3)
        from smoke_and_workers import merge
        merge(args.manifest, [args.out / f"worker{i}/receipt.json" for i in range(4)], args.out / "predictions.json")
        receipt.update(complete=True, formal_completed_diffusion_calls=60,
            total_smoke_plus_formal_diffusion_calls=64, predictions=c.pin(args.out / "predictions.json"),
            freeze_gate=c.pin(args.out / "freeze_gate.json"), wall_seconds=time.perf_counter()-started,
            finished_UTC=datetime.now(timezone.utc).isoformat())
        c.write(args.out / "parallel_receipt.json", receipt)
        print(c.json.dumps(dict(complete=True, formal=60, smoke=4, power_policy="keep_running")), flush=True)
    except BaseException as error:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        receipt.update(complete=False, failed=True, error_type=type(error).__name__, error=str(error),
            wall_seconds=time.perf_counter()-started, return_codes=[p.returncode for p in processes])
        c.write(args.out / "parallel_receipt.json", receipt)
        raise
    finally:
        for logfile in logs:
            logfile.close()


if __name__ == "__main__":
    main()
