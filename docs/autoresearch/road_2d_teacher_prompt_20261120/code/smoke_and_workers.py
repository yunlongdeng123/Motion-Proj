"""真实4次smoke、固定矩阵分片与合并；不改实验方法，不执行电源操作。"""
import argparse
from pathlib import Path
import time

import infer_next_round as c


def execute(plan_path, output, mode, worker_index=0, worker_count=4):
    m = c.plan_check(plan_path)
    if mode == "smoke":
        selected = [j for j in m["jobs"] if j["case_id"] in ("scene065_raw086", "scene065_raw110") and j["seed"] == 101]
        c.need(len(selected) == 4, "smoke固定2case×2prompt")
    else:
        c.need(0 <= worker_index < worker_count <= 4, "当前并发上限4")
        selected = [j for index, j in enumerate(m["jobs"]) if index % worker_count == worker_index]
        c.need(selected and selected[0]["historical_replay_expected"], "每个分片先重放历史1152")
    output = Path(output)
    c.need(not output.exists(), "保留已有现场，请使用新目录")
    output.mkdir(parents=True)
    receipt = dict(schema="road_2d_teacher_prompt_round_worker.v1", mode=mode, complete=False,
        manifest=c.pin(plan_path), worker_index=worker_index, worker_count=worker_count,
        selected_job_ids=[j["id"] for j in selected], attempted_diffusion_calls=0,
        completed_diffusion_calls=0, GT_opened=False, training_updates=0, jobs=[],
        execution_sources={name: c.pin(Path(__file__).parent / name) for name in
                           ("infer_next_round.py", "teacher_engine.py", "smoke_and_workers.py")})
    c.write(output / "receipt.json", receipt)
    started = time.perf_counter()
    try:
        from teacher_engine import TeacherEngine
        engine = TeacherEngine(plan_path)
        for job in selected:
            receipt["attempted_diffusion_calls"] += 1
            receipt["active_job"] = job["id"]
            c.write(output / "receipt.json", receipt)
            result = engine.predict(job, output / job["id"])
            receipt["jobs"].append(result)
            receipt["completed_diffusion_calls"] += 1
            c.write(output / "receipt.json", receipt)
            print(c.json.dumps(dict(completed=job["id"], calls=receipt["completed_diffusion_calls"], mode=mode)), flush=True)
        receipt.update(engine.finalize())
        receipt.update(complete=True, active_job=None, wall_seconds=time.perf_counter()-started)
        c.write(output / "receipt.json", receipt)
        # 实际回读保存的入口与输出pins，检验下一阶段可以完整读取。
        restored = c.read(output / "receipt.json")
        c.need(restored["complete"] and len(restored["jobs"]) == len(selected), "receipt完整回读失败")
        for job in restored["jobs"]:
            for name in ("raw_model", "raw_native", "final_native", "candidate_native_valid"):
                c.check_pin(job[name])
        receipt["saved_receipt_and_outputs_restored"] = True
        c.write(output / "receipt.json", receipt)
    except BaseException as error:
        receipt.update(complete=False, failed=True, error_type=type(error).__name__, error=str(error),
                       wall_seconds=time.perf_counter()-started)
        c.write(output / "receipt.json", receipt)
        raise


def merge(plan_path, receipts, output):
    m = c.plan_check(plan_path)
    workers = [c.read(path) for path in receipts]
    c.need(workers and all(w["complete"] and w["mode"] == "worker" and w["manifest"] == c.pin(plan_path)
                          and w["saved_receipt_and_outputs_restored"] for w in workers), "分片未完成")
    jobs = [j for w in workers for j in w["jobs"]]
    c.need(len(jobs) == len({j["id"] for j in jobs}) == 60, "正式60组合重复/缺失")
    lookup = {j["id"]: j for j in jobs}
    p = dict(schema=c.PREDICTION_SCHEMA, complete=True, manifest=c.pin(plan_path),
        attempted_diffusion_calls=sum(w["attempted_diffusion_calls"] for w in workers),
        completed_diffusion_calls=sum(w["completed_diffusion_calls"] for w in workers),
        maximum_diffusion_calls=60, first_historical_replay_passed=True,
        historical_replays_passed=sum(j["historical_replay_expected"] and bool(j["historical_replay_exact"])
                                      and all(j["historical_replay_exact"].values()) for j in jobs),
        training_updates=0, GT_opened=False, automatic_candidate_selection=False,
        adapter_unchanged=all(w["adapter_unchanged"] for w in workers),
        checkpoint_unchanged=all(w["checkpoint_unchanged"] for w in workers),
        jobs=[lookup[j["id"]] for j in m["jobs"]], worker_receipts=[c.pin(path) for path in receipts],
        additional_VAE_control_calls=0, reuse_historical_outputs=False,
        peak_single_worker_GPU_allocated_bytes=max(w["peak_GPU_allocated_bytes"] for w in workers))
    output = Path(output)
    c.need(not output.exists(), "保留旧合并结果")
    c.write(output, p)
    c.write(output.with_name("freeze_gate.json"), c.freeze_gate(output, plan_path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    for mode in ("smoke", "worker"):
        q = s.add_parser(mode)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--execute-gpu", action="store_true")
        q.add_argument("--worker-index", type=int, default=0)
        q.add_argument("--worker-count", type=int, default=4)
    q = s.add_parser("merge")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--receipts", type=Path, nargs="+", required=True)
    q.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "merge":
        merge(a.manifest, a.receipts, a.out)
    else:
        c.need(a.execute_gpu, "必须显式--execute-gpu；未加载GPU")
        execute(a.manifest, a.out, a.mode, a.worker_index, a.worker_count)


if __name__ == "__main__":
    main()
