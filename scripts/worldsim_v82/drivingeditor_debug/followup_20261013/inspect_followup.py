"""只读小型后续状态；不导入训练框架、不采样GPU、不启动worker。"""
import argparse
import datetime
import json
from pathlib import Path
import shutil

DEFAULT_ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013")


def read_if(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def process_check(pid, expected):
    if type(pid) is not int or pid <= 1:
        return {"pid": pid, "alive": False, "matches_owned_job": False}
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return {"pid": pid, "alive": False, "matches_owned_job": False}
    parts = [part.decode("utf-8", errors="replace") for part in command if part]
    return {"pid": pid, "alive": bool(parts), "matches_owned_job": any(expected in part for part in parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "root": str(root), "read_only": True, "gpu_used": False}
    if not root.is_dir():
        result["status"] = "preparation_root_missing"
    else:
        result["disk_free_GiB"] = shutil.disk_usage(root).free / 1024**3
        ready = read_if(root / "metadata/scene0008_cpu_readiness.json")
        if ready:
            result["scene0008_CPU"] = {k: ready.get(k) for k in
                ("status", "train_cameras", "dynamic_points", "static_points_after_exact0p1m_exclusion",
                 "fixed_pose_anchor", "training_ready", "training_gates_pending")}
        search = read_if(root / "red_search_cpu/status.json")
        if search:
            result["red_search"] = {k: search.get(k) for k in
                ("status", "mode", "updated_utc", "network_reserved_bytes", "network_received_bytes",
                 "new_control_passed", "cached_pass_completed", "cached_images_ranked",
                 "new_candidate_review_pending", "new_segments_this_invocation", "error",
                 "temporal_progress", "temporal_indices_started", "temporal_contract",
                 "temporal_candidate_review_pending", "temporal_candidate")}
            result["red_search"].update(
                completed_new_segments_count=len(search.get("completed_new_segments", [])),
                contract=search.get("contract"),
                process=process_check(search.get("pid"), "search_red_scene_"))
        download = read_if(root / "inputs/raw_scene0029/download_status.json")
        if download:
            result["red_raw_download"] = {k: download.get(k) for k in
                ("run_status", "updated_utc", "expected_bytes", "received_offset", "error")}
            result["red_raw_download"]["process"] = process_check(download.get("pid"), "download_red_raw_cpu.py")
        download_receipt = read_if(root / "inputs/raw_scene0029/download_receipt.json")
        if download_receipt:
            result["red_full_source"] = {k: download_receipt.get(k) for k in
                ("status", "raw_size_bytes", "size_verified", "full_sha256_verified")}
        identity = read_if(root / "metadata/scene0029_identity_review.json")
        if identity:
            result["scene0029_identity"] = {k: identity.get(k) for k in
                ("status", "identity_verified", "author_video_verified_raw_range",
                 "local_training_range", "anchor_pose_frame", "author_training_window_publicly_confirmed")}
            if "red_search" in result:
                result["red_search"]["candidate_search_resume_allowed"] = identity.get("identity_verified") is not True
        red_ready = read_if(root / "metadata/red_preparation/scene0029_069_117/cpu_raw_readiness.json")
        if red_ready:
            result["scene0029_CPU_raw"] = {k: red_ready.get(k) for k in
                ("status", "raw_training_range", "requested_frames", "num_points", "fixed_pose_anchor_for_later_training",
                 "training_ready", "pending", "pending_after_CPU", "GPU_used", "Torch_imported", "TF_imported")}
        prepared = []
        for path in sorted((root / "training_metadata").glob("*_prepared.json")):
            row = read_if(path)
            prepared.append({k: row.get(k) for k in
                            ("status", "scene_id", "anchor_pose_frame", "gpu_started", "gate_required_at_run")})
        result["prepared_training_sources"] = prepared
        active_queue = read_if(root / "metadata/red_masks_active_queue.json")
        attempt = active_queue.get("attempt", 1) if active_queue else 1
        if type(attempt) is not int or not 1 <= attempt <= 9:
            raise ValueError("Invalid owned mask queue attempt registry")
        suffix = "" if attempt == 1 else "_attempt%d" % attempt
        mask_queue = read_if(root / ("metadata/red_masks_20261013%s/status.json" % suffix))
        result["red_GPU_mask_attempts"] = [{"directory": str(p.parent), **{k: read_if(p).get(k) for k in ("status", "pid", "error")}}
                                         for p in sorted((root / "metadata").glob("red_masks_20261013*/status.json"))]
        if mask_queue:
            result["red_GPU_mask_queue"] = {k: mask_queue.get(k) for k in
                ("status", "pid", "updated_utc", "active_child_pid", "error", "sky_frames",
                 "training_started", "actor_selection_complete", "training_ready", "human_verdict")}
            result["red_GPU_mask_queue"]["controller_process"] = process_check(mask_queue.get("pid"), "run_red_masks_after_scene8.py")
            result["red_GPU_mask_queue"]["stages"] = [{k: s.get(k) for k in
                ("stage", "pid", "returncode", "wall_seconds", "memory_sampling")} for s in mask_queue.get("stages", [])]
            result["red_GPU_mask_queue"]["attempt"] = attempt
            launch = read_if(root / ("metadata/red_masks_queue_launch_20261013%s.json" % suffix))
            result["red_GPU_mask_queue"]["launch_log"] = launch.get("log") if launch else None
        elif active_queue:
            result["red_GPU_mask_queue"] = {"attempt": attempt, "status": "dispatch_pending_runtime_status",
                "pid": active_queue.get("pid"), "launch_log": active_queue.get("log"),
                "controller_process": process_check(active_queue.get("pid"), "run_red_masks_after_scene8.py")}
        runs = []
        for path in sorted((root / "runs").glob("*/status.json")):
            row = read_if(path)
            run = {"output": str(path.parent), **{k: row.get(k) for k in
                         ("status", "iteration", "wall_seconds", "pid", "worker_pid", "error")}}
            run["worker_process"] = process_check(row.get("pid", row.get("worker_pid")), "run_scene_gn11.py")
            runs.append(run)
        result["followup_runs"] = runs
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
