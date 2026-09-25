"""有限的 exhaustive → 全新训练 → 基础评估队列；关机由任务收口时执行。"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/external/worldsim_v75/VAD-GS")
RUN = Path("/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000")
PY = "/root/autodl-tmp/envs/vadgs-v76/bin/python"
CONFIG = str(ROOT / "configs/v76/nuscenes_000_repro_exhaustive.yaml")
state_path = RUN / "pipeline_state.json"
env = dict(os.environ, OMP_NUM_THREADS="12", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
           CUDA_HOME="/usr/local/cuda-11.8", TORCH_CUDA_ARCH_LIST="8.6", PYTHONUNBUFFERED="1")
env["PATH"] = "/usr/local/cuda-11.8/bin:" + env["PATH"]
state = {"run_id":"VADGS-P0R1-000", "controller_pid":os.getpid(), "completed_stages":[],
         "shutdown_when_pipeline_exits":False, "failure_ledger_refs":["V76-F01"]}

def save():
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2)+"\n")
    tmp.replace(state_path)

def run(name, cmd, log, cuda=True):
    state.update(stage=name, status="running", command=cmd, log=str(RUN/log))
    stage_env = dict(env)
    if not cuda:
        stage_env["CUDA_VISIBLE_DEVICES"] = ""
    save()
    with (RUN/log).open("xb") as stream:
        child = subprocess.Popen(cmd, cwd=ROOT, env=stage_env, stdout=stream, stderr=subprocess.STDOUT)
        state["child_pid"] = child.pid
        save()
        code = child.wait()
    if code:
        state.update(status="failed", returncode=code)
        save()
        raise SystemExit(code)
    state["completed_stages"].append(name)
    state.pop("child_pid", None)
    save()

if __name__ == "__main__":
    assert RUN.is_dir() and not state_path.exists(), "refuse duplicate pipeline"
    assert not (RUN / "trained_model").exists(), "fresh run must not contain weights"
    base = RUN / "colmap"
    run("exhaustive_matching", ["colmap", "exhaustive_matcher", "--database_path", str(base/"database.db"),
        "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", "12", "--random_seed", "0"],
        "colmap_exhaustive.log", cuda=False)
    run("triangulation", ["colmap", "point_triangulator", "--database_path", str(base/"database.db"),
        "--image_path", str(base/"train_imgs"), "--input_path", str(base/"created/sparse/model"),
        "--output_path", str(base/"triangulated/sparse/model"), "--Mapper.num_threads", "12",
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.max_extra_param", "0", "--clear_points", "0", "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.filter_max_reproj_error", "4", "--Mapper.filter_min_tri_angle", "0.5",
        "--Mapper.tri_min_angle", "0.5", "--Mapper.tri_ignore_two_view_tracks", "1",
        "--Mapper.tri_complete_max_reproj_error", "4", "--Mapper.tri_continue_max_angle_error", "4"],
        "colmap_triangulation.log", cuda=False)
    assert (base / "triangulated/sparse/model/points3D.bin").stat().st_size > 8
    run("train_30000", [PY, "train.py", "--config", CONFIG], "train.log")
    assert (RUN / "trained_model/iteration_30000.pth").is_file()
    run("official_test_30000", [PY, "script/v76/evaluate_official_test.py", "--config", CONFIG,
        "--iteration", "30000", "--output-dir", str(RUN/"official_test_30k")], "evaluate_official_30k.log")
    run("camera5_extrapolation_30000", [PY, "script/v76/evaluate_heldout.py", "--config", CONFIG,
        "--iteration", "30000", "--heldout-camera", "5", "--output-dir", str(RUN/"cam5_extrapolation_30k")], "evaluate_cam5_30k.log")
    run("lateral_sweep_30000", [PY, "script/v76/render_counterfactual.py", "--config", CONFIG,
        "--iteration", "30000", "--intervention", "sweep", "--camera", "0", "--frame", "20",
        "--offsets", "0,0.5,1,2,3.5", "--output-dir", str(RUN/"counterfactual_30k")], "render_sweep_30k.log")
    state.update(status="completed_needs_review", stage="pipeline_complete",
        next="Review artifacts, prepare same-scene continuation if justified, save/push final report, then authorized AutoDL shutdown when no work remains")
    save()
