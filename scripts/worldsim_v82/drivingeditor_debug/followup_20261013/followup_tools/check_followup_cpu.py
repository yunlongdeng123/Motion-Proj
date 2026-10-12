"""高影响启动边界CPU检查：fake assets/训练module；不调用GPU、nvidia-smi或真实worker。"""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent
from contracts import (below, build_config, disk_forecast, gate_contract, input_inventory, read_json,
                       readiness_contract, require, require_disk_space, scene_spec, write_new)
from prepare_followup import copy_code, prepare, verify_prepared, worker_source
import run_scene_gn11


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def png_header(width, height):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)


class FollowupChecks(unittest.TestCase):
    tools = None
    reference = None
    source = None
    staging = None

    def setUp(self):
        work = HERE / "work"
        work.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=work)
        self.root = below(self.temp.name, work)
        self.addCleanup(self.checked_cleanup)
        self.input = self.root / "inputs"
        frames = ["0008%03d" % i for i in range(23, 72)]
        for directory, suffix in (("image/image_0", ".png"), ("sky/sky_0", ".png"),
                                  ("dynamic_masks/dynamic_mask_0", ".png"),
                                  ("dynamic_image/dynamic_image_0", ".png"), ("pose", ".txt"),
                                  ("calib", ".txt"), ("velodyne", ".bin")):
            folder = self.input / directory
            folder.mkdir(parents=True)
            for name in frames:
                (folder / (name + suffix)).write_bytes(png_header(1920, 1280) if suffix == ".png" else b"fixture")
        for name in ("points3d.ply", "points3d_dynamic_0.ply", "transform_matrix.txt"):
            (self.input / name).write_text("fixture", encoding="utf-8")
        self.spec_value = {"scene_id": "scene0008_023_071", "source_path": str(self.input),
                           "frame_ids": frames, "frame_count": 49, "camera": "FRONT",
                           "raw_frame_range": [23, 71], "original_WH": [1920, 1280], "render_WH": [480, 320],
                           "resolution_scale": 4, "fixed_pose_anchor": "0008033", "fixed_pose_camera_index": 10,
                           "model_timestamp_values": [-.48 + .02 * i for i in range(49)],
                           "actor_PLY_paths": [str(self.input / "points3d_dynamic_0.ply")]}
        self.spec = scene_spec(self.spec_value)
        self.ready = {"status": "PASS_CPU_PREPARATION_TRAINING_GATED", "gpu_used": False,
                      "cuda_initialized": False, "training_started": False, "training_ready": False,
                      "scene_copy": str(self.input), "train_cameras": 49, "test_cameras": 0,
                      "fixed_pose_anchor": "0008033",
                      "canonical_full_mask_validation": [{"frame_id": f, "actual_canonical_mask_loader_exact": True} for f in frames],
                      "SGCM_view_inputs": {"count": 49, "ids": frames},
                      "training_guard_enabled_in_candidate_config": True, "cpu_geometry_guard_enabled": False,
                      "SGCM_geometry": {"status": "validated", "meter_to_model_scale": .1, "ground_fit": {"rmse_m": .0289}},
                      "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        self.config = read_json(self.reference)
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.spec_file = self.root / "scene.json"
        self.ready_file = self.root / "ready.json"
        self.config_file = self.root / "reference.json"
        write_new(self.spec_file, self.spec_value)
        write_new(self.ready_file, self.ready)
        write_new(self.config_file, self.config)
        self.source_fixture = self.root / "source"
        if self.source:
            copy_code(self.source, self.source_fixture)
        else:
            self.make_actual_patched_source()
        self.paths = {"campaign_root": str(self.campaign), "scene_spec": str(self.spec_file),
                      "readiness_file": str(self.ready_file), "reference_config": str(self.config_file),
                      "source_repository": str(self.source_fixture), "matrix_tools_dir": str(self.tools)}
        self.follow_cfg = build_config(self.config, self.spec, self.campaign)
        self.gate_file = self.root / "gate_passed.json"
        self.make_gate()

    def checked_cleanup(self):
        below(self.root, HERE / "work")
        self.temp.cleanup()

    def make_actual_patched_source(self):
        train_patch = load(self.tools / "train_patch.py", "followup_fixture_train_patch")
        sgcm_patch = load(self.tools / "sgcm_patch.py", "followup_fixture_sgcm_patch")
        sources = {"train.py": train_patch.patch_train_source,
                   "utils/paper_checkpoint.py": train_patch.patch_checkpoint_source,
                   "scene/paper_sgcm.py": sgcm_patch.patch_sgcm_source,
                   "scene/gaussian_model.py": sgcm_patch.patch_gaussian_source}
        for name, patch in sources.items():
            target = self.source_fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch((self.staging / name).read_text(encoding="utf-8")), encoding="utf-8")
        helpers = {"runtime.py": "utils/hybrid30k_runtime.py", "losses.py": "utils/hybrid30k_losses.py",
                   "metrics.py": "utils/hybrid30k_metrics.py", "static_time.py": "utils/hybrid30k_static_time.py",
                   "sgcm_helpers.py": "scene/hybrid_sgcm_helpers.py"}
        for source, name in helpers.items():
            target = self.source_fixture / name
            target.write_bytes((self.tools / source).read_bytes())

    def make_gate(self):
        matrix = self.root / "reference_matrix"
        matrix.mkdir()
        write_new(matrix / "matrix_status.json", {"status": "all_four_arms_complete"})
        events = {}
        for arm in ("C00", "GN11"):
            run = matrix / "runs" / arm
            (run / "checkpoint_events").mkdir(parents=True)
            config = copy.deepcopy(self.config)
            config.update(hybrid_arm=arm, model_path=str(run),
                          paper_sgcm_enabled=arm == "GN11", hybrid_nps_enabled=arm == "GN11")
            records = []
            for i in range(49):
                fid = "image_0/0017%03d" % i
                p = run / ("%03d.json" % i)
                write_new(p, {"frame_id": fid, "regions": {"full": {"composite_psnr_db": 35 if arm == "C00" else 34.8},
                                                         "actor": {"composite_psnr_db": 33 if arm == "C00" else 32.8}}})
                records.append({"frame_id": fid, "timestamp": -.48 + .02 * i, "width": 480, "height": 320,
                                "hybrid_metrics": str(p)})
            event = {"status": "complete", "iteration": 30000, "config": config, "frames": records,
                     "checkpoints": {name: {"bytes": size} for name, size in
                                     (("static", 800000000), ("dynamic", 800000000),
                                      ("environment", 226494608), ("training_state", 30000))},
                     "parameters_evaluated": "after_optimizer_and_sgcm_same_as_checkpoint"}
            path = run / "checkpoint_events/iteration_030000.json"
            write_new(path, event)
            events[arm] = str(path)
        self.gate = {"schema_version": 1, "status": "passed", "candidate_arm": "GN11", "reference_arm": "C00",
                     "candidate_checkpoint": 30000, "reference_checkpoint": 30000,
                     "paired_contract_pass": True, "visual_review_complete": True,
                     "severe_numeric_regression": False, "severe_visual_or_static_regression": False,
                     "allowed_scene_ids": ["scene0008_023_071"], "paired_steps_evidence": {"status": "passed", "rows": 30000},
                     "events": events, "max_psnr_drop_db": 1.0}
        write_new(self.gate_file, self.gate)

    def rewrite_gate(self):
        self.gate_file.write_text(json.dumps(self.gate), encoding="utf-8")

    def args(self, *extra):
        values = []
        for name, path in self.paths.items():
            values += ["--" + name.replace("_", "-"), path]
        return run_scene_gn11.parser().parse_args(values + list(extra))

    def test_real_candidate_types_and_anchor(self):
        self.assertEqual(self.spec["anchor_frame"], "0008033")
        self.assertEqual(readiness_contract(self.ready, self.spec)["status"],
                         "SCENE_CPU_PREPARED_ACTUAL_GPU_BINDING_PENDING")
        self.assertEqual(self.follow_cfg["start_checkpoint"], None)
        self.assertEqual(self.follow_cfg["hybrid_static_anchor_frame"] if "hybrid_static_anchor_frame" in self.follow_cfg else
                         self.follow_cfg["followup_static_anchor_frame"], "0008033")

    def test_default_dry_run_does_not_import_torch_or_spawn(self):
        program = "import sys;sys.path.insert(0,sys.argv[1]);import run_scene_gn11 as r;a=r.parser().parse_args(sys.argv[2:]);x=r.validate(a);assert 'torch' not in sys.modules;print(x[-1]['gpu_started'])"
        args = self.args()
        cli = []
        for name, path in self.paths.items():
            cli += ["--" + name.replace("_", "-"), path]
        result = subprocess.run([sys.executable, "-c", program, str(HERE), *cli],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")
        with mock.patch("subprocess.Popen", side_effect=AssertionError("不可spawn")):
            output = run_scene_gn11.validate(args)[-1]
        self.assertFalse(output["gate_passed"])
        self.assertFalse(output["gpu_started"])

    def test_prepare_succeeds_while_gate_pending_and_isolated(self):
        before = (self.source_fixture / "utils/hybrid30k_static_time.py").read_bytes()
        record = prepare(self.paths, self.follow_cfg, self.spec, readiness_contract(self.ready, self.spec))
        self.assertIn("GPU_BLOCKED", record["status"])
        self.assertFalse(record["gpu_started"])
        self.assertEqual(before, (self.source_fixture / "utils/hybrid30k_static_time.py").read_bytes())
        self.assertEqual(Path(record["training_source"]).parent.name, "training_source")
        self.assertEqual(verify_prepared(self.paths, self.follow_cfg, self.spec)["scene_id"], self.spec["scene_id"])
        worker = (Path(record["training_support"]) / "matrix_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("0017010", worker)
        self.assertNotIn("def main(", worker)
        self.assertIn("preserve_rng()", worker)
        self.assertIn("gpu_slot.lock", worker)
        self.assertIn("bind_training_contract(training, cfg)", worker)
        with self.assertRaises(ValueError):
            prepare(self.paths, self.follow_cfg, self.spec, {})
        altered = Path(record["training_source"]) / "train.py"
        altered.write_text(altered.read_text(encoding="utf-8") + "\n# stale source\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_prepared(self.paths, self.follow_cfg, self.spec)

    def test_passed_filename_cannot_bypass_gate(self):
        self.gate["status"] = "blocked"
        self.rewrite_gate()
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, self.config)
        with self.assertRaises(ValueError):
            run_scene_gn11.validate(self.args("--run", "--gate-file", str(self.gate_file)))

    def test_passed_gate_rechecks_actual_events(self):
        result = gate_contract(self.gate_file, self.spec, self.config)
        self.assertEqual(result["status"], "GATE_PASSED_RECHECKED")
        config = copy.deepcopy(self.config)
        config["hybrid_lambda_dynamic_opacity"] = 99.0
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, config)
        config = copy.deepcopy(self.config)
        config["followup_static_anchor_frame"] = "0008033"
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, config)

    def test_mismatched_timestamp_or_nonterminal_matrix_rejected(self):
        event = Path(self.gate["events"]["GN11"])
        data = read_json(event)
        data["frames"][10]["timestamp"] += .001
        event.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, self.config)
        data["frames"][10]["timestamp"] -= .001
        event.write_text(json.dumps(data))
        matrix = event.resolve().parents[3] / "matrix_status.json"
        matrix.write_text(json.dumps({"status": "running"}))
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, self.config)

    def test_severe_numeric_regression_cannot_be_hidden_by_passed_marker(self):
        event = read_json(self.gate["events"]["GN11"])
        for frame in event["frames"]:
            p = Path(frame["hybrid_metrics"])
            row = read_json(p)
            row["regions"]["actor"]["composite_psnr_db"] = 31.5
            p.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, self.config)

        # 用户确认1dB后，修改receipt阈值也不能掩盖严重回退。
        gate = read_json(self.gate_file)
        gate["max_psnr_drop_db"] = 2.0
        self.gate_file.write_text(json.dumps(gate))
        with self.assertRaises(ValueError):
            gate_contract(self.gate_file, self.spec, self.config)

    def test_unknown_red_and_wrong_anchor_rejected(self):
        value = copy.deepcopy(self.spec_value)
        value["scene_id"] = "urban_red_crossing"
        with self.assertRaises(ValueError):
            scene_spec(value)
        value = copy.deepcopy(self.spec_value)
        value["fixed_pose_anchor"] = "0017010"
        with self.assertRaises(ValueError):
            scene_spec(value)

    def test_missing_image_or_changed_header_rejected(self):
        path = self.input / "sky/sky_0/0008033.png"
        path.write_bytes(png_header(1919, 1280))
        with self.assertRaises(ValueError):
            input_inventory(self.spec)
        path.unlink()
        with self.assertRaises(ValueError):
            input_inventory(self.spec)

    def test_cpu_geometry_config_and_restore_rejected(self):
        config = copy.deepcopy(self.config)
        config["sgcm_guard_enabled"] = False
        with self.assertRaises(ValueError):
            build_config(config, self.spec, self.campaign)
        config = copy.deepcopy(self.config)
        config["start_checkpoint"] = "old30k.pth"
        with self.assertRaises(ValueError):
            build_config(config, self.spec, self.campaign)

    def test_low_disk_rejected_without_cleaning(self):
        prediction = disk_forecast(self.config)
        self.assertGreater(prediction["required_free_bytes"], 0)
        self.assertLess(prediction["required_free_bytes"], 25 * 1024**3)
        survivor = self.root / "keep_old_checkpoint"
        survivor.write_bytes(b"keep")
        with mock.patch("contracts.shutil.disk_usage", return_value=types.SimpleNamespace(free=1)):
            with self.assertRaises(ValueError):
                require_disk_space(self.campaign, prediction)
        self.assertEqual(survivor.read_bytes(), b"keep")

    def test_slot_busy_attempt_receipts_do_not_block_retry(self):
        forecast = disk_forecast(self.config)
        first = run_scene_gn11.record_start_attempt(self.campaign, self.spec, forecast)
        second = run_scene_gn11.record_start_attempt(self.campaign, self.spec, forecast)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file() and second.is_file())
        self.assertFalse(Path(self.follow_cfg["model_path"]).exists())

    def test_local_import_factory_binding_smoke_without_train_global(self):
        # 精确覆盖training() local-import factory，trainmodule本身没有该属性。
        import torch
        from runtime_binding import bind_training_contract
        cameras = [types.SimpleNamespace(image_path=str(self.input / "image/image_0" / (name + ".png")),
                   image_name=name, timestamp=stamp, image_width=480, image_height=320)
                   for name, stamp in zip(self.spec["frame_ids"], self.spec["model_timestamps"])]
        model = types.SimpleNamespace(get_xyz=torch.zeros(10, 3), _sgcm_lineage_ids=torch.arange(10),
                                      _sgcm_clone_blocked=torch.zeros(10, dtype=torch.bool))
        scene = types.SimpleNamespace(getTrainCameras=lambda: cameras, getTestCameras=lambda **kwargs: [],
                                      model_path=self.follow_cfg["model_path"])
        controller = types.SimpleNamespace(config=types.SimpleNamespace(guard_enabled=True),
            training_views=[{"id": c.image_path} for c in cameras],
            metadata={"world_to_model": self.ready["transform_matrix"]})
        scene_package = types.ModuleType("scene")
        factory_module = types.ModuleType("scene.paper_sgcm")
        factory_module.create_sgcm_controller = lambda *args, **kwargs: controller
        scene_package.paper_sgcm = factory_module
        utils = types.ModuleType("utils")
        checkpoint = types.ModuleType("utils.paper_checkpoint")
        checkpoint.save_components = lambda *args, **kwargs: {}
        runtime = types.ModuleType("utils.hybrid30k_runtime")
        runtime.camera_id = lambda c: Path(c.image_path).parent.name + "/" + c.image_name
        utils.paper_checkpoint = checkpoint
        training = types.SimpleNamespace(Scene=lambda *args, **kwargs: scene)
        self.assertFalse(hasattr(training, "create_sgcm_controller"))
        module_map = {"scene": scene_package, "scene.paper_sgcm": factory_module,
                      "utils": utils, "utils.paper_checkpoint": checkpoint, "utils.hybrid30k_runtime": runtime}
        with mock.patch.dict(sys.modules, module_map):
            bind_training_contract(training, self.follow_cfg, self.ready)
            training.Scene(None)
            result = factory_module.create_sgcm_controller(str(self.input), self.follow_cfg,
                                                           training_views=cameras, gaussians=model)
        self.assertIs(result, controller)
        binding = read_json(Path(self.follow_cfg["model_path"]) / "actual_sgcm_binding.json")
        self.assertEqual(binding["status"], "PASS_ACTUAL_GPU_SGCM_LINEAGE_BINDING")
        self.assertFalse(torch.cuda.is_initialized())

    def test_scene_time_anchor_and_exception_restore_cpu(self):
        import torch
        from static_time_scene import run_static_time_check
        cameras = [types.SimpleNamespace(image_path=str(self.input / "image/image_0" / (name + ".png")),
                   image_name=name, timestamp=stamp) for name, stamp in zip(self.spec["frame_ids"], self.spec["model_timestamps"])]
        model = types.SimpleNamespace(_velocity=torch.zeros(3, 3), get_scaling_t=torch.ones(3, 1), T=.2)
        scene = types.SimpleNamespace(getTrainCameras=lambda: cameras, followup_scene_spec=self.spec,
                                      model_path=str(self.root / "time_ok"), gaussians=model, dynamic_gaussians=None)
        Path(scene.model_path).mkdir()
        runtime = types.ModuleType("utils.hybrid30k_runtime")
        runtime.camera_id = lambda c: Path(c.image_path).parent.name + "/" + c.image_name
        def renderer(camera, *args, **kwargs):
            return {"render_s": torch.full((3, 320, 480), .5 + camera.timestamp / 10),
                    "static_alpha": torch.full((1, 320, 480), .8 + camera.timestamp / 20)}
        with mock.patch.dict(sys.modules, {"utils.hybrid30k_runtime": runtime}):
            result = run_static_time_check(scene, renderer, (), None, 10000)
            self.assertEqual(result["anchor_pose_frame"], "0008033")
            self.assertEqual(result["rows"][10]["timestamp_model"], self.spec["anchor_timestamp"])
            self.assertEqual(result["rows"][10]["static_rgb_MAE_from_anchor"], 0)
            self.assertEqual(cameras[10].timestamp, self.spec["anchor_timestamp"])
            scene.model_path = str(self.root / "time_fail")
            Path(scene.model_path).mkdir()
            def failing(camera, *args, **kwargs):
                if camera.timestamp == self.spec["model_timestamps"][-1]:
                    raise RuntimeError("fixture")
                return renderer(camera)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                run_static_time_check(scene, failing, (), None, 20000)
            self.assertEqual(cameras[10].timestamp, self.spec["anchor_timestamp"])
        self.assertFalse(torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-tools-dir", required=True, type=Path)
    parser.add_argument("--reference-config", required=True, type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source-repository", type=Path, help="已准备的actual frozen repository，仅读/复制小代码")
    sources.add_argument("--staging-source", type=Path, help="本地原stage，用实际train/SGCM patch生成fixture")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    FollowupChecks.tools = args.matrix_tools_dir.resolve()
    FollowupChecks.reference = args.reference_config.resolve()
    FollowupChecks.source = args.source_repository.resolve() if args.source_repository else None
    FollowupChecks.staging = args.staging_source.resolve() if args.staging_source else None
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FollowupChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"status": "PASS_FOLLOWUP_CPU_CHECKS" if result.wasSuccessful() else "FAILED_FOLLOWUP_CPU_CHECKS",
               "python": sys.version, "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "GPU_used": False, "checkpoint_loaded": False, "fake_train_module_only": True,
               "actual_GPU_model_and_guard_binding_verified": False, "restoration_verified": False}
    if args.output:
        write_new(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
