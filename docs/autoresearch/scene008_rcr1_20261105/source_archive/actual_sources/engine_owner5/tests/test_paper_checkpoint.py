"""CPU tests: atomic bundles, sampler/RNG state, actual capture/restore + Adam."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from utils import paper_checkpoint as checkpoint

ROOT = Path(__file__).resolve().parents[1]


def args_for(root):
    return OmegaConf.create({"model_path": str(root), "source_path": "/source/scene", "iterations": 50000,
                            "including_dynamic": True, "env_map_res": 2})


def camera(name, timestamp, uid):
    return types.SimpleNamespace(image_path=f"/source/scene/image_0/{name}.png", image_name=name,
                                 timestamp=timestamp, colmap_id=uid, image_width=4, image_height=3)


class Scene:
    def __init__(self):
        self.resolution_scales = [1, 4]
        self.scale_index = 1
        self.train_cameras = {scale: [camera("0002", .2, 2), camera("0001", .1, 1), camera("0000", 0, 0)]
                              for scale in self.resolution_scales}

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]

    def getTestCameras(self, scale):
        return []


class TinyModel:
    def __init__(self, number):
        self.value = torch.nn.Parameter(torch.tensor([float(number)]))
        self.optimizer = torch.optim.Adam([self.value], lr=.01)
        self.value.square().sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.no_time_split = False

    def capture(self):
        return self.value, self.optimizer.state_dict()


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = args_for(self.root)
        self.scene = Scene()
        self.models = {"static": TinyModel(1), "dynamic": TinyModel(2), "environment": TinyModel(3)}

    def test_complete_event_requires_all_components_then_preserves_each_payload(self):
        components = checkpoint.save_components(self.root, 10000, self.models)
        event_path = self.root / "checkpoint_events/iteration_010000.json"
        self.assertFalse(event_path.exists())
        state = checkpoint.capture_training_state(10000, self.scene, [2, 0], self.models, self.args)
        event = checkpoint.publish_checkpoint_event(self.root, 10000, components, state, [], self.args, self.scene)
        self.assertEqual(json.loads(event_path.read_text())["status"], "complete")
        self.assertEqual(set(event["checkpoints"]), {"static", "dynamic", "environment", "training_state"})
        for key, model in self.models.items():
            (parameter, optimizer), iteration = torch.load(event["checkpoints"][key]["path"], weights_only=False)
            self.assertEqual(iteration, 10000)
            torch.testing.assert_close(parameter, model.value)
            self.assertEqual(optimizer["state"][0]["step"].item(), 1)
        with self.assertRaises(FileExistsError):
            checkpoint.save_components(self.root, 10000, self.models)

    def test_failed_component_write_never_publishes_event_or_leaves_tempfile(self):
        original = torch.save
        count = 0
        def broken_save(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated interrupted write")
            return original(*args, **kwargs)
        with patch.object(torch, "save", broken_save), self.assertRaises(OSError):
            checkpoint.save_components(self.root, 10000, self.models)
        self.assertFalse(list(self.root.glob("checkpoint_events/*.json")))
        self.assertFalse(list(self.root.glob(".*")))

    def test_missing_evaluation_frame_blocks_complete_publication(self):
        components = checkpoint.save_components(self.root, 10000, self.models)
        state = checkpoint.capture_training_state(10000, self.scene, [], self.models, self.args)
        with self.assertRaisesRegex(ValueError, "Evaluation frame missing"):
            checkpoint.publish_checkpoint_event(self.root, 10000, components, state,
                [{"frames": [{"channels": {"gt": str(self.root / "missing.png")}}]}], self.args, self.scene)
        self.assertFalse(list(self.root.glob("checkpoint_events/*.json")))

    def test_requested_full_train_evaluation_cannot_silently_publish_empty(self):
        self.args.paper_eval_train_each_checkpoint = True
        components = checkpoint.save_components(self.root, 10000, self.models)
        state = checkpoint.capture_training_state(10000, self.scene, [], self.models, self.args)
        with self.assertRaisesRegex(ValueError, "full training-camera evaluation"):
            checkpoint.publish_checkpoint_event(self.root, 10000, components, state, [], self.args, self.scene)
        with self.assertRaisesRegex(ValueError, "missing or duplicating"):
            checkpoint.publish_checkpoint_event(self.root, 10000, components, state,
                [{"split": "train", "frames": []}], self.args, self.scene)
        self.assertFalse(list(self.root.glob("checkpoint_events/*.json")))

    def test_rng_camera_order_scale_and_sampler_continue_exactly_on_cpu(self):
        random.seed(8); np.random.seed(9); torch.manual_seed(10)
        state = checkpoint.capture_training_state(10000, self.scene, [2, 0], self.models, self.args)
        state_path = self.root / "training_state10000.pth"
        torch.save(state, state_path)
        expected = (random.random(), np.random.rand(5), torch.rand(5))
        new_scene = Scene()
        new_scene.scale_index = 0
        for cameras in new_scene.train_cameras.values():
            cameras.reverse()
        self.models["static"].no_time_split = True
        random.seed(98); np.random.seed(99); torch.manual_seed(100)
        stack, _ = checkpoint.restore_training_state(state_path, 10000, new_scene, self.models, self.args)
        self.assertEqual(stack, [2, 0])
        self.assertEqual(new_scene.scale_index, 1)
        self.assertEqual([c.image_name for c in new_scene.getTrainCameras()], ["0002", "0001", "0000"])
        self.assertFalse(self.models["static"].no_time_split)
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.rand(5), expected[1])
        torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)

    def test_wrong_camera_set_rejected_and_legacy_sidecar_explicitly_warned(self):
        state = checkpoint.capture_training_state(10000, self.scene, [], self.models, self.args)
        path = self.root / "state.pth"
        torch.save(state, path)
        self.scene.train_cameras[4].pop()
        with self.assertRaisesRegex(ValueError, "camera identities"):
            checkpoint.restore_training_state(path, 10000, self.scene, self.models, self.args)
        with self.assertWarnsRegex(UserWarning, "Legacy checkpoint"):
            self.assertEqual(checkpoint.restore_training_state(self.root / "absent", 10000, self.scene, self.models, self.args), (None, {}))

    def test_intermediate_train_eval_empty_test_is_full_fixed_order(self):
        configurations = checkpoint.paper_evaluation_configs(self.scene)
        self.assertEqual(configurations[0]["cameras"], [])
        self.assertEqual([camera.image_name for camera in configurations[1]["cameras"]], ["0000", "0001", "0002"])


def actual_class(filename, class_name, method_names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition.body = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    exec(compile(ast.Module(body=[definition], type_ignores=[]), filename, "exec"), namespace)
    return namespace[class_name]


class ActualModelRestoreTests(unittest.TestCase):
    def test_both_actual_gaussian_branches_restore_adam_and_next_lr_update(self):
        args = OmegaConf.merge(OmegaConf.load(ROOT / "configs/base.yaml"),
                               OmegaConf.load(ROOT / "configs/paper_waymo_reconstruction.yaml"))
        utils_tree = ast.parse((ROOT / "utils/general_utils.py").read_text(encoding="utf-8"))
        scheduler = next(node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_expon_lr_func")
        namespace = {"torch": torch, "np": np}
        exec(compile(ast.Module(body=[scheduler], type_ignores=[]), "actual-scheduler", "exec"), namespace)
        original_zeros = torch.zeros
        def zeros_on_cpu(*values, **kwargs):
            kwargs["device"] = "cpu"
            return original_zeros(*values, **kwargs)
        with patch.object(torch, "zeros", zeros_on_cpu):
            for name in ("GaussianModel", "Dynamic_GaussianModel"):
                cls = actual_class("scene/gaussian_model.py", name,
                                   {"capture", "restore", "training_setup", "update_learning_rate", "get_xyz"}, namespace)
                model = cls()
                model.setup_functions = lambda: None
                for field in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_t", "_scaling_t", "_velocity"):
                    setattr(model, field, torch.nn.Parameter(torch.tensor([[.2, .3]])))
                model.active_sh_degree = 2; model.max_radii2D = torch.tensor([7.])
                model.spatial_lr_scale = .7; model.T = .2; model.velocity_decay = 1.
                model.training_setup(args)
                model.update_learning_rate(10000)
                sum(group["params"][0].square().sum() for group in model.optimizer.param_groups).backward()
                model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
                restored = cls(); restored.setup_functions = lambda: None
                restored.restore(copy.deepcopy(model.capture()), args)
                for current, previous in zip(restored.optimizer.param_groups, model.optimizer.param_groups):
                    self.assertEqual(current["lr"], previous["lr"])
                    self.assertEqual(restored.optimizer.state[current["params"][0]]["step"].item(), 1)
                    torch.testing.assert_close(restored.optimizer.state[current["params"][0]]["exp_avg"],
                                               model.optimizer.state[previous["params"][0]]["exp_avg"])
                for branch in (model, restored):
                    branch.update_learning_rate(10001)
                    sum(group["params"][0].square().sum() for group in branch.optimizer.param_groups).backward()
                    branch.optimizer.step()
                for current, previous in zip(restored.optimizer.param_groups, model.optimizer.param_groups):
                    self.assertEqual(current["lr"], previous["lr"])
                    torch.testing.assert_close(current["params"][0], previous["params"][0], rtol=0, atol=0)

    def test_actual_environment_restore_binds_optimizer_to_loaded_parameter(self):
        namespace = {"torch": torch}
        cls = actual_class("scene/envlight.py", "EnvLight", {"capture", "restore", "training_setup"}, namespace)
        original = cls(); original.base = torch.nn.Parameter(torch.full((6, 2, 2, 3), .4))
        args = types.SimpleNamespace(envmap_lr=.001)
        original.training_setup(args)
        original.base.square().sum().backward(); original.optimizer.step(); original.optimizer.zero_grad(set_to_none=True)
        restored = cls(); restored.base = torch.nn.Parameter(torch.zeros_like(original.base))
        restored.restore(copy.deepcopy(original.capture()), args)
        self.assertIs(restored.optimizer.param_groups[0]["params"][0], restored.base)
        for model in (original, restored):
            model.base.square().sum().backward(); model.optimizer.step()
        torch.testing.assert_close(original.base, restored.base, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
