"""CPU export/parser/factory protocol tests; no actual CUDA renderer is invoked."""
from __future__ import annotations
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
try:
    from . import v44_cli as cli
except ImportError:
    import v44_cli as cli


class FakeScene:
    def __init__(self, fail_key=None, fail_after=False):
        self.events, self.calls, self.checks = [], [], 0
        self.fail_key, self.fail_after = fail_key, fail_after

    def assert_persistent_unchanged(self):
        self.events.append("guard")
        self.checks += 1
        if self.fail_after and self.checks == 2:
            raise RuntimeError("persistent state changed")

    def render(self, key, **query):
        self.events.append("render")
        self.calls.append((key, query))
        if key == self.fail_key:
            raise RuntimeError("renderer protocol failure")
        alpha = np.full((1, 2, 3), .75, np.float32)
        rgb = np.stack([np.full((2, 3), value, np.float32) for value in (.1, .2, 1.2)])
        groups = cli._module("v44_scene").GROUPS
        diagnostics = {name: np.zeros_like(alpha) for name in groups}
        diagnostics["unassigned"] = alpha.copy()
        ids = query["diagnostic_actor_ids"]
        owners = np.concatenate([alpha * .25 for _ in ids], axis=0) if ids else np.empty((0, 2, 3), np.float32)
        enabled = query.get("diagnostic_branch_alpha", False)
        branches = np.concatenate([alpha * .5, alpha * .5], 0) if enabled else np.empty((0, 2, 3), np.float32)
        return dict(rgb=rgb, raw_rgb=rgb.copy(), env_map=np.zeros_like(rgb), depth_sum=alpha * 4,
            alpha=alpha, diagnostics=diagnostics, owner_alpha_T=owners, owner_alpha_actor_ids=ids,
            branch_alpha_T=branches, branch_alpha_names=cli._module("v44_scene").BRANCH_ALPHA_NAMES if enabled else (),
            diagnostic_alpha_error_max=0., coverage=[dict(provider=0, available=False, native_fallback=True)])


class CLIProtocols(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.data = dict(schema="v44.scene.v1", model="unloaded-model", engine="unloaded-engine", iteration=314,
            actors=[dict(track_id="target/α", actor_id=7), dict(track_id="retained-track", actor_id=991)],
            frames=[dict(key="night/alpha:β", image_name="native-A.png", model_time=-.6, source_time_s=200.),
                    dict(key=77, image_name="native-B.png", model_time=.1, source_time_s=200.1),
                    dict(key="77", image_name="native-C.png", model_time=.8, source_time_s=200.2)])
        self.manifest_path.write_text(json.dumps(self.data), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        return cli._module("v44_manifest").SceneManifest.load(self.manifest_path)

    def factory(self, scene):
        calls = []
        def load(path):
            calls.append(Path(path))
            return scene, self.manifest()
        return load, calls

    def test_inspect_is_static_and_preserves_typed_frame_subset(self):
        with mock.patch.object(cli, "_factory", side_effect=AssertionError("inspect must not call factory")):
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["inspect", "--manifest", str(self.manifest_path), "--frames-json", '["77",77]'])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual([row["key"] for row in report["frames"]], ["77", 77])
        self.assertEqual(report["full_frame_count"], 3)
        self.assertEqual(report["frames"][0]["model_time"], .8)
        self.assertEqual(report["frames"][0]["source_time_s"], 200.2)

    def test_query_resolves_registered_tracks_and_actors_without_case_rules(self):
        manifest = self.manifest()
        tracks = cli.resolve_query(manifest, dict(delete_track_ids=["target/α"], focus_track_ids=["retained-track"], diagnostic_actor_ids=[991, 7]))
        actors = cli.resolve_query(manifest, dict(delete_actor_ids=[7], focus_actor_ids=[991], diagnostic_track_ids=["retained-track", "target/α"]))
        self.assertEqual(tracks, actors)
        self.assertEqual(tracks["diagnostic_actor_ids"], (991, 7))
        for bad in (dict(delete_actor_ids=[-1]), dict(delete_actor_ids=[123]),
                    dict(delete_actor_ids=[7, 7]), dict(delete_track_ids=["missing"]),
                    dict(delete_actor_ids=[7], delete_track_ids=["target/α"]), dict(case_override=True)):
            with self.subTest(query=bad), self.assertRaises(ValueError):
                cli.resolve_query(manifest, bad)

    def test_render_exports_float_arrays_png_and_original_frame_provenance(self):
        scene = FakeScene()
        factory, calls = self.factory(scene)
        output = self.root / "artifacts"
        report = cli.render_manifest(self.manifest_path, output,
            query=dict(delete_track_ids=["target/α"], diagnostic_actor_ids=[991, 7]),
            frame_keys=["77", "night/alpha:β"], factory=factory)
        self.assertEqual(calls, [self.manifest_path.resolve()])
        self.assertEqual(scene.events, ["guard", "render", "render", "guard"])
        self.assertEqual([row[0] for row in scene.calls], ["77", "night/alpha:β"])
        self.assertEqual(report["status"], "COMPLETE")
        self.assertTrue(report["persistent_check_after"])
        with np.load(output / "frames/000000/arrays.npz", allow_pickle=False) as values:
            self.assertEqual(values["alpha"].dtype, np.float32)
            self.assertEqual(values["group_alpha_T"].shape, (6, 2, 3))
            self.assertEqual(values["owner_alpha_T"].shape, (2, 2, 3))
            self.assertEqual(values["owner_alpha_actor_ids"].tolist(), [991, 7])
            np.testing.assert_allclose(values["group_alpha_T"].sum(0, keepdims=True), values["alpha"])
            self.assertGreater(float(values["rgb"][2, 0, 0]), 1.)
        pixel = np.array(Image.open(output / "frames/000000/rgb.png"))[0, 0].tolist()
        self.assertEqual(pixel, [26, 51, 255])
        metadata = json.loads((output / "frames/000001/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["frame"]["key"], "night/alpha:β")
        self.assertFalse(metadata["training_approved"])

    def test_render_failure_preserves_provenance_and_runs_final_guard(self):
        scene = FakeScene(fail_key=77)
        factory, _ = self.factory(scene)
        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "protocol failure"):
            cli.render_manifest(self.manifest_path, output, factory=factory)
        receipt = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "FAILED")
        self.assertEqual(len(receipt["frames"]), 1)
        self.assertTrue(receipt["persistent_check_before"])
        self.assertTrue(receipt["persistent_check_after"])
        self.assertEqual(scene.events[-1], "guard")

    def test_persistent_guard_failure_cannot_claim_complete(self):
        scene = FakeScene(fail_after=True)
        factory, _ = self.factory(scene)
        output = self.root / "mutated"
        with self.assertRaisesRegex(RuntimeError, "persistent state"):
            cli.render_manifest(self.manifest_path, output, frame_keys=[77], factory=factory)
        receipt = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "FAILED")
        self.assertFalse(receipt["persistent_check_after"])

    def test_invalid_subset_query_or_existing_output_never_loads_factory(self):
        factory = mock.Mock(side_effect=AssertionError("invalid input must not load models"))
        for index, keys in enumerate(([77, 77], ["missing"], [])):
            with self.assertRaises(ValueError):
                cli.render_manifest(self.manifest_path, self.root / f"bad{index}", frame_keys=keys, factory=factory)
        with self.assertRaises(ValueError):
            cli.render_manifest(self.manifest_path, self.root / "badquery", query=dict(delete_actor_ids=[123]), factory=factory)
        existing = self.root / "existing"
        existing.mkdir(); (existing / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            cli.render_manifest(self.manifest_path, existing, factory=factory)
        factory.assert_not_called()
        self.assertEqual((existing / "keep.txt").read_text(), "keep")

    def test_main_query_file_and_direct_flags_call_same_factory_contract(self):
        query_path = self.root / "query.json"
        query_path.write_text(json.dumps(dict(focus_track_ids=["retained-track"], diagnostic_actor_ids=[991])), encoding="utf-8")
        scene = FakeScene()
        factory, _ = self.factory(scene)
        with mock.patch.object(cli, "_factory", return_value=factory), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["render", "--manifest", str(self.manifest_path), "--query", str(query_path),
                "--frame-key", '"77"', "--output", str(self.root / "queryfile")]), 0)
            self.assertEqual(cli.main(["render", "--manifest", str(self.manifest_path), "--focus-actor-id", "991",
                "--diagnostic-track-id", "retained-track", "--frame-key", "77", "--output", str(self.root / "direct")]), 0)
        self.assertEqual(scene.calls[0][0], "77")
        self.assertEqual(scene.calls[1][0], 77)
        self.assertEqual(scene.calls[0][1], scene.calls[1][1])

    def test_help_and_query_errors_do_not_invoke_factory(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        with mock.patch.object(cli, "_factory", side_effect=AssertionError("not reached")), redirect_stderr(io.StringIO()):
            code = cli.main(["render", "--manifest", str(self.manifest_path), "--delete-actor-id", "999",
                "--output", str(self.root / "error")])
        self.assertEqual(code, 1)

    def test_branch_alpha_query_flag_and_float_export(self):
        query_path = self.root / "branch-query.json"
        query_path.write_text(json.dumps(dict(diagnostic_actor_ids=[7, 991])), encoding="utf-8")
        scene = FakeScene()
        factory, _ = self.factory(scene)
        output = self.root / "branch-output"
        with mock.patch.object(cli, "_factory", return_value=factory), redirect_stdout(io.StringIO()):
            code = cli.main(["render", "--manifest", str(self.manifest_path), "--query", str(query_path),
                "--diagnostic-branch-alpha", "--frames-json", "[77]", "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertTrue(scene.calls[0][1]["diagnostic_branch_alpha"])
        with np.load(output / "frames/000000/arrays.npz", allow_pickle=False) as values:
            self.assertEqual(values["branch_alpha_T"].shape, (2, 2, 3))
            self.assertEqual(values["branch_alpha_names"].tolist(), ["nonfocus_static", "nonfocus_dynamic"])
            np.testing.assert_allclose(values["group_alpha_T"].sum(0, keepdims=True), values["alpha"])

    def test_branch_mode_owner_capacity_and_bool_validation_precede_factory(self):
        data = dict(self.data, actors=[dict(track_id=f"actor-{i}", actor_id=i) for i in range(1, 17)])
        manifest = cli._module("v44_manifest").SceneManifest(data, self.root)
        self.assertEqual(len(cli.resolve_query(manifest, dict(diagnostic_actor_ids=list(range(1, 17))))["diagnostic_actor_ids"]), 16)
        resolved = cli.resolve_query(manifest, dict(diagnostic_actor_ids=list(range(1, 15)), diagnostic_branch_alpha=True))
        self.assertEqual(len(resolved["diagnostic_actor_ids"]), 14)
        with self.assertRaisesRegex(ValueError, "At most 14"):
            cli.resolve_query(manifest, dict(diagnostic_actor_ids=list(range(1, 16)), diagnostic_branch_alpha=True))
        with self.assertRaises(ValueError):
            cli.resolve_query(manifest, dict(diagnostic_branch_alpha="true"))


if __name__ == "__main__":
    unittest.main()
