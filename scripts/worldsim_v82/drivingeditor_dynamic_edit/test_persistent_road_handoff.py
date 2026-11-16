"""CPU protocol tests: no native model/CUDA load and no GPU forward."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest

import numpy as np
import torch


PACKAGE = 'persistent_handoff_cpu_fixture'
package = ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules[PACKAGE] = package
api = importlib.import_module(PACKAGE + '.persistent_road_handoff')
scene_api = importlib.import_module(PACKAGE + '.v44_scene')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def pin(path):
    return dict(path=str(path.resolve()), sha256=api._sha(path))


class PersistentRoadHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='persistent-road-handoff-cpu-')
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model'; self.model.mkdir()
        sources = []
        for role, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt'), ('environment', 'env_light_chkpnt')]:
            path = self.model / f'{prefix}7.pth'
            path.write_bytes(('explicit CPU-only checkpoint fixture ' + role).encode())
            sources.append(dict(role=role, **pin(path)))
        self.ownership = self.root / 'ownership.npz'
        np.savez(self.ownership, static_actor_ids=np.zeros(2, np.int64), dynamic_actor_ids=np.array([51]))
        self.road = self.root / 'road.npz'
        self.asset = dict(active=np.array([True, False, True, True]),
            model_xyz=np.array([[0, 0, .05], [0, 0, 9], [0, 0, .15], [0, 0, .4]], np.float32),
            model_from_world=np.diag([.1, .1, .1, 1.]), meter_to_model=np.array(.1),
            minimum_model_depth=np.array([0, 0, .2, .2], np.float32))
        np.savez(self.road, **self.asset)
        self.indices = self.root / 'combined_indices.npz'; np.savez(self.indices, ids=np.array([0, 1], np.int64))
        self.checkpoint = self.root / 'combined.pt'
        self.manifest = self.root / 'scene.json'
        self.source = dict(schema='v44.scene.v1', model=str(self.model), iteration=7, road=str(self.road),
            ownership=str(self.ownership), static_gates=dict(indices=str(self.indices), checkpoint=str(self.checkpoint)))
        write(self.manifest, self.source)
        self.binding = dict(manifest_sha256=api._sha(self.manifest), road_sha256=api._sha(self.road), static_rows=2, dynamic_rows=1)
        self.saved = dict(schema=api.CHECKPOINT_SCHEMA, gate=torch.tensor([1., 2.]),
            contributors=torch.tensor([0, 1]), binding=copy.deepcopy(self.binding))
        torch.save(self.saved, self.checkpoint)
        self.proof = dict(schema=api.PROOF_SCHEMA, manifest=pin(self.manifest),
            road=dict(**pin(self.road), total_rows=4, active_rows=3, meter_to_model=.1),
            native_sources=sources, ownership=pin(self.ownership), topology=dict(static_rows=2, dynamic_rows=1),
            gates=dict(indices=pin(self.indices), checkpoint=pin(self.checkpoint), checkpoint_binding=copy.deepcopy(self.binding)),
            raster_units=dict(unit_scale=10., legacy_native_near=.2))
        self.proof_path = self.root / 'proof.json'; write(self.proof_path, self.proof)
        self.calls = 0
        self.captures = []

    def tearDown(self):
        self.temporary.cleanup()

    def factory(self, path):
        self.calls += 1
        source = json.loads(Path(path).read_text())
        camera = SimpleNamespace(image_height=2, image_width=3, FoVx=1., FoVy=1.,
            world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4),
            projection_matrix=torch.eye(4), camera_center=torch.zeros(3))
        def make_model(depths):
            xyz = torch.tensor([[0., 0., z] for z in depths])
            n = len(xyz)
            rows = dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
                colors_precomp=torch.ones(n, 3) * .3, features=torch.stack((xyz[:, 2], torch.ones(n)), 1),
                opacities=torch.full((n, 1), .5), scales=torch.ones(n, 3) * .01,
                rotations=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1), cov3D_precomp=None,
                mask=torch.ones(n, dtype=torch.bool))
            return SimpleNamespace(get_xyz=xyz, active_sh_degree=0, rows=rows)
        static, dynamic = make_model([.1, .4]), make_model([.3])
        with np.load(source['road']) as data:
            road_xyz = torch.as_tensor(data['model_xyz'][data['active']].copy())
        road = make_model(road_xyz[:, 2].tolist()); road.xyz = road.get_xyz
        def merge(left, right):
            return {key: None if value is None else torch.cat((value, right[key]), 0) for key, value in left.items()}
        def raster(cam, pipe, degree, values, background):
            self.captures.append((cam, values))
            return dict(sentinel='CPU forwarding only')
        ops = scene_api.NativeOps(lambda cam, obj, pipe, dynamic=False: obj.rows, None, merge, raster)
        provider = scene_api.RowProvider(lambda *_: road.rows, 0, 'road', state_objects=(road,))
        indices = np.load(source['static_gates']['indices'])['ids']
        saved = torch.load(source['static_gates']['checkpoint'], map_location='cpu', weights_only=False)
        gates = np.ones(2, np.float32); gates[indices] = saved['gate'].sigmoid().numpy()
        scene = scene_api.V44Scene(static, dynamic, None, SimpleNamespace(including_dynamic=True), lambda _: camera,
            source_owner_ids={'static': 0, 'dynamic': 51}, row_providers=(provider,), native_ops=ops, static_gates=gates)
        manifest = SimpleNamespace(model=Path(source['model']), iteration=source['iteration'], road=Path(source['road']))
        return scene, manifest

    def load(self):
        return api.load_persistent_handoff(self.manifest, self.proof_path, factory=self.factory)

    def update_proof(self):
        write(self.proof_path, self.proof)

    def update_road(self):
        np.savez(self.road, **self.asset)
        self.proof['road'].update(pin(self.road))
        self.binding['road_sha256'] = api._sha(self.road)
        self.update_checkpoint()

    def update_checkpoint(self):
        self.saved['binding'] = copy.deepcopy(self.binding)
        torch.save(self.saved, self.checkpoint)
        self.proof['gates']['checkpoint'] = pin(self.checkpoint)
        self.proof['gates']['checkpoint_binding'] = copy.deepcopy(self.binding)
        self.update_proof()

    def test_saved_reload_combined_gate_and_depth_masks(self):
        scene, _, receipt = self.load()
        values, _, camera, _ = scene._rows(12345, (), (), False)
        self.assertTrue(torch.equal(values['mask'], torch.tensor([False, True, True, True, False, True])))
        expected_depth = torch.tensor([.1, .4, .3, .05, .15, .4])
        torch.testing.assert_close(values['features'][:, 0], expected_depth)
        self.assertEqual(receipt['bounded_active_road_rows'], 2)
        self.assertEqual(receipt['unit_scale'], 10.)
        self.assertFalse(receipt['quality_pass_claimed'])
        scene._bindings().raster(camera, scene.pipe, 0, values, torch.zeros(3))
        forwarded_camera, forwarded = self.captures[-1]
        torch.testing.assert_close(forwarded['means3D'], values['means3D'] * 10)
        self.assertIs(forwarded['features'], values['features'])
        scene.assert_persistent_unchanged()
        second, _, _ = self.load()
        replay, _, _, _ = second._rows(-99, (), (), False)
        for key in ['mask', 'opacities', 'features', 'means3D']:
            self.assertTrue(torch.equal(values[key], replay[key]))

    def test_save_new_road_preserves_gates_and_public_reload(self):
        changed = self.root / 'trained_road.npz'
        asset = dict(self.asset, reference_srgb=np.full((4, 3), .4, np.float32))
        np.savez(changed, **asset)
        manifest, proof = api.save_persistent_road_handoff(self.manifest, self.proof_path,
            changed, self.root / 'trained', training_record=dict(formal_steps=200, role='teacher_distillation'))
        scene, loaded, receipt = api.load_persistent_handoff(manifest, proof, factory=self.factory)
        self.assertEqual(loaded.road, changed)
        self.assertEqual(receipt['gated_static_rows'], 2)
        saved = torch.load(self.root / 'trained' / 'static_opacity.pt', map_location='cpu', weights_only=False)
        self.assertTrue(torch.equal(saved['gate'], self.saved['gate']))
        self.assertTrue(torch.equal(saved['contributors'], self.saved['contributors']))
        self.assertEqual(json.loads((self.root / 'trained' / 'build.json').read_text())['retired_static_rows'], 0)
        scene.assert_persistent_unchanged()
        with self.assertRaises(FileExistsError):
            api.save_persistent_road_handoff(self.manifest, self.proof_path, changed, self.root / 'trained')

    def test_save_rejects_corrupted_parent_before_creating_output(self):
        self.saved['gate'][0] = -8.
        torch.save(self.saved, self.checkpoint)
        output = self.root / 'rejected'
        with self.assertRaises(ValueError):
            api.save_persistent_road_handoff(self.manifest, self.proof_path, self.road, output)
        self.assertFalse(output.exists())

    def test_minimum_tensor_is_readonly_watched(self):
        scene, _, _ = self.load()
        scene._persistent_road_handoff['minimum_model_depth'].add_(.01)
        with self.assertRaisesRegex(RuntimeError, 'mutated'):
            scene.assert_persistent_unchanged()

    def test_default_factory_not_modified(self):
        scene, _ = self.factory(self.manifest)
        original, _, _, _ = scene._rows(0, (), (), False)
        self.assertTrue(bool(original['mask'].all()))
        self.load()
        again, _, _, _ = scene._rows(0, (), (), False)
        self.assertTrue(torch.equal(original['mask'], again['mask']))

    def test_road_without_optional_minimum_remains_supported(self):
        del self.asset['minimum_model_depth']; self.update_road()
        scene, _, receipt = self.load()
        values, _, _, _ = scene._rows(0, (), (), False)
        self.assertTrue(torch.equal(values['mask'], torch.tensor([False, True, True, True, True, True])))
        self.assertFalse(receipt['optional_minimum_model_depth'])

    def test_input_road_payload_is_not_mutated(self):
        before = self.road.read_bytes()
        scene, _, _ = self.load()
        scene._rows(0, (), (), False)
        self.assertEqual(before, self.road.read_bytes())

    def test_changed_native_source_rejected_before_factory(self):
        (self.model / 'chkpnt7.pth').write_bytes(b'changed topology under same path')
        with self.assertRaisesRegex(ValueError, 'file binding'):
            self.load()
        self.assertEqual(self.calls, 0)

    def test_changed_ownership_rejected_before_factory(self):
        self.ownership.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'file binding'):
            self.load()
        self.assertEqual(self.calls, 0)

    def test_checkpoint_contributors_must_match_indices(self):
        self.saved['contributors'] = torch.tensor([1, 0]); self.update_checkpoint()
        with self.assertRaisesRegex(ValueError, 'contributor topology'):
            self.load()

    def test_duplicate_gate_indices_rejected(self):
        np.savez(self.indices, ids=np.array([0, 0], np.int64))
        self.proof['gates']['indices'] = pin(self.indices); self.update_proof()
        with self.assertRaisesRegex(ValueError, 'unique valid'):
            self.load()

    def test_binding_mismatch_rejected(self):
        self.saved['binding']['unexpected_parent'] = 'different'
        torch.save(self.saved, self.checkpoint)
        self.proof['gates']['checkpoint'] = pin(self.checkpoint); self.update_proof()
        with self.assertRaisesRegex(ValueError, 'checkpoint binding differs'):
            self.load()

    def test_nonfinite_minimum_depth_rejected(self):
        self.asset['minimum_model_depth'][2] = np.nan; self.update_road()
        with self.assertRaisesRegex(ValueError, 'minimum_model_depth'):
            self.load()

    def test_wrong_minimum_depth_length_rejected(self):
        self.asset['minimum_model_depth'] = np.zeros(3, np.float32); self.update_road()
        with self.assertRaisesRegex(ValueError, 'minimum_model_depth'):
            self.load()

    def test_wrong_reciprocal_scale_rejected(self):
        self.proof['raster_units']['unit_scale'] = 9.; self.update_proof()
        with self.assertRaisesRegex(ValueError, 'convert'):
            self.load()

    def test_wrong_road_metadata_rejected(self):
        self.proof['road']['active_rows'] = 4; self.update_proof()
        with self.assertRaisesRegex(ValueError, 'topology'):
            self.load()

    def test_double_metric_wrap_rejected(self):
        self.source['raster_units'] = {'unit_scale': 10., 'legacy_native_near': .2}
        write(self.manifest, self.source)
        self.proof['manifest'] = pin(self.manifest); self.update_proof()
        with self.assertRaisesRegex(ValueError, 'double wrapping'):
            self.load()


if __name__ == '__main__':
    unittest.main()
