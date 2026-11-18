"""三态与fresh native gate的CPU合同；不导入Torch/native。"""
import unittest
import tempfile
import json
import hashlib
from pathlib import Path
import numpy as np
from module2_activation_state import prepare_request,FreshNativeGate,apply_fresh_gate,load_fresh_native_gate


class StateTests(unittest.TestCase):
    def request(self):
        return prepare_request(parent_render_active=np.array([1,0,0,0,0],bool),
            geometry_support=np.array([0,1,1,0,1],bool),material_qualified=np.array([0,1,0,1,1],bool),
            requested=np.array([0,1,1,1,1],bool),material_qualification_source='synthetic CPU source audit')

    def test_three_states_are_independent_and_CPU_does_not_activate(self):
        r=self.request()
        np.testing.assert_array_equal(r.candidate_world_rows,[1,4])
        self.assertFalse(r.candidate_world_rows.flags.writeable)
        np.testing.assert_array_equal(r.render_active,[1,0,0,0,0])
        self.assertTrue(r.render_active[0])
        self.assertFalse(r.geometry_support[0])
        self.assertFalse(r.material_qualified[0])

    def test_requires_explicit_material_source(self):
        with self.assertRaises(ValueError):
            prepare_request(parent_render_active=np.zeros(2,bool),geometry_support=np.ones(2,bool),
                material_qualified=np.ones(2,bool),requested=np.ones(2,bool),material_qualification_source='')

    def test_stale_oldactive_K_table_rejected(self):
        r=self.request()
        gate=FreshNativeGate('body',np.array([1,4]),np.array([0]),np.zeros(5,bool),np.zeros(5,bool),(86,110),10,'old gate')
        with self.assertRaises(ValueError):
            apply_fresh_gate(r,gate,candidate_road_sha256='body',required_frames=(86,110),required_K_pixel_frames=10)

    def test_native_guard_rejects_new_but_preserves_old_active(self):
        r=self.request();K=np.array([1,0,0,0,1],bool)
        gate=FreshNativeGate('body',np.array([1,4]),np.array([0,1,4]),K,np.zeros(5,bool),(86,110),10,'fresh gate')
        d=apply_fresh_gate(r,gate,candidate_road_sha256='body',required_frames=(86,110),required_K_pixel_frames=10)
        np.testing.assert_array_equal(d.render_active,[1,1,0,0,0])
        self.assertEqual(int(d.K_rejected.sum()),1)

    def test_body_or_denominator_mismatch_rejected(self):
        r=self.request()
        gate=FreshNativeGate('body',np.array([1,4]),np.array([0,1,4]),np.zeros(5,bool),np.zeros(5,bool),(86,110),10,'fresh gate')
        with self.assertRaises(ValueError):
            apply_fresh_gate(r,gate,candidate_road_sha256='different body',required_frames=(86,110),required_K_pixel_frames=10)

    def test_serialized_mask_is_bound_with_body_and_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);body=d/'road.npz';body.write_bytes(b'synthetic body')
            scene=d/'scene.json';scene.write_text('{}')
            mask=d/'guard.npz'
            np.savez(mask,world_K_guard=np.zeros(5,bool),world_near_guard=np.zeros(5,bool),
                candidate_world_rows=np.array([1,4]),packet_active_world_rows=np.array([0,1,4]))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            receipt=d/'receipt.json';metadata=dict(schema='module2_fresh_native_gate.v1',
                GPU_executed=True,candidate_road_sha256=sha(body),mask_sha256=sha(mask),scene_manifest_sha256=sha(scene),
                fixed_geometry_sigma_opacity=True,method='actual_native_fixed_sigma_tile_footprint_and_near',
                frames=[86,110],K_pixel_frames=10)
            receipt.write_text(json.dumps(metadata))
            load_fresh_native_gate(mask,receipt,candidate_road_path=body,candidate_scene_manifest_path=scene)
            receipt.write_text(json.dumps(dict(metadata,GPU_executed=False,native_producer_GPU_executed=True)))
            with self.assertRaises(ValueError):
                load_fresh_native_gate(mask,receipt,candidate_road_path=body,candidate_scene_manifest_path=scene)
            receipt.write_text(json.dumps(metadata))
            np.savez(mask,world_K_guard=np.ones(5,bool),world_near_guard=np.zeros(5,bool),
                candidate_world_rows=np.array([1,4]),packet_active_world_rows=np.array([0,1,4]))
            with self.assertRaises(ValueError):
                load_fresh_native_gate(mask,receipt,candidate_road_path=body,candidate_scene_manifest_path=scene)


if __name__=='__main__':unittest.main()
