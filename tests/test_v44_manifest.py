"""Manifest/NPZ/frozen LoD CPU contracts; dummy arrays only, no CUDA or model loads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from v44_manifest import SceneManifest, _pose
from v44_factory import _load_ownership, _owner_array, _row_mask, _load_actor_poses, _load_road_arrays
from v44_surface_lod import SurfaceLODAsset, stats


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); (self.base / 'model').mkdir()
        # Actual native cameras.json field layout/SE3 from the prior scene-065 artifact.
        self.camera = dict(img_name='0065000', width=1920, height=1280,
            position=[-.9999998018315818, .0005939333065375709, -.0007724531671628145],
            rotation=[[.0008516122844337774, .015066851119983493, .9998861258932691],
                      [.9690659345055274, -.24678501412927825, .0028933340881135464],
                      [.24680050514019095, .9689531189889972, -.014810937310293336]],
            fx=2081.696719057, fy=2081.696719057, cx=954.9586638747, cy=659.1282934505)
        self.data = dict(schema='v44.scene.v1', model='model', engine='engine', iteration=173,
            frames=[dict(key='arbitrary/frame UUID', image_name='0065000', model_time=-1.98, source_time_s=1507240547.844102)],
            actors=[dict(track_id='original raw UUID', actor_id=701)], render_wh=[480, 320])
        self.binding = dict(frame_ids=['image_0/0065000'], timestamps=[-1.98], render_wh=[480, 320])

    def saved(self, cameras=None, binding=None):
        (self.base/'model/cameras.json').write_text(json.dumps([self.camera] if cameras is None else cameras))
        (self.base/'model/actual_scene_binding.json').write_text(json.dumps(self.binding if binding is None else binding))

    def test_actual_camera_binding_adapter_preserves_explicit_key_clock_and_se3(self):
        self.saved(); m = SceneManifest(self.data, self.base); h = m.camera_headers()['arbitrary/frame UUID']
        c = np.eye(4); c[:3, :3] = self.camera['rotation']; c[:3, 3] = self.camera['position']
        np.testing.assert_allclose(h['W'] @ c, np.eye(4), atol=1e-12)
        self.assertEqual(h['timestamp'], -1.98); self.assertEqual(h['image_name'], '0065000')
        self.assertAlmostEqual(h['fx'], self.camera['fx']/4); self.assertAlmostEqual(h['cy'], self.camera['cy']/4)
        self.assertEqual(m.actor_ids(['original raw UUID']), (701,))
        self.assertEqual(m.model, (self.base/'model').resolve())

    def test_integer_frame_keys_arbitrary_camera_names_and_windows_binding(self):
        d = deepcopy(self.data); d['frames'][0].update(key=-273, image_name='camera-A UUID.png')
        camera = {**self.camera, 'img_name': 'camera-A UUID.png'}
        binding = dict(frame_ids=['image_7\\camera-A UUID.png'], timestamps=[-1.98])
        self.saved([camera], binding)
        self.assertEqual(SceneManifest(d, self.base).camera_headers()[-273]['frame_key'], -273)

    def test_duplicate_or_nonfinite_camera_binding_is_rejected(self):
        cases = [dict(frame_ids=['a','b'], timestamps=[-1.98]),
                 dict(frame_ids=['image_0/0065000'], timestamps=[float('nan')]),
                 dict(frame_ids=['image_0/0065000'], timestamps=[float('inf')]),
                 dict(frame_ids=['a/0065000','b/0065000'], timestamps=[-1.98,-1.98])]
        for binding in cases:
            self.saved(binding=binding)
            with self.subTest(binding=binding), self.assertRaises(ValueError): SceneManifest(self.data, self.base).camera_headers()

    def test_camera_pose_and_positive_intrinsics_shapes_are_strict(self):
        bads = [dict(position=0), dict(position=[0,0,float('nan')]), dict(rotation=np.ones((3,3)).tolist()),
                dict(width=0), dict(width=-1920, fx=-1), dict(width=1920.5), dict(fy=0), dict(cx=float('inf'))]
        for change in bads:
            self.saved([{**self.camera, **change}])
            with self.subTest(change=change), self.assertRaises(ValueError): SceneManifest(self.data, self.base).camera_headers()

    def test_schema_actor_ids_paths_and_reference_hw_do_not_cast(self):
        bads = [dict(schema='scene-specific'), dict(model=''), dict(iteration=True), dict(iteration=1.7),
                dict(actors=[dict(track_id='raw',actor_id=7.8)]), dict(actors=[dict(track_id='raw',actor_id=2**63)]),
                dict(frames=[dict(key=True,image_name='x',model_time=0)]),
                dict(frames=[dict(key='x',image_name='x',model_time=float('nan'))]),
                dict(static_gates={}), dict(world_to_model=np.diag([1,2,1,1]).tolist()),
                dict(surfaces=[dict(track_id='original raw UUID', geometry='x', poses='p', posterior='q', reference_hw=[4.2,5])])]
        for change in bads:
            with self.subTest(change=change), self.assertRaises(ValueError): SceneManifest({**self.data,**change},self.base)
        with self.assertRaises(ValueError): SceneManifest(self.data,self.base).actor_ids('original raw UUID')

    def test_float32_similarity_roundoff_is_not_confused_with_shear(self):
        angle=.73; c,s=np.cos(angle),np.sin(angle)
        pose=np.eye(4,dtype=np.float32);pose[:3,:3]=1000*np.array([[c,-s,0],[s,c,0],[0,0,1]])
        out=_pose(pose,'float32 model scale',similarity=True)
        np.testing.assert_array_equal(out,pose.astype(np.float64))
        pose[0,1]+=.1
        with self.assertRaises(ValueError):_pose(pose,'actual shear',similarity=True)

    def test_actor_pose_file_accepts_explicit_nonclip_keys_and_rejects_silent_reshape(self):
        p = self.base/'poses.json'
        value = dict(schema='v44.actor_poses.v1',track_id='raw',frames=[dict(key='UUID-key',world_from_actor=np.eye(4).ravel().tolist())])
        p.write_text(json.dumps(value)); poses = _load_actor_poses(p,'raw',{'UUID-key',837})
        np.testing.assert_array_equal(poses['UUID-key'],np.eye(4))
        for matrix in [np.eye(4).reshape(2,8), np.diag([2,1,1,1]), np.full((4,4),np.nan)]:
            value['frames'][0]['world_from_actor']=matrix.tolist();p.write_text(json.dumps(value))
            with self.subTest(shape=matrix.shape),self.assertRaises(ValueError):_load_actor_poses(p,'raw',{'UUID-key'})

    def test_ownership_npz_rejects_float_fraction_bool_negative_and_overflow(self):
        p = self.base/'owners.npz'
        for a in [np.array([1.,7.8]), np.array([1.,7.]), np.array([True,False]), np.array([-2,7]),np.array([2**63,7],np.uint64),np.array([[1],[7]])]:
            np.savez(p,static_actor_ids=a,dynamic_actor_ids=np.array([701],np.int64))
            with self.subTest(dtype=a.dtype,shape=a.shape),self.assertRaises(ValueError):_load_ownership(p,2,1)
        np.savez(p,static_actor_ids=np.array([-1,701],np.int64),dynamic_actor_ids=np.array([29],np.uint16))
        out=_load_ownership(p,2,1);self.assertEqual(out['dynamic'].dtype,np.int64);self.assertEqual(out['static'].tolist(),[-1,701])
        np.testing.assert_array_equal(_owner_array(np.array([],np.int64),0,'empty'),np.array([],np.int64))

    def test_retirement_indices_do_not_truncate_or_accept_bad_topology(self):
        p=self.base/'rows.npz';np.savez(p,ids=np.array([1,3],np.int64))
        np.testing.assert_array_equal(_row_mask(p,5),[0,1,0,1,0])
        for a in [np.array([1.9]),np.array([-1]),np.array([5]),np.ones((5,1),bool)]:
            np.savez(p,ids=a)
            with self.subTest(a=a),self.assertRaises(ValueError):_row_mask(p,5)

    def road(self, **updates):
        a=dict(active=np.array([True,False]),model_from_world=np.eye(4),meter_to_model=np.array(1.),
               model_xyz=np.zeros((2,3)),gaussian_sigma_m=np.full((2,3),.1),
               normal_world=np.tile([0.,0.,1.],(2,1)),reference_srgb=np.full((2,3),.2),road_opacity=np.array([.99,.99]))
        a.update(updates);p=self.base/'road.npz';np.savez(p,**a);return p

    def test_road_arrays_validate_without_importing_torch_or_cuda(self):
        valid=_load_road_arrays(self.road(active=np.array([1,0],np.uint8)))
        self.assertEqual(valid['active'].dtype,bool)
        bads=[dict(active=np.array([.1,1.])),dict(active=np.array([2,0])),dict(meter_to_model=np.array(0.)),
              dict(meter_to_model=np.array(np.inf)),dict(meter_to_model=np.array([1.])),
              dict(gaussian_sigma_m=np.zeros((2,3))),dict(normal_world=np.zeros((2,3))),
              dict(model_xyz=np.full((2,3),np.nan)),dict(reference_srgb=np.zeros((2,))),
              dict(road_opacity=np.array([.5,1.2])),dict(model_from_world=np.diag([2,2,2,1]))]
        for change in bads:
            with self.subTest(change=list(change)),self.assertRaises(ValueError):_load_road_arrays(self.road(**change))


class SurfaceLODTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.p=Path(self.temp.name)/'posterior.npz'
        self.fields=dict(u_knots_pixels=np.array([0.,0,0,0,3,3,3,3]),v_knots_pixels=np.array([0.,0,0,0,3,3,3,3]),
                         control_delta_z_m=np.zeros((4,4)),z_reference_m=np.array(2.),reference_K=np.diag([4.,4.,1.]),
                         reference_camera_from_local_m=np.eye(4))
        self.uv=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])

    def base(self, uv=None):
        uv=self.uv.copy() if uv is None else np.asarray(uv);n=len(uv)
        xyz=np.c_[(uv+.5)*.5,np.full(n,2.)].astype(np.float32)
        return SimpleNamespace(xyz=xyz,representation_kind=np.ones(n,np.int32),resample_reference_pixel_xy=uv,
            coefficients=np.zeros((n,16,3),np.float32),sigma=np.full(n,.3,np.float32),
            opacity=np.full(n,.99,np.float32),rgb=np.tile([.2,.4,1.2],(n,1)).astype(np.float32),
            assert_geometry_unchanged=lambda:True)

    def build(self, base=None, hw=(4,4), **changes):
        np.savez(self.p,**{**self.fields,**changes})
        return SurfaceLODAsset(self.base() if base is None else base,self.p,hw)

    def test_reference_surface_material_and_identity_lod_math_are_unchanged(self):
        base=self.base();lod=self.build(base)
        xyz,du,dv,z=lod.surface(self.uv,derivatives=True)
        np.testing.assert_allclose(xyz,base.xyz,atol=1e-7)
        np.testing.assert_allclose(du,np.tile([.5,0,0],(4,1)));np.testing.assert_allclose(dv,np.tile([0,.5,0],(4,1)))
        rgb,alpha,_=lod.material(self.uv,1.)
        np.testing.assert_allclose(rgb,base.rgb);np.testing.assert_allclose(alpha,np.ones(4))
        self.assertGreater(rgb.max(),1.)  # 不增加上色裁剪/面积opacity补偿。
        sample,record=lod.sampling(np.eye(4),SimpleNamespace(full_proj_transform=np.eye(4),image_width=4,image_height=4))
        self.assertIsNone(sample);self.assertTrue(record['canonical_grid_reused_exact']);self.assertEqual(record['du'],1.)

    def test_reference_hw_and_uv_cannot_truncate_or_wrap_texture_indices(self):
        for hw in [(4.9,4),(True,4),np.array(4),(4,0)]:
            with self.subTest(hw=hw),self.assertRaises(ValueError):self.build(hw=hw)
        for uv in [np.array([[-1.,0.]]),np.array([[4.,0.]]),np.array([[.2,0.]]),np.array([[0.,0.],[0.,0.]])]:
            with self.subTest(uv=uv),self.assertRaises(ValueError):self.build(self.base(uv))

    def test_posterior_shapes_finiteness_and_positive_camera_contracts(self):
        bads=[dict(z_reference_m=np.array(0.)),dict(z_reference_m=np.array(np.nan)),dict(z_reference_m=np.array([2.])),
              dict(reference_K=np.zeros((3,3))),dict(reference_K=np.array([[4.,.1,0],[0,4,0],[0,0,1]])),
              dict(reference_camera_from_local_m=np.diag([1,1,-1,1])),
              dict(control_delta_z_m=np.zeros((2,2,4))),dict(control_delta_z_m=np.full((4,4),np.inf)),
              dict(u_knots_pixels=np.array([0.,0,0,0,3,2,3,3]))]
        for change in bads:
            with self.subTest(change=list(change)),self.assertRaises(ValueError):self.build(**change)

    def test_base_row_fields_and_queries_are_validated(self):
        for field,value in [('sigma',np.zeros(4)),('opacity',np.full(4,1.)),('rgb',np.full((4,3),np.nan)),
                            ('coefficients',np.zeros((4,0,3))),('representation_kind',np.zeros(4,np.int32))]:
            base=self.base();setattr(base,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):self.build(base)
        lod=self.build()
        for du in [0.,-1.,np.inf,np.nan]:
            with self.subTest(du=du),self.assertRaises(ValueError):lod.material(self.uv,du)
        with self.assertRaises(ValueError):lod.surface([[5.,5.]])
        with self.assertRaises(ValueError):lod.surface(1.)
        self.assertEqual(stats(np.array([]))['count'],0)

    def test_frozen_posterior_scalars_and_derived_arrays_are_checked(self):
        lod=self.build();self.assertFalse(lod.invK.flags.writeable);self.assertTrue(lod.assert_persistent_unchanged())
        lod.zref+=.1
        with self.assertRaises(RuntimeError):lod.assert_persistent_unchanged()


if __name__=='__main__':unittest.main()
