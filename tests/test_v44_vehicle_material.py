"""Only CPU equations/metadata tests. Native CUDA collection/SpMM remain unrun."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse import csr_matrix, eye
from v44_deterministic_csr import Matrix, validated_csr
from v44_linear_weights import reference_weights, Kernel
from v44_vehicle_material import (source_camera_header, source_mask, observation_files,
                                   validate_design, fit_rgb, parser, design_binding)


class MaterialMathTests(unittest.TestCase):
    def test_inherited_ray_transmittance_and_termination(self):
        matrix = reference_weights(np.array([0,2,3]),np.array([0,1,1]),np.array([.5,.5,.2]),2)
        np.testing.assert_allclose(matrix.toarray(),[[.5,.25],[0,.2]],atol=1e-15)
        np.testing.assert_allclose(np.asarray(matrix.sum(1)).ravel(),[.75,.2])
        terminated = reference_weights(np.array([0,3]),np.array([0,1,2]),np.array([.99,.99,.99]),3)
        self.assertEqual(terminated[0,2],0.)
        with self.assertRaises(ValueError):reference_weights(np.array([0,1]),np.array([0]),np.array([1.]),1)

    def test_csr_reference_matches_explicit_rgb_and_rejects_dtype_and_nonfinite(self):
        A=csr_matrix(np.array([[1.,.25],[0.,2.]],np.float32));x=np.array([[.1,.2,.3],[.4,.5,.6]],np.float32)
        matrix=Matrix(A,device='cpu');np.testing.assert_allclose(matrix(x),A.toarray()@x)
        with self.assertRaises(ValueError):matrix(x.astype(np.float64))
        with self.assertRaises(ValueError):validated_csr(csr_matrix([[np.nan]]))
        bad=A.copy();bad.indices=bad.indices.astype(float)+.5
        with self.assertRaises(ValueError):validated_csr(bad)

    def test_1000_fista_steps_match_ridge_closed_form_on_fixed_geometry(self):
        A=eye(2,format='csr',dtype=np.float32);target=np.array([[.1,.8,.3],[.9,.2,.5]],np.float32)
        prior=np.full((2,3),.5,np.float32);checkpoints=[]
        fitted,receipt=fit_rgb(A,target,prior,device='cpu',checkpoint=lambda state:checkpoints.append(state))
        expected=(target+.001*prior)/1.001
        np.testing.assert_allclose(fitted,expected,atol=2e-6)
        self.assertEqual(receipt['steps'],1000);self.assertEqual(receipt['ridge'],.001)
        self.assertEqual([c['step'] for c in checkpoints],[500,1000]);self.assertEqual(len(receipt['objective_trace']),10)
        self.assertEqual(receipt['backend'],'scipy_cpu_reference')

    def test_unobserved_columns_preserve_prior_exactly(self):
        A=csr_matrix(np.array([[1,0,0],[0,1,0]],np.float32));target=np.array([[0,1,.2],[.2,.3,.4]],np.float32)
        prior=np.array([[.5,.5,.5],[.5,.5,.5],[.123,.234,.345]],np.float32);snapshot=prior.copy()
        fitted,receipt=fit_rgb(A,target,prior,device='cpu')
        np.testing.assert_array_equal(fitted[2],prior[2]);np.testing.assert_array_equal(prior,snapshot)
        self.assertEqual(receipt['unknown_columns'],1)

    def test_source_grid_from_actual_camera_size_preserves_clock_and_projection(self):
        saved=dict(width=480,height=320,fx=520.,fy=521.,cx=240.,cy=160.,timestamp=-.92,
                   image_name='opaque-image-name',frame_key='not-a-raw-number',W=np.eye(4))
        native=source_camera_header(saved,dict(camera_size=[1920,1280]))
        self.assertEqual((native['width'],native['height']),(1920,1280));self.assertEqual(native['fx'],2080.)
        self.assertEqual(native['timestamp'],-.92);np.testing.assert_array_equal(native['W'],saved['W'])
        self.assertEqual(saved['width'],480)
        self.assertEqual(native['fx']/native['width'],saved['fx']/saved['width'])

    def test_mask_erosion_and_resize_match_old_repeat_without_fixed_actor_id(self):
        labels=np.zeros((8,9),np.int64);labels[1:7,1:8]=701
        native,coarse=source_mask(labels,701,(9,8),(36,32))
        expected=binary_erosion(labels==701,iterations=1)
        np.testing.assert_array_equal(native,np.repeat(np.repeat(expected,4,axis=0),4,axis=1))
        self.assertEqual(coarse,int(expected.sum()))
        wrong,_=source_mask(labels,29,(9,8),(36,32));self.assertFalse(wrong.any())

    def test_observation_mapping_uses_image_names_not_three_digit_frame_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            bank=Path(td);(bank/'observations').mkdir()
            path=bank/'observations/UUID-source.json';path.write_text(json.dumps(dict(image_path='image/image_4/source-UUID.png')))
            frame=SimpleNamespace(key='arbitrary-key',image_name='source-UUID',observation=None)
            paths=observation_files(SimpleNamespace(frames=(frame,)),bank)
            self.assertEqual(paths['arbitrary-key'],path.resolve())
            explicit=SimpleNamespace(key=-777,image_name='anything',observation=path)
            self.assertEqual(observation_files(SimpleNamespace(frames=(explicit,)),bank)[-777],path)

    def test_bad_design_or_target_cannot_be_silently_fit(self):
        valid=csr_matrix([[1.,0.],[0.,1.]])
        for A,target,prior in [(valid,np.ones((1,3)),np.ones((2,3))),
            (csr_matrix([[-1.]]),np.zeros((1,3)),np.zeros((1,3))),
            (valid,np.full((2,3),np.nan),np.zeros((2,3))),
            (valid,np.zeros((2,3)),np.full((2,3),1.1))]:
            with self.subTest(shape=A.shape),self.assertRaises(ValueError):validate_design(A,target,prior)

    def test_cli_declares_portable_inputs_and_only_optional_cached_design(self):
        args=parser().parse_args(['--manifest','model.json','--bank','bank','--geometry','geometry.npz',
                                 '--track-id','raw UUID with spaces','--actor-id','701','--output','new-dir'])
        self.assertEqual(args.track_id,'raw UUID with spaces');self.assertEqual(args.actor_id,701)
        self.assertIsNone(args.design_source)

    def test_cached_design_binds_camera_clock_and_world_transform(self):
        with tempfile.TemporaryDirectory() as td:
            model=Path(td);(model/'cameras.json').write_text('saved-camera-A');(model/'actual_scene_binding.json').write_text('saved-clock')
            frame=SimpleNamespace(key='UUID-key',image_name='source-A',model_time=-.2,source_time_s=123.45)
            m=SimpleNamespace(model=model,frames=(frame,),world_to_model=np.eye(4))
            first=design_binding(m,'original-track',701,'geometry-hash')
            (model/'cameras.json').write_text('saved-camera-B')
            self.assertNotEqual(first,design_binding(m,'original-track',701,'geometry-hash'))
            (model/'cameras.json').write_text('saved-camera-A');m.world_to_model[0,3]=2.
            self.assertNotEqual(first,design_binding(m,'original-track',701,'geometry-hash'))

    def test_replay_count_and_gather_declare_64bit_stream_pointer(self):
        import ctypes
        class Fn: pass
        fake=SimpleNamespace(count_pass=Fn(),gather_pass=Fn())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'replay.so';path.touch()
            with patch('v44_linear_weights.C.CDLL',return_value=fake):Kernel(path)
        self.assertEqual(len(fake.count_pass.argtypes),17)
        self.assertEqual(len(fake.gather_pass.argtypes),20)
        self.assertIs(fake.count_pass.argtypes[-1],ctypes.c_void_p)


if __name__=='__main__':unittest.main()
