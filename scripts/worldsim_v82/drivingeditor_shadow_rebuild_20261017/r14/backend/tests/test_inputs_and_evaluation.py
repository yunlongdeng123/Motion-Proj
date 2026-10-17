import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch
from shadow_rebuild.prepare import build
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import preflight
from shadow_rebuild.config import Config
from shadow_rebuild.model import SceneModel
from shadow_rebuild.evaluate import evaluate
from fixture import create


def raw_fixture(root):
    scene, prepared = root/"raw", root/"reviewed"
    for folder in ("image/image_0", "calib", "pose", "velodyne"):
        (scene/folder).mkdir(parents=True)
    (prepared/"instance_masks").mkdir(parents=True)
    ids = [f"{i:04d}" for i in range(4)]
    xyz = np.array([[x, y, .015*x*x] for x in np.linspace(-2, 2, 21) for y in np.linspace(-2, 2, 21)], np.float32)
    annotations = []
    for i, fid in enumerate(ids):
        image = np.full((32, 32, 3), 130, np.uint8)
        center = -.7+i*.4
        # 所有标注/测量均明确为测试夹具；用于检查坐标与数据路径。
        mask = np.zeros((32, 32), np.uint16)
        px = int(round(16+16*center/3.4))
        mask[14:18, px-2:px+3] = 1
        image[mask > 0] = [180, 40, 30]
        Image.fromarray(image).save(scene/"image/image_0"/(fid+".png"))
        Image.fromarray(mask).save(prepared/"instance_masks"/(fid+".png"))
        np.savetxt(scene/"pose"/(fid+".txt"), np.eye(4))
        (scene/"calib"/(fid+".txt")).write_text("P0: 16 0 15.5 0 0 16 15.5 0 0 0 1 0\nTr_velo_to_cam_0: 1 0 0 0 0 -1 0 0 0 0 -1 4\n")
        body = np.array([[center+x, y, .6+z] for x in (-.12, .12) for y in (-.12, .12) for z in (-.1, .1)], np.float32)
        points = np.concatenate((xyz, body))
        np.pad(points, ((0, 0), (0, 3))).astype(np.float32).tofile(scene/"velodyne"/(fid+".bin"))
        label = dict(type=1, id="fixture_track", box=dict(center_x=center, center_y=0, center_z=.6,
                         length=.5, width=.5, height=.5, heading=0))
        annotations.append(dict(frame_id=fid, source_index=i, laser_labels=[label], frame_pose_world_from_vehicle=np.eye(4).ravel().tolist()))
    (scene/"preparation_manifest.json").write_text(json.dumps(dict(frames=[dict(id=fid, source_index=i) for i, fid in enumerate(ids)])))
    annotation_path = root/"annotations.json"
    annotation_path.write_text(json.dumps(dict(frames=annotations)))
    np.savez_compressed(prepared/"instances.npz", instance_ids=np.array(["fixture_track"]), frame_ids=np.array(ids), mask_known=np.ones((4, 1), bool))
    return argparse.Namespace(scene=str(scene), reviewed_instances=str(prepared), annotations=str(annotation_path),
                              output=str(root/"data"), test_raw=[2], downsample=1, static_voxel=.12, actor_voxel=.04, max_depth=80)


class InputContract(unittest.TestCase):
    def test_raw_adapter_nonplanar_initialization_and_no_test_RGB_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = raw_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                build(args)
            data = Dataset(root/"data")
            road = data.seed["xyz"][data.seed["road"]]
            self.assertGreater(np.ptp(road[:, 2]), .03)
            receipt = preflight(data, Config())
            self.assertEqual(receipt["train_frames"], 3)
            self.assertEqual(receipt["test_frames"], 1)
            self.assertGreater(receipt["reference_points"], 0)
            Image.fromarray(np.full((32, 32, 3), 255, np.uint8)).save(Path(args.scene)/"image/image_0/0002.png")
            args.output = str(root/"changed_test")
            with contextlib.redirect_stdout(io.StringIO()):
                build(args)
            second = Dataset(root/"changed_test")
            for key, value in data.seed.items():
                np.testing.assert_array_equal(value, second.seed[key], err_msg=key)
            # 不准覆盖原始或已有数据。
            with self.assertRaises(FileExistsError):
                build(args)

    def test_missing_real_mask_does_not_turn_into_known_background(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = raw_fixture(root)
            prepared = Path(args.reviewed_instances)
            with np.load(prepared/"instances.npz") as f:
                values = dict(f)
            values["mask_known"][1, 0] = False
            np.savez_compressed(prepared/"instances.npz", **values)
            with contextlib.redirect_stdout(io.StringIO()):
                build(args)
            dataset = Dataset(root/"data")
            with self.assertRaisesRegex(ValueError, "mask"):
                dataset.frame(1)
            allowed = Dataset(root/"data", allow_partial_masks=True)
            self.assertFalse(bool(allowed.frame(1).mask_known[0]))

    def test_reject_learned_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"fixture"
            create(root)
            path = root/"manifest.json"
            value = json.loads(path.read_text())
            value["learned_checkpoint_used"] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                Dataset(root, synthetic=True)


class Evaluation(unittest.TestCase):
    def test_multi_actor_dynamic_branches_and_each_actor_edit_matrix(self):
        from shadow_rebuild.multi_evaluate import evaluate_many,stream_table
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=create(root/'fixture')
            seed=dict(np.load(root/'fixture/seed.npz'));n=len(seed['xyz']);body=seed['owner']==0
            for key,value in list(seed.items()):
                extra=value[body].copy()
                if key=='owner':extra[:]=1
                if key=='neighbor':extra=np.arange(n,n+body.sum())
                seed[key]=np.concatenate((value,extra))
            np.savez_compressed(root/'fixture/seed.npz',**seed)
            trajectory=dict(np.load(root/'fixture/trajectory.npz'))
            second=trajectory['poses'].copy();second[:,:,0,3]+=.7
            trajectory['poses']=np.concatenate((trajectory['poses'],second),axis=1)
            trajectory['valid']=np.repeat(trajectory['valid'],2,axis=1)
            np.savez_compressed(root/'fixture/trajectory.npz',**trajectory)
            path=root/'fixture/manifest.json';manifest=json.loads(path.read_text());manifest['actor_ids'].append('synthetic_auxiliary')
            path.write_text(json.dumps(manifest))
            for row in manifest['frames']:
                path=root/'fixture'/row['supervision'];sup=dict(np.load(path))
                sup['masks']=np.concatenate((sup['masks'],np.zeros_like(sup['masks'])),axis=-1)
                sup['mask_known']=np.array([True,False])
                np.savez_compressed(path,**sup)
            dataset=Dataset(root/'fixture',synthetic=True,allow_partial_masks=True)
            model=SceneModel(dataset.seed,config)
            rows=evaluate_many(model,dataset,root/'eval',[0,1],backend='reference',videos=False)
            self.assertEqual(len(rows),5)
            self.assertEqual(len(list((root/'eval').rglob('*.png'))),len(stream_table([0,1]))*5)
            for actor in (0,1):
                for case in ('delete','translate','yaw15','yaw30','isolated_rgb','isolated_alpha','visible_alpha'):
                    self.assertEqual(len(list((root/f'eval/actors/a{actor:02d}'/case).glob('*.png'))),5)
            table=json.loads((root/'eval/actor_table.json').read_text())
            self.assertEqual([a['raw_track_id'] for a in table['actors']],list(dataset.actor_ids))
            self.assertTrue((root/'eval/dynamic/visible_rgb/f000.png').is_file())
            self.assertTrue((root/'eval/dynamic/isolated_rgb/f000.png').is_file())

    def test_continuous_edit_streams_delete_identity_fixed_camera_and_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = create(root/"fixture")
            dataset = Dataset(root/"fixture", synthetic=True)
            model = SceneModel(dataset.seed, config)
            records = evaluate(model, dataset, root/"eval", backend="reference", videos=False)
            self.assertEqual(len(records), 5)
            self.assertEqual(len(list((root/"eval").rglob("*.png"))), 40)
            self.assertEqual(max(r["delete_all_shadow_off_max"] for r in records), 0)
            self.assertEqual(max(r["fixed_static_max"] for r in records), 0)
            self.assertTrue(all(r["body_pixels"] > 0 for r in records))
            self.assertTrue(all(r["road_pixels"] > 0 for r in records))
            verdict = json.loads((root/"eval/evaluation.json").read_text())
            self.assertIsNone(verdict["human_verdict"])
            self.assertIsNone(verdict["problem_solved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
