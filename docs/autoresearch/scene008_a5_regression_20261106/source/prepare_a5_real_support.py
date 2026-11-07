"""CPU-only a5 real-source support arm; preserve selected dynamic=a37."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
import copy
import hashlib
import json
import shutil
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_support_ablation_20261106_r1/support')
BASE = ROOT.parents[1] / 'scene008_a37_only_B10k_20261103_r1'
OLD = BASE / 'inputs/scene0008_089_137'
FULL = ROOT.parents[1] / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
ENGINE = ROOT.parents[1] / 'r15_baseline_owner5_20261018/training_source/scene0008_089_137'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def xyz(rows):
    return np.column_stack([rows[name] for name in ['x', 'y', 'z']]).astype(np.float64)


def keys(rows):
    # nx is the model-time channel actually used by readWaymoInfo_add.
    fields = ['x', 'y', 'z', 'nx']
    packed = np.empty(len(rows), dtype=np.dtype([(name, rows.dtype.fields[name][0]) for name in fields]))
    for name in fields:
        packed[name] = rows[name]
    return packed


def main():
    scene = ROOT / 'inputs/scene0008_089_137'
    scene.mkdir(parents=True, exist_ok=False)
    assets = ROOT / 'support_assets'
    assets.mkdir()
    watched = [OLD / 'points3d.ply', OLD / 'points3d_dynamic_0.ply', OLD / 'transform_matrix.txt',
               FULL / 'points3d_dynamic_0.ply', BASE / 'config.json']
    before = {str(path): dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha(path)) for path in watched}
    original = PlyData.read(OLD / 'points3d.ply')
    full_a5 = PlyData.read(FULL / 'points3d_dynamic_0.ply')
    dynamic = PlyData.read(OLD / 'points3d_dynamic_0.ply')
    old_rows, a5_rows, dynamic_rows = original['vertex'].data, full_a5['vertex'].data, dynamic['vertex'].data
    assert len(old_rows) == 600000 and len(a5_rows) == 21358 and len(dynamic_rows) == 98452
    assert old_rows.dtype == a5_rows.dtype
    assert [element.name for element in original.elements] == ['vertex']
    a5_keys = keys(a5_rows)
    _, first = np.unique(a5_keys, return_index=True)
    source_order = np.sort(first)
    unique_a5 = a5_keys[source_order]
    present = np.isin(unique_a5, keys(old_rows))
    missing = source_order[~present]
    assert len(source_order) == 21358 and int(present.sum()) == 1442 and len(missing) == 19916
    extra = a5_rows[missing].copy()
    combined = np.concatenate((old_rows, extra))
    assert combined[:600000].tobytes() == old_rows.tobytes()
    assert combined[600000:].tobytes() == a5_rows[missing].tobytes()
    added_element = PlyElement.describe(combined, 'vertex', comments=original['vertex'].comments)
    PlyData([added_element], text=original.text, byte_order=original.byte_order,
            comments=original.comments, obj_info=original.obj_info).write(scene / 'points3d.ply')
    reread = PlyData.read(scene / 'points3d.ply')['vertex'].data
    assert reread.dtype == old_rows.dtype and reread.tobytes() == combined.tobytes()
    for name in ['calib', 'image', 'pose', 'sky', 'velodyne', 'annotations', 'timestamp',
                 'dynamic_image', 'dynamic_masks', 'mask', 'dynamic_actor_assets']:
        assert (OLD / name).exists()
        (scene / name).symlink_to(OLD / name, target_is_directory=(OLD / name).is_dir())
    shutil.copy2(OLD / 'transform_matrix.txt', scene / 'transform_matrix.txt')
    shutil.copy2(OLD / 'points3d_dynamic_0.ply', scene / 'points3d_dynamic_0.ply')
    assert sha(scene / 'points3d_dynamic_0.ply') == sha(OLD / 'points3d_dynamic_0.ply')
    assert list(scene.glob('points3d_dynamic_*.ply')) == [scene / 'points3d_dynamic_0.ply']
    assert np.isin(unique_a5, keys(reread)).all()

    tree = cKDTree(xyz(dynamic_rows))
    old_distance, _ = tree.query(xyz(old_rows), eps=0, distance_upper_bound=.1, workers=4)
    extra_distance, _ = tree.query(xyz(extra), eps=0, distance_upper_bound=.1, workers=4)
    keep_old, keep_extra = old_distance >= .1, extra_distance >= .1
    retained = reread[np.concatenate((keep_old, keep_extra))]
    assert int(keep_old.sum()) == 591672 and int(keep_extra.sum()) == 19916
    assert retained[:591672].tobytes() == old_rows[keep_old].tobytes()
    assert retained[591672:].tobytes() == extra.tobytes()
    expected_static = len(retained)
    assert expected_static == 611588

    prep = json.loads((OLD / 'preparation_manifest.json').read_text())
    prep['support_ablation'] = dict(arm='support', actor_alias=5, selected_dynamic_aliases=[37],
        original_scene=str(OLD), added_real_static_rows=19916, actor5_added_to_dynamic=False)
    write(scene / 'preparation_manifest.json', prep)
    for name in ['dynamic_multi_actor_preparation.json', 'dynamic_multi_actor_validation.json']:
        value = json.loads((OLD / name).read_text())
        assert len(value['actors']) == 1 and value['actors'][0]['raw_track_id'] == 'vBOasuXpjI7h7d-X_VEQdA'
        value.update(scene=str(scene), static_support_ablation='Only missing real a5 source rows appended to original sampled static scene; dynamic identity unchanged')
        write(scene / name, value)
    source_index_archive = np.load(FULL / 'dynamic_actor_assets/actor_0/source_lidar_indices.npz', allow_pickle=False)
    frame_ids, lidar_ids = [], []
    for frame in prep['frames']:
        ids = source_index_archive[frame['id']]
        frame_ids.extend([frame['id']] * len(ids))
        lidar_ids.extend(ids.tolist())
    assert len(frame_ids) == len(lidar_ids) == 21358
    np.savez_compressed(assets / 'added_a5_source_indices.npz', full_a5_ply_row_indices=missing,
        frame_ids=np.asarray(frame_ids)[missing], source_lidar_indices=np.asarray(lidar_ids, dtype=np.int32)[missing],
        model_time=extra['nx'], xyz=xyz(extra).astype(np.float32), retained_by_original_a37_exclusion=keep_extra)
    config = json.loads((BASE / 'config.json').read_text())
    original_config = copy.deepcopy(config)
    config.update(source_path=str(scene), model_path=str(ROOT / 'fresh5k'))
    config['followup_scene_spec']['source_path'] = str(scene)
    config['followup_scene_spec']['actor_ply_paths'] = [str(scene / 'points3d_dynamic_0.ply')]
    assert config['num_pts'] == 600000 and config['iterations'] == config['position_lr_max_steps'] == 30000
    assert config['start_checkpoint'] is None and config['initialization'] is False
    changes = {name: dict(baseline=original_config[name], support=config[name]) for name in config if config[name] != original_config[name]}
    assert set(changes) == {'source_path', 'model_path', 'followup_scene_spec'}
    write(ROOT / 'config.json', config)
    identity = json.loads((BASE / 'identity_contract.json').read_text())
    identity.update(task_id='SCENE008-A5-REAL-STATIC-SUPPORT-20261106-R1', source_scene=str(scene),
        dynamic_ply_files=config['followup_scene_spec']['actor_ply_paths'], formal_updates=5000,
        expected_initial_static_rows=expected_static, expected_initial_dynamic_rows=98452,
        dynamic_source_real_points=[98452],
        added_static_support_alias=5, added_static_support_rows=19916, a5_in_dynamic=False,
        retained_representation='Original native temporal-static branch plus missing real a5 source rows; all original600K records preserved.',
        config_changes=changes, inherited_masks_and_source_identity_unchanged=True)
    write(ROOT / 'identity_contract.json', identity)
    engine_sha = {str(path): sha(path) for path in [ENGINE / 'train.py', ENGINE / 'scene/gaussian_model.py',
                  ENGINE / 'gaussian_renderer/__init__.py', ENGINE / 'scene/waymo_loader2.py']}
    assert engine_sha[str(ENGINE / 'scene/gaussian_model.py')] == 'e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'
    assert engine_sha[str(ENGINE / 'gaussian_renderer/__init__.py')] == '35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
    after = {str(path): dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha(path)) for path in watched}
    assert before == after
    contract = dict(status='PASS_CPU_A5_REAL_STATIC_SUPPORT_INPUT_READY', arm='support', root=str(ROOT), source_path=str(scene),
        engine=str(ENGINE), source_engine_sha256=engine_sha, no_time_or_radius_patch=True,
        baseline_source=str(OLD), original_scene_rows=600000, original_scene_all_rows_retained=True,
        original_scene_vertex_prefix_byte_exact=True, original_native_static_rows=591672,
        full_a5_source=str(FULL / 'points3d_dynamic_0.ply'), full_a5_source_rows=21358, full_a5_unique_xyz_time=21358,
        existing_a5_unique_xyz_time=1442, existing_a5_sampled_rows=1496, added_missing_real_rows=19916,
        xyz_time_match_fields=['x', 'y', 'z', 'nx'], model_time_channel='nx (the original loader uses normals[:,0]); all original fields including time and RGB are copied verbatim',
        added_rows_retained_by_original_dynamic_exclusion=19916, added_rows_removed_by_original_dynamic_exclusion=0,
        dynamic_exclusion='Original strict nearest XYZ distance<0.1m to selected a37 full prior; no actor-specific gate or bypass',
        new_points3d_rows=619916, expected_initial_static_rows=611588, expected_initial_dynamic_rows=98452,
        original_static_retained_prefix_byte_exact=True, added_vertex_rows_byte_exact_to_full_a5_source=True,
        dynamic_actor_aliases=[37], a5_is_static=True, a5_selected_for_delete=False,
        dynamic_PLY_and_mask_RGB_assets_unchanged=True, num_pts_configuration=600000,
        native_static_PLY_resampled=False,
        loader_rule='readWaymoInfo_add reads existing points3d.ply, applies original selected-dynamic0.1m exclusion, and returns that static cloud. Its separate600K np.random.choice on rebuilt LiDAR is not used for static initialization when initialization=False.',
        transform_new_independent_copy=True, fresh_start=True, formal_stop=5000, learning_rate_horizon=30000,
        capacity_delta_original_PLY_fraction=19916/600000, capacity_delta_native_static_fraction=19916/591672,
        old_xyz_RGB_time_order_preserved=True, all_old_initial_parameters_identical_claimed=False,
        KNN_scale_note='Unmodified native distCUDA2 re-evaluates nearest-neighbor scales with the added support; nearby old Gaussian scales may change naturally. No manual scale restoration or other parameter patch is applied.',
        initialization_color_note='Original full a5 PLY RGB fields are all0; they are copied exactly. The unmodified native loader initializes BasicPointCloud colors to0 for both arms. No RGB or point is synthesized from a rendering.',
        scientific_scope='Specific real a5 initialization-support intervention, including local count/density and capacity changes; not a same-point-count geometry superiority claim.',
        source_indices_asset=str(assets / 'added_a5_source_indices.npz'), config_changes=changes,
        source_files_before=before, source_files_after=after, GPU_used=False, training_updates=0,
        automatic_shutdown_allowed=False, additional_approval_gate=False)
    write(ROOT / 'support_contract.json', contract)
    validation = dict(status=contract['status'], GPU_used=False, training_updates=0,
        prefix600K_all_vertex_bytes_exact=True, appended19916_all_vertex_bytes_exact=True,
        full_unique_a5_support_covered_after_merge=True, native_static_prefix591672_all_vertex_bytes_exact=True,
        original_0p1m_rule_used=True, added_retained=19916, added_removed=0,
        expected_initial_static_rows=expected_static, expected_initial_dynamic_rows=98452,
        selected_dynamic_PLY_files=[str(path) for path in scene.glob('points3d_dynamic_*.ply')],
        original_source_bytes_mtimes_unchanged=True,
        config_sha256=sha(ROOT / 'config.json'), identity_sha256=sha(ROOT / 'identity_contract.json'),
        points3d_sha256=sha(scene / 'points3d.ply'), support_contract_sha256=sha(ROOT / 'support_contract.json'),
        source_indices_sha256=sha(assets / 'added_a5_source_indices.npz'),
        auxiliary_time_field_equals_nx_original=bool(np.array_equal(old_rows['time'], old_rows['nx'])),
        auxiliary_time_field_equals_nx_added=bool(np.array_equal(extra['time'], extra['nx'])))
    write(ROOT / 'cpu_validation.json', validation)
    print(json.dumps({key: value for key, value in validation.items() if key != 'selected_dynamic_PLY_files'}, indent=2))


if __name__ == '__main__':
    main()
