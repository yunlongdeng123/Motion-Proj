"""用真实输入行和持久谱系识别旧 v44 static 的 a5 后代；不读渲染 mask。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

CAM = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN = CAM / 'scene008_blue_global_rgb_probe_20261105_r1/global_10k'
OUT = CAM / 'scene008_v44_full5_integration_20261107_r1/review/legacy_static_actor5'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    assert not torch.cuda.is_initialized()
    config = json.loads((RUN / 'effective_config.json').read_text())
    source = Path(config['source_path'])
    full_a5 = CAM / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137/points3d_dynamic_0.ply'
    cloud = o3d.io.read_point_cloud(str(source / 'points3d.ply'))
    xyz = np.asarray(cloud.points)
    time = np.asarray(cloud.normals)[:, :1]
    vertices = PlyData.read(str(source / 'points3d.ply'))['vertex'].data
    assert np.array_equal(xyz, np.column_stack([vertices[k] for k in ['x', 'y', 'z']]))
    assert np.array_equal(time[:, 0], vertices['nx'])
    actor = PlyData.read(str(full_a5))['vertex'].data
    actor4 = np.column_stack([actor[k] for k in ['x', 'y', 'z', 'nx']]).astype(np.float64)
    source4 = np.column_stack([xyz, time])
    witness = {tuple(value): index for index, value in enumerate(actor4)}
    assert len(actor4) == len(witness) == 21358
    witness_rows = np.fromiter((witness.get(tuple(value), -1) for value in source4), dtype=np.int64, count=len(source4))
    source_actor5_rows = np.flatnonzero(witness_rows >= 0)
    assert len(source_actor5_rows) == 1496
    assert len(np.unique(source4[source_actor5_rows], axis=0)) == 1442
    assert np.array_equal(source4[source_actor5_rows], actor4[witness_rows[source_actor5_rows]])
    dynamic_paths = sorted(source.glob('points3d_dynamic_*.ply'))
    dynamic = np.concatenate([np.asarray(o3d.io.read_point_cloud(str(path)).points) for path in dynamic_paths])
    assert len(xyz) == 600000 and len(dynamic) == 98452 and len(dynamic_paths) == 1
    nearest, _ = cKDTree(dynamic).query(xyz, k=1, eps=0, distance_upper_bound=.1, workers=1)
    keep = nearest >= .1
    assert int(keep.sum()) == 591672 and bool(keep[source_actor5_rows].all())
    original_static_source_rows = np.flatnonzero(keep)
    source_to_static = np.full(len(xyz), -1, dtype=np.int64)
    source_to_static[original_static_source_rows] = np.arange(591672, dtype=np.int64)
    initial_actor5_ids = source_to_static[source_actor5_rows]
    transform = np.loadtxt(source / 'transform_matrix.txt')
    initial_xyz = (np.pad(xyz[keep], ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3].astype(np.float32)
    initial_time = time[keep].astype(np.float32)
    assert digest(initial_xyz) == '7598f1556be5c128330365999bcb072b89100b0448e5b570a4a92a43b8e8f471'
    assert digest(initial_time) == '7eb38e589a7f17cf67d7fe0e65231df449fbf24547dbcdb07eec462ec15354fa'
    assignment = CAM / 'scene008_a6_a37_dynamic_20261104_r1/inputs/init_assignment.npz'
    assert sha(assignment) == '04671379a727ddca166b54030588cf865acf5aea2b947c9f6c9deeb3016bb0c1'
    with np.load(assignment, allow_pickle=False) as archive:
        moved = archive['original_static_indices'].astype(np.int64)
    assert len(moved) == 1023 and len(np.unique(moved)) == 1023
    assert len(np.intersect1d(moved, initial_actor5_ids)) == 0
    retained = np.ones(591672, dtype=bool)
    retained[moved] = False
    transfer = json.loads((RUN / 'initial_parameter_transfer.json').read_text())
    assert digest(initial_xyz[retained]) == transfer['per_field']['_xyz']['remaining_static_sha256']
    assert digest(initial_time[retained]) == transfer['per_field']['_t']['remaining_static_sha256']
    payload, step = torch.load(str(RUN / 'chkpnt10000.pth'), map_location='cpu', weights_only=False, mmap=True)
    lineage = payload[14]['sgcm_lineage_v1']
    assert step == 10000 and lineage['schema_version'] == 1
    ids = lineage['lineage_ids'].numpy()
    assert ids.dtype == np.int64 and ids.shape == (len(payload[1]),) and len(ids) == 1042776
    assert ids.min() >= 0 and ids.max() < 591672 and not np.isin(ids, moved).any()
    retirement = np.flatnonzero(np.isin(ids, initial_actor5_ids))
    current_seed_ids, descendant_counts = np.unique(ids[retirement], return_counts=True)
    road = CAM / 'scene008_blue_global_road1k_20261105_r1/formal/checkpoint001000.pt'
    road_payload = torch.load(str(road), map_location='cpu', weights_only=False)
    gate_ids = road_payload['repair']['ids'].numpy()
    assert gate_ids.min() >= 0 and gate_ids.max() < len(ids)
    with np.load(road.parent.parent / 'inputs/contributors.npz', allow_pickle=False) as archive:
        assert np.array_equal(gate_ids, archive['ids'])
    gate_intersection = np.intersect1d(retirement, gate_ids)
    OUT.mkdir(exist_ok=False)
    asset = OUT / 'strict_actor5_static_lineage.npz'
    np.savez_compressed(asset, source_ply_row_indices=source_actor5_rows,
        full_actor5_source_witness_rows=witness_rows[source_actor5_rows],
        initial_static_lineage_ids=initial_actor5_ids, current_static_retirement_rows=retirement,
        current_actor5_lineage_ids=current_seed_ids, current_actor5_descendant_counts=descendant_counts,
        road_gate_intersection_rows=gate_intersection)
    result = dict(status='PASS_EXACT_SOURCE_ROWS_AND_PERSISTENT_STATIC_ACTOR5_LINEAGE',
        native_parent=str(RUN), source_path=str(source), full_actor5_source=str(full_a5),
        source_points_rows=len(xyz), original_static_before_a6_transfer_rows=int(keep.sum()),
        actual_RCR_static_initial_rows_after_a6_transfer=int(retained.sum()),
        full_a5_rows=len(actor4), exact_actor5_sampled_rows=len(source_actor5_rows),
        exact_actor5_unique_xyz_time=1442, actor5_samples_removed_by_original_dynamic_exclusion=0,
        actor5_samples_moved_to_dynamic_a6=0, initial_xyz_hash_matches_executed_training_assert=True,
        initial_time_hash_matches_executed_training_assert=True, after_a6_transfer_xyz_time_hashes_exact=True,
        source_actor5_initial_static_lineage_examples=initial_actor5_ids[:20].tolist(),
        current_static_rows=len(ids), current_static_actor5_descendant_rows=len(retirement),
        surviving_actor5_seed_lineages=len(current_seed_ids),
        pruned_actor5_seed_lineages=len(initial_actor5_ids) - len(current_seed_ids),
        current_static_lineage_sha256=digest(ids),
        current_static_actor5_retirement_rows_sha256=digest(retirement),
        road_gate_count=len(gate_ids), road_gate_actor5_descendant_intersection=len(gate_intersection),
        gate_application_order='Original gate index_copy on original row indices first; then only render mask intersects NOT retired rows; no reindex or opacity mutation',
        evidence_asset=str(asset), evidence_asset_sha256=sha(asset),
        source_file_sha256={str(p): sha(p) for p in [source / 'points3d.ply', *dynamic_paths, full_a5,
            source / 'transform_matrix.txt', assignment, RUN / 'initial_parameter_transfer.json',
            RUN.parent / 'source/dual_actor_runtime.py', RUN.parent / 'source/train_loss_probe.py']},
        static_checkpoint_identity=dict(path=str(RUN / 'chkpnt10000.pth'),
            bytes=(RUN / 'chkpnt10000.pth').stat().st_size, mtime_ns=(RUN / 'chkpnt10000.pth').stat().st_mtime_ns),
        semantic_scope='Only descendants of exact real-source a5 points are identified; no label is inferred for other static rows.',
        no_predicted_mask_or_spatial_neighbour_owner_assignment=True, CUDA_initialized=torch.cuda.is_initialized(),
        optimizer_updates=0)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_file_sha256'}, indent=2))


if __name__ == '__main__':
    main()
