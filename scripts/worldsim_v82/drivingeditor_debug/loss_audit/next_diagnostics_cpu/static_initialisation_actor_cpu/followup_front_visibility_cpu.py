"""Separate cheap pin-preserving frustum context for the169 native witnesses.

Pinhole projection through existing input poses/calibration only, no rendering.
This never edits or reruns the frozen canonical initialization audit.
"""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
DATA = BASE / 'data/waymo_paper_scenes/scene0017_000_048'
OUT = BASE / 'campaign/vgpu_bugfix_20261012/loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'
CANONICAL_SHA = 'df97e5d48ee2be553b54e127dbeff2be12dafa8561c93d6bc87bc630059b8ecd'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    canonical = OUT / 'static_initialisation_actor_audit.json'
    if sha(canonical) != CANONICAL_SHA:
        raise ValueError('Frozen canonical audit changed')
    audit = json.loads(canonical.read_text())
    prep = json.loads((DATA / 'preparation_manifest.json').read_text())
    if sha(audit['lineage_map_path']) != audit['lineage_map_SHA256']:
        raise ValueError('Frozen lineage map changed')
    with np.load(audit['lineage_map_path']) as all_points:
        selected = all_points['static_keep'] & all_points['semantic_linked_target']
        xyz = all_points['world_xyz'][selected].astype(np.float64)
        frame_origin = all_points['frame_index'][selected]
        time_origin = all_points['model_time'][selected].astype(np.float64)
        raw_rows = all_points['global_raw_row'][selected]
        sample_ids = all_points['sample_index'][selected]
        original_front = all_points['front_in_frustum'][selected]
        original_mask = all_points['front_actor_mask'][selected]
    if len(xyz) != 169 or np.count_nonzero(original_mask) != 0:
        raise ValueError('Canonical native witness count/domain differs')
    eligibility = np.zeros((len(xyz), 49), dtype=bool)
    original_temporal_maxalpha_eligible = np.zeros_like(eligibility)
    cells = []
    t_init = 0.1
    sigma_sq = (prep['frames'][-1]['model_time'] - prep['frames'][0]['model_time']) * t_init
    for fi, frame in enumerate(prep['frames']):
        pose_path = DATA / 'pose' / (frame['id'] + '.txt')
        calib_path = DATA / 'calib' / (frame['id'] + '.txt')
        for p in (pose_path, calib_path):
            if sha(p) != audit['pins'][str(p)]:
                raise ValueError('Original camera input changed')
        pose = np.loadtxt(pose_path)
        calib = {line.split(':', 1)[0]: np.fromstring(line.split(':', 1)[1], sep=' ') for line in calib_path.read_text().splitlines() if ':' in line}
        body = np.column_stack((xyz, np.ones(len(xyz)))) @ np.linalg.inv(pose).T
        camera = body @ calib['Tr_velo_to_cam_0'].reshape(3, 4).T
        positive = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.01)
        projected = np.column_stack((camera, np.ones(len(xyz)))) @ calib['P0'].reshape(3, 4).T
        uv = np.full((len(xyz), 2), -100000., dtype=np.float64)
        uv[positive] = np.floor(projected[positive, :2] / projected[positive, 2:3] + 0.5)
        width, height = frame['image_size']
        eligible = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        eligibility[:, fi] = eligible
        marginal = np.exp(-0.5 * (time_origin - frame['model_time']) ** 2 / sigma_sq)
        original_temporal_maxalpha_eligible[:, fi] = eligible & (marginal > 0.05) & (0.01 * marginal >= 1 / 255)
        cells.append({'camera': 'image_0/' + frame['id'], 'input_model_time': frame['model_time'], 'native_witness_centers_in_front_pinhole': int(eligible.sum()), 'eligible_unique_original_returns': int(len(np.unique(raw_rows[eligible]))), 'original_initial_temporal_filter_and_maxalpha_upper_bound_eligible': int(original_temporal_maxalpha_eligible[:, fi].sum())})
    own_frame_parity = eligibility[np.arange(len(xyz)), frame_origin]
    if not np.array_equal(own_frame_parity, original_front):
        raise ValueError('Own-frame reprojection differs from frozen audit')
    any_camera = eligibility.any(axis=1)
    temporal_any = original_temporal_maxalpha_eligible.any(axis=1)
    proof = {'status': 'CPU_FRONT_PINHOLE_ELIGIBILITY_CONTEXT_ONLY', 'GPU_or_training_started': False, 'canonical_audit_modified': False,
             'canonical_audit_SHA256': CANONICAL_SHA, 'lineage_map_SHA256': audit['lineage_map_SHA256'],
             'native_target_retained_static_rows': 169, 'unique_original_returns': 160,
             'own_acquisition_frame': {'outside_FRONT_frustum_rows': int((~original_front).sum()), 'inside_FRONT_but_outside_actor_mask_rows': int(original_front.sum()), 'inside_actor_mask_rows': int(original_mask.sum())},
             'over_all49_observed_training_FRONT_poses': {'any_camera_eligible_rows': int(any_camera.sum()), 'any_camera_eligible_unique_returns': int(len(np.unique(raw_rows[any_camera]))), 'never_center_in_front_pinhole_rows': int((~any_camera).sum()), 'original_initial_temporal_filter_and_maxalpha_upper_bound_eligible_rows': int(temporal_any.sum()), 'original_initial_temporal_filter_and_maxalpha_upper_bound_eligible_unique_returns': int(len(np.unique(raw_rows[temporal_any])))} ,
             'method': 'Quantized original world point held fixed, inverse original Frame ego pose then original Tr0/P0, positivez>.01, nearestpixel, original native dimensions. At initialization v0, sigma_sq=.96*.1, basealpha=.01. No raw10 anchor or new mask used.',
             'eligibility_limit': 'Geometric center inclusion under original pinhole inputs only; no Gaussian footprint, covariance/SH, occlusion, per-pixel raster cutoff or actual Scene/runtime pose snapshot. Potential alpha upper bound is not proof of actual pixel contribution or confetti causality.',
             'initial_hypothesis_limit': 'The temporal-bound count describes the original inherited initialized fields, not later learned alpha/width/velocity. Static T would intentionally remove temporal eligibility restriction.',
             'all49_cameras': cells,
             'witnesses': [{'sample_index': int(sample_ids[i]), 'original_frame': prep['frames'][int(frame_origin[i])]['id'], 'eligible_camera_indices': np.flatnonzero(eligibility[i]).tolist(), 'original_initial_temporal_upper_bound_camera_indices': np.flatnonzero(original_temporal_maxalpha_eligible[i]).tolist()} for i in range(len(xyz))],
             'source_script_SHA256': sha(__file__)}
    if sha(canonical) != CANONICAL_SHA:
        raise ValueError('Canonical changed during separate followup')
    target = OUT / 'front_visibility_followup_cpu.json'
    target.write_text(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': proof['status'], 'own_frame': proof['own_acquisition_frame'], 'all49': proof['over_all49_observed_training_FRONT_poses'], 'receipt_SHA256': sha(target)}))


if __name__ == '__main__':
    main()
