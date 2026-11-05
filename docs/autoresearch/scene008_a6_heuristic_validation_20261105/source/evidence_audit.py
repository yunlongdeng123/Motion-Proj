"""Read-only CPU audit of Scene008 a6 observation and fixed lower domains."""

import argparse
import json
from pathlib import Path

import numpy as np


CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RAW_IDS = tuple(range(89, 138))
HELDOUT = (89, 95, 101, 105, 113, 121, 125, 130, 137)
BUILD = tuple(raw for raw in RAW_IDS if raw not in HELDOUT)
BOX_MARGIN_M = .25
LOWER_BIN_M = .20


def count(mask):
    return int(np.count_nonzero(mask))


def frame_domains(frame, lower_file):
    near = frame['box_near_m'][0]
    far = frame['box_far_m'][0]
    scene_z = frame['scene_depth_m']
    padded_box = near > 0
    free = frame['free'][0]
    lower = lower_file['lower']
    visible, occluded, unknown = (frame[name][0] for name in ('visible', 'occluded', 'unknown'))
    depth_valid = frame['depth_valid'][0]
    free_miss = free & ~padded_box
    free_through = free & padded_box
    projected_far = padded_box & (scene_z > far + BOX_MARGIN_M)
    other_depth = frame['depth_valid'][1] & (frame['depth_m'][1] > far + BOX_MARGIN_M)
    cutoff = lower_file['cutoff_bin'] * LOWER_BIN_M
    assert not (visible & occluded).any() and not ((visible | occluded) & free).any()
    assert np.array_equal(unknown, ~(visible | occluded | free))
    assert not (depth_valid & ~visible).any()
    assert not (lower & (depth_valid | frame['visible_reviewed'][0])).any()
    assert (cutoff[lower] > 0).all()

    domains = dict(
        fixed_lower=lower,
        reviewed_a6=frame['visible_reviewed'][0],
        observed_a6_rgb=visible,
        measured_a6_depth=depth_valid,
        observed_a37_rgb=frame['visible'][1],
        measured_a37_depth=frame['depth_valid'][1],
        occluded=occluded,
        possible_occluded=frame['possible_occluded'][0],
        unknown=unknown,
        unknown_inside_padded_box=unknown & padded_box,
        padded_box_hit=padded_box,
        exact_box_hit=frame['box_candidate'][0],
        bank_free=free,
        bank_free_box_miss=free_miss,
        bank_free_reviewed_through=free_through,
        bank_free_through_with_serialized_a37_depth=free_through & other_depth,
        bank_free_through_with_projected_far_depth=free_through & projected_far,
        projected_scene_return=scene_z > 0,
        projected_return_past_padded_box=projected_far,
        projected_return_before_padded_box=padded_box & (scene_z > 0) & (scene_z < near),
        projected_return_inside_padded_box=padded_box & (scene_z >= near) & (scene_z <= far),
        measured_road_edit=frame['road_edit'],
        lower_with_projected_return_before_cutoff=lower & (scene_z > 0) & (scene_z < cutoff),
        lower_with_projected_return_at_or_beyond_cutoff=lower & (scene_z >= cutoff),
    )
    return domains, cutoff


def cell_ray_counts(frame, domains, cutoff, boxes):
    """Intersect the archived PVD cells in camera-z metres at pixel centers."""
    selected = domains['fixed_lower'] | domains['bank_free'] | domains['measured_a6_depth']
    y, x = np.nonzero(selected)
    uv = np.column_stack((x + .5, y + .5, np.ones(len(x))))
    local_from_camera = np.linalg.inv(frame['world_to_camera'] @ frame['world_from_local'][0])
    directions = uv @ np.linalg.inv(frame['K']).T @ local_from_camera[:3, :3].T
    origin = local_from_camera[:3, 3]
    direction = directions[:, None, :]
    parallel = np.abs(direction) < 1e-12
    divisor = np.where(parallel, 1., direction)
    t0, t1 = ((edge - origin) / divisor for edge in boxes.transpose(1, 0, 2))
    entry = np.maximum(np.where(parallel, -np.inf, np.minimum(t0, t1)).max(2), .1)
    leave = np.where(parallel, np.inf, np.maximum(t0, t1)).min(2)
    parallel_outside = parallel & ((origin < boxes[:, 0]) | (origin > boxes[:, 1]))
    hit = (leave > entry) & ~parallel_outside.any(2)
    lower = domains['fixed_lower'][selected, None]
    own_z = frame['depth_m'][0][selected, None]
    scene_z = frame['scene_depth_m'][selected, None]
    counts = dict(
        lower_rays=hit & lower,
        lower_prefix_rays=hit & lower & (entry < cutoff[selected, None]),
        lower_with_projected_return_after_cell=hit & lower & (scene_z > leave + BOX_MARGIN_M),
        bank_free_box_miss_rays=hit & domains['bank_free_box_miss'][selected, None],
        bank_free_reviewed_through_rays=hit & domains['bank_free_reviewed_through'][selected, None],
        measured_a6_endpoint_in_cell=hit & domains['measured_a6_depth'][selected, None]
        & (own_z >= entry) & (own_z < leave),
    )
    return {key: value.sum(0) for key, value in counts.items()}


def asset_provenance(asset_path):
    with np.load(asset_path, allow_pickle=False) as asset:
        raw = asset['sample_raw_frame']
        rgb = asset['sample_rgb_valid']
        voxel = np.floor(asset['sample_xyz_local_m'] / .05).astype(np.int32)
        anchor_raw = asset['source_raw_frame']
        anchor_rgb = asset['rgb_valid']
    assert not (rgb & ~np.isin(raw, RAW_IDS)).any()
    result = {}
    for name, ids in (('build', BUILD), ('heldout', HELDOUT)):
        selected = np.isin(raw, ids)
        anchor_selected = np.isin(anchor_raw, ids)
        pairs = np.unique(np.column_stack((voxel[selected], raw[selected])), axis=0)
        _, distinct = np.unique(pairs[:, :3], axis=0, return_counts=True)
        result[name] = dict(
            raw_samples=count(selected), rgb_valid_samples=count(selected & rgb),
            rgb_voxels=len(np.unique(voxel[selected & rgb], axis=0)),
            geometry_voxels_in_two_or_more_split_frames=count(distinct >= 2),
            full_bank_representatives_from_split=count(anchor_selected),
            full_bank_rgb_representatives_from_split=count(anchor_selected & anchor_rgb),
        )
    return result


def pool_counts(rows, key):
    return {name: sum(row[key][name] for row in rows) for name in rows[0][key]}


def audit(observations, lower_dir, negative_file, volume_file):
    with negative_file.open(encoding='utf-8') as stream:
        negative = {row['raw']: row for row in json.load(stream)['rows']}
    assert tuple(sorted(negative)) == RAW_IDS
    with np.load(volume_file, allow_pickle=False) as archive:
        boxes = archive['boxes'].astype(np.float64)
    assert boxes.ndim == 3 and boxes.shape[1:] == (2, 3) and (boxes[:, 1] > boxes[:, 0]).all()

    rows, volumes = [], {}
    for raw in RAW_IDS:
        with np.load(observations / 'frames' / f'{raw:03d}.npz', allow_pickle=False) as frame:
            with np.load(lower_dir / f'{raw:03d}.npz', allow_pickle=False) as lower_file:
                assert int(frame['raw']) == raw
                domains, cutoff = frame_domains(frame, lower_file)
                volumes[raw] = cell_ray_counts(frame, domains, cutoff, boxes)
        lower = domains['fixed_lower']
        previous = negative[raw]
        assert count(domains['bank_free']) == previous['known_empty']
        assert not (domains['bank_free'] & lower).any()
        assert count(lower) > 0
        rows.append(dict(raw=raw, split='heldout' if raw in HELDOUT else 'build',
                         counts={name: count(mask) for name, mask in domains.items()},
                         fixed_lower_overlap={name: count(lower & mask) for name, mask in domains.items()},
                         prior_outside_overlap=dict(
                             bank_free=previous['empty_outside'],
                             without_bank_free=previous['variants']['previous']['outside_remaining_rgb']['count'])))

    pools, volume_pools = {}, {}
    for name, ids in (('all', RAW_IDS), ('build', BUILD), ('heldout', HELDOUT)):
        group = [row for row in rows if row['raw'] in ids]
        pools[name] = {key: pool_counts(group, key) for key in
                       ('counts', 'fixed_lower_overlap', 'prior_outside_overlap')}
        volume_pools[name] = {}
        for key in volumes[ids[0]]:
            per_frame = np.stack([volumes[raw][key] for raw in ids])
            volume_pools[name][key] = dict(ray_cell_hits=per_frame.sum(0).tolist(),
                                           distinct_frames=(per_frame > 0).sum(0).tolist())

    return dict(
        status='PASS', build=list(BUILD), heldout=list(HELDOUT), rows=rows, pooled=pools,
        a6_sample_provenance=asset_provenance(observations / 'a6.npz'),
        volume=dict(boxes_local_m=boxes.tolist(), by_split=volume_pools),
        notes={
            'bank_free_box_miss': 'Pixel collar outside the 0.18 m padded a6 box projection.',
            'bank_free_reviewed_through': 'Reviewed a37-body LiDAR return beyond the full padded a6 box plus 0.25 m.',
            'projected_scene_return': 'Nearest projected camera-z of all LiDAR returns. This does not identify a free camera ray.',
            'prior_outside_overlap': 'Legacy replay count using native a6 alpha and its fixed action collar.',
            'volume': 'Archived cells and fixed lower are inherited model proxies. One ray may hit several cells.',
            'split': 'Heldout here means exclusion from new heuristic evidence. Frozen prior models and domains predate this split.',
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--output', type=Path, help='Write JSON to this path; defaults to stdout.')
    args = parser.parse_args()
    root = args.campaign
    report = audit(
        root / 'scene008_instances_20261105_r1/observations',
        root / 'scene008_a6_local_fusion_20261105/evidence',
        root / 'scene008_a6_ray_switch_20261105/negative_evidence.json',
        root / 'scene008_a6_volume_carve_20261105/volume_accepted.npz',
    )
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + '\n', encoding='utf-8')
    else:
        print(payload)


if __name__ == '__main__':
    main()
