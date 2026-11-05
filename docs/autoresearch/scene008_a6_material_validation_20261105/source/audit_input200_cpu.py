"""CPU-only audit; never imports a renderer/model or changes source assets."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / 'a6_io_audit/source_evidence/scene008_a37_C2_20261103_r1'
PVD = EVIDENCE / 'source/prior/proposals/pvd_seed000'
HELD = np.array([89, 95, 101, 105, 113, 121, 125, 130, 137])
BUILD = np.setdiff1d(np.arange(89, 138), HELD)
RADIUS = .05
MIN_FRAMES = 3
MIN_SPAN = .5


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uid(raw, row):
    return raw.astype(np.int64) * (1 << 32) + row.astype(np.int64)


def quant(a):
    return np.quantile(a, [0, .05, .5, .95, 1]).tolist() if len(a) else []


def joined(query, values):
    order = np.argsort(values)
    at = np.searchsorted(values[order], query)
    if np.any(at == len(values)) or not np.array_equal(values[order[at]], query):
        raise ValueError('Unmatched source UID')
    return order[at]


def support(query, source_raw, xyz, raw, timestamps, poses):
    tree = cKDTree(xyz)
    masks = np.zeros((len(query), 199), bool)
    for start in range(0, len(query), 512):
        for j, near in enumerate(tree.query_ball_point(query[start:start+512], RADIUS)):
            masks[start+j, np.unique(raw[near])] = True
    count = masks.sum(1).astype(np.int16)
    other = count - masks[np.arange(len(query)), source_raw]
    span = np.zeros(len(query))
    motion = np.zeros(len(query))
    for j, mask in enumerate(masks):
        rr = np.flatnonzero(mask)
        if len(rr):
            span[j] = (timestamps[rr[-1]]-timestamps[rr[0]]) / 1e6
            motion[j] = np.linalg.norm(poses[rr[-1], :3, 3]-poses[rr[0], :3, 3])
    return dict(support_raw_frame_mask=masks, support_frame_count=count,
                support_other_frame_count=other.astype(np.int16), support_time_span_s=span,
                support_actor_endpoint_translation_m=motion)


def main():
    geo_path = EVIDENCE/'inputs/a6_geometry_minimal.npz'
    current_path = ROOT/'inputs/current_a6_observations.npz'
    full_path = ROOT/'inputs/c2_full199_observations.npz'
    flag_path = ROOT/'inputs/raw_other_box_flags.npz'
    paths = [geo_path, current_path, full_path, flag_path, PVD/'candidate_points_actor_native.npz',
             PVD/'fixed_partial_input.npz', EVIDENCE/'inputs/a6_prior_inputs.npz',
             ROOT/'inputs/raw_annotations_check.json', Path(__file__)]
    g, c, f, flags, pvd, fixed, prior = [bank(x) for x in paths[:7]]
    source_id = uid(g['raw_frame'], g['source_cloud_row'])
    current_id = uid(c['sample_raw_frame'], c['sample_source_cloud_row'])
    assert len(np.unique(source_id)) == len(source_id) == len(current_id) == 54715
    source_to_current = joined(source_id, current_id)
    source_to_full = joined(source_id, uid(f['raw_frame'], f['source_cloud_row']))
    ix = pvd['conditioning_observed_row_indices'].astype(np.int64)
    assert np.array_equal(ix, pvd['observed_raw_row_indices'])
    assert np.array_equal(ix, fixed['raw_row_indices'])
    assert np.array_equal(ix, np.random.default_rng(0).choice(len(source_id), 200, replace=False))
    assert np.array_equal(flags['input200_observation_index'], ix)
    assert np.array_equal(prior['source_raw_frame'], g['raw_frame'])
    assert np.array_equal(prior['source_cloud_row'], g['source_cloud_row'])
    map_error = np.max(np.abs(g['canonical_xyz']-c['sample_xyz_local_m'][source_to_current]), axis=1)
    partial = pvd['points_actor_native_m'][~pvd['generated_mask']]
    partial_error = np.max(np.abs(partial-g['canonical_xyz'][ix]), axis=1)
    assert len(ix) == len(np.unique(ix)) == 200 and partial_error.max() < 2e-5
    assert np.max(np.abs(prior['observed_points_actor_native_m']-g['canonical_xyz'])) == 0
    raw, xyz = c['sample_raw_frame'], c['sample_xyz_local_m']
    clean = ~np.isin(raw, HELD)
    # Rebuild representatives from admissible real source rows; historical fused
    # anchors can have HELD-dependent selection and are not reused as this pool.
    candidates = np.flatnonzero(clean)
    key, group = np.unique(np.floor(xyz[candidates]/RADIUS).astype(np.int32), axis=0, return_inverse=True)
    n = len(key)
    centroid = np.column_stack([np.bincount(group, weights=xyz[candidates, a], minlength=n)
                               / np.bincount(group, minlength=n) for a in range(3)])
    distance = np.linalg.norm(xyz[candidates]-centroid[group], axis=1)
    order = np.lexsort((current_id[candidates], distance, group))
    representatives = candidates[order[np.r_[True, np.diff(group[order]) != 0]]]
    pool_geo_index = joined(current_id[representatives], source_id)
    input_current = source_to_current[ix]
    sample = np.r_[input_current, representatives]
    ss = support(xyz[sample], raw[sample], xyz[clean], raw[clean], g['timestamp_micros'], g['world_from_canonical'])
    own = np.all(np.abs(xyz[sample]) <= g['box_lwh_m']/2 + 1e-5, axis=1)
    exact_index = np.r_[ix, pool_geo_index]
    reasons = np.zeros(len(sample), np.uint16)
    reasons[map_error[exact_index] > .0005] |= 1
    reasons[~own] |= 2
    reasons[flags['full199_other_box_pad03'][exact_index]] |= 4
    reasons[ss['support_frame_count'] < MIN_FRAMES] |= 8
    reasons[ss['support_other_frame_count'] < 2] |= 16
    reasons[ss['support_time_span_s'] < MIN_SPAN] |= 32
    held = np.isin(raw[sample], HELD)
    geo_consistent = reasons == 0
    reasons[held] |= 64
    eligible = reasons == 0
    rgb_valid = c['sample_rgb_valid'][sample]
    rgb_build = rgb_valid & np.isin(raw[sample], BUILD)
    rgb_payload = np.where(rgb_valid[:, None], c['sample_rgb'][sample], np.nan)
    fields = dict(xyz_local_m=xyz[sample], source_uid=current_id[sample], source_raw_frame=raw[sample],
                  voxel_key=np.floor(xyz[sample]/RADIUS).astype(np.int32),
                  source_cloud_row=c['sample_source_cloud_row'][sample], source_current_sample_index=sample,
                  source_original_observation_index=exact_index, source_current_fused_anchor=c['sample_anchor'][sample],
                  source_rgb_srgb=rgb_payload, source_rgb_valid=rgb_valid, appearance_BUILD_eligible=eligible & rgb_build,
                  source_HELD=held, geometry_consistent=geo_consistent, geometry_admission_eligible=eligible,
                  geometry_rejection_bits=reasons, semantic_class=np.full(len(sample), -1, np.int8), **ss)
    for name, sl in [('input200_eligibility.npz', slice(0, 200)), ('geometry_candidate_pool.npz', slice(200, None))]:
        np.savez_compressed(ROOT/name, **{k: v[sl] for k, v in fields.items()},
                            support_radius_m=RADIUS, minimum_raw_frames=MIN_FRAMES, minimum_span_s=MIN_SPAN,
                            support_raw_ids=np.setdiff1d(np.arange(199), HELD), legacy_HELD_raw_ids=HELD,
                            coordinates=np.array('x forward,y left,z up; metres'),
                            source_bank_sha256=np.array(sha(current_path)),
                            scene_context=np.array('10094743350625019937_3420_000_3440_000'),
                            target_track_id=np.array('7MxwK9zTRZ9g5lc_ZC0z6A'),
                            semantic_status=np.array('real-return geometry candidates; not per-point semantic truth'))
    def summary(sl):
        rr = raw[sample][sl]; mm = ss['support_raw_frame_mask'][sl]
        return dict(points=len(rr), geometry_consistent=int(geo_consistent[sl].sum()),
                    geometry_admission_eligible=int(eligible[sl].sum()), source_HELD=int(held[sl].sum()),
                    source_reviewed_RGB=int(rgb_valid[sl].sum()), appearance_BUILD_eligible=int((eligible & rgb_build)[sl].sum()),
                    support_frame_count_quantiles=quant(ss['support_frame_count'][sl]),
                    support_other_frame_count_quantiles=quant(ss['support_other_frame_count'][sl]),
                    support_time_span_s_quantiles=quant(ss['support_time_span_s'][sl]),
                    support_actor_endpoint_translation_m_quantiles=quant(ss['support_actor_endpoint_translation_m'][sl]),
                    has_support_before_raw89=int(mm[:, :89].any(1).sum()),
                    has_support_in_BUILD40=int(mm[:, BUILD].any(1).sum()),
                    has_support_after_raw137=int(mm[:, 138:].any(1).sum()),
                    has_support_in_all_three_periods=int((mm[:, :89].any(1)&mm[:, BUILD].any(1)&mm[:, 138:].any(1)).sum()),
                    rejection_counts={str(bit): int(((reasons[sl] & bit) != 0).sum()) for bit in [1, 2, 4, 8, 16, 32, 64]},
                    raw_frame_count=int(np.unique(rr).size), z_minmax_m=[float(xyz[sample][sl, 2].min()), float(xyz[sample][sl, 2].max())])
    pairs = pdist(partial.astype(np.float64))
    raw200 = raw[input_current]
    input_rgb = c['sample_rgb_valid'][input_current]
    legacy_rgb = f['rgb_valid_reviewed_native49'][source_to_full[ix]]
    height = [-.72, -.210449, 0, .24, .48, .72]
    result = dict(schema='a6_input200_eligibility_audit_v1', status='CPU_READ_ONLY_COMPLETE',
        source_selection=dict(seed=0, source_rows=54715, selected_rows=200, without_replacement=True,
                              seed0_selection_exactly_reproduced=True, input_aliases_and_fixed_partial_indices_equal=True,
                              partial_vs_source_max_abs_m=float(partial_error.max()),
                              source_vs_current_bank_max_abs_m=float(map_error.max()),
                              C2_full199_source_UID_set_equals_original=True,
                              C2_full199_vs_original_xyz_max_abs_m=float(np.abs(f['canonical_xyz'][source_to_full]-g['canonical_xyz']).max()),
                              prior_input_vs_minimal_geometry_exact=True, source_UIDs_unique=True,
                              current_sample_UID_set_equals_original=True,
                              current_fused_anchor_available_for_input200=int((c['sample_anchor'][input_current]>=0).sum())),
        temporal_source=dict(unique_raw_frames=int(np.unique(raw200).size), source_raw_ids=np.unique(raw200).tolist(),
                             before89=int((raw200<89).sum()), BUILD40=int(np.isin(raw200, BUILD).sum()),
                             HELD9=int(np.isin(raw200, HELD).sum()), after137=int((raw200>137).sum()),
                             reviewed_RGB=int(input_rgb.sum()), reviewed_RGB_BUILD=int((input_rgb & np.isin(raw200, BUILD)).sum()),
                             reviewed_RGB_HELD=int((input_rgb & np.isin(raw200, HELD)).sum()),
                             C2_legacy_reviewed_RGB=int(f['rgb_valid_reviewed_native49'][source_to_full[ix]].sum()),
                             C2_legacy_sparse_hint=int(f['rgb_valid_sparse_geometric_hint'][source_to_full[ix]].sum()),
                             both_current_and_C2_reviewed_RGB=int((input_rgb & legacy_rgb).sum()),
                             current_only_reviewed_RGB=int((input_rgb & ~legacy_rgb).sum()),
                             C2_only_reviewed_RGB=int((~input_rgb & legacy_rgb).sum()),
                             RGB_rule='Use current bank only; never union historical validity flags.'),
        duplication=dict(exact_source_uid_duplicates=0, exact_coordinate_duplicates=200-len(np.unique(partial, axis=0)),
                         unique5mm_voxels=len(np.unique(np.floor(partial/.005).astype(np.int32), axis=0)),
                         unique5cm_voxels=len(np.unique(np.floor(partial/.05).astype(np.int32), axis=0)),
                         pair_distance_m_quantiles=quant(pairs), pairs_within5mm=int((pairs<=.005).sum()),
                         pairs_within5cm=int((pairs<=.05).sum()),
                         note='多个return/多个点仅按distinct raw计支持；空间近邻不是重复源行。未重新解码200源行的sensor/return标签。'),
        height=dict(edges_m=height, input200_rows=np.histogram(partial[:, 2], height)[0].tolist(),
                    full199_rows=np.histogram(g['canonical_xyz'][:, 2], height)[0].tolist(),
                    lowest_all199_z_m=float(g['canonical_xyz'][:, 2].min())),
        raw_other_actor_check=json.loads((ROOT/'inputs/raw_annotations_check.json').read_text()),
        input200=summary(slice(0, 200)), candidate_pool=summary(slice(200, None)),
        input200_geometry_consistency_failed=[dict(input_index=int(i),
            source_uid=int(fields['source_uid'][i]), raw=int(fields['source_raw_frame'][i]),
            source_cloud_row=int(fields['source_cloud_row'][i]), xyz_local_m=fields['xyz_local_m'][i].tolist(),
            reason_bits=int(reasons[i]), source_reviewed_RGB=bool(rgb_valid[i]),
            support_frames=int(ss['support_frame_count'][i]), time_span_s=float(ss['support_time_span_s'][i]))
            for i in np.flatnonzero(~geo_consistent[:200])],
        candidate_pool_eligible_height_rows=np.histogram(fields['xyz_local_m'][200:][eligible[200:], 2], height)[0].tolist(),
        thresholds=dict(metric_radius_m=RADIUS, distinct_admitted_raw_frames=MIN_FRAMES,
                        other_raw_frames=2, minimum_time_span_s=MIN_SPAN, other_box_margin_m=.03,
                        canonical_mapping_tolerance_m=.0005, HELD_source_and_support_excluded=True,
                        no_parameter_sweep=True),
        reason_bits={'1':'source coordinate mapping mismatch', '2':'outside own annotated OBB',
                     '4':'inside other annotated OBB plus3cm', '8':'fewer than3 distinct admitted raw frames',
                     '16':'fewer than2 other admitted raw frames', '32':'support span below0.5s',
                     '64':'legacy HELD source; future clean conditioning excluded'},
        limits=['真实LiDAR/OBB关联不是逐点语义真值。所有semantic_class=-1。',
                '跨帧相近/宽时间跨度仍可能有静止期或重复面混淆，actor位移只作诊断未用于后验筛点。',
                '无他车盒体积重叠不证明RGB未遮挡；射线穿粗盒与点落他车盒是不同问题。',
                '原200含14旧HELD条件点；当前完整PVD几何不是独立HELD泛化。新候选代表和支持均排除9HELD。',
                'RGB仅携带自身源帧已审且实际深度可见颜色；未知为NaN，未做邻近颜色传播。',
                '候选池没有运行新PVD，没有抽选新的200，没有渲染/模型改动或图像质量选择。'],
        source_SHA256={str(x): sha(x) for x in paths},
        artifacts={name: dict(bytes=(ROOT/name).stat().st_size, sha256=sha(ROOT/name))
                   for name in ['input200_eligibility.npz', 'geometry_candidate_pool.npz']},
        GPU_used=False, training_updates=0, remote_mutations=False, AutoDL_left_running=True)
    (ROOT/'input200_eligibility_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['source_selection', 'temporal_source', 'duplication', 'input200', 'candidate_pool']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
