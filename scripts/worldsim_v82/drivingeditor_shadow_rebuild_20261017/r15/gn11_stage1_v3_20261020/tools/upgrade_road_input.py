"""只修复遗漏的mask门限，不重建既有查询/曲面；写新49帧包。"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import binary_dilation


def upgrade(source, output):
    from utils.v3_road_geometry import required_occluder_masks, sample
    from utils.v3_road_contract import ROAD_VERSION, ROAD_POLICY
    from utils.v3_road_prior import fixed_subcells, prior_graph
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads(source.read_text())
    with np.load(source.parent/metadata['units'], allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    before = data['base_weight'].copy()
    rows = []
    for index, fid in enumerate(metadata['frame_order']):
        mask, record = required_occluder_masks(metadata['original_engineering'], fid)
        clear = sample(binary_dilation(mask, iterations=ROAD_POLICY['mask_margin_px']), data['source_uv'][index]) == 0
        # 新门限只能收紧；保留全部旧LiDAR/盒/角度/分辨率/质量证据。
        data['gate_mask_clear'][index] &= clear
        accepted = np.logical_and.reduce([value[index] for key, value in data.items() if key.startswith('gate_')])
        data['base_weight'][index, ~accepted] = 0
        metadata['frames'][fid]['occluder_masks'] = record
        rows.append(dict(frame=fid, **record, accepted_before=int((before[index] > 0).sum()),
                         accepted_after=int((data['base_weight'][index] > 0).sum())))
    count = (data['base_weight'] > 0).sum(0)
    support = data['surface_supported']
    data['source_count'] = count
    data['provenance'] = np.where(support & (count >= ROAD_POLICY['minimum_views']), 1,
                                  np.where(support & (count == 0), 2, 0)).astype(np.uint8)
    data['coverage_model_xyz'] = fixed_subcells(data, metadata, ROAD_POLICY)
    graph, components = prior_graph(data['provenance'], support, data['is_query'], data['edges'])
    for name, value in graph.items():
        data['prior_graph_'+name] = value
    summaries = {}
    for actor in metadata['actors']:
        ids = np.unique(data['query_unit_id'][(data['query_frame'] == 0) & (data['query_actor'] == actor)])
        p = data['provenance'][ids]
        prior_ids = ids[p == 2]
        summaries[str(actor)] = dict(raw89_world_units=len(ids), observed=int((p == 1).sum()),
            prior=int((p == 2).sum()), unknown=int((p == 0).sum()),
            single_view_unconfirmed=int((count[ids] == 1).sum()),
            prior_reaches_observed_context=int((graph['context_hops'][prior_ids] >= 0).sum()),
            prior_unanchored_to_observed_context=int((graph['context_hops'][prior_ids] < 0).sum()))
    metadata.update(version=ROAD_VERSION, policy=ROAD_POLICY, parent_bundle=str(source),
        raw89_actor_summary=summaries, observed_units=int((data['provenance'] == 1).sum()),
        prior_units=int((data['provenance'] == 2).sum()), unknown_units=int((data['provenance'] == 0).sum()),
        source_observations=int((data['base_weight'] > 0).sum()),
        mask_union_repair=dict(omission_was_intentional=False, original_accepted=int((before > 0).sum()),
            restored_accepted=int((data['base_weight'] > 0).sum()), fixed_query_surface_unchanged=True,
            restored_native_mask_extra_pixels=sum(row['native_extra_pixels'] for row in rows)),
        prior_boundary_graph=dict(components=components,
            interpretation='context reach is graph evidence only; disabled prior arm has no prior anchor objective; unanchored stays explicit, never texture GT'))
    metadata['evidence_roles']['masks'] = 'required five-ID + all-vehicle + native proposal union'
    output.mkdir(parents=True)
    np.savez_compressed(output/'units.npz', **data)
    (output/'mask_input_audit.json').write_text(json.dumps(rows, indent=2))
    (output/'bundle.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return dict(version=ROAD_VERSION, **metadata['mask_union_repair'], actors=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.engine))
    print(json.dumps(upgrade(args.source, args.output), indent=2))
