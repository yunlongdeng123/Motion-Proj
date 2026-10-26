"""Read-only identity/physical-component audit; boxes never filter actor points."""
import argparse, importlib.util, json, os
from pathlib import Path
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from benchmark_geometry import ROOT, read
from prepare_motion_suite import rz


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', required=True)
    args = p.parse_args()
    scene = ROOT/'motion_scene_inputs_v1'/args.context
    proposal = read(scene/'native_preparation/selection_proposal.json')
    recipe_path = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/tools/red_actor_adapter/actor_cpu.py')
    spec = importlib.util.spec_from_file_location('frozen_original_actor_recipe', recipe_path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    records = []
    for row in proposal['frames']:
        name = row['frame_id']
        mask = np.array(Image.open(scene/'mask/mask_0/Annotations'/(name+'.png')))
        lidar = np.fromfile(scene/'velodyne'/(name+'.bin'), dtype=np.float32).reshape(-1, 6)
        pose = np.loadtxt(scene/'pose'/(name+'.txt'))
        projection = recipe.project(lidar, mask, recipe.calibration(scene/'calib'/(name+'.txt')))
        xyz, homogeneous, source_indices, uv, depth, bins, nearest = projection
        labels = read(scene/'annotations'/(name+'.json'))['laser_labels']
        b = next(x['box'] for x in labels if x['id'] == proposal['track'])
        target_center = np.array([b['center_x'], b['center_y'], b['center_z']])
        target_axes = rz(float(b['heading']))
        target_size = np.array([b['length'], b['width'], b['height']])
        for candidate in row['candidates']:
            instance = candidate['instance_id']
            hit = (mask[uv[:, 1], uv[:, 0]] == instance)
            indices = source_indices[hit & (depth <= nearest[bins]+.5)]
            counts = []
            if len(indices):
                edges = cKDTree(xyz[indices]).query_pairs(.5, output_type='ndarray')
                graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])), shape=(len(indices), len(indices)))
                _, components = connected_components(graph, directed=False)
                counts = sorted(np.bincount(components).tolist(), reverse=True)
            physical_pass = not len(indices) or counts[0] >= .5*len(indices)
            local = (xyz[indices]-target_center) @ target_axes
            inside = np.all(np.abs(local) <= target_size/2+.1, axis=1)
            records.append(dict(frame_id=name, raw=row['raw'], instance_id=instance,
                                automatic_proposal_selected=row['instance_ids'] == [instance],
                                projected_hit_points=int(hit.sum()), retained_before_component=len(indices),
                                component_counts=counts, original_component_guard_pass=physical_pass,
                                target_box_inside_fraction_plus01m=float(inside.mean()) if len(indices) else None,
                                diagnostic_boxes_only=True, selected_points_changed=False))
    by_id = {}
    for row in records:
        d = by_id.setdefault(str(row['instance_id']), dict(frames=0, guard_failures=[], before_component_points=0, points_inside_target_box=0))
        d['frames'] += 1
        d['before_component_points'] += row['retained_before_component']
        d['points_inside_target_box'] += round((row['target_box_inside_fraction_plus01m'] or 0)*row['retained_before_component'])
        if not row['original_component_guard_pass']:
            d['guard_failures'].append(row['frame_id'])
    for d in by_id.values():
        d['target_inside_fraction_plus01m'] = d['points_inside_target_box']/max(1, d['before_component_points'])
    result = dict(context=args.context, target_track=proposal['track'], rows=records, by_instance=by_id,
                  original_masks_modified=False, geometry_guard_modified=False, training_points_modified=False,
                  GT_boxes_used_only_for_diagnostic=True, any_training_started_here=False,
                  review_status='hold_remains_until_semantics_and_original_guard_both_valid')
    dest = scene/'native_preparation/association_candidate_audit.json'
    assert not dest.exists()
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
