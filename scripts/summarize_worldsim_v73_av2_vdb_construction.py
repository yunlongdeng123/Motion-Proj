"""Compact AV2 VDB construction records without copying Actor trajectory arrays."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = json.loads((args.run/'index.json').read_text())
manifest = json.loads((args.run/'manifest.json').read_text())
scenes = [{key: scene[key] for key in ['scene', 'log_id', 'role', 'background_points', 'background_triangles', 'tsdf', 'carving']}
          | {'actors': len(scene['cohort']), 'linked_heldout_frames': len(scene['frames']),
             'linked_heldout_rays': sum(frame['rays'] for frame in scene['frames'])} for scene in data['scenes']]
result = {key: value for key, value in data.items() if key != 'scenes'}
result.update(run=str(args.run), manifest=manifest, scenes=scenes, totals={
    'logs': len(scenes), 'actors': sum(scene['actors'] for scene in scenes),
    'build_raw_returns': sum(row['raw_returns'] for scene in scenes for row in scene['tsdf']['build']),
    'integrated_background_returns': sum(scene['tsdf']['integrated_returns'] for scene in scenes),
    'parent_unique_background_centers': sum(scene['background_points'] for scene in scenes),
    'origin_groups': sum(scene['tsdf']['origin_groups'] for scene in scenes),
    'native_triangles': sum(scene['tsdf']['native_triangles'] for scene in scenes),
    'carved_triangles': sum(scene['background_triangles'] for scene in scenes),
    'removed_triangles': sum(scene['carving']['removed_triangles'] for scene in scenes),
    'remaining_build_early_rays': sum(scene['carving']['remaining_build_early_rays'] for scene in scenes),
    'remaining_build_free_sum_m': sum(scene['carving']['remaining_build_free_sum_m'] for scene in scenes),
    'linked_heldout_rays': sum(scene['linked_heldout_rays'] for scene in scenes),
})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({key: result[key] for key in ['status', 'wall_s', 'peak_rss_gib', 'totals', 'model_quality_computed']}))
