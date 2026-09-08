"""Archive input construction counts without duplicating large known trajectory arrays."""
import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
source=json.loads((args.run/'index.json').read_text()); scenes=[]
for scene in source['scenes']:
    scenes.append({'scene':scene['scene'],'log_id':scene['log_id'],'role':scene['role'],
        'actors':len(scene['cohort']),'build_scans':len(scene['build']),'heldout_scans':len(scene['frames']),
        'raw_build_returns':sum(r['rays'] for r in scene['build']),
        'raw_heldout_returns':sum(r['rays'] for r in scene['frames']),
        'background_points':scene['background_points'],'background_triangles':scene['background_triangles'],
        'trajectory_poses':sum(len(t['timestamps_ns']) for t in scene['actor_trajectories'].values()),
        'pose_mode':scene['pose_mode'],'carving':scene['carving']})
result={key:value for key,value in source.items() if key!='scenes'}
result.update(source_index=str(args.run/'index.json'),source_index_bytes=(args.run/'index.json').stat().st_size,
    scenes=scenes,totals={key:sum(scene[key] for scene in scenes) for key in
        ['actors','build_scans','heldout_scans','raw_build_returns','raw_heldout_returns','background_points','background_triangles','trajectory_poses']},
    build_carving_totals={key:sum(scene['carving'][key] for scene in scenes) for key in
        ['original_triangles','removed_triangles','build_conflicting_rays_before','remaining_build_early_rays','remaining_build_free_sum_m']},
    boundary='construction counts and build-only carving diagnostics; no network inference or heldout model-quality scores; full raw scene/known trajectories remain in source run')
args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:result[key] for key in ['status','totals','build_carving_totals','wall_s','peak_parent_rss_gib','max_constructor_rss_gib']}))
