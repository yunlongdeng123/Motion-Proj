from pathlib import Path
import json,math,shutil
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
source=root/'actor_evidence_r1';out=root/'actor_evidence_r2';out.mkdir(exist_ok=False)
m=json.loads((source/'actor_evidence.json').read_text());m['thresholds']['min_unique_lidar_fraction']=.05
for row in m['actors']:
 for f in row['frames']:
  f['joint_evidence']=bool(f['mask_known'] and f['body_pixels']>=256 and f['visible_lidar_hits']>=8 and f['unique_lidar_pixels']>=max(4,math.ceil(f['body_pixels']*.05)))
 qualified=[f for f in row['frames'] if f['joint_evidence']]
 row['joint_evidence_frames']=len(qualified)
 row['max_joint_evidence_body_pixels']=max([f['body_pixels'] for f in qualified] or [0])
 row['eligible_delete']=bool(len(qualified)>=3 and row['max_joint_evidence_body_pixels']>=400 and row['seed_gaussians']>=100)
 if row['eligible_delete']:
  shutil.copyfile(source/row['preview'],out/row['preview']);row['root_evidence_preview_review']='camera body and projected same-Actor LiDAR jointly visible'
 else:
  row['excluded_reasons']=['insufficient repeated clear camera footprint with at least5% own-LiDAR pixel support'];row['preview']=None
m['eligible_actor_indices']=[r['actor_index'] for r in m['actors'] if r['eligible_delete']]
m['eligible_track_ids']=[r['raw_track_id'] for r in m['actors'] if r['eligible_delete']]
m['root_visual_review']='r1 contact sheet inspected; sparse top-only correspondence for a19 excluded by stronger joint-support gate'
(out/'actor_evidence.json').write_text(json.dumps(m,indent=2))
print(json.dumps(dict(eligible=m['eligible_actor_indices'],count=len(m['eligible_actor_indices']))))
protocol=json.loads((root/'comparison_protocol.json').read_text());protocol['eligible_delete_actor_indices']=m['eligible_actor_indices'];protocol['actor_evidence']='actor_evidence_r2/actor_evidence.json'
protocol['deletion_scope']='only clearly visible camera+LiDAR supported actors; not every LiDAR-seeded track'
(root/'comparison_protocol.json').write_text(json.dumps(protocol,indent=2))
