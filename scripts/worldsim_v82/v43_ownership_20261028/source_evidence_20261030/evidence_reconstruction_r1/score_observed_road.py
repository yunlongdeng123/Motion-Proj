"""Same-query reconstruction control split by old-observed vs old-unseen world road.

Visibility masks/independent edited geometry are consumed only after training/rendering.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030');OUT=ROOT/'evidence_reconstruction_r1'
references=json.loads((ROOT/'dataset_r2/privileged/reference_manifest.json').read_text())
lookup={(int(r['frame_id']),r['edit']):r for r in references['references'] if r['vehicle_shadow']}
rows=[]
for arm,folder in [('M0_original_observations',ROOT/'edit_r3_10k'),('same_world_richer_observations',OUT/'edit_common_10k')]:
    manifest=json.loads((folder/'render_manifest.json').read_text())
    for item in manifest['frames']:
        frame=int(item['frame']);edit=item['edit'];ref=lookup[(frame,edit)]
        with np.load(item['fp32']) as data:pred=data['final'].copy()
        target=np.asarray(Image.open(ROOT/'dataset_r2/privileged'/ref['rgb']).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
        with np.load(ROOT/'dataset_r2/privileged'/ref['geometry']) as data:current_road=np.isin(data['semantic_id'],[1,2])
        mask_path=ROOT/'visibility_control_r1'/('eval_masks_%06d.npz'%frame)
        with np.load(mask_path) as data:
            masks={key:data[key].astype(bool) for key in ['revealed_road','visible_old_any','visible_old_0','visible_old_1','visible_old_2plus']}
        groups=dict(all_revealed_road=masks['revealed_road'],old_observed_road=masks['revealed_road']&masks['visible_old_any'],old_unseen_road=masks['revealed_road']&masks['visible_old_0'],old_one_frame_road=masks['revealed_road']&masks['visible_old_1'],old_2plus_frames_road=masks['revealed_road']&masks['visible_old_2plus'])
        for name,mask in groups.items():
            mask=mask&current_road; pixels=int(mask.sum()); scores={}
            if pixels:
                for label,image in [('raw',pred),('clamped',pred.clip(0,1))]:
                    delta=image[:,mask].astype(np.float64)-target[:,mask].astype(np.float64)
                    scores[label+'_mae']=float(np.abs(delta).mean());scores[label+'_mse']=float(np.square(delta).mean())
            rows.append(dict(arm=arm,frame=frame,edit=edit,group=name,pixels=pixels,metrics=scores,visibility_mask=str(mask_path)))
summary=[]
for arm in sorted({r['arm'] for r in rows}):
    for edit in ['delete','shift','rotate']:
        for group in sorted({r['group'] for r in rows}):
            subset=[r for r in rows if r['arm']==arm and r['edit']==edit and r['group']==group and r['pixels']]
            pixels=sum(r['pixels'] for r in subset)
            metrics={key:sum(r['pixels']*r['metrics'][key] for r in subset)/pixels for key in ['raw_mae','raw_mse','clamped_mae','clamped_mse']} if pixels else {}
            summary.append(dict(arm=arm,edit=edit,group=group,pixels=pixels,metrics=metrics))
result=dict(status='SAME_QUERY_WORLD_VISIBILITY_STRATIFIED_AUDIT',rows=rows,pixel_weighted_summary=summary,
    mask_definition='original deleted-view newly exposed road/lane world points; visible_old groups from exact original16 camera FOV+BVH first-hit tests; intersect current edited independent road semantics',
    GT_used_only_for_evaluation=True,training_or_threshold_selection_uses_masks=False,quality_accepted=False,human_verdict=None)
(OUT/'observed_unseen_road_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],rows=len(rows))))
