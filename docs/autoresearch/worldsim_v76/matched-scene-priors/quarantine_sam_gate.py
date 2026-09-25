"""保留12视图失败证据，将不合格动态先验移出训练入口。"""
import datetime
import json
from pathlib import Path

root=Path('/root/autodl-tmp/data/v76_vadgs')
for name in ('scene_0230','scene_0255'):
    scene=root/name
    source=scene/'sam_masks'
    destination=scene/'sam_prior_evidence/rejected_box_only_sam_masks_20260926'
    assert source.is_dir() and not destination.exists()
    assert len(list(source.glob('*.png')))==6
    source.rename(destination)
    marker={'failure_id':'V76-F02','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reason':'Projected hidden pedestrian boxes lead SAM to label foreground car/sign as pedestrian; uint8 ID and shape checks alone miss this.',
        'evidence':'/root/autodl-tmp/data/v76_vadgs/sam_occlusion_audit',
        'preserved_dynamic_masks':str(destination),
        'decision':'Do not bulk-generate box-only actor labels or launch matched-scene training until visible identity gate passes.',
        'pre_registered_fallback':'Associate class-compatible visible 2D detections to projected 3D tracks, then prompt SAM; validate on the same fixed positives and occlusions before expansion.',
        'background_sam_may_continue':True,'human_verdict':None}
    (scene/'sam_prior_evidence/DYNAMIC_IDENTITY_BLOCKED.json').write_text(json.dumps(marker,indent=2)+'\n')
    print(name,'preserved six masks; dynamic entry blocked')
