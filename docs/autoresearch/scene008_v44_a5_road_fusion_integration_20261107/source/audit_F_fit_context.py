"""Supplement the F400 array audit with annotated BUILD target context (CPU)."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image, ImageDraw

R = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
O = R.parent/'scene008_instances_20261105_r1/observations/frames'
OUT = R/'review/identity_supervision_audit_F400_v1'
CFG = json.loads((R/'F_formal400_r1/executed_config.json').read_text())
ROWS = []


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def boundary(m):
    return m & ~(np.roll(m,1,0)&np.roll(m,-1,0)&np.roll(m,1,1)&np.roll(m,-1,1))


for raw in [94,105,128,137]:
    tp = Path(CFG['targets'][str(raw)]['path'])
    assert digest(tp) == CFG['targets'][str(raw)]['sha256']
    with np.load(tp) as z:
        target=z['edgeblend_rgb'].copy();fit=z['fit_mask'].copy();weight=z['edgeblend_weight'].copy()
    with np.load(O/f'{raw:03d}.npz') as z:
        observed=z['rgb'].copy();full37=z['actor_labels']==5;road=z['road'].copy()
    with np.load(R/f'F_formal400_r1/evaluation/frames/{raw:03d}/delete_float.npz') as z:
        candidate=z['rgb'].transpose(1,2,0).copy()
    roi=(180,230,360,320) if raw==94 else (150,250,380,320)
    x0,y0,x1,y1=roi;w=(x1-x0)*3;h=(y1-y0)*3
    board=Image.new('RGB',(3*w,h+65),'white');d=ImageDraw.Draw(board)
    d.text((6,5),f'raw{raw}: cyan=fit boundary; magenta=full37 boundary; nearest x3; post-result diagnostic only',fill='black')
    for i,(name,arr,overlay) in enumerate([('Observed',observed,False),('C target + original fit boundary',target,True),('F400 DELETE',candidate,False)]):
        rgb=np.rint(np.clip(arr,0,1)*255).astype(np.uint8)
        if overlay: rgb[boundary(fit)]=[0,255,255]
        rgb[boundary(full37)]=[255,0,255]
        tile=Image.fromarray(rgb).crop(roi).resize((w,h),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST)
        board.paste(tile,(i*w,50));d.text((i*w+8,28),name,fill='black')
    p=OUT/f'raw{raw:03d}_source_fit_context.png';board.save(p)
    # A fixed, explicitly diagnostic road patch below the observed car in raw128.
    patch=np.zeros_like(fit);patch[285:305,225:305]=True
    ys,xs=np.where(patch & fit)
    ROWS.append({'raw':raw,'crop_xyxy':list(roi),'path':str(p),'sha256':digest(p),'size':list(board.size),
                 'diagnostic_lower_patch_xyxy':[225,285,305,305],
                 'patch_fit_pixels':int((patch&fit).sum()),'patch_fit_actor37_pixels':int((patch&fit&full37).sum()),
                 'patch_fit_road_pixels':int((patch&fit&road).sum()),
                 'patch_fit_target_luma_mean':float((target[patch&fit] @ np.array([.2126,.7152,.0722])).mean()) if len(xs) else None,
                 'patch_target_observed_max_abs':float(np.abs(target[patch&fit]-observed[patch&fit]).max()) if len(xs) else None})

report={'status':'PASS_CPU_CONTEXT_RENDERED_NO_METHOD_CHANGE','script_sha256':digest(__file__),
        'actual_gpu_forwards':0,'actual_backwards':0,'actual_optimizer_updates':0,
        'interpretation_limits':['The fit mask excludes every labelled observed vehicle body pixel.',
            'It can include visually dark road/shadow pixels; the original target is observed RGB there.',
            'A dark source-road patch is not a clean-delete road target; these arrays alone do not assign every F stripe to a particular source shadow.',
            'The output stripe overlaps the old body footprint in raw94 while the actual fit mask excludes that footprint; direct vehicle-body loss leakage is not demonstrated.'],
        'rows':ROWS}
p=OUT.parent/'F_source_fit_context.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(p),'sha256':digest(p),'figures':len(ROWS)}))
