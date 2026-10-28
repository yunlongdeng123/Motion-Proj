"""渲染后CPU审计：只读验收；结果不会参与检测标签。"""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion
from PIL import Image
HERE=Path(__file__).resolve().parent
records=[]
for run in ['native_strict_smoke_r1','native_photo_smoke_r1','native_shadow_only_smoke_r1','native_target_contact_smoke_r1']:
    for raw in [98,113,114]:
        source=np.load(HERE/'shadow_evidence_r3'/f'{raw:03d}.npz')
        labels=np.load(HERE/'native_target_contact_smoke_r1/evidence'/f'{raw:03d}.npz')
        body=source['body'];effect=labels['pixel_shadow_effect'].astype(bool)
        regions={'body':body,'body_core':binary_erosion(body,iterations=2),'effect':effect,
                 'outside_effect':~binary_dilation(effect,iterations=2),
                 'outside_body_and_effect':~binary_dilation(body|effect,iterations=2),
                 'weak_road_counterevidence':source['road_counterevidence_valid']}
        base=np.array(Image.open(HERE/run/'baseline/native_rgb'/f'{raw:03d}.png')).astype(float)/255
        ba=np.load(HERE/run/'baseline'/f'{raw:03d}_maps.npz')['alpha'].astype(float)
        for mode in ['remove_car','remove_u','remove_car_u']:
            rgb=np.array(Image.open(HERE/run/mode/'native_rgb'/f'{raw:03d}.png')).astype(float)/255
            alpha=np.load(HERE/run/mode/f'{raw:03d}_maps.npz')['alpha'].astype(float)
            delta=np.abs(rgb-base).mean(-1)
            for name,mask in regions.items():
                records.append(dict(run=run,raw=raw,mode=mode,region=name,pixels=int(mask.sum()),
                    visible_rgb_changed_pixels=int((delta[mask]>1/255+1e-8).sum()),
                    visible_rgb_mean_absolute_change=float(delta[mask].mean()) if mask.any() else None,
                    alpha_mean_drop=float(np.maximum(ba-alpha,0)[mask].mean()) if mask.any() else None))
out=dict(schema='read-only-native-controls-audit-v1',not_used_for_labels=True,regions_are_diagnostic_not_GT=True,
         human_verdict=None,records=records,quality_accepted=False)
(HERE/'native_control_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('verified',len(records),'region records; no model or label writes')
