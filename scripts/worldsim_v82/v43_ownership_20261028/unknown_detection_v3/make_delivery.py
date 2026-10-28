from pathlib import Path
import json
import numpy as np
from classify_detection import REASONS,CONFIG
HERE=Path(__file__).resolve().parent
CTX='14705303724557273004_3105_000_3125_000'
RUN=HERE/'native_target_contact_smoke_r1'
summary=json.loads((RUN/'summary.json').read_text())
assert summary['source_checkpoint_unchanged'] and not summary['quality_accepted']
sources=set(range(77,126))
for p in (HERE/'shadow_evidence_r3').glob('[0-9][0-9][0-9].npz'):
    with np.load(p) as a:
        for k in ['source_raw','motion_witness_raw']:
            if k in a:sources.update(int(x) for x in np.unique(a[k]) if 0<=x<199)
delivery=dict(schema='target-detection-ui-v1',context=CTX,target_track='gTriXsyDrbzzU6apS1WErQ',static_total=918850,
    source_checkpoint='V42 static/environment 30K + V43 actor checkpoint010000.pth',checkpoint_frozen=True,
    geometry_base_group='v43_actor_10000',actor=dict(cloud='v43:10000',role_default=1,row_namespace='frozen_v43_actor10k',source_raws=list(range(77,126))),
    raws=list(range(77,126)),globalsource_frames=sorted(sources),reason_labels={str(v):k for k,v in REASONS.items()},thresholds=CONFIG,
    quality_accepted=False,human_verdict=None,source_note='检测候选尚未通过：整行剥离损伤背景，阴影不完整；紫色是独立弱effect，非car语义。',frames={})
for raw in [98,113,114]:
    receipt=dict(status='complete',renderer='native_gaussian',source_raw=raw,source_checkpoint_unchanged=True,training_updates=0)
    delivery['frames'][str(raw)]=dict(source_raw=raw,evidence=str(RUN/'evidence'/f'{raw:03d}.npz'),
        images={mode:str(RUN/mode/'native_rgb'/f'{raw:03d}.png') for mode in ['baseline','remove_car','remove_u','remove_car_u']},
        render_receipts={mode:receipt for mode in ['baseline','remove_car','remove_u','remove_car_u']},
        globalsource_frames=sorted(sources),metrics=[r for r in summary['records'] if r['raw']==raw],
        source_note='原始RGB/实测LiDAR独立候选；原生剥离只作验收，尚未通过。不把缺证视为车辆证据。')
(HERE/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
print(HERE/'delivery.json')
