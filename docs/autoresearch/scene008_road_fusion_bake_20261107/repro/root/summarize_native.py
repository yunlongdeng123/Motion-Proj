from pathlib import Path
import json
import numpy as np
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
d=json.loads((RUN/'results/full49/receipt.json').read_text())
pooled=[r for r in d['pooled'] if r['metric'] in ('a6','a37','road','protected')]
ref={(r['state'],r['metric']):r for r in pooled if r['variant']=='baseline'}
for r in pooled:
    b=ref[r['state'],r['metric']]['mean'];r['delta']=r['mean']-b if r['mean'] is not None and b is not None else None
materials={}
for name in ('direct','fusion'):
    with np.load(RUN/f'engineer/material_fit_v1/{name}/material_override.npz') as z:
        before=z['initial_colours_srgb'];after=z['colours_srgb'];delta=after-before
        materials[name]=dict(mean_abs_RGB_change=float(abs(delta).mean()),max_abs_RGB_change=float(abs(delta).max()),
            rows_changed_gt001=int((abs(delta).max(1)>.01).sum()),RGB_range=[float(after.min()),float(after.max())],
            zero_one_channels=int(((after==0)|(after==1)).sum()))
result=dict(pooled=pooled,material_changes=materials,
    exact_geometry_comparisons=sum(r['variant']!='baseline' and all(r['geometry_fields_bitwise_equal_baseline'].values()) for r in d['records']),
    baseline_PNG_exact=sum(r['baseline_parent_PNG_exact'] is True for r in d['records']))
(RUN/'root/native_summary_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
