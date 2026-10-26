"""审计颜色改善是否同时满足米制深度一致性；期望深度不冒充首交面。"""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
meta=json.loads(Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json').read_text())
rows=[]
for raw in range(89,138):
    d=dict(np.load(ROOT/f'phase2_reference49/crossview/{raw}.npz'))
    mask=d['valid']&np.isfinite(d['depth'])&(d['depth']>0)
    row=dict(raw=raw,labeled_pixels=int(mask.sum()),optimized_view=raw in [102,129,137],methods={})
    for name in ['delete_all_unrepaired','delete_all_repaired']:
        a=dict(np.load(ROOT/f'phase1_native/{name}/float/{raw:03d}.npz'))
        depth=(a['depth_sum']/np.maximum(a['alpha'],1e-8))[0]/meta['meter_to_model']
        error=np.abs(depth-d['depth'])
        row['methods'][name]=dict(mean_expected_depth_absolute_difference_m=float(error[mask].mean()),
            median_expected_depth_absolute_difference_m=float(np.median(error[mask])),
            fraction_difference_below_03m=float((error[mask]<.3).mean()))
    rows.append(row)
groups={}
for group,chosen in [('optimized_views',[x for x in rows if x['optimized_view']]),('other46_views',[x for x in rows if not x['optimized_view']])]:
    count=sum(x['labeled_pixels'] for x in chosen)
    groups[group]={method:{metric:sum(x['methods'][method][metric]*x['labeled_pixels'] for x in chosen)/count
        for metric in ['mean_expected_depth_absolute_difference_m','fraction_difference_below_03m']} for method in rows[0]['methods']}
summary=dict(rows=rows,groups=groups,render_depth='原生alpha归一化贡献加权期望深度，非首交面',
    reference_depth='固定道路曲面逐射线深度，标签另有跨帧实测几何支持；非独立密集GT',
    only_diagnosis=True,new_training_updates=0,quality_accepted=False,human_verdict=None)
(ROOT/'depth_consistency_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(groups,indent=2))
