"""同一PNG/clamp口径比较三维原图与预训练补全；三训练视图不冒称泛化集。"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--all-49',action='store_true');a=p.parse_args();root=Path(a.root)
methods={
 'unrepaired_native':root/'delete_all_unrepaired/native_rgb',
 'repaired_native':root/'delete_all_repaired/native_rgb',
 'unrepaired_video':root/'delete_all_unrepaired/propainter/model_png',
 'repaired_video':root/'delete_all_repaired/propainter/model_png'}
if (root/'delete_all_evidence_routed/propainter/summary.json').exists():
    methods['evidence_routed_video']=root/'delete_all_evidence_routed/propainter/model_png'
if (root/'delete_all_direct_reference/propainter/summary.json').exists():
    methods['direct_reference_video']=root/'delete_all_direct_reference/propainter/model_png'
rows=[]
views=list(range(89,138)) if a.all_49 else [102,129,137]
reference_root=root.parent/'phase2_reference49/crossview' if a.all_49 else Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026/crossview')
for raw in views:
    d=dict(np.load(reference_root/f'{raw}.npz'))
    refs=np.clip(d['rgb'],0,1);mask=d['valid'];row=dict(raw=raw,labeled_pixels=int(mask.sum()),full_target=int(d['target'].sum()),unknown=int((d['target']&~mask).sum()),methods={})
    for key,path in methods.items():
        image=np.asarray(Image.open(path/f'{raw:03d}.png'))/255.
        error=np.abs(image-refs).mean(2)
        row['methods'][key]=dict(MAE=float(error[mask].mean()),actors={})
        for actor,label in [(6,4),(37,5)]:
            selected=mask&(d['labels']==label)
            row['methods'][key]['actors'][str(actor)]=dict(labeled_pixels=int(selected.sum()),MAE=float(error[selected].mean()) if selected.any() else None)
    rows.append(row)
aggregate={method:sum(x['methods'][method]['MAE']*x['labeled_pixels'] for x in rows)/sum(x['labeled_pixels'] for x in rows) for method in methods}
groups={}
for group,chosen in [('prototype_optimized_views',[x for x in rows if x['raw'] in [102,129,137]]),('other_same_sequence_views',[x for x in rows if x['raw'] not in [102,129,137]])]:
    denominator=sum(x['labeled_pixels'] for x in chosen)
    if denominator:groups[group]=dict(labeled_pixels=denominator,frames=len(chosen),weighted_MAE={m:sum(x['methods'][m]['MAE']*x['labeled_pixels'] for x in chosen)/denominator for m in methods})
summary=dict(rows=rows,weighted_labeled_MAE=aggregate,reference_role='已参与原型优化的三个开发视图，仅作闭环/失效发现',
    metric_domain='clamp RGB PNG vs float crossview reference；共同量化口径',
    full49_video_frames=49,groups=groups,heldout_scene_count=0,quality_accepted=False,human_verdict=None)
if a.all_49:summary['reference_role']='三个原型优化视图与46同片段转移视图分开；来源曾进入初始化，均非独立GT/新场景测试'
(root/('pipeline_evaluation49.json' if a.all_49 else 'pipeline_evaluation.json')).write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
