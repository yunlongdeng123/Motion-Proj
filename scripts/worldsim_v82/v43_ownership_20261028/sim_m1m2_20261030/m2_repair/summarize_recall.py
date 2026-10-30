import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030'); out=root/'m2_repair_r1'
base=json.loads((root/'m1_assignment_r1/recall_verified/reference_metrics.json').read_text())
lookup={(r['frame'],r['edit']):r for r in base['rows']}
result=[]
for run in ['recall_position','recall_position_scale','recall_position_scale_split']:
    folder=out/run
    if not (folder/'reference_metrics.json').exists(): continue
    mine=json.loads((folder/'reference_metrics.json').read_text()); rows=[]; checks=[]
    for frame in ['000000','000008','000015']:
        with np.load(folder/('selection_'+frame+'.npz')) as a, np.load(root/'m1_assignment_r1/recall_verified'/('selection_'+frame+'.npz')) as b:
            checks.append(dict(frame=frame,indices_identical=bool(np.array_equal(a['selected'],b['selected']))))
    for row in mine['rows']:
        prior=lookup[(row['frame'],row['edit'])]
        for region in ['full','body_reference','original_body_footprint','newly_exposed_road','road_reference']:
            metric=row['regions'][region]
            if 'raw_mae' not in metric: continue
            old=prior['regions'][region]['raw_mae']; new=metric['raw_mae']
            rows.append(dict(frame=row['frame'],edit=row['edit'],region=region,m1_mae=old,m2_mae=new,delta_m2_minus_m1=new-old,relative_reduction_pct=100*(old-new)/old if old else None))
    result.append(dict(run=run,m1_baseline='recall_verified',selection_checks=checks,rows=rows,repairs=json.loads((folder/'responsibility_summary.json').read_text())['frames']))
(out/'recall_comparison.json').write_text(json.dumps(result,indent=2))
with np.load(root/'dataset_r2/privileged/heldout/000008.npz') as d:
    original_body=d['semantic_id']==10
heldout=[]
for run in ['recall_position','recall_position_scale','recall_position_scale_split']:
    folder=out/('heldout_'+run)
    if not (folder/'render_manifest.json').exists(): continue
    manifest=json.loads((folder/'render_manifest.json').read_text()); scores=[]
    for row in manifest['frames']:
        with np.load(row['fp32']) as data: pred=data['final']
        refpath=root/'heldout_edit_refs_r1'/row['edit']/'000008.png'
        target=np.asarray(Image.open(refpath).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
        regions={}
        for name,mask in [('full',np.ones_like(original_body)),('original_body_footprint',original_body)]:
            delta=pred[:,mask].astype(np.float64)-target[:,mask].astype(np.float64)
            regions[name]=dict(pixels=int(mask.sum()),raw_mae=float(np.abs(delta).mean()),raw_mse=float((delta**2).mean()))
        scores.append(dict(frame='000008',edit=row['edit'],regions=regions,reference=str(refpath),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest()))
    result_h=dict(status='SCORED_FIXED_TRAINFRAME_REPAIR_ON_HELDOUT_EDITS',run=run,rows=scores,
        repair='exact frame8 repaired geometry replay, no heldout geometry/GT RGB selection or optimization',
        region_scope='fixed original heldout body footprint, not edited-body label',
        reference_scope='new independent heldout edited renders; none differs from earlier none on 51/460800 channels, max5/255, per reference manifest',
        human_verdict=None,quality_accepted=False)
    (folder/'heldout_edit_metrics.json').write_text(json.dumps(result_h,indent=2)); heldout.append(result_h)
(out/'heldout_recall_comparison.json').write_text(json.dumps(heldout,indent=2))
print(json.dumps([dict(run=r['run'],midframe=[x for x in r['rows'] if x['frame']=='000008' and x['region']=='full'],splits=[x.get('split_parents',0) for x in r['repairs']]) for r in result],indent=2))
