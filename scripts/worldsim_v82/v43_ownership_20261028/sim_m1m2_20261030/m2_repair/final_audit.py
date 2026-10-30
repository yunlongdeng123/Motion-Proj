import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030'); out=root/'m2_repair_r1'
stages=['position','position_scale','position_scale_split']
parities=[]
for stage in stages:
    for prefix in ['', 'heldout_']:
        old=out/(prefix+'recall_'+stage); new=out/(prefix+'recall_'+stage+'_depth_r2')
        if not (new/'render_manifest.json').exists(): continue
        ma=json.loads((old/'render_manifest.json').read_text()); mb=json.loads((new/'render_manifest.json').read_text())
        values={k:0. for k in ['C','alpha','E','final']}; depth=0.
        for a,b in zip(ma['frames'],mb['frames']):
            with np.load(a['fp32']) as aa,np.load(b['fp32']) as bb:
                for key in values: values[key]=max(values[key],float(np.max(np.abs(aa[key]-bb[key]))))
                depth=max(depth,float(np.max(np.abs(aa['depth_sum_model']-bb['depth_sum_model']))))
        assert values['final']<=2e-6
        parities.append(dict(old=str(old),corrected=str(new),max_abs=values,depth_sum_change_max_abs=depth,training_updates=0))
(out/'depth_replay_parity.json').write_text(json.dumps(parities,indent=2))
with np.load(root/'dataset_r2/privileged/heldout/000008.npz') as g: body=g['semantic_id']==10
result=[]
for name in ['refit_assignment','refit_split']:
    folder=out/name
    if not (folder/'reference_metrics.json').exists(): continue
    summary=json.loads((folder/'responsibility_summary.json').read_text())
    local=summary['frames'][0]['local_refit']; assert local['updates']==200 and local['save_reload_render_max_abs']<=2e-6
    for fn in ['render_manifest.json','execution.json']:
        p=folder/fn; d=json.loads(p.read_text()); d['training_updates']=200; d['training_updates_scope']='single time8 local DC/opacity control'; p.write_text(json.dumps(d,indent=2))
    train=json.loads((folder/'reference_metrics.json').read_text())
    heldout=[]
    hf=out/('heldout_'+name)
    if (hf/'render_manifest.json').exists():
        manifest=json.loads((hf/'render_manifest.json').read_text())
        for row in manifest['frames']:
            with np.load(row['fp32']) as data: pred=data['final']
            ref=root/'heldout_edit_refs_r1'/row['edit']/'000008.png'
            target=np.asarray(Image.open(ref).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
            regions={}
            for key,mask in [('full',np.ones_like(body)),('original_body_footprint',body)]:
                regions[key]={}
                for label,x in [('raw',pred),('clamped',pred.clip(0,1))]:
                    delta=x[:,mask].astype(np.float64)-target[:,mask].astype(np.float64)
                    regions[key][label+'_mae']=float(np.abs(delta).mean())
                    regions[key][label+'_mse']=float((delta**2).mean())
            heldout.append(dict(edit=row['edit'],regions=regions,reference=str(ref)))
        (hf/'heldout_edit_metrics.json').write_text(json.dumps(dict(rows=heldout,region_scope='fixed original body footprint only; not edited body',refit='original trainRGB only, fixed DC/opacity replay'),indent=2))
    result.append(dict(name=name,local_refit=local,train_rows=train['rows'],heldout_rows=heldout))
(out/'local_refit_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(depth_replay_arms=len(parities),refits=[dict(name=x['name'],trace=x['local_refit']['loss_trace'],parity=x['local_refit']['save_reload_render_max_abs'],train=[dict(edit=y['edit'],raw=y['regions']['full']['raw_mae']) for y in x['train_rows']]) for x in result]),indent=2))
