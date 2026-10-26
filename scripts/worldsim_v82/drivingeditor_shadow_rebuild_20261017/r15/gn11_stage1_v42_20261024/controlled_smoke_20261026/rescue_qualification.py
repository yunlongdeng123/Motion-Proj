"""Adaptive zero-update control: retain accepted observations, rescue only checked temporal consensus."""
from editor_common import *
from scipy.ndimage import binary_erosion
dest=OUT/'qualification_rescue';dest.mkdir(exist_ok=True);n=Native();rows=[]
with torch.no_grad():
    for arm in ['window49','full199']:
        a=dict(np.load(FULL/f'{arm}_initialization.npz'));co=dict(np.load(FULL/f'{arm}_color_consensus.npz'));added=co['eligible']&~a['eligible'];before=int(a['eligible'].sum());a['eligible']=a['eligible']|added;a['rgb'][added]=co['rgb'][added]
        path=dest/f'{arm}_initialization.npz';np.savez_compressed(path,**a);road=Road(path)
        detail=dict(arm=arm,original_active=before,rescued=int(added.sum()),new_active=int(a['eligible'].sum()),raw={})
        for raw in [102,129,137]:
            old=np.load(OUT/'coverage_audit'/f'{arm}_init_{raw}.npz');uid=old['unit_id'];state=old['state'];labels=old['labels'];safe=np.maximum(uid,0)
            c=n.camera(raw);direct=render_road(c,road,n.args);edit=n.render(c,road);alpha=cpu(direct['alpha']).squeeze();con=cpu(edit['contribution']).squeeze();stats={}
            for actor,k in [(37,5),(6,4)]:
                for region,mask in [('core',binary_erosion(labels==k,iterations=5)),('lower_strip',(labels==k)&(np.indices(labels.shape)[0]>=270))]:
                    excluded=mask&(state==1);counts=a['source_count'][safe];stats[f'a{actor}_{region}']=dict(pixels=int(mask.sum()),unsupported=int((mask&(state==0)).sum()),previously_ineligible=int(excluded.sum()),ineligible_zero_sources=int((excluded&(counts==0)).sum()),ineligible_one_source=int((excluded&(counts==1)).sum()),ineligible_multiple_color_failed=int((excluded&(counts>=2)).sum()),rescued_ray_cells=int((excluded&added[safe]).sum()),isolated_alpha_ge_half=int((mask&(alpha>=.5)).sum()),previous_isolated_alpha_ge_half=int((mask&(old['alpha']>=.5)).sum()),actual_contribution_ge_005=int((mask&(con>=.05)).sum()))
            detail['raw'][str(raw)]=dict(stats=stats,critical={f'{x},{y}':dict(unit_id=int(uid[y,x]),rescued=bool(added[safe[y,x]]),alpha=float(alpha[y,x]),contribution=float(con[y,x])) for x,y in [(255,240),(256,219)]})
            save(dest/f'{arm}_{raw}_road.png',direct['rgb']);save(dest/f'{arm}_{raw}_edit.png',edit['rgb'])
        rows.append(detail)
assert n.unchanged()
(dest/'result.json').write_text(json.dumps(dict(adaptive_after_diagnosing_first_source_outlier=True,rule='retain original eligible; add only consensus accepted with >=3 fit inliers, span>=4 frames, >=2 check inliers, both fractions>=60%, RGB MAE<=.05; alternating3-frame blocks',geometry_thresholds_unchanged=True,training_updates=0,master_unchanged=True,rows=rows,quality_accepted=False,human_verdict=None),indent=2));print(json.dumps(rows,indent=2))
