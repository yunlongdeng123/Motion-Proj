"""Fixed49 coverage readout omitted by the first evaluator; no fitting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from fit_surface import ART,OLD,CAM,HELD,sha,write

def main():
    dest=ART/'scale/surface_lod_v1/coverage49';dest.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(CAM/'scene008_a6_asset_controls_20261105/source'));sys.path.insert(0,str(ART/'source/scale'))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime,raster,tagged
    from surface_lod_asset import SurfaceLODAsset
    from metrics import fixed_masks
    final=ART/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz';prior=OLD/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
    base=SHAsset.from_geometry(final).cuda();old=SHAsset.from_geometry(prior).cuda()
    lod=SurfaceLODAsset(base,ART/'surface_fit/masked_depth_v1/posterior.npz');rt=Runtime();records=[];start=time.time()
    with torch.no_grad():
        for raw in range(89,138):
            d=rt.raw(raw);fm=fixed_masks(d);masks=dict(visible=fm['a6_visible'],extra_eroded_visible=binary_erosion(fm['a6_visible'],iterations=1),reviewed=fm['a6_reviewed'])
            alpha={}
            for label,asset in [('old_native',old),('final_native',base),('surface_LOD',lod)]:
                rows=tagged(asset.rows(d['pose'],d['cam'],rt.pipe,1,precomputed=True),'asset')
                out=raster(d['cam'],rt.pipe,0,rows,rt.n.bg.new_zeros(3));k=rt.render_scene(raw,asset)
                alpha[label]=dict(solo=out['alpha'][0].cpu().numpy(),joint_keep=k['component_alpha_T'][0].cpu().numpy())
            row=dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',values={})
            for label,states in alpha.items():
                row['values'][label]={}
                for state,a in states.items():
                    row['values'][label][state]={}
                    for name,mask in masks.items():
                        x=a[mask]
                        row['values'][label][state][name]=dict(count=int(mask.sum()),sum=float(x.astype(float).sum()),mean=float(x.mean()) if len(x) else None,
                            min=float(x.min()) if len(x) else None,alpha_lt09=int((x<.9).sum()),alpha_lt05=int((x<.5).sum()),
                            zero_alpha_le1e6=int((x<=1e-6).sum()),
                            LOD_new_lt09_vs_final_native=int(((alpha['surface_LOD'][state]<.9)&(alpha['final_native'][state]>=.9)&mask).sum()) if label=='surface_LOD' else None)
            records.append(row);rt._data.clear();torch.cuda.empty_cache()
    summary={}
    for split in ('all49','BUILD','HELD'):
        rows=records if split=='all49' else [r for r in records if r['split']==split];values={}
        for label in records[0]['values']:
            values[label]={}
            for state in ('solo','joint_keep'):
                values[label][state]={}
                for name in masks:
                    fields=[r['values'][label][state][name] for r in rows];n=sum(x['count'] for x in fields);total=sum(x['sum'] for x in fields)
                    v=dict(count=n,sum=total,mean=total/n if n else None,min=min(x['min'] for x in fields if x['min'] is not None))
                    for key in ('alpha_lt09','alpha_lt05','zero_alpha_le1e6'):v[key]=sum(x[key] for x in fields)
                    if label=='surface_LOD':v['LOD_new_lt09_vs_final_native']=sum(x['LOD_new_lt09_vs_final_native'] for x in fields)
                    values[label][state][name]=v
        summary[split]=values
    write(dest/'metrics.json',records);write(dest/'summary.json',summary)
    assert rt.modelunchanged() and lod.assert_persistent_unchanged() and old.assert_geometry_unchanged()
    write(dest/'receipt.json',dict(status='PASS_FIXED49_COVERAGE_READOUT',frames=49,source_sha256=sha(__file__),
        output_only=True,method_or_parameter_or_training_updates=0,additional_raster_forwards=49*6,
        fixed_masks='Original visible, original visible eroded once, original complete reviewed silhouette',
        joint_alpha='actual asset alpha*T, not total scene alpha',
        role='Necessary coverage count omitted by earlier fixed49 metrics; no candidate tuning',seconds=time.time()-start))
    print(dict(status='PASS',seconds=time.time()-start,values=summary['all49']))

if __name__=='__main__':main()
