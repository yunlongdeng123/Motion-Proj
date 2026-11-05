"""Replay the persistent S/F viewport sampler in the existing native scene.

Requires the declared AutoDL runtime/background dependencies. A naked canonical
NPZ uses dense default rows and does not automatically reproduce the LOD image.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,copy,time
from pathlib import Path
import numpy as np
import torch
from probe_surface_lod import Runtime,SHAsset,SurfaceLODAsset,scene,render,sha,row_hash,camera_hr,environment
import probe_surface_lod as probe
from scale_probe import image
from diff_gaussian_rasterization import _C

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset',type=Path,default=Path(__file__).with_name('complete_a6.npz'))
    ap.add_argument('--surface',type=Path,default=Path(__file__).with_name('posterior.npz'))
    ap.add_argument('--raw',type=int,default=113)
    ap.add_argument('--width',type=int,choices=(480,1920),default=480)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    before=dict(asset=sha(a.asset),surface=sha(a.surface),active_SO=sha(_C.__file__))
    rt=Runtime();base=SHAsset.from_geometry(a.asset).cuda();asset=SurfaceLODAsset(base,a.surface)
    data=rt.raw(a.raw);cam=data['cam'] if a.width==480 else camera_hr(data['cam'])
    E=environment(rt,cam,data);states={};asset_hash=None
    for name,delete in [('keep',False),('delete',True)]:
        rows,A,meta=scene(rt,a.raw,cam,asset,delete)
        current=row_hash(A)
        if asset_hash is not None:assert current==asset_hash
        asset_hash=current
        if delete:assert meta['native_a37_packed_rows']==0
        assert meta['native_a6_rows_input']==0
        out=render(rt,cam,rows)
        image(out['rgb']+(1-out['alpha'])*E).save(a.output/(name+'_street.png'))
        np.savez_compressed(a.output/(name+'_native_float.npz'),**out)
        states[name]=dict(meta=meta,LOD=copy.deepcopy(asset.last_lod_record))
        if name=='keep':
            solo=render(rt,cam,A);image(solo['rgb']).save(a.output/'solo_black.png')
            np.savez_compressed(a.output/'solo_native_float.npz',**solo)
    assert before==dict(asset=sha(a.asset),surface=sha(a.surface),active_SO=sha(_C.__file__))
    assert asset.assert_persistent_unchanged() and rt.modelunchanged()
    receipt=dict(status='PASS_FIXED_SF_LOD_REPLAY',raw=a.raw,viewport=[cam.image_width,cam.image_height],sources=before,
        keep_delete_A_rows_byte_exact=True,states=states,seconds=time.time()-start,neural_weight_updates=0,
        persistent_surface_and_material_unchanged=True,source_retained_rows_unchanged=True,native_raster_calls=probe.CALLS,
        naked_NPZ_does_not_enable_LOD=True,automatic_adoption=False)
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
