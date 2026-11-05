"""Insert any persistent asset into the old native Runtime, without modifying it.

480 native/analytic-area scenes and 1920 black details are separate viewport
controls of one canonical asset. No intrinsic, pose or negative-FoV changes.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from fit_surface import CAM,ART,sha,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--raws',default='106,113,130');p.add_argument('--width',type=int,choices=[480,1920],default=480)
    p.add_argument('--viewport',choices=['native','area'],default='native');p.add_argument('--state',choices=['black','keep','delete'],default='black')
    args=p.parse_args()
    if args.width==1920 and args.state!='black':raise ValueError('Main street uses native480; HR is black detail only')
    sys.path.insert(0,str(CAM/'scene008_a6_asset_controls_20261105/source'));sys.path.insert(0,str(ART/'source/scale'))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime,raster,tagged
    from relative_area_asset import RelativeAreaOpacityAsset
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False);asset_path=Path(args.asset);before=sha(asset_path)
    base=SHAsset.from_geometry(asset_path).cuda();asset=base if args.viewport=='native' else RelativeAreaOpacityAsset(base)
    rt=Runtime();records=[]
    with torch.no_grad():
        for raw in map(int,args.raws.split(',')):
            data=rt.raw(raw);camera=data['cam']
            if args.width==1920:
                camera=copy.copy(camera);camera.image_width=1920;camera.image_height=1280
                for k in ('fx','fy','cx','cy'):setattr(camera,k,getattr(camera,k)*4)
            if args.state=='black':
                rows=tagged(asset.rows(data['pose'],camera,rt.pipe,1,precomputed=True),'asset')
                out=raster(camera,rt.pipe,0,rows,rt.n.bg.new_zeros(3));meta={}
            else:
                out=rt.render_scene(raw,asset,delete=args.state=='delete');meta=out['meta']
                assert meta['native_a6_rows_input']==0
                if args.state=='delete':assert meta['native_a37_packed_rows']==0
            cpu={k:out[k].detach().cpu().numpy() for k in ('rgb','alpha','depth_sum')}
            assert all(np.isfinite(v).all() for v in cpu.values())
            Image.fromarray(np.uint8(np.clip(cpu['rgb'].transpose(1,2,0),0,1)*255+.5)).save(dest/f'{raw:03d}.png')
            np.savez_compressed(dest/f'{raw:03d}.npz',**cpu,expected_depth_m=cpu['depth_sum']/np.maximum(cpu['alpha'],1e-5)/rt.scale,
                reference_pose=data['pose'].cpu().numpy(),world_view=camera.world_view_transform.cpu().numpy(),
                projection=camera.full_proj_transform.cpu().numpy())
            records.append(dict(raw=raw,state=args.state,width=args.width,viewport=args.viewport,meta=meta,
                area_record=asset.last_area_record if args.viewport=='area' else None))
            rt._data.clear();torch.cuda.empty_cache()
        assert rt.modelunchanged() and base.assert_geometry_unchanged() and sha(asset_path)==before
    write(dest/'receipt.json',dict(status='PASS_NATIVE_ASSET_INSERTION',asset=str(asset_path),asset_sha256=before,
        source_sha256=sha(__file__),geometry_hash=base.geometry_hash,frames=records,
        camera_pose_FoV_native_SO_unchanged=True,intrinsics_only_known_viewport_scaling=True,canonical_asset_opacity_unchanged=True,
        area_control_is_not_geometry_or_adoption_claim=True))

if __name__=='__main__':main()
