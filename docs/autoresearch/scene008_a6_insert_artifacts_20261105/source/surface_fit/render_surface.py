"""Short actual native renders of the fixed depth posterior; no colour refit."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
import argparse,json,hashlib,copy,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAM/'scene008_a6_appearance_formal_20261105'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True);p.add_argument('--raws',default='106,113,130');args=p.parse_args()
    sys.path.insert(0,str(CAM/'scene008_a6_asset_controls_20261105/source'))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime,raster,tagged
    root=Path(args.fit);outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=False)
    names=['masked_prototype','smooth_depth_only','masked_matte_initialization','smooth_depth_matte_initialization']
    paths={name:root/(name+'.npz') for name in names}
    paths={'unmasked_original_prototype':OLD/'output_adapter/projective_surface/projective_surface_asset.npz',**paths}
    hashes={k:sha(v) for k,v in paths.items()};assets={k:SHAsset.from_geometry(v).cuda() for k,v in paths.items()}
    for a,b in [('masked_prototype','smooth_depth_only'),('masked_matte_initialization','smooth_depth_matte_initialization')]:
        assert torch.equal(assets[a].coefficients,assets[b].coefficients)
        assert torch.equal(assets[a].opacity,assets[b].opacity)
        assert torch.equal(assets[a].resample_reference_pixel_xy,assets[b].resample_reference_pixel_xy)
    rt=Runtime();start=time.time();records=[]
    def hr(c):
        c=copy.copy(c);c.image_width=1920;c.image_height=1280
        for k in ('fx','fy','cx','cy'):setattr(c,k,getattr(c,k)*4)
        return c
    def cpu(x):return x.detach().cpu().numpy()
    def png(p,x):Image.fromarray(np.uint8(np.clip(x,0,1)*255+.5)).save(p)
    with torch.no_grad():
        for raw in map(int,args.raws.split(',')):
            d=rt.raw(raw);c=hr(d['cam']);images={};alpha={};depth={}
            folder=outdir/f'raw{raw}';folder.mkdir()
            for name,a in assets.items():
                rows=tagged(a.rows(d['pose'],c,rt.pipe,1,precomputed=True),'asset')
                result=raster(c,rt.pipe,0,rows,rt.n.bg.new_zeros(3))
                assert all(torch.isfinite(result[k]).all() for k in ('rgb','alpha','depth_sum'))
                images[name]=cpu(result['rgb']).transpose(1,2,0);alpha[name]=cpu(result['alpha'])[0]
                depth[name]=cpu(result['depth_sum']/result['alpha'].clamp_min(1e-6)/rt.scale)[0]
                png(folder/(name+'_black_1920.png'),images[name])
                del result,rows
            # One camera and one common physical crop across all variants.
            union=(alpha['unmasked_original_prototype']>.01)|(alpha['smooth_depth_only']>.01)
            y,x=np.where(union);crop=[max(0,int(x.min())-16),max(0,int(y.min())-16),min(1920,int(x.max())+17),min(1280,int(y.max())+17)]
            x0,y0,x1,y1=crop;w=x1-x0;h=y1-y0
            panel=Image.new('RGB',(w*len(assets),h+35),(242,246,249));dr=ImageDraw.Draw(panel)
            for i,(name,image) in enumerate(images.items()):
                im=Image.fromarray(np.uint8(np.clip(image[y0:y1,x0:x1],0,1)*255+.5));panel.paste(im,(i*w,35));dr.text((i*w+4,8),name,fill=(15,30,45))
                png(folder/(name+'_crop.png'),image[y0:y1,x0:x1])
            panel.save(folder/'depth_only_comparison.png')
            # Actual unchanged scene parameters, with changed asset geometry.
            ygrid,xgrid=torch.meshgrid(torch.arange(1280,device='cuda',dtype=torch.float32),torch.arange(1920,device='cuda',dtype=torch.float32),indexing='ij')
            directions=torch.stack(((xgrid+.5-c.cx)/c.fx,(ygrid+.5-c.cy)/c.fy,torch.ones_like(xgrid)),-1)
            directions=torch.nn.functional.normalize(directions,dim=-1)@c.c2w[:3,:3].T
            env=rt.n.env(directions).permute(2,0,1)
            for name in ('masked_prototype','smooth_depth_only','smooth_depth_matte_initialization'):
                for delete in (False,True):
                    scene=rt.scene(raw,assets[name],delete=delete)
                    result=raster(c,rt.pipe,rt.n.static.active_sh_degree,scene['rows'],scene['bg'])
                    image=result['rgb']+(1-result['alpha'])*env
                    assert scene['meta']['native_a6_rows_input']==0
                    if delete:assert scene['meta']['native_a37_packed_rows']==0
                    png(folder/(name+('_delete' if delete else '_keep')+'_1920.png'),cpu(image).transpose(1,2,0))
                    del scene,result,image
            np.savez_compressed(folder/'actual_black_crop.npz',crop_xyxy=crop,
                **{name+'_alpha':v[y0:y1,x0:x1] for name,v in alpha.items()},
                **{name+'_depth_m':v[y0:y1,x0:x1] for name,v in depth.items()},
                **{name+'_RGB':v[y0:y1,x0:x1] for name,v in images.items()})
            rec=dict(raw=raw,full_image_hw=[1280,1920],crop_xyxy=crop,finite=True,
                     native_a6_rows_input=0,delete_a37_packed_rows=0,geometry_changed=True,
                     matched_prototype_and_depth_only_RGB_coefficients_bitwise_identical=True,
                     mattetexture_pair_RGB_coefficients_bitwise_identical=True)
            records.append(rec);write(folder/'receipt.json',rec)
            print(json.dumps(dict(stage='actual_surface_depth_render',raw=raw)),flush=True)
            del env,directions,ygrid,xgrid,d,c;rt._data.clear();torch.cuda.empty_cache()
        assert rt.modelunchanged() and all(a.assert_geometry_unchanged() for a in assets.values())
        assert all(sha(paths[k])==h for k,h in hashes.items())
    receipt=dict(status='PASS_ACTUAL_NATIVE_DEPTH_ONLY_PREVIEW',frames=records,assets={k:dict(path=str(v),sha256=hashes[k],rows=len(assets[k].xyz),geometry_hash=assets[k].geometry_hash) for k,v in paths.items()},
        no_colour_optimization=True,PVD_training_updates=0,background_parameters_unchanged=True,
        no_geometry_unchanged_claim_across_variants=True,held_used_for_final_preview_only=True,seconds=time.time()-start,
        limitation='Three views are a geometry/initialization control, not all49 adoption or independent reconstruction evidence')
    write(outdir/'receipt.json',receipt);print(json.dumps(dict(status='PASS',seconds=receipt['seconds'],output=str(outdir))),flush=True)


if __name__=='__main__':main()
