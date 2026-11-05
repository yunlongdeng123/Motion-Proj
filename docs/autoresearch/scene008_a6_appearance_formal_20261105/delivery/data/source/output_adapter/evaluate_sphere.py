"""Fixed9HELD true-HR RGB and fixed49 LiDAR/coverage, no fitting."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_a6_appearance_formal_20261105'
sys.path.insert(0,str(RUN/'source/output_adapter'))
from fit_sphere import DATA_ROOT,BASE_ASSET,RAW,HELD,BUILD,camera_hr,write,sha
from scene_runtime import Runtime,raster,tagged
from sh_asset import SHAsset
from metrics import fixed_masks,expected_depth,metric,pool


def cpu(t):return t.detach().cpu().numpy()
def rgb8(a):return np.round(np.clip(a,0,1)*255).astype(np.uint8)


def black(rt,d,asset,cam):
    return raster(cam,rt.pipe,0,tagged(asset.rows(d['pose'],cam,rt.pipe,1,precomputed=True),'asset'),rt.n.bg.new_zeros(3))


@torch.no_grad()
def main():
    dest=DATA_ROOT/'evaluation';dest.mkdir(exist_ok=False);start=time.time()
    paths=dict(thin_HR=RUN/'highres/formal/complete_a6.npz',sphere_HR=DATA_ROOT/'formal/complete_a6.npz')
    rt=Runtime();assets={name:SHAsset.from_geometry(path).cuda() for name,path in paths.items()}
    before=SHAsset.from_geometry(BASE_ASSET).cuda()
    assert assets['sphere_HR'].geometry_hash==before.geometry_hash
    lidar=[];held=[];images=[]
    for raw in range(89,138):
        d=rt.raw(raw);masks=fixed_masks(d)
        record=dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',methods={})
        for name,asset in assets.items():
            out=black(rt,d,asset,d['cam'])
            alpha=out['alpha'][0];z=expected_depth(out,rt.scale)
            record['methods'][name]=dict(
                LiDAR_depth_error=metric(np.abs(z-d['obs']['depth_m'][0]),masks['a6_depth']),
                LiDAR_alpha_coverage=metric(alpha,masks['a6_depth']),
                visible_alpha=metric(alpha,masks['a6_visible']),
                reviewed_alpha=metric(alpha,masks['a6_reviewed']),
                reviewed_alpha_below05=metric((alpha<.5).float(),masks['a6_reviewed']))
        lidar.append(record)
        if raw in HELD or raw in (106,113):
            cam=camera_hr(d['cam']);packages={name:black(rt,d,asset,cam) for name,asset in assets.items()}
            if raw in HELD:
                image=np.asarray(Image.open(RAW/'image/image_0'/f'0008{raw:03d}.png').convert('RGB')).astype(np.float32)/255
                low=binary_erosion(d['a6_visible'].cpu().numpy(),iterations=1)
                mask=np.repeat(np.repeat(low,4,0),4,1)
                hr=dict(raw=raw,pixels=int(mask.sum()),methods={})
                for name,out in packages.items():
                    rgb=cpu(out['rgb']).transpose(1,2,0);a=cpu(out['alpha'])[0]
                    error=np.abs(rgb-image).mean(2);norm=np.abs(rgb/np.maximum(a[...,None],1e-5)-image).mean(2)
                    hr['methods'][name]=dict(RGB_MAE=float(error[mask].mean()),RGB_sum=float(error[mask].sum()),
                        normalized_MAE=float(norm[mask].mean()),normalized_sum=float(norm[mask].sum()),
                        mean_alpha=float(a[mask].mean()))
                held.append(hr)
            if raw in (106,113):
                pre=black(rt,d,before,cam)
                assert torch.equal(pre['alpha'],packages['sphere_HR']['alpha'])
                assert torch.equal(pre['depth_sum'],packages['sphere_HR']['depth_sum'])
                packages['sphere_before_color']=pre
                support=np.zeros((1280,1920),bool)
                for out in packages.values():support|=cpu(out['alpha'])[0]>.01
                y,x=np.where(support);box=(max(0,int(x.min())-24),max(0,int(y.min())-24),min(1920,int(x.max())+25),min(1280,int(y.max())+25))
                panels=[]
                for name in ('thin_HR','sphere_before_color','sphere_HR'):
                    out=packages[name];rgb=cpu(out['rgb']).transpose(1,2,0);alpha=cpu(out['alpha'])[0]
                    z=expected_depth(out,rt.scale)
                    per=dest/f'raw{raw:03d}'/name;per.mkdir(parents=True)
                    Image.fromarray(rgb8(rgb)).save(per/'rgb_black_hr.png')
                    Image.fromarray(np.round(np.clip(alpha,0,1)*65535).astype(np.uint16)).save(per/'alpha16_hr.png')
                    np.savez_compressed(per/'raster_hr.npz',rgb_premultiplied=rgb,alpha=alpha,expected_depth_m=z,
                        geometry_hash=np.asarray(before.geometry_hash if name=='sphere_before_color' else assets[name].geometry_hash))
                    tile=Image.fromarray(rgb8(rgb)).crop(box)
                    tile=tile.resize((round(tile.width*430/tile.height),430),Image.Resampling.LANCZOS)
                    panel=Image.new('RGB',(tile.width,462),(237,243,247));panel.paste(tile,(0,32));ImageDraw.Draw(panel).text((8,9),name,fill=(12,25,38));panels.append(panel)
                strip=Image.new('RGB',(sum(p.width for p in panels),462));offset=0
                for p in panels:strip.paste(p,(offset,0));offset+=p.width
                strip.save(dest/f'{raw:03d}_objects.png')
                images.append(dict(raw=raw,crop_xyxy=list(box),sphere_alpha_depth_before_after_colour_bitwise_identical=True))
        print(json.dumps(dict(stage='sphere_evaluate',raw=raw,complete=len(lidar))),flush=True)
        rt._data.clear();torch.cuda.empty_cache()
    totals=sum(r['pixels'] for r in held)
    hrpooled={name:dict(RGB_MAE=sum(r['methods'][name]['RGB_sum'] for r in held)/totals,
        normalized_MAE=sum(r['methods'][name]['normalized_sum'] for r in held)/totals) for name in assets}
    metrics=('LiDAR_depth_error','LiDAR_alpha_coverage','visible_alpha','reviewed_alpha','reviewed_alpha_below05')
    aggregate={split:{name:{key:pool(r['methods'][name][key] for r in rows) for key in metrics} for name in assets}
        for split,rows in [('all49',lidar),('BUILD',[r for r in lidar if r['split']=='BUILD']),('HELD',[r for r in lidar if r['split']=='HELD'])]}
    write(dest/'held_hr.json',dict(status='PASS',HELD=HELD,pixels=totals,pooled=hrpooled,rows=held,
        domain='Original1920x1280 RGB, same fixed old eroded visible mask nearest lifted4; no alpha-conditioned evaluation denominator',
        used_for_fitting=False,geometry_context_not_independently_HELD=True))
    write(dest/'lidar49.json',dict(status='PASS',all49_complete=True,rows=lidar,pooled=aggregate,
        resolution=[480,320],domain='Unchanged original observations depth_valid[0], visible[0], visible_reviewed[0]',
        depth_definition='Alpha-normalized expected camera-z, converted model units to metres',
        no_new_depth_supervision=True))
    assert rt.modelunchanged() and all(a.assert_geometry_unchanged() for a in assets.values()) and before.assert_geometry_unchanged()
    write(dest/'receipt.json',dict(status='PASS',held9_HR_pooled=hrpooled,LiDAR49_pooled=aggregate['all49'],images=images,
        assets={n:dict(path=str(p),sha256=sha(p),geometry_hash=assets[n].geometry_hash) for n,p in paths.items()},
        source_thickness_changed=True,PVD_weight_updates=0,positions_updates=0,sigma_updates=0,opacity_updates=0,
        evaluation_only=True,seconds=time.time()-start))
    print(json.dumps(dict(stage='sphere_eval_complete',status='PASS',held_hr=hrpooled,lidar49=aggregate['all49'],seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
