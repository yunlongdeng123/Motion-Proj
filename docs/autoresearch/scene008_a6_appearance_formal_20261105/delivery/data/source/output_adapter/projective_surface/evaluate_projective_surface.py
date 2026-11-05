"""Fixed49 evaluation and three-view native joint rendering of persistent3D chart."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import hashlib,json
from pathlib import Path
import sys,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_a6_appearance_formal_20261105'
ROOT=RUN/'output_adapter/projective_surface'
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr,RAW as RGBROOT,HELD
from scene_runtime import Runtime,raster,tagged
from sh_asset import SHAsset
from metrics import fixed_masks,expected_depth,metric,pool


def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu(t):return t.detach().cpu().numpy()
def rgb8(v):return np.round(np.clip(v,0,1)*255).astype(np.uint8)
def black(rt,d,asset,c):return raster(c,rt.pipe,0,tagged(asset.rows(d['pose'],c,rt.pipe,1,precomputed=True),'asset'),rt.n.bg.new_zeros(3))


def hr_env(rt,c):
    y,x=torch.meshgrid(torch.arange(c.image_height,device='cuda',dtype=torch.float32),torch.arange(c.image_width,device='cuda',dtype=torch.float32),indexing='ij')
    c.grid=torch.stack((x,y),-1)
    directions=c.get_world_directions(False).permute(1,2,0)
    return rt.n.env(directions).permute(2,0,1)


def strip(images,labels,height=400):
    panels=[]
    for value,label in zip(images,labels):
        tile=Image.fromarray(rgb8(value));tile=tile.resize((round(tile.width*height/tile.height),height),Image.Resampling.LANCZOS)
        p=Image.new('RGB',(tile.width,height+30),(237,243,247));p.paste(tile,(0,30));ImageDraw.Draw(p).text((8,8),label,fill=(12,25,38));panels.append(p)
    out=Image.new('RGB',(sum(p.width for p in panels),height+30));left=0
    for p in panels:out.paste(p,(left,0));left+=p.width
    return out


@torch.no_grad()
def main():
    dest=ROOT/'evaluation';dest.mkdir(exist_ok=False);start=time.time();rt=Runtime()
    paths=dict(sphere_strong_v2=RUN/'output_adapter/sphere_strong_v2/complete_a6.npz',
               projective_surface=ROOT/'projective_surface_asset.npz')
    assets={name:SHAsset.from_geometry(path).cuda() for name,path in paths.items()}
    records=[];renders=[];maxconservation=0.
    for raw in range(89,138):
        d=rt.raw(raw);masks=fixed_masks(d);c=camera_hr(d['cam'])
        target=np.asarray(Image.open(RGBROOT/'image/image_0'/f'0008{raw:03d}.png').convert('RGB')).astype(np.float32)/255
        low=binary_erosion(d['a6_visible'].cpu().numpy(),iterations=1)
        hr_mask=np.repeat(np.repeat(low,4,0),4,1)
        record=dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',HR_pixels=int(hr_mask.sum()),methods={})
        visuals={}
        for name,asset in assets.items():
            lowout=black(rt,d,asset,d['cam']);a=lowout['alpha'][0];depth=expected_depth(lowout,rt.scale)
            out=black(rt,d,asset,c);rgb=cpu(out['rgb']).transpose(1,2,0);alpha=cpu(out['alpha'])[0]
            error=np.abs(rgb-target).mean(2);norm=np.abs(rgb/np.maximum(alpha[...,None],1e-5)-target).mean(2)
            record['methods'][name]=dict(
                HR_RGB_error=metric(error,hr_mask),
                HR_normalized_RGB_error=metric(norm,hr_mask),
                HR_alpha_fixed_visible=metric(alpha,hr_mask),
                LiDAR_depth_error=metric(np.abs(depth-d['obs']['depth_m'][0]),masks['a6_depth']),
                LiDAR_alpha_coverage=metric(a,masks['a6_depth']),
                visible_alpha=metric(a,masks['a6_visible']),reviewed_alpha=metric(a,masks['a6_reviewed']),
                reviewed_alpha_below05=metric((a<.5).float(),masks['a6_reviewed']))
            if raw in (106,113,130):
                per=dest/f'raw{raw:03d}'/name;per.mkdir(parents=True)
                Image.fromarray(rgb8(rgb)).save(per/'rgb_black_hr.png')
                Image.fromarray(np.round(np.clip(alpha,0,1)*65535).astype(np.uint16)).save(per/'alpha16_hr.png')
                np.savez_compressed(per/'black_raster_hr.npz',alpha=alpha,expected_depth_m=expected_depth(out,rt.scale),
                    rgb_premultiplied=rgb,geometry_hash=np.asarray(asset.geometry_hash),model_from_local=d['pose'].cpu().numpy())
                visuals[name]=(rgb,alpha)
            del out,lowout
        if raw in (106,113,130):
            y,x=np.where((visuals['sphere_strong_v2'][1]>.01)|(visuals['projective_surface'][1]>.01))
            crop=(max(0,int(x.min())-24),max(0,int(y.min())-24),min(1920,int(x.max())+25),min(1280,int(y.max())+25))
            x0,y0,x1,y1=crop
            observed=np.zeros_like(target);vis=np.repeat(np.repeat(d['a6_visible'].cpu().numpy(),4,0),4,1);observed[vis]=target[vis]
            labels=['Observed visible only','sphere strongv2','view-conditioned surface resampling']
            strip([observed[y0:y1,x0:x1]]+[visuals[n][0][y0:y1,x0:x1] for n in assets],labels).save(dest/f'{raw:03d}_objects.png')
            environment=hr_env(rt,c);streets=[];scene_receipts=[]
            # BOTH states use one complete persistent local asset and the
            # original native joint depth sort with the actual a37/static/road.
            for delete in (False,True):
                scene=rt.scene(raw,assets['projective_surface'],delete=delete)
                assert scene['meta']['native_a6_rows_input']==0
                if delete:assert scene['meta']['native_a37_packed_rows']==0
                out=raster(c,rt.pipe,rt.n.static.active_sh_degree,scene['rows'],scene['bg'])
                rgb=out['rgb']+(1-out['alpha'])*environment
                groups=out['extra'][:4]
                err=float((groups.sum(0)-out['alpha'][0]).abs().max());maxconservation=max(maxconservation,err)
                assert err<2e-5
                label='delete a37' if delete else 'a37 present'
                per=dest/f'raw{raw:03d}'/'projective_surface'
                Image.fromarray(rgb8(cpu(rgb).transpose(1,2,0))).save(per/('delete_hr.png' if delete else 'keep_hr.png'))
                np.savez_compressed(per/('delete_native_groups.npz' if delete else 'keep_native_groups.npz'),
                    component_alpha_T=cpu(groups).astype(np.float16),alpha=cpu(out['alpha']).astype(np.float16))
                streets.append(cpu(rgb).transpose(1,2,0));scene_receipts.append(dict(state=label,**scene['meta'],group_conservation_maxabs=err))
                del scene,out
            strip(streets,['persistent3D chart / a37 present','persistent3D chart / delete a37'],height=480).save(dest/f'{raw:03d}_street.png')
            renders.append(dict(raw=raw,crop_xyxy=list(crop),states=scene_receipts))
            del environment,streets,visuals
        records.append(record)
        write(dest/f'{raw:03d}_metrics.json',record)
        print(json.dumps(dict(stage='projective_eval',raw=raw,complete=len(records))),flush=True)
        rt._data.clear();torch.cuda.empty_cache()
    keys=('HR_RGB_error','HR_normalized_RGB_error','HR_alpha_fixed_visible','LiDAR_depth_error','LiDAR_alpha_coverage','visible_alpha','reviewed_alpha','reviewed_alpha_below05')
    pooled={split:{name:{key:pool(r['methods'][name][key] for r in rows) for key in keys} for name in assets}
        for split,rows in [('all49',records),('BUILD',[r for r in records if r['split']=='BUILD']),('HELD',[r for r in records if r['split']=='HELD'])]}
    assert rt.modelunchanged() and all(a.assert_geometry_unchanged() for a in assets.values())
    write(dest/'metrics49.json',dict(status='PASS',rows=records,pooled=pooled,all49_complete=True,
        HR_RGB_domain='Same original1920 RGB on fixed old eroded visible interiors nearest4; no predicted alpha selection',
        LiDAR_domain='Original480 depth_valid[0] fixed across methods, expected camera-z in metres',
        reference_view_raw106_was_used_for_chart_coloring=True,HELD_RGB_used_for_fitting=False,
        geometry_context_not_independently_HELD=True,view_conditioned_upper_bound=True))
    summary={split:{name:{key:pooled[split][name][key]['mean'] for key in keys} for name in assets} for split in pooled}
    write(dest/'receipt.json',dict(status='PASS',representation='view-conditioned surface resampling',summary=summary,
        three_views=renders,native_group_conservation_maxabs=maxconservation,actual_native_joint_sort=True,
        PVD_weight_updates=0,colour_fit_updates=0,image_overlay_used=False,
        assets={n:dict(path=str(p),sha256=sha(p),geometry_hash=assets[n].geometry_hash,rows=len(assets[n].xyz)) for n,p in paths.items()},
        limitation='Reference-camera surface chart is persistent3D but is not full360 reconstruction; depth comes from PVD proposal not truth',
        seconds=time.time()-start))
    print(json.dumps(dict(stage='projective_eval_complete',status='PASS',summary=summary,seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
