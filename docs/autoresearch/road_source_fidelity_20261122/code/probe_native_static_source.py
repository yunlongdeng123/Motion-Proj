"""手工核验同位置原生静态高斯；固定参考时刻几何，不改任何父资产。"""
import argparse,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import probe_source_chain as p
h=p.h;torch=p.torch;ROOT=p.ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--patch',default='patch_02');ap.add_argument('--reference',type=int,default=115)
    ap.add_argument('--output',default='native_static_02_115');a=ap.parse_args()
    out=ROOT/a.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    plan=h.read(ROOT/'probe_plan.json');records={int(v['raw']):v for v in plan['records'] if v['candidate_id']==a.patch}
    frames=sorted(records);ctx=h.context(frames);torch.set_num_threads(2)
    entry=next(v for v in h.read(h.PRIOR/'source_selection/manifest.json')['candidates'] if v['candidate_id']==a.patch)
    source=h.load(entry['packet']);frame=source['source_tangent_frame_world'];center=source['core_center_tangent'];half=source['core_size_m']/2
    world_from_model=np.linalg.inv(source['model_from_world']);F=0;results=[]
    h.write(out/'contract.json',dict(task='ROAD-SOURCE-FIDELITY-20261122',patch=a.patch,reference=a.reference,
        source='Original static model below road handoff, before current retire gates',
        selection='Manual same physical source patch; centers in core plus .15m halo, reference-chart height within .10m; no RGB-based filtering',
        parameters='Original native geometry, opacity and SH captured at reference time then frozen; SH evaluated per view',
        native_model_geometry_modified=False,target_migration=False,optimizer_updates=0,new_diffusion_calls=0))
    def prepare(raw):
        return ctx.scene._bindings().prepare_rasterization(ctx.camera(raw),ctx.scene.static,ctx.scene.pipe,dynamic=False)
    def rgb(raw,packet):
        nonlocal F
        ctx.require_lease();ctx.native_calls+=1;F+=1
        native=ctx.scene._native_colors(packet,ctx.scene.static.active_sh_degree,ctx.camera(raw))
        result=ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,native,ctx.scene._background)
        alpha=result['alpha'].squeeze().cpu().numpy()
        return result['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(alpha[:,:,None],1e-8),alpha
    try:
      with torch.no_grad():
        reference=prepare(a.reference)
        xyz=reference['means3D'].cpu().numpy()
        world=xyz@world_from_model[:3,:3].T+world_from_model[:3,3]
        local=(world-frame[:3,3])@frame[:3,:3]
        select=(np.abs(local[:,:2]-center)<half+.15).all(1)&(np.abs(local[:,2])<.10)
        # 原生mask对应参考时刻的实际活动状态；未知owner不改写为道路真值。
        select &= reference['mask'].cpu().numpy()
        source_ids=np.flatnonzero(select)
        fixed={}
        for k,v in reference.items():
            if torch.is_tensor(v) and v.ndim and len(v)==len(xyz):fixed[k]=v[select].clone()
            else:fixed[k]=v
        arrays={k:v.cpu().numpy() for k,v in fixed.items() if torch.is_tensor(v)}
        arrays.update(native_static_row_ids=source_ids,world_xyz=world[select],reference_raw=np.asarray(a.reference))
        np.savez_compressed(out/'frozen_native_source.npz',**arrays)
        for raw in frames:
            photo,project,photo_path,K,T=p.projection(raw);uv,depth=project(entry['world_roi_corners'])
            canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255)
            core=np.asarray(canvas)>0
            legal=np.asarray(Image.open(records[raw]['core_photo_valid_mask_full1920_remote_stage']))>0
            lo=np.maximum(0,np.floor(uv.min(0)-12).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+13).astype(int))
            x0,y0=lo;x1,y1=hi;crop=np.s_[y0:y1,x0:x1]
            live=prepare(raw);live_rgb,live_alpha=rgb(raw,live)
            frozen={k:v.clone() if torch.is_tensor(v) else v for k,v in fixed.items()}
            frozen_rgb,frozen_alpha=rgb(raw,frozen)
            # 评分域取两者都有高覆盖的同一照片合法核心，并另报覆盖分母。
            base=binary_erosion(core,iterations=5)&legal
            score=base&(live_alpha>=.95)&(frozen_alpha>=.95)
            folder=out/f'{raw:03d}';folder.mkdir()
            for key,img in [('observation',photo),('full_native_static',live_rgb),('frozen_source',frozen_rgb),('frozen_alpha',frozen_alpha)]:h.image(folder/(key+'.png'),img[crop])
            Image.fromarray(np.uint8(score[crop])*255).save(folder/'score_mask.png')
            np.savez_compressed(folder/'native_static_float.npz',observation=photo[crop],live_rgb=live_rgb[crop],
                frozen_rgb=frozen_rgb[crop],live_alpha=live_alpha[crop],frozen_alpha=frozen_alpha[crop],
                score_mask=score[crop],base_mask=base[crop],core=core[crop],bbox_xyxy=np.r_[lo,hi])
            metrics={}
            if score.sum():
              for key,img in [('full_native_static',live_rgb),('frozen_source',frozen_rgb)]:
                metrics[key]=dict(RGB_MAE=float(np.abs(img-photo)[score].mean()),spectral=p.spectral(photo,img,score))
            results.append(dict(raw=raw,source_rows=len(source_ids),base_n=int(base.sum()),score_n=int(score.sum()),
                live_alpha_ge95=int((base&(live_alpha>=.95)).sum()),frozen_alpha_ge95=int((base&(frozen_alpha>=.95)).sum()),
                metrics=metrics,output=str(folder)))
            print(results[-1],flush=True)
        h.write(out/'receipt.json',dict(complete=True,rows=results,nativeF=F,nativeB=0,optimizer_updates=0,
            source_rows=len(source_ids),new_diffusion_calls=0,target_migration=False,original_asset_modified=False))
    finally:h.native.close()

if __name__=='__main__':main()
