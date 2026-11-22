"""按实际像素贡献而非中心高度选原生源高斯；仅梯度测量，无参数更新。"""
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import probe_source_chain as p
h=p.h;torch=p.torch;ROOT=p.ROOT

def main():
 out=ROOT/'native_static_contributors_02_115'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);patch='patch_02';ref=115
 plan=h.read(ROOT/'probe_plan.json');records={int(v['raw']):v for v in plan['records'] if v['candidate_id']==patch}
 frames=sorted(records);ctx=h.context(frames);torch.set_num_threads(2);F=B=0
 entry=next(v for v in h.read(h.PRIOR/'source_selection/manifest.json')['candidates'] if v['candidate_id']==patch)
 source=h.load(entry['packet']);frame=source['source_tangent_frame_world'];world_from_model=np.linalg.inv(source['model_from_world'])
 h.write(out/'contract.json',dict(task='ROAD-SOURCE-FIDELITY-20261122',patch=patch,reference=ref,
  source='Original native static model below current road handoff',
  selection='Actual RGB VJP summed over qualified reference core; smallest contributor prefix retaining 99.95% measured alpha*T mass',
  geometry='Capture prepared reference-time means/covariance/opacity/SH and freeze; no spatial truncation',
  gradient_measurements=1,optimizer_updates=0,new_diffusion_calls=0,target_migration=False))
 def preparation(raw):return ctx.scene._bindings().prepare_rasterization(ctx.camera(raw),ctx.scene.static,ctx.scene.pipe,dynamic=False)
 def render(raw,packet,measure=False):
  nonlocal F
  ctx.require_lease();ctx.native_calls+=1;F+=1
  return ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,ctx.scene._background)
 def masks(raw):
  photo,project,path,K,T=p.projection(raw);uv,depth=project(entry['world_roi_corners'])
  canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255)
  core=np.asarray(canvas)>0;legal=np.asarray(Image.open(records[raw]['core_photo_valid_mask_full1920_remote_stage']))>0
  lo=np.maximum(0,np.floor(uv.min(0)-12).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+13).astype(int))
  return photo,core,binary_erosion(core,iterations=5)&legal,lo,hi
 try:
  photo,core,base,lo,hi=masks(ref)
  with torch.no_grad():prepared=preparation(ref);native=ctx.scene._native_colors(prepared,ctx.scene.static.active_sh_degree,ctx.camera(ref))
  colors=native['colors_precomp'].detach().clone().requires_grad_(True);probe=dict(native,colors_precomp=colors)
  output=render(ref,probe,True)
  score=base&(output['alpha'].detach().squeeze().cpu().numpy()>=.95)
  loss=output['rgb'][:,torch.tensor(score,device='cuda')].sum()/3
  loss.backward();B+=1
  mass=colors.grad.detach().mean(1).cpu().numpy();order=np.argsort(-mass)
  count=int(np.searchsorted(np.cumsum(mass[order]),mass.sum()*.9995)+1)
  ids=np.sort(order[:count]);selected_mass=float(mass[ids].sum()/mass.sum())
  with torch.no_grad():
   fixed={k:(v[ids].clone() if torch.is_tensor(v) and v.ndim and len(v)==len(mass) else v) for k,v in prepared.items()}
   xyz=fixed['means3D'].cpu().numpy();world=xyz@world_from_model[:3,:3].T+world_from_model[:3,3]
   local=(world-frame[:3,3])@frame[:3,:3]
   fields={k:v.cpu().numpy() for k,v in fixed.items() if torch.is_tensor(v)}
   fields.update(native_static_row_ids=ids,world_xyz=world,reference_mass=mass[ids])
   np.savez_compressed(out/'frozen_contributor_source.npz',**fields)
   scales=fixed.get('scales');sigma=None if scales is None else scales.cpu().numpy()*np.linalg.norm(world_from_model[:3,:3],axis=0)
   geometry=dict(source_rows=len(ids),retained_reference_mass=selected_mass,
    row_center_height_abs_quantiles_m=np.quantile(abs(local[:,2]),[.5,.9,.95,1]).tolist(),
    contribution_mass_within_10cm_plane=float(mass[ids][abs(local[:,2])<.10].sum()/mass[ids].sum()),
    sigma_max_quantiles_m=None if sigma is None else np.quantile(sigma.max(1),[.5,.9,.95,1]).tolist())
   results=[]
   for raw in frames:
    photo,core,base,lo,hi=masks(raw)
    live=ctx.scene._native_colors(preparation(raw),ctx.scene.static.active_sh_degree,ctx.camera(raw))
    live_out=render(raw,live)
    packet={k:v.clone() if torch.is_tensor(v) else v for k,v in fixed.items()}
    packet=ctx.scene._native_colors(packet,ctx.scene.static.active_sh_degree,ctx.camera(raw))
    frozen_out=render(raw,packet)
    la=live_out['alpha'].squeeze().cpu().numpy();fa=frozen_out['alpha'].squeeze().cpu().numpy()
    lr=live_out['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(la[:,:,None],1e-8)
    fr=frozen_out['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(fa[:,:,None],1e-8)
    score=base&(la>=.95)&(fa>=.95)
    x0,y0=lo;x1,y1=hi;crop=np.s_[y0:y1,x0:x1];folder=out/f'{raw:03d}';folder.mkdir()
    for key,img in [('observation',photo),('full_native_static',lr),('frozen_source',fr),('frozen_alpha',fa)]:h.image(folder/(key+'.png'),img[crop])
    Image.fromarray(np.uint8(score[crop])*255).save(folder/'score_mask.png')
    np.savez_compressed(folder/'native_static_float.npz',observation=photo[crop],live_rgb=lr[crop],frozen_rgb=fr[crop],
     live_alpha=la[crop],frozen_alpha=fa[crop],score_mask=score[crop],base_mask=base[crop],core=core[crop],bbox_xyxy=np.r_[lo,hi])
    metrics={}
    if score.any():
     for k,img in [('full_native_static',lr),('frozen_source',fr)]:metrics[k]=dict(RGB_MAE=float(abs(img-photo)[score].mean()),spectral=p.spectral(photo,img,score))
    results.append(dict(raw=raw,base_n=int(base.sum()),score_n=int(score.sum()),frozen_alpha_ge95=int((base&(fa>=.95)).sum()),metrics=metrics,output=str(folder)))
    print(results[-1],flush=True)
   h.write(out/'receipt.json',dict(complete=True,geometry=geometry,rows=results,nativeF=F,nativeB=B,optimizer_updates=0,new_diffusion_calls=0,
    original_asset_modified=False,target_migration=False))
 finally:h.native.close()

if __name__=='__main__':main()
