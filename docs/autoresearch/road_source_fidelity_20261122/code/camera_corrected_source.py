"""Brown原照片与针孔渲染统一相机模型；复用已测固定算子，不改源几何。"""
import argparse,time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,binary_erosion
from scipy.sparse import load_npz,vstack,eye
from scipy.optimize import lsq_linear
import probe_source_chain as p
h=p.h;torch=p.torch;ROOT=p.ROOT

def brown_grid(raw):
 d=h.read(h.BANK/f'observations/{raw:03d}.json')
 fx,fy,cx,cy,k1,k2,p1,p2,k3=d['camera_intrinsic_Brown']
 yy,xx=np.mgrid[:1280,:1920].astype(np.float64)
 x=(xx+.5-cx)/fx;y=(yy+.5-cy)/fy;r=x*x+y*y
 radial=1+k1*r+k2*r*r+k3*r*r*r
 dx=x*radial+2*p1*x*y+p2*(r+2*x*x)
 dy=y*radial+p1*(r+2*y*y)+2*p2*x*y
 forward=np.stack((fy*dy+cy-.5,fx*dx+cx-.5))
 # 逆向Brown点用于将针孔图重采样回原始照片域；固定点在本畸变范围内收敛。
 ux=x.copy();uy=y.copy()
 for _ in range(12):
  rr=ux*ux+uy*uy;rad=1+k1*rr+k2*rr*rr+k3*rr*rr*rr
  tx=2*p1*ux*uy+p2*(rr+2*ux*ux);ty=p1*(rr+2*uy*uy)+2*p2*ux*uy
  ux=(x-tx)/rad;uy=(y-ty)/rad
 inverse=np.stack((fy*uy+cy-.5,fx*ux+cx-.5))
 rr=ux*ux+uy*uy;rad=1+k1*rr+k2*rr*rr+k3*rr*rr*rr
 bx=fx*(ux*rad+2*p1*ux*uy+p2*(rr+2*ux*ux))+cx-.5
 by=fy*(uy*rad+p1*(rr+2*uy*uy)+2*p2*ux*uy)+cy-.5
 return forward,inverse,float(np.hypot(bx-xx,by-yy).max())

def warp(a,grid,order=1):
 if a.ndim==2:return map_coordinates(a,grid,order=order,mode='nearest',prefilter=False)
 return np.stack([map_coordinates(a[:,:,c],grid,order=order,mode='nearest',prefilter=False) for c in range(a.shape[2])],2)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--patch',required=True);ap.add_argument('--reference',type=int,required=True);ap.add_argument('--prior-run',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 out=ROOT/args.output;previous=ROOT/args.prior_run
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 plan=h.read(ROOT/'probe_plan.json');records={int(v['raw']):v for v in plan['records'] if v['candidate_id']==args.patch};frames=sorted(records)
 entry=next(v for v in h.read(h.PRIOR/'source_selection/manifest.json')['candidates'] if v['candidate_id']==args.patch)
 source=h.load(entry['packet']);core=source['core_mask'];ids=source['world_row_ids'];n=int(core.sum())
 prior_source=h.load(previous/args.patch/'source_fixed_RGB.npz')
 state=h.load(previous/args.patch/'linear_state.npz');A=load_npz(previous/args.patch/'native_color_operator.npz')
 ctx=h.context(frames);torch.set_num_threads(2);base=ctx.base_rgb.detach().clone();positions=np.searchsorted(ctx.active_world_rows,ids)
 assert np.array_equal(ctx.active_world_rows[positions],ids)
 corepos=torch.tensor(positions[core],device='cuda');allpos=torch.tensor(positions,device='cuda');F=0;data={}
 h.write(out/'contract.json',dict(task='ROAD-SOURCE-FIDELITY-20261122',stage=args.output,patch=args.patch,reference=args.reference,
  correction='Raw Waymo Brown image deterministically sampled into native pinhole pixel centers; photo mask nearest, RGB bilinear',
  original_images_unchanged=True,geometry_scale_opacity_frozen=True,halo_RGB_frozen=True,
  solver='Same fixed native color operator and bounded LS settings as prior run; prior=1e-5, no new basis measurement',
  extra_validation='Warp native renders back into raw Brown camera and compare against unresampled original photograph',
  RGB_is_encoded_sRGB=True,new_diffusion_calls=0,target_migration=False,human_verdict=None,production_promoted=False))
 def render(raw,colors):
  nonlocal F
  packet=ctx.base_packet(raw);packet['mask']=packet['mask']&torch.tensor(np.isin(ctx.active_world_rows,ids),device='cuda');packet['colors_precomp']=colors
  ctx.require_lease();ctx.native_calls+=1;F+=1
  r=ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,ctx.scene._background)
  alpha=r['alpha'].squeeze().cpu().numpy();rgb=r['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(alpha[:,:,None],1e-8)
  return rgb,alpha
 try:
  for raw in frames:
   photo,project,path,K,T=p.projection(raw);grid,inverse,roundtrip=brown_grid(raw)
   old=h.load(previous/args.patch/f'{raw:03d}/source_float.npz');x0,y0,x1,y1=old['bbox_xyxy']
   legal_raw=np.asarray(Image.open(records[raw]['core_photo_valid_mask_full1920_remote_stage']))>0
   legal_pin=warp(legal_raw.astype(np.float32),grid,0)>.5
   oldmask=np.zeros((1280,1920),bool);oldmask[y0:y1,x0:x1]=old['score_mask']
   mask=oldmask&legal_pin
   data[raw]=dict(photo_raw=photo,photo=warp(photo,grid),grid=grid,inverse=inverse,roundtrip=roundtrip,
    mask=mask,oldmask=oldmask,legal_raw=legal_raw,project=project,bbox=(x0,y0,x1,y1),old=old)
  d=data[args.reference];selection=d['mask'][state['score_mask']]
  Af=A[selection];fixed=state['fixed_core_image'][selection];target=d['photo'][d['mask']]-fixed
  uv,depth=d['project'](source['world_xyz'])
  x0rgb=np.stack([map_coordinates(d['photo'][:,:,c],[uv[:,1],uv[:,0]],order=1,mode='nearest') for c in range(3)],1)[core]
  m=int(selection.sum());weight=1e-5*m/n
  AA=vstack([Af,np.sqrt(weight)*eye(n)],format='csr');channels=[];solvers=[]
  for ch in range(3):
   result=lsq_linear(AA,np.r_[target[:,ch],np.sqrt(weight)*x0rgb[:,ch]],bounds=(0,1),method='trf',lsq_solver='lsmr',tol=1e-7,lsmr_tol=1e-8,max_iter=100,lsmr_maxiter=2000)
   channels.append(result.x);solvers.append(dict(channel=ch,status=int(result.status),success=bool(result.success),iterations=int(result.nit),cost=float(result.cost),optimality=float(result.optimality)))
   print(solvers[-1],flush=True)
  fitted=np.stack(channels,1).astype(np.float32);repaired={k:v.copy() for k,v in source.items()};repaired['reference_srgb'][core]=fitted
  h.write(out/'solver_receipt.json',dict(channels=solvers))
  np.savez_compressed(out/'source_camera_corrected_RGB.npz',**repaired)
  reloaded=h.load(out/'source_camera_corrected_RGB.npz')
  assert all(np.array_equal(v,reloaded[k]) for k,v in source.items() if k!='reference_srgb')
  np.savez_compressed(out/'linear_state.npz',fitted_core_RGB=fitted,direct_core_RGB=x0rgb,target_minus_fixed=target,
   fixed_core_image=fixed,score_mask=d['mask'],reused_operator_row_selection=selection)
  colors={'original':base,'uncorrected_solve':base.index_copy(0,allpos,torch.tensor(prior_source['reference_srgb'],device='cuda')),
   'camera_corrected_solve':base.index_copy(0,allpos,torch.tensor(reloaded['reference_srgb'],device='cuda'))}
  evaluations=[]
  with torch.no_grad():
   for raw,v in data.items():
    x0,y0,x1,y1=v['bbox'];crop=np.s_[y0:y1,x0:x1];folder=out/f'{raw:03d}';folder.mkdir();metrics={};raw_metrics={};arrays={}
    for name,col in colors.items():
     rgb,alpha=render(raw,col);raw_rgb=warp(rgb,v['inverse']);raw_alpha=warp(alpha,v['inverse'])
     raw_mask=warp(v['mask'].astype(np.float32),v['inverse'],0)>.5
     raw_mask &= v['legal_raw']&(raw_alpha>=.95)
     for field,target_image,image,mask,collection in [('pinhole',v['photo'],rgb,v['mask'],metrics),('raw',v['photo_raw'],raw_rgb,raw_mask,raw_metrics)]:
      collection[name]=dict(score_n=int(mask.sum()),RGB_MAE=float(abs(target_image-image)[mask].mean()),spectral=p.spectral(target_image,image,mask))
      h.image(folder/(name+'_'+field+'.png'),image[crop]);arrays[name+'_'+field]=image[crop]
     arrays['alpha']=alpha[crop];arrays['raw_mask']=raw_mask[crop]
    h.image(folder/'observation_raw.png',v['photo_raw'][crop]);h.image(folder/'observation_pinhole.png',v['photo'][crop])
    Image.fromarray(np.uint8(v['mask'][crop])*255).save(folder/'score_mask.png')
    np.savez_compressed(folder/'camera_float.npz',**arrays,observation_raw=v['photo_raw'][crop],observation_pinhole=v['photo'][crop],score_mask=v['mask'][crop],bbox_xyxy=np.asarray(v['bbox']))
    evaluations.append(dict(raw=raw,role='REFERENCE_FIT' if raw==args.reference else 'QUERY_not_fit',metrics=metrics,raw_camera_metrics=raw_metrics,
     Brown_inverse_roundtrip_max_px=v['roundtrip'],output=str(folder),bbox_xyxy=[int(q) for q in v['bbox']] ))
    h.write(out/'evaluation_progress.json',dict(evaluations=evaluations,nativeF=F,nativeB=0))
  h.write(out/'receipt.json',dict(complete=True,patch=args.patch,reference=args.reference,solver=solvers,evaluations=evaluations,
   nativeF=F,nativeB=0,RGB_linear_solves=3,gradient_optimizer_updates=0,source_non_RGB_fields_exact=True,halo_RGB_exact=True,
   new_diffusion_calls=0,target_migration=False,original_asset_modified=False))
 finally:h.native.close()

if __name__=='__main__':main()
