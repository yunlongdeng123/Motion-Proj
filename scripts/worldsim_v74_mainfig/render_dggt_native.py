"""DGGT official mode2 Gaussian equations; predicted semantic sky mask replaces unavailable GT mask explicitly."""
import torch,numpy as np,json,time
from PIL import Image
def render(model,pred,images,out):
 from dggt.utils.gs import get_split_gs,concat_list
 from dggt.utils.geometry import unproject_depth_map_to_point_map
 from dggt.utils.pose_enc import pose_encoding_to_extri_intri
 from gsplat.rendering import rasterization
 t=time.time();H,W=images.shape[-2:];N=pred['depth'].shape[1];gs=pred['gs_map'].float();dy=pred['dynamic_conf'].float().squeeze(-1)
 E,K=pose_encoding_to_extri_intri(pred['pose_enc'],(H,W));E=E[0].float();K=K[0].float()
 point=torch.from_numpy(unproject_depth_map_to_point_map(pred['depth'][0].float().cpu(),E.cpu(),K.cpu())).float().to('cuda')[None]
 bg=pred['semantic_logits'].argmax(-1)!=9;static=(dy<.5)&bg
 ts=torch.tensor([i//6*.5 for i in range(N)],device='cuda',dtype=torch.float32)
 stime=ts[torch.where(static)[1]];sp=point[static];sr,so,ss,sq=get_split_gs(gs,static);so=so*(1-dy[static].sigmoid());gamma=pred['gs_conf'].float().squeeze(-1)[static]
 dest=out/'native_render';dest.mkdir(exist_ok=True);records=[]
 with torch.inference_mode():
  # Official sky model is retained; this is an input-view reconstruction check, not heldout rendering.
  E4=torch.eye(4,device='cuda')[None].repeat(N,1,1);E4[:,:3,:]=E
  sky=model.sky_model(images if images.ndim==5 else images[None],E4,K);sky=(sky-sky.min())/(sky.max()-sky.min()+1e-8)
  for i in range(N):
   mask=bg[:,i];dr,do,ds,dq=get_split_gs(gs[:,i],mask);do=do*dy[:,i][mask].sigmoid()
   timed=so*torch.exp(torch.log(torch.tensor(.1,device='cuda'))/(gamma.square()+1e-6)*(stime-ts[i]).square())
   means,colors,opacity,scales,quats=concat_list([sp,sr,timed,ss,sq],[point[:,i][mask],dr,do,ds,dq])
   T=torch.eye(4,device='cuda')[None];T[:,:3,:]=E[i:i+1]
   img,alpha,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacity,colors=colors,viewmats=T,Ks=K[i:i+1],width=W,height=H,render_mode='RGB+ED')
   rgb=alpha*img[...,:3]+(1-alpha)*sky[i:i+1]
   Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8')).save(dest/f'{i:02d}_rgb.png')
   np.savez_compressed(dest/f'{i:02d}_depth_alpha.npz',expected_depth=img[0,...,3].cpu().numpy(),alpha=alpha[0,...,0].cpu().numpy())
   records.append({'view':i,'gaussians':len(means),'finite_fraction':float(torch.isfinite(img).float().mean()),'alpha_mean':float(alpha.mean())})
 (dest/'result.json').write_text(json.dumps({'status':'DONE','renderer':'official gsplat RGB+ED mode2 equations plus official sky background','mask':'predicted semantic argmax non-sky, instead of official dataset GT sky mask','timestamps':'sample3=0, sample4=0.5s; simultaneous six camera approximation','evaluation':'input-view RGB reconstruction only; ED is opacity-weighted expected camera depth, not LiDAR first return; no simulator or planning claims','views':records,'elapsed_s':time.time()-t},indent=2));print('RENDER_DONE',out,flush=True)
