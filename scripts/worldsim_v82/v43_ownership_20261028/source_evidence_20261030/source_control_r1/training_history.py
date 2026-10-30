"""Fixed postdelete source IDs traced through all original observed native times."""
from pathlib import Path
import json,numpy as np,torch
from PIL import Image,ImageDraw

def diagnose(static,dynamic,camera,pipe,degree,helpers,environment,transform,reference_root,frame,output):
 from utils.sh_utils import eval_sh
 from evaluate_edits import save_rgb
 root=Path(reference_root);experiment=root.parent.parent
 fixed=json.loads((experiment/"source_control_r1/run_r2_color_maps/source_000008/source_receipt.json").read_text())["groups"]["top_error_128"]
 ids=torch.tensor(fixed,device="cuda",dtype=torch.long);n=len(static["means3D"])
 geometry=np.load(root/"original"/(frame+".npz"));sem=torch.tensor(geometry["semantic_id"].astype(np.int32),device="cuda")
 masks=[sem==10,(sem==1)|(sem==2),~((sem==10)|(sem==1)|(sem==2))]
 legal_rgb=camera.original_image[:3].detach().cuda()
 rows=helpers.merge_rows(static,dynamic)
 with torch.enable_grad():
  features=rows["features"].detach().clone().requires_grad_(True);values=dict(rows,features=features)
  package=helpers.raster(camera,pipe,degree,values,torch.zeros(3,device="cuda"))
  def g(field):return torch.autograd.grad((package["alpha"][0]*field).sum(),features,retain_graph=True)[0][:,1].detach()
  weights=[];checks=[]
  for mask in masks:
   w=g(mask);total=float((package["alpha"][0]*mask).sum());err=abs(float(w.sum())-total)/max(1,total)
   if err>2e-5:raise RuntimeError("category contribution conservation"+str(err))
   checks.append(dict(alpha_sum=total,gradient_sum=float(w.sum()),relative_error=err));weights.append(w[ids])
  mass=torch.stack(weights,1)
  total=g(torch.ones_like(sem));rgbmoment=torch.stack([g(legal_rgb[c])[ids] for c in range(3)],1)
  rgb2=g(legal_rgb.square().mean(0))[ids]
 dirs=static["means3D"][ids]-camera.camera_center;dirs=dirs/dirs.norm(dim=1,keepdim=True)
 color=(eval_sh(degree,static["shs"][ids].transpose(1,2),dirs)+.5).clamp_min(0)
 w=total[ids];safe=w.clamp_min(1e-12)
 fit_rms=(color.square().mean(1)-2*(color*rgbmoment).mean(1)/safe+rgb2/safe).clamp_min(0).sqrt()
 model_cam=torch.cat((static["means3D"][ids],torch.ones((len(ids),1),device="cuda")),1)@camera.world_view_transform
 scale=float(np.linalg.norm(transform[:3,0]));z=model_cam[:,2]/scale
 u=camera.fx*model_cam[:,0]/model_cam[:,2]+camera.cx;v=camera.fy*model_cam[:,1]/model_cam[:,2]+camera.cy
 uu=u.floor().long();vv=v.floor().long();h,wimg=sem.shape
 inside=(model_cam[:,2]>0)&(uu>=0)&(uu<wimg)&(vv>=0)&(vv<h)
 centers=[]
 for j,rid in enumerate(fixed):
  valid=bool(inside[j]);x=int(uu[j]);y=int(vv[j])
  centers.append(dict(row_id=rid,u=float(u[j]),v=float(v[j]),in_image=valid,center_camera_z_world_m=float(z[j]),center_pixel_firsthit_z_m=float(geometry["depth_z_m"][y,x]) if valid else None,center_pixel_semantic=int(geometry["semantic_id"][y,x]) if valid else None,center_pixel_surface=int(geometry["surface_id"][y,x]) if valid else None,warning="center-pixel firsthit only; not Gaussian surface or identity"))
 folder=output/"history";folder.mkdir(exist_ok=True)
 np.savez_compressed(folder/(frame+".npz"),static_index=np.array(fixed),alphaT_by_original_category=mass.cpu().numpy(),total_alphaT=total[ids].cpu().numpy(),native_view_rgb=color.cpu().numpy(),direction=dirs.cpu().numpy(),observed_rgb_moment=rgbmoment.cpu().numpy(),observed_rgb2_moment=rgb2.cpu().numpy(),observed_rgb_fit_rms=fit_rms.cpu().numpy(),center_z_world_m=z.cpu().numpy(),u=u.cpu().numpy(),v=v.cpu().numpy())
 if frame in ("000000","000004","000008","000012","000015"):
  im=Image.fromarray(np.round(np.clip(legal_rgb.cpu().numpy().transpose(1,2,0),0,1)*255).astype("uint8"));draw=ImageDraw.Draw(im)
  palette=["#ffff00","#00ffff","#ff00ff","#00ff00","#ff8800","#ffffff","#ff0000","#0088ff"]
  for j,item in enumerate(centers[:8]):
   if not item["in_image"]:continue
   x,y=item["u"],item["v"];draw.ellipse((x-3,y-3,x+3,y+3),outline=palette[j],width=2)
   draw.text((x+4,y-8-j%2*10),str(item["row_id"]),fill=palette[j])
  im.save(folder/(frame+"_top8_center_projection.png"))
  semantic=geometry["semantic_id"];rgb=np.zeros(semantic.shape+(3,),dtype=np.uint8);rgb[semantic==10]=[220,60,60];rgb[np.isin(semantic,[1,2])]=[90,90,90];rgb[~np.isin(semantic,[1,2,10])]=[80,130,180]
  im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
  for j,item in enumerate(centers[:8]):
   if item["in_image"]:draw.text((item["u"]+3,item["v"]),str(item["row_id"]),fill=palette[j])
  im.save(folder/(frame+"_original_firsthit_categories.png"))
 summary=dict(frame=frame,tracked_static_ids=fixed,categories=["vehicle_firsthit","road_firsthit","other_including_nohit"],unified_static_dynamic_time_correct=True,category_conservation=checks,centers=centers,actual_alphaT_categories=mass.cpu().tolist(),native_view_rgb=color.cpu().tolist(),observed_rgb_fit_rms=fit_rms.cpu().tolist(),total_alphaT=total[ids].cpu().tolist(),identity_gt=False)
 (folder/(frame+".json")).write_text(json.dumps(summary,indent=2))
 selected=torch.tensor(np.load(experiment/"m1_assignment_r1/recall_verified/selection_000008.npz")["selected"],device="cuda")
 return selected,dict(frame=frame,history=str(folder/(frame+".json")),tracked=128,identity_gt=False)
