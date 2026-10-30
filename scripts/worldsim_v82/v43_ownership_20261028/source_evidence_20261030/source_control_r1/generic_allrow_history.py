"""All static rows frozen observed-view replay, CPU accumulation, no top-subset inference."""
from pathlib import Path
import numpy as np,json,torch
ACCUM=None
MAXIMUM=None
COUNTS=None
FRAMES=[]

def diagnose(static,dynamic,camera,pipe,degree,helpers,environment,transform,reference_root,frame,output):
 global ACCUM,MAXIMUM,COUNTS,FRAMES
 root=Path(reference_root);experiment=root.parent.parent;n=len(static["means3D"])
 original=np.load(root/"original"/(frame+".npz"));sem=torch.tensor(original["semantic_id"].astype(np.int32),device="cuda")
 masks=[sem==10,(sem==1)|(sem==2),~((sem==10)|(sem==1)|(sem==2))]
 if ACCUM is None:ACCUM=np.zeros((n,3),np.float64);MAXIMUM=np.zeros(n,np.float64);COUNTS=np.zeros((n,3),np.int16)
 if len(ACCUM)!=n:raise ValueError("Row cardinality changed during frozen replay")
 values=helpers.merge_rows(static,dynamic);checks=[];weights=[]
 with torch.enable_grad():
  f=values["features"].detach().clone().requires_grad_(True);p=helpers.raster(camera,pipe,degree,dict(values,features=f),torch.zeros(3,device="cuda"))
  for mask in masks:
   g=torch.autograd.grad((p["alpha"][0]*mask).sum(),f,retain_graph=True)[0][:,1].detach()
   total=float((p["alpha"][0]*mask).sum());error=abs(float(g.sum())-total)/max(1,total)
   if error>2e-5:raise RuntimeError("Feature gradient conservation"+str(error))
   checks.append(dict(alpha_sum=total,gradient_sum=float(g.sum()),relative_error=error));weights.append(g[:n].cpu().numpy())
 w=np.stack(weights,1).astype(np.float64);ACCUM+=w;MAXIMUM=np.maximum(MAXIMUM,w.sum(1));COUNTS+=(w>1e-4).astype(np.int16);FRAMES.append(frame)
 folder=output/"allrow_history";folder.mkdir(exist_ok=True)
 (folder/(frame+".json")).write_text(json.dumps(dict(frame=frame,static_rows=n,category_conservation=checks,categories=["original_vehicle_firsthit","original_road_firsthit","other_including_nohit"]),indent=2))
 # Publish aggregate at each frame for recoverability; never stores16xfull arrays.
 np.savez_compressed(folder/"aggregate.npz",static_index=np.arange(n),sum_alphaT_by_category=ACCUM,total_alphaT_over_original_views=ACCUM.sum(1),max_frame_total_alphaT=MAXIMUM,frames_category_gt1e4=COUNTS,frames_seen=np.array(FRAMES))
 selected=torch.zeros(n,device="cuda",dtype=torch.bool) # no edits are allowed in this observed-view replay
 return selected,dict(frame=frame,allrow_aggregate=str(folder/"aggregate.npz"),static_rows=n,identity_gt=False)
