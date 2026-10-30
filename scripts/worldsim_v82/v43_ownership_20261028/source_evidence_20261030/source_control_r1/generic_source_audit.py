"""Frozen native source attribution. Pixel responsibility is never semantic identity."""
import json,csv
from pathlib import Path
import numpy as np
import torch
CONFIG = {}
def configure(config):
 global CONFIG
 CONFIG = dict(config)
 if CONFIG.get('recall_threshold',1e-4) != 1e-4: raise ValueError('Fixed source recall threshold is1e-4; do not tune on outcomes')

def diagnose(static,dynamic,camera,pipe,degree,helpers,environment,transform,reference_root,frame,output):
 from evaluate_edits import reference_records,rgb,region_error,save_rgb
 from utils.sh_utils import eval_sh
 root=Path(reference_root);folder=output/("source_"+frame);folder.mkdir()
 # Fail closed if common query calibration differs from the independent reference camera.
 obs=json.loads((root.parent/"observations/manifest.json").read_text())
 reference_camera=next(x for x in obs["frames"] if str(x["id"]).zfill(6)==frame)
 model_c2w=camera.c2w.detach().cpu().numpy();linear=transform[:3,:3];world_c2w=np.eye(4)
 world_c2w[:3,:3]=np.linalg.inv(linear)@model_c2w[:3,:3]*float(np.linalg.norm(linear[:,0]))
 world_c2w[:3,3]=np.linalg.inv(linear)@(model_c2w[:3,3]-transform[:3,3])
 reference_K=np.asarray(reference_camera["K"]);calib_error=float(np.max(np.abs(world_c2w-np.asarray(reference_camera["camera_to_world"]))))
 intrinsic_error=max(abs(float(camera.fx)-reference_K[0,0]),abs(float(camera.fy)-reference_K[1,1]),abs(float(camera.cx)-reference_K[0,2]),abs(float(camera.cy)-reference_K[1,2]))
 if calib_error>1e-5 or intrinsic_error>1e-5:raise RuntimeError("Common independent query calibration mismatch: "+str((calib_error,intrinsic_error)))
 records,_=reference_records(root);refrow=records[(frame,"delete",True)]
 original=np.load(root/"original"/(frame+".npz"));deleted=np.load(root/refrow["geometry"])
 body=torch.tensor((original["semantic_id"]==10)&np.isin(deleted["semantic_id"],[1,2]),device="cuda")
 road=torch.tensor(np.isin(deleted["semantic_id"],[1,2])&(original["semantic_id"]!=10),device="cuda")
 target=torch.tensor(rgb(root/refrow["rgb"]),device="cuda")
 depth=torch.tensor(deleted["depth_z_m"],device="cuda",dtype=torch.float32)
 n=len(static["means3D"])
 selection_audit=None
 if CONFIG.get("selection_directory"):
  selectfile=Path(CONFIG["selection_directory"])/("selection_"+frame+".npz")
  selected=torch.tensor(np.load(selectfile)["selected"],device="cuda",dtype=torch.bool)
  if selected.shape!=(n,):raise ValueError("Fixed selection row cardinality does not match this checkpoint; rows cannot be transferred across reconstructions")
 else:
  # Recompute the SAME diagnostic alphaT rule on this checkpoint; do not transfer row identity.
  original_body=torch.tensor(original["semantic_id"]==10,device="cuda")
  combined=helpers.merge_rows(static,dynamic)
  with torch.enable_grad():
   f=combined["features"].detach().clone().requires_grad_(True)
   p=helpers.raster(camera,pipe,degree,dict(combined,features=f),torch.zeros(3,device="cuda"))
   allw=torch.autograd.grad((p["alpha"][0]*original_body).sum(),f)[0][:,1].detach()
   selected=(allw[:n]>1e-4)&static["mask"].bool().reshape(-1)
   selection_audit=dict(static_original_body_alphaT=float(allw[:n].sum()),selected_static_original_body_alphaT=float(allw[:n][selected].sum()),selected_static_body_coverage_fraction=float(allw[:n][selected].sum()/allw[:n].sum().clamp_min(1e-12)),dynamic_original_body_alphaT=float(allw[n:].sum()),all_feature_gradient_sum=float(allw.sum()),native_original_body_alpha_sum=float((p["alpha"][0]*original_body).sum()))
  (output/"selection").mkdir(exist_ok=True)
  np.savez_compressed(output/"selection"/("selection_"+frame+".npz"),selected=selected.cpu().numpy(),static_index=np.arange(n),original_body_alphaT=allw[:n].cpu().numpy())
 kept={k:None if v is None else v[~selected] for k,v in static.items()};ids=torch.arange(n,device="cuda")[~selected]
 env=environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
 scale=float(np.linalg.norm(transform[:3,0]));bg=torch.zeros(3,device="cuda")
 def raster(rows):return helpers.raster(camera,pipe,degree,rows,bg)
 def sliced(rows,mask):return {k:None if v is None else v[mask] for k,v in rows.items()}
 def stats(rows):
  with torch.enable_grad():
   f=rows["features"].detach().clone().requires_grad_(True);pack=raster(dict(rows,features=f));final=pack["rgb"]+(1-pack["alpha"])*env
   error=(final.detach()-target).abs().mean(0)
   def weights(field):return torch.autograd.grad((pack["alpha"][0]*field).sum(),f,retain_graph=True)[0][:,1].detach()
   w=weights(body);outside=weights(road);ew=weights(body*error)
   dm=weights(body*depth);d2=weights(body*depth.square())
   cm=torch.stack([weights(body*target[c]) for c in range(3)],1);c2=weights(body*target.square().mean(0))
   alpha_sum=float((pack["alpha"][0]*body).sum());grad_sum=float(w.sum());rel=abs(alpha_sum-grad_sum)/max(1,alpha_sum)
   if rel>2e-5:raise RuntimeError("alphaT conservation "+str(rel))
  return dict(body=w,road=outside,error=ew,depth_moment=dm,depth2_moment=d2,color_moment=cm,color2_moment=c2),dict(body_alpha_sum=alpha_sum,featuregrad_sum=grad_sum,relative_error=rel),{k:v.detach() for k,v in pack.items()}
 pre,precheck,prepack=stats(helpers.merge_rows(static,dynamic));post,postcheck,postpack=stats(kept)
 postfull={k:torch.zeros((n,)+tuple(v.shape[1:]),device="cuda",dtype=v.dtype) for k,v in post.items()}
 for k,v in post.items():postfull[k][ids]=v
 dirs=static["means3D"]-camera.camera_center;dirs=dirs/dirs.norm(dim=1,keepdim=True)
 colors=torch.clamp_min(eval_sh(degree,static["shs"].transpose(1,2),dirs)+.5,0) if static["colors_precomp"] is None else static["colors_precomp"]
 z=(torch.cat((static["means3D"],torch.ones_like(static["means3D"][:,:1])),1)@camera.world_view_transform[:,2])/scale
 w=postfull["body"];safe=w.clamp_min(1e-12)
 depth_rms=(z.square()-2*z*postfull["depth_moment"]/safe+postfull["depth2_moment"]/safe).clamp_min(0).sqrt()
 color_rms=(colors.square().mean(1)-2*(colors*postfull["color_moment"]).mean(1)/safe+postfull["color2_moment"]/safe).clamp_min(0).sqrt()
 gain=w/(pre["body"][:n]+1e-8)
 score=postfull["error"]
 order=torch.argsort(score,descending=True);order=order[(w[order]>1e-4)&(~selected[order])]
 newly=(gain>5)&(w>1e-4)&~selected
 mixed=(postfull["road"]>1e-4)&(w>1e-4)&~selected
 groups={}
 for count in (32,128,512):groups["top_error_"+str(count)]=order[:count]
 groups["newly_revealed_top512"]=order[newly[order]][:512]
 groups["mixed_tail_top512"]=order[mixed[order]][:512]
 groups["color_mismatch_top512"]=order[color_rms[order]>.15][:512]
 arrays=dict(static_index=np.arange(n),selected=selected.cpu().numpy(),color_native=colors.cpu().numpy(),depth_world_z=z.cpu().numpy(),scales_model=static["scales"].cpu().numpy(),means_model=static["means3D"].cpu().numpy(),opacity=static["opacities"].cpu().numpy(),post_depth_rms_reference_m=depth_rms.cpu().numpy(),post_color_rms_reference=color_rms.cpu().numpy(),occlusion_reveal_gain=gain.cpu().numpy())
 for k,v in pre.items():arrays["before_"+k]=v[:n].cpu().numpy();arrays["before_dynamic_"+k]=v[n:].cpu().numpy()
 for k,v in postfull.items():arrays["after_"+k]=v.cpu().numpy()
 np.savez_compressed(folder/"per_row_sources.npz",**arrays)
 fields=["static_id","selected","before_body_alphaT","after_body_alphaT","after_road_alphaT","error_weighted_alphaT","reveal_gain","depth_z_m","depth_rms_m","color_rms","scale_max_model","r","g","b"]
 with (folder/"top_rows.csv").open("w",newline="") as f:
  writer=csv.writer(f);writer.writerow(fields)
  for idx in order[:2000].cpu().numpy():writer.writerow([int(idx),bool(arrays["selected"][idx]),float(arrays["before_body"][idx]),float(arrays["after_body"][idx]),float(arrays["after_road"][idx]),float(arrays["after_error"][idx]),float(arrays["occlusion_reveal_gain"][idx]),float(arrays["depth_world_z"][idx]),float(arrays["post_depth_rms_reference_m"][idx]),float(arrays["post_color_rms_reference"][idx]),float(arrays["scales_model"][idx].max()),*map(float,arrays["color_native"][idx])])
 masks=dict(full=np.ones(tuple(body.shape),bool),original_body_revealed_road=body.cpu().numpy(),road_outside_original_body=road.cpu().numpy())
 scores=[]
 def save_case(name,pack,fill_env,removed=None):
  c,a=pack["rgb"],pack["alpha"];fill=(1-a)*fill_env;final=c+fill
  arr={k:v.cpu().numpy().astype(np.float32) for k,v in dict(C=c,alpha=a,E=fill_env,fill=fill,final=final,white=c+1-a,depth_sum_model=pack["depth_sum"]).items()}
  np.savez_compressed(folder/(name+".npz"),**arr)
  for k in ("C","fill","final","white"):save_rgb(folder/(name+"_"+k+".png"),arr[k])
  target_np=target.cpu().numpy();metrics={k:region_error(arr["final"],target_np,mask) for k,mask in masks.items()}
  row=dict(case=name,removed_static_ids=[] if removed is None else removed.cpu().tolist(),regions=metrics,body_alpha_sum=float((a[0]*body).sum()),environment_fill_body_abs_mean=float(fill[:,body].abs().mean()))
  scores.append(row)
 save_case("before_none",prepack,env);save_case("baseline_delete",postpack,env)
 for name,group in groups.items():
  keepmask=~selected.clone();keepmask[group]=False
  pack=raster(sliced(static,keepmask));save_case("remove_"+name,pack,env,group)
 # Exact source maps keep all retained rows and their actual shared depth ordering.
 for rank,rowid in enumerate(order[:8]):
  f=kept["features"].clone();indicator=(ids==rowid).to(f.dtype)[:,None]
  pmap=raster(dict(kept,features=torch.cat((f,indicator),1)))
  actual=pmap["extra"][0].detach().cpu().numpy()
  np.save(folder/("top%02d_row%d_alphaT.npy"%(rank,int(rowid))),actual)
  from PIL import Image
  Image.fromarray(np.round(np.clip(actual,0,1)*255).astype("uint8")).save(folder/("top%02d_row%d_alphaT.png"%(rank,int(rowid))))
 # Replacing only view-direction RGB holds geometry/opacity/T fixed; privileged diagnostic only.
 color_rows=dict(kept,shs=None,colors_precomp=colors[ids])
 color_parity=raster(color_rows)
 color_parity_max=float((color_parity["rgb"]-postpack["rgb"]).abs().max())
 if color_parity_max>2e-5:raise RuntimeError("native color reconstruction parity "+str(color_parity_max))
 replacement=colors.clone();g=groups["top_error_128"]
 replacement[g]=postfull["color_moment"][g]/safe[g,None]
 color_rows["colors_precomp"]=replacement[ids]
 color_package=raster(color_rows)
 if float((color_package["alpha"]-postpack["alpha"]).abs().max())>1e-7:raise RuntimeError("color-only alpha changed")
 save_case("top128_color_only_reference_moment",color_package,env)
 save_case("baseline_delete_env_white",postpack,torch.ones_like(env));save_case("baseline_delete_env_zero",postpack,torch.zeros_like(env));save_case("baseline_delete_env_reference",postpack,target)
 summary=dict(selection_original_body_audit=selection_audit,common_query_world_c2w_max_abs=calib_error,common_query_intrinsic_max_abs=intrinsic_error,source_config=CONFIG,selection_rule="fixed1e-4 actual original-body alphaT or explicitly bound frozen same-checkpoint selection; no row identity transfer",native_color_precompute_parity_max_abs=color_parity_max,frame=frame,static_count=n,selected_count=int(selected.sum()),roi_pixels=int(body.sum()),pre_conservation=precheck,post_conservation=postcheck,original_static_body_alphaT=float(pre["body"][:n].sum()),original_dynamic_body_alphaT=float(pre["body"][n:].sum()),removed_static_original_body_alphaT=float(pre["body"][:n][selected].sum()),retained_static_original_body_alphaT=float(pre["body"][:n][~selected].sum()),postdelete_body_alphaT=float(w.sum()),postdelete_newly_revealed_body_alphaT=float(w[newly].sum()),postdelete_mixed_tail_body_alphaT=float(w[mixed].sum()),thresholds=dict(source="unchanged recall_verified",diagnostic_body_alphaT=1e-4,newly_revealed_gain=5,mixed_body_and_road_alphaT=1e-4,color_rms=.15,rank="postdelete alphaT times per-pixel raw RGB MAE versus independent delete reference",top_k=[32,128,512]),groups={k:v.cpu().tolist() for k,v in groups.items()},cases=scores,identity_claim=False,notes="Pixel contribution and RGB/depth mismatch are causal diagnostic evidence, never Gaussian semantic identity. Revealed road may be correct background. Environment-reference replacement is privileged evaluation-only sensitivity, not a method output.")
 (folder/"source_receipt.json").write_text(json.dumps(summary,indent=2));return selected,summary
