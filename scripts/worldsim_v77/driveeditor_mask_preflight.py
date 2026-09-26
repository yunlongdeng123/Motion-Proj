"""DriveEditor删除输入预检：只执行官方mask几何，不加载生成权重。"""
import ast,json,pathlib,subprocess,time,sys
import cv2,numpy as np,torch
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

SOURCE=pathlib.Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
OUT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-ASSESS-20260926/r1');OUT.mkdir(parents=True,exist_ok=True)
def getbox(row,f):
 a=row['frame_annotations']
 if f not in a['frame_idx']:return None
 i=a['frame_idx'].index(f);return np.asarray(a['obj_to_world'][i]),np.asarray(a['box_size'][i])
def project(item,c2w,K):
 pose,size=item;p=np.linalg.inv(c2w)@pose;l,w,h=size
 b=Box(p[:3,3],[w,l,h],Quaternion(matrix=p[:3,:3]))
 if (b.corners()[2]<=.1).any():return None
 return view_points(b.corners(),K,normalize=True)[:2].T

raw=subprocess.check_output(['git','-C',str(SOURCE),'show','HEAD:interactive_gui.py'],text=True)
tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scale_bbox')
ns={'np':np,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'official_scale_bbox','exec'),ns)
scale_bbox=ns['scale_bbox'];torch.set_num_threads(4)
manifest={'task_id':'WS-V77-DRIVEEDITOR-ASSESS-20260926','run_id':'r1','input_roles':'原GT相机/轨迹、SAM2 mask；仅mask输入预检，无背景真值或模型成绩','seed':42,'upstream_commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'model_forward_count':0,'training_steps':0,'resource':'CPU 4 threads; no checkpoint loading','checkpoint_exists':(SOURCE/'checkpoints/model.safetensors').is_file(),'demo_data_exists':(SOURCE/'checkpoints/data.pkl').is_file(),'torch_version':torch.__version__,'failure_ledger_refs':['V77-F02','V75-F02'],'failure_ledger_delta':'none; preflight is not generated quality evidence','human_verdict':None}
scenes=[]
for s,aid,cam,start,ref in [('scene_0230','22',2,0,5),('scene_0255','25',3,15,20)]:
 torch.manual_seed(42);D=pathlib.Path('/root/autodl-tmp/data/v76_vadgs')/s;inst=json.loads((D/'instances/instances_info.json').read_text());v=np.loadtxt(D/'intrinsics'/f'{cam}.txt');K=np.array([[v[0],0,v[2]],[0,v[1],v[3]],[0,0,1]])
 rows=[]
 for f in range(start,start+10):
  c2w=np.loadtxt(D/'extrinsics'/f'{f:03}_{cam}.txt');corners=project(getbox(inst[aid],f),c2w,K);assert corners is not None
  bbox=scale_bbox(corners,(900,1600),1.9,1.9);coords=np.round(bbox*.64).astype(int);mask=np.zeros((576,1024),np.uint8);mask[coords[1]:coords[3],coords[0]:coords[2]]=1
  path=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')/s/'masks'/f'cam{cam}'/f'{f:05}.png';sam=cv2.resize(cv2.imread(str(path),0),(1024,576),interpolation=cv2.INTER_NEAREST)>0
  overlaps=[]
  for other_id,other in inst.items():
   if other_id==aid:continue
   item=getbox(other,f)
   if item is None:continue
   oc=project(item,c2w,K)
   if oc is None:continue
   poly=cv2.convexHull(np.round(oc*.64).astype(np.int32));om=np.zeros_like(mask);cv2.fillConvexPoly(om,poly,1);area=int(om.sum());inter=int((om&mask).sum())
   if inter>0:overlaps.append({'actor_id':other_id,'intersection_pixels':inter,'fraction_of_projected_box':inter/max(1,area),'note':'3D框投影包络相交，不等于可见实例mask损伤'})
  rows.append({'frame':f,'mask_fraction_image':float(mask.mean()),'sam_fraction_image':float(sam.mean()),'sam_coverage':float(mask[sam].mean()) if sam.any() else None,'mask_to_sam_area_ratio':float(mask.sum()/max(1,sam.sum())),'bbox_original_xyxy':bbox.tolist(),'neighbor_projected_box_overlaps':overlaps})
  if f==ref:
   im=cv2.resize(cv2.imread(str(D/'images'/f'{f:03}_{cam}.jpg')),(1024,576));overlay=im.copy();tint=np.full_like(im,(50,150,240));overlay[mask>0]=(im[mask>0]*.55+tint[mask>0]*.45).astype(np.uint8)
   contours,_=cv2.findContours(sam.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(255,255,0),2)
   for other in sorted(overlaps,key=lambda x:x['intersection_pixels'],reverse=True)[:2]:
    oc=project(getbox(inst[other['actor_id']],f),c2w,K);poly=cv2.convexHull(np.round(oc*.64).astype(np.int32));cv2.polylines(overlay,[poly],True,(160,80,245),2);x,y=np.mean(poly[:,0,:],axis=0).astype(int);cv2.putText(overlay,'actor'+other['actor_id'],(max(0,min(x,900)),max(25,min(y,550))),cv2.FONT_HERSHEY_SIMPLEX,.6,(160,80,245),2)
   cv2.rectangle(overlay,(0,0),(1024,34),(30,30,30),-1);cv2.putText(overlay,f'{s} / target {aid} / f{f:03} / CAM{cam} | orange: official removal mask; cyan: SAM2',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
   cv2.imwrite(str(OUT/f'{s}_mask.jpg'),overlay);cv2.imwrite(str(OUT/f'{s}_source.jpg'),im)
  (OUT/s).mkdir(exist_ok=True);cv2.imwrite(str(OUT/s/f'mask_{f:03}.png'),mask*255)
 row={'scene':s,'actor_id':aid,'camera':cam,'frame_range':[start,start+9],'frames':rows,'reference_frame':ref,'frames_with_neighbor_projected_box_overlap':sum(bool(x['neighbor_projected_box_overlaps']) for x in rows),'mask_fraction_image_range':[min(x['mask_fraction_image'] for x in rows),max(x['mask_fraction_image'] for x in rows)],'sam_coverage_range':[min(x['sam_coverage'] for x in rows),max(x['sam_coverage'] for x in rows)],'seed_note':'isolated official mask function seeded42; actual full inference RNG order may differ; save/reuse masks for matched controls','human_verdict':None};scenes.append(row)
 print(json.dumps({k:v for k,v in row.items() if k!='frames'},ensure_ascii=False),flush=True)
(OUT/'summary.json').write_text(json.dumps({'registration':manifest,'scenes':scenes},ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False))
