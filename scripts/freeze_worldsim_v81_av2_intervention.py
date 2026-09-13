"""模型前按可见表面限定矩形；不换日志，不增加候选，不查看模型输出。"""
import json,datetime
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from motion_proj.worldsim_v81.geometry import project,plane_fit,roi_mask,coverage,apply
O=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01');A=O/'av2_atlas';rows=json.loads((A/'selected_before_model.json').read_text())
# 序号来自冻结的16张RGB/参考图；矩形只按同一可见表面边界指定。
rects={2:[.48,.20,.80,.95],5:[.06,.06,.94,.94],10:[.05,.57,.95,.95],13:[.06,.06,.68,.65]}
notes=['桥梁上缘与天空混合，低梯度由天空主导','桥梁边缘与天空混合','灰色立柱正面；限制到中央同一可见面','玻璃店面与门框，多层','有文字的墙面，保留为来源筛查资产，不进入本轮低纹理干预','砖墙内部；保留天然砖纹，不称无纹理','窗口与砖墙交界','栏杆、凹窗与砖墙混合','门窗和砖墙混合','牌匾、窗与墙混合','浅色建筑墙面；限制到格栅下方平面内部','斑马线、立柱与条纹墙混合','门框、灯具与屋檐混合','灰色灰泥墙；避开右侧砖边与下方窗口','植被、树干、车身多层','玻璃反射与窗框，不属于可靠漫反射硬平面']
review=[];selected=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);canvas=Image.new('RGB',(1200,4*260),'white');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(rows):
 entry={'roi_id':r['roi_id'],'reviewer':'assistant_visual_review','human_verdict':None,'model_outputs_accessed':False,'observation':notes[i],'status':'NOT_SELECTED_FOR_INTERVENTION'};review.append(entry)
 if i not in rects:continue
 raw=dict(np.load(A/'reference_geometry'/f'{r["roi_id"]}.npz'));x0,y0,x1,y1=r['box'];f=rects[i];box=np.array([x0+f[0]*(x1-x0),y0+f[1]*(y1-y0),x0+f[2]*(x1-x0),y0+f[3]*(y1-y0)]).round().astype(int);ok=roi_mask(raw['uv'],raw['depth_z'],box);ref={**raw,**{k:raw[k][ok] for k in ['uv','xyz','depth_z','source_scan','world']}};fit=plane_fit(ref['xyz']);spread=[]
 if fit:
  for s in np.unique(ref['source_scan']):
   q=plane_fit(ref['xyz'][ref['source_scan']!=s])
   if q:spread.append(float(np.degrees(np.arccos(np.clip(abs(q['normal']@fit['normal']),0,1)))))
 pass_ref=bool(fit and len(ref['uv'])>=30 and coverage(ref['uv'],box)>=.25 and fit['rms']<=.12 and fit['inlier_fraction']>=.85 and len(spread)>=2 and max(spread)<=5)
 entry.update(status='INTERIOR_REFERENCE_PASS' if pass_ref else 'INTERIOR_REFERENCE_REJECT',box=box.tolist(),reference_points=len(ref['uv']),reference_coverage=coverage(ref['uv'],box),reference_normal_spread_deg=max(spread) if spread else None)
 if not pass_ref:continue
 ref.update(center=fit['center'],normal=fit['normal'],box=box);np.savez_compressed(A/'reference_geometry'/f'{r["roi_id"]}_interior.npz',**ref)
 m=json.loads((A/'input_manifests'/f'{r["window_id"]}.json').read_text());target=next(v for v in m['views'] if v['camera']==r['camera']);context=sorted([v for v in m['context_views'] if v['camera']==r['camera']],key=lambda v:v['timestamp_us']);manifest={**m,'views':[target],'context_views':context};dest=A/'input_manifests'/f'{r["roi_id"]}_intervention.json';dest.write_text(json.dumps(manifest,indent=2))
 clouds=np.load(m['input_lidar_world']);geom=[];ri=len(selected);point_corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]);rays=np.c_[point_corners,np.ones(4)]@np.linalg.inv(ref['K']).T;xyz=rays*((fit['center']@fit['normal'])/(rays@fit['normal']))[:,None];world=apply(xyz,ref['world_from_camera']);w,h=box[2:]-box[:2];destuv=np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
 for j,v in enumerate([context[0],target,context[1]]):
  C=np.array(v['world_from_camera']);K=np.array(v['K']);uv,z,_=project(ref['world'],C,K);W,H=v['size'];visible=roi_mask(uv,z,[0,0,W,H]);pu,pz,_=project(clouds,C,K);pok=roi_mask(pu,pz,[0,0,W,H]);dist,nn=cKDTree(pu[pok]).query(uv);visible &= ~((dist<10)&(z>pz[pok][nn]+.5+.01*z));a=ref['world']-np.array(target['world_from_camera'])[:3,3];b=ref['world']-C[:3,3];angles=np.degrees(np.arccos(np.clip((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)), -1,1)));geom.append({'timestamp_us':v['timestamp_us'],'visible_frustum_and_sparse_occlusion_fraction':float(visible.mean()),'median_parallax_deg':float(np.median(angles[visible])) if visible.any() else None,'visibility_boundary':'current sparse cloud rejection plus RGB plane-warp review; not dense visibility ground truth'})
  uv4,_,_=project(world,C,K);Hmat=cv2.getPerspectiveTransform(uv4.astype(np.float32),destuv);rgb=np.array(Image.open(v['image']).convert('RGB'));warped=cv2.warpPerspective(rgb,Hmat,(int(w),int(h)));im=Image.fromarray(warped);im.thumbnail((390,205));canvas.paste(im,(j*400,ri*260+45));draw.text((j*400+3,ri*260+3),f'{r["log"][:8]} {[-.5,0,.5][j]:+.1f}s | angle {(geom[-1]["median_parallax_deg"] or 0):.2f}\nvisible {visible.mean():.2f}',fill='black',font=font)
 eligible=all(g['visible_frustum_and_sparse_occlusion_fraction']>=.7 and (g['median_parallax_deg'] or 0)>=1 for g in [geom[0],geom[2]])
 entry['effective_evidence_geometry_pass']=eligible;selected.append({'roi_id':r['roi_id'],'log':r['log'],'camera':r['camera'],'manifest':str(dest),'reference':str(A/'reference_geometry'/f'{r["roi_id"]}_interior.npz'),'box':box.tolist(),'reference_pass':pass_ref,'geometry':geom,'effective_evidence_geometry_pass':eligible,'model_status':'NOT_RUN','human_verdict':None,'texture_note':notes[i]})
(A/'visual_reviews_before_model.json').write_text(json.dumps(review,ensure_ascii=False,indent=2));(A/'interventions_before_model.json').write_text(json.dumps({'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_role':'DISCOVERY_MODEL_BLIND','interior_selection':'same-surface rectangles from fixed top2/8 logs; no replacement; frozen before any new forward','selected':selected},ensure_ascii=False,indent=2));canvas.save(O/'figures/av2_effective_evidence_before_model.png');print(json.dumps(selected,indent=2))
