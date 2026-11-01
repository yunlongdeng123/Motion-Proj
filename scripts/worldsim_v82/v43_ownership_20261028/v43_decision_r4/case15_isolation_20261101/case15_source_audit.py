import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import pathlib,json,math,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
W=pathlib.Path(__file__).resolve().parent;H=W/'case15';O=W.parent/'outputs'/'机制隔离15例_20261101';(O/'assets').mkdir(parents=True,exist_ok=True)
BASE=pathlib.Path(r'C:\Users\dengyunlong\Documents\Codex\2026-10-31\g\work\actor_diagnostics_20261101')
M=BASE/'media/generated';G=BASE/'root/gs_cache';SH=pathlib.Path(r'C:\Users\dengyunlong\Documents\Codex\2026-10-22\ni-x-a\outputs\GS_Scene_Diagnostics_20261027')
R=H/'remote/evidence'; selected=json.loads((H/'shortlist.json').read_text(encoding='utf8'))
views=json.loads((BASE/'media/spatial/actual_spatial_view_vectors.json').read_text(encoding='utf8'))
spatial=json.loads((BASE/'media/spatial/actual_spatial_stats.json').read_text(encoding='utf8'));smap={(r['context_id'],r['track_id']):r for r in spatial['rows']}
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
def norm(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def angles(v,bank):return np.degrees(np.arccos(np.clip(v@bank.T,-1,1))).min(1)
def cropframe(im,bounds):
 x0,y0,x1,y1=bounds;cx=(x0+x1)/2;cy=(y0+y1)/2;w=max(x1-x0+60,180);h=max(y1-y0+55,w*0.62);w=max(w,h/0.62)
 return im.crop((round(cx-w/2),round(cy-h/2),round(cx+w/2),round(cy+h/2))).resize((380,240),Image.Resampling.LANCZOS)
def bbox(box,wh):
 official=(box.get('projection_comparison') or {}).get('official_box')
 if official:
  return [max(0,official[0]),max(0,official[1]),min(wh[0],official[2]),min(wh[1],official[3])]
 # Clipped edge endpoints include image-boundary intersections; surviving 3D
 # corners alone would omit them and make truncated boxes artificially thin.
 pts=[e[:2] for e in box.get('edges',[])]+[e[2:] for e in box.get('edges',[])]
 if not pts:pts=[p for p in box.get('corners2D',[]) if p is not None]
 if not pts:return None
 xy=np.array(pts);lo=np.maximum(xy.min(0),[0,0]);hi=np.minimum(xy.max(0),wh)
 if np.any(hi<=lo):return None
 return [*lo,*hi]
results=[]
for context,subrows in __import__('itertools').groupby(sorted(selected,key=lambda r:r['context_id']),key=lambda r:r['context_id']):
 media=json.loads((M/'scenes'/context/'media.json').read_text(encoding='utf8'));frames={f['raw']:f for f in media['frames']};cal=media['projection']['source_calibration'];ext=np.array(cal['extrinsic']).reshape(4,4)
 geom=json.loads((G/(context+'_geometry.json')).read_text(encoding='utf8')) if (G/(context+'_geometry.json')).exists() else None
 for r in subrows:
  key=r['scene_key']+'_'+r['alias'];win=r['native_window'] or r['recommended49'];first,last=win['first_raw_frame'],win['last_raw_frame'];raws=list(range(first,last+1));anchor=[first,(first+last)//2,last]
  actor=media['actors'][r['track_id']];poses={x['raw']:x for x in actor['trajectory']};used=[poses[k] for k in raws if k in poses]
  centers=np.array([p['center_world'] for p in used]);times=np.array([p['timestamp_micros'] for p in used],dtype=np.float64)/1e6;rot=np.array([p['orientation_world'] for p in used]).reshape(-1,3,3);dims=np.array([p['dimensions'] for p in used]);dt=np.diff(times);speed=np.linalg.norm(np.diff(centers,axis=0),axis=1)/dt;az=np.unwrap(np.arctan2(rot[:,1,0],rot[:,0,0]));yawrate=np.degrees(np.diff(az))/dt
  bankrec=[x for x in views[context+'|'+r['track_id']] if first<=x['raw']<=last and x['official_FRONT_visible']];bank=norm(np.array([x['camera_direction_actor'] for x in bankrec]));viewspan=float(np.degrees(np.arccos(np.clip(bank@bank.T,-1,1))).max())
  cameras=[];native=[];closure=[];native_pose=[]
  cams_path=R/key/'cameras.json';Tpath=R/key/'transform_matrix.txt'
  cams=json.loads(cams_path.read_text(encoding='utf8')) if cams_path.exists() else [];cams={x['img_name']:x for x in cams}
  inv=np.linalg.inv(np.loadtxt(Tpath)) if Tpath.exists() else None
  for raw in raws:
   p=poses[raw];c2w_raw=np.array(frames[raw]['camera_pose_world']).reshape(4,4)@ext;C=c2w_raw[:3,3];cameras.append(C)
   if cams:
    fid=(r['cpu_input_audit']['frames'][raw-first]['frame_id'] if 'frames' in r.get('cpu_input_audit',{}) else None)
    # frame identifiers are joined via preparation records, never guessed asset aliases.
    prep=json.loads((R/key/'preparation.json').read_text(encoding='utf8')) if raw==first else prep
    pa=next(a for a in prep['actors'] if a['actor_index']==r['dynamic_actor_index'] and a['raw_track_id']==r['track_id']);fid=pa['frames'][raw-first]['frame_id'];c=cams[fid];Cn=inv[:3,:3]@c['position']+inv[:3,3];closure.append(float(np.linalg.norm(Cn-C)));native.append(Cn)
   elif geom:
    sf=geom['scene']['frames'][str(raw)];im=np.linalg.inv(np.array(geom['geometry']['local_from_world']));Cn=im[:3,:3]@np.array(sf['c2w'])[:3,3]+im[:3,3];closure.append(float(np.linalg.norm(Cn-C)));native.append(Cn)
   if geom and r['track_id']==geom['scene']['target_track']:
    sf=geom['scene']['frames'][str(raw)];L=np.array(geom['geometry']['local_from_world']);native_pose.append(float(np.max(abs(L[:3,:3]@p['center_world']+L[:3,3]-sf['target_center']))))
  cameras=np.array(cameras);rz=np.array([[math.cos(math.pi/6),-math.sin(math.pi/6),0],[math.sin(math.pi/6),math.cos(math.pi/6),0],[0,0,1]])
  dirs_rotate=norm(np.einsum('ni,nij->nj',(cameras-centers)@rz,rot));dirs_shift=norm(np.einsum('ni,nij->nj',cameras-centers-np.array([0,1.5,0]),rot));rot_dist=angles(dirs_rotate,bank);shift_dist=angles(dirs_shift,bank)
  rec={'scene_key':r['scene_key'],'alias':r['alias'],'case':key,'raw_window':[first,last],'source_track_id':r['track_id'],'source_context':context,'source_pose_frames':len(used),'source_track_gap_count':sum(np.diff([p['raw'] for p in used])!=1),'timestamps_strictly_increasing':bool((dt>0).all()),'box_records_finite_positive':bool(np.isfinite(centers).all() and np.isfinite(rot).all() and (dims>0).all()),'rotation_orthogonality_max_abs':float(np.max(abs(np.transpose(rot,(0,2,1))@rot-np.eye(3)))),'source_speed_mps':{'median':float(np.median(speed)),'max':float(speed.max())},'source_abs_yaw_rate_degps_max':float(abs(yawrate).max()),'source_dimension_max_minus_min_m':np.ptp(dims,axis=0).tolist(),'calibrated_original_view_pair_max_deg':viewspan,'view_bank_role':'native-window FRONT pose-bearing proxy' if r['scene_checkpoint_recorded'] else 'planned preparation-window source proxy, no frozen G training support','rotate30_worldZ_nearest_original_direction_deg':{'median':float(np.median(rot_dist)),'max':float(rot_dist.max())},'shift_worldY1p5m_nearest_original_direction_deg':{'median':float(np.median(shift_dist)),'max':float(shift_dist.max())},'request_semantics':'Inherited R4 query convention: +30deg world Z; +1.5m world Y. Source direction probe only; no edited image generated.','native_camera_origin_vs_source_pose_distance_m':{'max':max(closure),'median':float(np.median(closure))} if closure else None,'native_target_center_source_parity_max_abs_m':max(native_pose) if native_pose else None,'native_visualization_checkpoints':'original 30K eval for four existing scenes; repaired10k platform legacy RGB explicitly excluded','source_spatial_full_sequence':smap[(context,r['track_id'])],'mask_anchor_diagnostics':[],'render_layer_anchor_diagnostics':[],'human_verdict':None}
  manifest=R/key/'preparation_manifest.json'
  if manifest.exists():
   mf=json.loads(manifest.read_text(encoding='utf8')); mr=mf.get('frames',[])
   if isinstance(mr,list) and mr and 'source_index' in mr[0] and 'timestamp_micros' in mr[0]:
    joined=[p for p in mr if p['source_index'] in raws]
    rec['native_preparation_source_index_timestamp_matches']=sum(p['timestamp_micros']==frames[p['source_index']]['timestamp_micros'] for p in joined)
    rec['native_preparation_source_index_timestamp_denominator']=len(raws)
  # Source/camera/mask/contact proof is per actor and per fixed first/middle/last anchor.
  atlas=Image.new('RGB',(1160,920),'#f4f6fa');dr=ImageDraw.Draw(atlas);dr.text((12,8),key+' · 机制隔离源证据（固定三锚）',fill='#18324b',font=font)
  for j,raw in enumerate(anchor):
   fr=frames[raw];box=next(b for b in fr['boxes'] if b['track']==r['track_id']);bb=bbox(box,media['imageWH']);image=Image.open(M/fr['image']).convert('RGB');native_rgb=None;layers=None
   if bb is None:raise ValueError((key,raw,'no expected FRONT box'))
   dr.text((j*385+10,42),f'raw {raw} · 原始影像 + 目标框/已有mask',fill='#18324b',font=small)
   mask_path=R/key/(str(raw)+'_mask.png')
   if mask_path.exists():
    mk=Image.open(mask_path);mk=mk.resize(image.size,Image.Resampling.NEAREST);arr=np.asarray(mk)>0;ys,xs=np.where(arr);x0,y0,x1,y1=[int(x) for x in bb];rect=np.zeros_like(arr);rect[max(0,y0):min(arr.shape[0],y1+1),max(0,x0):min(arr.shape[1],x1+1)]=True
    rec['mask_anchor_diagnostics'].append({'raw':raw,'downsampled_nonzero_pixels':int(arr.sum()),'mask_inside_projected_box_fraction':float((arr&rect).sum()/arr.sum()),'mask_bbox_resized_pixels':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],'projected_box_pixels':bb,'meaning':'box is source projected cuboid, not independent instance mask; outside pixels are engineering review triggers, not semantic failure labels'})
    overlay=Image.new('RGB',image.size,'#26cc70');image=Image.composite(Image.blend(image,overlay,.3),image,mk)
   ImageDraw.Draw(image).rectangle(bb,outline='#1d7dce',width=2);atlas.paste(cropframe(image,bb),(j*385+5,70))
   if geom:
    sg=next(g for g in geom['scene']['groups'] if g['id']=='reconstruction_30000');paths=sg['frames'][str(raw)]['channels'];layers={k:Image.open(SH/context/p).convert('RGB') for k,p in paths.items()};native_rgb=layers['rgb']
   elif r['scene_key']=='waymo-train-008':
    p=H/'legacy_export'/('eval_%03d.png'%(raw-first))
    if p.exists():
     grid=Image.open(p).convert('RGB');ww=(grid.width-6)//2;hh=(grid.height-6)//2;layers={name:grid.crop((2+col*(ww+2),2+row*(hh+2),2+col*(ww+2)+ww,2+row*(hh+2)+hh)) for name,row,col in [('gt',0,0),('rgb',0,1),('static',1,0),('dynamic',1,1)]};native_rgb=layers['rgb']
   if layers:
    scaled=[bb[0]*native_rgb.width/image.width,bb[1]*native_rgb.height/image.height,bb[2]*native_rgb.width/image.width,bb[3]*native_rgb.height/image.height]
    for row,(name,title) in enumerate([('rgb','原30K合成（原生GPU存档）'),('static','静态层 + 环境填充'),('dynamic','动态层（多个对象可能共层）')]):
     dr.text((j*385+10,325+row*194),title,fill='#18324b',font=small);atlas.paste(cropframe(layers[name],scaled).resize((380,165)),(j*385+5,350+row*194))
    region=np.zeros((native_rgb.height,native_rgb.width),bool)
    if mask_path.exists():region=np.asarray(Image.open(mask_path).resize(native_rgb.size,Image.Resampling.NEAREST))>0;sem='associated native body-mask ROI, not independent GT'
    else:
     x0,y0,x1,y1=[int(round(x)) for x in scaled];region[max(0,y0):min(region.shape[0],y1+1),max(0,x0):min(region.shape[1],x1+1)]=True;sem='projected box ROI includes background, not a body mask'
    aa={k:np.asarray(im,dtype=float)/255 for k,im in layers.items()};rec['render_layer_anchor_diagnostics'].append({'raw':raw,'ROI_semantics':sem,'ROI_pixels':int(region.sum()),'native_composite_vs_source_GT_MAE':float(abs(aa['rgb'][region]-aa['gt'][region]).mean()),'static_with_env_vs_composite_MAE':float(abs(aa['static'][region]-aa['rgb'][region]).mean()),'dynamic_layer_RGB_absolute_mean':float(abs(aa['dynamic'][region]).mean()),'warning':'static image contains environment contribution; cannot infer road-G color pollution or ownership from this layer alone'})
   else:dr.text((j*385+10,342),'无冻结 G / 原生层结果；不制造渲染结论',fill='#985c20',font=small)
  imagepath=O/'assets'/(key+'_contact.png');atlas.save(imagepath);rec['contact_sheet']='assets/'+imagepath.name
  results.append(rec);print(key,'source49_checked','view_span',round(viewspan,3),'turn30_nearest',round(float(np.median(rot_dist)),3),flush=True)
(H/'source_audit.json').write_text(json.dumps({'task_id':'WS-V43-CASE15-ISOLATION-20261101-R1','new_GPU_queries':0,'rows':results},ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf8')
