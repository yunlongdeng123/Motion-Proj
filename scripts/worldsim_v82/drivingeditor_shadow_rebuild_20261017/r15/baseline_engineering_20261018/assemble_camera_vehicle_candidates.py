"""Actual camera33 mask/identity/real-initialization candidate set, CPU only.

The output is deliberately not the training gate. An unchanged full native
proposal may replace an instance candidate when it better matches the actual
camera-label extent. No rectangle filling, GT-box point cropping, or new points.
"""
from pathlib import Path
import os,json,collections
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
E=Path(__file__).resolve().parent;SRC=E.parent/'inputs/scene0008_089_137';OUT=E/'all_vehicle_camera33_candidates';N=E/'all_vehicle_camera_prompt49/native_predictions'
def component_seed(xyz,indices):
 before=len(indices)
 if not before:return indices,None,None
 edges=cKDTree(xyz[indices]).query_pairs(.5,output_type='ndarray');graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(before,before));_,cc=connected_components(graph,directed=False);counts=np.bincount(cc);dom=cc==counts.argmax();fraction=float(dom.mean())
 return (indices[dom] if fraction>=.5 else indices[:0]),fraction,('entire_observation_no_dominant_component' if fraction<.5 else None)
def metrics(m,bb):
 ys,xs=np.where(m)
 if not len(xs):return 0.,0.
 inter=float(((xs>=bb[0])&(xs<=bb[2])&(ys>=bb[1])&(ys<=bb[3])).sum());area=(bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)
 return inter/(len(xs)+area-inter),inter/len(xs)
def main():
 assert not OUT.exists(),'preserve prior candidate set'
 receipt=json.loads((N.parent/'inference_receipt.json').read_text());assert receipt['frames']==49
 contract=json.loads((N.parent/'input_contract.json').read_text());ids=contract['actual_camera_vehicle_track_ids'];assert len(ids)==33
 OUT.mkdir();(OUT/'frames').mkdir();(OUT/'labels').mkdir();(OUT/'unknown_overlap').mkdir()
 for i in range(len(ids)):(OUT/f'actors/actor_{i}/body_masks').mkdir(parents=True)
 tracks={t:dict(camera_actor_index=i,raw_camera_track_id=t,frames=[],total_real_seed_points=0,zero_seed_frames=0,mask_sources=collections.Counter()) for i,t in enumerate(ids)}
 clouds={t:[] for t in ids};times={t:[] for t in ids};sources={t:{} for t in ids};votes={t:collections.Counter() for t in ids};frames=[]
 prep=json.loads((SRC/'preparation_manifest.json').read_text());model_times={r['id']:r['model_time'] for r in prep['frames']}
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 for f in prep['frames']:
  fid=f['id'];queries=json.loads((N/(fid+'.json')).read_text())['queries'];z=np.load(N/(fid+'.npz'));inst=z['masks'];qualities=z['scores'];nv=np.load(E/'all_vehicle_native49/native_predictions'/(fid+'.npz'));native=nv['masks'];native_scores=nv['scores']
  rgb=np.array(Image.open(SRC/'image/image_0'/(fid+'.png')).convert('RGB'));xyz=np.fromfile(SRC/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3].astype(float);pose=np.loadtxt(SRC/'pose'/(fid+'.txt'));hom=np.c_[xyz,np.ones(len(xyz))]
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/(fid+'.txt')).read_text().splitlines() if ':' in s};cam=hom@cal['Tr_velo_to_cam_0'].reshape(3,4).T
  pos=np.flatnonzero(cam[:,2]>.01);q=np.c_[cam[pos],np.ones(len(pos))]@cal['P0'].reshape(3,4).T;uv=np.floor(q[:,:2]/q[:,2:3]+.5).astype(int);inside=(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1280);idx=pos[inside];uv=uv[inside];depth=cam[idx,2];bins=(uv[:,1]//2)*960+uv[:,0]//2;nearest=np.full(960*640,np.inf);np.minimum.at(nearest,bins,depth);vis=depth<=nearest[bins]+.5;idx=idx[vis];uv=uv[vis]
  anns={a['id']:a for a in json.loads((SRC/'annotations'/(fid+'.json')).read_text())['laser_labels'] if a['type']=='TYPE_VEHICLE'};gt_hits={}
  for t,a in anns.items():
   b=a['box'];c,s=np.cos(b['heading']),np.sin(b['heading']);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   gt_hits[t]=((((xyz[idx]-[b['center_x'],b['center_y'],b['center_z']])@rot).__abs__())<=np.array([b['length'],b['width'],b['height']])/2).all(1)
  selected=[];union=np.zeros((1280,1920),bool);multiplicity=np.zeros((1280,1920),np.uint8);labels=np.zeros((1280,1920),np.uint16);overlay=rgb.copy();texts=[]
  # All masks remain complete predictions. The actual camera-label extent is
  # used only to choose an existing proposal, never to clip its pixels.
  native_assignment={}
  for j,m in enumerate(native):
   scores=[metrics(m,r['prompt_box_xyxy'])[0] for r in queries]
   if scores:native_assignment[j]=int(np.argmax(scores))
  for qi,row in enumerate(queries):
   t=row['raw_camera_track_id'];bb=row['prompt_box_xyxy'];pool=[]
   for j in range(row['proposal_offset'],row['proposal_offset']+3):
    iou,fraction=metrics(inst[j],bb);pool.append((iou*max(0,float(qualities[j]))*(1 if fraction>=.65 else .2),'native_instance',j,float(qualities[j]),iou,fraction,inst[j]))
   for j,m in enumerate(native):
    if native_assignment.get(j)!=qi:continue
    iou,fraction=metrics(m,bb)
    if iou>=.3 and fraction>=.65:pool.append((iou*float(native_scores[j])*1.1,'native_video',j,float(native_scores[j]),iou,fraction,m))
   score,kind,j,quality,iou,fraction,m=max(pool,key=lambda x:x[0]);hit=m[uv[:,1],uv[:,0]];retained,dominance,reject=component_seed(xyz,idx[hit]);v=tracks[t]
   raw_votes={other:int((hit&h).sum()) for other,h in gt_hits.items()};raw_votes={other:n for other,n in raw_votes.items() if n};votes[t].update(raw_votes)
   world=(hom[retained]@pose.T)[:,:3].astype(np.float32);clouds[t].append(world);times[t].append(np.full(len(world),model_times[fid],np.float32));sources[t][fid]=retained.astype(np.int32)
   Image.fromarray(m.astype(np.uint8)*255).save(OUT/f"actors/actor_{row['camera_actor_index']}/body_masks"/(fid+'.png'))
   union|=m;multiplicity+=m;labels[m]=row['camera_actor_index']+1
   v['total_real_seed_points']+=len(retained);v['zero_seed_frames']+=not len(retained);v['mask_sources'][kind]+=1
   rec=dict(frame_id=fid,mask_source=kind,native_index=j,pixels=int(m.sum()),quality=quality,mask_rectangle_iou=iou,fraction_inside_camera_rectangle=fraction,real_points_before_component=int(hit.sum()),retained_real_points=len(retained),dominant_fraction=dominance,whole_observation_rejected=reject,raw_LiDAR_annotation_votes=sorted(raw_votes.items(),key=lambda x:-x[1]),visual_review_passed=False)
   v['frames'].append(rec);selected.append(dict(raw_camera_track_id=t,**rec));col=np.array([((row['camera_actor_index']+1)*k)%220+30 for k in [71,43,103]]);overlay[m]=np.rint(.7*rgb[m]+.3*col).astype(np.uint8)
   ys,xs=np.where(m)
   if len(xs):texts.append((int(xs.mean()),max(0,int(ys.min())-22),'c'+str(row['camera_actor_index'])+(' ?' if not len(retained) else '')))
  Image.fromarray(labels).save(OUT/'labels'/(fid+'.png'));Image.fromarray((multiplicity>1).astype(np.uint8)*255).save(OUT/'unknown_overlap'/(fid+'.png'))
  im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
  for x,y,t in texts:d.text((x,y),t,font=font,fill='yellow',stroke_width=2,stroke_fill='black')
  panel=Image.new('RGB',(1920,2560));panel.paste(Image.fromarray(rgb),(0,0));panel.paste(im,(0,1280));panel.save(OUT/'frames'/(fid+'.jpg'),quality=88)
  frames.append(dict(frame_id=fid,full_RGB_retained=True,selected=selected,union_pixels=int(union.sum()),overlap_pixels=int((multiplicity>1).sum()),native_vehicle_pixels_not_in_selected_union=int((native.any(0)&~union).sum())))
  print(fid,flush=True)
 sys_path=str(E.parent);import sys;sys.path.insert(0,sys_path);from waymo_numpy_cpu import save_ply
 for t,v in tracks.items():
  v['mask_sources']=dict(v['mask_sources']);v['raw_LiDAR_alias_candidates']=votes[t].most_common();v['human_verdict']=None
  points=np.concatenate(clouds[t]) if clouds[t] else np.empty((0,3),np.float32);ts=np.concatenate(times[t]) if times[t] else np.empty(0,np.float32);a=OUT/f"actors/actor_{v['camera_actor_index']}"
  np.savez_compressed(a/'source_lidar_indices.npz',**sources[t]);save_ply(a/'points3d_dynamic.ply',points,ts)
 inventory=dict(status='CAMERA33_REAL_MASK_AND_SEED_CANDIDATES_NOT_TRAINING_READY',actual_camera_vehicle_track_ids=ids,frame_ids=[r['id'] for r in prep['frames']],frames=frames,vehicles=list(tracks.values()),candidate_initialization_identity_list_equals_mask_identity_list=True,future_stage2_identity_list=ids,unresolved_visible_vehicles=[t for t,v in tracks.items() if not v['total_real_seed_points']],all49_full_RGB_retained=True,GT_boxes_used_to_filter_individual_points=False,geometry_filter='unchanged2x2px/.5m_depth,.5m_dominant_component; rawGT physical-geometryQA/alias and full visual review still pending',ready=False,human_verdict=None,training_updates=0)
 (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2));print(json.dumps([(v['camera_actor_index'],v['total_real_seed_points']) for v in tracks.values()]))
if __name__=='__main__':main()
