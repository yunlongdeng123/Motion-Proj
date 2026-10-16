"""Refine conservative projected-box guards using existing DEVA/SAM segments.

No segmentation inference, installation or target RGB editing. Unmatched boxes
stay conservatively protected. Large road segments cannot match by containment.
"""
import os,sys,json,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,binary_fill_holes,binary_closing
from observation_guards import project_guard
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');NEW=ROOT/'shadow_sequence_20261016_v2';OLD=ROOT/'shadow_BC_20261016_v1'
def surface_proxy(mask,corners,cam):
    y,x=np.nonzero(mask);pixels=np.c_[x,y];corners=np.asarray(corners);origin=np.array(cam['camera_center'])
    if not len(x):return np.empty((0,3),np.float32)
    ndc=np.c_[(2*pixels[:,0]+1)/480-1,(2*pixels[:,1]+1)/320-1,np.full(len(x),.5),np.ones(len(x))]
    points=ndc@np.linalg.inv(np.array(cam['full_proj_transform']));direction=points[:,:3]/points[:,3:4]-origin
    center=corners.mean(0);edges=np.stack([corners[4]-corners[0],corners[2]-corners[0],corners[1]-corners[0]],1)
    size=np.linalg.norm(edges,axis=0);rot=edges/size[None];o=(origin-center)@rot;dv=direction@rot
    safe=np.where(abs(dv)>1e-9,dv,1e-9);t1=(-size/2-o)/safe;t2=(size/2-o)/safe
    near=np.minimum(t1,t2).max(1);far=np.maximum(t1,t2).min(1)
    fallback=((center-origin)*direction).sum(1)/np.maximum((direction**2).sum(1),1e-9)
    t=np.where((far>=near)&(near>0),near,fallback)
    return (origin+direction*t[:,None]).astype('float32')
def main():
    out=NEW/'guard_refinement_v2';out.mkdir(exist_ok=False);start=time.time()
    spec=json.loads((ROOT/'rigid_specs/scene0008_089_137.json').read_text());src=Path(spec['source_scene'])
    ev=ROOT/'rigid_post30k/scene0008_089_137/evaluation';frames=json.loads((ev/'scene_evaluation_manifest.json').read_text())['streams']['observed']['frames']
    boxes=json.loads((NEW/'surface_guarded_data/protection_boxes.json').read_text())
    with np.load(NEW/'surface_guarded_data/surface.npz') as f:d={k:f[k] for k in f.files}
    with np.load(OLD/'training_data_v2/training_data.npz') as f:poses=f['poses'];sizes=f['sizes_model'];posevalid=f['pose_valid']
    guards=[];bodies=[];rows=[];clouds={};fallback_boxes=[]
    for i,row in enumerate(frames):
        maskpath=src/'mask/mask_0/Annotations'/(row['frame_id']+'.png')
        maskim=Image.open(maskpath).resize((480,320),Image.Resampling.NEAREST);labels=np.asarray(maskim)
        if labels.ndim==3:labels=labels[:,:,0].astype(np.int64)+256*labels[:,:,1].astype(np.int64)+65536*labels[:,:,2].astype(np.int64)
        ids,index=np.unique(labels,return_inverse=True);index=index.reshape(320,480)
        area=np.bincount(index.ravel(),minlength=len(ids))
        with np.load(ev/row['npz']) as f:body=f['observed_masks'].any(0);rgb=np.moveaxis(f['gt'],0,-1)
        selected_overlap=np.bincount(index[body],minlength=len(ids))/np.maximum(area,1)
        protected=np.zeros((320,480),bool);matched=0;unmatched=0;debug=[];cloud=[];fallback=[]
        for j,corners in enumerate(boxes[i]):
            bbox=project_guard([corners],row['camera']);barea=bbox.sum()
            if not barea:continue
            overlap=np.bincount(index[bbox],minlength=len(ids));inside=overlap/np.maximum(area,1);iou=overlap/np.maximum(area+barea-overlap,1)
            good=(inside>.70)&(iou>.15)&(selected_overlap<.20)&(area>=8)
            if good.any():
                score=np.where(good,iou,0);best=score.max();keep=good&(iou>best*.40)
                mask=np.isin(index,np.flatnonzero(keep));mask=binary_fill_holes(binary_closing(mask,iterations=1))
                protected|=binary_dilation(mask,iterations=1);matched+=1;method='SAM_segments'
                proxy_mask=binary_dilation(mask,iterations=1)&~binary_dilation(body,iterations=1)
                cloud.append(surface_proxy(proxy_mask,corners,row['camera']))
            else:protected|=binary_dilation(bbox,iterations=1);fallback.append(corners);unmatched+=1;method='conservative_box_fallback'
            if i==0 and bbox[202,178]:debug.append({'box_index':j,'bbox_area':int(barea),'method':method,'best_iou':float(np.max(np.where(inside>.70,iou,0))),'best_segment':int(ids[int(np.argmax(np.where(inside>.70,iou,0)))])})
        protected&=~binary_dilation(body,iterations=1)
        guards.append(protected);bodies.append(binary_dilation(body,iterations=3));fallback_boxes.append(fallback)
        clouds['points_%02d'%i]=np.concatenate(cloud) if cloud else np.empty((0,3),np.float32)
        rows.append({'frame':row['frame_id'],'matched_boxes':matched,'fallback_boxes':unmatched,'old_guard_pixels':int(d['protection'][i].sum()),'new_guard_pixels':int(protected.sum())})
        if i in [0,12,48]:
            overlay=rgb.copy();overlay[protected]=overlay[protected]*.45+np.array([.50,.05,.02])
            Image.fromarray(np.rint(overlay.clip(0,1)*255).astype('uint8')).save(out/('guard_review_'+row['frame_id']+'.png'))
        if i==0:(out/'early_shadow_box_matches.json').write_text(json.dumps(debug,indent=2))
    guards=np.array(guards);bodies=np.array(bodies);free=[];near=[];scale=float(d['world_scale'])
    for i in range(49):
        uv=d['uv'][i];x=np.rint(uv[:,0]).astype(int).clip(0,479);y=np.rint(uv[:,1]).astype(int).clip(0,319)
        valid=(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)&~guards[i,y,x]&~bodies[i,y,x]&d['support']
        free.append(valid)
        local=np.einsum('kni,kij->knj',d['xyz'][None]-poses[i,:,:3,3][:,None],poses[i,:,:3,:3])/scale
        distance=np.linalg.norm(np.maximum(abs(local[:,:,:2])-sizes[i,:,None,:2]/scale/2,0),axis=-1);distance[~posevalid[i]]=np.inf;near.append(distance.min(0))
    free=np.array(free);near=np.array(near);ref=d['reference'];obs=d['observed']
    clean=free&(abs(obs-ref[None]).max(2)<.045)
    luma=np.array([.2126,.7152,.0722]);ratio=(obs@luma)/np.maximum(ref@luma,.01)[None]
    chroma=np.std(np.log(np.maximum(obs/np.maximum(ref[None],.015),.02)),axis=2)
    shape=d['shape'];lum=(ref@luma).reshape(int(shape[1]),int(shape[0]));edge=np.zeros_like(lum,bool)
    edge[:,1:]|=abs(np.diff(lum,axis=1))>.07;edge[1:]|=abs(np.diff(lum,axis=0))>.07;edge=binary_dilation(edge,iterations=1).ravel()
    shadow=free&(ratio<.70)&(chroma<.55)&(near<4)&~edge[None]&~clean
    np.savez_compressed(out/'overrides.npz',protection=guards,free=free,clean=clean,shadow=shadow)
    np.savez_compressed(out/'foreground_proxies.npz',**clouds)
    (out/'fallback_boxes.json').write_text(json.dumps(fallback_boxes))
    summary={'existing_segmentation_source':'cached DEVA+SAM automatic full-scene masks; category-free segment matching to unselected 3D boxes',
        'thresholds':'segment mass inside3Dprojection>70%;IoU>15%; selected-body overlap<20%; unmatched boxes retain protection',
        'selected_actor_body_excluded_from_guards':True,'early_black_pixel_178_202_protected_old':bool(d['protection'][0,202,178]),
        'early_black_pixel_178_202_protected_new':bool(guards[0,202,178]),'clean_samples':int(clean.sum()),'shadow_pseudo_samples':int(shadow.sum()),'per_frame':rows,'elapsed_seconds':time.time()-start}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='per_frame'}))
if __name__=='__main__':main()
