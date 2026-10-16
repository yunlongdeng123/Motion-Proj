"""CPU-only, real RGB/pose preflight. Exports candidates; does not create shadow GT.

No checkpoint loading, optimizer, GPU, source mutation or queue. Local road depth
is a fitted-plane hypothesis, not measured per-pixel surface truth.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, json, datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def read(p): return json.loads(Path(p).read_text())
def sample(image, uv):
    h,w=image.shape[:2]
    x,y=uv.T; x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    valid=(x>=0)&(y>=0)&(x<w-1)&(y<h-1)
    x0=x0.clip(0,w-2); y0=y0.clip(0,h-2)
    dx=(x-x0)[:,None]; dy=(y-y0)[:,None]
    values=(image[y0,x0]*(1-dx)*(1-dy)+image[y0,x0+1]*dx*(1-dy)
            +image[y0+1,x0]*(1-dx)*dy+image[y0+1,x0+1]*dx*dy)
    return values,valid
def project(xyz,cam):
    c=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cam['full_proj_transform'])
    ndc=c[:,:3]/c[:,3:4]
    uv=(ndc[:,:2]+1)*np.array([cam['width'],cam['height']])/2-.5
    return uv,c[:,3]>0
def rays_to_plane(pixels,cam,plane):
    ndc=np.c_[(2*pixels[:,0]+1)/cam['width']-1,(2*pixels[:,1]+1)/cam['height']-1,
              np.full(len(pixels),.5),np.ones(len(pixels))]
    pts=ndc@np.linalg.inv(np.asarray(cam['full_proj_transform']))
    pts=pts[:,:3]/pts[:,3:4]; origin=np.asarray(cam['camera_center'])
    direction=pts-origin; distance=-(origin@plane[:3]+plane[3])/(direction@plane[:3])
    assert np.all(distance>0)
    return origin+distance[:,None]*direction
def tile(image,title,size=(452,148)):
    pic=Image.fromarray((image.clip(0,1)*255).astype('uint8')).resize(size)
    out=Image.new('RGB',(size[0],size[1]+32),'white'); out.paste(pic,(0,32))
    ImageDraw.Draw(out).text((6,7),title,fill='black');return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
    p.add_argument('--output',required=True); a=p.parse_args()
    root=Path(a.root); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    scene='scene0008_089_137'; ev=root/'rigid_post30k'/scene/'evaluation'
    m=read(ev/'scene_evaluation_manifest.json'); frames=m['streams']['observed']['frames']
    target=next(r for r in frames if r['frame_id']=='0008101'); cam=target['camera']
    plane=np.asarray(read(root/'metadata'/scene/'CPU_readiness.json')['SGCM_geometry']['plane_model'])
    diagnostic=read(root/'black_shadow_audit_20261016_v1/focus_cpu/focus_attribution.json')
    x0,y0,x1,y1=50,228,163,265
    yy,xx=np.mgrid[y0:y1,x0:x1]; pixels=np.c_[xx.ravel(),yy.ravel()]
    xyz=rays_to_plane(pixels,cam,plane)
    uv,front=project(xyz,cam); repro=float(np.abs(uv-pixels).max());assert repro<1e-4
    im=Image.new('L',(480,320));ImageDraw.Draw(im).polygon([tuple(x) for x in diagnostic['polygon']],fill=255)
    footprint=np.asarray(im)[y0:y1,x0:x1]>0
    samples=[]; valids=[]; mask_hits=[]; rows=[]
    selected=['0008089','0008095','0008099','0008101','0008107','0008113','0008125','0008137']
    canvas=Image.new('RGB',(452*2,180*4),'#dddddd')
    for i,row in enumerate(frames):
        with np.load(ev/row['npz']) as f:
            gt=np.moveaxis(f['gt'],0,-1)
            bodies=f['observed_masks'].any(0)
        q,fr=project(xyz,row['camera']); rgb,v=sample(gt,q)
        ix=np.rint(q[:,0]).astype(int).clip(0,cam['width']-1)
        iy=np.rint(q[:,1]).astype(int).clip(0,cam['height']-1)
        # Conservative guard around known moving-body boundaries, not all occluders.
        guarded=np.zeros_like(bodies)
        for dy in range(-2,3):
            for dx in range(-2,3):
                ys=slice(max(0,dy),min(320,320+dy));xs=slice(max(0,dx),min(480,480+dx))
                guarded[ys,xs]|=bodies[slice(max(0,-dy),min(320,320-dy)),slice(max(0,-dx),min(480,480-dx))]
        hit=guarded[iy,ix]; valid=v&fr&~hit
        samples.append(rgb);valids.append(valid);mask_hits.append(hit)
        rows.append({'frame_id':row['frame_id'],'physical_timestamp':row['physical_timestamp'],
            'candidate_body_free_in_footprint':int(valid[footprint.ravel()].sum()),
            'mean_RGB_footprint_no_semantic_filter':rgb[footprint.ravel()].mean(0).tolist(),
            'unselected_occluders_and_shadows_verified':False})
        if row['frame_id'] in selected:
            j=selected.index(row['frame_id']); patch=rgb.reshape(y1-y0,x1-x0,3)
            canvas.paste(tile(patch,row['frame_id']+' real RGB warped by plane'),((j%2)*452,(j//2)*180))
    samples=np.asarray(samples);valids=np.asarray(valids)
    # Two independently visible pre-arrival road observations are only candidates.
    # Frame 8113 is kept out of the candidate atlas as an agreement check.
    train_ids=['0008089','0008095'];check_id='0008113'
    lookup={r['frame_id']:i for i,r in enumerate(frames)}
    fit=np.array([lookup[k] for k in train_ids]);check=lookup[check_id]
    good=valids[fit].all(0)&valids[check]&footprint.ravel()
    atlas=np.median(samples[fit],axis=0)
    # sRGB values retained for inspection only. B/C fitting must linearize first.
    delta=np.abs(atlas-samples[check])
    target_delta=np.abs(atlas-samples[lookup['0008101']])
    comparison=Image.new('RGB',(452*2,180*2),'white')
    for k,(value,label) in enumerate([(samples[lookup['0008101']],'8101: observed RGB with vehicle-associated dark region'),
                     (atlas,'Candidate road: median of 8089 / 8095; not clean GT'),
                     (samples[check],'8113: independent-time real RGB agreement check'),
                     (np.repeat(good.reshape(-1,1),3,axis=1).astype(float),'White: in-view, selected-body-free candidate support')]):
        comparison.paste(tile(value.reshape(y1-y0,x1-x0,3),label),((k%2)*452,(k//2)*180))
    canvas.save(out/'real_warped_candidates.png');comparison.save(out/'candidate_atlas_check.png')
    np.savez_compressed(out/'road_observations.npz',frame_ids=np.array([r['frame_id'] for r in frames]),
        target_pixels=pixels,target_world_model_xyz=xyz,srgb=samples,candidate_body_free=valids,
        diagnostic_footprint=footprint.ravel(),candidate_atlas_srgb=atlas,candidate_support=good)
    cases=[]
    for sid in ['scene0008_089_137','scene0017_079_127']:
        spec=read(root/'rigid_specs'/f'{sid}.json')
        with np.load(Path(spec['prepared'])/'instances.npz') as f:
            scale=float(f['world_scale']);poses=f['poses']; valid=f['valid'];ids=f['instance_ids']
            actors=[]
            for k,track in enumerate(ids):
                xyz_track=poses[valid[:,k],k,:3,3]
                distance=float(np.linalg.norm(np.diff(xyz_track,axis=0),axis=1).sum()/scale)
                displacement=float(np.linalg.norm(xyz_track[-1]-xyz_track[0])/scale)
                actors.append({'actor_index':k,'track_id':str(track),'valid_poses':int(valid[:,k].sum()),
                  'path_length_m':distance,'net_displacement_m':displacement,
                  'median_box_size_model':np.median(f['box_size'][valid[:,k],k],axis=0).tolist()})
            cases.append({'scene_id':sid,'physical_span_seconds':float(np.ptp(f['physical_timestamps'])),
                'world_scale':scale,'all_selected_body_masks_known':bool(f['mask_known'].all()),'actors':actors,
                'evaluation_scope':'checkpoint trained on all49; reconstruction pilot only'})
    result={'schema':'shadow_BC_E0_RGB_preflight_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'GPU_started':False,'training_updates':0,'checkpoint_loaded':False,'source_files_modified':False,
      'coordinate_roundtrip_max_pixels':repro,'diagnostic_footprint_pixels':int(footprint.sum()),
      'candidate_source_frames':train_ids,'candidate_check_frame':check_id,'candidate_valid_pixels':int(good.sum()),
      'candidate_fraction':float(good.sum()/footprint.sum()),
      'candidate_atlas_vs_check_sRGB_MAE':float(delta[good].mean()),
      'candidate_atlas_vs_target_sRGB_MAE':float(target_delta[good].mean()),
      'not_shadow_GT':True,'not_clean_GT':True,'no_exposure_or_white_balance_correction_applied':True,
      'limitations':['Fitted plane correspondence; no new local LiDAR depth check',
         'Body-free only checks selected moving actors, not all scene occluders',
         'Dense road/shadow/unknown labels need review before supervision',
         'Candidate atlas agreement is preparation evidence, not reconstruction or editing score'],
      'frames':rows,'cases':cases}
    (out/'E0_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='frames'}))

if __name__=='__main__':main()
