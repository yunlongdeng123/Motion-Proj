"""Scene admission plumbing: real DEVA association, unchanged LiDAR actor recipe.

Track boxes propose an edit target; they never become segmentation or point masks.
The review flag is written separately after inspecting every contact-sheet tile.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
ACTOR=Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/tools/red_actor_adapter/actor_cpu.py')

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))

def propose(context):
    scene=ROOT/'motion_scene_inputs_v1'/context
    assert read(scene/'mask/mask_0/preparation_receipt.json')['frames']==49
    sensor=read(scene/'sensor_preparation.json')
    allraw=read(ROOT/'evaluation_set_v1/scenes'/context/'scene.json')['raw_annotations']
    proposals=read(ROOT/'evaluation_admission_v2/operation_window_proposals.json')
    entries=proposals['proposals']
    if isinstance(entries,dict):entry=entries[context]
    else:entry=next(x for x in entries if x['context']==context)
    target=entry['current_window_moving_target']
    track=target['target_track']
    assert track, target
    dest=scene/'native_preparation';dest.mkdir(exist_ok=True)
    frames=[];tiles=[]
    for index,row in enumerate(sensor['frames']):
        name=row['frame_id']; raw=allraw[row['raw']]
        obj=next(x for x in raw['front_vehicles'] if x['track']==track)
        x,y,w,h=obj['xywh'];box=np.array([x-w/2,y-h/2,x+w/2,y+h/2])
        mask=np.array(Image.open(scene/'mask/mask_0/Annotations'/f'{name}.png'))
        rgb=np.array(Image.open(scene/'image/image_0'/f'{name}.png').convert('RGB'))
        x0,y0,x1,y1=np.maximum(0,np.floor(box).astype(int));x1=min(x1,mask.shape[1]);y1=min(y1,mask.shape[0])
        counts=np.bincount(mask.ravel(),minlength=256);over=np.bincount(mask[y0:y1,x0:x1].ravel(),minlength=256)
        candidates=[]
        for instance in np.flatnonzero(over):
            if not instance or counts[instance]<20:continue
            yy,xx=np.where(mask==instance); bounds=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
            inter=max(0,min(x1,bounds[2])-max(x0,bounds[0]))*max(0,min(y1,bounds[3])-max(y0,bounds[1]))
            iou=inter/((x1-x0)*(y1-y0)+(bounds[2]-bounds[0])*(bounds[3]-bounds[1])-inter)
            containment=int(over[instance])/int(counts[instance])
            candidates.append(dict(instance_id=int(instance),score=.65*iou+.35*containment,iou=iou,containment=containment,pixels=int(counts[instance]),bbox=bounds))
        candidates.sort(key=lambda x:x['score'],reverse=True);assert candidates,name
        best=candidates[0];body=mask==best['instance_id']
        frames.append(dict(frame_id=name,raw=row['raw'],model_time=index*.02-.48,instance_ids=[best['instance_id']],visibility='visible',reviewed=False,box=box.tolist(),candidates=candidates[:3]))
        # Crop includes immediate context; all49 frames are visible in three sheets.
        xa,ya=max(0,x0-80),max(0,y0-80);xb,yb=min(mask.shape[1],x1+80),min(mask.shape[0],y1+80)
        overlay=rgb.copy();overlay[body]=(overlay[body]*.55+np.array([30,240,100])*.45).astype('uint8')
        tile=Image.new('RGB',(320,210),(20,25,30));pic=Image.fromarray(overlay[ya:yb,xa:xb]);pic.thumbnail((320,185));tile.paste(pic,((320-pic.width)//2,0))
        ImageDraw.Draw(tile).text((5,188),f'{name}: ID{best["instance_id"]} IoU{best["iou"]:.2f}',fill='white');tiles.append(tile)
    for offset in range(0,49,20):
        sheet=Image.new('RGB',(1600,840),(12,15,20))
        for j,tile in enumerate(tiles[offset:offset+20]):sheet.paste(tile,((j%5)*320,(j//5)*210))
        sheet.save(dest/f'actor_review_{offset//20}.jpg',quality=94)
    write(dest/'selection_proposal.json',dict(context=context,scene=str(scene),track=track,status='proposal_needs_agent_visual_review',frames=frames,target_stats=target,GT_boxes_used_for_point_filtering=False,box_use='target identity proposal only; actual body from original DEVA pixels'))
    print(json.dumps(dict(context=context,track=track,scores=[min(x['candidates'][0]['score'] for x in frames),max(x['candidates'][0]['score'] for x in frames)],instances=sorted(set(x['instance_ids'][0] for x in frames)))))

def build(context):
    scene=ROOT/'motion_scene_inputs_v1'/context;dest=scene/'native_preparation'
    selection=read(dest/'selection_reviewed.json');assert selection['status']=='agent_visually_reviewed'
    assert len(selection['frames'])==49 and all(x['reviewed'] for x in selection['frames'])
    spec=importlib.util.spec_from_file_location('original_actor_recipe',ACTOR);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for d in ['dynamic_image/dynamic_image_0','dynamic_masks/dynamic_mask_0','dynamic_actor_assets/actor_0/body_masks']:(scene/d).mkdir(parents=True,exist_ok=False)
    xyzs=[];times=[];static=[];statict=[];sourceids={};audit=[]
    for index,row in enumerate(selection['frames']):
        name=row['frame_id'];rgb=np.array(Image.open(scene/'image/image_0'/f'{name}.png').convert('RGB'))
        mask=np.array(Image.open(scene/'mask/mask_0/Annotations'/f'{name}.png'))
        lidar=np.fromfile(scene/'velodyne'/f'{name}.bin',dtype=np.float32).reshape(-1,6)
        pose=np.loadtxt(scene/'pose'/f'{name}.txt');calib=a.calibration(scene/'calib'/f'{name}.txt')
        projection=a.project(lidar,mask,calib)
        body,indices,world,projected,before=a.actor_subset(mask,row['instance_ids'],'visible',projection,pose,name,0)
        xyzs.append(world);times.append(np.full(len(world),row['model_time'],dtype=np.float32));sourceids[name]=indices
        allworld=(np.column_stack((lidar[:,:3].astype(np.float64),np.ones(len(lidar))))@pose.T)[:,:3].astype(np.float32)
        static.append(allworld);statict.append(np.full(len(allworld),row['model_time'],dtype=np.float32))
        output=np.zeros_like(rgb);output[body]=rgb[body]
        Image.fromarray(output).save(scene/'dynamic_image/dynamic_image_0'/f'{name}.png')
        for d in ['dynamic_masks/dynamic_mask_0','dynamic_actor_assets/actor_0/body_masks']:Image.fromarray(body.astype('uint8')*255).save(scene/d/f'{name}.png')
        audit.append(dict(frame_id=name,model_time=row['model_time'],retained_real_points=len(world),projected_points=projected,before_component=before,body_pixels=int(body.sum())))
    all_static=np.concatenate(static);all_times=np.concatenate(statict)
    selected=np.random.RandomState(0).choice(len(all_static),600000,replace=True)
    a.save_ply(scene/'points3d.ply',all_static[selected],all_times[selected])
    a.save_ply(scene/'points3d_dynamic_0.ply',np.concatenate(xyzs),np.concatenate(times))
    np.savez_compressed(scene/'dynamic_actor_assets/actor_0/source_lidar_indices.npz',**sourceids)
    # Native paper mask loader reads dynamic_masks preferentially; keep original
    # categorical DEVA untouched for provenance and future additional actors.
    sensor=read(scene/'sensor_preparation.json')
    manifest=dict(scene_id=context,training_ready=False,status='native_actor_built_sky_and_loader_pending',raw_training_range=[sensor['frames'][0]['raw'],sensor['frames'][-1]['raw']],frames=[dict(id=x['frame_id'],source_index=x['raw'],model_time=i*.02-.48) for i,x in enumerate(sensor['frames'])],model_frame_interval=.02)
    write(scene/'preparation_manifest.json',manifest)
    write(scene/'dynamic_multi_actor_preparation.json',dict(status='built_pending_native_loader',scene=str(scene),frames=49,actors=[dict(actor_index=0,raw_track_id=selection['track'],frames=audit,total_real_points=sum(len(x) for x in xyzs))],geometry='unchanged frozen dual_actor projection,2x2 nearest-depth tolerance0.5m,dominant component0.5m',gt_3d_boxes_used_to_filter_training_points=False,static_exclusion='unchanged loader nearest dynamic <0.1m',training_ready=False,selection=str(dest/'selection_reviewed.json')))
    write(dest/'actor_build_returned.json',dict(frames=49,static_source_points=sum(len(x) for x in static),static_seed_points=600000,static_seed_sampling='inherited RandomState(0).choice(all49 real world LiDAR,600000,replace=True)',dynamic_real_points=sum(len(x) for x in xyzs),finite=True,scene=str(scene),training_ready=False))
    print((dest/'actor_build_returned.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['propose','build']);p.add_argument('--context',required=True);args=p.parse_args();globals()[args.action](args.context)
