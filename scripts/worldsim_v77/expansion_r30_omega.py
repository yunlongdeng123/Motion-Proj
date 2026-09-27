"""只把r30已保存首10帧接回冻结Omega；保留六相机和纯几何对照。"""
from pathlib import Path
import sys,json,time,datetime,os,fcntl
import numpy as np,torch
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from geometry import project_bbox
from video_review import scene_frame
from evaluate import read_geometry
from delete_full_omega import render_depth,cross_view
BASE=Path('/root/autodl-tmp/runs/worldsim_v77');ROOT=BASE/'WS-V77-EXPAND-EDIT-20260928/r2';HYBRID=BASE/'WS-V77-HYBRID-BG-20260927';torch.set_num_threads(4)

def main():
    assert not ROOT.exists();ROOT.mkdir();old=read(BASE/'WS-V77-P0-24ACTOR-20260926/r1/registration.json');spec=next(s for s in old['scenes'] if s['name']=='official_000');inst=read(Path(spec['root'])/'instances/instances_info.json');frames=[scene_frame(spec,f,inst) for f in range(10)];dump(ROOT/'frames.json',frames)
    projections=[]
    for fr in frames:
        b=next(b for b in fr['all_boxes'] if b['actor_id']=='12')
        row=[]
        for c in range(6):
            cam,k=camera(fr,c);bb=project_bbox(b['pose'],b['size_lwh'],cam,k,(576,1024));area=(bb[2]-bb[0])*(bb[3]-bb[1]) if bb else 0;row.append(area)
        projections.append(row)
    assert max(max(r[1:]) for r in projections)==0,projections
    dump(ROOT/'registration.json',dict(task_id='WS-V77-EXPAND-EDIT-20260928',run_id='r2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',frames=list(range(10)),input='CAM0 saved r30 rect_feather_keep; other cameras original RGB after target frustum audit',target_projected_areas=projections,omega_source=old['omega_source'],checkpoint=old['checkpoint'],frozen=True,training_steps=0,input_roles='GT cameras/boxes and background LiDAR global scale retained; B_t independent per timestamp, not persistent world. Other actors remain baked.',user_scope='r30 transfer geometry validation, not claim that prior short video acceptance certifies reconstruction',human_verdict=None,failure_ledger_refs=['V77-F02']))
    sys.path.insert(0,old['omega_source']);from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    state=dict(state='loading',pid=os.getpid(),completed=[],human_verdict=None);dump(ROOT/'state.json',state)
    with torch.device('meta'):model=VGGTOmega()
    model.load_state_dict(torch.load(old['checkpoint'],map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True);model.requires_grad_(False);model.eval().cuda();torch.manual_seed(42);beg=time.time()
    for fr in frames:
        f=fr['frame'];out=ROOT/f'f{f:03}';out.mkdir();paths=[]
        for c in range(6):
            source=HYBRID/'r30/rect_feather_keep'/f'{f:05}.png' if c==0 else Path(spec['root'])/'images'/f'{f:03}_{c}.jpg'
            im=Image.open(source).convert('RGB').resize((1024,576),Image.Resampling.BILINEAR);p=out/f'input_cam{c}.png';im.save(p);paths.append(str(p))
        images=load_and_preprocess_images(paths,image_resolution=512,mode='balanced').cuda();assert tuple(images.shape[-2:])==(384,688);torch.cuda.reset_peak_memory_stats();start=time.time()
        with torch.inference_mode():raw=model(images);ex,k=encoding_to_camera(raw['pose_enc'],images.shape[-2:])
        pred={n:raw[n].float().cpu().numpy() for n in ['depth','depth_conf','pose_enc']};pred.update(extrinsics=ex.float().cpu().numpy(),intrinsics=k.float().cpu().numpy(),images=images.float().cpu().numpy());assert all(np.isfinite(v).all() for v in pred.values());np.savez_compressed(out/'prediction.npz',**pred)
        variants,cams,ks,lidar,rgb,align=read_geometry(fr,pred,out);points,colors,sources=variants['calibrated_control'];np.savez_compressed(out/'background_points.npz',points=points.astype('float32'),colors=colors,source_camera=sources,origin_world=fr['origin_world']);depth=pred['depth'][0,...,0]*align['calibrated_global_depth_scale'];np.savez_compressed(out/'metric_depth.npz',depth=depth,camera=cams,intrinsics=ks)
        masks=np.zeros((6,384,688),bool);masks[0]=np.array(Image.open(HYBRID/'r28/write_mask'/f'{f:05}.png').resize((688,384),Image.Resampling.NEAREST))>0;rows=[]
        for c in range(6):
            im,z=render_depth(points,colors,cams[c],ks[c],(384,688));hit=np.isfinite(z);hybrid=np.where(hit[...,None],im,rgb[c]);Image.fromarray(im).save(out/f'geometry_cam{c}.png');Image.fromarray(hybrid).save(out/f'hybrid_cam{c}.png');np.save(out/f'z_cam{c}.npy',z);rows.append(dict(camera=c,coverage=float(hit.mean()),mask_coverage=float(hit[masks[c]].mean()) if masks[c].any() else None))
        summary=dict(frame=f,seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,views=rows,cross_view=cross_view(depth,cams,ks,masks),human_verdict=None);dump(out/'summary.json',summary);state.update(state='running');state['completed'].append(summary);dump(ROOT/'state.json',state);print('OMEGA_DONE',f,summary['seconds'],flush=True);del raw,images,pred,variants,points,colors,sources;torch.cuda.empty_cache()
    state.update(state='complete_pending_visual_review',seconds=time.time()-beg);dump(ROOT/'state.json',state)
if __name__=='__main__':main()
