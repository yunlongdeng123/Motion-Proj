"""九例均跑补景后冻结Ω；包括标记为未准入的输入诊断。"""
from nine_common import *
import os,time,numpy as np,torch,traceback
from PIL import Image
from evaluate import read_geometry
from delete_full_omega import render_depth,cross_view
assert read(ROOT/'drive_state.json')['state']=='complete'
assert not (ROOT/'omega_state.json').exists()
reg=read(ROOT/'registration.json');torch.set_num_threads(4);torch.manual_seed(42)
sys.path.insert(0,reg['fixed']['omega_source'])
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
state=dict(state='loading',pid=os.getpid(),completed=[],human_verdict=None);dump(ROOT/'omega_state.json',state)
with torch.device('meta'):model=VGGTOmega()
model.load_state_dict(torch.load(reg['fixed']['omega_checkpoint'],map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True);model.requires_grad_(False);model.eval().cuda();beg=time.time()
try:
    for s in reg['scenes']:
        base=ROOT/s['name'];frames=read(base/'camera_frames.json')
        for fr in frames:
            f=fr['frame'];out=base/'background_world'/f'{f:03}';out.mkdir(parents=True)
            paths=[str(base/f'cam{c}/background'/f'{f:05}.png') for c in range(6)]
            images=load_and_preprocess_images(paths,image_resolution=512,mode='balanced').cuda();assert tuple(images.shape[-2:])==(384,688)
            torch.cuda.reset_peak_memory_stats();start=time.time()
            with torch.inference_mode():raw=model(images);ex,k=encoding_to_camera(raw['pose_enc'],images.shape[-2:])
            pred={n:raw[n].float().cpu().numpy() for n in ['depth','depth_conf','pose_enc']};pred.update(extrinsics=ex.float().cpu().numpy(),intrinsics=k.float().cpu().numpy(),images=images.float().cpu().numpy());assert all(np.isfinite(v).all() for v in pred.values());np.savez_compressed(out/'prediction.npz',**pred)
            variants,cams,ks,lidar,rgb,align=read_geometry(fr,pred,out);points,colors,sources=variants['calibrated_control'];np.savez_compressed(out/'background_points.npz',points=points.astype('float32'),colors=colors,source_camera=sources,origin_world=fr['origin_world']);depth=pred['depth'][0,...,0]*align['calibrated_global_depth_scale'];np.savez_compressed(out/'metric_depth.npz',depth=depth,camera=cams,intrinsics=ks)
            masks=np.zeros((6,384,688),bool)
            for v in s['streams']:
                if v['active']:masks[v['camera']]=np.array(Image.open(base/f"cam{v['camera']}/write_mask"/f'{f:05}.png').resize((688,384),Image.Resampling.NEAREST))>0
            rows=[]
            for c in range(6):
                im,z=render_depth(points,colors,cams[c],ks[c],(384,688));hit=np.isfinite(z);hybrid=np.where(hit[...,None],im,rgb[c])
                Image.fromarray(im).save(out/f'geometry_cam{c}.png');Image.fromarray(hybrid).save(out/f'hybrid_cam{c}.png');np.save(out/f'z_cam{c}.npy',z)
                rows.append(dict(camera=c,geometry_coverage=float(hit.mean()),deletion_mask_geometry_coverage=float(hit[masks[c]].mean()) if masks[c].any() else None))
            summary=dict(scene=s['name'],frame=f,seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,views=rows,cross_view=cross_view(depth,cams,ks,masks),human_verdict=None)
            dump(out/'summary.json',summary);state['completed'].append(summary);state['state']='running';dump(ROOT/'omega_state.json',state);progress('omega',scene=s['name'],frame=f,completed=len(state['completed']),total=270)
            del raw,images,pred,variants,points,colors,sources;torch.cuda.empty_cache()
    state.update(state='complete',seconds=time.time()-beg);dump(ROOT/'omega_state.json',state);progress('omega',state='complete',frames=270)
except Exception as ex:
    state.update(state='failed_engineering',error=repr(ex),trace=traceback.format_exc());dump(ROOT/'omega_state.json',state);raise
