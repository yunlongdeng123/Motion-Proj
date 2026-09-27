"""BUILD: 完整六相机时序的冻结Ω背景；不在QUERY中运行网络。"""
import argparse,fcntl,json,pathlib,sys,time
import numpy as np,torch
from PIL import Image
from delete_full_common import ROOT,REPO,dump
sys.path.insert(0,str(REPO/'scripts/worldsim_v77'))
from evaluate import read_geometry
from geometry import transform,unproject

def render_depth(points,colors,c2w,k,hw,radius=1):
    h,w=hw;device='cuda'
    xyz=torch.as_tensor(np.ascontiguousarray(points),dtype=torch.float32,device=device)
    mat=torch.as_tensor(np.linalg.inv(c2w),dtype=torch.float32,device=device)
    cp=xyz@mat[:3,:3].T+mat[:3,3];q=cp@torch.as_tensor(k,dtype=torch.float32,device=device).T
    uv=torch.round(q[:,:2]/q[:,2:3].clamp(min=1e-6)).long()
    good=(cp[:,2]>.1)&(uv[:,0]>=-radius)&(uv[:,0]<w+radius)&(uv[:,1]>=-radius)&(uv[:,1]<h+radius)
    ids=torch.nonzero(good).flatten();uv=uv[good];z=cp[good,2];ii=[];zz=[];pp=[]
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            u=uv[:,0]+dx;v=uv[:,1]+dy;ok=(u>=0)&(u<w)&(v>=0)&(v<h)
            ii.append(v[ok]*w+u[ok]);zz.append(z[ok]);pp.append(ids[ok])
    ii=torch.cat(ii);zz=torch.cat(zz);pp=torch.cat(pp)
    depth=torch.full((h*w,),float('inf'),device=device);depth.scatter_reduce_(0,ii,zz,reduce='amin',include_self=True)
    near=zz==depth[ii];chosen=torch.full((h*w,),len(points),dtype=torch.long,device=device)
    chosen.scatter_reduce_(0,ii[near],pp[near],reduce='amin',include_self=True)
    palette=torch.cat([torch.as_tensor(colors,device=device),torch.tensor([[22,27,35]],dtype=torch.uint8,device=device)])
    return palette[chosen].reshape(h,w,3).cpu().numpy(),depth.reshape(h,w).cpu().numpy()

def cross_view(depth,cams,ks,masks):
    """删除区域的跨相机深度兼容性诊断；遮挡也会导致不兼容，不作为真值准确率。"""
    h,w=depth.shape[1:];rows=[]
    for c in range(6):
        grid=np.zeros((h,w),bool);grid[::8,::8]=True
        sel=grid&masks[c]&(depth[c]>.5)&(depth[c]<80)
        pts=unproject(depth[c],ks[c],cams[c])[sel]
        for other in range(6):
            if c==other or not len(pts):continue
            cp=transform(pts,np.linalg.inv(cams[other]));uv=cp@ks[other].T
            uv=np.rint(uv[:,:2]/np.maximum(uv[:,2:3],1e-6)).astype(int)
            ok=(cp[:,2]>.5)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            z=cp[ok,2];xy=uv[ok];ref=depth[other,xy[:,1],xy[:,0]]
            valid=np.isfinite(ref)&(ref>.5)&(ref<80);z=z[valid];ref=ref[valid]
            if len(z):
                rel=np.abs(z-ref)/np.maximum(z,ref)
                rows.append({'source_camera':c,'other_camera':other,'count':len(z),'median_relative_depth_difference':float(np.median(rel)),'over_10pct_count':int((rel>.1).sum()),'source_behind_other_count':int((z>ref*1.1).sum())})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args()
    lock=open(ROOT/'omega.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'drive_state.json').read_text())['state']=='complete','DriveEditor须完成并释放GPU'
    reg=json.loads((ROOT/'registration.json').read_text());torch.set_num_threads(4);torch.manual_seed(42)
    sys.path.insert(0,reg['omega_source'])
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    state={'state':'loading','completed_frames':[],'expected_frames':sum(s['count'] for s in reg['scenes']),'training_steps':0,'human_verdict':None};dump(ROOT/'omega_state.json',state)
    with torch.device('meta'):model=VGGTOmega()
    model.load_state_dict(torch.load(reg['omega_checkpoint'],map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True)
    model.requires_grad_(False);model.eval().cuda();start=time.monotonic();n=0
    try:
        for s in reg['scenes']:
            name=s['name'];base=ROOT/name;frames=json.loads((base/'camera_frames.json').read_text());spec=json.loads((base/'spec.json').read_text())
            for fr in frames:
                f=fr['frame'];out=base/'background_world'/f'{f:03}';out.mkdir(parents=True,exist_ok=True)
                if (out/'summary.json').exists():state['completed_frames'].append({'scene':name,'frame':f});continue
                assert time.monotonic()-start<3600,'Ω有界预算'
                state.update(state='running',current={'scene':name,'frame':f});dump(ROOT/'omega_state.json',state)
                paths=[str(base/f'cam{c}/background/{f:05}.png') for c in range(6)]
                images=load_and_preprocess_images(paths,image_resolution=512,mode='balanced').cuda()
                assert tuple(images.shape[-2:])==(384,688)
                t=time.monotonic();torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():raw=model(images);ex,k=encoding_to_camera(raw['pose_enc'],images.shape[-2:])
                pred={v:raw[v].float().cpu().numpy() for v in ['depth','depth_conf','pose_enc']};pred.update(extrinsics=ex.float().cpu().numpy(),intrinsics=k.float().cpu().numpy(),images=images.float().cpu().numpy())
                assert all(np.isfinite(v).all() for v in pred.values())
                np.savez_compressed(out/'prediction.npz',**pred)
                scene=dict(fr,root=spec['root']);variants,cams,ks,lidar,bg,alignment=read_geometry(scene,pred,out)
                points,colors,sources=variants['calibrated_control'];metric=pred['depth'][0,...,0]*alignment['calibrated_global_depth_scale']
                np.savez_compressed(out/'background_points.npz',points=points.astype('float32'),colors=colors,source_camera=sources.astype('uint8'),origin_world=np.array(fr['origin_world']))
                np.savez_compressed(out/'metric_depth.npz',depth=metric.astype('float32'),camera=cams,intrinsics=ks)
                masks=np.stack([np.array(Image.open(base/f'cam{c}/mask/{f:05}.png').resize((688,384),Image.Resampling.NEAREST))>0 for c in range(6)])
                rows=[]
                for c in range(6):
                    rgb,z=render_depth(points,colors,cams[c],ks[c],(384,688));coverage=np.isfinite(z)
                    Image.fromarray(rgb).save(out/f'geometry_cam{c}.png');Image.fromarray(np.where(coverage[...,None],rgb,bg[c])).save(out/f'hybrid_cam{c}.png');np.save(out/f'z_cam{c}.npy',z)
                    rows.append({'camera':c,'geometry_coverage':float(coverage.mean()),'deletion_mask_pixels':int(masks[c].sum()),'deletion_mask_geometry_coverage':float(coverage[masks[c]].mean()) if masks[c].any() else None})
                summary={'scene':name,'frame':f,'point_count':len(points),'elapsed_s':time.monotonic()-t,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'views':rows,'cross_view':cross_view(metric,cams,ks,masks),'background_scope':'B_t in local frame origin; all other actors remain baked context','human_verdict':None}
                dump(out/'summary.json',summary);state['completed_frames'].append({'scene':name,'frame':f});dump(ROOT/'omega_state.json',state);print(json.dumps({k:summary[k] for k in ['scene','frame','elapsed_s','peak_gpu_gib']}),flush=True)
                del raw,images,pred,variants,points,colors,sources,lidar,bg;torch.cuda.empty_cache();n+=1
                if a.limit and n>=a.limit:state['state']='partial';dump(ROOT/'omega_state.json',state);return
        state.update(state='complete',elapsed_s=time.monotonic()-start);dump(ROOT/'omega_state.json',state)
    except Exception as exc:state.update(state='failed',error=repr(exc));dump(ROOT/'omega_state.json',state);raise

if __name__=='__main__':main()
