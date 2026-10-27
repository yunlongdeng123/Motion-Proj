"""Per-row alpha*T-weighted actual target-return depth compatibility.

Three adjoints give sum(w), sum(w*z_return), sum(w*z_return^2). Comparing
Gaussian center depth to this distribution tests support at the same observed
pixels, rather than treating every background contribution behind a car as leak.
This remains sparse evidence, not dense semantic/geometry GT.
"""
import sys,json,time,os,gc
from pathlib import Path
import numpy as np
from PIL import Image
from segmentation import OLD,ROOT,CONTEXTS,write
sys.path.insert(0,str(OLD/'code'))
from render_motion_suite import NativeMotion,prepare_rasterization,helpers,torch,cpu
from gpu_slot import exclusive

def main():
    dest=ROOT/'depth_support_v1';dest.mkdir(exist_ok=False)
    status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',frames=0,training_updates=0)
    def update(**kw):status.update(kw);write(dest/'status.json',status)
    update()
    with exclusive():
        torch.set_num_threads(1);started=time.monotonic()
        for ctx in CONTEXTS:
            out=dest/ctx;out.mkdir();n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);N=len(n.static._xyz)+len(n.dynamic._xyz);ns=len(n.static._xyz)
            compatible=torch.zeros(N,device='cuda');incompatible=torch.zeros_like(compatible);counts=torch.zeros(N,device='cuda',dtype=torch.int16);records=[]
            source=OLD/'motion_scene_inputs_v1'/ctx;sensor=json.loads((source/'sensor_preparation.json').read_text());indices=np.load(source/'dynamic_actor_assets/actor_0/source_lidar_indices.npz')
            for i,f in enumerate(sensor['frames']):
                fid=f['frame_id'];points=np.fromfile(source/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[indices[fid],:3]
                cal={l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in (source/'calib'/(fid+'.txt')).read_text().splitlines()};V=cal['Tr_velo_to_cam_0'].reshape(3,4);K=cal['P0'].reshape(3,4)[:,:3]/4
                cam=points@V[:3,:3].T+V[:3,3];uv=np.floor(cam[:,:2]/cam[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]).astype(int);good=(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)&(cam[:,2]>.1);uv=uv[good];z=cam[good,2];depth=np.full((320,480),np.nan,np.float32)
                for j in np.argsort(z)[::-1]:depth[uv[j,1],uv[j,0]]=z[j]
                valid=np.isfinite(depth);depth=np.nan_to_num(depth);maps=np.stack([valid.astype('float32'),depth,depth**2,np.ones_like(depth)])
                row=dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=i*.02-.48);camera=n.camera(row,{})
                with torch.no_grad():
                    s=prepare_rasterization(camera,n.static,n.args);d=prepare_rasterization(camera,n.dynamic,n.args,dynamic=True)
                    for kw in [s,d]:kw['means2D']=torch.zeros_like(kw['means3D'])
                    kw=helpers.merge_rows(s,d);zrows=(torch.cat((kw['means3D'],torch.ones_like(kw['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3])[:,0]/n.scale
                feat=torch.zeros((N,4),device='cuda',requires_grad=True);kw['features']=torch.cat((kw['features'],feat),1);result=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg)
                w=torch.autograd.grad((result['extra']*torch.from_numpy(maps).cuda()).sum(),feat)[0].clamp_min(0)
                closure=float((w[:,3].sum()-result['alpha'].sum()).abs()/result['alpha'].sum().clamp_min(1));assert closure<2e-4
                mass=w[:,0];mean=w[:,1]/mass.clamp_min(1e-8);variance=(w[:,2]/mass.clamp_min(1e-8)-mean**2).clamp_min(0);rms=((zrows-mean)**2+variance).sqrt();has=mass>.001;ok=has&(rms<=.5);bad=has&(rms>1.)
                compatible+=mass*ok;incompatible+=mass*bad;counts+=ok.to(torch.int16);ids=torch.nonzero(has).flatten()
                np.savez_compressed(out/('%03d.npz'%f['raw']),ids=cpu(ids).astype('int32'),rms_depth_m=cpu(rms[ids]).astype('float16'),return_mass=cpu(mass[ids]).astype('float16'),center_z_m=cpu(zrows[ids]).astype('float16'))
                records.append(dict(raw=f['raw'],actual_target_return_pixels=int(valid.sum()),static_compatible_rows=int(ok[:ns].sum()),dynamic_compatible_rows=int(ok[ns:].sum()),static_sparse_return_alpha_mass=float(mass[:ns].sum()),static_depth_compatible_mass=float((mass*ok)[:ns].sum()),static_depth_incompatible_mass=float((mass*bad)[:ns].sum()),adjoint_closure=closure))
                update(phase='attributing',context=ctx,frames=status['frames']+1,last_raw=f['raw']);print('DEPTH_SUPPORT',ctx,f['raw'],records[-1]['static_compatible_rows'],flush=True)
                del result,w,feat,kw,s,d;gc.collect()
            np.savez_compressed(out/'aggregate.npz',compatible_mass=cpu(compatible),incompatible_mass=cpu(incompatible),support_frames=cpu(counts),branch_sizes=np.array([ns,N-ns]))
            assert n.unchanged();write(out/'summary.json',dict(context=ctx,records=records,static_repeated_compatible_rows=int((counts[:ns]>=3).sum()),dynamic_repeated_compatible_rows=int((counts[ns:]>=3).sum()),single_ray_depth_not_full_3D_identity=True,source_checkpoint_unchanged=True,quality_accepted=False,human_verdict=None))
            del n;gc.collect();torch.cuda.empty_cache()
        update(phase='complete',wall_seconds=time.monotonic()-started)
if __name__=='__main__':main()
