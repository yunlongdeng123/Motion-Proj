"""Target-specific unknown evidence. Original static rows, exact native alpha*T.

Background and actor hypotheses compete; unknown is NOT a global deletion set.
Depth tests use Gaussian centers and cannot certify an entire footprint.
This read-only first-scene probe never changes a checkpoint or original owner.
"""
import argparse,json,time,os,gc
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation
from train_actor import NativeMotion,ROOT,OLD,torch,cpu,prepare_rasterization,helpers
from segmentation import write
from v43_resource_control import require_start
from review_gpu_slot import slot

CTX='14705303724557273004_3105_000_3125_000'

def source_maps(source,f,ds,seg):
    raw=f['raw'];fid=f['frame_id'];ob=np.load(ds/('%03d.npz'%raw));body=ob['body']
    labels=seg['labels'];confidence=seg['confidence'].astype('float32')
    vehicle=np.isin(labels,np.arange(12,20))|body
    safe=(~binary_dilation(vehicle,iterations=5))&(labels>=1)&(labels<=10)&(confidence>=.8)
    pts=np.fromfile(source/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3]
    cal={l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in (source/'calib'/(fid+'.txt')).read_text().splitlines()}
    V=cal['Tr_velo_to_cam_0'].reshape(3,4);K=cal['P0'].reshape(3,4)[:,:3]/4
    cam=pts@V[:3,:3].T+V[:3,3];valid=cam[:,2]>.1;cam=cam[valid];pts=pts[valid]
    uv=np.floor(cam[:,:2]/cam[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]).astype(int)
    ok=(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320);uv=uv[ok];z=cam[ok,2]
    depth=np.full((320,480),np.inf,np.float32);np.minimum.at(depth,(uv[:,1],uv[:,0]),z)
    bg=safe&np.isfinite(depth);bz=np.where(bg,depth,0)
    tz=np.nan_to_num(ob['depth']);tv=np.isfinite(ob['depth'])&body
    core=binary_erosion(body,iterations=2);fringe=binary_dilation(body,iterations=5)&~core
    target_sem=body&np.isin(labels,np.arange(14,20))
    maps=np.stack([core,body,fringe,bg,bz,bz*bz,tv,tz*tv,tz*tz*tv,np.ones_like(body),target_sem,safe]).astype('float32')
    return ob,body,safe,depth,maps

def save(p,t):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.round(np.clip(cpu(t).transpose(1,2,0),0,1)*255).astype('uint8')).save(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--managed-round',type=Path);a=p.parse_args()
    require_start(CTX);a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(**d):write(a.output/'status.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],training_updates=0,**d))
    status(phase='waiting_GPU')
    with slot(a.managed_round):
        torch.set_num_threads(1);n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000);ns=len(n.static._xyz)
        ds=ROOT/'datasets'/CTX;frames=json.loads((ds/'manifest.json').read_text())['frames'];source=n.source
        counts={k:np.zeros(ns,np.uint16) for k in ['target','depth_target','depth_background','clean','fringe']}
        bins={k:np.zeros(ns,np.uint8) for k in counts};sums={k:np.zeros(ns,np.float64) for k in ['total','target','core','fringe','vehicle_target','clean']};records=[]
        for i,f in enumerate(frames):
            require_start(CTX);raw=f['raw'];sem=np.load(ROOT/'segformer_b0_attempt3'/CTX/('%03d.npz'%raw));ob,body,safe,depth,maps=source_maps(source,f,ds,sem)
            camera=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{})
            with torch.no_grad():
                kw=prepare_rasterization(camera,n.static,n.args);kw['means2D']=torch.zeros_like(kw['means3D']);kw['features']=kw['features'][:,:2]
                z=(torch.cat((kw['means3D'],torch.ones_like(kw['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3])[:,0]/n.scale
                sigma=kw['scales'].amax(1)/n.scale
            aux=torch.zeros((ns,len(maps)),device='cuda',requires_grad=True);kw['features']=torch.cat((kw['features'],aux),1)
            result=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg)
            w=torch.autograd.grad((result['extra']*torch.from_numpy(maps).cuda()).sum(),aux)[0].clamp_min(0)
            closure=float((w[:,9].sum()-result['alpha'].sum()).abs()/result['alpha'].sum().clamp_min(1));assert closure<2e-4
            def rms(first):
                m=w[:,first];mean=w[:,first+1]/m.clamp_min(1e-8);var=(w[:,first+2]/m.clamp_min(1e-8)-mean*mean).clamp_min(0)
                return ((z-mean)**2+var).sqrt()
            rb=rms(3);rt=rms(6);total=w[:,9];tf=w[:,1]/total.clamp_min(1e-8)
            # Exact pixel contributions plus source depth. No semantic label alone confirms identity.
            gates=dict(target=(w[:,1]>.01)&(tf>.02),depth_target=(w[:,6]>.001)&(rt<=.5)&(sigma<=.5),depth_background=(w[:,3]>.001)&(rb<=.5)&(sigma<=.5),clean=(w[:,11]>.1)&(w[:,11]/total.clamp_min(1e-8)>.8),fringe=(w[:,2]>.01))
            for name,hit in gates.items():
                hit=cpu(hit);counts[name]+=hit;bins[name][hit]|=1<<min(2,i*3//len(frames))
            for name,j in [('core',0),('target',1),('fringe',2),('total',9),('vehicle_target',10),('clean',11)]:sums[name]+=cpu(w[:,j])
            active=cpu((w[:,1]>.001)|(w[:,2]>.001)|(w[:,3]>.001));ids=np.flatnonzero(active)
            np.savez_compressed(a.output/('%03d.npz'%raw),ids=ids.astype('int32'),weights=cpu(w)[ids].astype('float32'),target_depth_rms_m=cpu(rt)[ids],background_depth_rms_m=cpu(rb)[ids],center_z_m=cpu(z)[ids])
            save(a.output/'baseline_delete'/('%03d.png'%raw),result['rgb'].detach()+(1-result['alpha'].detach())*n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1))
            records.append(dict(raw=raw,body_pixels=int(body.sum()),safe_background_pixels=int(safe.sum()),safe_background_return_pixels=int((np.isfinite(depth)&safe).sum()),target_return_pixels=int(maps[6].sum()),adjoint_closure=closure,static_target_mass=float(w[:,1].sum()),target_depth_rows=int(gates['depth_target'].sum()),background_depth_rows=int(gates['depth_background'].sum())))
            status(phase='attributing_static_after_actor_delete',frames=i+1,last_raw=raw);print('TARGET_UNKNOWN',raw,records[-1],flush=True)
            del result,w,kw,aux;gc.collect()
        bitcount=np.array([bin(j).count('1') for j in range(8)])
        repeated=lambda k:(counts[k]>=3)&(bitcount[bins[k]]>=2)
        candidate=(counts['target']>0) # Target-associated unknown; global missing/low-confidence rows are excluded.
        background=repeated('depth_background')&repeated('clean')
        actor=repeated('depth_target')&repeated('target')&(sums['target']/np.maximum(sums['total'],1e-8)>.7)&(sums['vehicle_target']/np.maximum(sums['target'],1e-8)>.6)
        mixed=candidate&background;pure=actor&~background;unsupported=candidate&~pure&~background
        masks=dict(target_candidate=candidate,background_protected=background,mixed_target_background=mixed,vehicle_candidate=pure,unsupported_target=unsupported)
        np.savez_compressed(a.output/'evidence.npz',**{'count_'+k:v for k,v in counts.items()},**{'bins_'+k:v for k,v in bins.items()},**{'sum_'+k:v for k,v in sums.items()},**masks)
        # Read-only causal controls; these are not approved deletion masks.
        controls=dict(baseline=np.zeros(ns,bool),target_upper_bound=candidate,target_preserve_observed_bg=candidate&~background,strict_vehicle_only=pure)
        rows=json.loads((OLD/'motion_edit_suite_v1'/CTX/'delete_target.json').read_text())['frames'];render_records=[]
        with torch.no_grad():
            for row in rows:
                raw=row['raw'];cam=n.camera(row,{});base=prepare_rasterization(cam,n.static,n.args);base['means2D']=torch.zeros_like(base['means3D']);base['features']=base['features'][:,:2];env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1);ob=np.load(ds/('%03d.npz'%raw));body=torch.from_numpy(ob['body']).cuda();ref=None
                for name,mask in controls.items():
                    kw={**base,'opacities':base['opacities'].clone()};kw['opacities'][torch.from_numpy(mask).cuda()]=0;r=helpers.raster(cam,n.args,n.static.active_sh_degree,kw,n.bg);rgb=r['rgb']+(1-r['alpha'])*env
                    if ref is None:ref=rgb
                    delta=(rgb-ref).abs().mean(0);save(a.output/name/'native_rgb'/('%03d.png'%raw),rgb)
                    render_records.append(dict(raw=raw,control=name,rows_removed=int(mask.sum()),outside_target_change=float(delta[~body].mean()),outside_target_changed_pixels=int((delta[~body]>1/255).sum())))
        assert n.unchanged()
        write(a.output/'summary.json',dict(schema='target-unknown-v2-probe',context=CTX,static_rows=ns,frames=records,mask_counts={k:int(v.sum()) for k,v in masks.items()},controls=render_records,target_candidate_total_target_mass_fraction=float(sums['target'][candidate].sum()/sums['target'].sum()),source_checkpoint_unchanged=True,not_ground_truth=True,whole_row_removal_accepted=False,center_depth_not_surface_intersection=True,quality_accepted=False,human_verdict=None))
        status(phase='complete',frames=len(frames),wall_seconds=time.monotonic()-started,cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()
