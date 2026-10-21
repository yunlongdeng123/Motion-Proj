"""Frozen V3 native CUDA diagnostics. No optimizer, training, or model writes.

Only this new helper/output tree is writable. Existing engine and shared inputs
remain unchanged. A native buffer trace is gated by RGB/alpha/depth/T replay.
"""
import argparse
import base64
from contextlib import contextmanager
import datetime
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_serial_r1_20261021')
HERE=Path(__file__).resolve().parent
os.environ['PATH']=str(Path(sys.executable).parent)+':/usr/local/cuda/bin:'+os.environ.get('PATH','')
os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v3')
REL=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020')
ARMS=('full','real_only','prior_only','ownership_only')
MODES=('deleted_five','joint','ordinary','road_only','dynamic')
MANIFEST_SHA='eb926ef96b21f5b7382bc7e9eff7f4424cb13fc914246482d90ad179f6b72465'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    return v

@contextmanager
def global_slot():
    with (CAMPAIGN.parent/'gpu_slot.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if active.strip():raise RuntimeError('GPU occupied; diagnostics must run alone')
        yield

class GPUAdapter:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        from torch.utils.cpp_extension import load
        os.environ['MAX_JOBS']='2'
        build=HERE/'build';build.mkdir(exist_ok=True)
        self.ext=load(name='gsdiag_native_trace_r1',sources=[str(HERE/'trace.cpp'),str(HERE/'trace.cu')],
            build_directory=str(build),extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],verbose=False)
        sys.path.insert(0,str(CAMPAIGN/'engine'));sys.path.insert(0,str(REPO/REL/'tools'))
        import evaluate_v3_final49 as final
        self.final=final
        manifest_path=CAMPAIGN/'final49/summary_r2/asset_manifest.json'
        if sha(manifest_path)!=MANIFEST_SHA:raise ValueError('canonical manifest mismatch')
        self.manifest=json.loads(manifest_path.read_text())['files']
        import diff_gaussian_rasterization as raster
        self.native=raster
        expected='0eaff4d7a9b8ef6e711627f4c19824bf427c4387f7c6453fcb769a2c919c920e'
        if sha(raster._C.__file__)!=expected:raise RuntimeError('uninspected native rasterizer binary')
        self.binary_sha=expected;self.arm=None;self.kwargs={};self.before={}
        self.cameras=None;self.protected={}
        self.output=HERE/'runs'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);self.output.mkdir(parents=True,exist_ok=False)
        self.sequence=0

    def model(self,arm):
        if arm not in ARMS:raise ValueError('invalid arm')
        if self.arm==arm:return
        import torch
        from omegaconf import OmegaConf
        from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
        from scene.waymo_loader2 import readWaymoInfo_add
        from utils.camera_utils import cameraList_from_camInfos
        if self.arm:
            self.check_protected();self.kwargs.clear();del self.models
            gc.collect();torch.cuda.empty_cache()
        run=CAMPAIGN/f'arms/{arm}/run30k'
        config=json.loads((run/'effective_config.json').read_text())
        self.config=OmegaConf.create(config)
        if config['iterations']!=30000 or config['seed']!=0:raise ValueError('scientific config mismatch')
        self.meta=json.loads(Path(config['v3_bundle']).read_text());self.scale=float(self.meta['meter_to_model'])
        self.invA=np.linalg.inv(np.asarray(self.meta['world_to_model']))
        if self.cameras is None:
            with self.final.preserve_transform_file(Path(config['source_path'])/'transform_matrix.txt'):
                info=readWaymoInfo_add(self.config)
            cameras=cameraList_from_camInfos(info.train_cameras,4,self.config)
            self.cameras=sorted(cameras,key=lambda c:float(c.timestamp));self.duration=info.time_duration
            if len(self.cameras)!=49:raise ValueError('49 native cameras required')
            for raw,c in zip(range(89,138),self.cameras):
                if c.image_width!=480 or c.image_height!=320 or abs(c.timestamp-(raw-113)*.02)>1e-7:raise ValueError('camera binding')
                actual=c.world_view_transform.detach().cpu().numpy().T@np.asarray(self.meta['world_to_model']);actual[:3]/=self.scale
                if not np.allclose(actual,self.meta['frames'][f'0008{raw:03d}']['world_to_camera'],atol=2e-4,rtol=2e-4):raise ValueError('native camera/world mismatch')
        models=[GaussianModel(self.config),Dynamic_GaussianModel(self.config),EnvLight(resolution=self.config.env_map_res).cuda()]
        self.bindings={}
        for name,prefix,model in zip(('static','dynamic','environment'),('chkpnt','dynamic_chkpnt','env_light_chkpnt'),models):
            path=run/f'{prefix}30000.pth';entry=self.manifest[path.relative_to(CAMPAIGN).as_posix()]
            if sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise ValueError('checkpoint binding mismatch')
            self.bindings[name]=entry
            self.protected[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
            payload,step=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
            if step!=30000:raise ValueError('not30K')
            payload=list(payload)
            if name=='environment':payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
            else:
                for i in range(1,11):payload[i]=payload[i].detach().cuda()
                model.time_duration=self.duration
            model.restore(tuple(payload),training_args=None)
            if getattr(model,'optimizer',None) is not None:raise RuntimeError('optimizer created')
            for value in vars(model).values():
                if torch.is_tensor(value):value.requires_grad_(False)
            del payload;gc.collect()
        self.models=models;self.arm=arm;self.kwargs={}
        for path in (run/'effective_config.json',run/'status.json',Path(config['source_path'])/'transform_matrix.txt',Path(config['v3_bundle'])):
            self.protected[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)

    def check_protected(self):
        for path,state in self.protected.items():
            p=Path(path)
            if (p.stat().st_size,p.stat().st_mtime_ns)!=state:raise RuntimeError('protected input modified')

    def prepare(self,arm,raw):
        import torch
        from gaussian_renderer import prepare_rasterization,merge_kwargs
        if type(raw) is not int or not 89<=raw<=137:raise ValueError('invalid raw')
        self.model(arm)
        if self.kwargs.get('raw')==raw:return self.kwargs
        self.kwargs.clear();gc.collect()
        camera=self.cameras[raw-89];static,dynamic,environment=self.models
        settings=self.native.GaussianRasterizationSettings(image_height=320,image_width=480,
            tanfovx=math.tan(-.5 if self.config.neg_fov else camera.FoVx*.5),
            tanfovy=math.tan(-.5 if self.config.neg_fov else camera.FoVy*.5),
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,sh_degree=static.active_sh_degree,
            campos=camera.camera_center,prefiltered=False,debug=False)
        sk=prepare_rasterization(camera,static,self.config)
        dk=prepare_rasterization(camera,dynamic,self.config,dynamic=True)
        # The active dynamic=True path keeps all rows and passes marginal_t as
        # its mask. Only the unused split=True path physically filters rows.
        dynamic_rows=torch.arange(len(dynamic._xyz),device='cuda')
        static_rows=torch.arange(len(static._xyz),device='cuda')
        if len(dynamic_rows)!=len(dk['means3D']):raise RuntimeError('dynamic row mapping')
        env=environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1).detach()
        self.kwargs=dict(raw=raw,camera=camera,settings=settings,sk=sk,dk=dk,dynamic_rows=dynamic_rows,static_rows=static_rows,env=env)
        return self.kwargs

    def branch(self,arm,raw,mode):
        import torch
        from gaussian_renderer import merge_kwargs
        if mode not in MODES:raise ValueError('invalid mode')
        p=self.prepare(arm,raw);static=self.models[0];sr=p['static_rows'];dr=p['dynamic_rows']
        if mode=='joint':k=merge_kwargs(p['sk'],p['dk']);mapping=torch.cat((sr,dr));dyn=torch.cat((torch.zeros_like(sr),torch.ones_like(dr))).bool()
        elif mode=='dynamic':k=dict(p['dk']);mapping=dr;dyn=torch.ones_like(dr).bool()
        else:
            k=dict(p['sk']);mapping=sr;dyn=torch.zeros_like(sr).bool()
            if mode in ('road_only','ordinary'):
                select=static.road_unit_mask if mode=='road_only' else ~static.road_unit_mask
                k={key:(v[select] if torch.is_tensor(v) else v) for key,v in k.items()};mapping=sr[select];dyn=dyn[select]
        return p,k,mapping,dyn

    def native_render(self,k,settings):
        import torch
        e=torch.empty(0,device='cuda');v=lambda name:k.get(name) if k.get(name) is not None else e
        args=(settings.bg,k['means3D'].contiguous(),v('colors_precomp').contiguous(),k['features'].contiguous(),
            k['opacities'].contiguous(),v('scales').contiguous(),v('rotations').contiguous(),settings.scale_modifier,
            v('cov3D_precomp').contiguous(),k['mask'].contiguous(),settings.viewmatrix,settings.projmatrix,
            settings.tanfovx,settings.tanfovy,320,480,v('shs').contiguous(),settings.sh_degree,settings.campos,False,False)
        result=self.native._C.rasterize_gaussians(*args)
        if not all(torch.isfinite(t).all() for t in (result[2],result[3])):raise FloatingPointError('nonfinite native render')
        return result

    def trace(self,arm,raw,x,y,mode='deleted_five',verify_ids=True):
        import torch
        if type(x) is not int or type(y) is not int or not(0<=x<480 and 0<=y<320):raise ValueError('invalid pixel')
        started=time.monotonic();p,k,mapping,dyn=self.branch(arm,raw,mode);out=self.native_render(k,p['settings'])
        colors=k.get('colors_precomp');colors=colors if colors is not None else torch.empty(0,device='cuda')
        ids,records,counts,sums=self.ext.trace(out[5],out[6],out[7],colors.contiguous(),k['features'].contiguous(),
            torch.tensor([[x,y]],device='cuda',dtype=torch.int32),out[4],len(mapping),480,320,32768)
        cnt=counts[0].cpu().tolist();n=cnt[1]
        if n>32768:raise RuntimeError('trace overflow; no truncated Top-K allowed')
        rec=records[0,:n].cpu().numpy();local=ids[0,:n].long();mapped=mapping[local];dynamic=dyn[local]
        sums=sums[0].cpu().numpy();native_rgb=out[2][:,y,x].cpu().numpy();native_feat=out[3][:,y,x].cpu().numpy()
        errors=dict(alpha_abs=float(abs(sums[4]-native_feat[-1])),depth_sum_model_abs=float(abs(sums[3]-native_feat[-2])),
            rgb_pre_environment_abs_max=float(abs(sums[:3]-native_rgb).max()),T_abs=float(abs(sums[5]-sums[6])))
        if max(errors.values())>2e-5:raise RuntimeError('native-buffer trace failed aggregate replay: '+str(errors))
        saved=CAMPAIGN/f'final49/{arm}_r2/0008{raw:03d}/render_float.npz'
        with np.load(saved) as z:
            fields={key:z[key][:,y,x].copy() for key in ('joint','deleted_five','gt','dynamic_raw','road_raw','road_in_background','road_in_joint','static_alpha','joint_alpha','dynamic_alpha','road_alpha','static_depth_m','joint_depth_m')}
            lidar=float(z['lidar_depth_m'][0,y,x]);prefix={'joint':'joint','dynamic':'dynamic','road_only':'road','deleted_five':'static'}.get(mode)
            old_errors=None
            if prefix:
                expected_rgb=z[{'joint':'joint','deleted_five':'deleted_five','road_only':'road_raw','dynamic':'dynamic_raw'}[mode]][:,y,x]
                rgb=native_rgb+(1-native_feat[-1])*p['env'][:,y,x].cpu().numpy() if mode in ('joint','deleted_five') else native_rgb
                old_errors=dict(rgb=float(abs(rgb-expected_rgb).max()),alpha=float(abs(native_feat[-1]-z[prefix+'_alpha'][0,y,x])),depth=float(abs(native_feat[-2]-z[prefix+'_depth_sum'][0,y,x])))
                if max(old_errors.values())>3e-5:raise RuntimeError('saved final49 replay failed '+str(old_errors))
        picked=set(np.argsort(-rec[:,3],kind='stable')[:40].tolist())|set(np.argsort(-np.linalg.norm(rec[:,4:7]*rec[:,3:4],axis=1),kind='stable')[:40].tolist()) if n else set()
        top=[];mapped_cpu=mapped.cpu().numpy();dyn_cpu=dynamic.cpu().numpy()
        roads=self.models[0]._road_ids();profile=[]
        for i,r in enumerate(rec):
            row=int(mapped_cpu[i]);dynamic_row=bool(dyn_cpu[i]);model=self.models[int(dynamic_row)];road_id=-1 if dynamic_row else int(roads[row])
            role='dynamic' if dynamic_row else ('road' if road_id>=0 else 'ordinary')
            profile.append(dict(z_m=float(r[0]/self.scale),weight=float(r[3]),T=float(r[2]),branch=role))
            if i not in picked:continue
            name='dynamic' if dynamic_row else 'static';binding=self.bindings[name]
            xyz=k['means3D'][local[i]].cpu().numpy();world=self.invA[:3,:3]@xyz+self.invA[:3,3]
            a,b,c=r[9:12];det=a*c-b*b;cov2=[c/det,-b/det,a/det]
            lineage=getattr(model,'_sgcm_lineage_ids',None)
            top.append(dict(id=f'{arm}:{binding["sha256"][:12]}:{name}:{row}',row=row,branch=role,checkpoint_sha256=binding['sha256'],road_id=road_id,
                source_lineage=None if lineage is None else int(lineage[row]),world_m=world,model_xyz=xyz,camera_z_m=r[0]/self.scale,
                alpha=r[1],T=r[2],weight=r[3],color=r[4:7],color_contribution=r[4:7]*r[3],sigma_m=torch.exp(model._scaling[row]).cpu().numpy()/self.scale,
                quaternion=torch.nn.functional.normalize(model._rotation[row],dim=0).cpu().numpy(),cov2=cov2,uv=r[7:9],radius_px=r[12],opacity_base=float(torch.sigmoid(model._opacity[row])),
                time_center=float(model._t[row]),sigma_t=float(torch.exp(model._scaling_t[row])),velocity_model=model._velocity[row].cpu().numpy()))
        id_validation=None
        if verify_ids and n:
            # Independent native feature rasterization: one-hot IDs retain the
            # complete opacity/sort order and directly measure each traced Tα.
            sample=sorted(picked,key=lambda i:float(rec[i,3]),reverse=True)[:20]
            flags=torch.zeros((len(mapping),len(sample)+2),device='cuda')
            for column,i in enumerate(sample):flags[local[i],column]=1
            flags[:,-2:]=k['features'][:,-2:]
            fk=dict(k,features=flags,shs=None,colors_precomp=torch.zeros_like(k['means3D']))
            validation=self.native_render(fk,p['settings'])[3][:len(sample),y,x].cpu().numpy()
            expected=rec[sample,3]
            error=float(np.abs(validation-expected).max())
            if error>2e-6:raise RuntimeError('native one-hot ID weight replay failed')
            id_validation=dict(status='NATIVE_ONE_HOT_ID_WEIGHT_REPLAY_PASSED',count=len(sample),max_weight_error=error)
        self.check_protected()
        return clean(dict(status='NATIVE_BUFFER_CUDA_TRACE_VERIFIED',arm=arm,raw=raw,pixel=[x,y],mode=mode,
            candidate_count=cnt[0],accepted_count=n,top=top,profile=profile,no_optimizer=True,parameter_updates=0,gpu_used=True,
            reference=dict(rgb_gaussians=sums[:3],alpha=sums[4],T=sums[5],depth_sum_model=sums[3]),
            native=dict(alpha=native_feat[-1],depth_sum_model=native_feat[-2],lidar_m=lidar if lidar>0 else None,**fields),
            replay_errors=errors,saved_final49_replay_errors=old_errors,numeric_alpha_depth_agree=True,id_weight_validation=id_validation,
            native_rasterizer_binary_sha256=self.binary_sha,environment_color_not_in_reference_rgb=True,
            trace_scope='read native CUDA preprocessing/sorted tile buffers; same alpha/early termination expressions; aggregate replay checked',
            causal_claim=False,elapsed_s=time.monotonic()-started,limitations=['center-z order is not ellipsoid first intersection','source lineage is not unique lifetime ID','Top-K responsibility alone is not evidence of reconstruction error']))

    def counterfactual(self,arm,raw,x,y,mode,selected):
        import torch
        from PIL import Image
        if not isinstance(selected,list) or not 1<=len(selected)<=200:raise ValueError('select1..200 bound Gaussian IDs')
        p,k,mapping,dyn=self.branch(arm,raw,mode);mask=k['mask'].clone();removed=[]
        for ident in selected:
            if not isinstance(ident,str):raise ValueError('invalid ID')
            parts=ident.split(':')
            if len(parts)!=4 or parts[0]!=arm or parts[2] not in ('static','dynamic'):raise ValueError('invalid ID binding')
            name=parts[2]
            if parts[1]!=self.bindings[name]['sha256'][:12]:raise ValueError('wrong checkpoint ID')
            row=int(parts[3]);match=(mapping==row)&(dyn==(name=='dynamic'))
            if not match.any():raise ValueError('selected primitive absent from chosen branch/time prefilter')
            mask[match]=False;removed.append(dict(id=ident,row=row,branch=name,checkpoint_sha256=self.bindings[name]['sha256']))
        base=self.native_render(k,p['settings']);afterk=dict(k,mask=mask);after=self.native_render(afterk,p['settings'])
        def channels(out,kw):
            feat=out[3];alpha=feat[-1:];raw_rgb=out[2]
            # All modes show the same environment for this controlled compositing test.
            rgb=raw_rgb+(1-alpha)*p['env'];road=((~dyn)&(self.models[0]._road_ids()[mapping.clamp_max(len(self.models[0]._xyz)-1)]>=0)).float()[:,None]
            featk=dict(kw,features=torch.cat((road,kw['features'][:,-2:]),1),shs=None,colors_precomp=torch.zeros_like(kw['means3D']))
            r=self.native_render(featk,p['settings'])[3][:1]
            return dict(rgb=rgb,alpha=alpha,road=r,depth=feat[-2:-1]/alpha.clamp_min(1e-8)/self.scale)
        b=channels(base,k);a=channels(after,afterk)
        capture=self.output/f'cf_{self.sequence:05d}_{arm}_{raw}_{x}_{y}.npz';self.sequence+=1
        np.savez_compressed(capture,**{f'{side}_{key}':value.detach().cpu().numpy() for side,pack in (('before',b),('after',a)) for key,value in pack.items()})
        delta=a['rgb']-b['rgb'];change=delta.abs().mean(0);roi=torch.zeros((320,480),device='cuda',dtype=torch.bool)
        roi[max(0,y-10):min(320,y+11),max(0,x-10):min(480,x+11)]=True
        # A full-frame removal is not assumed to be beneficial; report all pixels.
        metrics=dict(mean_abs_RGB_change=float(change.mean()),max_abs_RGB_change=float(delta.abs().max()),
            selected_pixel_RGB_before=b['rgb'][:,y,x].cpu().numpy(),selected_pixel_RGB_after=a['rgb'][:,y,x].cpu().numpy(),
            selected_pixel_road_before=float(b['road'][0,y,x]),selected_pixel_road_after=float(a['road'][0,y,x]),
            selected_pixel_alpha_before=float(b['alpha'][0,y,x]),selected_pixel_alpha_after=float(a['alpha'][0,y,x]),
            selected_ROI_RGB_change=float(change[roi].mean()),outside_ROI_RGB_change=float(change[~roi].mean()),
            changed_pixel_count=int((change>1/255).sum()),selected_pixel_L1_change=float(change[y,x]))
        frames={}
        for name,value in dict(before=b['rgb'],after=a['rgb'],difference=delta.abs()*5,
                road_before=b['road'].expand(3,-1,-1),road_after=a['road'].expand(3,-1,-1)).items():
            pixels=(value.permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='WEBP',lossless=True)
            frames[name]='data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()
        self.check_protected()
        return clean(dict(status='FULL_NATIVE_RERASTERIZATION_COMPLETE',arm=arm,raw=raw,pixel=[x,y],mode=mode,
            removed=removed,metrics=metrics,images=frames,optimizer_updates=0,parameters_mutated=False,
            float_asset=dict(path=str(capture),sha256=sha(capture),bytes=capture.stat().st_size),
            gpu_used=True,environment_fixed=True,removal='fresh mask copy, complete native preprocessing/sorting/alpha compositing rerun; not RGB subtraction',
            interpretation='change is causal for this fixed rendering intervention, not proof of ground-truth improvement; no hidden-road GT',
            quality_accepted=False,human_verdict=None))

    def render_view(self,arm,raw,mode,view=None):
        """Full native splatting at the geometry viewer's camera, never samples."""
        import torch
        from PIL import Image
        from scene.cameras import Camera
        from gaussian_renderer import prepare_rasterization,merge_kwargs
        p,k,mapping,dyn=self.branch(arm,raw,mode)
        camera=p['camera'];settings=p['settings'];env=p['env']
        if view is not None:
            if not isinstance(view,dict) or set(view)!={'R','T','intrinsics'}:raise ValueError('view fields')
            R=np.asarray(view['R'],dtype=np.float32);T=np.asarray(view['T'],dtype=np.float32);K=np.asarray(view['intrinsics'],dtype=np.float32)
            if R.shape!=(3,3) or T.shape!=(3,) or K.shape!=(4,) or not all(np.isfinite(v).all() for v in (R,T,K)):raise ValueError('view shapes/finite')
            if not np.allclose(R@R.T,np.eye(3),atol=1e-4) or abs(np.linalg.det(R)-1)>1e-4:raise ValueError('proper right-handed camera rotation required')
            if abs(T).max()>10000 or not(50<=K[0]<=3000 and 50<=K[1]<=3000 and -480<=K[2]<=960 and -320<=K[3]<=640):raise ValueError('camera bounds')
            camera=Camera(colmap_id=-1,R=R.T,T=T,fx=float(K[0]),fy=float(K[1]),cx=float(K[2]),cy=float(K[3]),
                image=torch.zeros((3,320,480),device='cuda'),resolution=(480,320),timestamp=p['camera'].timestamp)
            settings=settings._replace(viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,campos=camera.camera_center)
            sk=prepare_rasterization(camera,self.models[0],self.config);dk=prepare_rasterization(camera,self.models[1],self.config,dynamic=True)
            if mode=='joint':k=merge_kwargs(sk,dk)
            elif mode=='dynamic':k=dk
            else:
                k=sk
                if mode in ('ordinary','road_only'):
                    select=self.models[0].road_unit_mask if mode=='road_only' else ~self.models[0].road_unit_mask
                    k={key:(v[select] if torch.is_tensor(v) else v) for key,v in k.items()}
            env=self.models[2](camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        result=self.native_render(k,settings);rgb=result[2];alpha=result[3][-1:];depth=result[3][-2:-1]
        use_env=mode in ('joint','deleted_five','ordinary')
        if use_env:rgb=rgb+(1-alpha)*env
        pixels=(rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='WEBP',lossless=True)
        capture=self.output/f'view_{self.sequence:05d}_{arm}_{raw}_{mode}.npz';self.sequence+=1
        np.savez_compressed(capture,rgb=rgb.cpu().numpy(),alpha=alpha.cpu().numpy(),depth_sum_model=depth.cpu().numpy())
        saved_error=None
        if view is None and mode!='ordinary':
            channel={'joint':'joint','deleted_five':'deleted_five','road_only':'road_raw','dynamic':'dynamic_raw'}[mode]
            with np.load(CAMPAIGN/f'final49/{arm}_r2/0008{raw:03d}/render_float.npz') as saved:saved_error=float(abs(rgb.cpu().numpy()-saved[channel]).max())
            if saved_error>3e-5:raise RuntimeError('full-frame native view replay mismatch')
        self.check_protected()
        return clean(dict(status='FULL_NATIVE_VIEW_RENDER_COMPLETE',arm=arm,raw=raw,mode=mode,view=view,
            image='data:image/webp;base64,'+base64.b64encode(buffer.getvalue()).decode(),image_size=[480,320],
            gaussian_count=len(k['means3D']),render_scope='full model native splatting, not geometry sample',environment_enabled=use_env,
            full_frame_saved_replay_max_error=saved_error,float_asset=dict(path=str(capture),bytes=capture.stat().st_size,sha256=sha(capture)),
            gpu_used=True,optimizer_updates=0,parameters_mutated=False,novel_view_has_ground_truth=False,human_verdict=None))

def dispatch(adapter,q):
    import torch
    from unittest.mock import patch
    if set(q)-{'action','arm','raw','x','y','mode','ids','view'}:raise ValueError('unknown query fields')
    for key in ('raw','x','y'):
        if type(q.get(key)) is not int:raise ValueError('integer query required')
    if not(89<=q['raw']<=137 and 0<=q['x']<480 and 0<=q['y']<320):raise ValueError('query bounds')
    with patch.object(torch.optim.Adam,'step',side_effect=RuntimeError('diagnostics forbid optimization')),torch.no_grad():
        if q.get('action','trace')=='trace':return adapter.trace(q['arm'],q['raw'],q['x'],q['y'],q.get('mode','deleted_five'))
        if q['action']=='counterfactual':return adapter.counterfactual(q['arm'],q['raw'],q['x'],q['y'],q.get('mode','deleted_five'),q.get('ids'))
        if q['action']=='render_view':return adapter.render_view(q['arm'],q['raw'],q.get('mode','deleted_five'),q.get('view'))
        raise ValueError('unsupported diagnostic action')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stdio',action='store_true');p.add_argument('--test',action='store_true');a=p.parse_args()
    with global_slot():
        # Model loaders print progress; keep stdout a JSON-only RPC channel.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):adapter=GPUAdapter()
        for line in sys.stdin:
            try:
                q=json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):result=dispatch(adapter,q)
                import torch
                result['execution_profile']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='this diagnostic worker only; not device-wide or training peak',gpu_name=torch.cuda.get_device_name())
                path=adapter.output/f'query_{adapter.sequence:05d}.json';adapter.sequence+=1
                with path.open('x') as f:json.dump(result,f,allow_nan=False)
                result['diagnostic_record']=dict(path=str(path),sha256=sha(path))
                print(json.dumps(result,allow_nan=False),flush=True)
            except Exception as e:
                import traceback;traceback.print_exc(file=sys.stderr)
                print(json.dumps(dict(error=repr(e),gpu_used=True,optimizer_updates=0)),flush=True)

if __name__=='__main__':main()
