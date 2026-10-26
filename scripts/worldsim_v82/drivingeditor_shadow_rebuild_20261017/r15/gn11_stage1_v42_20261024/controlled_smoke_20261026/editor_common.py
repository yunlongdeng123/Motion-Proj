"""固定B主副本和独立编辑门控；不向主副本写回任何参数。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023')
B=C/'arms/B/run30k'
FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')
OUT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
sys.path.insert(0,str(C/'engine'))
from omegaconf import OmegaConf
from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
from scene.cameras import Camera
from utils.v4_road_render import render_road,road_kwargs,merge_rows,raster
from gaussian_renderer import prepare_rasterization

def read(p):return json.loads(Path(p).read_text())
def cpu(t):return t.detach().cpu().numpy()
def save(p,t):Image.fromarray(np.uint8(np.round(np.clip(cpu(t).transpose(1,2,0),0,1)*255))).save(p)

class Road:
    def __init__(self,path):
        a=dict(np.load(path));m=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
        A=np.asarray(m['world_to_model']);scale=m['meter_to_model'];ids=np.flatnonzero(a['surface_supported']&a['eligible'])
        tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
        self.ids=ids;self.xyz=tensor(a['model_xyz'][ids]);self.scaling=tensor(a['gaussian_sigma_m'][ids]*scale)
        n=a['normal_world'][ids]@A[:3,:3].T/scale;n/=np.linalg.norm(n,axis=1)[:,None]
        q=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))];q/=np.linalg.norm(q,axis=1)[:,None]
        self.rotation=tensor(q);self.dc=torch.nn.Parameter(tensor((a['rgb'][ids]-.5)/.28209479177387814)[:,None])
        self.opacity=torch.nn.Parameter(tensor(np.zeros((len(ids),1))));self.active=torch.ones(len(ids),dtype=torch.bool,device='cuda')
    def colors(self):return .5+.28209479177387814*self.dc[:,0]
    def output_mask(self):return self.active

class Native:
    def __init__(self):
        torch.set_num_threads(1);self.config=read(B/'effective_config.json');self.args=OmegaConf.create(self.config)
        self.files=[B/f'{p}30000.pth' for p in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']]
        self.before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in self.files}
        self.static=GaussianModel(self.args);self.dynamic=Dynamic_GaussianModel(self.args);self.env=EnvLight(resolution=self.args.env_map_res).cuda()
        for p,model in zip(self.files,[self.static,self.dynamic,self.env]):
            payload,step=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True);assert step==30000;payload=list(payload)
            if 'env_light' in p.name:payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
            else:
                for i in range(1,11):payload[i]=payload[i].detach().cuda()
                model.time_duration=tuple(self.config['time_duration'])
            model.restore(tuple(payload),training_args=None)
            assert getattr(model,'optimizer',None) is None
            for v in vars(model).values():
                if torch.is_tensor(v):v.requires_grad_(False)
            del payload;gc.collect()
        self.records=read(B/'cameras.json');self.stamps=read(B/'actual_scene_binding.json')['timestamps']
        self.bg=torch.tensor([1.,1.,1.] if self.args.white_background else [0.,0.,0.],device='cuda')
    def camera(self,raw):
        if 89<=raw<=137:
            r=next(x for x in self.records if x['img_name']==f'0008{raw:03d}');c2w=np.eye(4);c2w[:3,:3]=r['rotation'];c2w[:3,3]=r['position'];W=np.linalg.inv(c2w)
            K=np.array([[r['fx']/4,0,r['cx']/4],[0,r['fy']/4,r['cy']/4],[0,0,1.]])
            stamp=self.stamps[raw-89]
        else:
            f=np.load(FULL/'frames'/f'{raw:03d}.npz');m=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
            # local -> world -> model, remove the uniform scale from camera rotation.
            T=np.asarray(m['world_to_model'])@np.asarray(m['coordinate_anchor']);c2w=T@np.linalg.inv(f['w2l']);c2w[:3,:3]/=m['meter_to_model'];W=np.linalg.inv(c2w);K=f['K']
            time0=read(FULL/'frames/089.json')['timestamp_micros'];time1=read(FULL/'frames/137.json')['timestamp_micros']
            stamp=self.stamps[0]+(read(FULL/'frames'/f'{raw:03d}.json')['timestamp_micros']-time0)/(time1-time0)*(self.stamps[-1]-self.stamps[0])
        return Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=K[0,2],cy=K[1,2],fx=K[0,0],fy=K[1,1],image=torch.zeros((3,320,480)),image_name=str(raw),resolution=(480,320),timestamp=stamp)
    def render(self,camera,road=None,suppress=None,dynamic=False,static_multiplier=None):
        kwargs=prepare_rasterization(camera,self.static,self.args);kwargs['means2D']=torch.zeros_like(kwargs['means3D'])
        if suppress is not None:kwargs['mask']=kwargs['mask']&~suppress
        if static_multiplier is not None:kwargs['opacities']=kwargs['opacities']*static_multiplier[:,None]
        if dynamic:
            dy=prepare_rasterization(camera,self.dynamic,self.args,dynamic=True);dy['means2D']=torch.zeros_like(dy['means3D']);kwargs=merge_rows(kwargs,dy)
        if road is not None:
            n=kwargs['shs'].shape[1] if kwargs['shs'] is not None else 1
            rr=road_kwargs(camera,road,self.args,n,precomputed_color=kwargs['colors_precomp'] is not None)
            kwargs['features']=torch.cat((kwargs['features'],torch.zeros_like(kwargs['features'][:,:1])),1)
            rr['features']=torch.cat((rr['features'],torch.ones_like(rr['features'][:,:1])),1);kwargs=merge_rows(kwargs,rr)
        out=raster(camera,self.args,self.static.active_sh_degree,kwargs,self.bg)
        environment=self.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        out['rgb']=out['rgb']+(1-out['alpha'])*environment
        out['contribution']=out['extra'] if road is not None else torch.zeros_like(out['alpha'])
        return out
    def unchanged(self):return self.before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in self.files}

def make_gate(native, footprint=False, ray_depth=False):
    """一次登记的三维诊断门控：raw102 a37/a6轮廓射线体积内、路面上方0.25~3m。

    门控不随渲染帧变化；不是已证明的污染分类，不删除主模型，也不删整条祖先后代。
    """
    from scipy.ndimage import binary_dilation,map_coordinates
    from scipy.spatial import cKDTree
    m=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
    T=np.asarray(m['world_to_model'])@np.asarray(m['coordinate_anchor']);inv=np.linalg.inv(T)
    xyz=cpu(native.static.get_xyz);local=xyz@inv[:3,:3].T+inv[:3,3]
    a=np.load(FULL/'full199_initialization.npz');ids=np.flatnonzero(a['surface_supported']);tree=cKDTree(a['local_xyz'][ids,:2]);dist,nn=tree.query(local[:,:2]);height=local[:,2]-a['local_xyz'][ids[nn],2]
    z=np.load('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/0008102.npz');labels=z['actor_labels'];mask=binary_dilation((labels==4)|(labels==5),iterations=2)
    f=np.load(FULL/'frames/102.npz');cam=local@f['w2l'][:3,:3].T+f['w2l'][:3,3];uvh=cam@f['K'].T;uv=uvh[:,:2]/np.maximum(cam[:,2:],1e-8)-.5
    inside=map_coordinates(mask.astype(float),[uv[:,1],uv[:,0]],order=0,mode='constant',cval=0)>0
    if ray_depth:
        from scipy.ndimage import distance_transform_edt
        sys.path.insert(0,str(C/'tools'))
        from v41_dense_geometry import FixedSurface
        road_surface=FixedSurface(m,dict(keys=a['keys'],world=a['world_xyz'],surface_supported=a['surface_supported'],provenance=a['eligible'].astype(np.uint8),base_weight=a['source_valid'],normal_world=a['normal_world']))
        road_map=road_surface.camera(f['K'],f['w2l']@np.linalg.inv(np.asarray(m['coordinate_anchor'])))
        dmap,nearest=distance_transform_edt(~mask,return_indices=True)
        # Use actual time-deformed raster means. Shared immutable row IDs across all later views.
        means=cpu(prepare_rasterization(native.camera(102),native.static,native.args)['means3D'])
        lm=means@inv[:3,:3].T+inv[:3,3];cc=lm@f['w2l'][:3,:3].T+f['w2l'][:3,3];ph=cc@f['K'].T;uv=ph[:,:2]/np.maximum(cc[:,2:],1e-8)-.5
        ij=np.clip(np.round(uv).astype(int),[0,0],[479,319]);ny,nx=nearest[:,ij[:,1],ij[:,0]]
        road_depth=road_map['depth'].reshape(320,480)[ny,nx];support=road_map['supported'].reshape(320,480)[ny,nx]
        sigma=cpu(native.static.get_scaling).max(1)/m['meter_to_model'];radius=3*sigma*np.max(f['K'][:2,:2])/np.maximum(cc[:,2],.1)
        gate=support&(cc[:,2]>.1)&(cc[:,2]<road_depth-.1)&(dmap[ij[:,1],ij[:,0]]<=radius)&(uv[:,0]+radius>=0)&(uv[:,0]-radius<480)&(uv[:,1]+radius>=0)&(uv[:,1]-radius<320)
        policy='fixed raw102 a37/a6 3sigma image footprint overlap AND actual raster centre depth in front of independently measured road by0.1m; destructive diagnostic in EDIT COPY ONLY, no claim of safe segmentation'
    elif footprint:
        # Additional preregistered diagnostic after centre-only gate failed:
        # finite Gaussian footprint rather than mean alone; can remove valid road,
        # therefore must report suppression-only damage and never label this a clean GT.
        from scipy.ndimage import distance_transform_edt
        sigma=cpu(native.static.get_scaling).max(1)/m['meter_to_model']
        radius=3*sigma*np.max(f['K'][:2,:2])/np.maximum(cam[:,2],.1)
        distance_to_mask=map_coordinates(distance_transform_edt(~mask),[uv[:,1],uv[:,0]],order=1,mode='constant',cval=1e6)
        gate=(dist<=.8)&(height>=-.15)&(height<=3)&(height+3*sigma>=.25)&(distance_to_mask<=radius)&(cam[:,2]>.1)&(local[:,0]>=6)&(local[:,0]<=40)
        policy='fixed raw102 a37/a6 Gaussian 3sigma footprint overlap; centre height -0.15..3m, upper extent>=0.25m; x6..40,nearest support<=0.8m; diagnostic upper-bound, may remove valid background'
    else:
        gate=(dist<=.4)&(height>=.25)&(height<=3)&inside&(cam[:,2]>.1)
        policy='fixed raw102 a6/a37 dilated2px ray volume; height0.25..3m above observed surface; nearest support<=0.4m; diagnostic not semantic truth'
    np.savez_compressed(OUT/('fixed_gate_ray_depth.npz' if ray_depth else 'fixed_gate_footprint.npz' if footprint else 'fixed_gate.npz'),suppress=gate,static_rows=np.flatnonzero(gate),height_m=height[gate])
    return torch.as_tensor(gate,device='cuda'),dict(rows=int(gate.sum()),total=len(gate),policy=policy)
