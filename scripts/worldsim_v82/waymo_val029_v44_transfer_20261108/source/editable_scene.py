"""A single persistent scene and one depth sort for KEEP and DELETE."""
from pathlib import Path
import json,sys
import numpy as np
from native_runtime import ROOT,load,camera,native_rows
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs')
class Scene:
    def __init__(self,with_asset=True,with_road=False):
        import torch
        self.pipe,self.models,self.cfg=load(ROOT/'formal10k',10000);self.static,self.dynamic,self.env=self.models
        from drivingeditor_dynamic_edit import actor_ids
        z=np.load(ROOT/'semantic_all158/semantic_ownership.npz');self.sids=torch.tensor(z['static_actor_ids'],device='cuda');self.source_ids=actor_ids(self.dynamic);ids=torch.tensor(z['dynamic_actor_ids'],device='cuda');self.dids=torch.where(ids>0,ids,self.source_ids)
        self.T=np.loadtxt(ROOT/'inputs/native/transform_matrix.txt');self.asset=None;self.road=None;self.gates=None;self.gate_ids=None
        if with_asset:
            sys.path[:0]=[str(ROOT.parent/'scene008_a6_asset_controls_20261105/source'),str(ROOT.parent/'scene008_a6_insert_artifacts_20261105/source/scale')]
            from sh_asset import SHAsset
            from surface_lod_asset import SurfaceLODAsset
            self.asset=SurfaceLODAsset(SHAsset.from_geometry(ROOT/'revealed_material_r2/complete_asset.npz').cuda().requires_grad_(False),ROOT/'revealed_surface/posterior.npz')
        if with_road:
            self.road=Road(ROOT/'world_road/reference.npz')
        self.raws=[int(s[-3:]) for s in self.cfg['followup_scene_spec']['frame_ids']]
        self.versions={(b,k):v._version for b,m in enumerate(self.models[:2]) for k,v in vars(m).items() if torch.is_tensor(v)}
    def unchanged(self):return all(vars(self.models[b])[k]._version==v for (b,k),v in self.versions.items())
    def camera(self,raw):return camera(ROOT/'formal10k',self.cfg,raw)
    def observation(self,raw):
        path=BANK/'observations'/f'{raw:03d}.json'
        if not path.exists():path=ROOT/'heldout/observations'/f'{raw:03d}.json'
        return json.loads(path.read_text())
    def rows(self,raw,cam=None):
        import torch
        from utils.v4_road_render import merge_rows,road_kwargs
        cam=cam or self.camera(raw);s=native_rows(cam,self.static,self.pipe);d=native_rows(cam,self.dynamic,self.pipe,True)
        for branch,ids in [(s,self.sids),(d,self.dids)]:
            f=branch['features'].new_zeros((len(ids),24));f[:,:2]=branch['features'][:,:2];f[:,2]=(ids==70);f[:,3]=(ids==25);f[:,4]=1-f[:,2]-f[:,3];branch['features']=f
        if self.asset is not None:s['mask'] &= self.sids!=25;d['mask'] &= (self.dids!=25)&(self.source_ids!=25)
        if self.gates is not None:s['opacities']=s['opacities'].index_copy(0,self.gate_ids,s['opacities'][self.gate_ids]*self.gates[:,None])
        values=merge_rows(s,d);ids=torch.cat((self.sids,self.dids))
        if self.asset is not None:
            obs=self.observation(raw);body=next((b for b in obs['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA'),None)
            if body is not None:
                pose=torch.tensor(self.T@np.array(body['world_from_actor']).reshape(4,4),device='cuda',dtype=torch.float32);a=self.asset.rows(pose,cam,self.pipe,1,precomputed=True);f=a['features'].new_zeros((len(a['means3D']),24));f[:,:2]=a['features'];f[:,3]=1;a['features']=f;values=merge_rows(values,a);ids=torch.cat((ids,ids.new_full((len(a['means3D']),),25)))
        if self.road is not None:
            r=road_kwargs(cam,self.road,self.pipe,precomputed_color=True);f=r['features'].new_zeros((len(r['means3D']),24));f[:,:2]=r['features'];f[:,5]=1;r['features']=f;values=merge_rows(values,r);ids=torch.cat((ids,ids.new_zeros(len(r['means3D']))))
        return values,ids,cam
    def render(self,raw,delete=False,cam=None,rows=None):
        import torch
        from utils.v4_road_render import raster
        values,ids,cam=rows if rows is not None else self.rows(raw,cam);values=dict(values,mask=values['mask'] & ((ids!=70) if delete else torch.ones_like(ids,dtype=torch.bool)));out=raster(cam,self.pipe,0,values,torch.zeros(3,device='cuda'));out['rgb']=out['rgb']+(1-out['alpha'])*self.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        return out
class Road:
    def __init__(self,path):
        import torch
        z=np.load(path);a=z['active'];tensor=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32);T=z['model_from_world'];scale=float(z['meter_to_model']);self.xyz=tensor(z['model_xyz'][a]);self.scaling=tensor(z['gaussian_sigma_m'][a]*scale);n=z['normal_world'][a]@T[:3,:3].T/scale;n/=np.linalg.norm(n,axis=1)[:,None];q=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))];q/=np.linalg.norm(q,axis=1)[:,None];self.rotation=tensor(q);self.rgb=tensor(z['reference_srgb'][a]);self.opacity=torch.logit(tensor(z['road_opacity'][a,None]).clamp(.00001,.99999));self.active=torch.ones(len(self.xyz),device='cuda',dtype=torch.bool)
    def colors(self):return self.rgb
    def output_mask(self):return self.active
