"""唯一Full5 checkpoint的原生joint场景；可选完整C道路，不加载V44/C2。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import torch

from scripts.worldsim_v82.drivingeditor_dynamic_edit.scene_io import (
    checkpoint_actors,load_native_models,camera_headers,make_camera,model_tensors)
from scripts.worldsim_v82.drivingeditor_dynamic_edit.ownership import actor_ids,ActorEditQuery

MODEL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1/inference10k')
C_MANIFEST=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/scene_manifest.json')
GROUP_ACTOR_IDS=[-1,5,6,18,26,37,-2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()


def module(path):
    name='_full5_road_pinned_'+sha(path)[:16]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]


def merge(a,b):
    if set(a)!=set(b):raise ValueError('joint字段不匹配')
    result={}
    for k in a:
        if a[k] is None and b[k] is None:result[k]=None
        elif a[k] is None or b[k] is None:raise ValueError('joint表示不匹配：'+k)
        else:result[k]=torch.cat((a[k],b[k]),0).contiguous()
    return result


class Full5RoadScene:
    def __init__(self,model=MODEL,iteration=10000,road_manifest=None,transform_path=None):
        self.model=Path(model).resolve();self.iteration=int(iteration)
        info=checkpoint_actors(self.model,self.iteration)
        if info['dynamic_actor_ids']!=[5,6,18,26,37]:raise ValueError('需要完整五源身份模型')
        self.args,self.models,self.renderer=load_native_models(self.model,self.iteration)
        self.static=self.models['static'];self.dynamic=self.models['dynamic'];self.environment=self.models['environment']
        self.owners=actor_ids(self.dynamic).detach().clone();self.headers=camera_headers(self.model)
        self.dynamic_actor_ids=info['dynamic_actor_ids'];self.torch=torch;self.np=np
        self.bg=torch.tensor([1.,1.,1.] if self.args.white_background else [0.,0.,0.],device='cuda')
        self.n=SimpleNamespace(static=self.static,dynamic=self.dynamic,env=self.environment,bg=self.bg)
        if not self.args.including_dynamic:raise ValueError('Full5场景必须包括完整动态分支')
        self.degree=int(self.static.active_sh_degree);self._raw={};self.render_calls=0
        self.transform_path=Path(transform_path).resolve() if transform_path else Path(self.args.source_path)/'transform_matrix.txt'
        self.world_to_model=np.loadtxt(self.transform_path);units=np.linalg.norm(self.world_to_model[:3,:3],axis=0)
        if not np.allclose(units,units[0],atol=1e-14,rtol=0):raise ValueError('非统一米制比例')
        self.scale=float(units[0]);self.road_manifest=None;self.asset=None;self.surface=None;self.mip=None
        paths=[self.model/n for n in [f'chkpnt{self.iteration}.pth',f'dynamic_chkpnt{self.iteration}.pth',f'env_light_chkpnt{self.iteration}.pth','effective_config.json','cameras.json','actual_scene_binding.json']]+[self.transform_path,Path(__file__)]
        if road_manifest is not None:
            self.road_manifest_path=Path(road_manifest).resolve();self.road_manifest=json.loads(self.road_manifest_path.read_text())
            p=self.road_manifest_path.parent/self.road_manifest['new_road']['file']
            if sha(p)!=self.road_manifest['new_road']['sha256']:raise ValueError('C资产SHA不符')
            with np.load(p,allow_pickle=False) as z:self.asset={k:z[k].copy() for k in z.files}
            a=self.asset
            if len(a['active'])!=40392 or not a['active'].all():raise ValueError('C必须完整40392行')
            if not np.array_equal(a['world_to_model'],self.world_to_model):raise ValueError('C与Full5坐标不同')
            if abs(float(a['meter_to_model'])-self.scale)>1e-14:raise ValueError('C米制比例不同')
            cv=lambda key:torch.as_tensor(a[key].copy(),device=self.bg.device)
            self.surface=SimpleNamespace(xyz=cv('render_model_xyz'),scales=cv('render_scales_model'),rotations=cv('render_rotation_wxyz'),
                logits=cv('render_opacity_logit'),rgb=cv('render_colours_srgb'),active=cv('active'))
            policy=self.road_manifest['renderer_policy'];mp=self.road_manifest_path.parent/policy['source_file']
            if sha(mp)!=self.road_manifest['replay_sources'][policy['source_file']]['sha256']:raise ValueError('relativeMip源码不符')
            parent=module(mp)
            class ExactMip(parent.RelativeMip):
                def __init__(m):
                    m.parent_asset_path=p;m.parent_asset_sha256=sha(p)
                    m.parent_xyz=a['parent_render_model_xyz'].copy();m.parent_scales=a['parent_render_scales_model'].copy();m.parent_rotation=a['parent_render_rotation_wxyz'].copy()
                    m.neg_fov=bool(policy['legacy_neg_fov']);m.normalize_quaternion=bool(policy['normalize_quaternion'])
                    m._geometry={};m._parent_factors={};m.records=[]
            if bool(self.args.neg_fov)!=bool(policy['legacy_neg_fov']) or bool(self.args.compute_cov3D_python)!=bool(policy['normalize_quaternion']):
                raise ValueError('Full5与C原相机/协方差规则不同')
            self.mip=ExactMip();self.parent_indices=torch.as_tensor(a['child_to_parent_ids'],device=self.bg.device,dtype=torch.long)
            if not np.array_equal(a['child_to_parent_ids'],np.repeat(np.arange(10098),4)):raise ValueError('C父子映射不符')
            paths += [self.road_manifest_path,p,mp]
        self._files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
        self._tensors=model_tensors(self.models)
        if self.surface is not None:self._tensors.update({'road.'+k:v for k,v in vars(self.surface).items() if torch.is_tensor(v)})
        self._tensors['owner_snapshot']=self.owners
        self._versions={k:(id(v),v._version) for k,v in self._tensors.items()}
        self.owner_counts={str(a):int((self.owners==a).sum()) for a in self.dynamic_actor_ids}

    def raw(self,raw):
        if raw not in self.headers:raise KeyError(raw)
        if raw not in self._raw:
            cam=make_camera(self.headers[raw])
            with torch.no_grad():env=self.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1).detach()
            self._raw[raw]=dict(cam=cam,env=env)
        return self._raw[raw]

    def _native_rows(self,camera,model,dynamic=False):
        rows=self.renderer.prepare_rasterization(camera,model,self.args,dynamic=dynamic)
        colour=rows['colors_precomp']
        if colour is None:
            direction=rows['means3D']-camera.camera_center;direction=direction/direction.norm(dim=-1,keepdim=True)
            colour=(self.renderer.eval_sh(self.degree,rows['shs'].transpose(1,2),direction)+.5).clamp_min(0)
        return dict(rows,means2D=torch.zeros_like(rows['means3D']),shs=None,colors_precomp=colour)

    def scene(self,raw,delete=False,delete_actor_ids=None,road_rgb=None):
        if type(delete) is not bool:raise TypeError('delete必须是bool')
        chosen=[37] if delete_actor_ids is None and delete else ([] if delete_actor_ids is None else delete_actor_ids)
        query=ActorEditQuery(self.dynamic_actor_ids,chosen);self.assert_unchanged()
        with torch.no_grad():
            data=self.raw(raw);camera=data['cam']
            s=self._native_rows(camera,self.static);d=self._native_rows(camera,self.dynamic,True)
            keep=torch.ones_like(self.owners,dtype=torch.bool)
            for actor in query.delete_actor_ids:keep&=self.owners!=actor
            d=dict(d,mask=d['mask']&keep)
            sf=s['features'].new_zeros((len(s['mask']),24));sf[:,:2]=s['features'][:,:2];sf[:,2]=1
            df=d['features'].new_zeros((len(d['mask']),24));df[:,:2]=d['features'][:,:2]
            for slot,actor in enumerate(self.dynamic_actor_ids,3):df[self.owners==actor,slot]=1
            rows=merge(dict(s,features=sf),dict(d,features=df));road_offset=len(rows['mask'])
            if self.surface is not None:
                x=self.surface;depth=(x.xyz@camera.world_view_transform[:3,:3]+camera.world_view_transform[3,:3])[:,2:3]
                extra=dict(means3D=x.xyz,means2D=torch.zeros_like(x.xyz),shs=None,colors_precomp=x.rgb,
                    features=torch.cat((depth,torch.ones_like(depth)),1),opacities=x.logits.sigmoid(),
                    scales=x.scales,rotations=x.rotations,cov3D_precomp=None,mask=x.active)
                if self.args.compute_cov3D_python:raise ValueError('当前完整C tuple只接受已核原scales/quaternion路径')
                extra['opacities']=self.mip(camera,extra,self.parent_indices)
                f=depth.new_zeros((len(depth),24));f[:,:2]=extra['features'];f[:,8]=1;f[:,9:12]=x.rgb
                extra=dict(extra,features=f)
        # 显式训练接口只允许外部新road RGB图；背景/车辆/Env始终detach。
        if self.surface is not None:
            if road_rgb is not None:
                if road_rgb.shape!=(40392,3) or road_rgb.dtype!=torch.float32 or road_rgb.device!=self.bg.device or not bool(torch.isfinite(road_rgb).all()):raise ValueError('road_rgb需finite CUDA float32[40392,3]')
                f=extra['features'].clone();f[:,9:12]=road_rgb
                extra=dict(extra,colors_precomp=road_rgb,features=f)
            rows=merge(rows,extra)
        elif road_rgb is not None:raise ValueError('无C几何时不能传road RGB')
        meta=dict(raw=raw,delete_actor_ids=list(query.delete_actor_ids),dynamic_actor_ids=self.dynamic_actor_ids,
            group_actor_ids=GROUP_ACTOR_IDS,static_rows=len(s['mask']),dynamic_rows=len(d['mask']),
            road_offset=road_offset,road_rows=40392 if self.surface is not None else 0,
            sole_scene_model=str(self.model),sole_scene_iteration=self.iteration,
            source_owner_identity_is_ancestry_not_validated_semantics=True,
            model_source_components=['Full5_static','Full5_dynamic','Full5_environment'],
            V44_or_C2_or_semantic_subset_model_loaded=False,C_geometry_complete=self.surface is not None,
            old_scene_parameters_frozen=True,newroad_RGB_override=road_rgb is not None,
            full5_native_SH_degree_for_all_rows=self.degree,precomputed_RGB_bridge_requires_native_parity=True)
        return dict(rows=rows,cam=camera,env=data['env'],bg=self.bg,meta=meta)

    def raster(self,camera,rows,background):
        from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
        tx=math.tan(-.5) if self.args.neg_fov else math.tan(camera.FoVx*.5)
        ty=math.tan(-.5) if self.args.neg_fov else math.tan(camera.FoVy*.5)
        settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),
            tanfovx=tx,tanfovy=ty,bg=background,scale_modifier=1.,viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,sh_degree=0,campos=camera.camera_center,prefiltered=False,debug=self.args.debug)
        _,rgb,features,radii=GaussianRasterizer(raster_settings=settings)(**rows)
        return dict(rgb=rgb,depth_sum=features[:1],alpha=features[1:2],extra=features[2:],radii=radii)

    def render(self,raw,scene):
        # 调用方若用外部road_rgb训练，可直接复用无no_grad的本函数；原模型全冻结。
        out=self.raster(scene['cam'],scene['rows'],scene['bg']);self.render_calls+=1
        out['rgb_premultiplied']=out['rgb'];out['env_rgb_T']=(1-out['alpha'])*scene['env'];out['rgb']=out['rgb']+out['env_rgb_T']
        out['render']=out['rgb'];out['group_alpha_T']=out['extra'][:7];out['owner_alpha_T']=out['group_alpha_T'][:6]
        out['newroad_rgb_T']=out['extra'][7:10];out['depth_m']=out['depth_sum']/out['alpha'].clamp_min(1e-8)/self.scale
        error=float((out['group_alpha_T'].sum(0)-out['alpha'][0]).abs().max())
        if error>2e-5:raise ValueError('单joint owner/road alpha不守恒')
        out['group_conservation_maxabs']=error;out['meta']=scene['meta'];return out

    def render_scene(self,raw,delete=False,**kwargs):return self.render(raw,self.scene(raw,delete=delete,**kwargs))

    def assert_unchanged(self,full=False):
        if any((id(v),v._version)!=self._versions[k] for k,v in self._tensors.items()):raise RuntimeError('冻结模型/身份/道路tensor被改写')
        if not torch.equal(actor_ids(self.dynamic),self.owners):raise RuntimeError('原source owner变更')
        for path,before in self._files.items():
            p=Path(path)
            if (p.stat().st_size,p.stat().st_mtime_ns)!=(before['bytes'],before['mtime_ns']) or (full and sha(p)!=before['sha256']):raise RuntimeError('原checkpoint/输入文件改变：'+path)
        return True

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(status='FULL5_ONLY_OPTIONAL_C_NATIVE_SCENE_ENGINEERING',model=str(self.model),iteration=self.iteration,
            static_rows=len(self.static._xyz),dynamic_rows=len(self.dynamic._xyz),source_owner_rows=self.owner_counts,
            source_owner_identity_is_ancestry_not_validated_semantics=True,dynamic_actor_ids=self.dynamic_actor_ids,
            group_actor_ids=GROUP_ACTOR_IDS,road_rows=40392 if self.surface is not None else 0,
            V44_or_C2_or_semantic_subset_model_loaded=False,one_joint_depth_sort=True,
            frozen_input_files=self._files,relative_mip=self.mip.receipt() if self.mip else None,
            original_Gaussian_parameter_updates=0,optimizer_created=False,render_calls=self.render_calls,
            quality_accepted=False,human_verdict=None)
