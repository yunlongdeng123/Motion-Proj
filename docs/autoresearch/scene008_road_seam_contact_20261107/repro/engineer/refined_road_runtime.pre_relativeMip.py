"""Independent seam/contact engineering wrapper; frozen parent is never edited.

Existing24 feature ABI is retained: original4groups, newroad18/19:22,
newroad/asset centre-depth-in-metres22/23. Oldroad is totalroad minus newroad.
No training, per-view culling policy, screen mask, height offset, or AA is added.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
from world_surface_sampling import read_surface,resample_surface,summarize,array_hash,sha


class _RefinedSurface:
    def __init__(self,rt,parent_extension,cpu,parent_indices,factor):
        import torch
        device=rt.n.bg.device
        cv=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
        self.xyz=cv(cpu['model_xyz']);self.scaling=cv(cpu['gaussian_sigma_m']*rt.scale)
        ids=torch.as_tensor(parent_indices,dtype=torch.long,device=device)
        self.rotation=parent_extension.rotation.index_select(0,ids).clone()
        self._rgb=parent_extension.colors().index_select(0,ids).clone()
        self.opacity=parent_extension.opacity.index_select(0,ids).clone()
        self.active=torch.ones(len(ids),dtype=torch.bool,device=device)
        self.dc=((self._rgb-.5)/.28209479177387814)[:,None]
        self.parent_indices=ids
        self._tensors=(self.xyz,self.scaling,self.rotation,self._rgb,self.opacity,self.active,self.dc)
        assert all(not t.requires_grad for t in self._tensors)
        self._versions=tuple((id(t),t._version) for t in self._tensors)
    def colors(self):return self._rgb
    def output_mask(self):return self.active
    def unchanged(self):return self._versions==tuple((id(t),t._version) for t in self._tensors)


class RefinedRoadRuntime:
    def __init__(self,manifest_path,factor=2,verify_dependencies=False,opacity_fn=None,opacity_policy_id=None):
        if factor not in (1,2,4):raise ValueError('factor must be explicit1/2/4')
        if opacity_fn is not None and (factor==1 or not opacity_policy_id):
            raise ValueError('Opacity hook needs named reviewed policy and cannot alter exact n1 baseline')
        self.manifest_path=Path(manifest_path).resolve();manifest=json.loads(self.manifest_path.read_text())
        source=self.manifest_path.parent/'baked_road_runtime.py'
        assert sha(source)==manifest['wrapper']['sha256']
        spec=importlib.util.spec_from_file_location('_seam_contact_parent_baked',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.parent=module.BakedRoadRuntime(self.manifest_path,verify_dependencies=verify_dependencies)
        self.base=self.parent.base;self.rt=self.parent.rt;self.factor=factor
        self.opacity_fn=opacity_fn;self.opacity_policy_id=opacity_policy_id or 'copy_parent_no_AA'
        self.surface_path=self.parent.extension.path;self.surface_sha=sha(self.surface_path)
        self.cpu_parent=read_surface(self.surface_path);self.cpu_child=resample_surface(self.cpu_parent,factor)
        self.summary=summarize(self.cpu_parent,self.cpu_child,factor)
        self.summary.update(parent_asset_sha256=self.surface_sha,parent_manifest_sha256=sha(self.manifest_path),
            factor1_physical_path='direct parent R0 rows, no property reconstruction',opacity_policy_id=self.opacity_policy_id,
            feature_ABI=24,source_parent_indices_sha256=array_hash(self.cpu_child['parent_indices']))
        self._cpu_hash={k:array_hash(v) for k,v in self.cpu_child.items()}
        self.refined=None if factor==1 else _RefinedSurface(self.rt,self.parent.extension,self.cpu_child,self.cpu_child['parent_indices'],factor)

    def extension_rows(self,raw,opacity_fn=None):
        import torch
        callback=opacity_fn or self.opacity_fn
        if self.factor==1:
            if callback is not None:raise ValueError('n1 must remain exact parent')
            rows=self.base.extension_rows(raw)
        else:
            data=self.rt.raw(raw)
            rows=self.rt.road_kwargs(data['cam'],self.refined,self.rt.pipe,sh_channels=1,precomputed_color=True)
            if callback is not None:
                # Functional hook only. No AA policy is supplied by this module.
                alpha=callback(data['cam'],rows,self.refined.parent_indices)
                if alpha.shape!=rows['opacities'].shape or not bool(torch.isfinite(alpha).all()) or bool(((alpha<0)|(alpha>1)).any()):
                    raise ValueError('Opacity callback must return finite linear alpha[N,1] in[0,1]')
                assert self.refined.unchanged(),'Callback mutated persistent newroad geometry/material'
                rows=dict(rows,opacities=alpha)
            rows=self.rt._tag(rows,'road')
            rows['features'][:,18]=1;rows['features'][:,19:22]=rows['colors_precomp']
        camera=self.rt.raw(raw)['cam'];feature=rows['features'].clone()
        depth=(rows['means3D']@camera.world_view_transform[:3,:3]+camera.world_view_transform[3,:3])[:,2]/self.rt.scale
        feature[:,22]=depth;feature[:,23]=0
        rows=dict(rows,features=feature)
        return rows

    def scene(self,raw,delete=False):
        if self.factor==1:
            scene=self.base.scene(raw,delete=delete,extension=True,retire=False,material=False)
        else:
            scene=self.rt.scene(raw,delete=delete)
            scene=dict(scene,rows=self.rt.merge(scene['rows'],self.extension_rows(raw)))
        rows=dict(scene['rows']);feature=rows['features'].clone()
        # Exact expression used by the parent's diagnose_layers.py.
        depth=(rows['means3D']@scene['cam'].world_view_transform[:3,:3]+scene['cam'].world_view_transform[3,:3])[:,2]/self.rt.scale
        feature[:,22]=depth*(feature[:,18]>0);feature[:,23]=depth*(feature[:,2]>0)
        rows['features']=feature;scene=dict(scene,rows=rows)
        scene['meta'].update(refinement_factor=self.factor,newroad_rows=len(self.cpu_child['parent_indices']),
            extension_rows=len(self.cpu_child['parent_indices']),R0_extension_enabled=True,
            newroad_parent_sha256=self.surface_sha,opacity_policy_id=self.opacity_policy_id,
            same_world_refinement_in_keep_delete=True,screen_space_hole_policy=False,training_updates=0)
        return scene

    def render_scene(self,raw,delete=False):
        out=self.rt.render(raw,self.scene(raw,delete=delete))
        out['newroad_alpha_T']=out['extra'][16];out['newroad_rgb_T']=out['extra'][17:20]
        # Keep raw differences: tiny negative subtraction roundoff is not clipped.
        out['oldroad_alpha_T']=out['group_alpha_T'][2]-out['newroad_alpha_T']
        out['oldroad_rgb_T']=out['group_rgb_T'][2]-out['newroad_rgb_T']
        out['newroad_centre_depth_T_m']=out['extra'][20];out['asset_centre_depth_T_m']=out['extra'][21]
        out['components']={
            'legacy':dict(alpha_T=out['group_alpha_T'][3],rgb_T=out['group_rgb_T'][3]),
            'oldroad':dict(alpha_T=out['oldroad_alpha_T'],rgb_T=out['oldroad_rgb_T']),
            'newroad':dict(alpha_T=out['newroad_alpha_T'],rgb_T=out['newroad_rgb_T']),
            'asset':dict(alpha_T=out['group_alpha_T'][0],rgb_T=out['group_rgb_T'][0]),
            'a37':dict(alpha_T=out['group_alpha_T'][1],rgb_T=out['group_rgb_T'][1]),
            'env':dict(transmittance_T=1-out['alpha'][0],rgb_T=out['env_rgb_T'])}
        return out

    def receipt(self):
        assert sha(self.surface_path)==self.surface_sha
        assert all(array_hash(self.cpu_child[k])==h for k,h in self._cpu_hash.items())
        assert self.refined is None or self.refined.unchanged()
        return dict(status='ENGINEERING_CONTROL_NOT_ADOPTED',refinement=self.summary,
            parent=self.parent.receipt(),old_model_parameter_updates=0,newroad_training_updates=0,
            new_height_evidence=False,geometry_operation='fixed parent tangent patch subcells; no vertical bias',
            keep_delete_operator='a37 physical existence only',
            diagnostic_semantics=dict(legacy='frozen native_static; includes its reconstructed static vehicles',
                oldroad='totalroad-newroad raw same-sort contribution',env='(1-alpha)*EnvLight; T is not a geometry alpha',
                depth_tags='alphaT-weighted Gaussian centre depths in metres, not surface first-hit'),
            AA_implemented=False,opacity_hook_policy_id=self.opacity_policy_id,
            automatic_factor4_fallback=False,shutdown_performed=False)
