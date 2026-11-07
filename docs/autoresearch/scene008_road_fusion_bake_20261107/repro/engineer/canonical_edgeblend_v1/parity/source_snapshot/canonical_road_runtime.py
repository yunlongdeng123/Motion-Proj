"""Replay an explicit fixed world-road layer in the untouched V44 native scene.

New road is always present; delete changes only a37. Authoritative rendering
uses saved fp32 tuples, never a normal->quaternion reconstruction.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import subprocess
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):digest.update(block)
    return digest.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def gpu_preflight():
    query=lambda fields:subprocess.check_output(['nvidia-smi',fields,'--format=csv,noheader'],text=True).strip()
    gpu=query('--query-gpu=index,memory.total,memory.used,memory.free')
    apps=query('--query-compute-apps=pid,used_memory,process_name')
    ids=[line.split(',')[0].strip() for line in apps.splitlines() if line.strip()]
    processes=subprocess.check_output(['ps','-p',','.join(ids),'-o','pid,args'],text=True).strip() if ids else ''
    return dict(gpu_memory_MiB_csv=gpu,concurrent_compute_apps_csv=apps,concurrent_commands=processes,
        inspection_only=True,other_processes_stopped_or_modified=False)


class ExactSurface:
    def __init__(self,rt,asset):
        import torch
        cv=lambda key:torch.as_tensor(asset[key].copy(),device=rt.n.bg.device)
        self.xyz=cv('render_model_xyz');self.scaling=cv('render_scales_model')
        self.rotation=cv('render_rotation_wxyz');self.opacity=cv('render_opacity_logit')
        self._rgb=cv('render_colours_srgb');self.active=cv('active')
        self.dc=((self._rgb-.5)/.28209479177387814)[:,None]
        self._tensors=(self.xyz,self.scaling,self.rotation,self.opacity,self._rgb,self.active,self.dc)
        assert all(not t.requires_grad for t in self._tensors)
        self._versions=tuple((id(t),t._version) for t in self._tensors)
    def colors(self):return self._rgb
    def output_mask(self):return self.active
    def unchanged(self):return self._versions==tuple((id(t),t._version) for t in self._tensors)


def exact_relative_mip(module,asset_path,asset,policy):
    class ExactRelativeMip(module.RelativeMip):
        def __init__(self):
            self.parent_asset_path=Path(asset_path);self.parent_asset_sha256=sha(asset_path)
            self.parent_xyz=asset['parent_render_model_xyz'].copy()
            self.parent_scales=asset['parent_render_scales_model'].copy()
            self.parent_rotation=asset['parent_render_rotation_wxyz'].copy()
            self.neg_fov=bool(policy['legacy_neg_fov'])
            self.normalize_quaternion=bool(policy['normalize_quaternion'])
            self._geometry={};self._parent_factors={};self.records=[]
        def receipt(self):
            receipt=super().receipt();receipt.update(parent_geometry_source='canonical exact fp32 parent render tuple',
                parent_geometry_reconstruction=False)
            return receipt
    return ExactRelativeMip()


class CanonicalRoadRuntime:
    def __init__(self,manifest_path,verify_dependencies=True):
        self.manifest_path=Path(manifest_path).resolve();self.manifest_hash=sha(self.manifest_path)
        self.manifest=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest['status']!='GEOMETRY_SUPPORT_RESEARCH_CANDIDATE' or self.manifest['adoption_claim']:
            raise ValueError('Explicit geometry-support research-candidate label required')
        if not self.manifest['native_capture_verified']:raise ValueError('CPU preview is not a render-ready native tuple')
        self.asset_path=self.manifest_path.parent/self.manifest['new_road']['file']
        assert sha(self.asset_path)==self.manifest['new_road']['sha256']
        for item in self.manifest['replay_sources'].values():
            path=self.manifest_path.parent/item['file'];assert sha(path)==item['sha256'],str(path)
        if verify_dependencies:
            for item in self.manifest['runtime_dependencies'].values():
                path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
        with np.load(self.asset_path,allow_pickle=False) as z:self.asset={key:z[key].copy() for key in z.files}
        a=self.asset;n=len(a['active']);assert n==self.manifest['new_road']['rows'] and a['active'].all()
        for key,width in [('render_model_xyz',3),('render_scales_model',3),('render_rotation_wxyz',4),
                          ('render_opacity_logit',1),('render_colours_srgb',3)]:
            assert a[key].shape==(n,width) and a[key].dtype==np.float32 and np.isfinite(a[key]).all(),key
        assert np.all((a['render_colours_srgb']>=0)&(a['render_colours_srgb']<=1))
        assert len(a['parent_source_valid'])==199 and a['parent_source_valid'].shape[1]==10098
        assert np.array_equal(a['child_to_parent_ids'],np.repeat(np.arange(10098),4))
        source=Path(self.manifest['v44_runtime_path'])
        module=load_module('_canonical_pinned_v44',source)
        self.rt=module.V44Runtime(self.manifest['v44_contract_path'])
        assert self.rt.binding_sha256==self.manifest['frozen_v44_binding_sha256']
        self.extension=ExactSurface(self.rt,a);self.factor=2
        import torch
        self.parent_indices=torch.as_tensor(a['child_to_parent_ids'],device=self.rt.n.bg.device,dtype=torch.long)
        self.opacity_fn=None;policy=self.manifest['renderer_policy']
        if policy['relative_mip']:
            mip=load_module('_canonical_relative_mip',self.manifest_path.parent/policy['source_file'])
            self.opacity_fn=exact_relative_mip(mip,self.asset_path,a,policy)
        assert bool(self.rt.pipe.neg_fov)==bool(policy['legacy_neg_fov'])
        assert bool(self.rt.pipe.compute_cov3D_python)==bool(policy['normalize_quaternion'])

    def extension_rows(self,raw):
        rt=self.rt;camera=rt.raw(raw)['cam']
        rows=rt.road_kwargs(camera,self.extension,rt.pipe,sh_channels=1,precomputed_color=True)
        if self.opacity_fn is not None:rows=dict(rows,opacities=self.opacity_fn(camera,rows,self.parent_indices))
        assert self.extension.unchanged()
        rows=rt._tag(rows,'road');feature=rows['features'].clone()
        feature[:,18]=1;feature[:,19:22]=rows['colors_precomp']
        depth=(rows['means3D']@camera.world_view_transform[:3,:3]+camera.world_view_transform[3,:3])[:,2]/rt.scale
        feature[:,22]=depth;feature[:,23]=0
        return dict(rows,features=feature)

    def scene(self,raw,delete=False):
        rt=self.rt;scene=rt.scene(raw,delete=delete)
        rows=rt.merge(scene['rows'],self.extension_rows(raw));feature=rows['features'].clone()
        depth=(rows['means3D']@scene['cam'].world_view_transform[:3,:3]+scene['cam'].world_view_transform[3,:3])[:,2]/rt.scale
        feature[:,22]=depth*(feature[:,18]>0);feature[:,23]=depth*(feature[:,2]>0)
        rows=dict(rows,features=feature);scene=dict(scene,rows=rows)
        scene['meta'].update(canonical_newroad_enabled=True,parent_newroad_loaded=False,newroad_rows=len(self.asset['active']),
            canonical_newroad_sha256=sha(self.asset_path),keep_delete_only_a37=True,training_updates=0,
            renderer_relative_mip=self.opacity_fn is not None,candidate_not_adopted=True)
        return scene

    def render_scene(self,raw,delete=False):return self.rt.render(raw,self.scene(raw,delete=delete))

    def receipt(self):
        assert sha(self.manifest_path)==self.manifest_hash and sha(self.asset_path)==self.manifest['new_road']['sha256']
        assert self.extension.unchanged()
        return dict(status='GEOMETRY_SUPPORT_RESEARCH_CANDIDATE',manifest_sha256=self.manifest_hash,
            scene_manifest=str(self.manifest_path),newroad_asset_sha256=sha(self.asset_path),native=self.rt.receipt(),
            original_parameter_updates=0,canonical_parameter_updates=0,parent_newroad_loaded=False,newroad_always_enabled=True,
            edit_operator='physical a37 existence only',single_joint_sort=True,native_CUDA_modified=False,
            view_filter_override_enabled=self.opacity_fn is not None,
            view_filter_override_receipt=self.opacity_fn.receipt() if self.opacity_fn else None,
            material_role='persistent fitted appearance; not independently observed hidden texture',
            adoption_claim=False,known_unresolved=self.manifest['known_unresolved'])
