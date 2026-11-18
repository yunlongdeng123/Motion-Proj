"""受单GPU租约约束的固定道路原生上下文；导入本文件不加载Torch/native。

颜色变量直接进入当前原生Gaussian raster。相机、几何、opacity、全部scale、
原场景和Gaussian顺序固定；共享参数化与独立ci使用同一上下文。
"""
from pathlib import Path
import gc
import json
import sys
import numpy as np

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CPU_CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
GPU_CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
CANONICAL_ASSET=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
FRAMES=(86,110,134,179,198)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load_npz(path,keys=None):
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in (saved.files if keys is None else keys)}

def assert_gpu_lease(lease_file,lease_owner):
    if lease_file is None or not lease_owner:raise RuntimeError('native加载需要显式lease_file与lease_owner')
    lease=json.loads(Path(lease_file).read_text())
    actual=str(lease.get('owner','')).rstrip('/').split('/')[-1]
    expected=str(lease_owner).rstrip('/').split('/')[-1]
    if lease.get('gpu_authorized') is not True or lease.get('active') is not True or actual!=expected:
        raise RuntimeError('当前GPU租约不属于本调用者: '+str(lease))
    return lease


class NativeRoadContext:
    def __init__(self,scene,manifest,loader,asset_dir,frames,lease_file,lease_owner):
        import torch
        from PIL import Image
        self.scene=scene;self.manifest=manifest;self.loader=loader
        self.asset_dir=Path(asset_dir);self.frames=tuple(map(int,frames))
        self.lease_file=Path(lease_file);self.lease_owner=lease_owner
        self.road=load_npz(manifest.road);self.provenance=load_npz(self.asset_dir/'provenance.npz')
        self.world_row_ids=np.flatnonzero(self.road['active'])
        self.active_world_rows=self.world_row_ids
        self.world_xyz=self.provenance['world_xyz'][self.world_row_ids].astype(np.float64)
        self.provider=next(value for value in scene._providers if value.kind=='road')
        self._packets={};self._packet_snapshots={};self._baselines={};self._joint_baselines={}
        self.K_masks={};self.native_calls=0;self._closed=False
        if not torch.equal(scene._background,torch.zeros_like(scene._background)):
            raise ValueError('固定颜色容量上下文要求黑底预乘road RGB')
        frame_records={int(row['key']):row for row in manifest.data['frames']}
        for raw in self.frames:
            camera=self.camera(raw)
            packet=self.provider.rows(raw,camera,scene.pipe)
            self._packets[raw]={key:value.detach().clone() if torch.is_tensor(value) else value for key,value in packet.items()}
            self._packet_snapshots[raw]={key:self._packets[raw][key].clone() for key in
                ('means3D','opacities','scales','rotations','mask') if torch.is_tensor(self._packets[raw].get(key))}
            labels=np.asarray(Image.open(frame_records[raw]['instance_mask']).resize((camera.image_width,camera.image_height),Image.Resampling.NEAREST))
            self.K_masks[raw]=(labels>0)&(labels!=10)
        self.base_rgb=self._packets[self.frames[0]]['colors_precomp'].detach().clone()
        self.c0=self.base_rgb
        if len(self.base_rgb)!=len(self.world_row_ids):raise ValueError('native road顺序/active数不一致')
        source=torch.as_tensor(self.road['reference_srgb'][self.world_row_ids],device=self.base_rgb.device,dtype=self.base_rgb.dtype)
        if not torch.equal(source,self.base_rgb):raise ValueError('canonical c0与实际native shader RGB不一致')
        for raw in self.frames:
            if not torch.equal(self._packets[raw]['colors_precomp'],self.base_rgb):raise ValueError('road shader颜色随相机变化，不满足共同ci上下文')

    def require_lease(self):
        if self._closed:raise RuntimeError('native context已关闭')
        return assert_gpu_lease(self.lease_file,self.lease_owner)

    def camera(self,raw):return self.scene.camera_provider(int(raw))

    def base_packet(self,raw):
        """只读固定模板；调用者只能在新dict里替换colors_precomp。"""
        return dict(self._packets[int(raw)])

    def render_road(self,raw,colors=None):
        self.require_lease();raw=int(raw)
        if colors is None:colors=self.base_rgb
        if colors.shape!=self.base_rgb.shape:raise ValueError('颜色须为Nactive×3，使用固定native行顺序')
        camera=self.camera(raw);packet=dict(self._packets[raw],colors_precomp=colors)
        self.native_calls+=1
        return self.scene._bindings().raster(camera,self.scene.pipe,0,packet,self.scene._background)

    def baseline(self,raw):
        import torch
        raw=int(raw)
        if raw not in self._baselines:
            with torch.no_grad():value=self.render_road(raw,self.base_rgb)
            self._baselines[raw]={name:value[name].detach().clone() for name in ('rgb','alpha')}
        return self._baselines[raw]

    @property
    def fixed_alpha(self):return {raw:self.baseline(raw)['alpha'].detach() for raw in self.frames}

    def render_joint(self,raw,colors=None):
        import torch
        self.require_lease();raw=int(raw)
        scene=self.scene;values,owners,camera,_=scene._rows(raw,(10,),(),False)
        values=dict(values,mask=values['mask']&~scene._contains(owners,(10,)))
        if colors is not None:
            n=len(self.base_rgb);offset=len(values['means3D'])-n
            if colors.shape!=self.base_rgb.shape or not torch.equal(values['means3D'][offset:],self._packets[raw]['means3D']):
                raise ValueError('联合road尾部与固定native顺序不一致')
            updated=values['colors_precomp'].clone();updated[offset:]=colors
            values=dict(values,colors_precomp=updated)
        self.native_calls+=1
        out=scene._bindings().raster(camera,scene.pipe,0,values,scene._background)
        environment=(1-out['alpha'])*scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        return dict(rgb=out['rgb']+environment,alpha=out['alpha'],raw=out)

    def joint_baseline(self,raw):
        import torch
        raw=int(raw)
        if raw not in self._joint_baselines:
            with torch.no_grad():self._joint_baselines[raw]=self.render_joint(raw)['rgb'].detach().clone()
        return self._joint_baselines[raw]

    def attribute(self,raw,regions):
        import torch
        self.require_lease()
        from drivingeditor_dynamic_edit.native_row_contribution import attribute_prepared_rows
        packet=self.base_packet(raw);owners=torch.zeros(len(self.base_rgb),device=self.base_rgb.device,dtype=torch.long)
        return attribute_prepared_rows(self.scene,packet,owners,self.camera(raw),regions)

    def audit_K(self,colors,output=None):
        import torch
        rows=[]
        with torch.no_grad():
            for raw in self.frames:
                mask=torch.as_tensor(self.K_masks[raw],device=self.base_rgb.device)
                difference=(self.render_joint(raw,colors)['rgb']-self.joint_baseline(raw)).abs().amax(0)
                rows.append(dict(raw=raw,pixels=int(mask.sum()),RGB_max=float(difference[mask].max()),changed_pixels=int((difference[mask]>0).sum())))
        result=dict(all_exact=all(row['RGB_max']==0 for row in rows),protected_pixel_frames=sum(row['pixels'] for row in rows),rows=rows)
        if output:write_json(output,result)
        if not result['all_exact']:raise RuntimeError('完整joint K保护失败: '+str(result))
        return result

    def assert_frozen(self):
        import torch
        for raw,fields in self._packet_snapshots.items():
            for key,value in fields.items():
                if not torch.equal(self._packets[raw][key],value):raise RuntimeError('固定road模板变化: '+str((raw,key)))
        self.scene.assert_persistent_unchanged()

    def save_asset(self,colors,eligible_world,output_dir,*,training_record=None):
        self.require_lease();self.assert_frozen()
        from drivingeditor_dynamic_edit.persistent_road_handoff import save_persistent_road_handoff
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
        mask=np.asarray(eligible_world,bool)
        if mask.shape!=(len(self.road['active']),):raise ValueError('save_asset需要完整world域eligible mask')
        if np.any(mask&~self.road['active']):raise ValueError('不能通过颜色导出激活新road行')
        rgb=colors.detach().cpu().numpy()
        if rgb.shape!=self.road['reference_srgb'][self.world_row_ids].shape or not np.isfinite(rgb).all() or np.any((rgb<0)|(rgb>1)):
            raise ValueError('导出有效shader RGB须有限[0,1]')
        road={key:value.copy() for key,value in self.road.items()}
        local=mask[self.world_row_ids]
        road['reference_srgb'][self.world_row_ids[local]]=rgb[local]
        if not np.array_equal(rgb[~local],self.road['reference_srgb'][self.world_row_ids[~local]]):raise ValueError('有效模型改动了未授权row颜色')
        np.savez_compressed(output/'road.npz',**road)
        p={key:value.copy() for key,value in self.provenance.items()}
        p['capacity_source_prior_reference_srgb']=self.road['reference_srgb'].copy()
        p['capacity_effective_rgb']=road['reference_srgb'].copy();p['effective_row_rgb']=road['reference_srgb'].copy()
        p['capacity_color_eligible']=mask.copy();p['capacity_prior_ledger_immutable']=np.asarray(True)
        p['asset_mode']=np.asarray('fixed_geometry_native_color_capacity')
        np.savez_compressed(output/'provenance.npz',**p)
        record=dict(training_record or {},GPU_executed=True,native_raster_calls_in_context=self.native_calls,
            only_road_field_modified='reference_srgb',all_original_observed_prior_ledger_preserved=True)
        write_json(output/'receipt.json',record)
        save_persistent_road_handoff(self._manifest_path,self._proof_path,output/'road.npz',output/'persistent',training_record=record)
        return output

    def validate_public_reload(self,colors,asset_dir,output):
        import torch
        self.require_lease()
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        asset=Path(asset_dir)
        scene,_,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
        provider=next(value for value in scene._providers if value.kind=='road');rows=[]
        with torch.no_grad():
            for raw in self.frames:
                expected=self.render_road(raw,colors);camera=scene.camera_provider(raw)
                actual=scene._bindings().raster(camera,scene.pipe,0,provider.rows(raw,camera,scene.pipe),scene._background)
                rows.append(dict(raw=raw,RGB_max=float((expected['rgb']-actual['rgb']).abs().max()),
                    alpha_max=float((expected['alpha']-actual['alpha']).abs().max()),
                    RGB8_max=int((torch.round(expected['rgb']*255)-torch.round(actual['rgb']*255)).abs().max())))
        scene.assert_persistent_unchanged()
        result=dict(public_loader=True,GPU_executed=True,tolerance=1e-6,rows=rows)
        write_json(output,result)
        if any(row['RGB_max']>1e-6 or row['alpha_max']>1e-6 for row in rows):raise RuntimeError('公共loader重载偏差超过1e-6')
        del provider,scene;gc.collect();torch.cuda.empty_cache()
        return result

    def close(self):
        if self._closed:return
        import torch
        self._packets.clear();self._packet_snapshots.clear();self._baselines.clear();self._joint_baselines.clear()
        self.scene=None;self.provider=None;self.base_rgb=None;self.c0=None
        self._closed=True;gc.collect();torch.cuda.empty_cache()


def load_context(*,asset_dir=CANONICAL_ASSET,manifest=None,proof=None,frames=FRAMES,
                 lease_file=None,lease_owner=None):
    assert_gpu_lease(lease_file,lease_owner)
    import torch
    sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    asset=Path(asset_dir);manifest=Path(manifest or asset/'persistent/scene.json');proof=Path(proof or asset/'persistent/proof.json')
    scene,description,loader=load_persistent_handoff(manifest,proof)
    if Path(description.road).resolve()!=(asset/'road.npz').resolve():raise ValueError('manifest road与canonical asset不一致')
    for component in (scene.static,scene.dynamic,scene.environment):
        for parameter in component.parameters() if hasattr(component,'parameters') else ():
            parameter.requires_grad_(False)
    context=NativeRoadContext(scene,description,loader,asset,frames,lease_file,lease_owner)
    context._manifest_path=manifest;context._proof_path=proof
    return context
