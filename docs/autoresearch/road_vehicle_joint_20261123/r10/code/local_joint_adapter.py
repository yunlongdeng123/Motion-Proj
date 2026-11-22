"""Target-free persistent new local road + original native joint depth sort.

Import/readiness is CPU-only. GPU construction requires enable_gpu=True.
Native retirement is a measured global-row sidecar, never owner-wide deletion.
"""
from pathlib import Path
import sys
import numpy as np
from local_gaussian_asset import ACTOR_SHA,FIELDS,read,sha,load_local_asset,export_colour_update,bank,view_matrix

PACKET_KEYS=('means3D','means2D','colors_precomp','shs','features','opacities','scales','rotations','cov3D_precomp','mask')

def readiness(asset_directory):
    try:
        asset=load_local_asset(asset_directory,verify_parent_pins=True)
        return dict(ready=True,CPU_only=True,GPU_calls=0,rows=len(asset.field['xyz']),new_local_asset=True,native_gate_mode=asset.manifest['native_gate'].get('mode','qualified_sparse_retirement'),native_gate_rows=len(asset.native_gate['global_row_ids']),fit_authorized=False)
    except (OSError,ValueError,KeyError,TypeError,AssertionError) as error:
        return dict(ready=False,CPU_only=True,GPU_calls=0,issues=[str(error) or type(error).__name__],fit_authorized=False)

class LocalJointAdapter:
    def __init__(self,asset_directory,output,*,enable_gpu=False):
        if not enable_gpu:raise RuntimeError('CPU-only prototype: root-scheduled GPU construction requires enable_gpu=True')
        self.asset=load_local_asset(asset_directory,verify_parent_pins=True);manifest=self.asset.manifest
        parent=read(manifest['parent_contract']['path'])
        assert len(parent['actors'])==1 and parent['actors'][0]['sha256']==ACTOR_SHA
        assert sha(parent['actors'][0]['asset'])==ACTOR_SHA
        sys.path.insert(0,manifest['r8_code'])
        from joint_identity_adapter import JointIdentityAdapter
        self.base=JointIdentityAdapter(parent,output);self.torch=self.base.torch;torch=self.torch
        selected={key:self.asset.field[key] for key in [*FIELDS,'active','source_id','teacher_prior_weight']}
        self.local=self.base.actor_backend.JointField(selected)
        for parameter in self.local.parameters():parameter.requires_grad_(False)
        model_from_patch=self.base.transform@self.asset.world_from_patch
        scale=np.linalg.norm(model_from_patch[:3,:3],axis=0);assert np.allclose(scale,self.base.meter_to_model,rtol=1e-7)
        self.model_from_patch=torch.as_tensor(model_from_patch.astype(np.float32),device='cuda')
        self.local_channel=5+len(self.base.actors);assert self.local_channel<24
        self.gate_ids=torch.as_tensor(self.asset.native_gate['global_row_ids'],device='cuda',dtype=torch.long)
        self.gate_values=torch.as_tensor(self.asset.native_gate['gate'],device='cuda',dtype=torch.float32)
        self.gate_proof=read(self.asset.directory/'native_gate_proof.json');self._rgb_guard=None;self._rgb_hook=None
        self.local_rgb_parameter=self.local.dc;self.ready=True

    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)

    def local_packet(self,raw):
        if self.local.dc.requires_grad and self._rgb_guard is None:raise RuntimeError('New local RGB training needs actual all-view K/body guard first')
        cam=self.base.cameras[raw]
        packet=self.base.actor_backend.Runtime.actor_rows(self.base.actor_math,self.local,self.model_from_patch,cam,False)
        depth=packet['means3D']@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2]
        packet['mask']=(depth>self.asset.manifest['near_m']*self.base.meter_to_model)&self.local.active
        packet=self.base._features(packet,raw,2);packet['features'][:,self.local_channel]=1
        assert all(key in packet for key in PACKET_KEYS) and packet['features'].shape[1]==24
        return packet

    def _native_gate(self,raw,packet):
        if not len(self.gate_ids):
            # Explicit no-retirement input preserves the prepared packet object.
            # It confers no K/body qualification on appended local Gaussians.
            assert self.gate_values.shape==(0,)
            return packet
        torch=self.torch;_,group,owners,gids,offset=self.base.context[raw];n=len(gids)
        assert offset==self.gate_proof['native_global_row_count']
        positions=torch.searchsorted(self.gate_ids,gids);safe=positions.clamp_max(len(self.gate_ids)-1)
        selected=(positions<len(self.gate_ids))&(self.gate_ids[safe]==gids)&(group==0)
        allowed=owners<=0
        for owner in self.asset.manifest['native_gate']['deleted_target_owner_ids']:allowed|=owners==owner
        assert allowed[selected].all() and not torch.isin(owners[selected],torch.as_tensor([spec['owner_id'] for spec,_ in self.base.actors],device='cuda')).any()
        assert not torch.isin(gids[selected],self.base.preserve).any()
        ids=torch.nonzero(selected).reshape(-1);copy=dict(packet);opacity=packet['opacities'].clone()
        opacity[ids]=opacity[ids]*self.gate_values[safe[ids]][:,None];copy['opacities']=opacity
        # base._prepare already retained the original native legacy-near mask.
        # No native mean/scale/camera or masked-out native row is reintroduced.
        return copy

    def packet(self,raw,component='shared'):
        if component=='local':return self.local_packet(raw)
        if component=='actor':return self.base.packet(raw,'actor')
        if component=='existing_road':return self.base.packet(raw,'road')
        if component=='road':return self.base.scene._bindings().merge_rows(self.base.packet(raw,'road'),self.local_packet(raw))
        if component not in ('shared','context'):raise ValueError('Unknown local joint component: '+component)
        packet=self._native_gate(raw,self.base.packet(raw,component))
        if component=='shared':packet=self.base.scene._bindings().merge_rows(packet,self.local_packet(raw))
        assert all(key in packet for key in PACKET_KEYS) and packet['features'].shape[1]==24
        return packet

    def render(self,raw,component='shared'):
        packet=self.packet(raw,component);cam=self.base.cameras[raw]
        out=self.base.scene._bindings().raster(cam,self.base.scene.pipe,0,packet,self.base.scene._background)
        self.base.calls+=1;self.base.trace.append(dict(call=self.base.calls,raw=raw,component=component,new_local_asset=True,single_joint_native_raster=True))
        rgb=out['rgb']
        if component in ('shared','context'):rgb=rgb+(1-out['alpha'])*self.base.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        local=out['extra'][self.local_channel-2]
        return dict(rgb=rgb,alpha=out['alpha'],depth_sum_m=out['depth_sum']/self.base.meter_to_model,road_alpha=out['extra'][0],local_road_alpha=local,existing_road_alpha=out['extra'][0]-local,actor_alpha=out['extra'][1],context_alpha=out['extra'][2],unified_actor_alpha=out['extra'][3:3+len(self.base.actors)])

    def enable_local_rgb(self,guard_npz,guard_receipt):
        """Only actual zero-K/body rows may receive local colour gradients."""
        guard=read(guard_receipt);arrays=bank(guard_npz)
        assert guard['complete'] and guard['decision']=='PASS' and guard['field_sha256']==self.asset.manifest['files']['local_field.npz']['sha256']
        assert guard['all_five_K_body_verified'] and sha(guard_npz)==guard['npz_sha256']
        mask=np.asarray(arrays['trainable_rgb_rows'],bool);assert mask.shape==(len(self.local.xyz),)
        frames=self.asset.manifest['frames'];assert np.asarray(arrays['frames']).tolist()==frames
        km=view_matrix(arrays['allview_K_mass'],len(mask),frames);bm=view_matrix(arrays['allview_body_mass'],len(mask),frames)
        assert np.array_equal(mask,(km==0).all(0)&(bm==0).all(0))
        self._rgb_guard=guard;allowed=self.torch.as_tensor(mask,device='cuda')
        if self._rgb_hook is not None:self._rgb_hook.remove()
        self.local.dc.requires_grad_(True);self._rgb_hook=self.local.dc.register_hook(lambda grad:grad*allowed[:,None])
        return dict(trainable_local_RGB_rows=int(mask.sum()),all_other_fields_frozen=True)

    def assert_fixed(self):
        self.base.assert_fixed();torch=self.torch
        for (spec,field),source in zip(self.base.actors,self.base.actor_sources):
            assert sha(spec['asset'])==ACTOR_SHA
            assert torch.equal(field.xyz,field.base_xyz) and torch.equal(field.opacity,field.base_opacity)
            assert not field.log_scales.any() and torch.equal(field.coefficients(),field.base_coefficients)
            assert torch.equal(field.rotation(),field.base_rotation)
            assert torch.equal(field.active,torch.as_tensor(source['active'],device='cuda').bool())
            for key,value in source.items():
                if key not in FIELDS:
                    actual=getattr(field,'meta_'+key);expected=torch.as_tensor(value,device='cuda')
                    assert actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected)
        assert not self.base.road_rgb_delta.any() and not self.base.road_tangent_log_sigma.any() and not self.base.road_opacity_delta.any()
        assert torch.equal(self.local.xyz,self.local.base_xyz) and torch.equal(self.local.opacity,self.local.base_opacity)
        assert not self.local.log_scales.any() and torch.equal(self.local.rotation(),self.local.base_rotation)

    def export(self,destination,training_record):
        self.assert_fixed();assert isinstance(training_record,dict) and int(training_record['actual_optimizer_updates'])>=0
        coefficients=self.local.coefficients().detach().cpu().numpy()
        return export_colour_update(self.asset,destination,coefficients,training_record)
