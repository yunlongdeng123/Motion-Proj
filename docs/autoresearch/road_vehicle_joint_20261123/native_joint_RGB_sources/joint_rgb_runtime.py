"""Only gathered road RGB and canonical actor DC are differentiable.

All original native rows, centres, scales, opacities, rotations and trajectories
remain fixed. One shared depth sort supplies joint RGB and instance readouts.
"""
from pathlib import Path
import numpy as np
from joint_identity_adapter import JointIdentityAdapter
from joint_rgb_common import load,read,sha,write,all4_and_sample,linear_to_srgb_numpy


class JointColourRuntime(JointIdentityAdapter):
    def __init__(self,cfg,base,output):
        super().__init__(base,output)
        torch=self.torch;self.fit_cfg=cfg;self.display_parent=False
        self.road_source=load(Path(base['road_asset'])/'road.npz')
        self.road_provenance=load(Path(base['road_asset'])/'provenance.npz')
        assert np.array_equal(self.road_base['colors_precomp'].cpu().numpy(),self.road_source['reference_srgb'][self.road_active]),'Parent shader RGB differs from road.reference_srgb'
        assert self.road_provenance['effective_row_rgb'].tobytes()==self.road_source['reference_srgb'].tobytes()
        self.atlas=load(cfg['source']['field']);self.uv=load(cfg['row_uv'])['row_uv']
        guard_dir=Path(cfg['colour_guard']);gr=read(guard_dir/'receipt.json');guard=load(guard_dir/'colour_guard.npz')
        assert gr['complete'] and gr['updates']==0 and gr['source_RGBA_exact'] is True
        assert gr['source_sha256']==cfg['source']['sha256'] and gr['guard_sha256']==sha(guard_dir/'colour_guard.npz')
        assert np.array_equal(guard['frames'],cfg['frames']) and np.array_equal(guard['road_active_world_rows'],self.road_active)
        self.guard=guard;self.guard_receipt=gr;self.guard_sha256=gr['guard_sha256']
        permitted=self.atlas['permission_mask']&(self.atlas['source_kind']>0)&np.isfinite(self.atlas['atlas_linear']).all(-1)
        row_linear,row_available=all4_and_sample(self.atlas['atlas_linear'],self.uv[self.road_active],permitted)
        available=load(cfg['road_eligible'])['world_row_ids'];eligible=np.isin(self.road_active,available)&row_available&~guard['road_Kpositive_union']
        if 'guarded_world_rows' in self.road_provenance:eligible&=~self.road_provenance['guarded_world_rows'][self.road_active]
        self.road_positions=np.flatnonzero(eligible);self.road_world_ids=self.road_active[eligible]
        assert len(self.road_positions)>0
        self.road_ix=torch.as_tensor(self.road_positions,device='cuda',dtype=torch.long)
        initial=np.clip(linear_to_srgb_numpy(row_linear[eligible]),0,1).astype(np.float32)
        self.road_initial=torch.as_tensor(initial,device='cuda');self.road_parameter=torch.nn.Parameter(self.road_initial.clone())
        self.actor_parameters=[];self.actor_initial=[];self.actor_allowed=[];self.actor_ix=[]
        for spec,field in self.actors:
            key=spec['name'];m=field.active.detach().cpu().numpy()&~guard[key+'_Kpositive_union']&guard[key+'_strict_train_Vpositive_union']
            ids=np.flatnonzero(m);ix=torch.as_tensor(ids,device='cuda',dtype=torch.long)
            initial=field.dc.detach()[ix].clone();self.actor_allowed.append(ids);self.actor_ix.append(ix);self.actor_initial.append(initial)
            self.actor_parameters.append(torch.nn.Parameter(initial.clone()))
        self.targets={};self.target_receipts=[]
        gain={int(x['raw']):float(x['gain']) for x in read(cfg['exposure_gains'])['records']}
        real=self.atlas['real_locks'].astype(np.float32);influence=np.clip(self.atlas['effective_teacher_blend'],0,1)
        source=np.stack([real,(1-real)*(1-influence),(1-real)*influence],-1).astype(np.float32)
        self.row_source,_=all4_and_sample(source,self.uv[self.road_world_ids],permitted)
        self.row_source/=np.maximum(self.row_source.sum(-1,keepdims=True),1e-12)
        for raw in self.frames:
            cache_path=Path(base['camera_audit_cache'])/f'{raw:03d}/view_cache.npz';cache=load(cache_path)
            target,valid=all4_and_sample(self.atlas['atlas_linear'],cache['atlas_uv_map'],permitted)
            sw,_=all4_and_sample(source,cache['atlas_uv_map'],permitted)
            weight=sw@np.array([cfg['source_weights'][k] for k in ['same_world','similar_reference','teacher_continuous']],np.float32)
            masks=load(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz')
            support=valid&cache['geometry_valid']&cache['projection_valid']&masks['RGB_valid']&~masks['protected']
            V=masks['actor_visible']&masks['RGB_valid']&~masks['protected'] if raw in cfg['train_frames'] else np.zeros_like(support)
            proxy=masks['actor_visible_proxy']&masks['RGB_valid']&~masks['protected']
            self.targets[raw]=dict(target=torch.as_tensor(target.transpose(2,0,1),device='cuda'),
                weight=torch.as_tensor(weight,device='cuda'),support=torch.as_tensor(support,device='cuda'),
                V=torch.as_tensor(V,device='cuda'),proxy=torch.as_tensor(proxy,device='cuda'),
                photo=torch.as_tensor(cache['photo_pinhole_srgb'].transpose(2,0,1),device='cuda'),gain=gain[raw],
                K=torch.as_tensor(masks['protected'],device='cuda'),source_weight=sw,
                target_numpy=target,support_numpy=support)
            self.target_receipts.append(dict(raw=raw,source_material_pixels=int(support.sum()),strict_train_V_pixels=int(V.sum()),
                proxy_pixels=int(proxy.sum()),gain=gain[raw],view_cache_sha256=sha(cache_path),
                mask_sha256=sha(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz')))
        self.trainable=[self.road_parameter]+self.actor_parameters
        assert all(not p.requires_grad for _,field in self.actors for p in field.parameters())
        write(self.output/'colour_scope.json',dict(road_world_rows=self.road_world_ids.tolist(),
            actor_DC_rows={spec['name']:len(ids) for (spec,_),ids in zip(self.actors,self.actor_allowed)},
            guard_sha256=self.guard_sha256,source_sha256=cfg['source']['sha256'],target_domains=self.target_receipts,
            all_geometry_fixed=True,native_context_fixed=True,all_five_Kpositive_rows_frozen=True,
            source1_atlas_locks_unchanged=True,teacher_weight_is_continuous_influence=True))

    def road_packet(self,raw):
        p=dict(self.road_templates[raw])
        if hasattr(self,'road_parameter') and not self.display_parent:
            p['colors_precomp']=self.road_base['colors_precomp'].index_copy(0,self.road_ix,self.road_parameter)
        return p

    def actor_packet(self,index,raw):
        p=super().actor_packet(index,raw)
        if not hasattr(self,'actor_parameters') or self.display_parent:return p
        torch=self.torch;spec,field=self.actors[index];pose=self.poses[(spec['name'],raw)];cam=self.cameras[raw]
        scale=pose[:3,:3].norm(dim=0).mean();R=pose[:3,:3]/scale
        local=torch.nn.functional.normalize(torch.nn.functional.normalize(p['means3D']-cam.camera_center,dim=1)@R,dim=1)
        dc=field.dc.detach().index_copy(0,self.actor_ix[index],self.actor_parameters[index])
        coeff=torch.cat([dc[:,None],field.base_coefficients[:,1:]],1)
        p['colors_precomp']=(torch.einsum('ni,nic->nc',self.actor_math.m.basis3(local),coeff)+.5).clamp_min(0)
        return p

    def linear(self,x):
        z=x.clamp_min(0);return self.torch.where(z<=.04045,z/12.92,((z+.055)/1.055).pow(2.4))

    def losses(self,raw,joint,road):
        t=self.targets[raw];torch=self.torch
        material=t['support']&(road['alpha'].squeeze().detach()>self.fit_cfg['road_material_loss_requires_standalone_alpha_gt'])
        jointmask=t['support']&(joint['road_alpha'].detach()>self.fit_cfg['joint_road_loss_requires_actual_road_alphaT_gt'])
        assert material.any() and jointmask.any() and t['V'].any(),'Training domain empty'
        def weighted(residual,mask):
            w=t['weight'][mask];return (residual[:,mask].square().mean(0)*w).sum()/w.sum().clamp_min(1e-12)
        jm=self.linear(joint['rgb'])*t['gain'];straight=road['rgb']/road['alpha'].clamp_min(1e-6)
        material_loss=weighted(self.linear(straight)*t['gain']-t['target'],material)
        jointroad_loss=weighted(jm-t['target'],jointmask)
        actor_loss=(jm[:,t['V']]-self.linear(t['photo'])[:,t['V']]*t['gain']).square().mean()
        prior=(self.road_parameter-self.road_initial).square().mean()
        for p,initial in zip(self.actor_parameters,self.actor_initial):
            if p.numel():prior=prior+((p-initial)*.28209479177387814).square().mean()
        loss=material_loss+jointroad_loss+actor_loss+self.fit_cfg['RGB_prior']*prior
        metrics=dict(material_MSE=float(material_loss.detach()),joint_road_MSE=float(jointroad_loss.detach()),
            joint_actor_real_MSE=float(actor_loss.detach()),prior_RGB_MSE=float(prior.detach()),
            material_pixels=int(material.sum()),joint_road_pixels=int(jointmask.sum()),actor_strictV_pixels=int(t['V'].sum()))
        assert torch.isfinite(loss);return loss,metrics

    def assert_colour_only(self):
        self.assert_fixed();torch=self.torch
        assert not self.road_tangent_log_sigma.any() and not self.road_opacity_delta.any()
        for _,field in self.actors:
            assert torch.equal(field.dc,field.base_coefficients[:,0])
            assert not field.log_scales.any() and torch.equal(field.opacity,field.base_opacity)
        assert sha(self.fit_cfg['source']['field'])==self.fit_cfg['source']['sha256']

    def export_asset(self,destination,steps,updates):
        torch=self.torch;destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        road={k:v.copy() for k,v in self.road_source.items()};road['reference_srgb'][self.road_world_ids]=self.road_parameter.detach().cpu().numpy()
        from joint_rgb_common import srgb_to_linear_numpy
        road['reference_linear']=srgb_to_linear_numpy(road['reference_srgb']).astype(np.float32)
        road_dir=destination/'road';road_dir.mkdir();np.savez_compressed(road_dir/'road.npz',**road)
        provenance={k:v.copy() for k,v in self.road_provenance.items()}
        # capacity_effective_rgb is an immutable historical ledger in this parent.
        # The public provider reads reference_srgb; effective_row_rgb is current.
        provenance['effective_row_rgb']=road['reference_srgb'].copy()
        np.savez_compressed(road_dir/'provenance.npz',**provenance)
        from drivingeditor_dynamic_edit.persistent_road_handoff import save_persistent_road_handoff
        save_persistent_road_handoff(self.cfg['scene_manifest'],self.cfg['scene_proof'],road_dir/'road.npz',road_dir/'persistent',
            training_record=dict(method='actual shared road/actor RGB only',logical_steps=steps,actual_optimizer_updates=updates,source_sha256=self.fit_cfg['source']['sha256']))
        road_unchanged={k:bool(np.array_equal(v,road[k]) and v.tobytes()==road[k].tobytes()) for k,v in self.road_source.items() if k not in {'reference_srgb','reference_linear'}}
        assert all(road_unchanged.values())
        unchanged_colour=np.ones(len(road['active']),bool);unchanged_colour[self.road_world_ids]=False
        assert road['reference_srgb'][unchanged_colour].tobytes()==self.road_source['reference_srgb'][unchanged_colour].tobytes()
        assert provenance['capacity_effective_rgb'].tobytes()==self.road_provenance['capacity_effective_rgb'].tobytes()
        hard=self.road_active[self.guard['road_Kpositive_union']]
        assert road['reference_srgb'][hard].tobytes()==self.road_source['reference_srgb'][hard].tobytes()
        actors=[];actor_invariants={};joint=dict(self.cfg);joint['road_asset']=str(road_dir);joint['scene_manifest']=str(road_dir/'persistent/scene.json');joint['scene_proof']=str(road_dir/'persistent/proof.json')
        for index,((spec,field),source) in enumerate(zip(self.actors,self.actor_sources)):
            value={k:v.copy() for k,v in source.items()}
            # Keep original optional metadata exactly. Never call JointField.export.
            original=load(spec['asset']);value={k:v.copy() for k,v in original.items()}
            value['coefficients'][self.actor_allowed[index],0]=self.actor_parameters[index].detach().cpu().numpy()
            exact={k:bool(v.dtype==value[k].dtype and v.shape==value[k].shape and v.tobytes()==value[k].tobytes()) for k,v in original.items() if k!='coefficients'}
            exact['SH_degree1_to3']=value['coefficients'][:,1:].tobytes()==original['coefficients'][:,1:].tobytes();assert all(exact.values())
            km=self.guard[spec['name']+'_Kpositive_union'];assert value['coefficients'][km,0].tobytes()==original['coefficients'][km,0].tobytes()
            folder=destination/'actors'/spec['name'];folder.mkdir(parents=True);path=folder/'actor.npz';np.savez_compressed(path,**value)
            actors.append(dict(spec,asset=str(path),sha256=sha(path)));actor_invariants[spec['name']]=exact
        joint['actors']=actors;write(destination/'joint_contract.json',joint)
        np.savez_compressed(destination/'colour_lineage.npz',road_world_rows=self.road_world_ids,road_row_uv=self.uv[self.road_world_ids],
            road_source_weight=self.row_source,source_names=np.asarray(['same_world_photo','similar_photo','teacher_continuous_influence']),
            **{spec['name']+'_canonical_DC_rows':ids for (spec,_),ids in zip(self.actors,self.actor_allowed)})
        proof=dict(schema='R8.persistent_joint_RGB_asset.proof.v1',complete=True,baseline_road=str(Path(self.cfg['road_asset'])/'road.npz'),
            baseline_road_sha256=sha(Path(self.cfg['road_asset'])/'road.npz'),road_sha256=sha(road_dir/'road.npz'),
            source_sha256=self.fit_cfg['source']['sha256'],source_field_sha256=self.fit_cfg['source']['sha256'],source_field=self.fit_cfg['source']['field'],guard_sha256=self.guard_sha256,K_guard_sha256=self.guard_sha256,
            logical_steps=steps,actual_optimizer_updates=updates,road_noncolour_byteexact=road_unchanged,actor_invariants=actor_invariants,
            actor_only_DC_modified=True,Kpositive_row_colours_byteexact=True,native_context_fixed=True,
            road_effective_RGB_equals_reference_sRGB=True,reference_linear_is_EOTF_reference_sRGB=True,
            outside_trainable_colour_rows_byteexact=True,historical_capacity_effective_rgb_preserved=True,
            geometry_sigma_opacity_normal_tracks_fixed=True,teacher_new_calls=0,
            joint_contract_sha256=sha(destination/'joint_contract.json'),fresh_GPU_reload='PENDING separate verify process',whole_scene_restoration_accepted=False)
        write(destination/'proof.json',proof);self.assert_colour_only();return proof
