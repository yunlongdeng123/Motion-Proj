"""One native depth-sorted packet for fixed context, RoadRows and canonical actors.

Import is CPU-only. Construction is an explicit GPU action. No optimizer,
teacher, Gaussian insertion, asset write, or camera/trajectory parameter exists.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys,numpy as np


def read(path):return json.loads(Path(path).read_text())
def bank(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def subset(rows,ids):
    n=len(rows['means3D'])
    return {k:v[ids] if hasattr(v,'ndim') and v.ndim and len(v)==n else v for k,v in rows.items()}


class JointNativeAdapter:
    """Contract-driven actor list; no actor/frame identity is hardcoded in math."""
    def __init__(self,contract,output):
        import torch
        self.torch=torch;self.cfg=read(contract) if isinstance(contract,(str,Path)) else contract
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.calls=0;self.backward_calls=0;self.frames=tuple(self.cfg['frames']);self.trace=[]
        torch.set_num_threads(2)
        sys.path.insert(0,str(Path(self.cfg['repo'])/'scripts/worldsim_v82'))
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        from drivingeditor_dynamic_edit.v44_sh_asset import basis3
        self.scene,self.manifest,self.loader=load_persistent_handoff(self.cfg['scene_manifest'],self.cfg['scene_proof'])
        data=dict(self.manifest.data);data['render_wh']=self.cfg['render_wh']
        headers=SceneManifest(data,Path(self.cfg['scene_manifest']).parent).camera_headers()
        self.cameras={raw:make_native_camera(headers[raw]) for raw in self.frames}
        self.scene.camera_provider=lambda raw:self.cameras[int(raw)]
        for model in [self.scene.static,self.scene.dynamic,self.scene.environment]:
            if hasattr(model,'parameters'):
                for parameter in model.parameters():parameter.requires_grad_(False)
            for value in vars(model).values():
                if torch.is_tensor(value) and value.is_leaf:value.requires_grad_(False)
        self.transform=np.asarray(self.manifest.world_to_model,np.float64)
        road=bank(Path(self.cfg['road_asset'])/'road.npz');self.road_active=np.flatnonzero(road['active'])
        self.meter_to_model=float(road['meter_to_model'])
        self.provider=next(x for x in self.scene._providers if x.kind=='road')
        initial=self.provider.rows(self.frames[0],self.cameras[self.frames[0]],self.scene.pipe)
        self.road_base={k:initial[k].detach().clone() for k in ['means3D','scales','rotations','opacities','colors_precomp']}
        n=len(self.road_active);assert len(initial['means3D'])==n
        self.road_local_mask=torch.zeros(n,device='cuda')
        self.road_rgb_delta=torch.nn.Parameter(torch.zeros_like(initial['colors_precomp']),requires_grad=False)
        self.road_tangent_log_sigma=torch.nn.Parameter(torch.zeros((n,2),device='cuda'),requires_grad=False)
        self.road_opacity_delta=torch.nn.Parameter(torch.zeros_like(initial['opacities']),requires_grad=False)
        # Keep the selected field module immutable; do not mislabel a final asset as its source init.
        module_path=Path(self.cfg['actor_field_module']);sys.path.insert(0,str(module_path.parent))
        self.actor_backend=load_module(module_path,'_r8_immutable_actor_field')
        self.actor_math=SimpleNamespace(m=SimpleNamespace(basis3=basis3))
        self.actors=[];self.poses={};self.actor_sources=[]
        for spec in self.cfg['actors']:
            asset=Path(spec['asset']);assert sha(asset)==spec['sha256'],'Actor baseline SHA changed'
            a=bank(asset);n=len(a['xyz'])
            for key,shape in [('xyz',(n,3)),('scales',(n,3)),('rotation',(n,4)),('coefficients',(n,16,3)),('opacity',(n,))]:
                assert a[key].shape==shape and a[key].dtype==np.float32 and np.isfinite(a[key]).all(),key
            assert (a['scales']>0).all() and ((a['opacity']>=0)&(a['opacity']<=1)).all()
            # Optional lineage fields are diagnostic only. Canonical geometry/SH stays exact.
            a.setdefault('source_id',np.zeros(n,np.int64));a.setdefault('teacher_prior_weight',np.zeros(n,np.float32))
            field=self.actor_backend.JointField(a)
            for parameter in field.parameters():parameter.requires_grad_(False)
            field.xyz.requires_grad_(False);field.raw_rotation.requires_grad_(False)
            self.actors.append((spec,field));self.actor_sources.append(a)
            for raw in self.frames:
                obs=read(Path(self.cfg['observations'])/f'{raw:03d}.json')
                box=next(x for x in obs['boxes'] if x['track']==spec['track'])
                pose=self.transform@np.asarray(box['world_from_actor']).reshape(4,4)
                self.poses[(spec['name'],raw)]=torch.as_tensor(pose.astype(np.float32),device='cuda')
        self.plan=read(self.cfg['context_plan'])
        self.remove=torch.as_tensor(self.plan['remove_native_global_row_ids'],device='cuda',dtype=torch.long)
        self.preserve=torch.as_tensor(self.plan.get('preserved_native_context_global_row_ids',[]),device='cuda',dtype=torch.long)
        assert not torch.isin(self.remove,self.preserve).any()
        self.relabels=[]
        for spec in self.cfg.get('owner_relabels',[]):
            a=bank(spec['path']);assert sha(spec['path'])==spec['sha256']
            self.relabels.append((torch.as_tensor(a[spec.get('key','global_row_ids')],device='cuda',dtype=torch.long),int(spec['owner_id'])))
        self.context={};self.road_templates={};self.reference_context={};self.camera_audit=[]
        self._prepare()
        self.frozen_centers=[(spec['name'],field.xyz.detach().clone()) for spec,field in self.actors]
        self.frozen_rotations=[field.raw_rotation.detach().clone() for _,field in self.actors]
        self.source_receipt=dict(actor_assets=[dict(name=s['name'],asset=s['asset'],sha256=s['sha256'],rows=len(a['xyz']),active=int(a['active'].sum())) for (s,_),a in zip(self.actors,self.actor_sources)],
            road_asset=self.cfg['road_asset'],road_SHA=sha(Path(self.cfg['road_asset'])/'road.npz'),road_active_rows=len(self.road_active),
            context_plan_SHA=sha(self.cfg['context_plan']),meter_to_model=self.meter_to_model,
            renderer='one persistent metric-native raster, native prefix legacy-near preserved',
            centers_fixed=True,normal_frame_fixed=True,rigid_pose_fixed=True,stage_A_updates=0,teacher_calls=0)

    def _prepare(self):
        torch=self.torch
        for raw in self.frames:
            cam=self.cameras[raw]
            with torch.no_grad():
                values,owners,_,_=self.scene._rows(raw,tuple(x['owner_id'] for x,_ in self.actors),(),False)
                n=len(values['means3D']);gids=torch.arange(n,device='cuda');owners=owners.clone()
                road=self.provider.rows(raw,cam,self.scene.pipe);nr=len(road['means3D']);offset=n-nr
                assert torch.equal(values['means3D'][offset:],road['means3D']),'Road provider is not the exact native suffix'
                for key in ['means3D','scales','rotations','opacities','colors_precomp']:
                    assert torch.equal(road[key],self.road_base[key]),'Road field changed across cameras: '+key
                self.road_templates[raw]=road
                for ids,owner in self.relabels:
                    pick=torch.isin(gids,ids)
                    assert not ((owners[pick]>0)&(owners[pick]!=owner)).any(),'Owner sidecar contradicts another retained actor'
                    owners[pick]=owner
                keep=values['mask']&~torch.isin(gids,self.remove)
                for owner in self.cfg.get('exclude_native_owner_ids',[]):keep&=owners!=owner
                idx=torch.nonzero(keep).reshape(-1)
                packet=subset(values,idx);packet['mask']=torch.ones(len(idx),device='cuda',dtype=torch.bool)
                group=(idx>=offset).long() # road=1; all retained native context=0
                self.context[raw]=(packet,group,owners[idx],gids[idx],offset)
                # Replay old selected r3 context preparation independently: original active packing then exact retirement.
                active=torch.nonzero(values['mask']).reshape(-1);packed=subset(values,active)
                kept=~torch.isin(active,self.remove)
                for owner in self.cfg.get('exclude_native_owner_ids',[]):kept&=owners[active]!=owner
                ref=subset(packed,kept);ref['mask']=torch.ones(len(ref['means3D']),device='cuda',dtype=torch.bool)
                self.reference_context[raw]=(ref,owners[active][kept],active[kept])
            cache=bank(Path(self.cfg['camera_audit_cache'])/f'{raw:03d}/view_cache.npz')
            obs=read(Path(self.cfg['observations'])/f'{raw:03d}.json')
            camera_model=cam.world_view_transform.detach().cpu().numpy().T.astype(np.float64)
            physical=camera_model@self.transform;physical[:3]/=self.meter_to_model
            K=np.array([[float(cam.fx),0,float(cam.cx)],[0,float(cam.fy),float(cam.cy)],[0,0,1]])
            frame=next(x for x in self.manifest.data['frames'] if int(x['key'])==raw)
            self.camera_audit.append(dict(raw=raw,timestamp_micros=obs['timestamp_micros'],source_time_s=frame['source_time_s'],native_model_time=frame['model_time'],
                native_image_name=frame['image_name'],original_image_path=obs['image_path'],K_max_abs_vs_R7=float(abs(K-cache['camera_K']).max()),
                world_to_camera_max_abs_vs_R7=float(abs(physical-cache['world_to_camera']).max()),
                reference_K=cache['camera_K'].tolist(),actual_K=K.tolist(),rigid_pose_source='model_from_world @ same-raw observation.box.world_from_actor'))

    def road_packet(self,raw):
        torch=self.torch;p=dict(self.road_templates[raw]);mask=self.road_local_mask[:,None]
        tangent=self.road_tangent_log_sigma*mask;delta=torch.cat([tangent,torch.zeros_like(tangent[:,:1])],1)
        p['scales']=p['scales']*delta.exp()
        p['opacities']=(p['opacities']+self.road_opacity_delta*mask).clamp(0,1)
        p['colors_precomp']=p['colors_precomp']+self.road_rgb_delta*mask
        return p

    def actor_packet(self,index,raw):
        spec,field=self.actors[index];cam=self.cameras[raw];pose=self.poses[(spec['name'],raw)]
        p=self.actor_backend.Runtime.actor_rows(self.actor_math,field,pose,cam,False)
        # Generic depth guard is explicit in this experiment contract, never inferred from actor identity.
        depth=p['means3D']@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2]
        p['mask']=(depth>float(spec['minimum_camera_depth_model']))&field.active
        return p

    def _features(self,p,raw,group):
        torch=self.torch;p=dict(p);features=torch.zeros((len(p['means3D']),24),device='cuda',dtype=p['means3D'].dtype)
        cam=self.cameras[raw];features[:,0]=p['means3D']@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2]
        features[:,1]=1
        # 2 road, 3 actor, 4 frozen context, then per actor; output extra starts at feature2.
        if isinstance(group,int):features[:,group]=1
        else:
            features[:,2]=(group==1).float();features[:,4]=(group==0).float()
        p['features']=features
        return p

    def packet(self,raw,component='shared'):
        if component=='road':return self._features(self.road_packet(raw),raw,2)
        if component=='actor':
            rows=None
            for i,_ in enumerate(self.actors):
                p=self._features(self.actor_packet(i,raw),raw,3)
                rows=p if rows is None else self.scene._bindings().merge_rows(rows,p)
            return rows
        context,group,owners,gids,offset=self.context[raw]
        p=dict(context);roadpos=torch_ids=self.torch.nonzero(group==1).reshape(-1)
        current=self.road_packet(raw);worldidx=gids[roadpos]-offset
        for key in ['colors_precomp','scales','opacities']:
            p[key]=p[key].index_copy(0,roadpos,current[key][worldidx])
        p=self._features(p,raw,group)
        if component=='context':return p
        for i,_ in enumerate(self.actors):p=self.scene._bindings().merge_rows(p,self._features(self.actor_packet(i,raw),raw,3))
        return p

    def render(self,raw,component='shared'):
        packet=self.packet(raw,component);return self.raster_packet(raw,packet,component,component in ('shared','context'))

    def raster_packet(self,raw,packet,tag,environment):
        cam=self.cameras[raw];out=self.scene._bindings().raster(cam,self.scene.pipe,0,packet,self.scene._background)
        self.calls+=1;self.trace.append(dict(call=self.calls,raw=raw,tag=tag,rows=len(packet['means3D']),single_native_raster=True))
        rgb=out['rgb']
        if environment:
            env=self.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            rgb=rgb+(1-out['alpha'])*env
        return dict(rgb=rgb,alpha=out['alpha'],depth_sum_m=out['depth_sum']/self.meter_to_model,
                    road_alpha=out['extra'][0],actor_alpha=out['extra'][1],context_alpha=out['extra'][2])

    def legacy_reference(self,raw,actor_only=False):
        """Actual old selected-r3 render method, without its unrelated training caches."""
        assert len(self.actors)==1,'This legacy equivalence oracle is only for the declared single-actor control'
        backend=self.actor_backend;ref=object.__new__(backend.Runtime);spec,field=self.actors[0]
        # Old render consumes only these fields. No init is relabeled, no Runtime.__init__ cache/optimization runs.
        ref.actor=field;ref.m=SimpleNamespace(basis3=self.actor_math.m.basis3);ref.blank=self.torch.zeros(3,device='cuda');ref.calls=0
        env={raw:self.scene.environment(self.cameras[raw].get_world_directions(False).permute(1,2,0)).permute(2,0,1)}
        ref.e=SimpleNamespace(cameras={raw:self.cameras[raw]},poses={raw:self.poses[(spec['name'],raw)]},scene=self.scene,env=env,scale=self.meter_to_model)
        p,owner,gids=self.reference_context[raw]
        ref.context={raw:(p,owner,self.torch.full_like(owner,-1),gids)};ref.K_preserve=self.preserve
        out=backend.Runtime.render(ref,raw=raw,actor_only=actor_only)
        self.calls+=ref.calls;self.trace.append(dict(call=self.calls,raw=raw,tag='legacy_r3_actor' if actor_only else 'legacy_r3_shared',single_native_raster=True))
        return out

    def assert_fixed(self):
        torch=self.torch
        for (name,xyz),rotation,(_,field) in zip(self.frozen_centers,self.frozen_rotations,self.actors):
            assert torch.equal(xyz,field.xyz) and torch.equal(rotation,field.raw_rotation),'Actor canonical center/normal frame changed'
        self.scene.assert_persistent_unchanged()
