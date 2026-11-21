"""Fixed-budget real-GS surface extension: activation, residual append, coverage, color."""
from pathlib import Path
import copy,gc,hashlib,json,os,random,subprocess,sys,time,traceback
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_surface_adaptive_20261121_r1';OLD=C/'road_common1152_20261121'
PARENT=OLD/'formal/r3/G_NUMERIC/asset'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
Q=C/'road_teacher_detail_cpu_20261119_r1';B=Q/'gpu_20261119/native_capacity_20261120_r1'
FRAMES=(86,110,134,179,198);OWNER='road_surface_adaptive'
sys.path[:0]=[str(REPO/'scripts/worldsim_v82'),str(C/'road_capacity_gpu_20261118_r1/code'),
              str(B/'code'),str(Q/'gpu_20261119/field_training/code')]
import torch
from PIL import Image
from field_training_state import cpu_clone,rng_state,restore_rng,max_tree_difference,balanced_schedule
from native_capacity_loss import NativeCapacityLoss

LEDGER={};EXPORTS=[];CURRENT=None;PRIOR_CALLS=0
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(v,q);q.replace(p)
def tensor(a,dtype=torch.float32):return torch.as_tensor(np.array(a,copy=True),device='cuda',dtype=dtype)
def status(phase,**kw):write(ROOT/'status.json',dict(schema='surface_adaptive_status.v1',state='running',status='running',phase=phase,complete=False,PID=os.getpid(),updated_at=time.time(),**kw))
def count(name,n=1,role='formal'):LEDGER[role+':'+name]=LEDGER.get(role+':'+name,0)+n
def account():write(ROOT/'resource_ledger.json',dict(counters=LEDGER,native_calls=PRIOR_CALLS+(CURRENT.native_calls if CURRENT else 0),power_operations=0,updated_at=time.time()))
def close_ctx():
    global CURRENT,PRIOR_CALLS
    if CURRENT is not None:PRIOR_CALLS+=CURRENT.native_calls;CURRENT.close();CURRENT=None
    account();gc.collect();torch.cuda.empty_cache()
def choose(d,*keys):
    for k in keys:
        if k in d:return d[k]
    raise KeyError('required candidate field; choices='+str(keys))
def current_contract():return read(ROOT/'contract.json')

def context(asset):
    global CURRENT
    close_ctx()
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
    from road_capacity_native import NativeRoadContext
    asset=Path(asset);scene,m,loader=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
    for obj in (scene.static,scene.dynamic,scene.environment):
        for p in obj.parameters() if hasattr(obj,'parameters') else ():p.requires_grad_(False)
    data=dict(m.data);data['render_wh']=[1152,768];headers=SceneManifest(data,asset/'persistent').camera_headers();cams={}
    def camera(raw):
        if raw not in cams:cams[raw]=make_native_camera(headers[raw])
        return cams[raw]
    scene.camera_provider=camera
    CURRENT=NativeRoadContext(scene,m,loader,asset,FRAMES,ROOT/'gpu_lease.json',OWNER)
    CURRENT._manifest_path=asset/'persistent/scene.json';CURRENT._proof_path=asset/'persistent/proof.json'
    return CURRENT

def pin(parent,asset,record):
    from drivingeditor_dynamic_edit.persistent_road_handoff import save_persistent_road_handoff
    save_persistent_road_handoff(Path(parent)/'persistent/scene.json',Path(parent)/'persistent/proof.json',asset/'road.npz',asset/'persistent',training_record=record)
    # New artifacts default to the actual1152 camera, with every proof binding renewed.
    scene=asset/'persistent/scene.json';proof=asset/'persistent/proof.json';gate=asset/'persistent/static_opacity.pt'
    d=read(scene);d['render_wh']=[1152,768];write(scene,d);h=sha(scene)
    g=torch.load(gate,map_location='cpu',weights_only=False);g['binding']['manifest_sha256']=h;save(gate,g)
    p=read(proof);p['manifest']['sha256']=h;p['gates']['checkpoint_binding']['manifest_sha256']=h;p['gates']['checkpoint']['sha256']=sha(gate);write(proof,p)

def initial_rgb(world):
    from scipy.spatial import cKDTree
    r=load(PARENT/'road.npz');p=load(PARENT/'provenance.npz');permission=load(OLD/'common_permission/teacher_evidence.npz')
    ids=permission['active_world_rows'][permission['common_write_active']]
    distance,index=cKDTree(p['world_xyz'][ids]).query(world,k=1)
    return r['reference_srgb'][ids[index]],ids[index],distance

def candidate_spec(path,activation=False,limit=None):
    d=load(path)
    if activation:
        ids=choose(d,'existing_world_rows','existing_world_row_ids','world_row_ids','proposed_world_rows').astype(np.int64)
        base=load(PARENT/'road.npz');world=load(PARENT/'provenance.npz')['world_xyz'][ids]
        assert not base['active'][ids].any() and len(np.unique(ids))==len(ids)
        sigma=base['gaussian_sigma_m'][ids].copy();normal=base['normal_world'][ids].copy()
    else:
        world=choose(d,'world_xyz','append_world_xyz','xyz_world').astype(np.float64)
        ids=None;normal=choose(d,'normal_world','append_normal_world').astype(np.float64)
        sigma=choose(d,'scales_m','initial_sigma_m','sigma_m').astype(np.float64)
        if sigma.ndim==1:sigma=np.repeat(sigma[:,None],3,axis=1)
    n=len(world)
    upper=choose(d,'max_sigma_m','tangent_sigma_max_m','scale_upper_m').astype(np.float64)
    if upper.ndim==1:upper=np.repeat(upper[:,None],2,axis=1)
    if upper.shape[1]==3:upper=upper[:,:2]
    # The provider fixes tangent axes by shortest-arc q. A spherical cap is yaw-safe.
    cap=np.min(upper,axis=1);initial=np.min(sigma[:,:2],axis=1)
    if not activation:sigma[:,:2]=initial[:,None]
    assert np.isfinite(world).all() and np.isfinite(normal).all() and np.isfinite(sigma).all()
    assert np.all(sigma>0) and np.all(cap>=np.max(sigma[:,:2],axis=1)-1e-10)
    rgb,donor,distance=initial_rgb(world)
    spec=dict(ids=ids,world=world,normal=normal,sigma=sigma,cap=np.repeat(cap[:,None],2,axis=1),
        minimum=sigma[:,:2]*.5,rgb=rgb,donor=donor,distance=distance,
        triangle=d.get('triangle_id',d.get('triangle_ids',np.full(n,-1,np.int64))),
        barycentric=d.get('barycentric',d.get('barycentric_weights',np.full((n,3),np.nan))),
        triangle_world_rows=d.get('triangle_world_rows',np.full((n,3),-1,np.int64)),
        triangle_vertex_world=d.get('triangle_vertex_world',np.full((n,3,3),np.nan)),
        inferred=d.get('geometry_inferred_from_fixed_measurements',np.zeros(n,bool)),
        uv=d.get('atlas_uv',d.get('uv',np.full((n,2),np.nan))),source=str(path),source_SHA256=sha(path))
    if limit is not None:
        for k,v in list(spec.items()):
            if isinstance(v,np.ndarray):spec[k]=v[:limit]
    if not activation:
        assert np.isfinite(spec['barycentric']).all() and np.isfinite(spec['triangle_vertex_world']).all()
        assert np.all(spec['barycentric']>=-1e-8) and np.max(np.abs(spec['barycentric'].sum(1)-1))<1e-7
        reconstructed=np.einsum('ni,nij->nj',spec['barycentric'],spec['triangle_vertex_world'])
        assert np.max(np.abs(reconstructed-spec['world']))<1e-6,'new geometry must lie on the pinned surface triangles'
    return spec

def extend_provenance(p,n,world,normal,rgb,donor,distance,triangle,bary,uv):
    m=len(world);out={}
    for k,v in p.items():
        if v.ndim and v.shape[0]==n:
            shape=(m,*v.shape[1:]);fill=False if v.dtype.kind=='b' else -1 if v.dtype.kind in 'iu' and v.dtype.kind=='i' else 0
            extra=np.full(shape,fill,dtype=v.dtype);out[k]=np.concatenate([v,extra])
        else:out[k]=v.copy()
    out['world_xyz'][n:]=world;out['world_keys'][n:]=np.c_[np.full(m,-1),-1-np.arange(m)]
    out['atlas_uv'][n:]=uv;out['row_rgb'][n:]=rgb
    for k in ['effective_row_rgb','effective_model_xyz','effective_sigma_m','effective_road_opacity']:
        if k not in out:continue
    for k in ['source','confidence','known']:
        if k in out:out[k][n:]=0
    for k in ['unknown','surface_supported','legal_write','surface_interpolated']:
        if k in out:out[k][n:]=True
    if 'observed_reference_srgb' in out:out['observed_reference_srgb'][n:]=np.nan
    for k in ['source_world_row','source_raw','parent_world_row_id']:
        if k in out:out[k][n:]=-1
    for k in ['source_distance_m','source_uv_native']:
        if k in out:out[k][n:]=np.nan
    # Novel Gaussian rows are inferred geometry and neighbor-initialized material,
    # never promoted to a real observed source or an atlas grid anchor.
    out['adaptive_append_new_geometry']=np.r_[np.zeros(n,bool),np.ones(m,bool)]
    out['adaptive_color_donor_world_row']=np.r_[np.full(n,-1,np.int64),donor]
    out['adaptive_color_donor_distance_m']=np.r_[np.zeros(n),distance]
    out['adaptive_triangle_id']=np.r_[np.full(n,-1,np.int64),triangle]
    out['adaptive_barycentric']=np.r_[np.zeros((n,3)),bary]
    out['adaptive_geometry_kind']=np.r_[np.zeros(n,np.uint8),np.full(m,2,np.uint8)]
    return out

def build_asset(parent,out,activation=None,append=None):
    out=Path(out)
    if (out/'persistent/proof.json').exists():return out
    if out.exists():raise FileExistsError('preserve incomplete asset '+str(out))
    r=load(Path(parent)/'road.npz');p=load(Path(parent)/'provenance.npz');n=len(r['active']);parent_active=r['active'].copy()
    actids=np.zeros(0,np.int64) if activation is None else activation['ids'];m=0 if append is None else len(append['world'])
    if len(actids):
        assert not r['active'][actids].any();r['active'][actids]=True
        # Existing inactive geometry and source labels stay intact; material init is explicit.
        r['reference_srgb'][actids]=activation['rgb']
    if m:
        a=append;T=r['model_from_world'];new={
            'active':np.ones(m,bool),'model_xyz':a['world']@T[:3,:3].T+T[:3,3],
            'normal_world':a['normal'],'gaussian_sigma_m':a['sigma'],
            'reference_srgb':a['rgb'].astype(np.float32),'road_opacity':np.full(m,.1,np.float32),
            'minimum_model_depth':np.zeros(m,np.float32)}
        for k,v in list(r.items()):
            if v.ndim and v.shape[0]==n:
                if k not in new:raise KeyError('unregistered road row field '+k)
                r[k]=np.concatenate([v,np.asarray(new[k],dtype=v.dtype)])
        p=extend_provenance(p,n,a['world'],a['normal'],a['rgb'],a['donor'],a['distance'],a['triangle'],a['barycentric'],a['uv'])
        p['adaptive_triangle_world_rows']=np.r_[np.full((n,3),-1,np.int64),a['triangle_world_rows']]
        p['adaptive_triangle_vertex_world']=np.r_[np.zeros((n,3,3)),a['triangle_vertex_world']]
    else:p['adaptive_append_new_geometry']=np.zeros(n,bool)
    if 'adaptive_activate_existing' not in p:p['adaptive_activate_existing']=np.zeros(n+m,bool)
    p['adaptive_activate_existing'][actids]=True
    if 'adaptive_effective_surface_supported' not in p:p['adaptive_effective_surface_supported']=p['surface_supported'].copy()
    if 'adaptive_fixed_measurement_recertified' not in p:p['adaptive_fixed_measurement_recertified']=np.zeros(n+m,bool)
    p['adaptive_effective_surface_supported'][actids]=True
    if activation is not None:p['adaptive_fixed_measurement_recertified'][actids]=activation['inferred']
    if m:p['adaptive_effective_surface_supported'][n:]=True
    for key,value in [('effective_active',r['active']),('effective_row_rgb',r['reference_srgb']),('effective_model_xyz',r['model_xyz']),('effective_sigma_m',r['gaussian_sigma_m']),('effective_road_opacity',r['road_opacity'])]:p[key]=value.copy()
    p['asset_mode']=np.asarray('adaptive_surface_ordinary_GS')
    assert len(r['active'])==n+m
    for k,v in load(Path(parent)/'road.npz').items():
        if v.ndim and v.shape[0]==n:
            assert np.array_equal(r[k][:n][parent_active],v[parent_active]),'parent active rows changed at initialization '+k
    out.mkdir(parents=True);np.savez_compressed(out/'road.npz',**r);np.savez_compressed(out/'provenance.npz',**p)
    counts=dict(parent_total_rows=n,total_rows=n+m,parent_active_rows=int(parent_active.sum()),active_rows=int(r['active'].sum()),activate_existing=len(actids),append_new_geometry=m)
    pin(parent,out,dict(stage='geometry_initialization',point_counts=counts,new_geometry_is_not_activation=True))
    write(out/'extension_receipt.json',dict(complete=True,point_counts=counts,parent=str(parent),parent_road_SHA256=sha(Path(parent)/'road.npz'),road_SHA256=sha(out/'road.npz'),
        provenance_SHA256=sha(out/'provenance.npz'),activation_source=None if activation is None else activation['source'],
        activation_source_SHA256=None if activation is None else activation['source_SHA256'],
        append_source=None if append is None else append['source'],append_source_SHA256=None if append is None else append['source_SHA256'],
        surface_certificate_SHA256=sha(ROOT/'proposal/surface_certificate.npz')))
    return out

def geometry_render(ctx,raw,opacity=None,scales=None,colors=None,joint=False,feature_ids=None):
    if joint:
        rows,owners,camera,_=ctx.scene._rows(raw,(10,),(),False)
        rows=dict(rows,mask=rows['mask']&~ctx.scene._contains(owners,(10,)));offset=len(rows['means3D'])-len(ctx.base_rgb)
        assert torch.equal(rows['means3D'][offset:],ctx.base_packet(raw)['means3D'])
    else:rows=ctx.base_packet(raw);camera=ctx.camera(raw);offset=0
    n=len(ctx.base_rgb);pos=torch.arange(offset,offset+n,device='cuda')
    rows={k:(v.detach() if torch.is_tensor(v) and k!='means2D' else v) for k,v in rows.items()}
    if opacity is not None:rows['opacities']=rows['opacities'].index_copy(0,pos,opacity.reshape(-1,1))
    if scales is not None:rows['scales']=rows['scales'].index_copy(0,pos,scales)
    if colors is not None:rows['colors_precomp']=rows['colors_precomp'].index_copy(0,pos,colors)
    if feature_ids is not None:
        flags=torch.zeros(len(rows['means3D']),device='cuda');flags[offset+feature_ids]=1
        f=torch.zeros((len(flags),24),device='cuda');f[:,:2]=rows['features'][:,:2];f[:,2]=flags;rows['features']=f
    ctx.require_lease();ctx.native_calls+=1;out=ctx.scene._bindings().raster(camera,ctx.scene.pipe,0,rows,ctx.scene._background)
    rgb=out['rgb']
    if joint:rgb=rgb+(1-out['alpha'])*ctx.scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    result=dict(rgb=rgb,alpha=out['alpha'].squeeze())
    if feature_ids is not None:result['new_weight']=out['extra'][0]
    return result

def active_positions(ctx,ids):
    pos=np.searchsorted(ctx.active_world_rows,ids);assert np.array_equal(ctx.active_world_rows[pos],ids)
    return tensor(pos,torch.long)

class Coverage:
    def __init__(self,ctx,act,app=None,role='formal',previous=None):
        self.ctx=ctx;self.act=act;self.app=app;self.role=role;self.cursor=0;self.trace=[];self.schedule=balanced_schedule(600,101)
        self.binding=dict(asset_road=sha(ctx.asset_dir/'road.npz'),activation=act['source_SHA256'],
            append=None if app is None else app['source_SHA256'],contract=sha(ROOT/'contract.json'),
            domains={str(r):sha(ROOT/'proposal/domains'/f'{r:03d}.npz') for r in (86,110)})
        # Preserve the exact evaluated native values of every frozen old row.
        native=ctx.base_packet(86)
        if native.get('scales') is None:raise RuntimeError('coverage requires native differentiable scale ABI, not precomputed covariance')
        self.base_op=native['opacities'].detach().reshape(-1).clone();self.base_scale=native['scales'].detach().clone()
        self.params={};self.specs={};groups=[]
        original_n=len(load(PARENT/'road.npz')['active'])
        for name,spec,ids in [('act',act,act['ids']),('app',app,None if app is None else np.arange(original_n,original_n+len(app['world'])) )]:
            if spec is None or not len(spec['world']):continue
            pos=active_positions(ctx,ids);op=torch.nn.Parameter(torch.logit(self.base_op[pos].clamp(1e-4,.999)))
            sig=torch.nn.Parameter(torch.log(self.base_scale[pos,:2]/float(ctx.road['meter_to_model'])))
            self.params[name+'_opacity']=op;self.params[name+'_sigma']=sig
            self.specs[name]=(pos,tensor(spec['minimum']),tensor(spec['cap']))
            newgroups=[dict(params=[op],lr=.05),dict(params=[sig],lr=.005)]
            if name=='act':groups+=newgroups
            else:self.append_groups=newgroups
        self.optimizer=torch.optim.Adam(groups,eps=1e-15)
        if previous is not None:self.restore(previous,act_only=True)
        for group in getattr(self,'append_groups',[]):self.optimizer.add_param_group(group)
        self.domains={}
        for raw in (86,110):
            d=load(ROOT/'proposal/domains'/f'{raw:03d}.npz')
            self.domains[raw]={k:tensor(d[k],torch.bool) for k in ['fit','outside','retained']}
            if 'retained_confirmed_A' in d:self.domains[raw]['retained']|=tensor(d['retained_confirmed_A'],torch.bool)
        self.parent_alpha={r:tensor(load(ROOT/'parent/rendered'/f'{r:03d}.npz')['road_alpha']) for r in (86,110)}
    def values(self,worst=False):
        op=self.base_op;scale=self.base_scale
        for name,(pos,minimum,cap) in self.specs.items():
            value=torch.full_like(self.params[name+'_opacity'],.999) if worst else self.params[name+'_opacity'].sigmoid()
            sig=cap if worst else self.params[name+'_sigma'].exp()
            xyz=torch.cat([sig*float(self.ctx.road['meter_to_model']),self.base_scale[pos,2:]],1)
            op=op.index_copy(0,pos,value);scale=scale.index_copy(0,pos,xyz)
        return op,scale
    def step(self):
        self.optimizer.zero_grad(set_to_none=True);op,scale=self.values();loss=0.;parts={};t=time.monotonic()
        for raw in (86,110):
            alpha=geometry_render(self.ctx,raw,op,scale)['alpha'];d=self.domains[raw];base=self.parent_alpha[raw]
            mean=lambda x,m:x[m].mean() if m.any() else x.sum()*0
            cover=mean((alpha-.99).square(),d['fit']);outside=mean((alpha-base).square(),d['outside']);keep=mean((alpha-base).square(),d['retained'])
            loss=loss+(cover+outside+4*keep)/2;parts[str(raw)]=dict(coverage=float(cover.detach()),outside=float(outside.detach()),retained=float(keep.detach()))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite native coverage loss')
        loss.backward();grad={k:float(v.grad.abs().max()) if v.grad is not None else None for k,v in self.params.items()}
        if any(v is None or not np.isfinite(v) for v in grad.values()):raise FloatingPointError('new opacity/tangent gradients disconnected')
        axes={k:v.grad.abs().amax(0).detach().cpu().tolist() for k,v in self.params.items() if k.endswith('_sigma')}
        self.optimizer.step()
        with torch.no_grad():
            for name,(pos,lo,hi) in self.specs.items():
                self.params[name+'_opacity'].clamp_(float(torch.logit(torch.tensor(1e-4))),float(torch.logit(torch.tensor(.999))))
                self.params[name+'_sigma'].copy_(torch.minimum(torch.maximum(self.params[name+'_sigma'],lo.log()),hi.log()))
        self.trace.append(self.schedule[self.cursor]);self.cursor+=1;count('coverage_updates',1,self.role);count('coverage_nativeB',2,self.role)
        return dict(cursor=self.cursor,loss=float(loss.detach()),gradients=grad,tangent_axis_gradient_max=axes,parts=parts,seconds=time.monotonic()-t)
    def snapshot(self):return dict(schema='adaptive_coverage_full_state.v1',params=cpu_clone(self.params),optimizer=cpu_clone(self.optimizer.state_dict()),cursor=self.cursor,trace=self.trace.copy(),RNG=cpu_clone(rng_state('cuda')),role=self.role,binding=self.binding)
    def restore(self,s,act_only=False):
        assert s['schema']=='adaptive_coverage_full_state.v1' and s['role']==self.role
        if act_only:
            assert s['cursor']==200 and all(s['binding'][k]==self.binding[k] for k in ['activation','contract','domains'])
        else:assert s['binding']==self.binding
        with torch.no_grad():
            for k,v in s['params'].items():self.params[k].copy_(v.to('cuda'))
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));self.cursor=s['cursor'];self.trace=s['trace'].copy();restore_rng(s['RNG'],'cuda')
    def guard(self):
        op,sig=self.values(worst=True);ids=torch.cat([x[0] for x in self.specs.values()]);rows=[]
        with torch.no_grad():
            for raw in FRAMES:
                out=geometry_render(self.ctx,raw,op,sig,joint=True,feature_ids=ids);K=tensor(self.ctx.K_masks[raw],torch.bool)
                maximum=float(out['new_weight'][K].abs().max());rows.append(dict(raw=raw,K_new_weight_max=maximum))
        write(ROOT/('smoke' if self.role=='smoke' else 'qualification')/('K_'+str(len(self.ctx.base_rgb))+'.json'),dict(rows=rows,worst_opacity=.999,all_max_sigma=True))
        if any(r['K_new_weight_max']!=0 for r in rows):raise RuntimeError('candidate maximum footprint intersects actual joint K; no screen masking allowed')
        return rows

def freeze_asset(ctx,out,cov=None,colors=None,stage='coverage'):
    out=Path(out)
    if (out/'persistent/proof.json').exists():return out
    r=load(ctx.asset_dir/'road.npz');p=load(ctx.asset_dir/'provenance.npz');ids=ctx.active_world_rows
    if cov is not None:
        with torch.no_grad():op,sig=cov.values()
        changed=torch.cat([x[0] for x in cov.specs.values()]);worldids=ids[changed.cpu().numpy()]
        r['road_opacity'][worldids]=op[changed].detach().cpu().numpy()
        r['gaussian_sigma_m'][worldids]=sig[changed].detach().cpu().numpy().astype(np.float64)/float(r['meter_to_model'])
    if colors is not None:r['reference_srgb'][ids]=colors.detach().cpu().numpy()
    p['effective_active']=r['active'].copy();p['effective_row_rgb']=r['reference_srgb'].copy();p['effective_sigma_m']=r['gaussian_sigma_m'].copy();p['effective_road_opacity']=r['road_opacity'].copy()
    original=load(PARENT/'road.npz');old_active=original['active'];n0=len(old_active)
    for key in ['model_xyz','normal_world','gaussian_sigma_m','road_opacity','active','minimum_model_depth']:
        assert np.array_equal(r[key][:n0][old_active],original[key][old_active]),'parent active geometry changed: '+key
    out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'road.npz',**r);np.savez_compressed(out/'provenance.npz',**p)
    pin(ctx.asset_dir,out,dict(stage=stage,xyz_normal_rotation_fixed=True,existing_parent_active_geometry_frozen=True,appearance_only_color=colors is not None))
    write(out/'authority_receipt.json',dict(complete=True,road_SHA256=sha(out/'road.npz'),provenance_SHA256=sha(out/'provenance.npz'),
        source_asset=str(ctx.asset_dir),source_road_SHA256=sha(ctx.asset_dir/'road.npz'),teacher_manifest_SHA256=sha(OLD/'teacher/manifest.json'),
        surface_certificate_SHA256=sha(ROOT/'proposal/surface_certificate.npz')))
    return out

def export(ctx,stage,folder,values=None,role='formal'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rows=[]
    op,sig=values if values is not None else (None,None)
    r=ctx.road;p=ctx.provenance
    counts=dict(total_rows=len(r['active']),active_rows=int(r['active'].sum()),
        activate_existing=int(p.get('adaptive_activate_existing',np.zeros(len(r['active']),bool)).sum()),
        append_new_geometry=int(p.get('adaptive_append_new_geometry',np.zeros(len(r['active']),bool)).sum()))
    counts.update(total=counts['total_rows'],active=counts['active_rows'],activated_existing=counts['activate_existing'],appended_new_geometry=counts['append_new_geometry'])
    names=dict(parent='父道路资产',activated='启用既有锚点 + 覆盖200步',dense_init='真实追加表面高斯（覆盖前）',coverage='覆盖训练完成（累计600步）',appearance='外观训练完成（1200步）')
    with torch.no_grad():
        for raw in FRAMES:
            rd=geometry_render(ctx,raw,op,sig);jn=geometry_render(ctx,raw,op,sig,joint=True)
            arrays=dict(road_rgb=rd['rgb'].cpu().numpy(),road_alpha=rd['alpha'].cpu().numpy(),joint_rgb=jn['rgb'].cpu().numpy(),joint_alpha=jn['alpha'].cpu().numpy(),K=ctx.K_masks[raw])
            if stage!='parent':
                parent=load(ROOT/'parent/rendered'/f'{raw:03d}.npz');K=arrays['K']
                error=float(np.abs(arrays['joint_rgb']-parent['joint_rgb'])[:,K].max())
                if error!=0:raise RuntimeError('actual persistent joint K changed '+str((stage,raw,error)))
            else:error=0.
            if stage=='appearance':
                fixed=load(ROOT/'coverage_fit/rendered'/f'{raw:03d}.npz')
                assert np.array_equal(arrays['road_alpha'],fixed['road_alpha']) and np.array_equal(arrays['joint_alpha'],fixed['joint_alpha']),'appearance changed frozen coverage alpha'
            path=folder/f'{raw:03d}.npz';np.savez_compressed(path,**arrays)
            images={}
            for view in ['road','joint']:
                img=folder/f'{raw:03d}_{view}.png';Image.fromarray(np.rint(np.clip(arrays[view+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(img);images[view+'_png']=str(img.relative_to(ROOT))
            rows.append(dict(stage=stage,arm=stage,group='surface_adaptive',name=names.get(stage,stage),raw=raw,packet=str(path.relative_to(ROOT)),**images,resolution=[1152,768],asset_dir=str(ctx.asset_dir),road_SHA256=sha(ctx.asset_dir/'road.npz'),
                provenance_SHA256=sha(ctx.asset_dir/'provenance.npz'),point_counts=counts,joint_K_RGB_max=error,renderer='actual_ordinary_GS1152',
                coverage_updates=200 if stage in ['activated','dense_init'] else 600 if stage in ['coverage','appearance'] else 0,appearance_updates=1200 if stage=='appearance' else 0))
    if role=='formal':
        EXPORTS[:]=[r for r in EXPORTS if r['stage']!=stage]+rows;write(ROOT/'export_manifest.json',dict(schema='adaptive_surface_exports.v1',rows=EXPORTS,complete=False))
    write(folder/'receipt.json',dict(complete=True,stage=stage,point_counts=counts,rows=rows));account();return rows

class Appearance:
    def __init__(self,ctx,role='formal'):
        self.ctx=ctx;self.role=role;self.cursor=0;self.trace=[];self.base=ctx.base_rgb.detach();n0=len(load(PARENT/'road.npz')['active'])
        self.binding=dict(asset_road=sha(ctx.asset_dir/'road.npz'),teacher=sha(OLD/'teacher/manifest.json'),contract=sha(ROOT/'contract.json'),
            domains={str(r):sha(ROOT/'proposal/domains'/f'{r:03d}.npz') for r in (86,110)})
        permission=load(OLD/'common_permission/teacher_evidence.npz');oldids=permission['active_world_rows'][permission['common_write_active']]
        newflags=ctx.provenance.get('adaptive_activate_existing',np.zeros(len(ctx.road['active']),bool))|ctx.provenance.get('adaptive_append_new_geometry',np.zeros(len(ctx.road['active']),bool))
        newids=np.flatnonzero(ctx.road['active']&newflags);oldids=np.setdiff1d(oldids,newids)
        self.oldpos=active_positions(ctx,oldids);self.newpos=active_positions(ctx,newids)
        self.old_delta=torch.nn.Parameter(torch.zeros((len(self.oldpos),3),device='cuda'));self.new_rgb=torch.nn.Parameter(self.base[self.newpos].clone())
        self.optimizer=torch.optim.Adam([self.old_delta,self.new_rgb],lr=.01,betas=(.9,.999),eps=1e-8)
        self.frames={};self.alpha={}
        with torch.no_grad():
            for raw in (86,110):
                d=load(ROOT/'proposal/domains'/f'{raw:03d}.npz');teacher=load(OLD/'teacher'/f'raw{raw:03d}'/'supervision.npz')
                alpha=geometry_render(ctx,raw)['alpha'].cpu().numpy();self.alpha[raw]=alpha
                masks={k:choose(d,'appearance_'+k,k) for k in ['A','C','S']}
                self.frames[raw]=dict(alpha=np.ones_like(alpha),targets={k:alpha[None]*teacher['target_'+k+'_srgb'] for k in ['A','C','S']},contexts=masks,scores=masks)
        self.loss=NativeCapacityLoss(self.frames,device='cuda')
        with torch.no_grad():pred={raw:geometry_render(ctx,raw,colors=self.colors())['rgb'] for raw in (86,110)}
        self.loss.calibrate(pred)
    def colors(self):
        value=self.base.index_copy(0,self.oldpos,(self.base[self.oldpos]+self.old_delta).clamp(0,1))
        return value.index_copy(0,self.newpos,self.new_rgb.clamp(0,1))
    def step(self):
        t=time.monotonic();self.optimizer.zero_grad(set_to_none=True);pred={raw:geometry_render(self.ctx,raw,colors=self.colors())['rgb'] for raw in (86,110)}
        loss,_=self.loss.objective(pred,'detail')
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite native appearance objective')
        loss.backward();grads=[float(p.grad.abs().max()) if p.grad is not None else None for p in [self.old_delta,self.new_rgb]]
        if any(x is None or not np.isfinite(x) for x in grads):raise FloatingPointError('native appearance color gradient disconnected')
        self.optimizer.step()
        with torch.no_grad():self.old_delta.clamp_(-.05,.05);self.new_rgb.clamp_(0,1)
        self.cursor+=1;count('appearance_updates',1,self.role);count('appearance_nativeB',2,self.role)
        return dict(cursor=self.cursor,loss=float(loss.detach()),old_color_gradient_max=grads[0],new_color_gradient_max=grads[1],seconds=time.monotonic()-t)
    def snapshot(self):return dict(schema='adaptive_appearance_full_state.v1',old_delta=cpu_clone(self.old_delta),new_rgb=cpu_clone(self.new_rgb),optimizer=cpu_clone(self.optimizer.state_dict()),cursor=self.cursor,RNG=cpu_clone(rng_state('cuda')),normalizers=self.loss.snapshot(),role=self.role,binding=self.binding)
    def restore(self,s):
        assert s['schema']=='adaptive_appearance_full_state.v1' and s['role']==self.role
        assert s['binding']==self.binding
        with torch.no_grad():self.old_delta.copy_(s['old_delta'].to('cuda'));self.new_rgb.copy_(s['new_rgb'].to('cuda'))
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));self.cursor=s['cursor'];restore_rng(s['RNG'],'cuda');self.loss.restore(s['normalizers'])

def run_steps(state,end,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);ck=folder/'checkpoint.pt'
    if ck.exists():state.restore(torch.load(ck,map_location='cpu',weights_only=False))
    while state.cursor<end:
        r=state.step()
        if state.cursor%25==0:
            save(ck,state.snapshot());account();status('coverage' if isinstance(state,Coverage) else 'appearance',cursor=state.cursor,budget=end)
            print(json.dumps(r),flush=True)
    save(ck,state.snapshot());return state.snapshot()

def smoke(act,reserve):
    folder=ROOT/'smoke';folder.mkdir(exist_ok=True);start=time.monotonic()
    app={k:(v[:min(len(v),16)] if isinstance(v,np.ndarray) else v) for k,v in reserve.items()}
    asset=build_asset(PARENT,folder/'init_asset',activation=act,append=app);ctx=context(asset)
    assert len(ctx.road['active'])==len(load(PARENT/'road.npz')['active'])+len(app['world'])
    cov=Coverage(ctx,act,app,role='smoke');K=cov.guard();a=cov.step();s=cov.snapshot();save(folder/'coverage_checkpoint.pt',s)
    expected=cov.step();wanted=cov.snapshot();cov.restore(torch.load(folder/'coverage_checkpoint.pt',map_location='cpu',weights_only=False));cov.step()
    difference=max_tree_difference(wanted,cov.snapshot());assert difference<2e-5
    assert all(x>0 for x in a['gradients'].values()),'new opacity and BOTH tangent-axis gradients must be nonzero'
    assert all(x>0 for axes in a['tangent_axis_gradient_max'].values() for x in axes),'each native tangent axis needs real alpha gradient'
    saved=freeze_asset(ctx,folder/'coverage_asset',cov=cov,stage='smoke_coverage')
    with torch.no_grad():reference=geometry_render(ctx,110,*cov.values())
    reference={k:v.detach().cpu().numpy() for k,v in reference.items()};ctx=context(saved)
    with torch.no_grad():loaded=geometry_render(ctx,110)
    reload={k:float(np.abs(loaded[k].cpu().numpy()-reference[k]).max()) for k in ['rgb','alpha']}
    assert max(reload.values())<=5e-6
    color=Appearance(ctx,role='smoke');c=color.step();save(folder/'appearance_checkpoint.pt',color.snapshot());color.step();expected=color.snapshot()
    color.restore(torch.load(folder/'appearance_checkpoint.pt',map_location='cpu',weights_only=False));color.step()
    cdiff=max_tree_difference(expected,color.snapshot());assert cdiff<2e-5
    final=freeze_asset(ctx,folder/'appearance_asset',colors=color.colors(),stage='smoke_appearance')
    ctx=context(final);export(ctx,'smoke_final',folder/'rendered',role='smoke')
    estimate=600*a['seconds']+1200*max(c['seconds'],.05)+180
    # Only if the actual measured queue is hour-scale, extend real discarded work.
    long_required=estimate>=3600
    while long_required and time.monotonic()-start<600:
        color.step()
        if color.cursor%50==0:save(folder/'extended_checkpoint.pt',color.snapshot());account();status('long_smoke',elapsed=time.monotonic()-start)
    write(folder/'receipt.json',dict(verified=True,true_append_rows=len(app['world']),activation_rows=len(act['world']),
        true_native_alpha_gradients=a['gradients'],coverage_full_restore_next_max=difference,appearance_full_restore_next_max=cdiff,
        native_tangent_axis_gradient_max=a['tangent_axis_gradient_max'],
        persistent_reload=reload,K_worst_case=K,coverage_to_appearance=True,estimated_formal_seconds=estimate,
        ten_minute_required=long_required,actual_smoke_seconds=time.monotonic()-start,formal_warm_start=False))
    close_ctx();print('SMOKE_PASSED',flush=True)

def main():
    global EXPORTS,LEDGER,PRIOR_CALLS
    torch.set_num_threads(2);random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
    ROOT.mkdir(parents=True,exist_ok=True)
    write(ROOT/'gpu_lease.json',dict(active=True,owner=OWNER,gpu_authorized=True,power_policy='keep_on'))
    if (ROOT/'resource_ledger.json').exists():old=read(ROOT/'resource_ledger.json');LEDGER=old['counters'];PRIOR_CALLS=old['native_calls']
    if (ROOT/'export_manifest.json').exists():EXPORTS=read(ROOT/'export_manifest.json')['rows']
    try:
        if not (ROOT/'parent/rendered/receipt.json').exists():
            status('public_parent_reload');ctx=context(PARENT);export(ctx,'parent',ROOT/'parent/rendered');close_ctx()
        status('waiting_proposal')
        while not (ROOT/'proposal/manifest.json').exists() or not read(ROOT/'proposal/manifest.json').get('complete'):
            if (ROOT/'proposal/failed.json').exists():raise RuntimeError((ROOT/'proposal/failed.json').read_text())
            time.sleep(5)
        act=candidate_spec(ROOT/'proposal/activation.npz',activation=True);reserve=candidate_spec(ROOT/'proposal/append_reserve.npz')
        assert len(act['world'])>0 and len(reserve['world'])>0
        if not (ROOT/'smoke/receipt.json').exists():status('actual_smoke');smoke(act,reserve)
        assert read(ROOT/'smoke/receipt.json')['verified']
        init=build_asset(PARENT,ROOT/'activation_init/asset',activation=act);ctx=context(init);state=Coverage(ctx,act)
        state.guard();s=run_steps(state,200,ROOT/'activation_fit');asset=freeze_asset(ctx,ROOT/'activation_fit/asset',cov=state,stage='activation_coverage200')
        ctx=context(asset);export(ctx,'activated',ROOT/'activation_fit/rendered')
        status('adaptive_residual_selection')
        if not (ROOT/'proposal/append_manifest.json').exists():
            subprocess.run([sys.executable,str(ROOT/'proposal/select_append.py'),'--activation-render-root',str(ROOT/'activation_fit/rendered')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        assert read(ROOT/'proposal/append_manifest.json')['complete']
        app=candidate_spec(ROOT/'proposal/append.npz');assert len(app['world'])>0,'no actual new geometry selected'
        dense=build_asset(asset,ROOT/'dense_init/asset',append=app);ctx=context(dense);export(ctx,'dense_init',ROOT/'dense_init/rendered')
        state=Coverage(ctx,act,app,previous=s);state.guard();run_steps(state,600,ROOT/'coverage_fit')
        covered=freeze_asset(ctx,ROOT/'coverage_fit/asset',cov=state,stage='coverage600');ctx=context(covered);export(ctx,'coverage',ROOT/'coverage_fit/rendered')
        state=Appearance(ctx);run_steps(state,1200,ROOT/'appearance_fit');final=freeze_asset(ctx,ROOT/'appearance_fit/asset',colors=state.colors(),stage='appearance1200')
        ctx=context(final);export(ctx,'appearance',ROOT/'appearance_fit/rendered');ctx.assert_frozen();close_ctx()
        manifest=read(ROOT/'export_manifest.json');manifest['complete']=True;write(ROOT/'export_manifest.json',manifest)
        status('CPU_bench_pack');subprocess.run([sys.executable,str(ROOT/'bench/evaluate_and_package.py'),'--manifest',str(ROOT/'export_manifest.json')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        bundle=ROOT/'final_bundle.tar.gz';assert bundle.exists()
        receipt=dict(complete=True,activate_existing=len(act['world']),append_new_geometry=len(app['world']),
            coverage_updates=600,appearance_updates=1200,final_asset=str(final),road_SHA256=sha(final/'road.npz'),
            bundle_sha256=sha(bundle),GPU_keep_on=True,production_promoted=False,human_verdict=None)
        write(ROOT/'receipt.json',receipt);write(ROOT/'status.json',dict(state='complete',status='complete',phase='done',PID=os.getpid(),**receipt))
    except Exception:
        error=traceback.format_exc();account();write(ROOT/'failure.json',dict(error=error,counters=LEDGER,time=time.time()))
        write(ROOT/'status.json',dict(state='failed',status='failed',phase='failed',complete=False,error=error,PID=os.getpid()));raise
    finally:close_ctx()

if __name__=='__main__':main()
