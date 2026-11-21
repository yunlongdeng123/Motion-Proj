"""R2 fixed local residual refinement. Actual native GS gradients, no new network."""
from pathlib import Path
import copy,importlib.util,json,os,random,subprocess,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_surface_residual_20261121_r2';R1=C/'road_surface_adaptive_20261121_r1'
PARENT=R1/'appearance_fit/asset';N0=595029;A0=161264
spec=importlib.util.spec_from_file_location('r1_frozen_helpers',R1/'code/run_adaptive.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=ROOT;m.PARENT=PARENT;m.OWNER='road_surface_residual_r2'
torch=m.torch;read=m.read;load=m.load;write=m.write;sha=m.sha;save=m.save;tensor=m.tensor
FRAMES=m.FRAMES;render=m.geometry_render;positions=m.active_positions
INPUT=ROOT/'train_inputs'

def inputs():
    manifest=read(INPUT/'manifest.json');assert manifest['complete']
    for p,h in manifest['source_SHA256'].items():assert sha(p)==h,p
    act=m.candidate_spec(INPUT/'activation.npz',activation=True)
    app=m.candidate_spec(INPUT/'append.npz')
    reuse=load(INPUT/'reuse_existing.npz');reuse['source']=str(INPUT/'reuse_existing.npz');reuse['source_SHA256']=sha(reuse['source'])
    assert len(act['ids'])==13 and len(app['world'])==1071 and len(reuse['ids'])==107
    return act,app,reuse

def all_existing(act,reuse):
    # Only these two old-row classes may change coverage; no duplicate stack rows.
    return dict(ids=np.r_[act['ids'],reuse['ids']],world=np.r_[act['world'],reuse['world']],
        sigma=np.r_[act['sigma'],reuse['sigma']],minimum=np.r_[act['minimum'],reuse['minimum']],
        cap=np.r_[act['cap'],reuse['cap']],source_SHA256=sha(INPUT/'manifest.json'))

def row_extension(p,n,a):
    out={};M=len(a['world'])
    for k,v in p.items():
        if v.ndim and len(v)==n:
            fill=False if v.dtype.kind=='b' else -1 if v.dtype.kind=='i' else 0
            out[k]=np.concatenate([v,np.full((M,*v.shape[1:]),fill,dtype=v.dtype)])
        else:out[k]=v.copy()
    def put(k,value,dtype=None,fill=0):
        if k not in out:out[k]=np.full((n+M,*value.shape[1:]),fill,dtype=dtype or value.dtype)
        out[k][n:]=value
    out['world_xyz'][n:]=a['world'];out['world_keys'][n:]=np.c_[np.full(M,-2),-1-np.arange(M)]
    out['atlas_uv'][n:]=a['uv'];out['row_rgb'][n:]=a['rgb']
    for k in ['source','confidence','known']:
        if k in out:out[k][n:]=0
    for k in ['unknown','surface_supported','legal_write','surface_interpolated']:
        if k in out:out[k][n:]=True
    for k in ['observed_reference_srgb','source_distance_m','source_uv_native']:
        if k in out:out[k][n:]=np.nan
    for k in ['source_world_row','source_raw','parent_world_row_id']:
        if k in out:out[k][n:]=-1
    put('adaptive_append_new_geometry',np.ones(M,bool));put('adaptive_effective_surface_supported',np.ones(M,bool))
    put('adaptive_color_donor_world_row',a['donor'],np.int64,-1);put('adaptive_color_donor_distance_m',a['distance'])
    put('adaptive_triangle_id',a['triangle'],np.int64,-1);put('adaptive_barycentric',a['barycentric'])
    put('adaptive_triangle_world_rows',a['triangle_world_rows'],np.int64,-1)
    put('adaptive_triangle_vertex_world',a['triangle_vertex_world']);put('adaptive_geometry_kind',np.full(M,2,np.uint8))
    return out

def allowed(ctx):
    d=load(INPUT/'allowed_color_world_rows.npz');ids=d['world_row_ids'];ids=ids[ids<len(ctx.road['active'])]
    assert ctx.road['active'][ids].all();return ids

def prefix_proof(r):
    parent=load(PARENT/'road.npz');permit=load(INPUT/'allowed_color_world_rows.npz')['world_row_ids']
    reuse=load(INPUT/'reuse_existing.npz')['ids'];act=load(INPUT/'activation.npz')['existing_world_rows']
    old=np.ones(N0,bool);old[np.r_[act,reuse]]=False
    rows={}
    for key in ['model_xyz','normal_world','minimum_model_depth']:
        rows[key]=bool(np.array_equal(r[key][:N0],parent[key]));assert rows[key],key
    rows['normal_axis_sigma']=bool(np.array_equal(r['gaussian_sigma_m'][:N0,2],parent['gaussian_sigma_m'][:,2]));assert rows['normal_axis_sigma']
    for key in ['active','gaussian_sigma_m','road_opacity']:
        rows['frozen_'+key]=bool(np.array_equal(r[key][:N0][old],parent[key][old]));assert rows['frozen_'+key],key
    keep=np.ones(N0,bool);keep[permit[permit<N0]]=False
    rows['frozen_RGB']=bool(np.array_equal(r['reference_srgb'][:N0][keep],parent['reference_srgb'][keep]));assert rows['frozen_RGB']
    assert np.isfinite(r['reference_srgb']).all() and np.min(r['reference_srgb'])>=0 and np.max(r['reference_srgb'])<=1
    return rows

def sync_effective(p,r):
    for k,v in [('effective_active',r['active']),('effective_row_rgb',r['reference_srgb']),('effective_model_xyz',r['model_xyz']),('effective_sigma_m',r['gaussian_sigma_m']),('effective_road_opacity',r['road_opacity'])]:p[k]=v.copy()

def build(parent,out,act=None,app=None,reuse=None):
    out=Path(out)
    if (out/'persistent/proof.json').exists():return out
    if out.exists():raise FileExistsError('preserving incomplete asset '+str(out))
    r=load(Path(parent)/'road.npz');p=load(Path(parent)/'provenance.npz');n=len(r['active'])
    if act is not None:
        ids=act['ids'];assert not r['active'][ids].any();r['active'][ids]=True;r['reference_srgb'][ids]=act['rgb']
    M=0 if app is None else len(app['world'])
    if M:
        a=app;T=r['model_from_world'];new=dict(active=np.ones(M,bool),model_xyz=a['world']@T[:3,:3].T+T[:3,3],
            normal_world=a['normal'],gaussian_sigma_m=a['sigma'],reference_srgb=a['rgb'].astype(np.float32),road_opacity=np.full(M,.1,np.float32),minimum_model_depth=np.zeros(M,np.float32))
        for k,v in list(r.items()):
            if v.ndim and len(v)==n:r[k]=np.concatenate([v,np.asarray(new[k],dtype=v.dtype)])
        p=row_extension(p,n,a)
    N=len(r['active'])
    for key in ['r2_activate_existing','r2_append_new_geometry','r2_reuse_existing_newgroup']:
        if key not in p:p[key]=np.zeros(N,bool)
    if act is not None:p['r2_activate_existing'][act['ids']]=True
    if M:p['r2_append_new_geometry'][n:]=True
    if reuse is not None:p['r2_reuse_existing_newgroup'][reuse['ids']]=True
    sync_effective(p,r);proof=prefix_proof(r);out.mkdir(parents=True)
    np.savez_compressed(out/'road.npz',**r);np.savez_compressed(out/'provenance.npz',**p)
    m.pin(parent,out,dict(stage='R2_initialization',true_appended_rows=M,old_source_prefix_preserved=True))
    write(out/'extension_receipt.json',dict(complete=True,parent=str(parent),parent_road_SHA256=sha(Path(parent)/'road.npz'),
        road_SHA256=sha(out/'road.npz'),provenance_SHA256=sha(out/'provenance.npz'),prefix_proof=proof,
        inputs_manifest_SHA256=sha(INPUT/'manifest.json'),source_provenance_SHA256=sha(Path(parent)/'provenance.npz'),
        activation_source_SHA256=None if act is None else act['source_SHA256'],append_source_SHA256=None if app is None else app['source_SHA256'],
        point_counts=counts(r,p)))
    return out

def counts(r,p):
    N=len(r['active']);act=int(p.get('r2_activate_existing',np.zeros(N,bool)).sum());app=int(p.get('r2_append_new_geometry',np.zeros(N,bool)).sum());reuse=int(p.get('r2_reuse_existing_newgroup',np.zeros(N,bool)).sum())
    assert N-N0==app and int(r['active'].sum())-A0==act+app
    return dict(total=N,active=int(r['active'].sum()),activated_existing=act,appended_new_geometry=app,reuse_existing_newgroup=reuse,
        parent_total=N0,parent_active=A0,total_rows=N,active_rows=int(r['active'].sum()),activate_existing=act,append_new_geometry=app)

def kcheck(ctx,label,values=None,colors=None,role='formal',feature=True):
    ids=allowed(ctx);pos=positions(ctx,ids);op,sig=values if values is not None else (None,None);rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            out=render(ctx,raw,op,sig,colors=colors,joint=True,feature_ids=pos if feature else None)
            parent=load(R1/'appearance_fit/rendered'/f'{raw:03d}.npz');K=ctx.K_masks[raw]
            rgb=float(np.abs(out['rgb'].cpu().numpy()-parent['joint_rgb'])[:,K].max())
            alpha=float(np.abs(out['alpha'].cpu().numpy()-parent['joint_alpha'])[K].max())
            unit=float(out['new_weight'][K].abs().max()) if feature else 0.
            rows.append(dict(raw=raw,joint_K_RGB_max=rgb,joint_K_alpha_max=alpha,unit_K_max=unit))
    result=dict(passed=all(x[k]==0 for x in rows for k in ['joint_K_RGB_max','joint_K_alpha_max','unit_K_max']),rows=rows,
        exact_zero_required=True,nativeF=5,role=role,changed_rows=len(ids),feature_S24=feature)
    write(ROOT/('smoke' if role=='smoke' else 'qualification')/'K'/f'{label}.json',result)
    m.count('K_nativeF',5,role)
    if not result['passed']:raise RuntimeError('PROTECTION_GATE failed: '+label)
    return result

class Coverage(m.Coverage):
    def __init__(self,ctx,act,app,role='formal'):
        super().__init__(ctx,act,app,role=role);self.schedule=m.balanced_schedule(400,101)
    def step(self):
        start=time.monotonic();self.optimizer.zero_grad(set_to_none=True);op,sig=self.values();loss=0.;parts={}
        for raw in (86,110):
            alpha=render(self.ctx,raw,op,sig)['alpha'];mask=self.domains[raw]['fit']
            cover=torch.relu(.99-alpha[mask]).square().mean();loss=loss+cover/2;parts[str(raw)]=float(cover.detach())
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite coverage hinge')
        loss.backward();grad={k:float(v.grad.abs().max()) if v.grad is not None else None for k,v in self.params.items()}
        if any(v is None or not np.isfinite(v) for v in grad.values()):raise FloatingPointError('disconnected coverage gradients')
        axes={k:v.grad.abs().amax(0).detach().cpu().tolist() for k,v in self.params.items() if k.endswith('_sigma')}
        self.optimizer.step()
        with torch.no_grad():
            for name,(pos,lo,hi) in self.specs.items():
                self.params[name+'_opacity'].clamp_(float(torch.logit(torch.tensor(1e-4))),float(torch.logit(torch.tensor(.999))))
                self.params[name+'_sigma'].copy_(torch.minimum(torch.maximum(self.params[name+'_sigma'],lo.log()),hi.log()))
        self.trace.append(self.schedule[self.cursor%400]);self.cursor+=1;m.count('coverage_updates',1,self.role);m.count('coverage_nativeB',2,self.role)
        return dict(cursor=self.cursor,loss=float(loss.detach()),gradients=grad,tangent_axis_gradient_max=axes,parts=parts,seconds=time.monotonic()-start)
    def max_values(self):
        op,scale=self.values(worst=True)
        for pos,_,_ in self.specs.values():op=op.index_copy(0,pos,torch.ones(len(pos),device='cuda'))
        return op,scale

class Appearance:
    def __init__(self,ctx,role='formal'):
        self.ctx=ctx;self.role=role;self.cursor=0;self.base=ctx.base_rgb.detach();self.ids=allowed(ctx);self.pos=positions(ctx,self.ids)
        self.rgb=torch.nn.Parameter(self.base[self.pos].clone());self.optimizer=torch.optim.Adam([self.rgb],lr=.01,betas=(.9,.999),eps=1e-8)
        self.binding=dict(road=sha(ctx.asset_dir/'road.npz'),allowed=sha(INPUT/'allowed_color_world_rows.npz'),teacher=sha(m.OLD/'teacher/manifest.json'),contract=sha(ROOT/'contract.json'))
        self.frames={}
        with torch.no_grad():
            for raw in (86,110):
                d=load(ROOT/'proposal/domains'/f'{raw:03d}.npz');t=load(m.OLD/'teacher'/f'raw{raw:03d}'/'supervision.npz');alpha=render(ctx,raw)['alpha'].cpu().numpy()
                masks={k:d['appearance_'+k] for k in ['A','C','S']}
                self.frames[raw]=dict(alpha=np.ones_like(alpha),targets={k:alpha[None]*t['target_'+k+'_srgb'] for k in ['A','C','S']},contexts=masks,scores=masks)
        self.loss=m.NativeCapacityLoss(self.frames,device='cuda')
        with torch.no_grad():self.loss.calibrate({raw:render(ctx,raw,colors=self.colors())['rgb'] for raw in (86,110)})
    def colors(self):return self.base.index_copy(0,self.pos,self.rgb.clamp(0,1))
    def step(self):
        start=time.monotonic();self.optimizer.zero_grad(set_to_none=True)
        prediction={raw:render(self.ctx,raw,colors=self.colors())['rgb'] for raw in (86,110)};loss,_=self.loss.objective(prediction,'detail')
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite appearance objective')
        loss.backward();grad=None if self.rgb.grad is None else float(self.rgb.grad.abs().max())
        if grad is None or not np.isfinite(grad):raise FloatingPointError('disconnected permitted RGB gradient')
        self.optimizer.step()
        with torch.no_grad():self.rgb.clamp_(0,1)
        self.cursor+=1;m.count('appearance_updates',1,self.role);m.count('appearance_nativeB',2,self.role)
        return dict(cursor=self.cursor,loss=float(loss.detach()),allowed_color_gradient_max=grad,allowed_rows=len(self.ids),seconds=time.monotonic()-start)
    def snapshot(self):return dict(schema='r2_appearance_full_state.v1',rgb=m.cpu_clone(self.rgb),optimizer=m.cpu_clone(self.optimizer.state_dict()),cursor=self.cursor,RNG=m.cpu_clone(m.rng_state('cuda')),normalizers=self.loss.snapshot(),role=self.role,binding=self.binding)
    def restore(self,s):
        assert s['schema']=='r2_appearance_full_state.v1' and s['role']==self.role and s['binding']==self.binding
        with torch.no_grad():self.rgb.copy_(s['rgb'].to('cuda'))
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));self.cursor=s['cursor'];m.restore_rng(s['RNG'],'cuda');self.loss.restore(s['normalizers'])

def freeze(ctx,out,cov=None,color=None,stage='coverage'):
    out=Path(out)
    if (out/'persistent/proof.json').exists():return out
    r=load(ctx.asset_dir/'road.npz');p=load(ctx.asset_dir/'provenance.npz')
    if cov is not None:
        with torch.no_grad():op,sig=cov.values()
        for name,(pos,minimum,cap) in cov.specs.items():
            ids=ctx.active_world_rows[pos.cpu().numpy()];r['road_opacity'][ids]=op[pos].cpu().numpy()
            physical=sig[pos,:2].cpu().numpy().astype(np.float64)/float(r['meter_to_model'])
            r['gaussian_sigma_m'][ids,:2]=np.clip(physical,minimum.cpu().numpy(),cap.cpu().numpy())
    if color is not None:
        ids=allowed(ctx);pos=positions(ctx,ids);r['reference_srgb'][ids]=color[pos].detach().cpu().numpy()
    proof=prefix_proof(r);sync_effective(p,r);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'road.npz',**r);np.savez_compressed(out/'provenance.npz',**p)
    m.pin(ctx.asset_dir,out,dict(stage=stage,point_counts=counts(r,p),all_xyz_normal_height_frozen=True,parent_except_reuse_coverage_frozen=True))
    write(out/'authority_receipt.json',dict(complete=True,road_SHA256=sha(out/'road.npz'),provenance_SHA256=sha(out/'provenance.npz'),
        source_asset=str(ctx.asset_dir),source_road_SHA256=sha(ctx.asset_dir/'road.npz'),prefix_proof=proof,
        teacher_manifest_SHA256=sha(m.OLD/'teacher/manifest.json'),allowed_color_rows_SHA256=sha(INPUT/'allowed_color_world_rows.npz'),
        reuse_caps_authority_SHA256=sha(INPUT/'reuse_caps_authority.json'),source_inputs_manifest_SHA256=sha(INPUT/'manifest.json')))
    return out

def export(ctx,stage,folder,role='formal'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rows=[];point_counts=counts(ctx.road,ctx.provenance)
    names=dict(parent='第二轮直接父资产',activated='启用13个既有锚点（未训练）',dense_init='新增1071个表面位置（未训练）',coverage='覆盖优化400步',appearance='外观优化1200步')
    with torch.no_grad():
        for raw in FRAMES:
            rd=render(ctx,raw);joint=render(ctx,raw,joint=True);K=ctx.K_masks[raw]
            data=dict(road_rgb=rd['rgb'].cpu().numpy(),road_alpha=rd['alpha'].cpu().numpy(),joint_rgb=joint['rgb'].cpu().numpy(),joint_alpha=joint['alpha'].cpu().numpy(),K=K)
            parent=load(R1/'appearance_fit/rendered'/f'{raw:03d}.npz')
            krgb=float(np.abs(data['joint_rgb']-parent['joint_rgb'])[:,K].max());ka=float(np.abs(data['joint_alpha']-parent['joint_alpha'])[K].max())
            if krgb!=0 or ka!=0:raise RuntimeError('PROTECTION_GATE persistent export '+str((stage,raw,krgb,ka)))
            if stage=='appearance':
                fixed=load(ROOT/'coverage_fit/rendered'/f'{raw:03d}.npz')
                assert np.array_equal(data['road_alpha'],fixed['road_alpha']) and np.array_equal(data['joint_alpha'],fixed['joint_alpha'])
            path=folder/f'{raw:03d}.npz';np.savez_compressed(path,**data);images={}
            for view in ['road','joint']:
                image=folder/f'{raw:03d}_{view}.png';m.Image.fromarray(np.rint(np.clip(data[view+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(image);images[view+'_png']=str(image.relative_to(ROOT))
            rows.append(dict(stage=stage,arm=stage,group='surface_adaptive',name=names.get(stage,stage),raw=raw,packet=str(path.relative_to(ROOT)),**images,
                resolution=[1152,768],asset_dir=str(ctx.asset_dir),road_SHA256=sha(ctx.asset_dir/'road.npz'),provenance_SHA256=sha(ctx.asset_dir/'provenance.npz'),
                point_counts=point_counts,joint_K_RGB_max=krgb,joint_K_alpha_max=ka,renderer='actual_ordinary_GS1152',
                coverage_updates=400 if stage in ['coverage','appearance'] else 0,appearance_updates=1200 if stage=='appearance' else 0))
    if role=='formal':
        m.EXPORTS[:]=[x for x in m.EXPORTS if x['stage']!=stage]+rows;write(ROOT/'export_manifest.json',dict(schema='residual_surface_exports.v1',rows=m.EXPORTS,complete=False))
    write(folder/'receipt.json',dict(complete=True,stage=stage,point_counts=point_counts,rows=rows));m.account();return rows

def run_steps(state,end,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():state.restore(torch.load(checkpoint,map_location='cpu',weights_only=False))
    while state.cursor<end:
        result=state.step();interval=25 if isinstance(state,Coverage) else 100
        if state.cursor%interval==0:
            save(checkpoint,state.snapshot());m.account();label=('coverage' if isinstance(state,Coverage) else 'appearance')+f'_{state.cursor:04d}'
            if isinstance(state,Coverage):kcheck(state.ctx,label,values=state.values())
            else:kcheck(state.ctx,label,colors=state.colors())
            m.status(label,cursor=state.cursor,budget=end);print(json.dumps(result),flush=True)
    save(checkpoint,state.snapshot())

def smoke(act,app,reuse):
    folder=ROOT/'smoke';folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    small={k:v[:16] if isinstance(v,np.ndarray) else v for k,v in app.items()};existing=all_existing(act,reuse)
    asset=build(PARENT,folder/'init_asset',act,small,reuse);ctx=m.context(asset)
    assert len(ctx.road['active'])==N0+16 and counts(ctx.road,ctx.provenance)['reuse_existing_newgroup']==107
    cov=Coverage(ctx,existing,small,role='smoke');initial=kcheck(ctx,'initial',values=cov.values(),role='smoke');maximum=kcheck(ctx,'max',values=cov.max_values(),role='smoke')
    first=cov.step();saved=cov.snapshot();save(folder/'coverage_checkpoint.pt',saved);cov.step();expected=cov.snapshot()
    cov.restore(torch.load(folder/'coverage_checkpoint.pt',map_location='cpu',weights_only=False));cov.step();difference=m.max_tree_difference(expected,cov.snapshot());assert difference<2e-5
    assert all(v is not None and v>0 for v in first['gradients'].values())
    assert all(v>0 for axes in first['tangent_axis_gradient_max'].values() for v in axes)
    covered=freeze(ctx,folder/'coverage_asset',cov=cov,stage='smoke_coverage');reference={}
    with torch.no_grad():reference={k:v.cpu().numpy() for k,v in render(ctx,110,*cov.values()).items()}
    ctx=m.context(covered)
    with torch.no_grad():loaded=render(ctx,110)
    reload={k:float(np.abs(reference[k]-loaded[k].cpu().numpy()).max()) for k in ['rgb','alpha']};assert max(reload.values())<5e-6
    kcheck(ctx,'coverage_reload',role='smoke');color=Appearance(ctx,role='smoke');firstcolor=color.step();save(folder/'appearance_checkpoint.pt',color.snapshot())
    color.step();expected=color.snapshot();color.restore(torch.load(folder/'appearance_checkpoint.pt',map_location='cpu',weights_only=False));color.step();cdiff=m.max_tree_difference(expected,color.snapshot());assert cdiff<2e-5
    final=freeze(ctx,folder/'appearance_asset',color=color.colors(),stage='smoke_appearance');ctx=m.context(final);export(ctx,'smoke_final',folder/'rendered',role='smoke');kcheck(ctx,'final_S2',role='smoke',feature=False)
    estimate=400*first['seconds']+1200*max(firstcolor['seconds'],.05)+240;long_required=estimate>=3600
    if long_required:
        # Extra discarded real work, never a formal warm start.
        color=Appearance(ctx,role='smoke')
        while time.monotonic()-start<600:
            color.step()
            if color.cursor%50==0:save(folder/'extended_checkpoint.pt',color.snapshot());m.account();m.status('long_smoke',elapsed=time.monotonic()-start)
    write(folder/'receipt.json',dict(verified=True,appended_rows=16,activated_rows=13,reused_existing_rows=107,point_count_gate=True,
        coverage_full_restore_next_max=difference,appearance_full_restore_next_max=cdiff,persistent_reload=reload,prefix_freeze_proof=prefix_proof(ctx.road),
        allowed_color_count=len(allowed(ctx)),true_alpha_gradients=first['gradients'],tangent_axis_gradient_max=first['tangent_axis_gradient_max'],
        appearance_gradient=firstcolor['allowed_color_gradient_max'],initial_K=initial,max_K=maximum,coverage_to_appearance=True,
        formal_warm_start=False,estimated_formal_seconds=estimate,ten_minute_required=long_required,actual_smoke_seconds=time.monotonic()-start))
    m.close_ctx();print('SMOKE_PASSED',flush=True)

def main():
    torch.set_num_threads(2);random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
    write(ROOT/'gpu_lease.json',dict(active=True,owner=m.OWNER,gpu_authorized=True,power_policy='keep_on'))
    if (ROOT/'resource_ledger.json').exists():ledger=read(ROOT/'resource_ledger.json');m.LEDGER=ledger['counters'];m.PRIOR_CALLS=ledger['native_calls']
    if (ROOT/'export_manifest.json').exists():m.EXPORTS=read(ROOT/'export_manifest.json')['rows']
    try:
        act,app,reuse=inputs();existing=all_existing(act,reuse)
        if not (ROOT/'parent/rendered/receipt.json').exists():
            m.status('parent_public_reload');ctx=m.context(PARENT);export(ctx,'parent',ROOT/'parent/rendered');m.close_ctx()
        if not (ROOT/'smoke/receipt.json').exists():m.status('actual_smoke');smoke(act,app,reuse)
        assert read(ROOT/'smoke/receipt.json')['verified']
        activated=build(PARENT,ROOT/'activation_init/asset',act=act);ctx=m.context(activated);export(ctx,'activated',ROOT/'activation_init/rendered')
        dense=build(activated,ROOT/'dense_init/asset',app=app,reuse=reuse);ctx=m.context(dense);export(ctx,'dense_init',ROOT/'dense_init/rendered')
        cov=Coverage(ctx,existing,app);kcheck(ctx,'formal_initial',values=cov.values());kcheck(ctx,'formal_max',values=cov.max_values())
        m.status('coverage',cursor=0,budget=400);run_steps(cov,400,ROOT/'coverage_fit');covered=freeze(ctx,ROOT/'coverage_fit/asset',cov=cov,stage='coverage400')
        ctx=m.context(covered);export(ctx,'coverage',ROOT/'coverage_fit/rendered');kcheck(ctx,'coverage_public_S2',feature=False)
        color=Appearance(ctx);m.status('appearance',cursor=0,budget=1200);run_steps(color,1200,ROOT/'appearance_fit');final=freeze(ctx,ROOT/'appearance_fit/asset',color=color.colors(),stage='appearance1200')
        ctx=m.context(final);export(ctx,'appearance',ROOT/'appearance_fit/rendered');kcheck(ctx,'appearance_public_S2',feature=False);ctx.assert_frozen();m.close_ctx()
        manifest=read(ROOT/'export_manifest.json');manifest['complete']=True;write(ROOT/'export_manifest.json',manifest)
        m.status('CPU_bench_pack');subprocess.run([sys.executable,str(ROOT/'bench/evaluate_and_package.py'),'--manifest',str(ROOT/'export_manifest.json')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        bundle=ROOT/'final_bundle.tar.gz';assert bundle.exists();receipt=dict(complete=True,activated_existing=13,appended_new_geometry=1071,reuse_existing_newgroup=107,
            final_total_rows=596100,final_active_rows=162348,coverage_updates=400,appearance_updates=1200,final_asset=str(final),road_SHA256=sha(final/'road.npz'),
            provenance_SHA256=sha(final/'provenance.npz'),bundle_sha256=sha(bundle),GPU_keep_on=True,production_promoted=False)
        write(ROOT/'receipt.json',receipt);write(ROOT/'status.json',dict(state='complete',status='complete',phase='done',PID=os.getpid(),**receipt))
    except Exception:
        error=traceback.format_exc();m.account();write(ROOT/'failure.json',dict(error=error,counters=m.LEDGER,time=time.time()))
        write(ROOT/'status.json',dict(state='failed',status='failed',phase='failed',complete=False,error=error,PID=os.getpid()));raise
    finally:
        m.close_ctx();write(ROOT/'gpu_lease.json',dict(active=False,owner=m.OWNER,gpu_authorized=True,power_policy='keep_on'))

if __name__=='__main__':main()
