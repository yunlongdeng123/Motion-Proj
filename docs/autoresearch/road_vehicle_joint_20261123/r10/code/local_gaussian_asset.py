"""CPU-only certified new local Gaussian asset and target-free persistent loader.

No raw-image material generation, renderer, torch import or placeholder asset.
"""
from pathlib import Path
from dataclasses import dataclass
import argparse,hashlib,json,shutil
import numpy as np

ACTOR_SHA='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
C0=0.28209479177387814
FIELDS=('xyz','scales','rotation','coefficients','opacity')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def bank(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def resolve(base,value):
    p=Path(value);return p if p.is_absolute() else base/p
def pinned(base,spec):
    p=resolve(base,spec['path'])
    if not p.is_file():raise ValueError('missing input: '+str(p))
    if sha(p)!=spec['sha256']:raise ValueError('input SHA mismatch: '+str(p))
    return p

def tangent_sigma(spacing_m=.02,opacity=.9,cell_alpha=.95):
    """Four equal kernels at a square-cell midpoint; not a rendered guarantee."""
    per_kernel=1-(1-cell_alpha)**.25;weight=per_kernel/opacity
    if not 0<weight<1:raise ValueError('Initial opacity cannot support requested four-kernel coverage')
    return (spacing_m/np.sqrt(2))/np.sqrt(-2*np.log(weight))

def normal_quaternions(normals):
    n=np.asarray(normals,np.float64);n=n/np.linalg.norm(n,axis=1,keepdims=True)
    ref=np.zeros_like(n);ref[:,0]=1;ref[np.abs(n[:,0])>.9]=[0,1,0]
    a=ref-(ref*n).sum(1,keepdims=True)*n;a/=np.linalg.norm(a,axis=1,keepdims=True);b=np.cross(n,a)
    R=np.stack([a,b,n],axis=-1);q=np.empty((len(n),4),np.float64)
    for i,r in enumerate(R):
        candidates=np.array([1+np.trace(r),1+r[0,0]-r[1,1]-r[2,2],1-r[0,0]+r[1,1]-r[2,2],1-r[0,0]-r[1,1]+r[2,2]])
        k=int(np.argmax(candidates));v=np.sqrt(max(candidates[k],0))/2;d=4*v
        if k==0:q[i]=[v,(r[2,1]-r[1,2])/d,(r[0,2]-r[2,0])/d,(r[1,0]-r[0,1])/d]
        elif k==1:q[i]=[(r[2,1]-r[1,2])/d,v,(r[0,1]+r[1,0])/d,(r[0,2]+r[2,0])/d]
        elif k==2:q[i]=[(r[0,2]-r[2,0])/d,(r[0,1]+r[1,0])/d,v,(r[1,2]+r[2,1])/d]
        else:q[i]=[(r[1,0]-r[0,1])/d,(r[0,2]+r[2,0])/d,(r[1,2]+r[2,1])/d,v]
    q/=np.linalg.norm(q,axis=1,keepdims=True);q[q[:,0]<0]*=-1
    return q.astype(np.float32)

def sample_linear(field,uv,valid):
    h,w=field.shape[:2];uv=np.asarray(uv,np.float64);finite=np.isfinite(uv).all(1);safe=np.where(finite[:,None],uv,0)
    x=np.floor(safe[:,0]).astype(int);y=np.floor(safe[:,1]).astype(int)
    legal=finite&(x>=0)&(y>=0)&(x+1<w)&(y+1<h);x=np.clip(x,0,w-2);y=np.clip(y,0,h-2)
    fx=safe[:,0]-x;fy=safe[:,1]-y;out=np.zeros((len(uv),3),np.float64)
    for ix,iy,weight in [(x,y,(1-fx)*(1-fy)),(x+1,y,fx*(1-fy)),(x,y+1,(1-fx)*fy),(x+1,y+1,fx*fy)]:
        legal&=valid[iy,ix]&np.isfinite(field[iy,ix]).all(1);out+=field[iy,ix]*weight[:,None]
    return out.astype(np.float32),legal

def view_matrix(value,n,frames):
    value=np.asarray(value)
    if value.shape==(n,len(frames)):return value.T
    if value.shape==(len(frames),n):return value
    raise ValueError('Candidate mass matrix does not match global IDs and frames')

def validate_native_gate(base_dir,cfg,parent):
    """Identity is explicit; nonempty retirement keeps actual VJP qualification."""
    spec=cfg['native_gate'];mode=spec.get('mode','qualified_sparse_retirement')
    path=pinned(base_dir,spec['npz']);proof_path=pinned(base_dir,spec['proof'])
    candidate_path=pinned(base_dir,spec['candidates'])
    gate=bank(path);proof=read(proof_path);candidate=bank(candidate_path)
    gids=np.asarray(gate['global_row_ids']);values=np.asarray(gate['gate'])
    frames=np.asarray(candidate['frames']).tolist()
    assert gids.dtype==np.int64 and gids.ndim==1 and np.array_equal(gids,np.unique(gids)) and np.array_equal(gids,candidate['global_row_ids'])
    assert frames==cfg['frames'] and values.dtype==np.float32 and values.shape==gids.shape
    assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    if not(proof.get('complete') and proof.get('decision')=='PASS'):raise ValueError('Native gate proof is incomplete')
    assert proof['baseline_contract_sha256']==cfg['parent_contract']['sha256']
    assert proof['candidate_npz_sha256']==spec['candidates']['sha256'] and proof['gate_npz_sha256']==spec['npz']['sha256']
    if mode=='identity_no_retirement':
        assert proof['policy']=='identity_native_noop' and proof['action']=='preserve_all_native_rows'
        assert len(gids)==0 and len(values)==0 and proof['candidate_rows']==0
        assert proof['K_body_condition_is_vacuous_for_empty_selection'] is True
        assert proof['all_old_native_parameters_unmodified'] is True and proof['native21_and_other_retained_actors_frozen'] is True
        assert proof['old_RoadRows_retirement_not_authorized_by_this_scope'] is True
        assert proof['diagnostic_5444_and_8816_gates_not_adopted'] is True
        assert proof['new_field_actual_K_and_body_measurement_required'] is True
        assert proof['accepted_actor_sha256']==ACTOR_SHA and spec.get('deleted_target_owner_ids',[])==[]
        if cfg.get('parent_pins'):
            parents={p['role']:p['sha256'] for p in cfg['parent_pins']}
            assert proof['old_context_plan_sha256']==parents['context_plan']
            assert proof['native_scene_proof_sha256']==parents['scene_proof']
        for key in ['allview_K_mass','allview_body_mass','allview_ground_mass','owners_by_view']:
            assert candidate[key].shape==(0,len(frames))
        return path,proof_path,dict(mode=mode,rows=0,identity_no_retirement=True)
    if mode!='qualified_sparse_retirement':raise ValueError('Unsupported native gate mode')
    measurement_path=pinned(base_dir,spec['measurement_receipt']);measurement=read(measurement_path)
    assert measurement.get('complete') is True and proof['actual_attribution_receipt_sha256']==sha(measurement_path)
    assert measurement.get('nativeF',measurement.get('actual_nativeF',0))>=len(frames) and measurement.get('nativeB',measurement.get('actual_nativeB',0))>=len(frames)
    if proof.get('policy')=='identity_native_noop' or proof.get('all_five_K_body_exact_zero') is not True:raise ValueError('Nonempty native retirement requires actual zero K/body in all views')
    assert ((gids>=0)&(gids<int(proof['native_global_row_count']))).all()
    key=lambda a,b:a if a in candidate else b
    k=view_matrix(candidate['allview_K_mass'],len(gids),frames)
    body=view_matrix(candidate[key('allview_body_mass','body_mass')],len(gids),frames)
    ground=view_matrix(candidate[key('allview_ground_mass','ground_mass')],len(gids),frames)
    assert all(np.isfinite(v).all() and (v>=0).all() for v in [k,body,ground])
    assert (k==0).all() and (body==0).all() and ((ground.sum(0)>0).all() or not len(gids))
    owners=np.asarray(candidate['owners_by_view']);assert owners.shape==(len(gids),len(frames))
    deleted=spec['deleted_target_owner_ids'];assert deleted==[10]
    assert ((owners<=0)|np.isin(owners,deleted)).all() and not (owners==21).any()
    plan=read(parent['context_plan']);assert not np.intersect1d(gids,plan.get('preserved_native_context_global_row_ids',[])).size
    return path,proof_path,dict(mode=mode,rows=len(gids),identity_no_retirement=False)


def preflight(contract):
    """A missing or unaccepted input returns not-ready; no directory is created."""
    path=Path(contract);status=dict(schema='R10.local_gaussian_asset_preflight.v1',ready=False,CPU_only=True,GPU_calls=0,optimizer_updates=0,asset_written=False,issues=[])
    try:
        cfg=read(path);base_dir=path.resolve().parent
        geometry_path=pinned(base_dir,cfg['geometry']['npz']);certificate_path=pinned(base_dir,cfg['geometry']['certificate']);certificate=read(certificate_path)
        if not(certificate.get('complete') and certificate.get('decision')=='PASS' and certificate.get('constant_z_plane_used') is False):raise ValueError('Geometry certificate is not explicit qualified nonconstant-height evidence')
        if certificate['geometry_npz']['sha256']!=cfg['geometry']['npz']['sha256']:raise ValueError('Geometry certificate binds another NPZ')
        coverage_path=pinned(base_dir,cfg['geometry']['target_coverage_gate']);coverage=read(coverage_path)
        if not(coverage.get('complete') and coverage.get('decision')=='PASS' and coverage.get('local_asset_initialization_authorized') is True):raise ValueError('Qualified height subset has no accepted target coverage for local reconstruction')
        if coverage['geometry_npz_sha256']!=cfg['geometry']['npz']['sha256']:raise ValueError('Target coverage refers to another surface')
        geometry=bank(geometry_path);vertices=np.asarray(geometry['world_xyz'],np.float64);h,w=vertices.shape[:2];cell_shape=(h-1,w-1)
        asset_mask=np.asarray(geometry['asset_mask'],bool);reconstruction=np.asarray(geometry['reconstruction_mask'],bool);blend=np.asarray(geometry['blend_ring_mask'],bool)
        assert asset_mask.shape==reconstruction.shape==blend.shape==cell_shape and not (asset_mask&~(reconstruction|blend)).any()
        ids=np.flatnonzero(asset_mask.reshape(-1))
        if not len(ids):raise ValueError('No authorized reconstruction footprint; context-only grid is not a new asset')
        average4=lambda a:(a[:-1,:-1]+a[:-1,1:]+a[1:,:-1]+a[1:,1:])*.25
        all4=lambda a:a[:-1,:-1]&a[:-1,1:]&a[1:,:-1]&a[1:,1:]
        uv_key='uv_global' if 'uv_global' in geometry else 'shared_atlas_uv';uv_vertices=np.asarray(geometry[uv_key],np.float64)
        if 'cell_world_xyz' in geometry:
            xyz_grid=np.asarray(geometry['cell_world_xyz'],np.float64);normal_grid=np.asarray(geometry['cell_normal_world'],np.float64)
            if certificate.get('candidate_geometry_accepted_for_zero_step') is True:
                # An immutable candidate may keep renderable_mask=false. A
                # separately signed acceptance binds its exact SHA and cell
                # footprint; it does not rewrite direct-witness evidence.
                if certificate.get('approval_scope')!='new_local_zero_step_only':raise ValueError('Candidate override is not explicitly scoped to the zero-step comparison')
                key=certificate['approved_cell_mask_key']
                approved=np.asarray(geometry[key],bool)
                assert approved.shape==cell_shape and np.array_equal(approved,asset_mask)
                cell_valid=np.asarray(geometry['cell_candidate_mask'],bool)&approved
            else:
                cell_valid=np.asarray(geometry['cell_geometry_valid'],bool)&np.asarray(geometry['cell_renderable_mask'],bool)
            source_kind=np.asarray(geometry['cell_geometry_source_kind']);centre_method='separately certified new cell-centre samples'
        else:
            if certificate.get('cell_center_interpolation_authorized') is not True:raise ValueError('Vertex-only geometry does not authorize new cell-centre interpolation')
            vertex_valid=np.asarray(geometry['geometry_valid'] if 'geometry_valid' in geometry else geometry['valid_mask'],bool);renderable=np.asarray(geometry['renderable_mask'],bool)
            cell_valid=all4(vertex_valid)&all4(renderable);xyz_grid=average4(vertices)
            normal_grid=average4(np.asarray(geometry['normal_world'] if 'normal_world' in geometry else geometry['normals'],np.float64))
            kinds=np.asarray(geometry['geometry_source_kind']);source_kind=np.maximum.reduce([kinds[:-1,:-1],kinds[:-1,1:],kinds[1:,:-1],kinds[1:,1:]])
            centre_method='new cell centres from explicitly authorized bilinear surface interpolation'
        assert xyz_grid.shape==normal_grid.shape==(*cell_shape,3) and cell_valid.shape==source_kind.shape==cell_shape
        if (asset_mask&~cell_valid).any():raise ValueError('Requested new-asset cells lack separately qualified geometry/renderability')
        cell_kinds=source_kind.reshape(-1)[ids];assert np.isin(cell_kinds,[1,2]).all()
        if (cell_kinds==2).any():
            inferred_path=pinned(base_dir,cfg['geometry']['inferred_gate']);inferred=read(inferred_path)
            if not(inferred.get('complete') and inferred.get('decision')=='PASS' and inferred.get('geometry_npz_sha256')==cfg['geometry']['npz']['sha256']):raise ValueError('Model-inferred geometry lacks separate acceptance')
        xyz=xyz_grid.reshape(-1,3)[ids];normal=normal_grid.reshape(-1,3)[ids];normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        uv=average4(uv_vertices).reshape(-1,2)[ids]
        if not np.allclose(uv-np.floor(uv),.5,atol=1e-5):raise ValueError('New centres must use the half-cell global UV lattice, not existing grid vertices')
        pose=np.asarray(geometry['world_from_patch'],np.float64);R=pose[:3,:3]
        assert pose.shape==(4,4) and np.allclose(pose[3],[0,0,0,1],atol=1e-12) and np.allclose(R.T@R,np.eye(3),atol=1e-10) and np.linalg.det(R)>0
        if not np.isfinite(xyz).all() or not np.isfinite(normal).all() or not np.allclose(np.linalg.norm(normal,axis=1),1,atol=1e-3):raise ValueError('Certified centres/normals invalid')
        local=((xyz-pose[:3,3])@R).astype(np.float32)
        roundtrip=float(abs(local.astype(np.float64)@R.T+pose[:3,3]-xyz).max())
        if roundtrip>1e-5:raise ValueError('Local origin transform does not preserve measured XYZ')
        source=cfg['source'];field_path=pinned(base_dir,source['field']);field=bank(field_path)
        for role in ['numeric','visual','root']:
            gate_path=pinned(base_dir,source[role+'_gate']);gate=read(gate_path)
            if not(gate.get('complete') and gate.get('decision')=='PASS' and gate.get('final_atlas_sha256',gate.get('field_sha256'))==source['field']['sha256']):raise ValueError('New local appearance '+role+' gate is not PASS on this field')
            if role=='root' and not(gate.get('local_patch_appearance_accepted') is True and gate.get('student_fit_authorized') is True):raise ValueError('Root did not authorize this new local appearance')
        rgb_key=source.get('rgb_key','atlas_linear');valid_key=source.get('valid_key','permission_mask');rgb=np.asarray(field[rgb_key],np.float32);appearance_valid=np.asarray(field[valid_key],bool)
        if 'canvas_uv_origin_global' in field:
            canvas_receipt_path=pinned(base_dir,source['canvas_receipt']);embedding=read(canvas_receipt_path)['same_global_UV_canvas_embedding']
            origin=np.asarray(field['canvas_uv_origin_global'],np.int64)
            assert np.array_equal(origin,np.asarray(source['uv_origin_global'],np.int64)) and np.array_equal(origin,np.asarray(embedding['origin_global'],np.int64))
            assert np.array_equal(field['canvas_shape'],np.asarray(rgb.shape[:2])) and np.array_equal(field['canvas_shape'],embedding['canvas_shape'])
            assert np.array_equal(field['canvas_parent_shape'],embedding['parent_shape'])
        if 'same_world_requalification' in source:
            stats=source['same_world_requalification'];assert int(field['real_locks'].sum())==stats['branch_real_lock_count']
            assert stats['ancestor_real_lock_count']-stats['unaffected_inherited_real_lock_count']==stats['qualification_repair_addresses']
            assert stats['unaffected_inherited_real_lock_count']+stats['fresh_recertified_real_count']==stats['branch_real_lock_count']
        if source.get('kind_key','source_kind') in field:appearance_valid&=field[source.get('kind_key','source_kind')]>0
        query=uv-np.asarray(source.get('uv_origin_global',[0,0]),np.float64);colour,legal=sample_linear(rgb,query,appearance_valid)
        if not legal.all():raise ValueError(f'New accepted appearance lacks all-four support for {int((~legal).sum())} certified geometry points')
        if not np.isfinite(colour).all() or (colour<0).any() or (colour>1).any():raise ValueError('Shared linear RGB is outside the declared domain')
        parent_path=pinned(base_dir,cfg['parent_contract']);parent=read(parent_path);assert parent['frames']==cfg['frames']
        if len(parent['actors'])!=1 or parent['actors'][0]['sha256']!=ACTOR_SHA:raise ValueError('Original accepted A20 r3 must remain the parent')
        actor_path=Path(parent['actors'][0]['asset']);assert actor_path.is_file() and sha(actor_path)==ACTOR_SHA
        parent_pins=[]
        required_roles={'actor_asset','road_asset','scene_manifest','scene_proof','context_plan','actor_field_module'}
        if not required_roles.issubset({pin['role'] for pin in cfg['parent_pins']}):raise ValueError('Frozen parent pin set is incomplete')
        for pin in cfg['parent_pins']:pinned(base_dir,pin);parent_pins.append(pin)
        gate_path,gate_proof_path,gate_status=validate_native_gate(base_dir,cfg,parent)
        init=cfg.get('initialization',{});spacing=float(certificate.get('spacing_m',.02));assert abs(spacing-.02)<1e-9
        opacity=float(init.get('opacity',.9));normal_sigma=float(init.get('normal_sigma_m',.0015));assert .001<=normal_sigma<=.002 and 0<opacity<=1 and float(cfg.get('new_patch_near_m',.2))>0
        sigma=tangent_sigma(spacing,opacity,float(init.get('cell_alpha_target',.95)))
        status.update(ready=True,geometry_rows=len(ids),context_grid_vertices=h*w,new_asset_cells=len(ids),direct_supported_surface_cells=int((cell_kinds==1).sum()),model_inferred_surface_cells=int((cell_kinds==2).sum()),centre_method=centre_method,native_gate_rows=gate_status['rows'],native_gate_mode=gate_status['mode'],native_identity_no_retirement=gate_status['identity_no_retirement'],local_XYZ_roundtrip_max_m=roundtrip,tangent_sigma_m=float(sigma),normal_sigma_m=normal_sigma)
        return status,dict(cfg=cfg,base_dir=base_dir,geometry_path=geometry_path,certificate_path=certificate_path,field_path=field_path,field=field,source_query_uv=query,local=local,normal_local=normal@R,uv=uv,grid_ids=ids,geometry_source_kind=cell_kinds,pose=pose,linear=colour,opacity=opacity,sigma=sigma,normal_sigma=normal_sigma,parent_pins=parent_pins,gate_path=gate_path,gate_proof_path=gate_proof_path)
    except (OSError,ValueError,KeyError,TypeError,AssertionError) as error:
        status['issues'].append(str(error) or type(error).__name__);return status,None

def persist(destination,field,pose,lineage,gate_path,gate_proof_path,manifest):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve previous local asset: '+str(destination))
    destination.mkdir(parents=True);np.savez_compressed(destination/'local_field.npz',**field);np.savez_compressed(destination/'lineage.npz',**lineage)
    write(destination/'pose.json',dict(world_from_patch=np.asarray(pose,np.float64).tolist(),units='metres',rigid=True));shutil.copy2(gate_path,destination/'native_gate.npz');shutil.copy2(gate_proof_path,destination/'native_gate_proof.json')
    manifest=dict(manifest);manifest['files']={name:dict(path=name,sha256=sha(destination/name)) for name in ['local_field.npz','lineage.npz','pose.json','native_gate.npz','native_gate_proof.json']}
    write(destination/'manifest.json',manifest);return manifest

def build_local_asset(contract,destination):
    status,data=preflight(contract)
    if not status['ready']:return status
    cfg=data['cfg'];n=len(data['local']);linear=data['linear'];srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055).astype(np.float32)
    coefficients=np.zeros((n,16,3),np.float32);coefficients[:,0]=(srgb-.5)/C0
    field=dict(xyz=data['local'],scales=np.tile([data['sigma'],data['sigma'],data['normal_sigma']],(n,1)).astype(np.float32),rotation=normal_quaternions(data['normal_local']),coefficients=coefficients,opacity=np.full(n,data['opacity'],np.float32),active=np.ones(n,bool),source_id=np.full(n,2,np.int64),teacher_prior_weight=np.zeros(n,np.float32),normal_local=data['normal_local'].astype(np.float32))
    lineage=dict(asset_local_row_ids=np.arange(n,dtype=np.int64),geometry_cell_flat_row_ids=data['grid_ids'].astype(np.int64),geometry_source_kind=data['geometry_source_kind'],shared_uv_global=data['uv'],source_linear_initial=linear,source_srgb_initial=srgb)
    query=data['source_query_uv'];x=np.floor(query[:,0]).astype(int);y=np.floor(query[:,1]).astype(int);fx=query[:,0]-x;fy=query[:,1]-y
    lineage['source_corner_bilinear_weights']=np.stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],1).astype(np.float32)
    kind_key=cfg['source'].get('kind_key','source_kind')
    if kind_key in data['field']:
        kinds=data['field'][kind_key];lineage['source_corner_kinds']=np.stack([kinds[y,x],kinds[y,x+1],kinds[y+1,x],kinds[y+1,x+1]],1)
    if 'effective_teacher_blend' in data['field']:
        values=np.asarray(data['field']['effective_teacher_blend']);weights=lineage['source_corner_bilinear_weights']
        lineage['source_conditional_teacher_influence']=sum(values[iy,ix]*weights[:,j] for j,(iy,ix) in enumerate([(y,x),(y,x+1),(y+1,x),(y+1,x+1)])).astype(np.float32)
    absolute_ref=lambda item:dict(item,path=resolve(data['base_dir'],item['path']).resolve().as_posix())
    manifest=dict(schema='R10.certified_new_local_Gaussian_asset.v1',complete=True,new_local_representation=True,CPU_initialized=True,GPU_calls=0,optimizer_updates=0,existing_representation_updates=0,quality_accepted=False,whole_scene_restoration_accepted=False,
        parent_contract=absolute_ref(cfg['parent_contract']),parent_pins=[absolute_ref(item) for item in cfg['parent_pins']],geometry=cfg['geometry'],source=cfg['source'],native_gate=cfg['native_gate'],frames=cfg['frames'],actor_parent_sha256=ACTOR_SHA,r8_code=resolve(data['base_dir'],cfg['r8_code']).resolve().as_posix(),near_m=float(cfg.get('new_patch_near_m',.2)),
        initialization=status,coverage_formula='four equal kernels at square-cell midpoint; not rendered acceptance',lineage_interpretation='shared-field corner categories/conditional influence; not raw-RGB causal mixture',frozen=['centres','normal_frame','scales','opacity','original_A20_all_fields','existing_road','native_context_except_certified_gate'],fresh_loader_requires_training_targets=False,fit_requires_actual_new_local_K_body_guard=True)
    persist(destination,field,data['pose'],lineage,data['gate_path'],data['gate_proof_path'],manifest)
    return dict(status,asset_written=True,asset_directory=str(destination),new_asset_rows=n)

@dataclass
class PersistentLocalAsset:
    directory:Path;manifest:dict;field:dict;lineage:dict;world_from_patch:np.ndarray;native_gate:dict

def load_local_asset(directory,verify_parent_pins=True):
    """Fresh read uses this package and frozen parents, never material targets."""
    directory=Path(directory);manifest=read(directory/'manifest.json');assert manifest['schema']=='R10.certified_new_local_Gaussian_asset.v1' and manifest['complete']
    for record in manifest['files'].values():assert sha(directory/record['path'])==record['sha256']
    if verify_parent_pins:
        pinned(directory,manifest['parent_contract'])
        for item in manifest['parent_pins']:pinned(directory,item)
    field=bank(directory/'local_field.npz');n=len(field['xyz'])
    for key,shape in [('xyz',(n,3)),('scales',(n,3)),('rotation',(n,4)),('coefficients',(n,16,3)),('opacity',(n,))]:assert field[key].shape==shape and field[key].dtype==np.float32 and np.isfinite(field[key]).all()
    assert (field['scales']>0).all() and ((field['opacity']>=0)&(field['opacity']<=1)).all() and np.allclose(np.linalg.norm(field['rotation'],axis=1),1,atol=1e-5)
    lineage=bank(directory/'lineage.npz');assert np.array_equal(lineage['asset_local_row_ids'],np.arange(n))
    pose=np.asarray(read(directory/'pose.json')['world_from_patch'],np.float64);gate=bank(directory/'native_gate.npz')
    proof=read(directory/'native_gate_proof.json')
    assert proof['gate_npz_sha256']==sha(directory/'native_gate.npz') and proof['baseline_contract_sha256']==manifest['parent_contract']['sha256']
    if manifest['native_gate'].get('mode')=='identity_no_retirement':
        assert proof['policy']=='identity_native_noop' and proof['action']=='preserve_all_native_rows'
        assert gate['global_row_ids'].dtype==np.int64 and gate['gate'].dtype==np.float32
        assert gate['global_row_ids'].shape==gate['gate'].shape==(0,)
        assert proof['all_old_native_parameters_unmodified'] is True
        assert proof['new_field_actual_K_and_body_measurement_required'] is True
    else:assert proof['all_five_K_body_exact_zero'] is True
    return PersistentLocalAsset(directory,manifest,field,lineage,pose,gate)

def export_colour_update(asset,destination,coefficients,training_record):
    field={key:value.copy() for key,value in asset.field.items()};coefficients=np.asarray(coefficients,np.float32)
    assert coefficients.shape==field['coefficients'].shape and np.array_equal(coefficients[:,1:],field['coefficients'][:,1:])
    rgb=coefficients[:,0]*C0+.5;assert np.isfinite(rgb).all() and ((rgb>=0)&(rgb<=1)).all();field['coefficients']=coefficients.copy()
    manifest=dict(asset.manifest,parent_local_manifest_sha256=sha(asset.directory/'manifest.json'),training_record=training_record,quality_accepted=False)
    return persist(destination,field,asset.world_from_patch,asset.lineage,asset.directory/'native_gate.npz',asset.directory/'native_gate_proof.json',manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--build',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.build and not args.output:parser.error('--build requires --output')
    result=build_local_asset(args.contract,args.output) if args.build else preflight(args.contract)[0];print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.build and not result['ready']:raise SystemExit(2)

if __name__=='__main__':main()
