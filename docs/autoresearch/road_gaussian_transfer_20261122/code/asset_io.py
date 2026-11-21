"""高斯块迁移资产的CPU构建、来源账本扩展与局部状态保存。"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import native
from drivingeditor_dynamic_edit.v44_factory import _legacy_road_world_quaternions

ROOT=native.ROOT;PARENT=native.PARENT

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def save(path,arrays):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def cpu(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

def extend(prior,n,m):
    result={}
    for key,a in prior.items():
        if a.ndim and len(a)==n:
            fill=np.nan if a.dtype.kind in 'fc' else -1 if a.dtype.kind=='i' else False if a.dtype.kind=='b' else 0
            extra=np.full((m,*a.shape[1:]),fill,dtype=a.dtype)
            result[key]=np.concatenate((a,extra))
        else:result[key]=a.copy()
    return result

def sync_effective(p,r):
    for key,value in [('effective_active',r['active']),('effective_row_rgb',r['reference_srgb']),
                      ('effective_model_xyz',r['model_xyz']),('effective_sigma_m',r['gaussian_sigma_m']),
                      ('effective_road_opacity',r['road_opacity'])]:p[key]=value.copy()

def build_asset(output_dir,reject_new_mask=None):
    """按ownership保留旧行、追加显式迁移行；已存在的输出目录一律保留。"""
    start=time.monotonic();out=Path(output_dir)
    if out.exists():raise FileExistsError('Preserve existing asset: '+str(out))
    r0=load(PARENT/'road.npz');p0=load(PARENT/'provenance.npz')
    a=load(ROOT/'proposal/transferred_rows.npz');own=load(ROOT/'proposal/parent_ownership.npz')
    n=len(r0['active']);m=len(a['world_xyz']);N=n+m
    reject=np.zeros(m,bool) if reject_new_mask is None else np.asarray(reject_new_mask)
    if reject.shape!=(m,) or reject.dtype.kind!='b':raise ValueError('reject_new_mask must be bool[newN]')
    assert own['parent_active'].shape==(n,) and own['parent_opacity'].shape==(n,)
    source=a['source_row_id'].astype(np.int64);assert ((source>=0)&(source<n)).all() and r0['active'][source].all()
    assert np.array_equal(a['reference_srgb'],r0['reference_srgb'][source]),'Transferred RGB must remain exact donor RGB'
    T=r0['model_from_world'];new=dict(active=a['valid'].astype(bool)&~reject,
        model_xyz=a['world_xyz']@T[:3,:3].T+T[:3,3],normal_world=a['normal_world'],
        gaussian_sigma_m=a['gaussian_sigma_m'],reference_srgb=a['reference_srgb'],
        road_opacity=a['road_opacity'],minimum_model_depth=a['minimum_model_depth'])
    r={}
    for key,value in r0.items():
        if value.ndim and len(value)==n:
            if key not in new:raise KeyError('Unregistered parent row field: '+key)
            r[key]=np.concatenate((value,np.asarray(new[key],dtype=value.dtype)))
        else:r[key]=value.copy()
    r['active'][:n]=own['parent_active'];r['road_opacity'][:n]=own['parent_opacity']
    r['gaussian_rotation_world_wxyz']=np.concatenate((_legacy_road_world_quaternions(r0['normal_world'],T,float(r0['meter_to_model'])),a['gaussian_rotation_world_wxyz']))
    r['gaussian_rotation_world_override']=np.r_[np.zeros(n,bool),np.ones(m,bool)]
    p=extend(p0,n,m)
    def newfield(key,value,oldfill=0):
        value=np.asarray(value);p[key]=np.full((N,*value.shape[1:]),oldfill,dtype=value.dtype);p[key][n:]=value
    for key,value in [('world_xyz',a['world_xyz']),('row_rgb',a['reference_srgb']),
                      ('base_row_rgb',a['reference_srgb']),('capacity_source_prior_reference_srgb',a['reference_srgb']),
                      ('capacity_effective_rgb',a['reference_srgb'])]:
        if key in p:p[key][n:]=value
    p['world_keys'][n:]=np.c_[np.full(m,-3,np.int64),-1-np.arange(m)]
    p['atlas_uv'][n:]=np.nan
    for key,value in [('source',2),('confidence',0.),('known',False),('unknown',True),
                      ('effective_material_source',2),('effective_material_available',True),('effective_material_unknown',False),
                      ('surface_supported',True),('adaptive_effective_surface_supported',True),('surface_interpolated',True),
                      ('legal_write',True),('legal_coverage',True),('effective_coverage_legal',True),('requested',True),('W_world',True)]:
        if key in p:p[key][n:]=value
    for key in ['source_raw','source_world_row','parent_world_row_id','effective_material_source_raw']:
        if key in p:p[key][n:]=-1
    for key in ['observed_reference_srgb','source_uv_native','source_distance_m','effective_material_source_uv_native']:
        if key in p:p[key][n:]=np.nan
    # The old row_rgb/source/known/unknown prefix remains its original source ledger.
    sync_effective(p,r)
    newfield('transfer_new_row',np.ones(m,bool),False)
    newfield('transfer_source_world_row_id',source,-1)
    newfield('transfer_patch_id',a['patch_id'].astype(np.int64),-1)
    newfield('transfer_source_uv',a['source_uv'],np.nan)
    newfield('transfer_target_uv',a['target_uv'],np.nan)
    newfield('transfer_source_world_xyz',p0['world_xyz'][source],np.nan)
    newfield('transfer_source_input_index',a['source_input_index'].astype(np.int64),-1)
    newfield('transfer_jacobian_world',a['jacobian_world'],np.nan)
    newfield('transfer_covariance_world',a['covariance_world'],np.nan)
    newfield('transfer_source_normal_residual_m',a['source_normal_residual_m'],np.nan)
    newfield('transfer_source_road_opacity',a['source_road_opacity'],np.nan)
    newfield('transfer_source_reference_srgb',r0['reference_srgb'][source],np.nan)
    newfield('transfer_ownership_weight',a['ownership_weight'],0.)
    p['transfer_ownership_weight'][:n]=1-own['transfer_weight']
    newfield('transfer_K_rejected',reject,False)
    p['transfer_parent_retired']=np.r_[own['parent_retired'],np.zeros(m,bool)]
    p['transfer_parent_modified']=np.r_[own['parent_modified'],np.zeros(m,bool)]
    p['transfer_parent_replacement_weight']=np.r_[own['transfer_weight'],np.zeros(m,np.float32)]
    p['transfer_source_asset']=np.asarray(str(PARENT));p['transfer_source_road_SHA256']=np.asarray(sha(PARENT/'road.npz'))
    p['transfer_source_provenance_SHA256']=np.asarray(sha(PARENT/'provenance.npz'))
    p['transfer_parent_asset_mode']=p0.get('asset_mode',np.asarray('unknown')).copy()
    p['asset_mode']=np.asarray('retrieved_gaussian_patch_transport')
    for key,value in r0.items():
        if key not in ('active','road_opacity'):assert np.array_equal(r[key][:n] if value.ndim and len(value)==n else r[key],value),key
    assert not p['known'][n:].any() and p['unknown'][n:].all() and (p['source'][n:]==2).all()
    assert np.isfinite(r['reference_srgb']).all() and ((r['reference_srgb']>=0)&(r['reference_srgb']<=1)).all()
    assert np.isfinite(r['gaussian_sigma_m']).all() and (r['gaussian_sigma_m']>0).all()
    out.mkdir(parents=True,exist_ok=False);save(out/'road.npz',r);save(out/'provenance.npz',p)
    retired=np.flatnonzero(own['parent_retired']);changed=np.flatnonzero((r0['active']!=r['active'][:n])|(r0['road_opacity']!=r['road_opacity'][:n]))
    trace=dict(parent_world_row_ids=changed,parent_active_before=r0['active'][changed],parent_active_after=r['active'][changed],
        parent_opacity_before=r0['road_opacity'][changed],parent_opacity_after=r['road_opacity'][changed],
        parent_replacement_weight=own['transfer_weight'][changed],retired_parent_world_row_ids=retired,
        retired_parent_opacity_before=r0['road_opacity'][retired],new_world_row_ids=n+np.arange(m),
        new_source_world_row_ids=source,new_patch_ids=a['patch_id'],new_K_rejected=reject,
        source_asset=np.asarray(str(PARENT)),source_road_SHA256=p['transfer_source_road_SHA256'])
    save(out/'transfer_trace.npz',trace)
    record=dict(stage='zero_step_retrieved_Gaussian_transfer',source_asset=str(PARENT),
        source_road_SHA256=p['transfer_source_road_SHA256'].item(),source_provenance_SHA256=p['transfer_source_provenance_SHA256'].item(),
        proposal_SHA256=sha(ROOT/'proposal/transferred_rows.npz'),ownership_SHA256=sha(ROOT/'proposal/parent_ownership.npz'),
        parent_rows=n,new_rows=m,total_rows=N,parent_retired=len(retired),parent_transition=int((own['parent_modified']&~own['parent_retired']).sum()),
        rejected_new_rows=int(reject.sum()),active_rows=int(r['active'].sum()),new_position_source='retrieved_similar_3D_Gaussian_block; not real observation at new location',
        source_code=2,new_known=False,new_unknown=True,new_observed_RGB_available=False,
        source_confidence=0.,source_confidence_meaning='No calibrated observation confidence at transferred target positions; independent of copied RGB/opacity.',
        parent_geometry_and_RGB_exact=True,old_source_ledger_prefix_preserved=True,legacy_quaternion_override_false=True,new_quaternion_override_true=True,
        GPU_calls=0,optimizer_updates=0,new_diffusion_calls=0,seconds=time.monotonic()-start)
    native.pin(PARENT,out,record)
    write(out/'transfer_receipt.json',dict(complete=True,**record))
    return out

def _native_difference(value,baseline,name):
    """先与原生模板逐值比较，不能拿重新编码的CPU物理尺度作为baseline。"""
    if hasattr(value,'detach'):
        v=value.detach();b=baseline.detach() if hasattr(baseline,'detach') else native.torch.as_tensor(baseline,device=v.device)
        if name=='opacity' and v.numel()==b.numel():v=v.reshape(b.shape)
        if tuple(v.shape)!=tuple(b.shape):raise ValueError(name+' shape must match native active order')
        b=b.to(device=v.device,dtype=v.dtype)
        if not bool(native.torch.isfinite(v).all()):raise ValueError('Nonfinite '+name)
        changed=(v!=b).cpu().numpy();values=v.cpu().numpy()
    else:
        values=np.asarray(value);b=cpu(baseline)
        if name=='opacity' and values.size==b.size:values=values.reshape(b.shape)
        if values.shape!=b.shape:raise ValueError(name+' shape must match native active order')
        if not np.isfinite(values).all():raise ValueError('Nonfinite '+name)
        changed=values!=b
    return values,changed.reshape(len(values),-1)

def write_state_asset(ctx,output_dir,colors=None,opacity=None,scales=None,record=None):
    """保存原生active顺序的局部状态；scales输入为原生model单位。

    RGB只允许新迁移行变化；opacity和切线sigma只允许资格认证的接缝行变化。
    未变分量直接保留ctx.road的CPU原值，不经历float32物理量往返。
    """
    start=time.monotonic();out=Path(output_dir)
    if out.exists():raise FileExistsError('Preserve existing state asset: '+str(out))
    r0=ctx.road;r={k:v.copy() for k,v in r0.items()};p={k:v.copy() for k,v in ctx.provenance.items()}
    active=np.asarray(ctx.active_world_rows,dtype=np.int64)
    assert np.array_equal(active,np.flatnonzero(r0['active'])),'Native active identity/order changed'
    new=np.asarray(p['transfer_new_row'],bool);new_active=new[active]
    from seam_domain import seam_rows
    seam=np.asarray(seam_rows(ROOT)['new_seam_mask'])
    if seam.dtype.kind!='b' or seam.shape!=(int(new.sum()),):raise ValueError('new_seam_mask must match full new-row order')
    seam_world=np.zeros(len(new),bool);seam_world[new]=seam
    boundary=new_active&seam_world[active]
    changes={};native_packet=ctx.base_packet(ctx.frames[0])
    if colors is not None:
        values,changed=_native_difference(colors,ctx.base_rgb,'colors');rows=changed.any(1)
        assert not (rows&~new_active).any(),'Frozen old RGB changed'
        if ((values<0)|(values>1)).any():raise ValueError('RGB must remain in [0,1]')
        r['reference_srgb'][active[rows]]=values[rows];changes['RGB']=active[rows]
    else:changes['RGB']=np.empty(0,np.int64)
    if opacity is not None:
        values,changed=_native_difference(opacity,native_packet['opacities'],'opacity');rows=changed.any(1)
        assert not (rows&~boundary).any(),'Opacity changed outside new boundary rows'
        if ((values<0)|(values>1)).any():raise ValueError('Opacity must remain in [0,1]')
        if (values.reshape(-1)[rows]>.999+1e-7).any():raise ValueError('Seam opacity exceeds qualified.999 maximum')
        r['road_opacity'][active[rows]]=values.reshape(-1)[rows];changes['opacity']=active[rows]
    else:changes['opacity']=np.empty(0,np.int64)
    if scales is not None:
        values,changed=_native_difference(scales,native_packet['scales'],'scales');rows=changed.any(1)
        assert not (rows&~boundary).any(),'Scales changed outside new boundary rows'
        assert not changed[:,2].any(),'Normal-axis sigma is frozen'
        if (values<=0).any():raise ValueError('Native scales must remain positive')
        ratio=values.astype(np.float64)[:,:2]/cpu(native_packet['scales']).astype(np.float64)[:,:2]
        if ((ratio[rows]<.9-1e-6)|(ratio[rows]>1.15+1e-6)).any():raise ValueError('Seam tangent sigma exceeds qualified[.9,1.15] multiplier')
        rr,axis=np.nonzero(changed)
        r['gaussian_sigma_m'][active[rr],axis]=values.astype(np.float64)[rr,axis]/float(r0['meter_to_model'])
        changes['sigma']=active[rows]
    else:changes['sigma']=np.empty(0,np.int64)
    for key,value in r0.items():
        if key not in ('reference_srgb','road_opacity','gaussian_sigma_m'):
            assert np.array_equal(r[key],value),'Frozen road field changed: '+key
    old=~new
    for key in ['reference_srgb','road_opacity','gaussian_sigma_m']:
        assert np.array_equal(r[key][old],r0[key][old]),'Old row changed: '+key
    assert np.array_equal(r['gaussian_rotation_world_wxyz'],r0['gaussian_rotation_world_wxyz'])
    assert np.array_equal(r['gaussian_rotation_world_override'],r0['gaussian_rotation_world_override'])
    # The source RGB is immutable even when local per-patch gain/bias changes current row RGB.
    if 'transfer_source_reference_srgb' not in p:
        p['transfer_source_reference_srgb']=np.full_like(r['reference_srgb'],np.nan)
        source=np.asarray(p['transfer_source_world_row_id'][new],np.int64)
        p['transfer_source_reference_srgb'][new]=load(PARENT/'road.npz')['reference_srgb'][source]
    if 'row_rgb' in p:p['row_rgb'][new]=r['reference_srgb'][new]
    if 'capacity_effective_rgb' in p:p['capacity_effective_rgb']=r['reference_srgb'].copy()
    sync_effective(p,r)
    p['transfer_fit_RGB_changed']=np.isin(np.arange(len(new)),changes['RGB'])
    p['transfer_fit_opacity_changed']=np.isin(np.arange(len(new)),changes['opacity'])
    p['transfer_fit_sigma_changed']=np.isin(np.arange(len(new)),changes['sigma'])
    assert not p['known'][new].any() and p['unknown'][new].all() and (p['source'][new]==2).all()
    out.mkdir(parents=True,exist_ok=False);save(out/'road.npz',r);save(out/'provenance.npz',p)
    source_trace=Path(ctx.asset_dir)/'transfer_trace.npz'
    if source_trace.exists():
        import shutil
        shutil.copy2(source_trace,out/'transfer_trace.npz')
    save(out/'state_changes.npz',dict(RGB_world_row_ids=changes['RGB'],opacity_world_row_ids=changes['opacity'],
          sigma_world_row_ids=changes['sigma'],input_asset=np.asarray(str(ctx.asset_dir))))
    receipt=dict(record or {})
    receipt.update(stage=receipt.get('stage','local_seam_fit_state'),input_asset=str(ctx.asset_dir),
        input_road_SHA256=sha(Path(ctx.asset_dir)/'road.npz'),RGB_changed_rows=len(changes['RGB']),
        opacity_changed_rows=len(changes['opacity']),tangent_sigma_changed_rows=len(changes['sigma']),
        old_rows_all_exact=True,XYZ_normal_active_minDepth_transforms_quaternions_exact=True,
        unchanged_native_components_CPU_bitexact=True,scales_input_units='native model; divide by meter_to_model only for actually changed components',
        seam_qualification=str(ROOT/'qualification/seam_rows.npz'),seam_opacity_max=.999,seam_tangent_sigma_multiplier=[.9,1.15],
        new_positions_remain_retrieved_inference=True,new_diffusion_calls=0,asset_save_nativeF=0,asset_save_nativeB=0,
        asset_save_optimizer_updates=0,asset_save_seconds=time.monotonic()-start)
    native.pin(PARENT,out,receipt)
    write(out/'state_receipt.json',dict(complete=True,**receipt))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'candidate/asset');args=parser.parse_args()
    print(build_asset(args.output),flush=True)
