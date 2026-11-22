"""Fresh-process CPU audit of persistent colour-only road/actor assets.

Never imports a renderer, torch or CUDA. Native renders are explicit external
inputs to render-check; preparing a command never executes it.
"""
import argparse,copy,hashlib,json,os,sys,zipfile,pickletools
from pathlib import Path
import numpy as np

SOURCE_SHA='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'
FRAMES=[86,110,134,179,198]
ROAD_COLOUR={'reference_srgb','reference_linear'}
PROVENANCE_COLOUR={'effective_row_rgb'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):
    p=Path(path);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise ValueError('Refusing to overwrite audit evidence: '+str(p))
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def array_record(a):
    a=np.asarray(a);return {'dtype':a.dtype.str,'shape':list(a.shape),'C_order_bytes_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
def same(a,record):return array_record(a)==record
def npz_record(path):
    with np.load(path,allow_pickle=False) as z:return {'file':pin(path),'fields':{k:array_record(z[k]) for k in z.files}}
def eotf(x):
    x=np.asarray(x,dtype=np.float32);return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4).astype(np.float32)
def scene_pose_record(cfg,frames):
    scene=read(cfg['scene_manifest']);binding=scene['world_to_model']
    transform=np.loadtxt(binding) if isinstance(binding,str) else np.asarray(binding,np.float64)
    if transform.shape!=(4,4):transform=transform.reshape(4,4)
    records=[]
    for raw in frames:
        path=Path(cfg['observations'])/f'{raw:03d}.json';obs=read(path)
        frame=next(x for x in scene['frames'] if int(x['key'])==raw)
        poses={}
        for actor in cfg['actors']:
            box=next(x for x in obs['boxes'] if x['track']==actor['track'])
            world=np.asarray(box['world_from_actor'],np.float64).reshape(4,4)
            poses[actor['name']]={'world_from_actor':array_record(world),
               'model_from_actor_float32':array_record((transform@world).astype(np.float32))}
        records.append({'raw':raw,'observation':pin(path),'timestamp_micros':obs['timestamp_micros'],
          'native_model_time':frame['model_time'],'source_time_s':frame['source_time_s'],
          'native_image_name':frame['image_name'],'poses':poses})
    return {'world_to_model':array_record(transform),'transform_file':pin(binding) if isinstance(binding,str) else None,'frames':records}


def freeze(a):
    cfg=read(a.baseline_contract);effective=copy.deepcopy(cfg);effective['frames']=a.frames
    assert a.frames==FRAMES and a.train_frames==[86,110]
    assert sha(a.source_field)==a.source_sha==SOURCE_SHA
    road_path=Path(effective['road_asset'])/'road.npz';provenance_path=Path(effective['road_asset'])/'provenance.npz'
    road=load(road_path);guard=load(a.guard);guard_receipt=read(Path(a.guard).parent/'receipt.json')
    n=len(road['active']);assert np.array_equal(guard['frames'],a.frames)
    assert np.array_equal(guard['road_active_world_rows'],np.flatnonzero(road['active']))
    assert guard_receipt['complete'] and guard_receipt['guard_sha256']==sha(a.guard)
    assert guard_receipt['source_sha256']==SOURCE_SHA and guard_receipt['source_RGBA_exact'] is True
    assert guard_receipt['missing_K_rows_filled_zero'] is False
    guard_baseline=read(a.guard_baseline_contract or a.baseline_contract);guard_baseline['frames']=a.frames
    assert guard_baseline==effective,'Guard and export effective baselines differ'
    assert sha(a.guard_baseline_contract or a.baseline_contract)==guard_receipt['baseline_contract_sha256']
    mass=guard['road_K_mass'];assert mass.shape==(5,len(guard['road_active_world_rows'])) and np.isfinite(mass).all() and (mass>=0).all()
    assert np.array_equal(guard['road_Kpositive_union'],(mass>0).any(0))
    assert np.array_equal(guard['road_Kzero_allviews'],(mass==0).all(0))
    frozen_road_ids=guard['road_active_world_rows'][guard['road_Kpositive_union']]
    actors=[]
    for spec in effective['actors']:
        asset=Path(spec['asset']);assert sha(asset)==spec['sha256'];z=load(asset);name=spec['name'];count=len(z['xyz'])
        km=guard[name+'_K_mass'];vm=guard[name+'_strict_train_V_mass']
        assert km.shape==vm.shape==(5,count) and np.isfinite(km).all() and np.isfinite(vm).all() and (km>=0).all() and (vm>=0).all()
        assert np.array_equal(guard[name+'_Kpositive_union'],(km>0).any(0))
        assert np.array_equal(guard[name+'_strict_train_Vpositive_union'],(vm>0).any(0))
        for i,raw in enumerate(a.frames):
            if raw not in a.train_frames:assert not vm[i].any()
        actors.append({'spec':spec,'asset':npz_record(asset),'rows':count,
          'higher_SH':array_record(z['coefficients'][:,1:,:]),
          'Kpositive_DC':array_record(z['coefficients'][guard[name+'_Kpositive_union'],0,:]),
          'Kpositive_rows':int(guard[name+'_Kpositive_union'].sum()),
          'Kpositive_row_ids':array_record(np.flatnonzero(guard[name+'_Kpositive_union']))})
    cameras=[];masks=[];baseline_native=[]
    native_camera=read(Path(a.guard).parent/'camera_timestamp_audit.json');native_camera=native_camera.get('rows',native_camera) if isinstance(native_camera,dict) else native_camera
    for raw in a.frames:
        path=Path(a.camera_reference_cache)/f'{raw:03d}/view_cache.npz';z=load(path)
        camera=next(x for x in native_camera if x['raw']==raw)
        assert np.array_equal(np.asarray(camera['actual_K'],np.float64),z['camera_K'])
        cameras.append({'raw':raw,'cache':pin(path),'K_float64':array_record(z['camera_K']),
          'expected_K':z['camera_K'].tolist(),'native_baseline_audit':camera})
        mask_path=Path(a.rectified_masks)/f'{raw:03d}/rectified_masks.npz';mm=load(mask_path)
        masks.append({'raw':raw,'file':pin(mask_path),'protected':array_record(mm['protected']),
          'strict_actor_V_available':raw in a.train_frames,
          'strict_actor_V_role':'training visibility supervision' if raw in a.train_frames else 'NOT OPENED: evaluation only, not zero-valued ground truth'})
        baseline_native.append({'raw':raw,'file':pin(Path(a.baseline_native)/f'{raw:03d}'/a.render_filename)})
    fixed=[pin(effective[k]) for k in ['scene_manifest','scene_proof','context_plan','actor_field_module']]
    for relabel in effective.get('owner_relabels',[]):
        assert sha(relabel['path'])==relabel['sha256'];fixed.append(pin(relabel['path']))
    result={'schema':'R9.joint_persistent_colour_asset_frozen_invariants.v1','complete':True,
      'baseline_contract':pin(a.baseline_contract),'effective_baseline_configuration':effective,
      'source_field':pin(a.source_field),'source_field_sha256':SOURCE_SHA,'frames':a.frames,'train_frames':a.train_frames,
      'road':npz_record(road_path),'road_provenance':npz_record(provenance_path),'actors':actors,
      'road_Kpositive_reference_srgb':array_record(road['reference_srgb'][frozen_road_ids]),
      'road_Kpositive_world_row_ids':array_record(frozen_road_ids),'road_Kpositive_rows':len(frozen_road_ids),
      'guard':pin(a.guard),'guard_receipt':pin(Path(a.guard).parent/'receipt.json'),
      'guard_shape_complete_finite_nonnegative':True,'guard_union_bitexact_recomputed':True,
      'fixed_dependencies':fixed,'poses':scene_pose_record(effective,a.frames),'cameras':cameras,'masks':masks,'baseline_native':baseline_native,
      'allowed_road_changes':sorted(ROAD_COLOUR),'allowed_new_road_fields':['reference_linear'],
      'allowed_provenance_changes':sorted(PROVENANCE_COLOUR),'allowed_new_provenance_fields':[],
      'allowed_actor_changes':'coefficients[:,0,:] only; all SH1..15 and all other original fields byte exact',
      'native_render_parity_executed':False,'fresh_CPU_process':{'pid':os.getpid(),'parent_pid':os.getppid(),'torch_imported':False},
      'GPU_forward':0,'diffusion_calls':0,'updates':0}
    write(a.output,result);print(json.dumps({'decision':'FROZEN_CPU_INVARIANTS','output':pin(a.output),'road_rows':n,
      'road_Kpositive_frozen_rows':len(frozen_road_ids),'actors':[{'name':x['spec']['name'],'rows':x['rows'],'Kpositive':x['Kpositive_rows']} for x in actors]}))


def mapped(path,maps):
    text=str(path).replace('\\','/')
    for value in sorted(maps or [],key=len,reverse=True):
        old,new=value.split('=',1);old=old.replace('\\','/').rstrip('/')
        if text==old or text.startswith(old+'/'):return Path(new+text[len(old):])
    return Path(path)
def record_matches(path,record):
    return Path(path).is_file() and sha(path)==record['sha256']
def immutable_fields(current,reference,allowed):
    return {k:k in current and (k in allowed or same(current[k],v)) for k,v in reference.items()}
def torch_zip_storage_record(path):
    """No unpickling: compare tensor metadata before binding and raw storages."""
    with zipfile.ZipFile(path) as z:
        names=z.namelist();pkl=next(n for n in names if n.endswith('/data.pkl'))
        prefix=[];found=False
        for op,arg,_ in pickletools.genops(z.read(pkl)):
            if arg=='binding':found=True;break
            prefix.append((op.name,repr(arg)))
        if not found:raise ValueError('Unrecognized gate checkpoint: no explicit binding after tensor fields')
        storage={n.split('/data/',1)[1]:{'bytes':z.getinfo(n).file_size,'sha256':hashlib.sha256(z.read(n)).hexdigest()}
                 for n in names if '/data/' in n and n.split('/data/',1)[1].isdigit()}
        if not storage:raise ValueError('No raw tensor storages in gate checkpoint')
        return {'tensor_prefix_pickle_opcodes':hashlib.sha256(repr(prefix).encode()).hexdigest(),'raw_tensor_storages':storage}


def check_assets(a,emit=True):
    frozen=read(a.frozen);asset=Path(a.asset);maps=a.path_map
    cfg=frozen['effective_baseline_configuration'];checks={};details={}
    source=mapped(a.source_field or frozen['source_field']['path'],maps)
    checks['source_field_SHA_unchanged']=record_matches(source,frozen['source_field']) and sha(source)==SOURCE_SHA
    guard_path=mapped(frozen['guard']['path'],maps);checks['guard_SHA_exact']=record_matches(guard_path,frozen['guard']);guard=load(guard_path)
    for label,record in [('road',frozen['road']),('road_provenance',frozen['road_provenance'])]:
        checks['immutable_baseline_'+label+'_file_SHA']=record_matches(mapped(record['file']['path'],maps),record['file'])
    for i,record in enumerate(frozen['fixed_dependencies']):
        checks['fixed_dependency_'+str(i)+'_SHA']=record_matches(mapped(record['path'],maps),record)
    for frame in frozen['poses']['frames']:
        record=frame['observation'];checks[f"observation_{frame['raw']}_SHA"]=record_matches(mapped(record['path'],maps),record)
    if frozen['poses'].get('transform_file'):
        record=frozen['poses']['transform_file'];checks['pose_transform_file_SHA']=record_matches(mapped(record['path'],maps),record)
    road_path=asset/'road/road.npz';road=load(road_path);reference=frozen['road']['fields']
    details['road_fields']=immutable_fields(road,reference,ROAD_COLOUR)
    checks['road_all_original_noncolour_fields_byteexact']=all(details['road_fields'].values())
    checks['road_field_set_exact_except_declared_linear']=set(road)==set(reference)|{'reference_linear'}
    colour=road['reference_srgb'];linear=road['reference_linear'];expected=eotf(colour)
    checks['road_sRGB_f32_N3_finite_range']=colour.dtype==np.float32 and colour.shape==(len(road['active']),3) and np.isfinite(colour).all() and (colour>=0).all() and (colour<=1).all()
    checks['road_linear_f32_N3_finite_range']=linear.dtype==np.float32 and linear.shape==colour.shape and np.isfinite(linear).all() and (linear>=0).all() and (linear<=1).all()
    err=float(abs(linear-expected).max());checks['road_linear_EOTF_consistent']=err<=3e-7
    details['road_linear_EOTF_max_abs']=err;details['road_linear_EOTF_bitexact']=bool(np.array_equal(linear,expected))
    ids=guard['road_active_world_rows'][guard['road_Kpositive_union']]
    checks['road_Kpositive_world_ids_exact']=same(ids,frozen['road_Kpositive_world_row_ids'])
    checks['road_Kpositive_RGB_byteexact']=same(colour[ids],frozen['road_Kpositive_reference_srgb'])
    pp=asset/'road/provenance.npz';provenance=load(pp);ref=frozen['road_provenance']['fields']
    details['provenance_fields']=immutable_fields(provenance,ref,PROVENANCE_COLOUR)
    checks['provenance_field_set_exact']=set(provenance)==set(ref)
    checks['provenance_all_original_noncolour_fields_byteexact']=all(details['provenance_fields'].values())
    for key in PROVENANCE_COLOUR:
        checks['provenance_'+key+'_equals_effective_sRGB']=key in provenance and array_record(provenance[key])==array_record(colour)
    actor_outputs=[]
    for original in frozen['actors']:
        name=original['spec']['name'];path=asset/'actors'/name/'actor.npz';actor=load(path);ref=original['asset']['fields']
        checks[name+'_immutable_baseline_SHA']=record_matches(mapped(original['asset']['file']['path'],maps),original['asset']['file'])
        fields=immutable_fields(actor,ref,{'coefficients'});details[name+'_fields']=fields
        checks[name+'_all_other_fields_byteexact']=all(fields.values());checks[name+'_field_set_exact']=set(actor)==set(ref)
        coef=actor['coefficients'];checks[name+'_coefficients_dtype_shape_exact']=coef.dtype.str==ref['coefficients']['dtype'] and list(coef.shape)==ref['coefficients']['shape']
        checks[name+'_coefficients_finite']=bool(np.isfinite(coef).all());checks[name+'_SH_1_through15_byteexact']=same(coef[:,1:,:],original['higher_SH'])
        union=guard[name+'_Kpositive_union'];checks[name+'_Kpositive_DC_byteexact']=same(coef[union,0,:],original['Kpositive_DC'])
        actor_outputs.append({'name':name,'asset':pin(path),'rows':len(coef),'active':int(actor['active'].sum()),
           'DC_changed':not same(coef,ref['coefficients']),'Kpositive_rows_frozen':int(union.sum())})
    joint_path=asset/'joint_contract.json';joint=read(joint_path);normalized=copy.deepcopy(joint)
    checks['joint_contract_same_top_level_keys']=set(joint)==set(cfg)
    for key in ['road_asset','scene_manifest','scene_proof']:normalized[key]=cfg[key]
    checks['joint_actor_count_exact']=len(joint['actors'])==len(cfg['actors'])
    for current,original in zip(normalized['actors'],cfg['actors']):
        current['asset']=original['asset'];current['sha256']=original['sha256']
    checks['joint_contract_except_asset_bindings_exact']=normalized==cfg
    checks['joint_road_points_to_exported_asset']=mapped(joint['road_asset'],maps).resolve()==(asset/'road').resolve()
    for spec in joint['actors']:
        path=asset/'actors'/spec['name']/'actor.npz'
        checks[spec['name']+'_joint_binding_exact']=mapped(spec['asset'],maps).resolve()==path.resolve() and spec['sha256']==sha(path)
    scene_path=mapped(joint['scene_manifest'],maps);new_scene=read(scene_path);old_scene=read(mapped(cfg['scene_manifest'],maps))
    checks['persistent_scene_same_field_set']=set(new_scene)==set(old_scene)
    checks['persistent_scene_except_relocated_road_and_static_gate_bindings_exact']=all(new_scene.get(k)==v for k,v in old_scene.items() if k not in {'road','static_gates'})
    checks['persistent_scene_and_proof_exist']=scene_path.is_file() and mapped(joint['scene_proof'],maps).is_file()
    old_proof=read(mapped(cfg['scene_proof'],maps));new_proof=read(mapped(joint['scene_proof'],maps))
    for label in ['indices','checkpoint']:
        old_record=old_proof['gates'][label];new_record=new_proof['gates'][label]
        old_path=mapped(old_record['path'],maps);new_path=mapped(new_record['path'],maps)
        checks['old_static_gate_'+label+'_SHA_exact']=record_matches(old_path,old_record)
        checks['exported_static_gate_'+label+'_proof_SHA_exact']=record_matches(new_path,new_record)
        if label=='indices':
            old_values=load(old_path);new_values=load(new_path)
            checks['static_gate_ids_all_byteexact']=set(old_values)==set(new_values) and all(array_record(v)==array_record(new_values[k]) for k,v in old_values.items())
        else:
            before=torch_zip_storage_record(old_path);after=torch_zip_storage_record(new_path)
            checks['static_gate_tensor_metadata_and_raw_storage_byteexact']=before==after
            details['static_gate_checkpoint_CPU_zip_storage']=after
    proof_path=asset/'proof.json';proof=read(proof_path)
    checks['asset_proof_source_field_SHA_exact']=proof.get('source_field_sha256')==SOURCE_SHA
    checks['asset_proof_guard_SHA_exact']=proof.get('K_guard_sha256')==frozen['guard']['sha256']
    for camera in frozen['cameras']:
        checks[f"camera_cache_{camera['raw']}_SHA"]=record_matches(mapped(camera['cache']['path'],maps),camera['cache'])
    for mask in frozen['masks']:
        checks[f"protected_masks_{mask['raw']}_SHA"]=record_matches(mapped(mask['file']['path'],maps),mask['file'])
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'R9.joint_persistent_colour_asset_CPU_check.v1','complete':True,'decision':'PASS' if not failed else 'FAIL',
      'frozen_contract':pin(a.frozen),'asset_proof':pin(proof_path),'joint_contract':pin(joint_path),'road_asset':pin(road_path),
      'road_provenance':pin(pp),'actor_assets':actor_outputs,'source_field':pin(source),'checks':{k:bool(v) for k,v in checks.items()},
      'failed_checks':failed,'field_details':details,'frames':frozen['frames'],
      'protected_K_colours_of_measured_positive_rows_byteexact':not any('Kpositive' in k for k in failed),
      'strict_actor_V_availability':{str(r):r in frozen['train_frames'] for r in frozen['frames']},
      'evaluation_only_missing_V_is_zero_GT':False,'native_render_parity':'NOT_EXECUTED: explicit native render evidence required',
      'fresh_CPU_process':{'pid':os.getpid(),'parent_pid':os.getppid(),'torch_imported':'torch' in sys.modules},
      'GPU_forward':0,'diffusion_calls':0,'updates':0}
    if emit:
        write(a.output,result);print(json.dumps({'decision':result['decision'],'failed_checks':failed,'output':pin(a.output)}))
    return result


def protected_rgb(rgb,mask):
    rgb=np.asarray(rgb)
    if rgb.ndim==3 and rgb.shape[:2]==mask.shape:return rgb[mask]
    if rgb.ndim==3 and rgb.shape[1:]==mask.shape:return rgb[:,mask]
    raise ValueError('RGB/mask shape mismatch')


def render_check(a):
    asset_result=check_assets(a,emit=False);frozen=read(a.frozen);checks={'CPU_asset_invariants_PASS':asset_result['decision']=='PASS'}
    native_receipt=read(a.native_run_receipt)
    checks['native_run_complete']=native_receipt.get('complete') is True
    checks['native_run_bound_to_persisted_asset_proof']=native_receipt.get('asset_proof_sha256')==sha(Path(a.asset)/'proof.json')
    checks['native_run_bound_to_source_field']=native_receipt.get('source_field_sha256')==SOURCE_SHA
    camera_rows=read(a.native_camera_audit);camera_rows=camera_rows.get('rows',camera_rows) if isinstance(camera_rows,dict) else camera_rows
    checks['fresh_native_all_five_cameras']=sorted(x['raw'] for x in camera_rows)==frozen['frames']
    rows=[]
    for raw in frozen['frames']:
        before_path=Path(a.native_expected)/f'{raw:03d}'/a.render_filename
        after_path=Path(a.native_reloaded)/f'{raw:03d}'/a.render_filename
        before=load(before_path);after=load(after_path);parity={}
        checks[f'{raw}_joint_all_key_set_exact']=set(before)==set(after)
        checks[f'{raw}_joint_required_keys_present']=set(a.parity_keys).issubset(before) and set(a.parity_keys).issubset(after)
        for key in sorted(set(before)|set(after)):
            parity[key]=key in before and key in after and array_record(before[key])==array_record(after[key])
            checks[f'{raw}_{key}_reload_byteexact']=parity[key]
            if key in before and key in after and before[key].dtype.kind in 'fc':
                checks[f'{raw}_{key}_all_finite']=bool(np.isfinite(before[key]).all() and np.isfinite(after[key]).all())
        component_before_path=Path(a.native_expected)/f'{raw:03d}'/a.component_render_filename
        component_after_path=Path(a.native_reloaded)/f'{raw:03d}'/a.component_render_filename
        component_before=load(component_before_path);component_after=load(component_after_path)
        checks[f'{raw}_component_all_key_set_exact']=set(component_before)==set(component_after)
        component_parity={}
        for key in sorted(set(component_before)|set(component_after)):
            exact=key in component_before and key in component_after and array_record(component_before[key])==array_record(component_after[key])
            component_parity[key]=exact;checks[f'{raw}_component_{key}_byteexact']=exact
            if key in component_before and key in component_after and component_before[key].dtype.kind in 'fc':
                checks[f'{raw}_component_{key}_all_finite']=bool(np.isfinite(component_before[key]).all() and np.isfinite(component_after[key]).all())
        ref=next(x for x in frozen['baseline_native'] if x['raw']==raw)['file'];baseline_path=mapped(ref['path'],a.path_map)
        checks[f'{raw}_baseline_render_SHA_unchanged']=record_matches(baseline_path,ref);baseline=load(baseline_path)
        mask_ref=next(x for x in frozen['masks'] if x['raw']==raw);mask_path=mapped(mask_ref['file']['path'],a.path_map)
        mask=load(mask_path)['protected'].astype(bool)
        checks[f'{raw}_K_mask_nonempty']=bool(mask.any())
        checks[f'{raw}_protected_K_RGB_byteexact_A0']=array_record(protected_rgb(after['rgb'],mask))==array_record(protected_rgb(baseline['rgb'],mask))
        checks[f'{raw}_alpha_global_byteexact_A0']=array_record(after['alpha'])==array_record(baseline['alpha'])
        camera=next(x for x in camera_rows if x['raw']==raw);reference=next(x for x in frozen['cameras'] if x['raw']==raw)
        checks[f'{raw}_native_camera_intrinsics_K_exact']=same(np.asarray(camera['actual_K'],np.float64),reference['K_float64'])
        pose=next(x for x in frozen['poses']['frames'] if x['raw']==raw)
        checks[f'{raw}_timestamp_and_native_model_time_exact']=camera['timestamp_micros']==pose['timestamp_micros'] and camera['native_model_time']==pose['native_model_time']
        rows.append({'raw':raw,'expected':pin(before_path),'reloaded':pin(after_path),'reload_channel_byteexact':parity,
          'expected_component':pin(component_before_path),'reloaded_component':pin(component_after_path),'component_channel_byteexact':component_parity,
          'protected_K_pixels':int(mask.sum()),'strict_V_available_for_training':raw in frozen['train_frames'],
          'evaluation_only_zero_V_is_GT':False})
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'R9.joint_persistent_asset_native_file_parity_CPU_readback.v1','complete':True,
      'decision':'PASS' if not failed else 'FAIL','frozen_contract':pin(a.frozen),'asset_CPU_check':asset_result,
      'native_execution_receipt':pin(a.native_run_receipt),'native_camera_audit':pin(a.native_camera_audit),
      'checks':checks,'failed_checks':failed,'rows':rows,'native_render_was_external_explicit_execution':True,
      'fresh_CPU_process':{'pid':os.getpid(),'parent_pid':os.getppid(),'torch_imported':'torch' in sys.modules},
      'GPU_forward_performed_by_this_checker':0,'diffusion_calls':0,'updates':0}
    write(a.output,result);print(json.dumps({'decision':result['decision'],'failed_checks':failed,'output':pin(a.output)}))
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    freeze_parser=sub.add_parser('freeze',help='CPU: bind original assets, source, trajectory and measured K guard before export')
    freeze_parser.add_argument('--baseline-contract',type=Path,required=True)
    freeze_parser.add_argument('--guard-baseline-contract',type=Path)
    freeze_parser.add_argument('--source-field',type=Path,required=True)
    freeze_parser.add_argument('--source-sha',default=SOURCE_SHA)
    freeze_parser.add_argument('--guard',type=Path,required=True)
    freeze_parser.add_argument('--camera-reference-cache',type=Path,required=True)
    freeze_parser.add_argument('--rectified-masks',type=Path,required=True)
    freeze_parser.add_argument('--baseline-native',type=Path,required=True)
    freeze_parser.add_argument('--frames',type=int,nargs='+',default=FRAMES)
    freeze_parser.add_argument('--train-frames',type=int,nargs='+',default=[86,110])
    freeze_parser.add_argument('--render-filename',default='joint_render.npz')
    freeze_parser.add_argument('--output',type=Path,required=True)
    for name in ['check','render-check']:
        q=sub.add_parser(name,help='CPU: verify persisted arrays' if name=='check' else 'CPU: verify files from a separately scheduled native reload')
        q.add_argument('--frozen',type=Path,required=True);q.add_argument('--asset',type=Path,required=True)
        q.add_argument('--source-field',type=Path);q.add_argument('--path-map',action='append',default=[])
        q.add_argument('--output',type=Path,required=True)
        if name=='render-check':
            q.add_argument('--native-expected',type=Path,required=True)
            q.add_argument('--native-reloaded',type=Path,required=True)
            q.add_argument('--native-run-receipt',type=Path,required=True)
            q.add_argument('--native-camera-audit',type=Path,required=True)
            q.add_argument('--render-filename',default='joint_render.npz')
            q.add_argument('--component-render-filename',default='component_render.npz')
            q.add_argument('--parity-keys',nargs='+',default=['rgb','alpha','depth_expected_m','road_alphaT','actor_alphaT','context_alphaT'])
    return p


def main():
    if any(x in sys.modules for x in ['torch','gsplat']):raise RuntimeError('Use a fresh CPU process, not a live renderer process')
    a=parser().parse_args()
    if a.mode=='freeze':freeze(a);return
    result=check_assets(a) if a.mode=='check' else render_check(a)
    if result['decision']!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
