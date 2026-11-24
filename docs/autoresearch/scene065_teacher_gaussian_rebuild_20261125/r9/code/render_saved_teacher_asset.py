"""Reload the R9 coverage-fix model at its configured target step without Adam.

python code/render_saved_teacher_asset.py --run /.../training_v1
Training paths are resolved from train/receipt.json and model_final.pt bindings.
The complete checkpoint is hashed only. --cpu-preflight-only never uses CUDA.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, inspect, json, sys, time, traceback
import numpy as np

TOL=1e-6
PARAMETERS=('xyz','log_scales','raw_quaternion','opacity_logits','dc')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=p.as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def pinned(s):
    p=Path(s['path']);assert p.is_file() and sha(p)==s['sha256'],('Changed/missing bound file',str(p));return p
def pins(q):
    if isinstance(q,dict):
        if isinstance(q.get('path'),str) and isinstance(q.get('sha256'),str):yield q
        else:
            for v in q.values():yield from pins(v)
    elif isinstance(q,list):
        for v in q:yield from pins(v)
def bound(q,*names):
    for name in names:
        if name in q:return pinned(q[name])
    raise AssertionError(('Missing actual binding',names,list(q)))
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def module(p,name):
    p=Path(p).resolve();sys.path.insert(0,str(p.parent));s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def array(a):
    return a.detach().cpu().numpy() if hasattr(a,'detach')else np.asarray(a)
def byte_equal(a,b):
    a,b=array(a),array(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def compare_arrays(old,new):
    assert old.keys()==new.keys(),('Readout arrays changed',list(old),list(new));out={}
    for key in old:
        a,b=old[key],new[key];assert a.shape==b.shape and a.dtype==b.dtype,(key,a.shape,b.shape,a.dtype,b.dtype)
        assert np.isfinite(a).all() and np.isfinite(b).all(),key
        delta=float(np.max(abs(a.astype(np.float64)-b.astype(np.float64))))if a.size else 0.
        out[key]=dict(max_abs=delta,pass_=delta<TOL,byte_equal=byte_equal(a,b))
    return out
def validate_model(model):
    assert model['schema']=='scene065.teacher_gaussian.model_state.v1'
    f=model['field'];p=f['parameters'];assert tuple(p)==PARAMETERS
    n=len(p['xyz']);assert n>0 and array(p['xyz']).shape==(n,3)
    for key,value in p.items():assert len(value)==n and np.isfinite(array(value)).all(),key
    ids=array(f['row_ids']);assert ids.shape==(n,) and len(np.unique(ids))==n
    assert int(f['next_row_id'])>int(ids.max())
    assert len(f['parent_ids'])==n and len(f['source_candidate_ids'])==n
    changes={}
    for name,prior in zip(PARAMETERS,['prior_xyz','prior_log_scales','prior_raw_quaternion','prior_opacity_logits','prior_dc']):
        a,b=array(p[name]),array(f[prior]);assert a.shape==b.shape
        d=abs(a.astype(np.float64)-b.astype(np.float64));changes[name]=dict(max_abs=float(d.max()),mean_abs=float(d.mean()),changed_values=int(np.count_nonzero(d)))
    events=model['topology_events'];totals={k:sum(int(e.get(k,0))for e in events)for k in ('birth','clone','split','split_parents','prune')}
    return dict(point_count=n,next_row_id=int(f['next_row_id']),row_ids_SHA256=hashlib.sha256(ids.tobytes()).hexdigest(),
                unique_row_ids=True,parameter_names=list(PARAMETERS),parameter_changes_from_soft_prior=changes,
                learned_xyz_changed=changes['xyz']['changed_values']>0,all_five_groups_changed=all(x['changed_values']>0 for x in changes.values()),
                topology_event_count=len(events),topology_totals=totals,candidate_cursor=int(model['candidate_cursor']),
                used_candidate_count=len(model['used_candidate_ids']))

def verify_reused_baseline(path,runtime,native):
    source=Path(path).resolve();manifest_path=source/'manifest.json';old=read(manifest_path)
    assert old['complete']is True and old.get('source_native_parameters_unchanged',False)
    for key,current in [('native30k_contract',runtime['native30k_contract']),('native_adapter_code',runtime['native_adapter_code']),
                        ('native_checkpoint_sources',native['native_sources']),('native_engine_sources',native.get('native_engine_sources',{})),
                        ('native_source_metadata',native['source_metadata']),('native_rasterizer_sources',native.get('rasterizer_sources',{}))]:
        assert old[key]==current,('Reused baseline source binding differs',key)
    pinned(old['native30k_contract']);pinned(old['native_adapter_code'])
    spec=old['baseline_receipt'];receipt_path=pinned(spec)
    assert receipt_path.resolve()==(source/'native30k_baseline/receipt.json').resolve()
    receipt=read(receipt_path);assert receipt['complete']and receipt['iteration']==30000 and receipt['original_A9_present']
    assert receipt['source_owner21_alpha_is_not_semantic21_alpha']
    rows=receipt['rows'];assert [int(row['raw'])for row in rows]==runtime['frames']and len(rows)==20
    verified=[]
    for row in rows:
        for artifact in pins({key:row[key]for key in ['joint','image','alpha_images']}):
            target=pinned(artifact);assert target.resolve().is_relative_to(source)
            verified.append(artifact)
    return dict(source_asset=pin(manifest_path),source_baseline_receipt=spec,rows=rows,files_SHA_verified=len(verified),
                source_pins_equal=True,all_twenty_baseline_frames_verified=True,source_native_parameters_unchanged=True)

def preflight(run,reuse_native_baseline=None):
    import torch
    run=Path(run).resolve();stage=run/'train';receipt_path=stage/'receipt.json';receipt=read(receipt_path)
    assert receipt['complete']is True and receipt.get('error')is None and receipt.get('phase','train')=='train'
    receipt_bindings=receipt['bindings']
    config=read(bound(receipt_bindings,'configuration','config','training_configuration'))
    step=int(config['target_unique_updates']);assert step>0
    assert int(receipt.get('logical_step',receipt.get('new_scene_unique_training_updates',-1)))==step
    for spec in pins(receipt_bindings):pinned(spec)
    model_path=stage/'model_final.pt';payload=torch.load(model_path,map_location='cpu',weights_only=False)
    assert isinstance(payload,dict) and {'model','bindings','frozen_assets','logical_step'}<=payload.keys()
    assert payload['schema']=='scene065.teacher_gaussian.inference_asset.v1'
    assert 'optimizer'not in payload and 'optimizer_state'not in payload
    assert payload['logical_step']==step
    life_path=bound(payload['bindings'],'lifecycle_code','lifecycle_module')
    life=module(life_path,'_saved_teacher_asset_lifecycle')
    assert life.equal(payload['bindings'],receipt_bindings),'Model and final receipt bindings differ'
    bindings=payload['bindings']
    runtime_path=bound(bindings,'runtime_contract','runtime');runtime=read(runtime_path)
    config=read(bound(bindings,'configuration','config','training_configuration'))
    assert int(config['target_unique_updates'])==step
    target_spec=bindings.get('teacher_targets',bindings.get('target_bindings'))
    assert isinstance(target_spec,dict)
    targets=read(pinned(target_spec))
    background_source=targets['background_completion_manifest'];background_seed=targets['background_seed'];background_pool=targets['background_candidate_pool']
    for spec in [background_source,background_seed,background_pool]:pinned(spec)
    execution=read(bound(bindings,'execution_contract'))
    continuation_append_seed=execution.get('background_seed')
    if isinstance(continuation_append_seed,dict):pinned(continuation_append_seed)
    objective_path=bound(bindings,'objective_code','objective_module','objective')
    runtime_code=bound(bindings,'runtime_code','runtime_module')
    for spec in pins(runtime):pinned(spec)
    assert runtime['schema']in('scene065.teacher_gaussian.runtime.v1','scene065.teacher_gaussian_reconstruction.runtime.v1')
    assert len(runtime['frames'])==len(set(runtime['frames']))==20
    native=read(pinned(runtime['native30k_contract']));assert native['complete_unedited_native_30k']and native['iteration']==30000
    teacher_manifest=read(pinned(runtime['teacher_manifest']));assert teacher_manifest['independent_teacher_count']==3
    for teacher in teacher_manifest['rows']:
        pinned(teacher['raw_teacher']);pinned(teacher['fullRGB_native'])
    stats=validate_model(payload['model'])
    if 'frozen_assets'in bindings:assert life.equal(payload['frozen_assets'],bindings['frozen_assets'])
    if 'frozen_assets'in receipt:assert life.equal(payload['frozen_assets'],receipt['frozen_assets'])
    field_path=stage/'new_field_final.npz'
    if not field_path.is_file():field_path=stage/'field_final.npz'
    field=load(field_path);f=payload['model']['field'];pars=f['parameters']
    assert len(field['xyz'])==stats['point_count']
    for key in ('row_ids','parent_ids','source_candidate_ids'):
        assert byte_equal(field[key],f[key]),('Field NPZ lineage differs',key)
    assert byte_equal(field['xyz'],pars['xyz'])and byte_equal(field['coefficients'][:,0],pars['dc'])
    assert not field['coefficients'][:,1:].any()
    reference=stage/f'actual_step{step:06d}';reference_receipt=read(reference/'receipt.json')
    configured_readout=[int(x)for x in config['readout_frames']];assert len(configured_readout)==len(set(configured_readout))==16
    readout=[int(x['raw'])for x in reference_receipt['rows']]
    # Use what the bound driver actually saved. R6 should save all configured
    # frames; the explicit historical ten-frame set remains a valid control.
    assert len(readout)==len(set(readout))
    assert set(readout).issubset(configured_readout)
    assert set(readout)==set(configured_readout)or sorted(readout)==[106,107,108,109,110,181,182,183,184,185]
    references={raw:pin(reference/f'{raw:03}'/'joint.npz')for raw in readout}
    reuse=verify_reused_baseline(reuse_native_baseline,runtime,native)if reuse_native_baseline is not None else None
    return dict(run=run,stage=stage,step=step,receipt=receipt,receipt_path=receipt_path,payload=payload,bindings=bindings,
                runtime=runtime,runtime_path=runtime_path,runtime_code=runtime_code,objective_path=objective_path,
                lifecycle_path=life_path,life=life,config=config,native=native,teachers=teacher_manifest,
                model_pin=pin(model_path),checkpoint_pin=pin(stage/'checkpoint_final.pt'),field_pin=pin(field_path),
                field=field,model_stats=stats,configured_readout_frames=configured_readout,readout_frames=readout,reference_receipt=pin(reference/'receipt.json'),reference_files=references,baseline_reuse=reuse,
                background_source=background_source,background_seed=background_seed,background_pool=background_pool,continuation_append_seed=continuation_append_seed)

def save_render(folder,arrays):
    from PIL import Image
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    assert all(np.isfinite(x).all()for x in arrays.values())
    np.savez_compressed(folder/'joint.npz',**arrays)
    rgb=arrays['rgb'];assert rgb.shape[0]==3
    Image.fromarray(np.rint(np.clip(rgb.transpose(1,2,0),0,1)*255).astype(np.uint8)).save(folder/'joint.png')
    alphas={}
    for key,value in arrays.items():
        if 'alpha'in key and np.squeeze(value).ndim==2:
            p=folder/(key+'.png');Image.fromarray(np.rint(np.clip(np.squeeze(value),0,1)*255).astype(np.uint8)).save(p);alphas[key]=pin(p)
    return dict(joint=pin(folder/'joint.npz'),image=pin(folder/'joint.png'),alpha_images=alphas)

def main(args):
    started=time.perf_counter();info=preflight(args.run,args.reuse_native_baseline);step=info['step'];comparison_name=f'comparison_step{step:06d}.json'
    if args.cpu_preflight_only:
        print(json.dumps(dict(CPU_preflight_pass=True,GPU_started=False,model=info['model_pin'],model_stats=info['model_stats'],complete_checkpoint_deserialized=False,Adam_created=False),ensure_ascii=False),flush=True);return
    out=(args.output or info['run']/'asset').resolve();assert not out.exists(),'Preserve earlier asset evidence; choose a fresh --output';out.mkdir(parents=True)
    rt=None;error=None;rows=[];baseline=[];compares=[];raster_count=[0];base_before=0;restored_exact=False
    rc=info['runtime'];binds=info['bindings'];state=info['payload']['model'];life=info['life']
    manifest=dict(schema='scene065.full3D_teacher_gaussian.saved_asset.v1',complete=False,model_step=step,
        model_final=info['model_pin'],complete_checkpoint=dict(**info['checkpoint_pin'],deserialized=False),learned_field=info['field_pin'],
        train_receipt=pin(info['receipt_path']),training_bindings=binds,runtime_contract=pin(info['runtime_path']),
        runtime_code=pin(info['runtime_code']),objective_code=pin(info['objective_path']),lifecycle_code=pin(info['lifecycle_path']),
        driver_code=binds['driver_code'],execution_contract=binds.get('execution_contract'),
        reloader_code=pin(__file__),native_adapter_code=rc['native_adapter_code'],native30k_contract=rc['native30k_contract'],
        native_checkpoint_sources=info['native']['native_sources'],native_source_metadata=info['native']['source_metadata'],
        native_engine_sources=info['native'].get('native_engine_sources',{}),native_rasterizer_sources=info['native'].get('rasterizer_sources',{}),
        native_semantic_ownership=info['native'].get('ownership_metadata'),context_retirement=rc['context_retirement'],
        retained_fragments=rc.get('retained_fragments'),
        background_source_manifest=info['background_source'],background_seed=info['background_seed'],background_candidate_pool=info['background_pool'],
        continuation_append_seed=info['continuation_append_seed'],
        background_permission_policy=dict(unknown_RGB_weight=0,geometry_scope='entire fixed background geometry scope, including RGB-unknown cells',unknown_protection_RGB_weight=0,evaluation_denominators='fixed full scope; never reduced to RGB-known support'),
        teacher_seed=rc.get('teacher_seed',rc.get('new_field')),candidate_pool=rc['candidate_pool'],pose=rc['pose'],
        teacher_manifest=rc['teacher_manifest'],seed_manifest=rc.get('seed_manifest'),three_fullRGB_teachers=info['teachers']['rows'],
        model_statistics=info['model_stats'],frozen_assets_saved=info['payload']['frozen_assets'],frames=rc['frames'],
        reference_readout_receipt=info['reference_receipt'],configured_readout_frames=info['configured_readout_frames'],compared_final_readout_frames=info['readout_frames'],strict_readout_max_abs_tolerance=TOL,
        restore_api='TeacherGaussianRuntime.restore_model_state(payload[model], None)',
        all_five_parameters_and_topology_restored=True,inference_Adam_created=False,inference_optimizer_state_loaded=False,
        complete_checkpoint_deserialized=False,source_native_parameters_unchanged=False,training_or_inputs_modified=False,
        one_actual_native_raster_per_output=True,permanent_green_carve=False,full_RGB_teacher_initialization=True,
        alpha_meanings=dict(edited_actor_alpha='effective alpha of semantic21 A20 in edited joint native raster',
                           baseline_source_owner21_alpha='effective alpha of native source dynamic owner21; not semantic21',
                           baseline_native_static_alpha='effective original static-branch alpha',baseline_native_dynamic_alpha='effective original dynamic-branch alpha'),
        source_owner21_and_semantic21_alphas_compared=False,baseline_original_A9_present=True,
        baseline_reused=info['baseline_reuse']is not None,baseline_reuse=info['baseline_reuse'])
    try:
        import torch as t
        assert t.cuda.is_available();seed=int(info['config']['seed']);t.manual_seed(seed);t.cuda.manual_seed_all(seed)
        runtime=module(info['runtime_code'],'_reloaded_full_teacher_runtime')
        objective=module(info['objective_path'],'_reloaded_full_teacher_objective')
        rt=runtime.TeacherGaussianRuntime(info['runtime_path'],out/'runtime_load')
        assert rt.frames==rc['frames']
        # The training driver registers its verified fixed-cache pin after
        # runtime construction. Recreate that exact binding before equality.
        cache_spec=binds['cache_manifest'];pinned(cache_spec)
        rt.frozen_assets['cache_manifest']=cache_spec['sha256']
        assert life.equal(rt.frozen_assets,info['payload']['frozen_assets'])
        assert all(getattr(model,'optimizer',None)is None for model in rt.base.models.values())
        result=rt.restore_model_state(state,None);assert result is None
        rt.assert_fixed();rt.base.assert_fixed()
        assert tuple(rt.named_new_parameters())==PARAMETERS
        for name,parameter in rt.named_new_parameters().items():
            assert byte_equal(parameter,state['field']['parameters'][name]),('Parameter restore differs',name)
        assert byte_equal(rt.field.row_ids,state['field']['row_ids'])and len(rt.field.xyz)==info['model_stats']['point_count']
        assert life.equal(rt.capture_model_state(),state),'Full model/topology restore is not exact'
        # Saved display arrays are derived from the learned five groups, with no
        # requirement that geometry equal the initial seed.
        for name,value in [('xyz',rt.field.xyz),('scales',rt.field.scales),('rotation',rt.field.rotation),('opacity',rt.field.opacity[:,0])]:
            assert byte_equal(value,info['field'][name]),('Learned field/model mismatch',name)
        restored_exact=True;rt.field.eval()
        manifest['frozen_assets_reloaded']=dict(rt.frozen_assets);manifest['native_frozen_assets_reloaded']=dict(rt.base.frozen_assets)
        manifest['native_scene_code']=pin(inspect.getsourcefile(type(rt.base.scene)))
        native_raster=rt.raster
        def counted(*a,**kw):raster_count[0]+=1;return native_raster(*a,**kw)
        rt.raster=counted;base_before=rt.base.calls;render_before=rt.calls
        with t.no_grad():
            for raw in rt.frames:
                before=raster_count[0];arrays={k:v.detach().cpu().numpy()for k,v in rt.render(raw).items()};assert raster_count[0]-before==1
                artifacts=save_render(out/'readout'/f'{raw:03}',arrays);rows.append(dict(raw=raw,actual_native_raster_calls=1,**artifacts))
                if raw in info['reference_files']:
                    source=info['reference_files'][raw];old=load(pinned(source));terms=compare_arrays(old,arrays)
                    q=dict(raw=raw,source=source,arrays=terms,max_abs=max(v['max_abs']for v in terms.values()),pass_=all(v['pass_']for v in terms.values()));compares.append(q)
                    assert q['pass_'],('Reloaded final render differs',q)
                print(json.dumps(dict(stage='reloaded_full3D',raw=raw,point_count=len(rt.field.xyz))),flush=True)
                del arrays
            if info['baseline_reuse']is None:
                for raw in rt.frames:
                    before=rt.base.calls;arrays={k:v.detach().cpu().numpy()for k,v in rt.base.render(raw).items()};assert rt.base.calls-before==1
                    arrays['source_owner21_alpha']=arrays.pop('original_A20_alpha')
                    artifacts=save_render(out/'native30k_baseline'/f'{raw:03}',arrays);baseline.append(dict(raw=raw,iteration=30000,unedited=True,original_A9_present=True,actual_native_raster_calls=1,**artifacts))
                    print(json.dumps(dict(stage='unedited_native30k',raw=raw)),flush=True);del arrays
            else:
                for original_row in info['baseline_reuse']['rows']:
                    row=dict(original_row);row['source_actual_native_raster_calls']=row.get('actual_native_raster_calls',1)
                    row['actual_native_raster_calls']=0;row['reused']=True;baseline.append(row)
        rt.assert_fixed();rt.base.assert_fixed();assert life.equal(rt.capture_model_state(),state),'Inference mutated model/topology state'
        assert len(rows)==len(baseline)==raster_count[0]==rt.calls-render_before==20
        expected_baseline_calls=0 if info['baseline_reuse']is not None else 20
        assert rt.base.calls-base_before==expected_baseline_calls and rt.backward_calls==rt.base.backward_calls==0
        assert sorted(x['raw']for x in compares)==sorted(info['readout_frames'])
        write(out/comparison_name,dict(schema='scene065.full3D_asset_reload_comparison.v1',pass_=True,model_step=step,compared_frames=len(info['readout_frames']),configured_frames=info['configured_readout_frames'],actual_saved_reference_frames=info['readout_frames'],strict_max_abs_tolerance=TOL,max_abs=max(x['max_abs']for x in compares),rows=compares))
        write(out/'readout/receipt.json',dict(schema='scene065.full3D_asset_actual_readout.v1',complete=True,model_step=step,point_count=len(rt.field.xyz),actual_native_raster_calls=20,rows=rows))
        write(out/'native30k_baseline/receipt.json',dict(schema='scene065.unedited_native30k_asset_baseline.v1',complete=True,iteration=30000,original_A9_present=True,source_owner21_alpha_is_not_semantic21_alpha=True,actual_native_raster_calls=expected_baseline_calls,baseline_reused=info['baseline_reuse']is not None,source_baseline_receipt=None if info['baseline_reuse']is None else info['baseline_reuse']['source_baseline_receipt'],rows=baseline))
        manifest.update(complete=True,source_native_parameters_unchanged=True,model_state_unchanged_after_inference=True,
                        comparison=pin(out/comparison_name),readout_receipt=pin(out/'readout/receipt.json'),baseline_receipt=pin(out/'native30k_baseline/receipt.json'))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raise
    finally:
        manifest.update(error=error,restored_full_model_exact=restored_exact,saved_edited_frames=len(rows),saved_baseline_frames=len(baseline),
            compared_frames=len(compares),compared_arrays_pass=bool(len(compares)==len(info['readout_frames']) and all(q['pass_']for q in compares)),
            edited_actual_native_raster_calls=raster_count[0],baseline_actual_native_render_calls=0 if rt is None else rt.base.calls-base_before,
            baseline_raster_calls=0 if rt is None else rt.base.calls-base_before,
            native_backward_calls=0 if rt is None else rt.backward_calls+rt.base.backward_calls,seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'manifest.json',manifest)
        print(json.dumps(dict(asset=out.as_posix(),complete=manifest['complete'],error=error,model_step=step,point_count=info['model_stats']['point_count'],
                              edited_frames=len(rows),baseline_frames=len(baseline),compared_frames=len(compares),inference_Adam_created=False,optimizer_state_loaded=False),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,help='Fresh output directory, default RUN/asset');p.add_argument('--reuse-native-baseline',type=Path,help='Verified completed earlier asset directory with the same frozen native30k sources');p.add_argument('--cpu-preflight-only',action='store_true');main(p.parse_args())
