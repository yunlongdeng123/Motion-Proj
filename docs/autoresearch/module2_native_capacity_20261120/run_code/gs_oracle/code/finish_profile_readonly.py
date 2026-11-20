"""Complete the existing profile's public1920 reload, with zero new updates/B.

The original profile/source/failure remain immutable. SceneManifest's base path
is explicitly asset/persistent, rather than the nonexistent manifest.parent.
"""
from pathlib import Path
import copy,gc,hashlib,json,sys,time,traceback
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
P=B/'gs_oracle/full_objective_profile_r1';OUT=B/'gs_oracle/full_objective_profile_completion_r1'
FRAMES=(86,110);OWNER='native_capacity_profile'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    failed=read(P/'failed.json');ledger=copy.deepcopy(failed['physical_ledger'])
    if 'SceneManifest'not in failed['error']or ledger['native_forwards']!=44 or ledger['native_per_view_backwards']!=16 or ledger['engineering_parameter_updates']!=7:
        raise RuntimeError('unexpected parent failure state; do not guess resume')
    write(OUT/'authorization.json',dict(schema='native_capacity_profile_readonly_completion.v1',GPU_authorized=True,
        root_authorization='Necessary mechanical publicloader path repair within original64F/16B/8update profile contract',
        actual_scope='read-only field/ci requery and public1920 reload, no new accepted step or backward',
        inherited_profile=str(P),inherited_nativeF=44,max_aggregate_nativeF=64,additional_B=0,additional_updates=0,GPU_keep_on=True))
    lease=OUT/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(OUT/'authorization.json'),phase='readonly_completion'))
    start=time.monotonic();ctx=None;rows=[];authority={}
    try:
        import torch
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import make_context,field_and_reader
        from gs_color_oracle import BoundedGSColors
        # The package native import requires the context factory's import paths.
        ctx=make_context(lease,OWNER)
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        model,reader,source=field_and_reader(ctx.base_rgb,'cuda')
        field,field_meta=type(model).load(P/'world_field.pt',device='cuda')
        oracle=BoundedGSColors.load(P/'oracle_colors.pt',device='cuda')
        with torch.no_grad():colors={'field':reader.colors(field).detach(),'oracle':oracle().detach()}
        saved_field=torch.load(P/'field_final_checkpoint.pt',map_location='cpu',weights_only=False)
        authority['field_state_all_exact']=all(torch.equal(v.detach().cpu(),saved_field['state'][k])for k,v in field.state_dict().items())
        saved_box=torch.load(P/'box_final_checkpoint.pt',map_location='cpu',weights_only=False)
        authority['oracle_state_exact']=np.array_equal(oracle.rgb.detach().cpu().numpy(),saved_box['solver']['x'].astype(np.float32))
        for name in colors:
            with np.load(P/(name+'_asset')/'road.npz',allow_pickle=False)as z:stored=z['reference_srgb'][ctx.active_world_rows]
            authority[name+'_authority_to_saved_GS_RGB_max']=float(np.max(abs(colors[name].cpu().numpy()-stored)))
            if authority[name+'_authority_to_saved_GS_RGB_max']>1e-6:raise RuntimeError('field/ci authority differs from already saved GS')
        if not authority['field_state_all_exact']or not authority['oracle_state_exact']:raise RuntimeError('saved authority state differs')
        def sync():torch.cuda.synchronize()
        def call(label,raw,fn):
            if ledger['native_forwards']>=64:raise RuntimeError('aggregate64-forward cap reached')
            ctx.require_lease();sync();t=time.perf_counter();value=fn();sync()
            ledger['native_forwards']+=1;ledger['rows'].append(dict(kind='native_forward',raw=raw,label=label,seconds=time.perf_counter()-t,readonly_completion=True))
            write(OUT/'aggregate_native_counter.json',ledger);return value
        for name,color in colors.items():
            asset=P/(name+'_asset')
            with torch.no_grad():expected={r:call(name+'_expected_recomputed',r,lambda r=r:ctx.render_road(r,color))for r in FRAMES}
            scene,m,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
            data=dict(m.data);data['render_wh']=[1920,1280]
            headers=SceneManifest(data,asset/'persistent').camera_headers();cameras={}
            def camera(raw):
                if raw not in cameras:cameras[raw]=make_native_camera(headers[raw])
                return cameras[raw]
            scene.camera_provider=camera;provider=next(v for v in scene._providers if v.kind=='road')
            with torch.no_grad():
                for raw in FRAMES:
                    cam=camera(raw)
                    actual=call(name+'_public_1920_clone',raw,lambda:scene._bindings().raster(cam,scene.pipe,0,provider.rows(raw,cam,scene.pipe),scene._background))
                    rgb=float((expected[raw]['rgb']-actual['rgb']).abs().max());alpha=float((expected[raw]['alpha']-actual['alpha']).abs().max())
                    row=dict(asset=name,raw=raw,RGB_max=rgb,alpha_max=alpha,RGB8_max=int((torch.round(expected[raw]['rgb']*255)-torch.round(actual['rgb']*255)).abs().max()),
                        resolution=[1920,1280],public_proof_loaded=True,camera_policy='saved own manifest+explicit render_wh1920, base=asset/persistent; no file mutation')
                    rows.append(row)
                    if rgb>1e-6 or alpha!=0:raise RuntimeError('actual native1920 public reload failed')
            scene.assert_persistent_unchanged();del expected,provider,scene;gc.collect();torch.cuda.empty_cache()
        Krows=[]
        with torch.no_grad():
            for raw in FRAMES:
                parent=call('parent_joint_K',raw,lambda r=raw:ctx.render_joint(r))
                mask=torch.as_tensor(ctx.K_masks[raw],device=ctx.base_rgb.device)
                for name,color in colors.items():
                    actual=call(name+'_joint_K',raw,lambda r=raw,c=color:ctx.render_joint(r,c))
                    error=float((actual['rgb']-parent['rgb']).abs().amax(0)[mask].max());alpha=float((actual['alpha']-parent['alpha']).abs().max())
                    Krows.append(dict(asset=name,raw=raw,K_native_pixels=int(mask.sum()),joint_K_RGB_max=error,joint_alpha_max=alpha))
                    if error!=0 or alpha!=0:raise RuntimeError('profile joint K/alpha changed')
        ctx.assert_frozen()
        old_bindings=read(P/'context_ready.json')['source_binding']
        if any(sha(path)!=h for path,h in old_bindings.items()):raise RuntimeError('original executed source/input changed')
        result=dict(schema='native_capacity_verified_profile_completion.v1',completed=True,formal_updates=0,
            original_profile=str(P),original_mechanical_failure_preserved=str(P/'failed.json'),original_executed_code=str(B/'gs_oracle/code/profile_full_objective.py'),
            readonly_completion_code=str(Path(__file__)),readonly_completion_code_sha256=sha(__file__),
            aggregate_physical_ledger=ledger,completion_nativeF=ledger['native_forwards']-44,completion_nativeB=0,completion_updates=0,
            field_full_restore_exact=True,field_nextstep_replay_passed_tolerance_1e_minus6=True,box_full_restore_exact=True,box_nextstep_replay_passed_tolerance_1e_minus6=True,
            replay_evidence='Original executed hard assertions passed before persistent assets were saved; exact replay error scalars were not persisted before later path failure. No replay is fabricated or rerun here.',
            authority_requery=authority,public_1920_reload=rows,joint_K=Krows,field_steps=read(P/'field_progress.json'),box_steps=read(P/'box_progress.json'),
            loss_snapshot=str(P/'loss_snapshot.json'),profile_seconds_original_failure=failed['seconds'],readonly_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            combined_profile_seconds=failed['seconds']+time.monotonic()-start,all_engineering_states_discard_for_formal=True,
            production_assets_written=False,GPU_keep_on=True)
        write(OUT/'receipt.json',result);print('COMPLETED',json.dumps({k:v for k,v in result.items()if k not in('aggregate_physical_ledger','field_steps','box_steps')}),flush=True)
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),aggregate_physical_ledger=ledger,completion_updates=0,completion_B=0));raise
    finally:
        if ctx is not None:ctx.close()
        state=read(lease);state.update(active=False,ended_at=time.time());write(lease,state)


if __name__=='__main__':main()
