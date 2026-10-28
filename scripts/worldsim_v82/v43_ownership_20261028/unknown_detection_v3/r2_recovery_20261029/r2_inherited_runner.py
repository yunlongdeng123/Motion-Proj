"""R2 inheritance queue. No GPU work unless a phase is explicitly selected.

Phase order: replay -> contact -> seeds -> extend. An exact saved-anchor replay
receipt is mandatory before new-frame rendering; failed/missing evidence blocks.
No empty-seed fallback, adjacent-frame substitute, or edited RGB training label.
"""
from pathlib import Path
import argparse, json, subprocess, sys, time
import numpy as np

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
NEW = ROOT / 'unknown_detection_v3'
CODE = NEW / 'code'
CTX = '14705303724557273004_3105_000_3125_000'
ANCHORS = [98, 113, 114]
GPU_PY = '/root/autodl-tmp/envs/drivingeditor/bin/python'
# The contact script explicitly disables CUDA. This installed environment has
# scipy; dggt_data_cpu lacks it on the current CPU host.
CPU_PY = GPU_PY
RAWS = list(range(77,126))


def preflight(work):
    """Pure CPU inventory; readiness does not claim CUDA or scientific validity."""
    records=[]
    for name,root,raws,suffix in [
        ('body_probe',NEW/'body_probe_r1',RAWS,'.npz'),
        ('shadow_strict',NEW/'shadow_evidence_r2',RAWS,'.npz'),
        ('shadow_photo',NEW/'shadow_evidence_r3',RAWS,'.npz'),
        ('original_contact',NEW/'contact_effect_evidence_r1',ANCHORS,'.npz'),
        ('regenerated_contact',work/'contact49',RAWS,'.npz'),
        ('regenerated_existing',work/'native_existing49/evidence',RAWS,'.npz')]:
        missing=[raw for raw in raws if not (root/(f'{raw:03d}'+suffix)).is_file()]
        records.append(dict(input=name,path=str(root),expected=len(raws),present=len(raws)-len(missing),missing_raws=missing))
    anchors=[]
    for raw in ANCHORS:
        paths=[NEW/'bounded_white_inputs'/f'{raw:03d}/diagnostic_gate.npz',
               NEW/'recall_ceiling_r1'/f'{raw:03d}/existing/contributors.npz',
               NEW/'recall_ceiling_r1'/f'{raw:03d}/existing/maps.npz',
               NEW/'recall_ceiling_r1'/f'{raw:03d}/domains.npz',
               NEW/'bounded_native_r1'/f'{raw:03d}/evidence.npz']
        anchors.append(dict(raw=raw,missing=[str(p) for p in paths if not p.is_file()]))
    gates={}
    for name in ['anchor_replay_validation','contact_anchor_validation','seed_anchor_validation','seed_full49_validation']:
        path=work/(name+'.json');gates[name]=json.loads(path.read_text()).get('passed',False) if path.is_file() else False
    common_paths=[CODE/'render_bounded.py',CODE/'render_detection.py',CODE/'build_contact_effect_evidence.py',
                  CODE/'probe_detection.py',ROOT/'datasets'/CTX/'manifest.json',
                  ROOT/'actor_campaign_v1'/CTX/'formal10k/checkpoint010000.pth']
    common_missing=[str(p) for p in common_paths if not p.is_file()]
    result=dict(schema='r2-inherited-preflight-v1',inputs=records,anchors=anchors,gates=gates,common_missing=common_missing,
        replay_inputs_ready=not common_missing and all(not r['missing'] for r in anchors),
        seeds_inputs_ready=all(not r['missing_raws'] for r in records[:5]) and gates['anchor_replay_validation'] and gates['contact_anchor_validation'],
        extend_inputs_ready=all(not r['missing_raws'] for r in records) and all(gates.values()),
        GPU_checked=False,GPU_calls=0,training_updates=0,
        no_empty_seed_or_adjacent_frame_fallback=True)
    write(work/'preflight.json',result);print(json.dumps(result,indent=2));return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run(argv):
    subprocess.run([str(x) for x in argv], check=True)


def required(path):
    if not path.exists():
        raise FileNotFoundError('Required original dependency missing; no fallback: ' + str(path))
    return path


def passed(work):
    gate = json.loads(required(work / 'anchor_replay_validation.json').read_text())
    if not gate.get('passed'):
        raise RuntimeError('R2 anchor replay has not passed; cannot extend')


def replay(work):
    # This is the original unchanged GPU algorithm, including its exact 8-pass
    # limit. R2 was this GPU result plus a CPU-local display-scope restriction.
    output = work / 'anchor_gpu_replay'
    run([GPU_PY, required(CODE / 'render_bounded.py'), '--inputs',
         required(NEW / 'bounded_white_inputs'), '--source',
         required(NEW / 'recall_ceiling_r1'), '--output', output])
    records = []
    for raw in ANCHORS:
        key = f'{raw:03d}'
        e = dict(np.load(output / key / 'evidence.npz'))
        ref = np.load(required(NEW / 'bounded_native_r1' / key / 'evidence.npz'))
        gate = np.load(required(NEW / 'bounded_white_inputs' / key / 'diagnostic_gate.npz'))
        scope, baseline = gate['unknown_display_gate'], gate['source_native_rgb']
        white = baseline.copy()
        white[:, scope] = (e['gaussian_rgb'] + (1-e['alpha'])[None])[:, scope]
        reference_white = baseline.copy()
        reference_white[:, scope] = (ref['gaussian_rgb'] + (1-ref['alpha'])[None])[:, scope]
        checks = {name: bool(np.array_equal(e[name], ref[name])) for name in
                  ['existing_removed_ids', 'new_unknown_ids', 'eligible_ids', 'allowed', 'query']}
        errors = {name: float(np.max(np.abs(e[name]-ref[name]))) for name in ['alpha', 'gaussian_rgb']}
        errors['localized_R2_white'] = float(np.max(np.abs(white-reference_white)))
        e.update(white_rgb=white, display_scope=scope, baseline_native_rgb=baseline)
        np.savez_compressed(output / key / 'localized_r2_evidence.npz', **e)
        from PIL import Image
        Image.fromarray(np.round(np.clip(white.transpose(1,2,0),0,1)*255).astype('uint8')).save(output / key / 'localized_r2_white.png')
        records.append(dict(raw=raw, exact_discrete_checks=checks, float_max_error=errors,
                            passed=all(checks.values()) and max(errors.values()) <= 1e-5))
    result = dict(passed=all(x['passed'] for x in records), records=records,
                  reference='original bounded_native_r1 G/alpha + original R2 local composition',
                  no_seed_replacement=True, training_updates=0)
    write(work / 'anchor_replay_validation.json', result)
    assert result['passed'], 'Anchor replay changed: stop before all-frame extension'


def contact(work):
    # Missing 46 contact-effect frames are recomputed with the original
    # independent raw-data algorithm and strict-r2 plane/source evidence.
    run([CPU_PY, required(CODE / 'build_contact_effect_evidence.py'), '--root', ROOT,
         '--strict', required(NEW / 'shadow_evidence_r2'), '--output', work / 'contact49',
         '--raw', *range(77,126)])
    checks=[]
    for raw in ANCHORS:
        original=np.load(required(NEW/'contact_effect_evidence_r1'/f'{raw:03d}.npz'))
        new=np.load(work/'contact49'/f'{raw:03d}.npz')
        checks.append(dict(raw=raw, pixels_equal=bool(np.array_equal(
            original['contact_ground_photo_effect'],new['contact_ground_photo_effect']))))
    write(work/'contact_anchor_validation.json',dict(passed=all(x['pixels_equal'] for x in checks),records=checks))
    assert all(x['pixels_equal'] for x in checks), 'Original contact evidence changed'


def seeds(work):
    passed(work)
    assert json.loads(required(work/'contact_anchor_validation.json').read_text())['passed']
    # Test the exact original photo classifier on the 3 anchors BEFORE spending
    # work on 49 frames; retain all49 body history in both calls.
    common=[GPU_PY,required(CODE/'render_detection.py'),'--shadow',required(NEW/'shadow_evidence_r3'),
            '--body-probe',required(NEW/'body_probe_r1'),'--contact',required(work/'contact49'),'--profile','photo']
    def check(source):
        records=[]
        for raw in ANCHORS:
            e=np.load(source/'evidence'/f'{raw:03d}.npz');ids=e['ids'][e['static_role']>=2]
            old=np.load(required(NEW/'recall_ceiling_r1'/f'{raw:03d}/existing/contributors.npz'))['removed_ids']
            records.append(dict(raw=raw,identical=bool(np.array_equal(ids,old)),rows=len(ids)))
        return dict(passed=all(x['identical'] for x in records),records=records)
    run(common+['--output',work/'native_existing_anchor','--raws',','.join(map(str,ANCHORS))])
    gate=check(work/'native_existing_anchor');write(work/'seed_anchor_validation.json',gate)
    assert gate['passed'], 'Anchor seeds disagree with R2 ancestry; all49 did not start'
    run(common+['--output',work/'native_existing49','--raws','all'])
    full=check(work/'native_existing49')
    full['missing_raws']=[r for r in RAWS if not (work/'native_existing49/evidence'/f'{r:03d}.npz').is_file()]
    full['passed'] &= not full['missing_raws'];write(work/'seed_full49_validation.json',full)
    assert full['passed'], 'Full49 original seed delivery is incomplete or changed anchors'


def extend(work):
    passed(work)
    assert json.loads(required(work/'seed_anchor_validation.json').read_text())['passed']
    assert json.loads(required(work/'seed_full49_validation.json').read_text())['passed']
    sys.path.insert(0,str(CODE));sys.path.insert(0,str(Path(__file__).resolve().parent))
    from probe_detection import load_native, kwargs, attribution, authorized, save, torch, cpu, exclusive, helpers
    from scope import build_scope, eligible_from_screen
    import gc
    from PIL import Image
    authorized();output=work/'r2_inherited49';output.mkdir(parents=True,exist_ok=False);records=[];started=time.monotonic()
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz)
        assert sorted(f['raw'] for f in frames)==RAWS, 'Unexpected original frame manifest'
        static_checkpoint=str(NEW.parent.parent/'persistent_world_20261027/native_scene_training_v1'/CTX/'formal30k')
        actor_checkpoint=str(ROOT/'actor_campaign_v1'/CTX/'formal10k/checkpoint010000.pth')
        namespace=static_checkpoint+'@30000:static'
        # Keep original camera index while evaluating anchors first. New-frame
        # rendering is forbidden until all three inherited anchors pass.
        order=sorted(enumerate(frames),key=lambda x:(x[1]['raw'] not in ANCHORS,x[0]))
        anchor_checks=[]
        for i,f in order:
            authorized();raw=f['raw'];key=f'{raw:03d}';dest=output/key;dest.mkdir()
            if raw not in ANCHORS:
                assert len(anchor_checks)==3 and all(x['passed'] for x in anchor_checks), 'Extension anchors not passed; no new frame rendering'
            write(output/'status.json',dict(phase='rendering',raw=raw,completed=len(records),training_updates=0))
            ob=np.load(ds/(key+'.npz'));body=ob['body'].astype(bool)
            source=np.load(required(work/'native_existing49/evidence'/(key+'.npz')))
            old_ids=source['ids'][source['static_role']>=2]
            if raw in ANCHORS:
                gate=np.load(required(NEW/'bounded_white_inputs'/key/'diagnostic_gate.npz'))
                allowed,contact_mask,display=gate['allowed'],gate['user_contact_query'],gate['unknown_display_gate']
            else:
                scope=build_scope(body,context=CTX,raw=raw,body_source=str(ds/(key+'.npz')))
                assert scope['status']=='ok', 'Missing source mask: stop, no borrowed frame'
                allowed,contact_mask,display=scope['allowed'],scope['user_contact_query'],scope['display_scope']
            cam,st,dy=kwargs(n,actor,f,i)
            with torch.no_grad():
                env=cpu(n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1))
                fullout=helpers.raster(cam,n.args,n.static.active_sh_degree,helpers.merge_rows(st,dy),n.bg)
                original=cpu(fullout['rgb'])+(1-cpu(fullout['alpha']))*env
                del fullout
                existing_kw={**st,'opacities':st['opacities'].clone()}
                existing_kw['opacities'][torch.from_numpy(old_ids.astype('int64')).cuda()]=0
                existing_out=helpers.raster(cam,n.args,n.static.active_sh_degree,existing_kw,n.bg)
                baseline=cpu(existing_out['rgb'])+(1-cpu(existing_out['alpha']))*env
                h=torch.cat((st['means3D'],torch.ones_like(st['means3D'][:,:1])),1)
                ndc=h@cam.full_proj_transform;ndc=ndc[:,:3]/ndc[:,3:4]
                screen=cpu(torch.stack(((ndc[:,0]+1)*480/2-.5,(ndc[:,1]+1)*320/2-.5),1))
                z=cpu((h@cam.world_view_transform)[:,2]);eligible=eligible_from_screen(screen,z,allowed)
            del existing_out
            clipped=((baseline<0).any(0)|(baseline>1).any(0))&allowed
            query=(body|contact_mask|clipped)&allowed
            maps=np.stack([query,allowed,~allowed,np.ones_like(allowed)]).astype('float32')
            removed=np.zeros(ns,bool);removed[old_ids]=True;initial=removed.copy();history=[];converged=False
            for step in range(8):
                kw={**st,'opacities':st['opacities'].clone()};kw['opacities'][torch.from_numpy(removed).cuda()]=0
                out,w,closure=attribution(n,cam,kw,maps);w=cpu(w);alpha=cpu(out['alpha'][0])
                g=cpu(out['rgb'])-(1-alpha)[None]*cpu(n.bg).reshape(3,1,1)
                add=(w[:,0]>1e-7)&eligible&~removed
                history.append(dict(step=step,new_rows=int(add.sum()),closure=closure))
                if not add.any():converged=True;break
                if step==7:break
                removed|=add;del out;gc.collect()
            white=baseline.copy();white[:,display]=(g+(1-alpha)[None])[:,display]
            native_env=g+(1-alpha)[None]*env
            native_local=baseline.copy();native_local[:,display]=native_env[:,display]
            new=removed&~initial
            assert not (new&~eligible).any()
            assert np.array_equal(white[:,~display],baseline[:,~display])
            for name,rgb in [('white',white),('existing',baseline),('original',original),('gaussian',g),
                             ('native',native_local),('native_env',native_env),('ungated_white',g+(1-alpha)[None])]:
                save(dest/(name+'.png'),torch.from_numpy(rgb))
            Image.fromarray(np.round(np.clip(alpha,0,1)*255).astype('uint8')).save(dest/'alpha.png')
            Image.fromarray(np.asarray(ob['rgb'],dtype='uint8')).save(dest/'observed.png')
            remaining=(w[:,0]>1e-7)&~removed
            residual_eligible=remaining&eligible
            final_ids=np.flatnonzero(removed).astype('int32')
            np.savez_compressed(dest/'evidence.npz',existing_removed_ids=old_ids,
                new_unknown_ids=np.flatnonzero(new).astype('int32'),final_removed_ids=final_ids,
                eligible_ids=np.flatnonzero(eligible).astype('int32'),screen_centers=screen[new],allowed=allowed,
                query=query,display_scope=display,alpha=alpha,gaussian_rgb=g,white_rgb=white,
                baseline_native_rgb=baseline,environment_rgb=env,original_rgb=original,native_rgb=native_local,
                native_env_rgb=native_env,body=body,contact=contact_mask,all_removed_screen_centers=screen[removed],
                starting_removed_ids=old_ids,residual_ids=np.flatnonzero(remaining).astype('int32'),residual_mass=w[remaining,0],
                residual_eligible_ids=np.flatnonzero(residual_eligible).astype('int32'),
                inherited_outside_ids=np.flatnonzero(initial&~eligible).astype('int32'),
                row_namespace=np.array(namespace),checkpoint_ref=np.array(actor_checkpoint),
                context=np.array(CTX),raw=np.array(raw),camera_index=np.array(i),model_time=np.array(f['model_time']))
            rec=dict(context=CTX,raw=raw,frame_id=f.get('frame_id',n.records[i]['img_name']),camera_index=i,
                model_time=f['model_time'],method='inherited_R2',status='ok' if converged else 'failed_nonconverged',
                row_namespace=namespace,static_checkpoint_ref=static_checkpoint,checkpoint_ref=actor_checkpoint,
                method_parameters=dict(mass_epsilon=1e-7,evaluation_pass_limit=8,maximum_selection_updates=7),
                existing_rows=int(initial.sum()),final_removed_rows=int(removed.sum()),static_rows=ns,
                inherited_outside_center_rows=int((initial&~eligible).sum()),new_rows=int(new.sum()),
                new_outside_center_rows=0,converged=converged,history=history,
                initialization='same-frame regenerated original existing role>=2',
                query='body|manual_contact|same-frame-existing-float-out-of-range clipped to original blue',
                outside_composite='same-frame existing; not actor-omitted',labels_for_training=False,
                query_pixels=int(query.sum()),display_pixels=int(display.sum()),
                after_attribution_recomputed=True,final_after_recomputed=True,
                final_after_summary=dict(residual_query_rows=int(remaining.sum()),
                    residual_eligible_rows=int(residual_eligible.sum()),residual_eligible_mass=float(w[residual_eligible,0].sum()),
                    residual_outside_rows=int((remaining&~eligible).sum()),residual_outside_mass=float(w[remaining&~eligible,0].sum()),
                    query_alpha_mean=float(alpha[query].mean()),contact_alpha_mean=float(alpha[contact_mask].mean()),closure=closure),
                outside_output_max_difference=0.,actual_rasterizer_rerendered=True,
                white_equation='G_after+(1-alpha_after)*white in original local display scope; existing elsewhere',
                independent_frame_reset=True,init_mode='same_frame_existing',training_updates=0,human_verdict=None)
            if raw in ANCHORS:
                ref=np.load(required(work/'anchor_gpu_replay'/key/'localized_r2_evidence.npz'))
                rec['anchor_ids_identical']=bool(np.array_equal(np.flatnonzero(new),ref['new_unknown_ids']))
                rec['anchor_white_max_error']=float(np.max(np.abs(white-ref['white_rgb'])))
                check=dict(raw=raw,passed=rec['anchor_ids_identical'] and rec['anchor_white_max_error']<=1e-5 and converged)
                anchor_checks.append(check)
                write(work/'extend_anchor_validation.json',dict(passed=len(anchor_checks)==3 and all(x['passed'] for x in anchor_checks),records=anchor_checks))
                if not check['passed']:rec['status']='failed_anchor_regression'
            write(dest/'receipt.json',rec);records.append(rec)
            if rec['status']!='ok':
                write(output/'status.json',dict(phase=rec['status'],raw=raw,completed=len(records),training_updates=0))
                write(output/'summary.json',dict(status='failed',records=records,training_updates=0))
                raise RuntimeError('Inherited R2 failed at raw '+str(raw)+': '+rec['status'])
            del out,w,st,dy;gc.collect()
        assert n.unchanged()
    write(output/'summary.json',dict(status='complete',records=sorted(records,key=lambda x:x['raw']),frames=len(records),training_updates=0,
          anchors_evaluated_first=True,wall_seconds=time.monotonic()-started,cuda_peak_reserved=torch.cuda.max_memory_reserved(),
          inherited_seed_outside_blue_is_legacy_not_new=True,scope='scene1 only; no unverified cross-scene transfer'))
    write(output/'status.json',dict(phase='complete',completed=len(records),training_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--phase',choices=['replay','contact','seeds','extend']);choice.add_argument('--preflight',action='store_true')
    p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    if a.preflight:preflight(a.work)
    else:
        try:
            write(a.work/'queue_status.json',dict(phase=a.phase,status='running',training_updates=0))
            {'replay':replay,'contact':contact,'seeds':seeds,'extend':extend}[a.phase](a.work)
            write(a.work/'queue_status.json',dict(phase=a.phase,status='complete',training_updates=0))
        except Exception as error:
            write(a.work/'queue_status.json',dict(phase=a.phase,status='failed',error_type=type(error).__name__,
                error=str(error),empty_seed_fallback_used=False,training_updates=0))
            raise
