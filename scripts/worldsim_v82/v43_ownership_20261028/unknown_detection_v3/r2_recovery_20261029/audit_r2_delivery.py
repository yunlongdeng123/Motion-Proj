"""CPU-only independent saved-output audit and lossless light delivery packer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,json,hashlib,shutil,tarfile,time
import numpy as np
from PIL import Image

CTX='14705303724557273004_3105_000_3125_000'
RAWS=list(range(77,126));ANCHORS=[98,113,114]
PNGS=['white','existing','original','gaussian','native','native_env','ungated_white','alpha','observed']
KEYS=['existing_removed_ids','new_unknown_ids','final_removed_ids','eligible_ids','screen_centers',
      'all_removed_screen_centers','allowed','query','display_scope','body','contact',
      'starting_removed_ids','residual_ids','residual_mass','residual_eligible_ids','inherited_outside_ids',
      'row_namespace','checkpoint_ref','context','raw','camera_index','model_time']
GATES=['anchor_replay_validation','contact_anchor_validation','seed_anchor_validation','seed_full49_validation','extend_anchor_validation']

def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def display(rgb):return np.round(np.clip(rgb.transpose(1,2,0),0,1)*255).astype('uint8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.monotonic();base=a.run/'r2_inherited49';root=a.run.parents[1];campaign=root.parent
    a.output.mkdir(parents=True,exist_ok=False)
    status=json.loads((base/'status.json').read_text());assert status['phase']=='complete' and status['completed']==49
    summary=json.loads((base/'summary.json').read_text());assert summary['status']=='complete' and summary['frames']==49
    dataset=campaign/'datasets'/CTX;manifest=json.loads((dataset/'manifest.json').read_text())
    frames={f['raw']:(i,f) for i,f in enumerate(manifest['frames'])};assert sorted(frames)==RAWS
    namespace=str(campaign.parent/'persistent_world_20261027/native_scene_training_v1'/CTX/'formal30k')+'@30000:static'
    actor=str(campaign/'actor_campaign_v1'/CTX/'formal10k/checkpoint010000.pth')
    gates={name:json.loads((a.run/(name+'.json')).read_text()) for name in GATES}
    assert all(g['passed'] for g in gates.values())
    reports=[];compact=[];all_ok=True
    for raw in RAWS:
        src=base/f'{raw:03d}';rec=json.loads((src/'receipt.json').read_text());ep=src/'evidence.npz'
        with np.load(ep) as f:e={k:f[k] for k in f.files}
        checks={};errors={};ns=rec['static_rows'];ix,frame=frames[raw]
        checks['frame_identity']=(rec['context']==CTX and rec['raw']==raw and rec['camera_index']==ix and
            float(rec['model_time'])==float(frame['model_time']) and str(e['row_namespace'])==namespace and
            str(e['checkpoint_ref'])==actor and int(e['raw'])==raw and int(e['camera_index'])==ix and
            float(e['model_time'])==float(frame['model_time']))
        checks['converged_no_training']=rec['status']=='ok' and rec['converged'] and rec['training_updates']==0
        checks['same_R2_parameters']=rec['method_parameters']==dict(mass_epsilon=1e-7,evaluation_pass_limit=8,maximum_selection_updates=7)
        checks['final_attribution_recomputed']=rec['after_attribution_recomputed'] and rec['final_after_recomputed']
        ids_names=['existing_removed_ids','new_unknown_ids','final_removed_ids','eligible_ids','residual_ids','residual_eligible_ids','inherited_outside_ids']
        checks['IDs_unique_in_static_namespace']=all(np.array_equal(e[k],np.unique(e[k])) and
            ((e[k]>=0)&(e[k]<ns)).all() for k in ids_names)
        legacy,new,final,eligible=[e[k] for k in ['existing_removed_ids','new_unknown_ids','final_removed_ids','eligible_ids']]
        checks['final_ID_union']=np.array_equal(np.union1d(legacy,new),final) and not np.intersect1d(legacy,new).size
        seeds=np.load(a.run/'native_existing49/evidence'/f'{raw:03d}.npz')
        checks['same_frame_existing_seed']=np.array_equal(legacy,seeds['ids'][seeds['static_role']>=2]) and np.array_equal(legacy,e['starting_removed_ids'])
        checks['new_IDs_eligible']=np.isin(new,eligible).all()
        xy=np.rint(e['screen_centers']).astype('int64');h,w=e['allowed'].shape
        on=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        checks['new_centers_in_blue']=bool(on.all() and e['allowed'][xy[:,1],xy[:,0]].all())
        checks['legacy_outside_separate']=np.array_equal(np.setdiff1d(legacy,eligible),e['inherited_outside_ids']) and len(e['inherited_outside_ids'])==rec['inherited_outside_center_rows']
        checks['residuals_not_removed']=not np.intersect1d(e['residual_ids'],final).size
        checks['residual_eligible_empty']=not len(e['residual_eligible_ids']) and not len(np.intersect1d(e['residual_ids'],eligible))
        checks['residual_mass_valid']=len(e['residual_ids'])==len(e['residual_mass']) and np.isfinite(e['residual_mass']).all() and (e['residual_mass']>1e-7).all()
        checks['final_summary_closure']=rec['final_after_summary']['closure']<3e-4 and rec['final_after_summary']['residual_eligible_rows']==0 and rec['final_after_summary']['residual_eligible_mass']==0
        floats=['alpha','gaussian_rgb','white_rgb','baseline_native_rgb','environment_rgb','original_rgb','native_rgb','native_env_rgb']
        checks['float_components_finite']=all(np.isfinite(e[k]).all() for k in floats)
        checks['alpha_range']=float(e['alpha'].min())>=-1e-6 and float(e['alpha'].max())<=1.000001
        scope=e['display_scope'];baseline=e['baseline_native_rgb'];g=e['gaussian_rgb'];alpha=e['alpha']
        predicted=baseline.copy();predicted[:,scope]=(g+(1-alpha)[None])[:,scope]
        errors['white_formula']=float(np.abs(predicted-e['white_rgb']).max())
        errors['outside_preserved']=float(np.abs(e['white_rgb'][:,~scope]-baseline[:,~scope]).max())
        errors['native_env_formula']=float(np.abs(g+(1-alpha)[None]*e['environment_rgb']-e['native_env_rgb']).max())
        checks['all_composition_exact']=all(v==0 for v in errors.values())
        checks['scopes_clipped']=not (scope&~e['allowed']).any() and not (e['query']&~e['allowed']).any()
        checks['scope_body_contact_query']=not ((e['body']|e['contact'])&e['allowed']&~e['query']).any()
        checks['final_ids_count']=len(final)==rec['final_removed_rows'] and len(new)==rec['new_rows'] and len(legacy)==rec['existing_rows']
        png_hashes={};png_arrays={}
        for name in PNGS:
            pp=src/(name+'.png')
            with Image.open(pp) as im:
                im.load();arr=np.asarray(im);assert im.size==(480,320)
            png_arrays[name]=arr;png_hashes[name]=digest(pp)
        pairs={'white':e['white_rgb'],'existing':baseline,'original':e['original_rgb'],'gaussian':g,
               'native':e['native_rgb'],'native_env':e['native_env_rgb'],'ungated_white':g+(1-alpha)[None]}
        checks['PNG_float_encoding_exact']=all(np.array_equal(png_arrays[k],display(v)) for k,v in pairs.items())
        checks['alpha_PNG_exact']=np.array_equal(png_arrays['alpha'],np.round(np.clip(alpha,0,1)*255).astype('uint8'))
        observed=np.load(dataset/f'{raw:03d}.npz');checks['observed_PNG_is_source']=np.array_equal(png_arrays['observed'],observed['rgb'].astype('uint8'))
        if raw in ANCHORS:
            ref=np.load(root/'bounded_native_r1'/f'{raw:03d}/evidence.npz')
            gate=np.load(root/'bounded_white_inputs'/f'{raw:03d}/diagnostic_gate.npz')
            oldwhite=gate['source_native_rgb'].copy();olds=gate['unknown_display_gate']
            oldwhite[:,olds]=(ref['gaussian_rgb']+(1-ref['alpha'])[None])[:,olds]
            checks['anchor_oldR2_IDs_exact']=np.array_equal(legacy,ref['existing_removed_ids']) and np.array_equal(new,ref['new_unknown_ids'])
            checks['anchor_oldR2_scope_query_exact']=np.array_equal(e['query'],ref['query']) and np.array_equal(scope,olds)
            errors['anchor_alpha']=float(np.abs(alpha-ref['alpha']).max());errors['anchor_G']=float(np.abs(g-ref['gaussian_rgb']).max())
            errors['anchor_white']=float(np.abs(e['white_rgb']-oldwhite).max())
            checks['anchor_oldR2_floats_exact']=all(errors[k]==0 for k in ['anchor_alpha','anchor_G','anchor_white'])
        ok=all(checks.values());all_ok &= ok
        reports.append(dict(raw=raw,passed=bool(ok),checks={k:bool(v) for k,v in checks.items()},max_errors=errors,
            full_evidence_bytes=ep.stat().st_size,full_evidence_sha256=digest(ep),PNG_sha256=png_hashes,
            rows=dict(existing=len(legacy),new=len(new),final=len(final),legacy_outside=len(e['inherited_outside_ids']))))
        compact.append({k:e[k] for k in KEYS})
    audit=dict(schema='r2-inherited-delivery-audit-v1',passed=bool(all_ok),raws=RAWS,frames=reports,
        checks=dict(all49_present=True,all_5_gates_passed=True,full_float_read_on_remote=True,
                    local_package_contains_full_float=False,CPU_only=True,PNG_decoded=49*len(PNGS)),
        read_source=str(base),GPU_calls=0,training_updates=0,visual_quality_accepted=False,
        audit_scope='independent checks of saved native components/IDs; no new GPU attribution',
        wall_seconds=time.monotonic()-started)
    write(a.output/'audit.json',audit)
    assert all_ok, 'Audit failed: preserve failed audit, do not package'
    delivery=a.output/'r2_delivery';delivery.mkdir();dstroot=delivery/'r2_inherited49';dstroot.mkdir()
    for raw,data,report in zip(RAWS,compact,reports):
        src=base/f'{raw:03d}';dst=dstroot/f'{raw:03d}';dst.mkdir()
        for name in PNGS:shutil.copy2(src/(name+'.png'),dst/(name+'.png'))
        shutil.copy2(src/'receipt.json',dst/'receipt.json')
        np.savez_compressed(dst/'evidence.npz',**data)
        report['packaged_evidence_sha256']=digest(dst/'evidence.npz')
        report['packaged_evidence_bytes']=(dst/'evidence.npz').stat().st_size
    for name in ['summary.json','status.json']:shutil.copy2(base/name,dstroot/name)
    for name in GATES:shutil.copy2(a.run/(name+'.json'),delivery/(name+'.json'))
    audit['compact_NPZ_keys']=KEYS;audit['wall_seconds']=time.monotonic()-started
    write(a.output/'audit.json',audit);write(delivery/'local_audit.json',audit)
    anchor_size=sum((base/f'{r:03d}/evidence.npz').stat().st_size for r in ANCHORS)
    if anchor_size<=20*1024*1024:
        for raw in ANCHORS:
            dst=delivery/'anchor_full_float'/f'{raw:03d}';dst.mkdir(parents=True)
            shutil.copy2(base/f'{raw:03d}/evidence.npz',dst/'evidence.npz')
    archive=a.output/'r2_inherited49_light.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(delivery,arcname='r2_delivery')
    result=dict(passed=True,archive=str(archive),bytes=archive.stat().st_size,archive_sha256=digest(archive),
        anchor_full_float_bytes=anchor_size,anchor_full_float_packaged=anchor_size<=20*1024*1024,
        full_float_preserved_remote=str(base),files=sum(p.is_file() for p in delivery.rglob('*')),wall_seconds=time.monotonic()-started)
    write(a.output/'delivery_manifest.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
