"""Prepare exact completed R2 sources without touching shared Git work/index."""
from pathlib import Path
import argparse,csv,hashlib,json,os,shutil,subprocess
import numpy as np
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PUB=BASE/'road_surface_residual_20261121_r2/publication'
BRANCH='research/worldsim-v8.2-drivingeditor-v44'
ROOTS={'surface':BASE/'road_surface_residual_20261121_r2'}
DOCS={'surface':'docs/autoresearch/module2_surface_residual_20261121_r2'}
ALLOWED={'.py','.json','.md','.svg','.txt','.js','.html','.ps1','.csv'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def append(a,b):return a.rstrip()+b'\n\n'+b.rstrip()+b'\n'

def export_evidence(p):
    source=p/'k_gate/per_candidate_K_evidence.npz'
    target=PUB/'derived';target.mkdir(exist_ok=True)
    z=np.load(source,allow_pickle=False)
    fields={k:dict(dtype=str(z[k].dtype),shape=list(z[k].shape),data=z[k].tolist()) for k in z.files}
    out=target/'per_candidate_K_evidence.json'
    dump(out,dict(schema='lossless_existing_K_VJP_evidence_export.v1',source=str(source),source_SHA256=sha(source),states=read(p/'k_gate/receipt.json')['states'],views=[86,110,134,179,198],boundary='Exact existing initial/max-state five-view evidence; no parameter update, interval or arbitrary-view claim.',fields=fields))
    back=read(out)['fields']
    assert all(np.array_equal(z[k],np.asarray(back[k]['data'],dtype=back[k]['dtype'])) for k in z.files)
    columns=[]
    for k in z.files:
        columns.extend([(k,ix,k+''.join('_'+str(i) for i in ix)) for ix in np.ndindex(z[k].shape[1:])])
    with (target/'per_candidate_K_evidence.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow([c[2] for c in columns])
        for i in range(len(z['candidate_id'])):writer.writerow([z[k][(i,)+ix].item() for k,ix,_ in columns])
    dump(target/'serialization_receipt.json',dict(complete=True,source=str(source),source_SHA256=sha(source),candidates=len(z['candidate_id']),all_fields_roundtrip_exact=True,CUDA_calls=0,new_native_calls=0,optimizer_updates=0,exports={f.name:dict(SHA256=sha(f),bytes=f.stat().st_size) for f in [out,target/'per_candidate_K_evidence.csv']}))
    numeric=p/'diagnostic/color_optimality'
    latest=read(numeric/'solver/latest.json')
    paths=[Path(latest['workspace']),Path(latest['RNG']),numeric/'solver/final_workspace.npz',numeric/'solver/final_RNG.pt',p/'appearance_fit/asset/road.npz',p/'appearance_fit/asset/provenance.npz',p/'appearance_fit/asset/persistent/scene.json',p/'appearance_fit/asset/persistent/proof.json',p/'appearance_fit/checkpoint.pt',p/'train_inputs/allowed_color_world_rows.npz']
    paths+=list((numeric/'training_masks').glob('*.npz'))+list((numeric/'cpu_masks').glob('*.npz'))
    deps={str(f):dict(bytes=f.stat().st_size,SHA256=sha(f),Git_payload=False) for f in paths}
    dump(target/'numeric_workspace_dependencies.json',dict(schema='fixed_geometry_numeric_workspace_dependencies.v1',workspace_and_RNG_binding='solver/latest.json in exact diagnostic snapshot',files=deps,full_restore_smoke_receipt=str(numeric/'smoke/receipt.json'),standalone_scene_backup=False,formal_R2_replaced=False,archive_GPU_calls=0,restore_note='Restore the complete L-BFGS-B reverse-communication workspace and paired RNG with its pinned SciPy ABI and exact frozen source asset, allowed row IDs, masks, teacher and original Adam1200 normalizers. Static/dynamic/environment and teacher dependencies remain external. NPZ/PT state is not in Git; original absolute-path manifests require explicit relocation/proof rebinding. The separately verified R2 takeback is the formal asset and Adam/coverage recovery state, not this diagnostic solver package.'))

def collect(require_reports=True):
    selected={}
    def add(dest,source,role):
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        if source.suffix not in ALLOWED or source.stat().st_size>(8_000_000 if source.suffix=='.json' else 2_000_000):raise ValueError('Not lightweight source: '+str(source))
        if dest in selected:raise ValueError('Duplicate destination '+dest)
        selected[dest]=dict(path=str(source),sha256=sha(source),bytes=source.stat().st_size,role=role)
    p=ROOTS['surface'];d=DOCS['surface']
    assert read(p/'status.json')['complete'] and read(p/'receipt.json')['complete'] and read(p/'bench/completed.json')['complete']
    assert read(p/'smoke/receipt.json')['verified'] and read(p/'bench/parent_freeze_proof.json')['passed']
    receipt=read(p/'receipt.json')
    assert sha(p/'appearance_fit/asset/road.npz')==receipt['road_SHA256']=='70cc545db6ce595b24c0ec2ba836d7058edbb73734d6873c880f3e634fb08d9f'
    assert sha(p/'appearance_fit/asset/provenance.npz')==receipt['provenance_SHA256']
    for rel,key in [('proposal/manifest.json','source_sha256'),('train_inputs/manifest.json','source_SHA256'),('k_gate/receipt.json','input_SHA256'),('bench/code_manifest.json','source_sha256'),('bench/metrics.json','source_sha256')]:
        mapping=read(p/rel)[key]
        for file,h in mapping.items():
            if isinstance(h,str) and len(h)==64 and file.startswith('/'):
                assert Path(file).is_file() and sha(file)==h,('Source binding changed',file)
    gate=read(p/'k_gate/receipt.json')
    assert sha(gate['source_code'])==gate['source_code_SHA256'] and gate['first_wave_formal_unchanged'] and gate['optimizer_updates']==0
    diagdir=p/'diagnostic/fixed_color_interval'
    diag=read(diagdir/'receipt.json')
    assert diag['complete'] and diag['formal_sources_unchanged'] and diag['optimizer_updates']==diag['nativeB']==0 and diag['nativeF']==4
    for f,h in diag['source_SHA256'].items():assert sha(f)==h,('Diagnostic source changed',f)
    for raw,h in diag['output_SHA256'].items():assert sha(diagdir/(f'{int(raw):03d}.npz'))==h
    audit=read(diagdir/'boxed_rgb_audit.json')
    assert audit['optimizer_updates']==audit['additional_GPU_calls']==0 and not audit['HF_recomputed'] and not audit['formal_results_changed']
    for f,h in audit['source_SHA256'].items():assert sha(f)==h,('CPU diagnostic source changed',f)
    assert sha(audit['source_code'])==audit['source_code_SHA256'] and sha(audit['receipt'])==audit['receipt_SHA256']
    if require_reports:assert sha(PUB/'inputs/surface/local_delivery/work/road_surface_residual_20261121_r2/probe_fixed_color_interval.py')==diag['script_SHA256']
    numeric=p/'diagnostic/color_optimality';n=read(numeric/'numeric_receipt.json')
    assert n['complete'] and n['FG']==2 and n['accepted']==1 and n['new_point_count']==0 and n['source_R2_formal1200_unchanged'] and n['formal_source_unchanged']
    for f,h in n['source_SHA256'].items():assert sha(f)==h,('Numeric source changed',f)
    assert sha(numeric/'code/probe_color_optimality.py')==n['numeric_script_SHA256']
    for name,rec in read(numeric/'code/source_manifest.json').items():
        assert sha(numeric/'code'/name)==rec['SHA256'] and sha(rec['original_path'])==rec['SHA256']
    assert read(numeric/'precision_qualification.json')['verified'] and read(numeric/'smoke/receipt.json')['verified']
    identity=read(numeric/'objective_identity.json');assert identity['verified'] and not identity['normalizers_recalibrated']
    for f,h in identity['source_SHA256'].items():assert sha(f)==h
    if require_reports:assert read(numeric/'status.json')['complete'],'Numeric CPU finalization pending'
    basics=['contract.json','launch_receipt.json','status.json','receipt.json','resource_ledger.json','export_manifest.json','teacher_manifest.json','train_inputs/manifest.json','train_inputs/domains_manifest.json','train_inputs/reuse_caps_authority.json','proposal/manifest.json','proposal/contract.json','proposal/audit.json','proposal/residual_gate_audit.json','proposal/geometry_support_legend.json','proposal/residual_pixel_K_table.csv','bench/metrics.json','bench/protocol.json','bench/completed.json','bench/interface_smoke.json','bench/code_manifest.json','bench/allowed_scope_preflight.json','bench/parent_freeze_proof.json','bench/residual_pixel_K_table.csv','delivery/latest_manifest.json','parent/rendered/receipt.json']
    for rel in basics:add(d+'/'+rel,p/rel,'completed_R2_exact_metadata')
    for sub in ['code','cpu','bench','k_gate']:
        for f in sorted((p/sub).glob('*.py')):add(d+'/run_code/'+f.relative_to(p).as_posix(),f,'actual_remote_run_source')
    add(d+'/run_code/prepare_training_inputs.py',p/'prepare_training_inputs.py','actual_training_input_preparation')
    for sub in ['smoke','qualification','k_gate']:
        for f in sorted((p/sub).rglob('*.json')):
            if f.name in ['scene.json','gpu_lease.json'] or '/probe_asset/' in f.as_posix():continue
            add(d+'/evidence/'+f.relative_to(p).as_posix(),f,'actual_smoke_K_and_persistence_receipt')
    for sub in ['activation_init','dense_init','coverage_fit','appearance_fit']:
        for f in sorted((p/sub).rglob('*.json')):
            if f.name not in ['authority_receipt.json','extension_receipt.json','proof.json','build.json','receipt.json']:continue
            add(d+'/evidence/'+f.relative_to(p).as_posix(),f,'endpoint_persistence_or_render_receipt')
    for rel in ['package_receipt.json','manifest.json','dependencies.json','checkpoints.json','RESTORE.md']:
        add(d+'/takeback/'+rel,p/'delivery/persistent_asset_takeback_r2'/rel,'separate_asset_delivery_metadata_no_payload')
    for f in sorted((p/'diagnostic').rglob('*')):
        if f.is_file() and f.suffix in ['.py','.json','.md','.txt','.csv'] and f.name!='gpu_lease.json':
            role='separate_fixed_geometry_numeric_2FG_not_formal1200' if '/color_optimality/' in f.as_posix() else 'separate_zero_update_color_interval_diagnostic'
            add(d+'/evidence/'+f.relative_to(p).as_posix(),f,role)
    for f in sorted((PUB/'inputs/surface').rglob('*')):
        if f.is_file():
            rel=f.relative_to(PUB/'inputs/surface').as_posix()
            if rel=='report.md' and not require_reports:continue
            add(d+'/'+rel,f,'exact_local_report_or_UI_source_snapshot')
    export_evidence(p)
    for f in sorted((PUB/'derived').iterdir()):add(d+'/evidence/K_VJP/'+f.name,f,'lossless_existing_candidate_evidence_export')
    deps=[('R1_run_adaptive.py',BASE/'road_surface_adaptive_20261121_r1/code/run_adaptive.py'),('road_detail_metrics.py',BASE/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1/metrics/road_detail_metrics.py'),('road_capacity_native.py',BASE/'road_capacity_gpu_20261118_r1/code/road_capacity_native.py')]
    for module in ['persistent_road_handoff','v44_factory','v44_scene','v44_manifest','road_asset']:deps.append(('native_loader/'+module+'.py',REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit'/(module+'.py')))
    for name,f in deps:add(d+'/dependencies/'+name,f,'exact_imported_source_snapshot_only')
    for f in [PUB/x for x in ['prepare_archive.py','publish_selected.py','validate_archive.py','verify_published.py']]:add(d+'/publication_tools/'+f.name,f,'private_index_publication_source')
    if require_reports:assert d+'/report.md' in selected and d+'/architecture.svg' in selected
    return selected

def prepare(output,proxy,catalog=False):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HTTP_PROXY=proxy,HTTPS_PROXY=proxy,ALL_PROXY=proxy)
    def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,env=env,timeout=90)
    selected=collect(not catalog)
    if catalog:
        dump(PUB/'selection_catalog.json',dict(complete=False,commit_allowed=False,files=selected,total_bytes=sum(x['bytes'] for x in selected.values())))
        print(json.dumps(dict(catalog_files=len(selected),bytes=sum(x['bytes'] for x in selected.values()),commit_allowed=False)));return
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    head=git('rev-parse','HEAD').decode().strip();index=REPO/git('rev-parse','--git-path','index').decode().strip();index_before=index.read_bytes()
    working_before={p:sha(REPO/p) for p in git('diff','--name-only').decode().splitlines() if (REPO/p).is_file()}
    working_2d={f.relative_to(REPO).as_posix():sha(f) for d in (REPO/'docs/autoresearch').glob('road_2d_teacher_*') if d.is_dir() for f in d.rglob('*') if f.is_file()}
    base=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]
    git('fetch','--no-write-fetch-head','origin',base)
    output.mkdir(parents=True);tree=output/'tree';tree.mkdir()
    files=[];overrides={};bindings={}
    for rel,rec in selected.items():
        dest=tree/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(rec['path']).read_bytes());assert sha(dest)==rec['sha256']
        files.append(rel);overrides[rel]=str(dest);bindings[rel]=dict(path=rec['path'],sha256=rec['sha256'])
    # Only the current 3D block is replaced; preserve every existing 2D paragraph.
    docs={}
    old=git('show',base+':docs/RESEARCH_STATUS.md').decode();marker='`ROAD-SURFACE-ADAPTIVE-20261121-R1`'
    if old.count(marker)!=1:raise ValueError('Review new current-status marker before publishing')
    start=old.index(marker);end=old.find('\n## ',start);end=len(old) if end<0 else end
    body=(PUB/'inputs/status_3d_snapshot.md').read_text();candidate=old[:start]+body.rstrip()+'\n'+old[end:]
    candidate=candidate.replace('更新：2026-11-20','更新：2026-11-21',1)
    for para in old.split('\n\n'):
        if 'ROAD-2D-TEACHER-' in para and para not in candidate:raise ValueError('Parallel 2D status would change')
    docs['docs/RESEARCH_STATUS.md']=candidate.encode()
    for rel,name in [('docs/EXPERIMENTS.md','experiments_append.md'),('docs/research_failures/entries/V82-F06.md','failure_append.md')]:
        before=git('show',base+':'+rel);addition=(PUB/'inputs'/name).read_bytes()
        if addition.strip() in before:raise ValueError('Duplicate publication addition')
        docs[rel]=append(before,addition)
    for rel,data in docs.items():
        dest=tree/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);files.append(rel);overrides[rel]=str(dest)
    scratch=output/'failure_index_scratch'
    names=['scripts/build_research_failure_index.py','scripts/query_research_failures.py','docs/research_failures/history_index.json']
    names+=git('ls-tree','-r','--name-only',base,'--','docs/research_failures/entries').decode().splitlines()
    for rel in names:
        f=scratch/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(git('show',base+':'+rel))
    shutil.copy2(tree/'docs/research_failures/entries/V82-F06.md',scratch/'docs/research_failures/entries/V82-F06.md')
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python','-B',str(scratch/'scripts/build_research_failure_index.py')],cwd=scratch,env=env,check=True,stdout=subprocess.DEVNULL)
    for f in sorted((scratch/'docs/research_failures').rglob('*')):
        if not f.is_file() or f.suffix not in ('.json','.md') or '/entries/' in f.as_posix():continue
        rel=f.relative_to(scratch).as_posix()
        try:before=git('show',base+':'+rel)
        except subprocess.CalledProcessError:before=b''
        if f.read_bytes()!=before:
            dest=tree/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes());files.append(rel);overrides[rel]=str(dest)
    for kind,d in DOCS.items():
        rel=d+'/source_manifest.json';dest=tree/rel
        dump(dest,dict(schema='byte_exact_lightweight_run_archive.v1',complete=True,run=str(ROOTS[kind]),archive_only=True,model_raw_media_included=False,quality_PASS=False,files={r:rec for r,rec in selected.items() if r.startswith(d+'/')}))
        files.append(rel);overrides[rel]=str(dest)
    protected=[line for line in git('ls-tree','-r',base,'--','docs/autoresearch/road_2d_teacher_hf_cpu_20261119','docs/autoresearch/road_2d_teacher_hf_gpu_20261119','docs/autoresearch/road_2d_teacher_prompt_20261120','AGENTS.md').decode().splitlines()]
    plan=dict(output=str(output/'stage'),branch=BRANCH,expected_base=base,expected_active_head=head,files=sorted(files),source_overrides=overrides,override_sha256={r:sha(overrides[r])for r in files},original_source_bindings=bindings,message='feat(road): archive certified residual surface repair R2\n\nRecord actual 13 activations, 1071 spatial appends and 107 qualified R1 newgroup reuses with original caps. Preserve direct-parent comparison, five-view joint K evidence, fixed-hinge contract, remaining rejected-K residuals and separate fixed-geometry diagnostics; no quality promotion or global capacity claim.')
    dump(output/'plan.json',plan);dump(output/'selected_sources.json',selected);dump(output/'protected_2d_tree.json',protected)
    if git('rev-parse','HEAD').decode().strip()!=head or index.read_bytes()!=index_before:raise RuntimeError('Shared HEAD/index changed')
    assert all(sha(REPO/p)==h for p,h in working_before.items()),'Existing dirty file changed during preparation'
    assert all(sha(REPO/p)==h for p,h in working_2d.items()),'Parallel 2D work changed during preparation'
    dump(output/'prepared.json',dict(complete=True,parent=base,active_HEAD=head,selected=len(files),working_dirty_sha256=working_before,working_2d_sha256=working_2d,shared_index_sha256=hashlib.sha256(index_before).hexdigest(),protected_2d_paths=len(protected),shared_worktree_modified=False,pushed=False))
    print(json.dumps(dict(plan=str(output/'plan.json'),parent=base,selected_files=len(files),protected_2d_paths=len(protected))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--proxy',required=True);p.add_argument('--catalog',action='store_true');a=p.parse_args();prepare(a.output,a.proxy,a.catalog)
