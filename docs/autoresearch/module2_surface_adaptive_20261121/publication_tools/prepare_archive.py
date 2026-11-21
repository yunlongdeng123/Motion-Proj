"""Prepare a byte-exact two-run documentation tree, without touching shared work/index."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PUB=BASE/'road_surface_adaptive_20261121_r1/publication'
BRANCH='research/worldsim-v8.2-drivingeditor-v44'
ROOTS={'common':BASE/'road_common1152_20261121','surface':BASE/'road_surface_adaptive_20261121_r1'}
DOCS={'common':'docs/autoresearch/module2_common1152_20261121','surface':'docs/autoresearch/module2_surface_adaptive_20261121'}
ALLOWED={'.py','.json','.md','.svg','.txt','.js','.html','.ps1'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def append(a,b):return a.rstrip()+b'\n\n'+b.rstrip()+b'\n'

def collect(require_reports=True):
    selected={}
    def add(dest,source,role):
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        if source.suffix not in ALLOWED or source.stat().st_size>(8_000_000 if source.suffix=='.json' else 2_000_000):raise ValueError('Not lightweight source: '+str(source))
        if dest in selected:raise ValueError('Duplicate destination '+dest)
        selected[dest]=dict(path=str(source),sha256=sha(source),bytes=source.stat().st_size,role=role)
    for kind,p in ROOTS.items():
        assert read(p/'status.json')['complete'] and read(p/'receipt.json')['complete']
        assert read(p/'bench/completed.json')['complete']
        d=DOCS[kind]
        basics=['contract.json','launch_receipt.json','status.json','receipt.json','resource_ledger.json','export_manifest.json','bench/metrics.json','bench/protocol.json','bench/completed.json','bench/interface_smoke.json']
        if kind=='common':
            basics+=['compute_ledger.json','camera_manifest.json','source_binding.json','pixel_interface_preflight.json','common_permission/receipt.json','common_permission/average_receipt.json','cpu_inputs/manifest.json','cpu_inputs/targets_manifest.json','cpu_inputs/source_footprint.json','r1_inputs/receipt.json','native/manifest.json','native/receipt.json','native/qualified_basis/receipt.json','teacher/manifest.json','teacher/qualification_cpu.json','formal/r3_normalizers.json','formal/r3/G_NUMERIC/receipt.json','formal/r3/G_NUMERIC/precision_qualification.json','formal/r3/G_NUMERIC/solver_latest.json']
            bindings=read(p/'source_binding.json')
            if not all(Path(f).is_file() and sha(f)==h for f,h in bindings.items()):raise ValueError('Frozen common source binding changed')
        else:
            basics+=['proposal/manifest.json','proposal/sampling_contract.json','proposal/surface_certificate.json','proposal/gap_diagnosis.json','proposal/geometry_support_legend.json','proposal/append_manifest.json','proposal/revision0/manifest.json','proposal/revision0/sampling_contract.json','proposal/revision0/gap_diagnosis.json']
            proposal=read(p/'proposal/manifest.json')
            if not all(Path(f).is_file() and sha(f)==h for f,h in proposal['source_sha256'].items()):raise ValueError('Frozen proposal source binding changed')
            diagnostic=read(p/'residual_feasibility/max_allowed_state/receipt.json')
            if not diagnostic['original_formal_files_unchanged'] or diagnostic['optimizer_updates']!=0:raise ValueError('Diagnostic is not separate zero-update evidence')
            if not all(Path(f).is_file() and sha(f)==h for f,h in diagnostic['input_SHA256'].items()):raise ValueError('Formal surface source changed after diagnostic')
            if sha(diagnostic['source_code'])!=diagnostic['source_code_SHA256']:raise ValueError('Diagnostic source changed')
        for rel in basics:add(d+'/'+rel,p/rel,'completed_run_metadata')
        for sub in ['code','cpu','cpu_inputs','bench','teacher']:
            for f in sorted((p/sub).glob('*.py')):add(d+'/run_code/'+f.relative_to(p).as_posix(),f,'actual_remote_run_source')
        if kind=='surface':add(d+'/run_code/proposal/select_append.py',p/'proposal/select_append.py','actual_once_only_selector')
        for f in sorted((p/'smoke').rglob('*.json')):
            rel=f.relative_to(p).as_posix()
            if f.name in ['scene.json','solver_latest.json','r3_normalizers.json']:continue
            add(d+'/evidence/'+rel,f,'actual_smoke_receipt')
        for f in sorted((p/'qualification').glob('*.json')):add(d+'/evidence/'+f.relative_to(p).as_posix(),f,'actual_five_view_protection')
        folders=[p/'formal'] if kind=='common' else [p/x for x in ['activation_init','activation_fit','dense_init','coverage_fit','appearance_fit']]
        for folder in folders:
            for f in sorted(folder.rglob('*.json')):
                if f.name not in ['authority_receipt.json','extension_receipt.json','proof.json','build.json','receipt.json']:continue
                add(d+'/evidence/'+f.relative_to(p).as_posix(),f,'endpoint_persistence_or_render_receipt')
        inputs=PUB/'inputs'/kind
        for f in sorted(inputs.rglob('*')):
            if not f.is_file():continue
            rel=f.relative_to(inputs).as_posix()
            if rel=='report.md' and not require_reports:continue
            add(d+'/'+rel,f,'exact_local_delivery_source_or_report')
        if require_reports:
            assert (d+'/report.md') in selected and (d+'/architecture.svg') in selected
    # Actual imported helper versions, copied as source snapshots instead of modifying the main modules.
    Q=BASE/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
    paths=[BASE/'road_capacity_gpu_20261118_r1/code',REPO/'scripts/worldsim_v82',Q/'code',G/'field_training/code',B/'code',B/'shared_field/code',B/'gs_oracle/code',Q/'pixel_render/code']
    helpers=['road_capacity_native','native_basis_adapter','world_texture_runtime','fixed_material_read_operator','material_readers','field_training_state','build_training_inputs','native_capacity_loss']
    for module in helpers:
        source=next((p/(module+'.py') for p in paths if (p/(module+'.py')).is_file()),None)
        if source is None:raise FileNotFoundError(module)
        add(DOCS['common']+'/dependencies/imported/'+module+'.py',source,'frozen_import_dependency')
    for module in ['checkpointed_lbfgsb','fp64_reduction_loss']:
        add(DOCS['common']+'/dependencies/numerical/'+module+'.py',B/'numerical_completion/code'/(module+'.py'),'frozen_numeric_dependency')
    for module in ['matched_i0_training','world_material_field']:
        add(DOCS['common']+'/dependencies/multiscale/'+module+'.py',BASE/'road_multiscale_material_20261118_r1/code'/(module+'.py'),'frozen_multiscale_dependency')
    for module in ['persistent_road_handoff','v44_factory','v44_scene','v44_manifest','road_asset']:
        add(DOCS['common']+'/dependencies/native_loader/'+module+'.py',REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit'/(module+'.py'),'exact_main_module_snapshot_only')
    for f in sorted((G/'quality_bench_20261120_r1/metrics').glob('*.py')):add(DOCS['common']+'/dependencies/quality_metrics/'+f.name,f,'actual_metrics_dependency')
    for f in sorted((ROOTS['surface']/'diagnostics').rglob('*')):
        if f.is_file() and f.suffix in ('.py','.json','.md') and f.stat().st_size<2_000_000:add(DOCS['surface']+'/evidence/'+f.relative_to(ROOTS['surface']).as_posix(),f,'zero_update_feasibility_diagnostic_separate_from_formal')
    for f in sorted((ROOTS['surface']/'residual_feasibility/max_allowed_state').glob('*')):
        if f.is_file() and f.suffix in ('.py','.json','.md') and f.stat().st_size<2_000_000:add(DOCS['surface']+'/evidence/'+f.relative_to(ROOTS['surface']).as_posix(),f,'zero_update_max_allowed_state_optimistic_diagnostic_not_formal_training')
    for f in [PUB/'prepare_archive.py',PUB/'publish_selected.py',PUB/'validate_archive.py',PUB/'verify_published.py']:
        add(DOCS['surface']+'/publication_tools/'+f.name,f,'private_index_publication_source')
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
    old=git('show',base+':docs/RESEARCH_STATUS.md').decode();marker='`MODULE2-NATIVE-CAPACITY-20261120`'
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
    plan=dict(output=str(output/'stage'),branch=BRANCH,expected_base=base,expected_active_head=head,files=sorted(files),source_overrides=overrides,override_sha256={r:sha(overrides[r])for r in files},original_source_bindings=bindings,message='feat(road): archive common1152 training and adaptive surface coverage\n\nFreeze the accepted teacher, two completed fixed-budget chains, real-GS append and five-view protection receipts. Preserve unsupported editing domains and residual black gaps; no quality promotion or representation-limit claim.')
    dump(output/'plan.json',plan);dump(output/'selected_sources.json',selected);dump(output/'protected_2d_tree.json',protected)
    if git('rev-parse','HEAD').decode().strip()!=head or index.read_bytes()!=index_before:raise RuntimeError('Shared HEAD/index changed')
    assert all(sha(REPO/p)==h for p,h in working_before.items()),'Existing dirty file changed during preparation'
    assert all(sha(REPO/p)==h for p,h in working_2d.items()),'Parallel 2D work changed during preparation'
    dump(output/'prepared.json',dict(complete=True,parent=base,active_HEAD=head,selected=len(files),working_dirty_sha256=working_before,working_2d_sha256=working_2d,shared_index_sha256=hashlib.sha256(index_before).hexdigest(),protected_2d_paths=len(protected),shared_worktree_modified=False,pushed=False))
    print(json.dumps(dict(plan=str(output/'plan.json'),parent=base,selected_files=len(files),protected_2d_paths=len(protected))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--proxy',required=True);p.add_argument('--catalog',action='store_true');a=p.parse_args();prepare(a.output,a.proxy,a.catalog)
