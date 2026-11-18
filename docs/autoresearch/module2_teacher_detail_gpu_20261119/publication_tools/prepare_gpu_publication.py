"""GPU来源发布准备：默认只盘点/备份，不写工作树、共享index或任何ref。"""
from pathlib import Path
import argparse,datetime,difflib,hashlib,json,os,shutil,subprocess

R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
G=Q/'gpu_20261119';PUB=G/'publication'
BRANCH='research/worldsim-v8.2-drivingeditor-v44'
D='docs/autoresearch/module2_teacher_detail_gpu_20261119'
TASK='MODULE2-TEACHER-DETAIL-GPU-20261119'
MODULES=('fixed_material_read_operator.py','world_texture_runtime.py','gpu_entry_guard.py')
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def git(*args):return subprocess.check_output(['git',*args],cwd=R,env=ENV)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def index_path():
    p=Path(git('rev-parse','--git-path','index').decode().strip());return p if p.is_absolute()else R/p
def live_remote():
    rows=git('ls-remote','origin','refs/heads/'+BRANCH).decode().splitlines()
    if len(rows)!=1:raise RuntimeError('Publication remote head is ambiguous')
    return rows[0].split()[0]

def selected_sources():
    sources={}
    # CPU mother source and GPU actual/failure engineering source have separate archive roots.
    for scope,folder in [('cpu_preparation',Q),('gpu',G)]:
        for p in sorted(folder.rglob('*.py')):
            rel=p.relative_to(folder)
            if any(s in rel.parts for s in ('publication','__pycache__')):continue
            if scope=='cpu_preparation'and rel.parts[0]in('gpu_20261119',):continue
            sources[D+'/run_code/'+scope+'/'+rel.as_posix()]=str(p)
    for name in MODULES:
        sources['scripts/worldsim_v82/'+name]=str(Q/'code'/name)
        sources['tests/worldsim_v82/test_'+name]=str(Q/'code'/('test_'+name))
    dependencies={
        'road_capacity_native.py':Q.parent/'road_capacity_gpu_20261118_r1/code/road_capacity_native.py',
        'ideal_capacity_metrics.py':Q.parent/'road_capacity_gpu_20261118_r1/code/ideal_capacity_metrics.py',
        'ideal_world_teacher.py':Q.parent/'road_capacity_cpu_20261117_r1/ideal_code/ideal_world_teacher.py'}
    for name,p in dependencies.items():sources[D+'/run_code/dependencies/'+name]=str(p)
    return sources

def inventory(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing inventory snapshot')
    output.mkdir(parents=True)
    idx=index_path();original=idx.read_bytes()if idx.exists()else None
    head=git('rev-parse','HEAD').decode().strip();remote=live_remote()
    for rel in ('docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V82-F06.md'):
        p=output/'working_before'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/rel,p)
    if original is not None:(output/'shared_index.original').write_bytes(original)
    status=git('status','--porcelain').decode();(output/'status_before.txt').write_text(status)
    sources=selected_sources()
    listing={rel:dict(path=p,bytes=Path(p).stat().st_size,sha256=sha(p))for rel,p in sources.items()}
    refs={ref:git('rev-parse',ref).decode().strip()for ref in ('HEAD','refs/heads/'+BRANCH,'refs/remotes/origin/'+BRANCH)}
    receipt=dict(schema='teacher_detail_gpu_publication_inventory.v1',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        active_HEAD=head,local_refs=refs,live_remote_parent=remote,branch=BRANCH,shared_index_path=str(idx),
        shared_index_sha256=None if original is None else hashlib.sha256(original).hexdigest(),
        selected_code_sources=listing,source_snapshot_role='provisional inventory; final prepare rebinds actual immutable source after evaluation freeze',
        working_docs_modified=False,shared_index_used=False,private_index_created=False,commit_created=False,pushed=False,
        CPU_and_GPU_native_executed_by_this_publication_step=False)
    if git('rev-parse','HEAD').decode().strip()!=head or (idx.read_bytes()if idx.exists()else None)!=original:raise RuntimeError('Active HEAD/index changed')
    write(output/'inventory.json',receipt);print(json.dumps(dict(output=str(output),files=len(sources),remote_parent=remote,commit_created=False)))

def prepare(output,templates,extra_tests=(),extra_test_count=0):
    output,templates=Path(output),Path(templates)
    if output.exists():raise FileExistsError('Preserve previous prepared tree')
    idx=index_path();original=idx.read_bytes()if idx.exists()else None;head=git('rev-parse','HEAD').decode().strip()
    # Read the actual remote afresh. A local research branch is explicitly not publication authority.
    base=live_remote()
    try:git('cat-file','-e',base+'^{commit}')
    except subprocess.CalledProcessError:
        git('fetch','origin','refs/heads/'+BRANCH)
        if git('rev-parse','FETCH_HEAD').decode().strip()!=base:raise RuntimeError('Remote moved while fetching; rerun preparation in a new directory')
    output.mkdir(parents=True);tree=output/'tree';tree.mkdir();files=[];bindings={};code={};roles={}
    def copy(rel,p,role):
        p=Path(p)
        if p.suffix.lower()not in('.py','.json','.md','.svg','.html','.txt')or p.stat().st_size>2_000_000:raise ValueError('Only small selected source/docs/evidence: '+str(p))
        target=tree/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        files.append(rel);bindings[rel]=dict(path=str(p),sha256=sha(p));roles[rel]=role
        if p.suffix=='.py':code[rel]=str(p)
    for rel,p in selected_sources().items():copy(rel,p,'source_actual_or_engineering_snapshot')
    canonical={'report.md':G/'report.md','results.json':G/'results.json','protocol.json':Q/'protocol.json',
        'authorization.json':G/'authorization.json','architecture.svg':G/'architecture.svg'}
    for name,p in canonical.items():copy(D+'/'+name,p,'root_frozen_canonical')
    copy(D+'/evidence/ideal_field_config.json',Q.parent/'road_capacity_cpu_20261117_r1/ideal_targets_r2/field_config.json','independent_analytic_control_configuration')
    for p in sorted((PUB/'tools').iterdir()):
        if p.suffix in('.py','.md'):copy(D+'/publication_tools/'+p.name,p,'publication_tool')
    results=json.loads((G/'results.json').read_text())
    if results.get('human_verdict','missing')is not None:raise ValueError('Human verdict cannot be filled by source publication')
    for p in sorted(G.rglob('*.json')):
        rel=p.relative_to(G)
        if any(x in rel.parts for x in ('publication','__pycache__'))or p.name=='gpu_lease.json'or p.stat().st_size>2_000_000:continue
        if len(rel.parts)==1 and p.name in canonical:continue
        copy(D+'/evidence/gpu/'+rel.as_posix(),p,'small_GPU_receipt')
    # Root provides complete publication status and explicit working candidate; no inferred splice of parallel work.
    work={};patch=[]
    for rel,name in [('docs/RESEARCH_STATUS.md','RESEARCH_STATUS.md'),('docs/EXPERIMENTS.md','experiments_append.md'),('docs/research_failures/entries/V82-F06.md','failure_append.md')]:
        before=(R/rel).read_bytes();parent=git('show',base+':'+rel);addition=(templates/name).read_bytes()
        if rel.endswith('RESEARCH_STATUS.md'):
            pub=addition;working=(templates/'RESEARCH_STATUS_working.md').read_bytes()
            # Require all task-labelled working paragraphs to survive unless they are the replaced 3D CPU snapshot.
            for paragraph in before.decode().split('\n\n'):
                if 'ROAD-2D-TEACHER-'in paragraph and paragraph not in working.decode():raise ValueError('Parallel 2D status paragraph was dropped')
        else:
            if TASK.encode()in parent or TASK.encode()in before:raise ValueError('Task already appended; review existing document')
            pub=parent.rstrip()+b'\n\n'+addition;working=before.rstrip()+b'\n\n'+addition
        target=tree/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(pub);files.append(rel);roles[rel]='root_authored_document'
        for folder,data in [('working_before',before),('working_docs',working)]:
            p=output/folder/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        work[rel]=dict(before_sha256=hashlib.sha256(before).hexdigest(),candidate=str(output/'working_docs'/rel))
        patch.extend(difflib.unified_diff(parent.decode().splitlines(True),pub.decode().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    (output/'publication_docs.patch').write_text(''.join(patch))
    # Regenerate failure navigation only in isolated scratch, never in the active worktree.
    scratch=output/'failure_index_scratch';names=['scripts/build_research_failure_index.py','scripts/query_research_failures.py','docs/research_failures/history_index.json']
    names+=git('ls-tree','-r','--name-only',base,'--','docs/research_failures/entries').decode().splitlines()
    for rel in names:
        p=scratch/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(git('show',base+':'+rel))
    shutil.copy2(tree/'docs/research_failures/entries/V82-F06.md',scratch/'docs/research_failures/entries/V82-F06.md')
    log=subprocess.check_output(['/root/autodl-tmp/envs/motionproj/bin/python',str(scratch/'scripts/build_research_failure_index.py')],env=ENV,text=True)
    for p in sorted((scratch/'docs/research_failures').rglob('*.md')):
        rel=p.relative_to(scratch).as_posix()
        if '/entries/'in rel:continue
        try:old=git('show',base+':'+rel)
        except subprocess.CalledProcessError:old=b''
        if p.read_bytes()!=old:copy(rel,p,'isolated_generated_failure_navigation')
    tests=['tests/worldsim_v82/test_'+name for name in MODULES]+[D+'/run_code/cpu_preparation/'+x for x in
        ('permission/code/test_permission_cpu.py','pixel_render/code/test_pixel_render_cpu.py','trellis_hf/code/test_trellis_hf_seed.py')]
    tests+=[D+'/run_code/gpu/field_training/code/test_field_training_state.py']
    extra_imports=[]
    for rel in extra_tests:
        rel=Path(rel)
        if rel.is_absolute()or '..'in rel.parts or rel.suffix!='.py':raise ValueError('Extra test must be an explicit relative GPU source path')
        target=D+'/run_code/gpu/'+rel.as_posix();tests.append(target);extra_imports.append(str(Path(target).parent))
    if bool(extra_tests)!=(extra_test_count>0):raise ValueError('Extra test files and declared successful call count must be supplied together')
    files=sorted(set(files))
    if any(x not in files for x in tests):raise ValueError('Final CPU test source missing')
    plan=dict(output=str(output/'stage'),branch=BRANCH,expected_base=base,expected_active_head=head,files=files,
        source_overrides={x:str(tree/x)for x in files},override_sha256={x:sha(tree/x)for x in files},
        original_source_bindings=bindings,executed_code_bindings=code,tests=tests,expected_tests=43+extra_test_count,archive_root=D,
        working_docs=work,message='feat(road): evaluate native pixel-world material and teacher detail\n\nArchive fixed-coverage reader and permission controls, native lifecycle and observed quality evidence without promoting the default scene.',
        import_paths=['scripts/worldsim_v82',D+'/run_code/cpu_preparation/code',D+'/run_code/cpu_preparation/permission/code',D+'/run_code/cpu_preparation/pixel_render/code',D+'/run_code/cpu_preparation/trellis_hf/code',D+'/run_code/gpu/field_training/code',*extra_imports])
    write(output/'plan.json',plan);write(output/'source_list.json',dict(roles=roles,bindings=bindings))
    write(output/'failure_index_regeneration.json',dict(log=log,active_worktree_modified=False))
    if git('rev-parse','HEAD').decode().strip()!=head or (idx.read_bytes()if idx.exists()else None)!=original:raise RuntimeError('Active HEAD/index changed')
    write(output/'prepared.json',dict(completed=True,live_remote_parent=base,active_HEAD=head,files=len(files),shared_index_unchanged=True,private_index_created=False,commit_created=False,pushed=False))
    print(json.dumps(dict(output=str(output),files=len(files),remote_parent=base,tests_expected=43+extra_test_count)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--canonical-frozen',action='store_true');p.add_argument('--templates',type=Path);p.add_argument('--extra-test',action='append',default=[]);p.add_argument('--extra-test-count',type=int,default=0);a=p.parse_args()
    if a.prepare:
        if not a.canonical_frozen or a.templates is None:raise RuntimeError('Root canonical freeze and explicit doc templates required')
        prepare(a.output,a.templates,a.extra_test,a.extra_test_count)
    else:inventory(a.output)
