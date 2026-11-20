"""准备当前实验的小型来源与文档覆盖树；不改活动分支或共享 index。"""
from pathlib import Path
import argparse,difflib,hashlib,json,os,shutil,subprocess

R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/native_capacity_20261120_r1')
D='docs/autoresearch/module2_native_capacity_20261120'
BRANCH='research/worldsim-v8.2-drivingeditor-v44'
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

def git(*args):return subprocess.check_output(['git',*args],cwd=R,env=ENV,timeout=45)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def append(a,b):return a+(b''if a.endswith(b'\n\n')else b'\n'if a.endswith(b'\n')else b'\n\n')+b

def replace_status(data,body):
    text=data.decode('utf-8');marker='`MODULE2-QUALITY-BENCH-20261120`'
    if text.count(marker)!=1:raise ValueError('expected single current3D status marker')
    start=text.index(marker);end=text.find('\n## ',start);end=len(text)if end<0 else end
    candidate=text[:start]+body.rstrip()+'\n\n'+text[end:]
    for para in text.split('\n\n'):
        if 'ROAD-2D-TEACHER-'in para and para not in candidate:raise ValueError('parallel 2D status changed')
    return candidate.encode('utf-8')

def prepare(output,selection,templates):
    output,templates=Path(output),Path(templates);s=json.loads(Path(selection).read_text())
    if output.exists():raise FileExistsError('preserve earlier preparation')
    head=git('rev-parse','HEAD').decode().strip()
    idx=Path(git('rev-parse','--git-path','index').decode().strip());idx=idx if idx.is_absolute()else R/idx
    original_index=idx.read_bytes()if idx.exists()else None
    base=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0];git('cat-file','-e',base+'^{commit}')
    output.mkdir(parents=True);tree=output/'tree';tree.mkdir();files=[];sources={};roles={}
    def copy(rel,path,role):
        path=Path(path)
        limit=8_000_000 if path.suffix=='.json'else 2_000_000
        if not path.is_file()or path.stat().st_size>limit:raise ValueError('missing or oversized selected artifact:'+str(path))
        if path.suffix not in('.py','.json','.md','.svg','.txt'):raise ValueError('unexpected artifact type')
        dest=tree/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        files.append(rel);sources[rel]=dict(path=str(path),sha256=sha(path));roles[rel]=role
    for name in s['canonical']:
        source,dest=(name['source'],name['dest'])if isinstance(name,dict)else(name,name)
        copy(D+'/'+dest,B/source,'canonical_report')
    for path in sorted(B.rglob('*.py')):
        relative=path.relative_to(B)
        if any(x in relative.parts for x in('publication','__pycache__')):continue
        copy(D+'/run_code/'+relative.as_posix(),path,'actual_or_engineering_source')
    copy(D+'/quality_bench_20261120_r1/metrics/road_detail_metrics.py',
        B.parent/'quality_bench_20261120_r1/metrics/road_detail_metrics.py','byte_exact_frozen_dependency_for_isolated_CPU_tests')
    for name in s['evidence']:copy(D+'/evidence/'+name,B/name,'selected_small_receipt')
    tools=Path(selection).parent
    for name in s.get('publication_tools',[]):copy(D+'/publication_tools/'+name,tools/name,'publication_tool')
    work={};patch=[]
    for rel,addition in [('docs/RESEARCH_STATUS.md','status_3d_snapshot.md'),('docs/EXPERIMENTS.md','experiments_append.md'),('docs/research_failures/entries/V82-F06.md','failure_append.md')]:
        before=(R/rel).read_bytes();parent=git('show',base+':'+rel);body=(templates/addition).read_bytes()
        if rel.endswith('RESEARCH_STATUS.md'):
            pub=replace_status(parent,body.decode('utf-8'));candidate=replace_status(before,body.decode('utf-8'))
        else:
            if body.strip()in parent:raise ValueError('duplicate experiment addition')
            pub=append(parent,body);candidate=append(before,body)
        dest=tree/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(pub);files.append(rel);roles[rel]='scoped_research_doc'
        for folder,data in [('working_before',before),('working_docs',candidate)]:
            dest=output/folder/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        work[rel]=dict(before_sha256=sha(R/rel),candidate=str(output/'working_docs'/rel))
        patch.extend(difflib.unified_diff(parent.decode().splitlines(True),pub.decode().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    scratch=output/'failure_index_scratch'
    names=['scripts/build_research_failure_index.py','scripts/query_research_failures.py','docs/research_failures/history_index.json']
    names+=git('ls-tree','-r','--name-only',base,'--','docs/research_failures/entries').decode().splitlines()
    for rel in names:
        dest=scratch/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(git('show',base+':'+rel))
    shutil.copy2(tree/'docs/research_failures/entries/V82-F06.md',scratch/'docs/research_failures/entries/V82-F06.md')
    log=subprocess.check_output(['/root/autodl-tmp/envs/motionproj/bin/python','-B',str(scratch/'scripts/build_research_failure_index.py')],env=ENV,text=True)
    navigation=[]
    for path in sorted((scratch/'docs/research_failures').rglob('*')):
        if not path.is_file()or path.suffix not in('.md','.json'):continue
        rel=path.relative_to(scratch).as_posix()
        if '/entries/'in rel:continue
        try:old=git('show',base+':'+rel)
        except subprocess.CalledProcessError:old=b''
        if path.read_bytes()!=old:copy(rel,path,'generated_failure_navigation');navigation.append(rel)
    write(output/'failure_index_regeneration.json',dict(log=log,changed=navigation,active_tree_modified=False))
    (output/'publication_docs.patch').write_text(''.join(patch),encoding='utf-8');files=sorted(set(files))
    groups=s['test_groups'];tests=[t for g in groups for t in g['tests']]
    plan=dict(output=str(output/'stage'),branch=BRANCH,expected_base=base,expected_active_head=head,
        files=files,source_overrides={f:str(tree/f)for f in files},override_sha256={f:sha(tree/f)for f in files},
        original_source_bindings=sources,executed_code_bindings={f:v['path']for f,v in sources.items()if f.endswith('.py')},
        archive_root=D,tests=tests,test_groups=groups,expected_tests=sum(g['expected']for g in groups),
        import_paths=s['import_paths'],working_docs=work,
        message='feat(road): evaluate native-resolution detail distillation and fixed Gaussian capacity\n\n'
        'Keep registered endpoints, conditional capacity controls, full coverage denominators and persistent five-view reload evidence. No validation-selected hyperparameters or scene promotion.')
    write(output/'plan.json',plan);write(output/'source_list.json',dict(roles=roles,bindings=sources))
    if git('rev-parse','HEAD').decode().strip()!=head or(idx.read_bytes()if idx.exists()else None)!=original_index:raise RuntimeError('active HEAD/index changed')
    write(output/'prepared.json',dict(completed=True,parent=base,active_HEAD=head,files=len(files),shared_index_unchanged=True,active_tree_modified=False,pushed=False))
    print(json.dumps(dict(output=str(output),parent=base,files=len(files))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--selection',required=True);p.add_argument('--templates',required=True)
    a=p.parse_args();prepare(a.output,a.selection,a.templates)
