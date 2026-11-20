"""只在本轮campaign准备选定覆盖树；不写活动工作树/index/分支。"""
from pathlib import Path
import argparse,datetime,difflib,hashlib,json,os,shutil,subprocess

R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1')
BRANCH='research/worldsim-v8.2-drivingeditor-v44'
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
         OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
HEADER=b'from pathlib import Path as _MetricPath\nimport sys as _metric_sys\n_metric_sys.path.insert(0, str(_MetricPath(__file__).resolve().parents[2] / "scripts/worldsim_v82/drivingeditor_dynamic_edit"))\n\n'


def git(*args):return subprocess.check_output(['git',*args],cwd=R,env=ENV)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def append(before,addition):
    if addition.strip()in before:raise ValueError('document addition already present')
    sep=b''if before.endswith(b'\n\n')else b'\n'if before.endswith(b'\n')else b'\n\n'
    return before+sep+addition


def prepare(output,selection,templates,canonical_frozen):
    if not canonical_frozen:raise RuntimeError('root final canonical freeze required')
    output,selection=Path(output),Path(selection);s=json.loads(selection.read_text());D=s['archive_root']
    if output.exists():raise FileExistsError('preserve previous prepared snapshot')
    head=git('rev-parse','HEAD').decode().strip()
    idx=Path(git('rev-parse','--git-path','index').decode().strip());idx=idx if idx.is_absolute()else R/idx
    before_index=idx.read_bytes()if idx.exists()else None
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().splitlines()
    if len(remote)!=1:raise RuntimeError('ambiguous origin head')
    base=remote[0].split()[0]
    try:git('cat-file','-e',base+'^{commit}')
    except subprocess.CalledProcessError:raise RuntimeError('root must fetch current remote object first; no implicit fetch/branch mutation')
    output.mkdir(parents=True);tree=output/'tree';tree.mkdir();files=[];sources={};code={};roles={}
    limit=s['maximum_selected_file_bytes']
    def copy(rel,source,role):
        source=Path(source)
        if not source.is_file()or source.stat().st_size>limit:raise ValueError('missing/oversized selected source: '+str(source))
        if source.suffix not in('.py','.json','.md','.svg','.txt'):raise ValueError('source type not selected')
        p=tree/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes())
        files.append(rel);sources[rel]=dict(path=str(source),sha256=sha(source));roles[rel]=role
        if source.suffix=='.py':code[rel]=str(source)
    for name in s['canonical']:copy(D+'/'+name,B/name,'root_frozen_canonical')
    for p in sorted(B.rglob('*.py')):
        rel=p.relative_to(B)
        if any(x in rel.parts for x in ('publication','__pycache__')):continue
        copy(D+'/run_code/'+rel.as_posix(),p,'actual_or_engineering_source_snapshot')
    for name in s['small_evidence']:copy(D+'/evidence/'+name,B/name,'explicit_small_evidence')
    for name in s['small_documents']:copy(D+'/source_docs/'+name,B/name,'API_or_source_document')
    tools=selection.parent
    for p in sorted(tools.iterdir()):
        if p.is_file()and p.suffix in('.py','.md','.json'):copy(D+'/publication_tools/'+p.name,p,'publication_tool_or_plan')
    target=s['public_reusable_module'];frozen=B/'metrics/road_detail_metrics.py'
    freeze=json.loads((B/'metric_freeze.json').read_text())
    if sha(frozen)!=freeze['source_sha256']['metrics/road_detail_metrics.py']:raise ValueError('metric formula changed')
    if (R/target).exists()and (R/target).read_bytes()!=frozen.read_bytes():raise ValueError('unknown reusable target collision')
    copy(target,frozen,'byte_exact_reusable_metric')
    test=s['public_test'];original=B/'metrics/test_road_detail_metrics.py';data=HEADER+original.read_bytes()
    if (R/test).exists()and (R/test).read_bytes()!=data:raise ValueError('unknown reusable test collision')
    p=tree/test;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);files.append(test);roles[test]='test_import_prefix_only'
    adaptations={test:dict(source=str(original),source_sha256=sha(original),prefix_utf8=HEADER.decode(),
        original_test_body_bytes_preserved=True,reason='direct module import avoids existing scene package initialization')}
    readme=('# 道路细节benchmark来源\n\n[报告与组件](report.md) · [结果](results.json) · [协议](protocol.json)。\n\n'
        '本轮重新评价已冻结资产；不由指标更新推断训练收益、盲测结果、标线真值或表示上限。'
        '原生1920与480像素指标、稳定道路颜色与legacy预乘颜色分别保留。\n\n'
        'run_code保存本轮实际Python源原字节；唯一可复用模块复制到scripts/worldsim_v82/drivingeditor_dynamic_edit/road_detail_metrics.py。'
        'CPU消费者请将该目录放入PYTHONPATH后直接import road_detail_metrics，避免现有package initializer加载场景依赖；未修改任何默认scene/initializer。'
        '测试只添加路径前导，原13个测试正文未改，另2个HF伴随测试独立验证。\n\n'
        '只选明确小收据；33MB lowres/full tile JSON、11MB native摘要、所有NPZ/PT/图像/缓存保留远端：`'+str(B)+'`。'
        '复现入口为run_code/root_code，真实HF/edge指标和梯度诊断各有冻结输入、单位与分母。'
        '原训练属于先前研究，本轮CPU求导无参数步；任何原生读取调用由实收据单列。\n')
    p=tree/D/'README.md';p.write_text(readme,encoding='utf-8');files.append(D+'/README.md');roles[D+'/README.md']='source_navigation'
    work={};patch=[];nav=[]
    if templates is not None:
        templates=Path(templates)
        doc_binding=json.loads((templates/'doc_source_receipt.json').read_text())
        if doc_binding['actual_remote_parent']!=base:raise RuntimeError('document parent changed; regenerate templates')
        for rel,expected in doc_binding['working_before_sha256'].items():
            if sha(R/rel)!=expected:raise RuntimeError('working document changed; regenerate scoped templates: '+rel)
        for rel,name in [('docs/RESEARCH_STATUS.md','RESEARCH_STATUS.md'),('docs/EXPERIMENTS.md','experiments_append.md'),('docs/research_failures/entries/V82-F06.md','failure_append.md')]:
            before=(R/rel).read_bytes();parent=git('show',base+':'+rel);addition=(templates/name).read_bytes()
            if rel=='docs/RESEARCH_STATUS.md':
                pub=addition;candidate=(templates/'RESEARCH_STATUS_working.md').read_bytes()
                for paragraph in before.decode().split('\n\n'):
                    if 'ROAD-2D-TEACHER-'in paragraph and paragraph not in candidate.decode():raise ValueError('parallel 2D status lost')
            else:pub=append(parent,addition);candidate=append(before,addition)
            p=tree/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(pub);files.append(rel);roles[rel]='root_authored_doc'
            for folder,data in [('working_before',before),('working_docs',candidate)]:
                p=output/folder/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            work[rel]=dict(before_sha256=sha(R/rel),candidate=str(output/'working_docs'/rel))
            patch.extend(difflib.unified_diff(parent.decode().splitlines(True),pub.decode().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
        scratch=output/'failure_index_scratch';names=['scripts/build_research_failure_index.py','scripts/query_research_failures.py','docs/research_failures/history_index.json']
        names+=git('ls-tree','-r','--name-only',base,'--','docs/research_failures/entries').decode().splitlines()
        for rel in names:
            p=scratch/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(git('show',base+':'+rel))
        shutil.copy2(tree/'docs/research_failures/entries/V82-F06.md',scratch/'docs/research_failures/entries/V82-F06.md')
        log=subprocess.check_output(['/root/autodl-tmp/envs/motionproj/bin/python','-B',str(scratch/'scripts/build_research_failure_index.py')],env=ENV,text=True)
        for p in sorted((scratch/'docs/research_failures').rglob('*.md')):
            rel=p.relative_to(scratch).as_posix()
            if '/entries/'in rel:continue
            try:old=git('show',base+':'+rel)
            except subprocess.CalledProcessError:old=b''
            if p.read_bytes()!=old:copy(rel,p,'scratch_generated_failure_navigation');nav.append(rel)
        write(output/'failure_index_regeneration.json',dict(log=log,changed=nav,active_worktree_modified=False))
    (output/'publication_docs.patch').write_text(''.join(patch),encoding='utf-8')
    files=sorted(set(files))
    plan=dict(output=str(output/'stage'),branch=BRANCH,expected_base=base,expected_active_head=head,
        files=files,source_overrides={f:str(tree/f)for f in files},override_sha256={f:sha(tree/f)for f in files},
        original_source_bindings=sources,executed_code_bindings=code,adaptations=adaptations,archive_root=D,
        tests=[test,D+'/run_code/gradient_probe/code/test_hf_gradient_probe.py'],expected_tests=15,
        test_groups=[dict(name='metric_integration',tests=[test],expected=13,forbid_torch=True),
            dict(name='HF_gradient_math',tests=[D+'/run_code/gradient_probe/code/test_hf_gradient_probe.py'],expected=2,forbid_torch=False)],
        import_paths=['scripts/worldsim_v82/drivingeditor_dynamic_edit',D+'/run_code/gradient_probe/code'],
        working_docs=work,parallel_status_preserved_in_working_candidate=bool(templates),
        message='feat(road): add domain-safe detail benchmark and native quality diagnostics\n\n'
                'Preserve frozen same-resolution teacher controls, explicit edge and high-frequency correspondence, CPU gradient alignment and full-denominator failure evidence without promoting the scene.')
    write(output/'plan.json',plan);write(output/'source_list.json',dict(roles=roles,bindings=sources,adaptations=adaptations))
    if git('rev-parse','HEAD').decode().strip()!=head or(idx.read_bytes()if idx.exists()else None)!=before_index:raise RuntimeError('active HEAD/index changed')
    write(output/'prepared.json',dict(completed=True,actual_remote_parent=base,active_HEAD=head,files=len(files),
        shared_index_unchanged=True,active_worktree_modified=False,private_index_created=False,pushed=False,
        canonical={n:sha(B/n)for n in s['canonical']},failure_navigation_changed=nav))
    print(json.dumps(dict(output=str(output),files=len(files),parent=base,tests=15,private_index_created=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--templates');p.add_argument('--canonical-frozen',action='store_true');a=p.parse_args()
    prepare(a.output,a.selection,a.templates,a.canonical_frozen)
