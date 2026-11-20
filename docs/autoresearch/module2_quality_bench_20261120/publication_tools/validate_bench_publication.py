"""只验收private index导出的选定源；CPU测试与纯metric接入分阶段隔离。"""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
PY='/root/autodl-tmp/envs/motionproj/bin/python'

TEST_RUNNER=r'''
import sys,json,contextlib,io
from pathlib import Path
source=Path(sys.argv[1]);group=json.loads(sys.argv[2]);paths=json.loads(sys.argv[3]);output=Path(sys.argv[4])
sys.path[:0]=[str(source/p)for p in paths]
import pytest
class Count:
 def __init__(self):self.passed=set();self.failed=[]
 def pytest_runtest_logreport(self,report):
  if report.when=='call'and report.passed:self.passed.add(report.nodeid)
  if report.failed:self.failed.append(report.nodeid)
c=Count();log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q','-p','no:cacheprovider','--import-mode=importlib','--rootdir',str(source),*[str(source/t)for t in group['tests']]],plugins=[c])
modules={n:str(Path(sys.modules[n].__file__).resolve())for n in ['road_detail_metrics','hf_gradient_probe']if n in sys.modules}
isolated=all(Path(p).is_relative_to(source.resolve())for p in modules.values())
torch_imported='torch'in sys.modules
passed=code==0 and len(c.passed)==group['expected']and not c.failed and isolated and not(group['forbid_torch']and torch_imported)
output.with_suffix('.log').write_text(log.getvalue())
output.write_text(json.dumps(dict(passed=passed,tests_passed=len(c.passed),expected=group['expected'],failed=c.failed,modules=modules,isolated=isolated,torch_imported=torch_imported,CUDA_visible='',native_renderer_invoked=False,optimizer_updates=0),indent=2)+'\n')
print(log.getvalue());print(json.dumps(dict(group=group['name'],passed=passed,tests=len(c.passed),torch_imported=torch_imported)))
raise SystemExit(0 if passed else 1)
'''


def main(plan_path):
    p=json.loads(Path(plan_path).read_text());out=Path(p['output'])
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_INDEX_FILE=str(out/'publication.index'),
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    def git(*a):return subprocess.check_output(['git',*a],cwd=R,env=env)
    head=git('rev-parse','HEAD').decode().strip()
    if head!=p['expected_active_head']:raise RuntimeError('active HEAD changed')
    idx=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')).decode().strip())
    idx=idx if idx.is_absolute()else R/idx;before=idx.read_bytes()if idx.exists()else None
    tree=git('write-tree').decode().strip();source=out/('validated_source_'+tree[:12]);source.mkdir(exist_ok=False)
    for rel in p['files']:
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(git('show',tree+':'+rel))
    checks={rel:(source/rel).read_bytes()==Path(original).read_bytes()for rel,original in p['executed_code_bindings'].items()}
    adaptations={}
    for rel,row in p['adaptations'].items():
        original=Path(row['source']).read_bytes()
        adaptations[rel]=(hashlib.sha256(original).hexdigest()==row['source_sha256']and
            (source/rel).read_bytes()==row['prefix_utf8'].encode()+original)
    if not all(checks.values())or not all(adaptations.values()):raise RuntimeError('byte binding/adaptation mismatch')
    groups={}
    for group in p['test_groups']:
        target=out/('source_validation_'+group['name']+'.json')
        subprocess.run([PY,'-B','-c',TEST_RUNNER,str(source),json.dumps(group),json.dumps(p['import_paths']),str(target)],
            cwd=source,env=env,check=True)
        groups[group['name']]=json.loads(target.read_text())
    paths=sorted(str(x)for x in source.rglob('*.py'));compiled={}
    for executable in [PY,'/root/autodl-tmp/envs/drivingeditor/bin/python']:
        subprocess.run([executable,'-m','py_compile',*paths],cwd=source,env=env,check=True);compiled[executable]=len(paths)
    if (idx.read_bytes()if idx.exists()else None)!=before or git('rev-parse','HEAD').decode().strip()!=head:raise RuntimeError('active HEAD/index changed')
    passed=all(v['passed']for v in groups.values())and sum(v['tests_passed']for v in groups.values())==p['expected_tests']
    receipt=dict(passed=passed,tree=tree,source_export=str(source),exact_source_bytes=checks,
        test_import_prefix_only_adaptations=adaptations,test_groups=groups,compiled=compiled,
        tests_passed=sum(v['tests_passed']for v in groups.values()),active_HEAD_unchanged=True,
        shared_index_unchanged=True,native_calls=0,GPU_operations=0,scene_optimizer_updates=0)
    (out/'source_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,tree=tree,tests=receipt['tests_passed'],source_files=len(checks))))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main(sys.argv[1])
