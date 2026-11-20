"""归档原字节与编译验证；复用本波已完成的实际CPU/GPU检查，不重跑实验。"""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def main(path):
 p=json.loads(Path(path).read_text());out=Path(p['output']);env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_INDEX_FILE=str(out/'publication.index'),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 def git(*a):return subprocess.check_output(['git',*a],cwd=R,env=env)
 tree=git('write-tree').decode().strip();dest=out/'validated_archive';dest.mkdir(exist_ok=False);checks={}
 for rel in p['files']:
  data=git('show',tree+':'+rel);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  checks[rel]=hashlib.sha256(data).hexdigest()==p['override_sha256'][rel]
  binding=p['original_source_bindings'].get(rel)
  if binding:checks[rel]&=hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest()==binding['sha256']
 assert all(checks.values())
 sources=[str(x)for x in dest.rglob('*.py')]
 subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python','-m','py_compile',*sources],env=env,check=True)
 git('-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check',p['expected_base'])
 result=dict(passed=True,tree=tree,source_byte_checks=len(checks),compiled_python_sources=len(sources),
  validation_mode='byte-exact archive of already CPU/GPU validated run sources; compilation and staged whitespace check',
  scientific_tests_rerun=0,new_experiments=0,GPU_operations=0,quality_PASS=False,shared_index_used=False)
 (out/'source_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
