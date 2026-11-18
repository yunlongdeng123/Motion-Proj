"""选定覆盖树发布：默认只暂存；真实index/活动HEAD始终保持。"""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(plan_path):
 p=json.loads(Path(plan_path).read_text());out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
 def git(*a,input=None,env=None):return subprocess.check_output(['git',*a],cwd=R,input=input,env=env)
 head=git('rev-parse','HEAD').decode().strip()
 if head!=p['expected_active_head']:raise RuntimeError('active HEAD changed')
 index=Path(git('rev-parse','--git-path','index').decode().strip())
 if not index.is_absolute():index=R/index
 original=index.read_bytes() if index.exists() else None
 try:
  git('fetch','origin','refs/heads/'+p['branch'])
  base=git('rev-parse','FETCH_HEAD').decode().strip()
  if base!=p['expected_base']:
   remote_changed=git('diff','--name-only',p['expected_base'],base).decode().splitlines()
   (out/'parent_changed.json').write_text(json.dumps(dict(expected=p['expected_base'],actual=base,remote_changed=remote_changed,overlapping_selected=sorted(set(remote_changed)&set(p['files'])),action='stop; root reviews overlap; no reset or merge'),indent=2)+'\n')
   raise RuntimeError('publication parent changed: '+base+'; see parent_changed.json')
  env=dict(os.environ,GIT_INDEX_FILE=str(out/'publication.index'));git('read-tree',base,env=env)
  for name in p['files']:
   source=Path(p['source_overrides'][name])
   if name.startswith('/') or '..' in Path(name).parts or sha(source)!=p['override_sha256'][name]:raise ValueError('invalid/changed override:'+name)
   origin=p['original_source_bindings'].get(name)
   if origin and sha(origin['path'])!=origin['sha256']:raise ValueError('canonical/source changed after preparation:'+name)
   blob=git('hash-object','-w','--stdin',input=source.read_bytes()).decode().strip()
   git('update-index','--add','--cacheinfo','100644,'+blob+','+name,env=env)
  changed=git('diff','--cached','--name-only',base,env=env).decode().splitlines()
  if not changed or set(changed)-set(p['files']):raise RuntimeError('unexpected publication paths')
  git('-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check',base,env=env)
  (out/'staged.diff').write_bytes(git('diff','--cached',base,env=env))
  (out/'staged_stat.txt').write_bytes(git('diff','--cached','--stat',base,env=env))
  tree=git('write-tree',env=env).decode().strip()
  (out/'stage_receipt.json').write_text(json.dumps(dict(staged=True,tree=tree,parent=base,selected_files=changed,active_HEAD=head,shared_index_used=False,commit_created=False,pushed=False),indent=2)+'\n')
  if '--commit' not in sys.argv:
   print((out/'staged_stat.txt').read_text());print(json.dumps(dict(tree=tree,stage_only=True)));return
  validation=json.loads((out/'source_validation.json').read_text())
  if validation.get('passed') is not True or validation['tree']!=tree:raise RuntimeError('exact staged tree not validated')
  commit=git('commit-tree',tree,'-p',base,input=(p['message'].rstrip()+'\n').encode()).decode().strip()
  git('push','origin',commit+':refs/heads/'+p['branch'])
  remote=git('ls-remote','origin','refs/heads/'+p['branch']).decode().split()[0]
  if remote!=commit:raise RuntimeError('pushed head mismatch')
  receipt=dict(pushed=True,branch=p['branch'],parent=base,commit=commit,verified_remote_head=remote,active_HEAD_unchanged=True,active_index_unchanged=True,
    selected_files=changed,data_models_media_published=False,URL='https://github.com/yunlongdeng123/Motion-Proj/commit/'+commit)
  (out/'publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
 finally:
  if git('rev-parse','HEAD').decode().strip()!=head or (index.read_bytes() if index.exists() else None)!=original:
   raise RuntimeError('active HEAD/index changed')
if __name__=='__main__':main(sys.argv[1])

