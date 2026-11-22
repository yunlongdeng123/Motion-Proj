"""Actually rebuild origin failure navigation in an isolated snapshot; no index/worktree mutation."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,io,json,os,subprocess,sys,tarfile,time

CARD='docs/research_failures/entries/V82-F06.md'
EXPORT=('docs/research_failures','scripts/build_research_failure_index.py','scripts/query_research_failures.py')
def digest(data):return hashlib.sha256(data).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rebuild(parent,files,output,git,origin):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);snapshot=output/'origin_snapshot';snapshot.mkdir()
    payload=git('archive',parent,*EXPORT)
    with tarfile.open(fileobj=io.BytesIO(payload)) as package:
        for item in package:
            name=PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or '\\' in item.name or item.issym() or item.islnk() or not(item.isfile() or item.isdir()):
                raise ValueError('Unsafe origin archive member: '+item.name)
            destination=snapshot/Path(*name.parts)
            assert destination.resolve().is_relative_to(snapshot.resolve())
            if item.isdir():destination.mkdir(parents=True,exist_ok=True)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with package.extractfile(item) as stream:destination.write_bytes(stream.read())
    assert CARD in files and isinstance(files[CARD],bytes)
    (snapshot/CARD).write_bytes(files[CARD]);before=origin(parent,CARD)
    result=subprocess.run([sys.executable,'scripts/build_research_failure_index.py'],cwd=snapshot,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    (output/'builder.stdout.txt').write_bytes(result.stdout);(output/'builder.stderr.txt').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError('Isolated failure-index builder failed; snapshot/log retained')
    candidates=[snapshot/'docs/research_failures'/x for x in ['IDS.md','VERSIONS.md','ENTRIES.md']]
    candidates+=sorted((snapshot/'docs/research_failures/ids').glob('*.md'))
    candidates+=sorted((snapshot/'docs/research_failures/versions').glob('*.md'))
    changed=[];generated=[]
    for path in candidates:
        rel=path.relative_to(snapshot).as_posix();value=path.read_bytes()
        try:previous=origin(parent,rel)
        except subprocess.CalledProcessError:previous=None
        different=value!=previous
        generated.append(dict(path=rel,sha256=digest(value),bytes=len(value),differs_from_origin=different))
        if different:files[rel]=value;changed.append(rel)
    receipt=dict(schema='publication.origin_failure_index_actual_rebuild.v1',complete=True,decision='PASS',parent=parent,
        snapshot=str(snapshot),origin_export_paths=list(EXPORT),candidate_card=CARD,
        origin_card_sha256=digest(before),candidate_card_sha256=digest(files[CARD]),card_changed=before!=files[CARD],
        actual_builder_command=[sys.executable,'scripts/build_research_failure_index.py'],builder_exit_code=result.returncode,
        builder_stdout=result.stdout.decode().strip(),generated_paths=changed,generated_navigation=generated,
        generated_navigation_files=len(generated),navigation_diff_files=len(changed),no_navigation_diff=len(changed)==0,
        changed_navigation_included_in_isolated_index_payload=True,shared_checkout_written=False,shared_index_used=False,
        stage_executed=False,commit_created=False,pushed=False,GPU_calls=0)
    write(output/'receipt.json',receipt)
    return dict(parent=parent,snapshot=str(snapshot),used_shared_dirty_docs=False,generated_paths=changed,
        generated_navigation_files=len(generated),no_navigation_diff=not changed,output=receipt['builder_stdout'],
        actual_rebuild_receipt=str(output/'receipt.json'),actual_rebuild_receipt_sha256=digest((output/'receipt.json').read_bytes()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--parent',required=True)
    p.add_argument('--candidate-v82',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.resolve().is_relative_to(a.repo.resolve()),'Keep the isolated snapshot outside the shared checkout'
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0');env.pop('GIT_INDEX_FILE',None)
    def git(*args):return subprocess.check_output(['git',*args],cwd=a.repo,env=env)
    parent=git('rev-parse',a.parent+'^{commit}').decode().strip()
    index=Path(git('rev-parse','--git-path','index').decode().strip());index=index if index.is_absolute() else a.repo/index
    before=dict(HEAD=git('rev-parse','HEAD').decode().strip(),index_sha256=digest(index.read_bytes()) if index.exists() else None,status_sha256=digest(git('status','--porcelain=v1','-z')))
    files={CARD:a.candidate_v82.read_bytes()}
    result=rebuild(parent,files,a.output,git,lambda commit,path:git('show',commit+':'+path))
    after=dict(HEAD=git('rev-parse','HEAD').decode().strip(),index_sha256=digest(index.read_bytes()) if index.exists() else None,status_sha256=digest(git('status','--porcelain=v1','-z')))
    assert before==after,'Shared checkout changed during this isolated CPU run; inspect before using it'
    write(a.output/'shared_checkout_readonly_check.json',dict(complete=True,decision='PASS',before=before,after=after,shared_checkout_written=False,stage_executed=False,Git_mutations=0))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
