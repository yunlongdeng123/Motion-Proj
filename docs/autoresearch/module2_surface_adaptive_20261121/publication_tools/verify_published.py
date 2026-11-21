"""Verify remote archive, parallel 2D paths, and untouched shared work/index."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(plan_path,proxy):
    p=read(plan_path);out=Path(p['output']);prepared=read(out.parent/'prepared.json');receipt=read(out/'publication_receipt.json')
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',HTTP_PROXY=proxy,HTTPS_PROXY=proxy,ALL_PROXY=proxy)
    def git(*args):return subprocess.check_output(['git',*args],cwd=R,env=env,timeout=90)
    commit=receipt['commit'];base=p['expected_base']
    assert git('ls-remote','origin','refs/heads/'+p['branch']).decode().split()[0]==commit
    assert git('show','-s','--format=%P',commit).decode().strip()==base
    changed=git('diff','--name-only',base,commit).decode().splitlines()
    assert set(changed)<=set(p['files']) and changed
    assert all(Path(n).suffix not in ['.png','.jpg','.webp','.npz','.pt','.pth','.bin']for n in changed)
    for rel in p['files']:
        assert hashlib.sha256(git('show',commit+':'+rel)).hexdigest()==p['override_sha256'][rel],rel
    for rel,b in p['original_source_bindings'].items():assert sha(b['path'])==b['sha256'],rel
    protected=git('ls-tree','-r',commit,'--','docs/autoresearch/road_2d_teacher_hf_cpu_20261119','docs/autoresearch/road_2d_teacher_hf_gpu_20261119','docs/autoresearch/road_2d_teacher_prompt_20261120','AGENTS.md').decode().splitlines()
    assert protected==read(out.parent/'protected_2d_tree.json')
    assert git('rev-parse','HEAD').decode().strip()==prepared['active_HEAD']
    index=R/git('rev-parse','--git-path','index').decode().strip()
    assert sha(index)==prepared['shared_index_sha256']
    assert all(sha(R/rel)==h for rel,h in prepared['working_dirty_sha256'].items())
    assert all(sha(R/rel)==h for rel,h in prepared['working_2d_sha256'].items())
    result=dict(schema='two_run_publication_verified.v1',passed=True,commit=commit,parent=base,branch=p['branch'],URL=receipt['URL'],remote_head_verified=True,remote_parent_nonselected_paths_preserved=True,protected_2d_and_AGENTS_byte_exact=True,protected_path_count=len(protected),parallel_2d_worktree_files_unchanged=True,parallel_2d_worktree_file_count=len(prepared['working_2d_sha256']),active_HEAD_unchanged=True,shared_index_unchanged=True,tracked_dirty_files_unchanged=True,archive_files=len(p['files']),changed_files=len(changed),models_raw_media_published=False,quality_PASS=False,new_experiments=0,GPU_operations=0)
    (out/'final_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('plan');a.add_argument('--proxy',required=True);s=a.parse_args();main(s.plan,s.proxy)
