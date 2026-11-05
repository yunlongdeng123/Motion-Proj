"""Read-only scoped review; preserve other agents' independent staged work."""
import hashlib,json,subprocess
from pathlib import Path

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105')
DOC='docs/autoresearch/scene008_a6_insert_artifacts_20261105'
PATHS=[DOC,'docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md',
    'docs/autoresearch/scene008_a6_appearance_formal_20261105/delivery/index.html',
    'docs/autoresearch/scene008_a6_appearance_formal_20261105/delivery/file_manifest.json']
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def sha(b):return hashlib.sha256(b).hexdigest()

manifest=json.loads((REPO/DOC/'publication_manifest.json').read_text())
for x in manifest['files']:
    data=git('show',':'+DOC+'/'+x['path'])
    assert len(data)==x['bytes'] and sha(data)==x['sha256'],x['path']
assert sha(git('show',':'+DOC+'/delivery/index.html'))=='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191'
for name in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md']:
    assert git('show',':'+name)==(REPO/name).read_bytes(),'Shared document changed concurrently:'+name
old=(RUN/'publication/backups/fixed_LOD49_review/docs/RESEARCH_STATUS.md').read_bytes()
start=old.find('RCR-1支线'.encode());end=old.find('历史：'.encode(),start);other=old[start:end].rstrip(b'\r\n')
assert other in (REPO/'docs/RESEARCH_STATUS.md').read_bytes()
check=subprocess.run(['git','-C',str(REPO),'diff','--cached','--check','--',*PATHS],capture_output=True,text=True)
if check.returncode:
    (RUN/'publication/scoped_diff_check.log').write_text(check.stdout+check.stderr)
    raise RuntimeError('Scoped whitespace check failed: '+str(check.returncode)+'; see scoped_diff_check.log')
ours=git('diff','--cached','--name-only','--',*PATHS).decode().splitlines()
all_staged=git('diff','--cached','--name-only').decode().splitlines();other=[x for x in all_staged if x not in ours]
assert 'scripts/run_worldsim_v72_e3_render_bridge.py' not in all_staged
stat=git('diff','--cached','--stat','--stat-count=12','--',*PATHS).decode()
docdiff=git('diff','--cached','--','docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md').decode()
before=json.loads((REPO/DOC/'historical_navigation_receipt.json').read_text())
report=dict(status='PASS_SCOPED_STAGED_REVIEW_NO_COMMIT',head=git('rev-parse','HEAD').decode().strip(),pathspec=PATHS,
    scoped_files=len(ours),scoped_worktree_bytes=sum((REPO/x).stat().st_size for x in ours),
    exact_publication_manifest_files=len(manifest['files']),publication_index_sha256=sha(git('show',':'+DOC+'/delivery/index.html')),
    other_staged_paths=other,other_staged_count=len(other),other_staged_blob_sha256={x:sha(git('show',':'+x)) for x in other},
    unrelated_user_script_unstaged=True,unrelated_user_script_sha256=sha((REPO/'scripts/run_worldsim_v72_e3_render_bridge.py').read_bytes()),
    shared_docs_index_equals_latest_worktree=True,other_RCR1_a5_status_blocks_preserved=True,
    historical_navigation=before,diff_cached_check_passed=True,commit_or_push_performed=False,
    commit_instruction='Use git commit --only -- exact pathspec, after parent review; retain other staged blobs.')
(RUN/'publication/staged_scope_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(RUN/'publication/staged_stat.txt').write_text(stat);(RUN/'publication/shared_docs.diff').write_text(docdiff)
print(json.dumps({k:v for k,v in report.items() if k not in ['other_staged_blob_sha256','historical_navigation']},ensure_ascii=False));print(stat);print(docdiff)
