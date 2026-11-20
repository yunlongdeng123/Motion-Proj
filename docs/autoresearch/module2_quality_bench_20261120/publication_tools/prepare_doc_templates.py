"""从实际发布父和当前脏文档准备三文档候选；不写活动仓库。"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess

R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
BRANCH='research/worldsim-v8.2-drivingeditor-v44'
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',CUDA_VISIBLE_DEVICES='')
MARKER='`MODULE2-TEACHER-DETAIL-GPU-20261119`'


def git(*args):return subprocess.check_output(['git',*args],cwd=R,env=ENV)
def sha(data):return hashlib.sha256(data).hexdigest()


def replace_3d(data,body):
    text=data.decode('utf-8')
    if text.count(MARKER)<1:raise ValueError('expected current 3D snapshot missing')
    start=text.index(MARKER)
    end=text.find('\n## ',start)
    end=len(text)if end<0 else end
    result=text[:start]+body.rstrip()+'\n\n'+text[end:]
    result=result.replace('更新：2026-11-19（Asia/Singapore）。','更新：2026-11-20（Asia/Singapore）。',1)
    for paragraph in text.split('\n\n'):
        if 'ROAD-2D-TEACHER-'in paragraph and paragraph not in result:raise ValueError('parallel status changed')
    return result.encode('utf-8')


def main(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('preserve existing template snapshot')
    base=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]
    parent=git('show',base+':docs/RESEARCH_STATUS.md')
    current=(R/'docs/RESEARCH_STATUS.md').read_bytes()
    body=(source/'status_3d_snapshot.md').read_text(encoding='utf-8')
    output.mkdir(parents=True)
    (output/'RESEARCH_STATUS.md').write_bytes(replace_3d(parent,body))
    (output/'RESEARCH_STATUS_working.md').write_bytes(replace_3d(current,body))
    for name in ['experiments_append.md','failure_append.md']:
        (output/name).write_bytes((source/name).read_bytes())
    before={rel:sha((R/rel).read_bytes())for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V82-F06.md']}
    receipt=dict(actual_remote_parent=base,working_before_sha256=before,
        published_status_only_replaced_3D_snapshot=True,working_status_parallel_2D_preserved=True,
        append_only_working_EXPERIMENTS_and_F06=True,active_files_written=False)
    (output/'doc_source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(templates=str(output),parent=base,active_files_written=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
