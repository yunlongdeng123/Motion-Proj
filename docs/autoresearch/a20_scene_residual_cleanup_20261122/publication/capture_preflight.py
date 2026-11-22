"""只读捕获最新v44 parent与共享状态；仅在本任务publication新目录写证据。"""
from pathlib import Path
import argparse
from archive_common import ARCHIVE, BRANCH, DEFAULT_REPO, LINK_UPDATES, Git, dump_json

TASK='A20-SCENE-RESIDUAL-CLEANUP-20261122'
RUN='a20_scene_residual_cleanup_20261122_r1'
PUB=Path('/root/autodl-tmp/drivingeditor_repro/campaign/'+RUN+'/publication')

def main(output,proxy):
    output=Path(output).resolve()
    if not output.is_relative_to(PUB) or output.exists():
        raise ValueError('仅使用本任务publication下的新目录')
    git=Git(DEFAULT_REPO,proxy)
    before=git.snapshot()
    try:
        base=git.remote_head(BRANCH);git.ensure_commit(base)
        status=git.call('show',base+':docs/RESEARCH_STATUS.md').decode('utf-8')
        marker='`A20-JOINT-RECONSTRUCTION-20261122`'
        if status.count(marker)!=1:raise ValueError('旧HQ当前状态标识不唯一，读取最新状态定位后再准备')
        paragraphs=status[status.index(marker):].strip().split('\n\n')
        selected=[paragraphs[0]]
        if len(paragraphs)>1 and paragraphs[1].startswith('免责声明：'):selected.append(paragraphs[1])
        old_fragment='\n\n'.join(selected)
        parallel=('ROAD-2D-TEACHER-','ROAD-MULTISOURCE','ROAD-MULTI-SOURCE','road_multisource')
        if any(x in old_fragment for x in parallel):raise ValueError('三维状态片段混入并行道路/二维任务')
        output.mkdir(parents=True)
        (output/'published_status_before.md').write_text(status,encoding='utf-8')
        (output/'status_replace_before.md').write_text(old_fragment+'\n',encoding='utf-8')
        for name in sorted(LINK_UPDATES):
            local=output/'prior_navigation_base'/name;local.parent.mkdir(parents=True,exist_ok=True)
            try:data=git.call('show',base+':'+name)
            except RuntimeError:data=b''
            local.write_bytes(data)
        (output/'published_failure_V82-F06_before.md').write_bytes(git.call('show',base+':docs/research_failures/entries/V82-F06.md'))
        dump_json(output/'remote_preflight.json',dict(task_id=TASK,run_id=RUN,target_branch=BRANCH,
            verified_remote_tip=base,archive_root=ARCHIVE,shared_snapshot=before,
            protected_objects=git.protected_objects(base),
            target_archive_preexists=bool(git.call('ls-tree','-r','--name-only',base,'--',ARCHIVE).strip()),
            prior_navigation_update_paths=sorted(LINK_UPDATES),
            status_replacement_strategy='Exact oldHQ task paragraph and its trailing disclaimer only; preserve all other task paragraphs',
            report_finalized=False,formal_stage=False,commit_created=False,pushed=False,
            gpu_or_power_operations=False,raw_media_or_model_content_read=False))
        print(output/'remote_preflight.json')
    finally:git.assert_snapshot(before)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--proxy',required=True);a=p.parse_args();main(a.output,a.proxy)
