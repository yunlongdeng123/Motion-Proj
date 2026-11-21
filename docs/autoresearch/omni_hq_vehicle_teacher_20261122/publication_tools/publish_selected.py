"""独立 index 发布；默认只暂存。--commit 仅在 root 最终完成信号后调用。"""
from pathlib import Path
import argparse
from archive_common import (Git, dump_json, read_json, sha, check_frozen_sources, private_index)


def publish(plan_path, proxy, commit_allowed=False):
    plan_path = Path(plan_path).resolve()
    p = read_json(plan_path)
    git = Git(p['repo'], proxy)
    git.assert_snapshot(p['shared_snapshot'])
    before = git.snapshot()
    out = Path(p['output'])
    out.mkdir(parents=True, exist_ok=True)
    index = private_index(p)
    try:
        if sha(p['input_manifest']) != p['input_manifest_sha256']:
            raise ValueError('最终清单发生变化，须重新准备')
        manifest = read_json(p['input_manifest'])
        if not (manifest.get('root_report_complete') is True and manifest.get('publication_ready') is True):
            raise ValueError('root 完成门未通过')
        actual = git.remote_head(p['branch'])
        if actual != p['expected_base']:
            dump_json(out / 'parent_changed.json', dict(expected=p['expected_base'], actual=actual,
                      action='停止；基于新 parent 重新审阅；不 reset/merge/force-push'))
            raise RuntimeError('发布 parent 已变化，当前未提交或推送')
        check_frozen_sources(p)
        git.call('read-tree', p['expected_base'], index=index)
        for name in p['files']:
            blob = git.call('hash-object', '-w', '--stdin', input=Path(p['source_overrides'][name]).read_bytes()).decode().strip()
            git.call('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + name, index=index)
        changed = git.call('diff', '--cached', '--name-only', p['expected_base'], index=index).decode().splitlines()
        if not changed or set(changed) - set(p['files']):
            raise RuntimeError('暂存 diff 范围异常')
        git.call('-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof',
                 'diff', '--cached', '--check', p['expected_base'], index=index)
        (out / 'staged.diff').write_bytes(git.call('diff', '--cached', p['expected_base'], index=index))
        (out / 'staged_stat.txt').write_bytes(git.call('diff', '--cached', '--stat', p['expected_base'], index=index))
        tree = git.call('write-tree', index=index).decode().strip()
        dump_json(out / 'stage_receipt.json', dict(staged=True, tree=tree, parent=p['expected_base'],
                  selected_files=changed, active_HEAD=before['active_HEAD'], shared_index_used=False,
                  commit_created=False, pushed=False))
        if not commit_allowed:
            print((out / 'staged_stat.txt').read_text())
            print('仅暂存：' + tree)
            return
        validation = read_json(out / 'source_validation.json')
        if validation.get('passed') is not True or validation.get('tree') != tree:
            raise RuntimeError('当前精确暂存树尚未通过源码验收')
        git.assert_snapshot(before)
        commit = git.call('commit-tree', tree, '-p', p['expected_base'],
                          input=(p['message'].rstrip() + '\n').encode()).decode().strip()
        dump_json(out / 'commit_receipt.json', dict(commit=commit, parent=p['expected_base'], pushed=False))
        git.call('push', 'origin', commit + ':refs/heads/' + p['branch'])
        remote = git.remote_head(p['branch'])
        if remote != commit:
            raise RuntimeError('推送后远端 head 不符')
        git.assert_snapshot(before)
        receipt = dict(pushed=True, branch=p['branch'], parent=p['expected_base'], commit=commit,
                       verified_remote_head=remote, active_HEAD_unchanged=True, active_index_unchanged=True,
                       existing_dirty_files_preserved=True, selected_files=changed,
                       model_raw_media_or_external_fullpaper_included=False,
                       URL='https://github.com/yunlongdeng123/Motion-Proj/commit/' + commit)
        dump_json(out / 'publication_receipt.json', receipt)
        print(receipt['URL'])
    finally:
        git.assert_snapshot(before)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--proxy', required=True)
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args()
    publish(args.plan, args.proxy, args.commit)
