"""确认远端分支与提交内容，并生成可取回的轻量归档 zip。"""
from pathlib import Path
import argparse
import zipfile
from archive_common import Git, dump_json, read_json, digest, sha, valid_destination


def verify(plan_path, proxy):
    p = read_json(plan_path)
    git = Git(p['repo'], proxy)
    git.assert_snapshot(p['shared_snapshot'])
    before = git.snapshot()
    out = Path(p['output'])
    try:
        receipt = read_json(out / 'publication_receipt.json')
        commit = receipt['commit']
        if git.remote_head(p['branch']) != commit:
            raise ValueError('当前远端分支 head 已变化，需要审阅')
        if git.call('rev-parse', commit + '^').decode().strip() != p['expected_base']:
            raise ValueError('提交 parent 不符')
        changed = git.call('diff', '--name-only', commit + '^', commit).decode().splitlines()
        if set(changed) - set(p['files']):
            raise ValueError('提交包含未授权路径')
        blobs = {}
        for name in p['files']:
            valid_destination(name, allow_current_docs=True, allow_link_updates=True)
            data = git.call('show', commit + ':' + name)
            if digest(data) != p['override_sha256'][name]:
                raise ValueError('已提交内容与冻结源不符：' + name)
            blobs[name] = data
        for name, oid in p['protected_objects'].items():
            if git.call('rev-parse', commit + ':' + name).decode().strip() != oid:
                raise ValueError('AGENTS 或二维教师归档被改变：' + name)
        git.assert_snapshot(before)
        result = dict(verified=True, commit=commit, remote_head=commit, parent=p['expected_base'],
                      selected_files=len(blobs), selected_bytes=sum(map(len, blobs.values())),
                      active_HEAD_and_index_unchanged=True, dirty_worktree_preserved=True,
                      protected_2d_and_agents_unchanged=True,
                      URL=receipt['URL'])
        dump_json(out / 'final_verification.json', result)
        zip_path = out / 'publication_takeback.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in blobs.items():
                z.writestr('committed/' + name, data)
            for name in ('publication_receipt.json', 'source_validation.json', 'stage_receipt.json', 'final_verification.json'):
                z.write(out / name, 'publication/' + name)
        dump_json(out / 'takeback_receipt.json', dict(path=str(zip_path), sha256=sha(zip_path),
                  bytes=zip_path.stat().st_size, committed_source_sha256=p['override_sha256'],
                  model_raw_media_or_external_fullpaper_included=False))
        print(zip_path)
    finally:
        git.assert_snapshot(before)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--proxy', required=True)
    args = parser.parse_args()
    verify(args.plan, args.proxy)
