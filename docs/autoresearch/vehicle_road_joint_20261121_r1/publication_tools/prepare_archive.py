"""按 root 最终确认的显式文件清单准备覆盖树，不写共享文档/index/HEAD。"""
from pathlib import Path
import argparse
import re
import subprocess
import sys
from archive_common import (ARCHIVE, BRANCH, DEFAULT_REPO, ALLOWED_ROLES, Git,
                            dump_json, read_json, digest, sha, lightweight_text, valid_destination)


def replace_status(old, spec, report_date):
    marker = spec['replace_start_marker']
    content = Path(spec['source']).read_text(encoding='utf-8').rstrip()
    if old.count(marker) != 1 or content.count(spec['new_task_marker']) != 1:
        raise ValueError('当前三维快照锚点或新任务标识不唯一')
    start = old.index(marker)
    end_marker = spec.get('replace_end_marker', '\n## ')
    end = old.find(end_marker, start + len(marker)) if end_marker else -1
    if end < 0:
        end = len(old)
    result = old[:start] + content + '\n' + old[end:]
    result = re.sub(r'(?m)^更新：[^\n]*$', '更新：' + report_date + '（Asia/Singapore）。', result, count=1)
    for para in old.split('\n\n'):
        if 'ROAD-2D-TEACHER-' in para and para not in result:
            raise ValueError('准备结果会更改并行二维教师快照')
    return result.encode('utf-8')


def prepare(manifest_path, output, proxy):
    manifest_path = Path(manifest_path).resolve()
    m = read_json(manifest_path)
    if not (m.get('publication_ready') is True and m.get('root_report_complete') is True):
        raise ValueError('等待 root 确认报告完成与最终 sourcefiles；当前只可准备工具')
    if m.get('archive_root') != ARCHIVE or m.get('branch') != BRANCH:
        raise ValueError('本轮归档根目录或目标分支不符')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('使用新归档目录，保留旧现场：' + str(output))
    git = Git(m.get('repo', DEFAULT_REPO), proxy)
    before = git.snapshot()
    try:
        base = git.remote_head(BRANCH)
        git.ensure_commit(base)
        existing = git.call('ls-tree', '-r', '--name-only', base, '--', ARCHIVE).decode().splitlines()
        if existing and not m.get('allow_existing_archive', False):
            raise ValueError('r1 归档已存在，需 root 明确选择后续修订')
        selected = {}
        payload = {}
        bindings = {}
        for rec in m['sources']:
            name = valid_destination(rec['destination'])
            if name in selected or name == ARCHIVE + '/source_manifest.json':
                raise ValueError('重复或保留的目标路径：' + name)
            if rec['role'] not in ALLOWED_ROLES:
                raise ValueError('非法归档角色；外部论文全文不可归档')
            source = Path(rec['source']).resolve()
            data = lightweight_text(source)
            payload[name] = data
            selected[name] = dict(path=str(source), sha256=digest(data), bytes=len(data), role=rec['role'])
            origin = rec.get('original_source', str(source))
            if sha(origin) != digest(data):
                raise ValueError('实际canonical来源与冻结副本不符：' + name)
            bindings[name] = dict(path=origin, sha256=digest(data))
            if rec.get('local_origin'):
                selected[name]['local_origin'] = rec['local_origin']
            selected[name]['original_source'] = origin
        if not selected:
            raise ValueError('最终显式 sourcefiles 为空')
        if not any(p.endswith('/architecture.svg') for p in selected):
            raise ValueError('技术报告归档缺 architecture.svg')
        if not any(r['role'] == 'report_source' for r in selected.values()):
            raise ValueError('技术报告源未进入最终清单')
        spec = m['status_update']
        status_before = git.call('show', base + ':docs/RESEARCH_STATUS.md').decode('utf-8')
        payload['docs/RESEARCH_STATUS.md'] = replace_status(status_before, spec, m['report_date'])
        bindings['docs/RESEARCH_STATUS.md'] = dict(path=str(Path(spec['source']).resolve()), sha256=sha(spec['source']))
        experiment = Path(m['experiments_append']['source']).resolve()
        addition = lightweight_text(experiment).strip()
        prior = git.call('show', base + ':docs/EXPERIMENTS.md')
        marker = m['experiments_append']['task_marker'].encode()
        if marker in prior or marker not in addition:
            raise ValueError('实验索引任务标识重复或缺失')
        payload['docs/EXPERIMENTS.md'] = prior.rstrip() + b'\n\n' + addition + b'\n'
        bindings['docs/EXPERIMENTS.md'] = dict(path=str(experiment), sha256=sha(experiment))
        output.mkdir(parents=True)
        if m.get('failure_append'):
            f = m['failure_append']
            rel = 'docs/research_failures/entries/V82-F06.md'
            old_failure = git.call('show', base + ':' + rel)
            extra = lightweight_text(f['source']).strip()
            if f['task_marker'].encode() in old_failure or f['task_marker'].encode() not in extra:
                raise ValueError('既有failure卡任务标识重复或缺失')
            payload[rel] = old_failure.rstrip() + b'\n\n' + extra + b'\n'
            bindings[rel] = dict(path=str(Path(f['source']).resolve()), sha256=sha(f['source']))
            scratch = output / 'failure_index_scratch'
            needed = ['scripts/build_research_failure_index.py','scripts/query_research_failures.py',
                      'docs/research_failures/history_index.json']
            needed += git.call('ls-tree','-r','--name-only',base,'--','docs/research_failures/entries').decode().splitlines()
            for name in needed:
                q=scratch/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(git.call('show',base+':'+name))
            (scratch/rel).write_bytes(payload[rel])
            subprocess.run([sys.executable,'-B',str(scratch/'scripts/build_research_failure_index.py')],
                           cwd=scratch,env=git.env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            # 保留既有标题/metadata时导航应相同；额外变更必须先人工核定。
            changed_indexes=[]
            for q in (scratch/'docs/research_failures').rglob('*'):
                if q.is_file() and q.suffix in ('.md','.json') and '/entries/' not in q.as_posix() and q.name!='history_index.json':
                    name=q.relative_to(scratch).as_posix()
                    if q.read_bytes()!=git.call('show',base+':'+name):changed_indexes.append(name)
            if changed_indexes:
                raise ValueError('failure索引重建出现额外变更，停止审阅：'+', '.join(changed_indexes))
        tree = output / 'tree'
        overrides = {}
        for name, data in payload.items():
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            overrides[name] = str(target)
        source_manifest = tree / ARCHIVE / 'source_manifest.json'
        dump_json(source_manifest, dict(schema='vehicle_road_joint_lightweight_archive.v1',
                  task_id=m['task_id'], run_id=m['run_id'], archive_only=True,
                  model_raw_media_or_external_fullpaper_included=False,
                  source_files=selected, method_verdict_is_in_report=True))
        overrides[ARCHIVE + '/source_manifest.json'] = str(source_manifest)
        plan = dict(repo=str(git.repo), output=str(output / 'stage'), branch=BRANCH,
                    expected_base=base, expected_active_head=before['active_HEAD'],
                    shared_snapshot=before, files=sorted(overrides), source_overrides=overrides,
                    override_sha256={p:sha(s) for p,s in overrides.items()},
                    original_source_bindings=bindings, protected_objects=git.protected_objects(base),
                    input_manifest=str(manifest_path), input_manifest_sha256=sha(manifest_path),
                    message=m['commit_message'])
        dump_json(output / 'plan.json', plan)
        dump_json(output / 'selected_sources.json', selected)
        dump_json(output / 'prepared.json', dict(prepared=True, commit_created=False, pushed=False,
                  parent=base, selected_files=len(overrides), selected_bytes=sum(len(x) for x in payload.values()),
                  shared_worktree_modified=False, active_HEAD=before['active_HEAD']))
        print(output / 'plan.json')
    finally:
        git.assert_snapshot(before)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--proxy', required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.proxy)
