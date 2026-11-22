"""本轮轻量材料的独立索引准备/暂存；无 commit 或 push 接口。"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_source_multiframe_20261122_r3')
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OUT = RUN / 'publication'
BRANCH = 'research/worldsim-v8.2-drivingeditor-v42'
PREFIX = 'docs/autoresearch/road_source_multiframe_20261122'
LABEL = 'ROAD-SOURCE-MULTIFRAME-20261122'
DOC_PATHS = ['docs/RESEARCH_STATUS.md', 'docs/EXPERIMENTS.md', 'docs/research_failures/entries/V82-F06.md']
ALLOWED_TYPES = {'.py', '.json', '.md', '.svg', '.js', '.mjs'}
R2_PARENT_COMMIT = '8d3b9bb81d2174115e82c100b131c31fcb444772'
R2_PARENT_README = 'docs/autoresearch/road_source_fidelity_20261122/README.md'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def git(*args, input=None, env=None):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    environment.pop('GIT_INDEX_FILE', None)
    if env:
        environment.update(env)
    proxy = os.environ.get('PUBLICATION_GIT_HTTP_PROXY', '')
    options = ['-c', 'http.proxy=' + proxy] if proxy else []
    try:
        return subprocess.check_output(['git', *options, *args], cwd=REPO, input=input, env=environment)
    except subprocess.CalledProcessError as error:
        if error.output:
            sys.stderr.write(error.output.decode(errors='replace'))
        raise


def checkout_snapshot():
    index_path = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index_path.is_absolute():
        index_path = REPO / index_path
    status = git('status', '--porcelain=v1', '--untracked-files=all', '-z')
    dirty_names = git('diff', '--name-only', '-z', 'HEAD').split(b'\0')
    content = {}
    for raw in dirty_names:
        if not raw:
            continue
        name = raw.decode()
        p = REPO / name
        content[name] = digest(p.read_bytes()) if p.is_file() else None
    return dict(HEAD=git('rev-parse', 'HEAD').decode().strip(), index_path=str(index_path),
                index_sha256=digest(index_path.read_bytes()) if index_path.exists() else None,
                porcelain_z_sha256=digest(status), tracked_dirty_content=content)


def verify_origin():
    ref = 'refs/heads/' + BRANCH
    expected = git('ls-remote', 'origin', ref).decode().split()[0]
    git('fetch', '--no-tags', 'origin', BRANCH)
    fetched = git('rev-parse', 'FETCH_HEAD').decode().strip()
    remote = git('ls-remote', 'origin', ref).decode().split()[0]
    if expected != fetched or fetched != remote:
        raise RuntimeError('Origin changed while fetching; repeat preparation on the new snapshot')
    git('cat-file', '-e', fetched + '^{commit}')
    git('merge-base', '--is-ancestor', R2_PARENT_COMMIT, fetched)
    git('cat-file', '-e', fetched + ':' + R2_PARENT_README)
    return fetched


def source_manifest():
    value = json.loads((OUT / 'selected_files.json').read_text(encoding='utf-8'))
    if value['docprefix'] != PREFIX or value['run_id'] != RUN.name:
        raise ValueError('Wrong publication task')
    names = [r['destination'] for r in value['files']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate publication path')
    for row in value['files']:
        p = Path(row['source'])
        target = Path(row['destination'])
        if p.is_absolute() or '..' in p.parts or target.is_absolute() or '..' in target.parts:
            raise ValueError('Paths must remain relative to current task/docprefix')
        if p.suffix not in ALLOWED_TYPES or target.suffix not in ALLOWED_TYPES:
            raise ValueError('Unexpected non-lightweight publication type: ' + str(p))
        if any(x in p.name.lower() for x in ('native_color_operator', 'complete_result_matrix', 'linear_state', 'actual_payload', 'paired_cpu_audit', 'pair_payload', '_sampling.json')):
            raise ValueError('Matrices and full result payloads belong in archive, not Git')
    return value


def inventory(selection):
    result = []
    for row in selection['files']:
        path = RUN / row['source']
        entry = dict(row, absolute_source=str(path), git_path=PREFIX + '/' + row['destination'], exists=path.is_file())
        if path.is_file():
            data = path.read_bytes()
            if len(data) > 300000:
                raise ValueError('Publication input exceeds 300KB: ' + str(path))
            entry.update(bytes=len(data), sha256=digest(data))
        result.append(entry)
    return result


def origin(parent, path):
    return git('show', parent + ':' + path)


def document_updates(parent, narrative):
    if narrative.get('finalized_by_root') is not True or narrative.get('task') != LABEL:
        raise RuntimeError('Root must finalize publication/narrative.json before staging')
    if narrative.get('failure_ledger_refs') != ['V82-F06'] or narrative.get('failure_ledger_delta') != 'updated:V82-F06':
        raise ValueError('Only the existing F06 card may be updated')
    files = {}
    status = origin(parent, DOC_PATHS[0]).decode()
    experiments = origin(parent, DOC_PATHS[1]).decode()
    failure = origin(parent, DOC_PATHS[2]).decode()
    if any(LABEL in data for data in (status, experiments, failure)):
        raise RuntimeError('This task already appears in origin; inspect existing publication')
    marker = '## ROAD-GAUSSIAN-TRANSFER-20261122'
    start = status.index(marker)
    end = status.find('\n## ', start + len(marker))
    if end < 0:
        end = len(status)
    paragraph = narrative['status_append'].strip()
    experiment = narrative['experiment_entry'].strip()
    failure_append = narrative['failure_append'].strip()
    if not all(LABEL in x for x in (paragraph, experiment, failure_append)):
        raise ValueError('Every final document addition must identify this task')
    files[DOC_PATHS[0]] = (status[:end].rstrip() + '\n\n' + paragraph + '\n' + status[end:]).encode()
    files[DOC_PATHS[1]] = (experiments.rstrip() + '\n\n' + experiment + '\n').encode()
    files[DOC_PATHS[2]] = (failure.rstrip() + '\n\n' + failure_append + '\n').encode()
    return files


def failure_index(parent, files):
    snapshot = OUT / ('origin_failure_snapshot_' + str(time.time_ns()))
    snapshot.mkdir()
    payload = git('archive', parent, 'docs/research_failures', 'scripts/build_research_failure_index.py', 'scripts/query_research_failures.py')
    with tarfile.open(fileobj=io.BytesIO(payload)) as package:
        for member in package.getmembers():
            target = (snapshot / member.name).resolve()
            if not target.is_relative_to(snapshot.resolve()) or member.issym() or member.islnk():
                raise RuntimeError('Unexpected snapshot archive member: ' + member.name)
        package.extractall(snapshot)
    (snapshot / DOC_PATHS[2]).write_bytes(files[DOC_PATHS[2]])
    log = subprocess.check_output([sys.executable, 'scripts/build_research_failure_index.py'], cwd=snapshot)
    (OUT / 'failure_index_build.log').write_bytes(log)
    candidates = [snapshot / 'docs/research_failures' / name for name in ['IDS.md', 'VERSIONS.md', 'ENTRIES.md']]
    candidates += list((snapshot / 'docs/research_failures/ids').glob('*.md'))
    candidates += list((snapshot / 'docs/research_failures/versions').glob('*.md'))
    changed = []
    for p in candidates:
        name = p.relative_to(snapshot).as_posix()
        value = p.read_bytes()
        try:
            previous = origin(parent, name)
        except subprocess.CalledProcessError:
            previous = None
        if value != previous:
            files[name] = value
            changed.append(name)
    return dict(parent=parent, snapshot=str(snapshot), used_shared_dirty_docs=False,
                generated_paths=changed, output=log.decode().strip())


def prepare(parent):
    selection = source_manifest()
    entries = inventory(selection)
    for path in DOC_PATHS:
        target = OUT / 'origin_snapshot' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(origin(parent, path))
    missing = [r['source'] for r in entries if not r['exists']]
    result = dict(prepared=True, staged=False, commit_created=False, pushed=False,
        origin_parent=parent, branch=BRANCH, docprefix=PREFIX,
        isolated_index=str(OUT / 'isolated.index'), files=entries,
        missing_final_inputs=missing, allowed_document_updates=DOC_PATHS,
        forbidden_payloads=['PNG', 'NPZ', 'dense/sparse matrices', 'full view-result matrix', 'model weights'],
        final_narrative_required=True, shared_checkout_written=False)
    write(OUT / 'prepare.json', result)
    return selection, entries, result


def stage(parent, selection, entries):
    if selection.get('finalized_by_root') is not True:
        raise RuntimeError('Root must finalize publication/selected_files.json before staging')
    missing = [r['source'] for r in entries if not r['exists']]
    if missing:
        raise RuntimeError('Final inputs are missing: ' + repr(missing))
    for entry in entries:
        if entry.get('require_ready'):
            value = json.loads(Path(entry['absolute_source']).read_text(encoding='utf-8-sig'))
            if value.get('ready') is not True:
                raise RuntimeError('Compact final input is not ready: ' + entry['source'])
    narrative = json.loads((OUT / 'narrative.json').read_text(encoding='utf-8'))
    files = {}
    for entry in entries:
        value = Path(entry['absolute_source']).read_bytes()
        if digest(value) != entry['sha256']:
            raise RuntimeError('Input changed during staging: ' + entry['source'])
        files[entry['git_path']] = value
    files.update(document_updates(parent, narrative))
    rebuilt = failure_index(parent, files)
    # 仅规范本发布的Git文本blob；不改源文件、归档字节或共享checkout。
    files = {name: (value.replace(b'\r\n', b'\n').rstrip(b'\n') + b'\n' if value else value)
             for name, value in files.items()}
    index = OUT / 'isolated.index'
    if index.exists():
        index.rename(OUT / ('isolated.index.previous.' + str(time.time_ns())))
    environment = {'GIT_INDEX_FILE': str(index)}
    git('read-tree', parent, env=environment)
    for name, value in files.items():
        blob = git('hash-object', '-w', '--stdin', input=value).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + name, env=environment)
    git('diff', '--cached', '--check', parent, env=environment)
    diff = git('diff', '--cached', parent, env=environment)
    (OUT / 'staged.diff').write_bytes(diff)
    (OUT / 'staged_stat.txt').write_bytes(git('diff', '--cached', '--stat', parent, env=environment))
    tree = git('write-tree', env=environment).decode().strip()
    result = dict(staged=True, parent=parent, tree=tree, branch=BRANCH, files=list(files),
        index=str(index), failure_index=rebuilt, shared_index_used=False,
        commit_created=False, pushed=False, review_required=True, diff=str(OUT / 'staged.diff'),
        git_text_blobs_CRLF_normalized_to_LF=True, archive_source_bytes_modified=False)
    result['git_text_blobs_single_final_LF'] = True
    write(OUT / 'stage.json', result)
    write(OUT / 'review.json', dict(parent=parent, tree=tree, passed=False, review_required=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--stage', action='store_true')
    action.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--http-proxy', default=os.environ.get('PUBLICATION_GIT_HTTP_PROXY', ''))
    args = parser.parse_args()
    if args.http_proxy:
        os.environ['PUBLICATION_GIT_HTTP_PROXY'] = args.http_proxy
    OUT.mkdir(parents=True, exist_ok=True)
    if args.inventory_only:
        selection = source_manifest()
        entries = inventory(selection)
        result = dict(prepared=False, inventory_ready=True, staged=False, commit_created=False, pushed=False,
                      run_id=RUN.name, docprefix=PREFIX, parent_binding_deferred=True,
                      expected_R2_ancestor=R2_PARENT_COMMIT, files=entries,
                      missing_final_inputs=[r['source'] for r in entries if not r['exists']],
                      shared_checkout_read=False, shared_checkout_written=False, shared_index_used=False)
        write(OUT / 'inventory.json', result)
        print(json.dumps({k: result[k] for k in ('inventory_ready', 'staged', 'commit_created', 'pushed', 'missing_final_inputs')}, ensure_ascii=False))
        return
    before = checkout_snapshot()
    try:
        parent = verify_origin()
        selection, entries, prepared = prepare(parent)
        result = stage(parent, selection, entries) if args.stage else prepared
        print(json.dumps({k: result[k] for k in result if k in ('prepared', 'staged', 'commit_created', 'pushed', 'origin_parent', 'parent', 'tree', 'missing_final_inputs')}, ensure_ascii=False))
    finally:
        after = checkout_snapshot()
        exact = before == after
        write(OUT / 'checkout_preservation.json', dict(preserved=exact, before=before, after=after,
              shared_worktree_written=False, shared_index_used=False, GIT_OPTIONAL_LOCKS=0))
        if not exact:
            raise RuntimeError('Shared checkout changed during publication preparation; preserve it and inspect differences')


if __name__ == '__main__':
    main()
