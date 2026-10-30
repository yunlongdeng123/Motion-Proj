"""远端文档发布入口：限定文件同步，可选局部暂存；不提交、推送、训练或关机。"""
from pathlib import Path, PurePosixPath
import argparse
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

DEFAULT_REPO = '/root/autodl-tmp/motion_proj_v82_gn11_v42'
DEFAULT_CAMPAIGN = '/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/publication_r5'
DOC = 'docs/autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030'
CODE = 'scripts/worldsim_v82/v43_ownership_20261028/gt_protocol_r5_20261030'
M0_DOC = 'docs/autoresearch/v43_ownership_20261028/sim_mechanism_20261030'
M0_CODE = 'scripts/worldsim_v82/v43_ownership_20261028/sim_mechanism_20261030'


def git(repo, *arguments):
    return subprocess.run(['git', '-C', str(repo), *arguments], check=True, capture_output=True).stdout


def read_bundle(bundle):
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError('归档含重复路径')
        if any(not member.isfile() for member in members):
            raise ValueError('发布归档只接受普通文件')
        manifest = json.loads(archive.extractfile('PUBLICATION_MANIFEST.json').read())
        if manifest['revision'] != 5:
            raise ValueError('只接受R5包')
        expected = set(manifest['files']) | {'PUBLICATION_MANIFEST.json'}
        if {member.name for member in members} != expected:
            raise ValueError('归档内容与显式清单不一致')
        payload = {}
        for name in manifest['files']:
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError(name)
            allowed = name == 'docs/RESEARCH_STATUS.md' or name.startswith((DOC + '/', CODE + '/'))
            allowed = allowed or (manifest['m0_report_included'] and name.startswith(M0_DOC + '/'))
            m0_source = (manifest['m0_report_included'] and name.startswith(M0_CODE + '/')
                         and relative.suffix.lower() in {'.py', '.json', '.cjs', '.md'}
                         and not any(part in {'__pycache__', 'output', 'outputs', 'tmp', 'temp', '.cache'}
                                     for part in relative.parts))
            allowed = allowed or m0_source
            if not allowed:
                raise ValueError('超出发布范围：' + name)
            payload[name] = archive.extractfile(name).read()
        actual_m0_sources = {name for name in payload if name.startswith(M0_CODE + '/')}
        if actual_m0_sources != set(manifest.get('m0_source_files', [])):
            raise ValueError('M0源码清单与实际归档不一致')
        return manifest, payload


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file() and not path.is_symlink()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(DEFAULT_REPO))
    parser.add_argument('--bundle', type=Path, default=Path(__file__).with_name('publication.tar.gz'))
    parser.add_argument('--campaign', type=Path, default=Path(DEFAULT_CAMPAIGN))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stage', action='store_true', help='只暂存清单文件；不提交和push')
    args = parser.parse_args()
    manifest, payload = read_bundle(args.bundle)
    repo = args.repo.resolve(strict=True)
    git(repo, 'rev-parse', '--show-toplevel')
    allowed_paths = sorted([*payload, 'docs/EXPERIMENTS.md'])
    for relative in allowed_paths:
        target = (repo / relative).resolve()
        if not target.is_relative_to(repo):
            raise ValueError('目标路径越界：' + relative)
    r4_roots = [repo / manifest['r4_doc_root'], repo / manifest['r4_code_root']]
    if not all(root.is_dir() for root in r4_roots):
        raise FileNotFoundError('R4文档和源码必须原位保留，缺失时停止发布')
    for name in ['r2_recovery_20261029', 'unknown_white_20261030', 'editable_ownership_20261030']:
        if not (repo / 'docs/autoresearch/v43_ownership_20261028' / name / 'index.html').is_file():
            raise FileNotFoundError('既有报告依赖缺失：' + name)
    known_dirty = repo / manifest['known_unrelated_dirty_path']
    known_dirty_before = known_dirty.read_bytes() if known_dirty.is_file() else None
    status_before = git(repo, 'status', '--porcelain=v1', '--untracked-files=normal').decode('utf8')
    staged_before = git(repo, 'diff', '--cached', '--name-only', '-z').decode('utf8').split('\0')
    unrelated_staged = sorted(path for path in staged_before if path and path not in allowed_paths)
    if args.stage and unrelated_staged:
        raise RuntimeError('保留已有无关暂存，不扩大提交范围；先单独处理这些暂存后重试--stage：' + repr(unrelated_staged))
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'write_paths': sorted(payload), 'append_only_index': 'docs/EXPERIMENTS.md',
                          'stage_paths_if_requested': allowed_paths, 'r4_paths_preserved': [str(path) for path in r4_roots],
                          'unrelated_staged_preserved': unrelated_staged, 'commits': 0, 'gpu_jobs': 0, 'power_actions': 0}, ensure_ascii=False))
        return
    base = args.campaign.resolve()
    base.mkdir(parents=True, exist_ok=True)
    backup = base / ('before_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    (backup / 'git_status_before.txt').write_text(status_before, encoding='utf8')
    r4_before = {str(root.relative_to(repo)): tree_bytes(root) for root in r4_roots}
    with tarfile.open(backup / 'revision4_remote_before.tar.gz', 'w:gz') as archive:
        for root in r4_roots:
            archive.add(root, arcname=root.relative_to(repo).as_posix())
    for relative, data in payload.items():
        target = repo / relative
        if target.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    index = repo / 'docs/EXPERIMENTS.md'
    old_index = backup / 'docs/EXPERIMENTS.md'
    old_index.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index, old_index)
    text = index.read_text(encoding='utf-8-sig')
    additions = []
    for entry in manifest['experiment_index_entries']:
        task_id = entry.split('：', 1)[0].removeprefix('- ')
        if task_id not in text:
            additions.append(entry)
    if additions:
        index.write_text(text.rstrip() + '\n\n' + '\n'.join(additions) + '\n', encoding='utf8')
    for root in r4_roots:
        assert tree_bytes(root) == r4_before[str(root.relative_to(repo))], 'R4档案发生意外修改'
    if known_dirty_before is not None:
        assert known_dirty.read_bytes() == known_dirty_before, '无关run_worldsim文件被修改'
    for relative, data in payload.items():
        assert (repo / relative).read_bytes() == data, relative
    if args.stage:
        git(repo, 'add', '--', *allowed_paths)
        git(repo, 'diff', '--cached', '--check', '--', *allowed_paths)
        staged_after = set(git(repo, 'diff', '--cached', '--name-only', '-z').decode('utf8').split('\0')) - {''}
        if not staged_after.issubset(set(allowed_paths)):
            raise RuntimeError('暂存范围超出清单；未执行提交')
        (base / 'staged_diff.txt').write_bytes(git(repo, 'diff', '--cached', '--stat', '--', *allowed_paths))
        (base / 'staged_diff.patch').write_bytes(git(repo, 'diff', '--cached', '--', *allowed_paths))
    status_after = git(repo, 'status', '--porcelain=v1', '--untracked-files=normal').decode('utf8')
    receipt = {'revision': 5, 'published_files': len(payload), 'backup': str(backup),
               'm0_report_included': manifest['m0_report_included'], 'experiment_index_entries_appended': additions,
               'stage_requested': args.stage, 'stage_allowlist': allowed_paths,
               'r4_original_directories_preserved': True, 'r4_remote_archive': str(backup / 'revision4_remote_before.tar.gz'),
               'known_unrelated_dirty_path_preserved': manifest['known_unrelated_dirty_path'],
               'unrelated_staged_before': unrelated_staged,
               'git_status_before': status_before, 'git_status_after': status_after,
               'commits': 0, 'pushes': 0, 'gpu_jobs': 0, 'power_actions': 0}
    (base / 'publication_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
