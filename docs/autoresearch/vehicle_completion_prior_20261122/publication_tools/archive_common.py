"""本轮轻量归档的路径、文本和共享 Git 状态保护。仅使用标准库。"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import subprocess

DEFAULT_REPO = '/root/autodl-tmp/motion_proj_v82_gn11_v42'
ARCHIVE = 'docs/autoresearch/vehicle_completion_prior_20261122'
BRANCH = 'research/worldsim-v8.2-drivingeditor-v44'
ALLOWED_SUFFIXES = {'.py', '.ps1', '.sh', '.svg', '.html', '.css', '.js', '.json', '.md', '.yaml', '.yml', '.toml', '.patch'}
ALLOWED_ROLES = {'run_source', 'contract', 'receipt', 'metrics', 'report_source',
                 'architecture_source', 'publication_tool', 'configuration', 'provenance', 'record'}
CURRENT_DOCS = {'docs/RESEARCH_STATUS.md', 'docs/EXPERIMENTS.md', 'docs/research_failures/entries/V82-F06.md'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def valid_destination(name, allow_current_docs=False):
    if not isinstance(name, str) or not name or '\\' in name or '\n' in name or '\r' in name:
        raise ValueError('归档路径格式错误：' + repr(name))
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ('.', '..', '.git') for x in p.parts) or p.as_posix() != name:
        raise ValueError('归档路径越界：' + name)
    if not name.startswith(ARCHIVE + '/') and not (allow_current_docs and name in CURRENT_DOCS):
        raise ValueError('非本轮归档路径：' + name)
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError('不允许模型、原图或二进制归档：' + name)
    return name


def lightweight_text(path):
    path = Path(path)
    lowered_parts = [x.lower() for x in path.parts]
    name = path.name.lower()
    if (any('road_multisource' in part for part in lowered_parts)
            or 'reference' in lowered_parts or name in {'hugsim_text.txt', 'old_report_before_vehicle_entry.html'}
            or name.startswith('target_visibility.')):
        raise ValueError('排除并行道路任务、外部全文、借阅源、备份或旧错误目标诊断：' + str(path))
    if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError('不是允许的源码/记录文件：' + str(path))
    limit = 8_000_000 if path.suffix.lower() == '.json' else 2_000_000
    if path.stat().st_size > limit:
        raise ValueError('超过轻量文件大小上限：' + str(path))
    data = path.read_bytes()
    decoded = data.decode('utf-8-sig')
    embedded = re.search(r'data:(?:image/[a-z0-9.+-]+|video/[a-z0-9.+-]+|audio/[a-z0-9.+-]+|application/octet-stream);base64,[a-z0-9+/=\r\n]{64,}', decoded, re.I)
    if '\0' in decoded or embedded:
        raise ValueError('检测到二进制或内嵌原始媒体：' + str(path))
    return data


class Git:
    def __init__(self, repo, proxy=None):
        self.repo = Path(repo).resolve()
        self.env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', CUDA_VISIBLE_DEVICES='',
                        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for name in ('GIT_INDEX_FILE', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR'):
            self.env.pop(name, None)
        if proxy:
            for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                self.env[name] = proxy
        top = self.call('rev-parse', '--show-toplevel').decode().strip()
        if Path(top).resolve() != self.repo:
            raise ValueError('Git 根目录不符')

    def call(self, *args, input=None, index=None):
        env = self.env if index is None else dict(self.env, GIT_INDEX_FILE=str(Path(index).resolve()))
        result = subprocess.run(['git', *args], cwd=self.repo, input=input, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode:
            raise RuntimeError('git ' + ' '.join(args[:4]) + ': ' + result.stderr.decode(errors='replace').strip())
        return result.stdout

    def shared_index(self):
        path = Path(self.call('rev-parse', '--git-path', 'index').decode().strip())
        return path if path.is_absolute() else self.repo / path

    def snapshot(self):
        index = self.shared_index()
        names = self.call('diff', '--name-only', '-z', 'HEAD').decode().split('\0')
        dirty = {}
        for name in filter(None, names):
            p = self.repo / name
            dirty[name] = sha(p) if p.is_file() else None
        return dict(active_HEAD=self.call('rev-parse', 'HEAD').decode().strip(),
                    active_branch=self.call('rev-parse', '--symbolic-full-name', 'HEAD').decode().strip(),
                    shared_index_path=str(index), shared_index_sha256=sha(index) if index.exists() else None,
                    tracked_status=self.call('status', '--porcelain=v1', '-z', '--untracked-files=no').decode(),
                    dirty_tracked_sha256=dirty)

    def assert_snapshot(self, before):
        after = self.snapshot()
        if before != after:
            changed = [k for k in before if before[k] != after.get(k)]
            raise RuntimeError('共享 Git 状态发生变化，停止且不重置：' + ', '.join(changed))

    def remote_head(self, branch):
        rows = self.call('ls-remote', 'origin', 'refs/heads/' + branch).decode().splitlines()
        if len(rows) != 1:
            raise RuntimeError('origin 分支查询为空或不唯一')
        return rows[0].split()[0]

    def ensure_commit(self, commit):
        try:
            self.call('cat-file', '-e', commit + '^{commit}')
        except RuntimeError:
            self.call('fetch', '--no-write-fetch-head', 'origin', commit)

    def protected_objects(self, tree):
        roots = ['AGENTS.md']
        for name in self.call('ls-tree', '-d', '--name-only', tree + ':docs/autoresearch').decode().splitlines():
            if name.startswith('road_2d_teacher_') or 'road_multisource' in name:
                roots.append('docs/autoresearch/' + name)
        return {p: self.call('rev-parse', tree + ':' + p).decode().strip() for p in roots}


def check_frozen_sources(plan):
    for name in plan['files']:
        valid_destination(name, allow_current_docs=True)
        if sha(plan['source_overrides'][name]) != plan['override_sha256'][name]:
            raise ValueError('冻结覆盖源发生变化：' + name)
    for name, rec in plan.get('original_source_bindings', {}).items():
        if sha(rec['path']) != rec['sha256']:
            raise ValueError('正式来源在准备后发生变化：' + name)


def private_index(plan):
    output = Path(plan['output']).resolve()
    index = output / 'publication.index'
    if str(index) == plan['shared_snapshot']['shared_index_path']:
        raise ValueError('拒绝使用共享 index')
    return index
