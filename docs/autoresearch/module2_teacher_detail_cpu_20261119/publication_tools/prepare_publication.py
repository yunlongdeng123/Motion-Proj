"""CPU 来源发布准备；只写 campaign，不触活动工作树或 Git index。"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess

R = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
BASE = '58b60ea32e130daea14869e82c62feb7f91538ba'
HEAD = 'e10b5c20545a2737bb799bd7f7693c3e9e7e28cc'
D = 'docs/autoresearch/module2_teacher_detail_cpu_20261119'
TASK = b'MODULE2-TEACHER-DETAIL-CPU-20261119'
MODULES = ['fixed_material_read_operator.py', 'world_texture_runtime.py', 'gpu_entry_guard.py']
CANONICAL = ['report.md', 'results.json', 'protocol.json', 'architecture.svg']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=R)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def append(before, addition):
    if TASK in before:
        raise ValueError('current task already present; preserve document and review')
    separator = b'' if before.endswith(b'\n\n') else b'\n' if before.endswith(b'\n') else b'\n\n'
    return before + separator + addition


def preserve_parallel_status(before, snapshot):
    """只在 working candidate 保留别人未提交的二维任务完整段落。"""
    paragraphs = before.decode('utf-8').split('\n\n')
    preserved = [p for p in paragraphs if 'ROAD-2D-TEACHER-HF-CPU-20261119' in p]
    if len(preserved) > 1:
        raise ValueError('ambiguous parallel status paragraphs')
    if not preserved:
        return snapshot, []
    if 'ROAD-2D-TEACHER-HF-CPU-20261119' in snapshot.decode('utf-8'):
        raise ValueError('publication snapshot must not absorb unselected parallel work')
    parts = snapshot.decode('utf-8').split('\n\n', 2)
    candidate = '\n\n'.join(parts[:2] + preserved + parts[2:]).encode('utf-8')
    assert preserved[0] in candidate.decode('utf-8')
    return candidate, preserved


def main(output, templates, canonical_frozen):
    if not canonical_frozen:
        raise ValueError('root canonical freeze notice is required')
    output, templates = Path(output), Path(templates)
    if output.exists():
        raise FileExistsError('preserve existing prepared snapshot')
    if git('rev-parse', 'HEAD').decode().strip() != HEAD:
        raise ValueError('active HEAD changed')
    git('cat-file', '-e', BASE + '^{commit}')
    index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = R / index
    index_before = index.read_bytes() if index.exists() else None
    output.mkdir(parents=True)
    tree = output / 'tree'
    tree.mkdir()
    sources, bindings, roles, files = {}, {}, {}, []

    def put(rel, data, source=None, role='frozen'):
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('target collision: ' + rel)
        target.write_bytes(data)
        files.append(rel)
        roles[rel] = role
        if source is not None:
            sources[rel] = dict(path=str(source), sha256=sha(source))

    def copy(rel, source, role='frozen'):
        source = Path(source)
        if not source.is_file() or source.stat().st_size > 2_000_000:
            raise ValueError('only small source/docs/evidence: ' + str(source))
        put(rel, source.read_bytes(), source, role)

    for name in CANONICAL:
        copy(D + '/' + name, Q / name, 'canonical')
    results = json.loads((Q / 'results.json').read_text())
    if results['GPU_authorized'] is not False or results['actual_scene_optimizer_updates'] != 0:
        raise ValueError('this publication is CPU preparation only')
    collisions = []
    for name in MODULES:
        for folder, filename in [('scripts/worldsim_v82', name), ('tests/worldsim_v82', 'test_' + name)]:
            rel, source = folder + '/' + filename, Q / 'code' / filename
            current = R / rel
            if current.exists() and current.read_bytes() != source.read_bytes():
                collisions.append(rel)
            copy(rel, source, 'reusable_CPU_module' if folder.startswith('scripts') else 'final_CPU_test')
            bindings[rel] = str(source)
    if collisions:
        save_json(output / 'collision.json', dict(paths=collisions, action='stop; do not overwrite unknown work'))
        raise ValueError('existing reusable target differs: ' + ','.join(collisions))
    for source in sorted(Q.rglob('*.py')):
        rel = source.relative_to(Q)
        if 'publication' in rel.parts or '__pycache__' in rel.parts:
            continue
        target = D + '/run_code/' + rel.as_posix()
        copy(target, source, 'CPU_source_snapshot')
        bindings[target] = str(source)
    for source in sorted(Q.rglob('*.json')):
        rel = source.relative_to(Q)
        if 'publication' in rel.parts or '__pycache__' in rel.parts or source.name == 'gpu_lease.json':
            continue
        if rel.as_posix() in CANONICAL or source.stat().st_size > 2_000_000:
            continue
        copy(D + '/evidence/' + rel.as_posix(), source, 'CPU_evidence')
    for source in sorted(Q.rglob('*.md')):
        rel = source.relative_to(Q)
        if 'publication' in rel.parts or rel.as_posix() in CANONICAL:
            continue
        if source.stat().st_size <= 2_000_000:
            copy(D + '/preparation_docs/' + rel.as_posix(), source, 'CPU_schema_document')
    for source in sorted(templates.iterdir()):
        if source.is_file() and source.suffix in ('.py', '.md'):
            copy(D + '/publication_tools/' + source.name, source, 'publication_tool_or_template')
    readme = ('# 教师细节：CPU 来源与复现\n\n'
              '[报告与组件图](report.md) · [结果](results.json) · [协议](protocol.json)。\n\n'
              'CPU_PREPARED_STOP_BEFORE_GPU。当前 GPU/native/场景优化/新推理均为 0；默认场景不变。'
              '未来四组主对照 2800 物理更新，次要部分投影 TRELLIS.2 HF 初始化最多另 600，均未执行。\n\n'
              '`run_code` 保留 CPU 准备及原生 adapter 的原字节；原生入口本轮未执行，默认拒绝未授权调用。'
              '仓库 scripts/worldsim_v82 的三个平铺模块使用相同源码，提供固定读取、共同世界材质运行时及执行硬门。'
              '不安装控制器，不接管默认 scene。\n\n'
              '完整数组、权威场、算子和 CPU 数据夹具保留远端：`' + str(Q) + '`。'
              '其中 `permission/final_r1/teacher_evidence.npz` 与 recipe 为共同输入；平均算子为 '
              '`permission/avg_sparse_gh31_r1/operator`，逐像素算子为 `pixel_render/read_operator_box_r1/{086,110,134,179,198}`。'
              '平均算子 row_ids 是原完整道路拓扑 global world ID，须显式映射 active positions；K_H 已含 valid/mH，不得二次乘门。\n\n'
              '最终六个测试文件由 publication plan 显式选择，只运行最终版本，不把失败工程快照的旧测试混入。'
              '验证在私有暂存树的独立导出中进行，CUDA 隐藏、1 线程；Python3.10 跑 CPU 测试，3.10/3.9 做源码编译检查。'
              '数据夹具可从冻结远端绝对路径读取，活动仓库代码不参与测试导入。\n\n'
              'publication_tools 提供 prepare → stage-only → source validation → root review/commit 的受限流程；'
              'working docs 应在 root 发布后带备份应用，保留并行二维任务段与全部原实验历史。\n')
    put(D + '/README.md', readme.encode('utf-8'), role='authored_document')
    docpairs = {'docs/EXPERIMENTS.md': 'experiments_append.md',
                'docs/research_failures/entries/V82-F06.md': 'failure_append.md'}
    work, patches, pubpatch = {}, [], []
    parallel = []
    for rel, name in {'docs/RESEARCH_STATUS.md': 'RESEARCH_STATUS.md', **docpairs}.items():
        before, addition = (R / rel).read_bytes(), (templates / name).read_bytes()
        if rel == 'docs/RESEARCH_STATUS.md':
            after, parallel = preserve_parallel_status(before, addition)
            pub = addition
        else:
            after = append(before, addition)
            pub = append(git('show', BASE + ':' + rel), addition)
        put(rel, pub, role='authored_document')
        for folder, data in [('working_before', before), ('working_docs', after)]:
            target = output / folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        work[rel] = dict(before_sha256=sha(R / rel), candidate=str(output / 'working_docs' / rel))
        old = git('show', BASE + ':' + rel)
        for bucket, first, second in [(patches, before, after), (pubpatch, old, pub)]:
            bucket.extend(difflib.unified_diff(first.decode().splitlines(True), second.decode().splitlines(True),
                                             fromfile='a/' + rel, tofile='b/' + rel))
    (output / 'working_docs.patch').write_text(''.join(patches), encoding='utf-8')
    (output / 'publication_docs.patch').write_text(''.join(pubpatch), encoding='utf-8')
    scratch = output / 'failure_index_scratch'
    names = ['scripts/build_research_failure_index.py', 'scripts/query_research_failures.py',
             'docs/research_failures/history_index.json']
    names += git('ls-tree', '-r', '--name-only', BASE, '--', 'docs/research_failures/entries').decode().splitlines()
    for rel in names:
        target = scratch / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git('show', BASE + ':' + rel))
    shutil.copy2(tree / 'docs/research_failures/entries/V82-F06.md', scratch / 'docs/research_failures/entries/V82-F06.md')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log = subprocess.check_output(['/root/autodl-tmp/envs/motionproj/bin/python',
                                   str(scratch / 'scripts/build_research_failure_index.py')], env=env, text=True)
    nav = []
    for source in sorted((scratch / 'docs/research_failures').rglob('*.md')):
        rel = source.relative_to(scratch).as_posix()
        if '/entries/' in rel:
            continue
        try:
            old = git('show', BASE + ':' + rel)
        except subprocess.CalledProcessError:
            old = b''
        if source.read_bytes() != old:
            put(rel, source.read_bytes(), role='generated_failure_index')
            nav.append(rel)
    save_json(output / 'failure_index_regeneration.json', dict(executed_in_isolated_scratch=True,
              log=log.strip(), changed=nav, active_tree_modified=False))
    files = sorted(set(files))
    tests = ['tests/worldsim_v82/test_' + name for name in MODULES] + [D + '/run_code/' + name for name in
             ['permission/code/test_permission_cpu.py', 'pixel_render/code/test_pixel_render_cpu.py',
              'trellis_hf/code/test_trellis_hf_seed.py']]
    if any(name not in files for name in tests):
        raise ValueError('final test source missing')
    plan = dict(output=str(output / 'stage'), branch='research/worldsim-v8.2-drivingeditor-v44', expected_base=BASE,
                expected_active_head=HEAD, files=files, source_overrides={name: str(tree / name) for name in files},
                override_sha256={name: sha(tree / name) for name in files}, original_source_bindings=sources,
                executed_code_bindings=bindings, tests=tests, expected_tests=37, archive_root=D,
                frozen_archive_prefixes=[D + '/run_code/', D + '/evidence/'],
                frozen_archive_files=[name for name in files if name.endswith(('.py', '.json', '.svg'))],
                append_basis={name: BASE for name in docpairs}, working_docs=work,
                parallel_status_preserved_in_working_candidate=parallel,
                message='feat(road): prepare teacher-detail permissions and shared material readers\n\n'
                        'Archive CPU teacher evidence, GH31 and pixel-footprint read operators with explicit layouts and execution gates. '
                        'Keep native and GPU execution pending, preserve original source ledgers, and do not promote default scenes.')
    save_json(output / 'plan.json', plan)
    save_json(output / 'source_list.json', dict(files=files, roles=roles, bindings=sources))
    if (index.read_bytes() if index.exists() else None) != index_before or git('rev-parse', 'HEAD').decode().strip() != HEAD:
        raise RuntimeError('active HEAD/index changed')
    receipt = dict(completed=True, files=len(files), actual_source_files=len(bindings), tests=tests, expected_tests=37,
                   private_stage_not_executed=True, active_worktree_modified=False, active_index_unchanged=True,
                   active_HEAD=HEAD, base=BASE, parallel_status_preserved=bool(parallel),
                   large_models_arrays_media_published=False, canonical_sources={name: sha(Q / name) for name in CANONICAL})
    save_json(output / 'prepared.json', receipt)
    print(json.dumps(dict(output=str(output), files=len(files), code=len(bindings), tests=tests, nav_changed=nav)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--templates', required=True, type=Path)
    parser.add_argument('--canonical-frozen', action='store_true')
    args = parser.parse_args()
    main(args.output, args.templates, args.canonical_frozen)
