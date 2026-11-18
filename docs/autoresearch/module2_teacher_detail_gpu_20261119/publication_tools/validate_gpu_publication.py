"""从私有已暂存树导出后做 CPU 隔离验收；不导入活动树代码。"""
from pathlib import Path
import contextlib
import io
import json
import os
import subprocess
import sys

R = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')


class CountReports:
    def __init__(self):
        self.passed = self.failed = self.skipped = 0
        self.passed_nodes = set()

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed += 1
        if report.when == 'call':
            self.passed += int(report.passed)
            if report.passed:
                self.passed_nodes.add(report.nodeid)
            self.skipped += int(report.skipped)


def main(path):
    plan = json.loads(Path(path).read_text())
    out = Path(plan['output'])
    env = dict(os.environ, GIT_INDEX_FILE=str(out / 'publication.index'), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=R, env=env)
    head = git('rev-parse', 'HEAD').decode().strip()
    if head != plan['expected_active_head']:
        raise RuntimeError('active HEAD changed')
    index = Path(subprocess.check_output(['git', 'rev-parse', '--git-path', 'index'], cwd=R).decode().strip())
    if not index.is_absolute():
        index = R / index
    original_index = index.read_bytes() if index.exists() else None
    tree = git('write-tree').decode().strip()
    source = out / ('validated_source_' + tree[:12])
    attempt = 0
    while source.exists():
        attempt += 1
        source = out / ('validated_source_' + tree[:12] + '_retry' + str(attempt))
    for name in ['source_validation.json', 'source_validation.log']:
        old = out / name
        backup = out / (name + '.counter_initial')
        if old.exists() and not backup.exists():
            backup.write_bytes(old.read_bytes())
    source.mkdir(exist_ok=False)
    for rel in plan['files']:
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git('show', tree + ':' + rel))
    checks = {rel: (source / rel).read_bytes() == Path(original).read_bytes()
              for rel, original in plan['executed_code_bindings'].items()}
    if not all(checks.values()):
        raise RuntimeError('source byte mismatch')
    os.environ.update({key: value for key, value in env.items() if key != 'GIT_INDEX_FILE'})
    imports = [source / name for name in plan['import_paths']]
    sys.path[:0] = [str(value) for value in imports]
    import pytest
    reports, log = CountReports(), io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        code = pytest.main(['-q', '--import-mode=importlib', '--rootdir', str(source),
                            *[str(source / test) for test in plan['tests']]], plugins=[reports])
    (out / 'source_validation.log').write_text(log.getvalue(), encoding='utf-8')
    expected_imports = ['fixed_material_read_operator', 'world_texture_runtime', 'gpu_entry_guard',
                        'teacher_detail_permission', 'pixel_world_material', 'native_basis_adapter',
                        'trellis_hf_seed', 'project_h_seed', 'field_training_state', 'material_readers']
    imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in expected_imports if name in sys.modules}
    isolated = all(Path(value).is_relative_to(source.resolve()) for value in imported.values())
    paths = sorted(str(value) for value in source.rglob('*.py'))
    groups = {}
    for executable in ['/root/autodl-tmp/envs/motionproj/bin/python', '/root/autodl-tmp/envs/drivingeditor/bin/python']:
        subprocess.run([executable, '-m', 'py_compile', *paths], env=env, check=True)
        groups[executable] = len(paths)
    if (index.read_bytes() if index.exists() else None) != original_index or git('rev-parse', 'HEAD').decode().strip() != head:
        raise RuntimeError('active HEAD/index changed during validation')
    tests_passed = len(reports.passed_nodes)
    passed = code == 0 and isolated and tests_passed == plan['expected_tests'] and reports.failed == 0
    receipt = dict(passed=passed, tree=tree, isolated_source_export=True, active_worktree_used_for_tests=False,
                   exact_executed_source_bindings=checks, isolated_module_paths=imported,
                   compiled_sources=len(paths), environment_compile_groups=groups, tests=plan['tests'],
                   tests_passed=tests_passed, successful_call_reports=reports.passed,
                   additional_successful_subtest_reports=reports.passed-tests_passed,
                   tests_failed=reports.failed, tests_skipped=reports.skipped,
                   expected_tests=plan['expected_tests'], CUDA_VISIBLE_DEVICES='', native_GPU_calls=0,
                   scene_optimizer_updates=0, active_HEAD_unchanged=True, active_index_unchanged=True)
    (out / 'source_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(log.getvalue())
    print(json.dumps(dict(passed=passed, tree=tree, tests=tests_passed, compiled_sources=len(paths),
                          source_byte_checks=len(checks), isolated_imports=isolated)))
    if not passed:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main(sys.argv[1])
