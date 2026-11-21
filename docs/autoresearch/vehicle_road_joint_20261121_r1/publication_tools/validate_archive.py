"""验收精确暂存树的来源、语法和选定路径；不运行训练或 GPU。"""
from pathlib import Path
import argparse
import ast
import json
import xml.etree.ElementTree as ET
from archive_common import (Git, dump_json, read_json, digest, check_frozen_sources, private_index, valid_destination)


def validate(plan_path):
    p = read_json(plan_path)
    git = Git(p['repo'])
    git.assert_snapshot(p['shared_snapshot'])
    before = git.snapshot()
    try:
        check_frozen_sources(p)
        stage = read_json(Path(p['output']) / 'stage_receipt.json')
        index = private_index(p)
        tree = git.call('write-tree', index=index).decode().strip()
        if tree != stage['tree']:
            raise ValueError('独立 index 与待验收树不符')
        changes = git.call('diff', '--name-only', p['expected_base'], tree).decode().splitlines()
        if set(changes) - set(p['files']):
            raise ValueError('出现选定文件之外的变更')
        checked = []
        for name in p['files']:
            valid_destination(name, allow_current_docs=True)
            data = git.call('show', tree + ':' + name)
            if digest(data) != p['override_sha256'][name]:
                raise ValueError('暂存内容与冻结覆盖源不符：' + name)
            text = data.decode('utf-8-sig')
            suffix = Path(name).suffix.lower()
            if suffix == '.py':
                ast.parse(text, filename=name)
            elif suffix == '.json':
                json.loads(text)
            elif suffix == '.svg':
                ET.fromstring(text)
            checked.append(name)
        for name, oid in p['protected_objects'].items():
            if git.call('rev-parse', tree + ':' + name).decode().strip() != oid:
                raise ValueError('二维教师或 AGENTS 对象发生变化：' + name)
        git.call('-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof',
                 'diff', '--check', p['expected_base'], tree)
        result = dict(passed=True, tree=tree, checked_files=checked, changed_files=changes,
                      validation='byte-exact source binding, Python AST, JSON, SVG XML, diff check',
                      training_or_gpu_executed=False, protected_2d_and_agents_unchanged=True,
                      scientific_or_visual_quality_verdict=False)
        dump_json(Path(p['output']) / 'source_validation.json', result)
        print('精确暂存树源码验收通过：' + tree)
    finally:
        git.assert_snapshot(before)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    args = parser.parse_args()
    validate(args.plan)
