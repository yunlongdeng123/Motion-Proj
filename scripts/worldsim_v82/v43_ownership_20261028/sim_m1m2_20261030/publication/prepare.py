"""Build a bounded M1/M2 research publication; no SSH or experiment launch."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
WORK = HERE.parent
OUT = BASE / 'outputs/V43_Ownership_20261027'
DOC = 'docs/autoresearch/v43_ownership_20261028/sim_m1m2_20261030'
CODE = 'scripts/worldsim_v82/v43_ownership_20261028/sim_m1m2_20261030'
R5DOC = 'docs/autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030'
R5CODE = 'scripts/worldsim_v82/v43_ownership_20261028/gt_protocol_r5_20261030'
spec = importlib.util.spec_from_file_location('previous_publisher', BASE / 'work/v43_gt_protocol_20261030/publication/prepare_publication.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
TEXT = {'.py', '.json', '.md', '.html', '.svg', '.cjs', '.js', '.txt'}


def main():
    report = OUT / 'gpu_v43/sim_m1m2_20261030'
    assert json.loads((report / 'validation.json').read_text(encoding='utf8'))['passed']
    entries = {}

    def add(name, data):
        if isinstance(data, str):
            data = data.encode('utf8')
        if Path(name).suffix in TEXT:
            data = data.replace(b'\r\n', b'\n')
        if Path(name).suffix == '.py':
            data = data.rstrip(b'\n') + b'\n'
        entries[name] = data

    for path in sorted(report.rglob('*')):
        if path.is_file():
            assert not path.is_symlink()
            data = path.read_bytes()
            if path.name == 'index.html':
                data = data.replace(b'../../index.html', b'../gt_protocol_r5_20261030/index.html')
            add(f'{DOC}/{path.relative_to(report).as_posix()}', data)
    for folder in ('m1_assignment', 'm2_repair', 'heldout_edit_refs'):
        for path in sorted((WORK / folder).iterdir()):
            if path.is_file() and path.suffix in TEXT:
                add(f'{CODE}/{folder}/{path.name}', path.read_bytes())
    for name in ('build_m1m2_report.py', 'validate_m1m2.cjs', 'M1_M2_CONTRACT.json', 'remote_exec.py'):
        add(f'{CODE}/{name}', (WORK / name).read_bytes())
    for path in sorted(HERE.glob('*.py')):
        add(f'{CODE}/publication/{path.name}', path.read_bytes())
    for name in ('V43_DESIGN.md', 'architecture.svg', 'architecture.png', 'data_architecture.svg', 'data_architecture.png',
                 'ownership_record_contract.json', 'mechanism_experiment_plan.json', 'direction_state.json',
                 'design_revision_validation.json', 'architecture_validation.json'):
        data = (OUT / name).read_bytes()
        if Path(name).suffix in {'.md', '.json'}:
            data = previous.remote_links(data.decode('utf8'))
        add(f'{R5DOC}/{name}', data)
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V43 Revision 5</title><style>' + previous.STYLE
    html += '</style><h1>V43 Revision5：世界GT与真实重建协议</h1><p><a href="../joint_direction_r4_20261030/index.html">保留的Revision4</a> · <a href="R4_ARCHIVE.md">R4档案</a></p><img src="architecture.svg" alt="V43嵌套组件图">'
    html += previous.remote_links(previous.DESIGN_HTML) + '</html>\n'
    add(f'{R5DOC}/index.html', html)
    for name in ('revision5_contract.py', 'design_revision5.md'):
        add(f'{R5CODE}/{name}', (BASE / 'work/v43_ownership_20261027' / name).read_bytes())
    add('docs/RESEARCH_STATUS.md', (WORK / 'RESEARCH_STATUS.md').read_bytes())
    add(f'{CODE}/README.md', '# M1/M2 实际机制对照\n\n本目录是已执行脚本归档；路径记录原 campaign 与本地工作区，不作为后台作业入口。独立 GT 定义在世界表面，不把条件贡献赋成唯一高斯真值。报告包含失败、局部拟合额外容量与 depth 修复收据。M0 数据与原生 helper 继承同级 sim_mechanism_20261030 及远端原现场，已有日志/检查点不覆盖。自动关机禁用。\n')
    manifest = {'task_id': 'WS-V43-SIM-M1-M2-20261030', 'files': sorted(entries),
                'sha256': {n: hashlib.sha256(b).hexdigest() for n, b in entries.items()},
                'experiment_entry': '- WS-V43-SIM-M1-M2-20261030：[真值辅助归属与同集合表示修复实际对照](autoresearch/v43_ownership_20261028/sim_m1m2_20261030/index.html)；3关键帧、1留出相机、两臂各200步局部拟合；有限删除收益，未编辑/横移仍失败，不是数学上界或泛化证明。',
                'power_actions': 0, 'new_jobs_started_by_publication': 0}
    manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf8')
    (HERE / 'manifest.json').write_bytes(manifest_data)
    (HERE / 'publication.tar.gz').write_bytes(previous.archive_bytes({**entries, 'PUBLICATION_MANIFEST.json': manifest_data}))
    print(json.dumps({'files': len(entries), 'bytes': sum(map(len, entries.values())), 'bundle': str(HERE / 'publication.tar.gz')}))


if __name__ == '__main__':
    main()
