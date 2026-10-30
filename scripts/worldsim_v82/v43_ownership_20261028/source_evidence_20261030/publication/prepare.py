"""Package completed source/evidence controls; no experiment or power operations."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
BASE = WORK.parents[1]
OUT = BASE/'outputs/V43_Ownership_20261027'
DOC = 'docs/autoresearch/v43_ownership_20261028/source_evidence_20261030'
CODE = 'scripts/worldsim_v82/v43_ownership_20261028/source_evidence_20261030'
R5DOC = 'docs/autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030'
R5CODE = 'scripts/worldsim_v82/v43_ownership_20261028/gt_protocol_r5_20261030'
spec = importlib.util.spec_from_file_location('previous', BASE/'work/v43_gt_protocol_20261030/publication/prepare_publication.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
TEXT = {'.py', '.json', '.md', '.html', '.svg', '.cjs', '.js', '.txt', '.csv'}


def main():
    report = OUT/'gpu_v43/source_evidence_20261030'
    assert json.loads((report/'validation.json').read_text())['passed']
    assert json.loads((report/'report_state.json').read_text())['evidence_reconstruction_report_attached']
    entries = {}

    def add(name, data):
        if isinstance(data, str):
            data = data.encode('utf8')
        if Path(name).suffix in TEXT:
            data = data.replace(b'\r\n', b'\n').rstrip(b'\n')+b'\n'
        entries[name] = data

    for path in sorted(report.rglob('*')):
        if path.is_file():
            assert not path.is_symlink()
            assert path.suffix not in {'.npz', '.pt', '.pth'}
            data = path.read_bytes()
            if path.suffix == '.html':
                data = data.replace(b'../../index.html', b'../gt_protocol_r5_20261030/index.html')
            add(f'{DOC}/{path.relative_to(report).as_posix()}', data)
    for folder in ('source_control_r1', 'visibility_control_r1', 'evidence_reconstruction_r1'):
        for path in sorted((WORK/folder).iterdir()):
            if path.is_file() and path.suffix in TEXT:
                name = path.name+'.source.txt' if path.suffix=='.html' else path.name
                add(f'{CODE}/{folder}/{name}', path.read_bytes())
    for name in ('build_source_evidence_report.py', 'validate_source_report.cjs', 'build_rich_section.py',
                 'SOURCE_EVIDENCE_CONTRACT.json', 'remote_exec.py'):
        add(f'{CODE}/{name}', (WORK/name).read_bytes())
    for path in sorted(HERE.glob('*.py')):
        add(f'{CODE}/publication/{path.name}', path.read_bytes())
    # Preserve old experiment artifacts; only correct the interpretation and next-control link.
    old = OUT/'gpu_v43/sim_m1m2_20261030'
    for name in ('index.html', 'validation.json'):
        data = (old/name).read_bytes()
        if name == 'index.html':
            data = data.replace(b'../../index.html', b'../gt_protocol_r5_20261030/index.html')
        add('docs/autoresearch/v43_ownership_20261028/sim_m1m2_20261030/'+name, data)
    add('scripts/worldsim_v82/v43_ownership_20261028/sim_m1m2_20261030/build_m1m2_report.py', (WORK/'build_m1m2_report.py').read_bytes())
    for name in ('V43_DESIGN.md', 'architecture.svg', 'architecture.png', 'data_architecture.svg', 'data_architecture.png',
                 'ownership_record_contract.json', 'mechanism_experiment_plan.json', 'direction_state.json',
                 'design_revision_validation.json', 'architecture_validation.json'):
        data = (OUT/name).read_bytes()
        if Path(name).suffix in {'.md', '.json'}:
            data = previous.remote_links(data.decode('utf8'))
        add(f'{R5DOC}/{name}', data)
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V43 Revision 5</title><style>'+previous.STYLE
    html += '</style><h1>V43 Revision5：世界GT与真实重建协议</h1><p><a href="../joint_direction_r4_20261030/index.html">保留Revision4</a> · <a href="R4_ARCHIVE.md">R4档案</a></p><img src="architecture.svg" alt="嵌套组件图">'
    html += previous.remote_links(previous.DESIGN_HTML)+'</html>\n'
    add(f'{R5DOC}/index.html', html)
    for name in ('revision5_contract.py', 'design_revision5.md'):
        add(f'{R5CODE}/{name}', (BASE/'work/v43_ownership_20261027'/name).read_bytes())
    add('docs/RESEARCH_STATUS.md', (WORK/'RESEARCH_STATUS.md').read_bytes())
    add(f'{CODE}/README.md', '# 删后来源与观测证据控制\n\n已执行脚本归档，原始路径指向保留的campaign现场，不是后台作业入口。来源控制固定旧M0权重及原M1集合；观测控制固定世界、共享规范坐标及正式10K预算，另计600秒真实smoke。新旧高斯ID不可互用。报告公开特权诊断、失败、桥接探针与同query比较；不提供唯一owner真值或理想修复上界。完整浮点数组/检查点保存在campaign与本地证据包，不进入Git。AutoDL保持开机。\n')
    manifest = dict(task_id='WS-V43-SIM-SOURCE-EVIDENCE-20261030', files=sorted(entries),
                    sha256={n: hashlib.sha256(b).hexdigest() for n,b in entries.items()},
                    experiment_entry='- WS-V43-SIM-SOURCE-EVIDENCE-20261030：[删后来源与同世界观测控制](autoresearch/v43_ownership_20261028/source_evidence_20261030/index.html)；全行αT、16观测回查、固定旧模型桥接及新观测10K重建。区分选点、表示与观测缺口；不将旧M1/M2实现失败推广为方向失败。',
                    power_actions=0, new_jobs_started_by_publication=0)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf8')
    (HERE/'manifest.json').write_bytes(raw)
    (HERE/'publication.tar.gz').write_bytes(previous.archive_bytes({**entries, 'PUBLICATION_MANIFEST.json':raw}))
    print(json.dumps(dict(files=len(entries), bytes=sum(map(len,entries.values())), bundle=str(HERE/'publication.tar.gz'))))


if __name__ == '__main__':
    main()
