"""准备本地R5文档发布包；不连接远端、不执行实验、Git或电源操作。"""
from pathlib import Path
import argparse
import gzip
import io
import json
import sys
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE / 'outputs/V43_Ownership_20261027'
PAYLOAD = HERE / 'payload'
DOC = 'docs/autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030'
CODE = 'scripts/worldsim_v82/v43_ownership_20261028/gt_protocol_r5_20261030'
M0_DOC = 'docs/autoresearch/v43_ownership_20261028/sim_mechanism_20261030'
M0_CODE = 'scripts/worldsim_v82/v43_ownership_20261028/sim_mechanism_20261030'
M0_ROOT = '/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030'
sys.path.insert(0, str(BASE / 'work/v43_ownership_20261027'))
from revision5_contract import DESIGN_HTML

STYLE = 'body{max-width:1300px;margin:30px auto;padding:0 24px;background:#f6fafc;color:#203d4e;font:17px/1.7 "Microsoft YaHei",sans-serif}a{color:#17639e}img{max-width:100%}section{background:white;border:1px solid #ccdbe5;padding:22px;margin:20px 0;border-radius:10px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd9e3;padding:8px;text-align:left}'


def archive_bytes(entries):
    """固定tar元数据，避免重复打包仅因mtime不同而变化。"""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, data in sorted(entries.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), mtime=0)


def remote_links(text):
    return (text.replace('(gpu_v43/', '(../').replace('gpu_v43/', '../')
            .replace('../bounded_white_v1/index.html', '../r2_recovery_20261029/index.html')
            .replace('../../work/v43_gpu_20261028/train_actor.py', 'LOCAL_SOURCE_REFERENCES.md#actor')
            .replace('../../work/v1/scene/envlight.py', 'LOCAL_SOURCE_REFERENCES.md#environment'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m0-report', type=Path, help='主代理已补齐的报告目录；默认不发布未填骨架')
    parser.add_argument('--status-file', type=Path, help='可选当前状态快照；未提供时使用已知的M0待smoke状态')
    args = parser.parse_args()
    checks = json.loads((OUT / 'design_revision_validation.json').read_text(encoding='utf8'))
    assert checks['passed'] and checks['architecture_revision'] == 5
    assert json.loads((HERE.parent / 'generator_idempotency.json').read_text(encoding='utf8'))['passed']
    entries = {}

    def add(relative, value):
        data = value if isinstance(value, bytes) else value.encode('utf8')
        if Path(relative).suffix.lower() in {'.md', '.py', '.json', '.html', '.svg', '.cjs', '.js', '.txt'}:
            data = data.replace(b'\r\n', b'\n')
        destination = PAYLOAD / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries[relative] = data

    for name in ['V43_DESIGN.md', 'architecture.svg', 'architecture.png', 'data_architecture.svg', 'data_architecture.png',
                 'ownership_record_contract.json', 'mechanism_experiment_plan.json', 'direction_state.json',
                 'design_revision_validation.json', 'architecture_validation.json']:
        data = (OUT / name).read_bytes()
        if name.endswith(('.md', '.json')):
            data = remote_links(data.decode('utf8')).encode('utf8')
        add(f'{DOC}/{name}', data)
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V43 Revision 5</title><style>' + STYLE
    html += '</style><h1>V43 Revision5：世界GT与真实重建协议</h1><p><a href="../joint_direction_r4_20261030/index.html">保留的Revision4</a> · <a href="R4_ARCHIVE.md">R4档案</a></p><img src="architecture.svg" alt="V43嵌套组件图">'
    html += remote_links(DESIGN_HTML) + '</html>\n'
    add(f'{DOC}/index.html', html)
    add(f'{DOC}/LOCAL_SOURCE_REFERENCES.md', '# 本地只读代码核对\n\n## actor\n\n`work/v43_gpu_20261028/train_actor.py`：完整场景 `CG+(1-alpha)E`，本次未改训练源码。\n\n## environment\n\n`work/v1/scene/envlight.py`：世界方向查询可学习cubemap，forward无显示颜色范围约束。\n')
    backup = HERE.parent / 'revision4_backup'
    r4_entries = {path.relative_to(backup).as_posix(): path.read_bytes() for path in backup.rglob('*') if path.is_file()}
    assert len(r4_entries) >= 15
    add(f'{DOC}/revision4_local_snapshot.tar.gz', archive_bytes(r4_entries))
    add(f'{DOC}/R4_ARCHIVE.md', '# Revision4保留档案\n\n[原远端Revision4页面](../joint_direction_r4_20261030/index.html)保持原位，只读保留。\n\n[本地修订前快照](revision4_local_snapshot.tar.gz)保存15份原始文档、图、合同、生成器及校验文件；快照内路径维持原相对结构，未为迁移而改写原始正文。\n\n远端发布脚本另在campaign备份目录保存发布前的R4文档与代码归档，不将R4目录纳入本次Git暂存。\n')
    for name in ['build_v43.py', 'build_architecture.py', 'revision4_contract.py', 'design_revision4.md',
                 'revision5_contract.py', 'design_revision5.md', 'render_architecture.cjs']:
        add(f'{CODE}/{name}', (BASE / 'work/v43_ownership_20261027' / name).read_bytes())
    add(f'{CODE}/validate_revision.py', (BASE / 'work/v43_joint_direction_20261030/validate_revision.py').read_bytes())
    add(f'{CODE}/verify_generation.py', (HERE.parent / 'verify_generation.py').read_bytes())
    for name in ['prepare_publication.py', 'publish_remote.py', 'validate_publication.py',
                 'render_report_template.cjs', 'README.md']:
        add(f'{CODE}/publication/{name}', (HERE / name).read_bytes())
    add(f'{CODE}/README.md', '# R5设计生成源\n\n这些源文件用于本地已有V43工作区的原位生成，保留本地路径、历史首页和备份依赖；远端归档不作为训练或作业启动入口。R5继承R4合同，故R4源与设计同时保留。远端直接阅读对应docs下的独立HTML、两张组件图、Markdown与JSON。\n')
    policy = {'scope': 'current_user_conversation', 'date': '2026-10-30', 'autodl_auto_shutdown': False,
              'explicit_user_shutdown_request_required': True, 'allowed_subagent_models': ['gpt-6.1-sol', 'gpt-6.1-terra']}
    add(f'{DOC}/SESSION_POLICY.json', json.dumps(policy, ensure_ascii=False, indent=2) + '\n')
    m0_source_paths = []
    if args.m0_report:
        report = args.m0_report.resolve()
        if report == (HERE / 'm0_report_template').resolve():
            raise ValueError('未填骨架不能作为最终M0报告发布；请先在目标报告目录补齐真实记录')
        report_state = json.loads((report / 'report_state.json').read_text(encoding='utf8'))
        if report_state.get('artifact_kind') == 'skeleton_not_final_report':
            raise ValueError('M0 report_state仍为骨架状态')
        for path in sorted(report.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'报告不打包符号链接：{path}')
            if path.is_file():
                add(f'{M0_DOC}/{path.relative_to(report).as_posix()}', path.read_bytes())
        m0_source = BASE / 'work/v43_sim_mechanism_20261030'
        source_paths = [path for path in m0_source.iterdir()
                        if path.is_file() and path.suffix.lower() in {'.py', '.json', '.cjs', '.md'}]
        for directory in ('simulation', 'reconstruction'):
            source_paths += [path for path in (m0_source / directory).rglob('*')
                             if path.is_file() and path.suffix.lower() in {'.py', '.json', '.cjs', '.md'}
                             and not any(part in {'__pycache__', 'output', 'outputs', 'tmp', 'temp', '.cache'}
                                         for part in path.relative_to(m0_source).parts)]
        for path in sorted(set(source_paths)):
            if path.is_symlink():
                raise ValueError(f'M0源码不打包符号链接：{path}')
            relative = f'{M0_CODE}/{path.relative_to(m0_source).as_posix()}'
            add(relative, path.read_bytes())
            m0_source_paths.append(relative)
    status = '''# 当前研究状态

当前任务：WS-V43-WORLD-GT-PROTOCOL-R5-20261030；最小M0实验为WS-V43-SIM-RECON-M0-20261030。V43采用共享证据、多头联合分解与分阶段训练，场景判断不接收编辑指令。GT定义在独立世界表面、对象与编辑干预结果，不预设每颗Gaussian唯一标签；原分支与网页语义为带噪证据，alphaT语义为派生责任，错深度、mixed、无对应、缺测分开。

R5设计、嵌套主架构、数据/GT隔离图、侧车合同及实验矩阵已在本地通过一致性、历史保留和重复生成校验。文档生成新增训练/场景渲染为0。R4档案和旧R2、204帧unknown-white、三锚点/200步child refit均保留；M1/M2改称真值辅助归属/表示修复参考，不是数学上界，额外信息、容量及优化预算单列。

最小M0位于 `''' + M0_ROOT + '''`。当前已知状态：r2仿真数据16帧验证通过，真实GPU smoke待跑；本快照不宣布重建、编辑评价或V43方法完成。程序化非写实车仅作接口与机制验证。oracle body/sky masks、ideal tracks和无噪声稀疏raycast是明确登记的传感器控制，不冒称可部署RGB预测管线。完整网格/稠密深度/表面对象ID和编辑参考与训练白名单隔离。

下一步在真实环境完成当前重建代码的整场景smoke：静态、动态、环境共同训练，保留继承损失，覆盖保存、评价、完整状态恢复和下一阶段实际渲染入口。如后续正式运行达到小时级，先满足约10分钟smoke并单列成本。检查自然污染，不承诺指定错误必出现；先1个最小实际重建，再3–5机制场景，随后小学习/批量。真实三场景无完整GT；场景1仅开发诊断/推理，不训练新可迁移模块；人工/subagent不是完美GT。

环境冻结四路对照保持：原环境/统一白底 × 基线/待测高斯，保存CG、alpha、回填、最终RGB和E/B。冻结参数不等于回填不变。阴影为world road receiver与vehicle effect owner；当前有限采样shadow差分仅是关联效果代理，有车减无车不直接作为纯阴影GT。

**本对话禁止自动AutoDL关机；完成或空闲保留实例，只有用户明确要求才关机。subagent仅gpt-6.1-sol或gpt-6.1-terra。** 发布脚本不调度GPU作业或电源操作，不恢复旧队列。

设计：[V43 Revision5](autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030/index.html)。M0状态须以独立运行记录更新；人工verdict=null；failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。
'''
    if args.status_file:
        status = args.status_file.read_text(encoding='utf-8-sig')
    add('docs/RESEARCH_STATUS.md', status)
    experiment_entries = [
        '- WS-V43-WORLD-GT-PROTOCOL-R5-20261030：[世界表面GT、真实重建污染与部署输入隔离](autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030/index.html)；R5设计修订，文档生成训练/场景渲染0，M1/M2为真值辅助参考。'
    ]
    if args.m0_report:
        experiment_entries.append('- WS-V43-SIM-RECON-M0-20261030：[最小普通场景实际重建与独立参考核验](autoresearch/v43_ownership_20261028/sim_mechanism_20261030/index.html)；具体状态、预算和边界以该报告及运行记录为准。')
    manifest = {'revision': 5, 'task_id': 'WS-V43-WORLD-GT-PROTOCOL-R5-20261030',
                'files': sorted(entries), 'doc_root': DOC, 'code_root': CODE, 'optional_m0_doc_root': M0_DOC,
                'm0_report_included': bool(args.m0_report), 'optional_m0_code_root': M0_CODE,
                'm0_source_files': sorted(m0_source_paths), 'experiment_index_entries': experiment_entries,
                'known_unrelated_dirty_path': 'scripts/run_worldsim_v72_e3_render_bridge.py',
                'r4_doc_root': 'docs/autoresearch/v43_ownership_20261028/joint_direction_r4_20261030',
                'r4_code_root': 'scripts/worldsim_v82/v43_ownership_20261028/joint_direction_r4_20261030',
                'remote_campaign': M0_ROOT, 'new_jobs_started_by_publication': 0, 'power_actions': 0}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf8')
    (HERE / 'publication_manifest.json').write_bytes(manifest_bytes)
    bundle_entries = {**entries, 'PUBLICATION_MANIFEST.json': manifest_bytes}
    (HERE / 'publication.tar.gz').write_bytes(archive_bytes(bundle_entries))
    print(json.dumps({'files': len(entries), 'bundle': str(HERE / 'publication.tar.gz'),
                      'm0_report_included': bool(args.m0_report), 'ssh_executed': False}))


if __name__ == '__main__':
    main()
