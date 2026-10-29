"""校验并发布当前诊断包；不修改旧实验或用户已有工作。"""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/editable_ownership_20261030')
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DOC = 'docs/autoresearch/v43_ownership_20261028/editable_ownership_20261030'
CODE = 'scripts/worldsim_v82/v43_ownership_20261028/editable_ownership_20261030'
TEMPLATE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/code/diagnostic_templates/v43_semantic_gpu_v2/editable_ownership_20261030')

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()

manifest = json.loads((BASE/'publication_manifest.json').read_text())
expected = {f['path']: f for f in manifest}
assert git('diff', '--name-only') == 'scripts/run_worldsim_v72_e3_render_bridge.py'
with tarfile.open(BASE/'publication.tar.gz', 'r:gz') as tar:
    members = tar.getmembers()
    assert len(members) == len(expected)
    for member in members:
        assert member.isfile() and member.name in expected
        destination = (REPO/member.name).resolve()
        assert destination.is_relative_to(REPO)
        assert member.name.startswith((DOC+'/', CODE+'/'))
        data = tar.extractfile(member).read()
        assert len(data) == expected[member.name]['bytes']
        assert hashlib.sha256(data).hexdigest() == expected[member.name]['sha256']
        if destination.exists():
            prior = destination.read_bytes()
            if destination.suffix in {'.py','.json','.md','.html','.svg'}:
                prior = prior.replace(b'\r\n', b'\n')
                prior = b'\n'.join(line.rstrip(b' \t') for line in prior.split(b'\n'))
            if prior != data:
                assert destination.name in {'publish_remote.py','prepare_publication.py'}, 'Refuse unrelated overwrite: '+member.name
                saved = BASE/'publication_source_before'/destination.name
                saved.parent.mkdir(exist_ok=True)
                if not saved.exists(): saved.write_bytes(destination.read_bytes())
            destination.write_bytes(data)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

backup = BASE/'publication_status_before'
backup.mkdir(exist_ok=True)
for name in ['RESEARCH_STATUS.md', 'EXPERIMENTS.md']:
    target = backup/name
    if not target.exists(): shutil.copy2(REPO/'docs'/name, target)

status = '''# 当前研究状态

当前任务：WS-V43-EDITABLE-OWNERSHIP-20261030。V43修订3保留三大模块及内部串联布局，补齐语义/实例或表面/运动/证据四字段、混合支撑、几何修复、独立阴影effect及编辑后贡献核验。场景1是后续分类器的推理与诊断对象，本轮分类器训练0、原模型训练0；仅14个诊断子高斯颜色和opacity累计200步短拟合。

已完成49帧CPU证据审计、raw98/113/114三锚帧原生CUDA贡献探针、全候选空间支撑审计，以及raw113的7父→14子两侧保留对照。原生完整基准与已有Actor10K PNG逐像素一致。删Actor后静态在车身域的可见贡献从约216/313/318增至1998/2192/2213；重新遮挡合成后的来源与删除前不同。三个锚帧452/777/810个投影双域候选中，没有候选的两侧实测质心同时处于父高斯3σ椭球内；该描述是深度/支撑不相容警报，非语义真值或全场景错误率。

两侧截断矩初始化与子高斯短拟合可改善无编辑RGB保真，但删Actor后假设两侧分离比例由0.3976降到0.3851，没有证明归属或编辑改善。因此当前不得将投影双域直接视为可拆分几何，更不能只依语义概率硬路由混合父点。下一步先核验实测深度与表面可观测性，修复不相容几何或重新建模，再作保留双侧的局部拆分与编辑后验证；阴影在固定道路接收面上的effect另行验证，不将阴影等同车体实体。

短拟合r1实际200步并保存状态，恢复验收遇到NumPy uint32 RNG兼容错误；修复后r2精确恢复200步状态、增加更新0，11项恢复检查通过。24项拆分数学检查、49张PNG与浮点显示核对、28项局部指标复算、4项两域比例复算通过。浏览器交互QA未完成；human_verdict=null、quality_accepted=false。这不是小时级正式训练，也没有新分类器、跨场景泛化或联合优化优于流水线的结论。

报告：[场景1身份与表示修复](autoresearch/v43_ownership_20261028/editable_ownership_20261030/index.html)；设计：[V43修订3](autoresearch/v43_ownership_20261028/editable_ownership_20261030/V43_DESIGN.md)。原R2和已交付unknown白材质六组204帧不变，新报告中的R2删行+原环境是来源对照，不替代已接受的局部白底R2。主工作台与场景1只新增独立报告入口。完整浮点、checkpoint和失败现场已取回并校验归档；代码同步独立远端模板。无其他实验排队。failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。
'''
(REPO/'docs/RESEARCH_STATUS.md').write_text(status, encoding='utf8')
index = REPO/'docs/EXPERIMENTS.md'
text = index.read_text()
row = '| WS-V43-EDITABLE-OWNERSHIP-20261030 / probe-anchors-r1 + child-refit-r2 | 49帧证据审计、3锚帧after贡献与空间支撑、7父14子保留双侧及200步短拟合；编辑后分离未改善，分类器/原模型训练0 | [报告与组件图](autoresearch/v43_ownership_20261028/editable_ownership_20261030/index.html)、[设计](autoresearch/v43_ownership_20261028/editable_ownership_20261030/V43_DESIGN.md) |\n'
if 'WS-V43-EDITABLE-OWNERSHIP-20261030' not in text:
    text = text.replace('|---|---|---|\n', '|---|---|---|\n'+row, 1)
index.write_text(text, encoding='utf8')
TEMPLATE.mkdir(parents=True, exist_ok=True)
for p in (REPO/CODE).iterdir():
    if p.is_file():
        out = TEMPLATE/p.name
        if out.exists() and out.read_bytes() != p.read_bytes():
            saved = BASE/'publication_template_before'/p.name
            saved.parent.mkdir(exist_ok=True)
            if not saved.exists(): shutil.copy2(out, saved)
        shutil.copy2(p, out)
receipt = dict(passed=True, verified_files=len(expected), repo=str(REPO), template=str(TEMPLATE),
    original_model_training_updates=0, classifier_training_updates=0, child_refit_updates=200,
    old_r2_and_unknown_white_unchanged=True, published_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(BASE/'source_deployment.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps(receipt))
