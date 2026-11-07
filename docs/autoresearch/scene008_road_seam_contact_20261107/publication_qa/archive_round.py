"""交付验收后更新现有研究快照和同一失败卡，保留其他分支现场。"""
from pathlib import Path
import json
import shutil
import subprocess

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
REL='docs/autoresearch/scene008_road_seam_contact_20261107'
LINK='autoresearch/scene008_road_seam_contact_20261107/index.html'
source=RUN/'delivery_release_v2'
target=REPO/REL
assert (source/'publication_manifest.json').is_file()
assert not target.exists(), '不覆盖既有归档'
backup=RUN/'publication_repo_backup'
backup.mkdir(exist_ok=False)
for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V82-F13.md','docs/RESEARCH_FAILURES.md']:
    p=backup/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,p)
shutil.copytree(source,target)
(target/'.gitattributes').write_text('# Preserve immutable research artifacts and original CRLF bytes.\n* -text -filter whitespace=cr-at-eol\n# Generated SVG path line endings are preserved.\n*.svg whitespace=-blank-at-eol\n')

old=(REPO/'docs/RESEARCH_STATUS.md').read_text()
markers=[old.find(value) for value in ('v44＋a5车顶恢复已完成','a5本轮已完成')]
start=min(value for value in markers if value>=0)
end=old.index('\n历史状态保存在')
a5=old[start:end].strip()
new=f'''# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008本轮已完成[路面接缝与车轮遮挡分项验证]({LINK})。红侧得到可复现的道路支撑改进候选，蓝侧跨帧接缝仍开放；不合并为整场景通过/失败，原V44正式版本保留，human_verdict=null。

主候选保持C2/SurfaceLOD车辆、父cellfit1000道路颜色、原生10K、旧道路1K/gates、环境与轨迹不变。10098个世界道路父格在同一切平面细分为40392子格；三轴sigma减半，intrinsic opacity仍.99。仅新道路使用parent-relative Mip幅度归一化，按相机协方差确定有效幅值，与删除指令无关；所有层仍共同进入原生Gaussian排序。没有车体前景优先级、逐像素挖路、车辆整体抬升或新静态退役。

红侧固定raw128的23条车面在前射线，车辆贡献均值.566785→.666535；raw130固定24条.622210→.734828，后者逐条增加且无>.01退化。两帧共23条地面在前射线仍不露出车体。相对V44仍有6/7条严重chart-first缺失，近接触另列；这是当前车面/实测支持地面的成像诊断，不是隐藏轮胎准确率。原生滤波地板与中心深度排序分开解释，不宣称完整Mip-Splatting、2DGS或MOG不透明首交。

蓝侧raw128/delete近似蓝框的1385个原观测道路像素，RGB MAE .031119→.029598；固定两侧6bins的残差跳变.012529→.009794。二者均非整条边界验收。独立的5关键帧世界patch副表显示跨帧回退：raw105/delete/lateral-y 8bins .011334→.019934；raw137/delete/lateral-y 15bins .010415→.015789。KEEP可有相反趋势；主蓝框与世界patch不同分母明确分表，far边界和无配对分桶保持unsupported，不记为0。

分组表显示蓝圈既有新/旧道路交接，也有新道路与暗蓝静态层混合，不能统一归成两套道路颜色不一致。固定6世界basis/18RGB系数的源颜色约束QP只略降蓝框RGB误差，没有实质改善固定接缝指标；该颜色控制和5帧冻结组合仅保留对照，不并入主资产，不继续调权重/步数。下一步应验证实测道路对冲突旧静态表示的持久接管，以及车地查询算子的兼容性，而不是继续补自由颜色。

主AA候选完成49帧双状态；相对父R0的98个状态帧，道路与车外保护MAE均下降，但局部接缝回退及少量a6逐帧微小RGB退化仍披露。raw137固定720像素新道路solo alpha最小.999794，0个低于.98。40392行资产和父来源映射已保存，5关键帧×双状态独立10次重放的全部保存浮点字段及PNG与最终结果逐值相同。删除只移除a37，两种状态共用同一份资产和滤波规则。

本轮1095次native前向，0反向；含92次蓝侧真实短程/正式/线性验证与10次独立重放。正式80F及smoke/正常方程新进程恢复分别记账；颜色QP的smoke与formal三通道各解一次，不能写成没有拟合。原模型、车辆与预训练网络权重更新为0；R1未扩，学习型道路生成先验未训练，R4决策泛化未验证。代码、资产、报告与必要状态已取回核验。failure_ledger_refs=[V82-F13]，failure_ledger_delta=updated:V82-F13。

subagent仅gpt-6.1-sol/max；本轮无待启动GPU队列。按用户要求AutoDL保持开机，不自动关机。RCR-1异常蓝色RGB的原限定采用决定与原V44回退入口保持；本轮几何支撑候选不代表整场景已采用。

{a5}

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
'''
(REPO/'docs/RESEARCH_STATUS.md').write_text(new)
index=REPO/'docs/EXPERIMENTS.md';text=index.read_text()
assert 'scene008_road_seam_contact_20261107_r1' not in text
entry=f'''## Scene008道路接缝与车轮遮挡分项验证（2026-11-07）

- `scene008_road_seam_contact_20261107_r1`：[同面道路细分、滤波归一化、边界材质强控制与49帧分项验收]({LINK})。红侧支撑有增量，蓝侧跨帧接缝仍开放；18系数颜色对照未选入主资产。资源、正式/短程、失败与精确重放见报告。failure_ledger_refs=[V82-F13]，failure_ledger_delta=updated:V82-F13。

'''
index.write_text(text.replace('# 实验索引\n\n','# 实验索引\n\n'+entry,1))
card=REPO/'docs/research_failures/entries/V82-F13.md';text=card.read_text()
assert '道路支撑与接缝分项后续证据' not in text
text+='''
## 道路支撑与接缝分项后续证据（2026-11-07）

同一世界道路细分与parent-relative Mip幅度控制减轻了固定车面在前射线的错误遮挡，地面在前组未被强行露车；但仍有严重缺失，不构成隐藏几何正确或完整首交保证。该成分候选没有更新C2车辆、父道路颜色或原生参数。

接缝证据进一步区分了新/旧道路交接与新道路/暗蓝静态层混合。raw128/delete原蓝框改善，但5关键帧世界patch的部分DELETE横向边界回退，KEEP可相反；18RGB系数的持久边界颜色强控制没有解决分项接缝，未选入主资产。原始RCR-1去蓝采用与本轮成分候选分开，不能用整图MAE下降或保护门通过替代局部接缝验收。见[分项报告](../../autoresearch/scene008_road_seam_contact_20261107/index.html)的固定ROI、双状态、分母和负例；没有新增原训练唯一根因或学习道路先验结论。
'''
card.write_text(text)
(target/'publication_ledger_receipt.json').write_text(json.dumps(dict(task_run='scene008_road_seam_contact_20261107_r1',
    failure_ledger_refs=['V82-F13'],failure_ledger_delta='updated:V82-F13',a5_status_preserved_verbatim=a5 in new,
    source_packet_manifest_preserved=True,human_verdict=None,AutoDL='KEEP_ON'),ensure_ascii=False,indent=2)+'\n')
subprocess.run(['/root/autodl-tmp/envs/drivingeditor/bin/python','scripts/build_research_failure_index.py'],cwd=REPO,check=True)
print('归档和同一账本已更新，未暂存、提交或推送。')
