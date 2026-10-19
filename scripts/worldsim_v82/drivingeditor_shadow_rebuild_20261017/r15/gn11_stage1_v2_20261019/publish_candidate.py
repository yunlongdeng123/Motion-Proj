"""Copy only this candidate/report; update current status and the existing failure card.

Does not stage, commit, push, call SSH or execute training. Review staged diff separately.
"""
from pathlib import Path
import datetime,json,shutil

W=Path(__file__).resolve().parent;BASE=W.parent.parent
O=BASE/'outputs/gn11_stage1_v2_20261019'
REPO=W.parent/'r15_closure_run/publish_repo'
DOC=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019')
CODE=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v2_20261019')
def long(p):return Path(chr(92)*2+'?'+chr(92)+str(p.resolve()))
def write(path,text):long(path).parent.mkdir(parents=True,exist_ok=True);long(path).write_text(text,encoding='utf-8')
def copy(src,dst):long(dst).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(long(src),long(dst))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
record=dict(task='WS-V82-R15-GN11-STAGE1-V2-20261019',run='gn11_stage1_v2_seed0',
    state='LOCAL_CANDIDATE_CPU_CHECKED_AWAITING_USER_POWER_ON',updated_utc=now,
    baseline='A fresh GN11 native firststage30K five identities',seed=0,updates_authorized=30000,
    updates_executed=0,second_stage_updates=0,actual_GPU_gate_passed=False,
    actors=[6,37,5,26,18],raw_frames=list(range(89,138)),human_verdict=None,
    input_roles={'RGB_masks_initialization':'same original A49 inputs, runtime binding pending',
                 'LiDAR_poses_calibration':'world-road geometry and measured visibility support',
                 'dataset_3D_boxes':'selected vehicle volume exclusion and all-object occlusion',
                 'donor_RGB':'user approved unannotated raw89 different-position asphalt prior',
                 'evaluation':'same49 training reconstruction, not heldout'},
    resources={'host':'wm-vgpu-1011','GPU_count_planned':1,'GPU_memory_admission_pending':True,'disk_admission_pending':True},
    failure_ledger_refs=['V82-F06'],failure_ledger_delta='candidate_and_CPU_evidence_only_no_new_scientific_verdict',
    report='index.html',code='candidate_source.zip')
write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
paths=[]
for src in sorted(O.iterdir()):
    if src.is_file() and src.name!='github_publish.json':
        rel=DOC/src.name;copy(src,REPO/rel);paths.append(rel.as_posix())
for src in sorted(W.rglob('*')):
    if src.is_file() and '__pycache__' not in src.parts and (src.suffix in {'.py','.json'} or src.name=='LICENSE_gaussian_splatting.md'):
        if src.name in ['publish_paths.json','github_publish.json']:continue
        rel=CODE/src.relative_to(W);copy(src,REPO/rel);paths.append(rel.as_posix())
status=REPO/'docs/RESEARCH_STATUS.md'
old=status.read_text(encoding='utf-8')
if 'task=WS-V82-R15-GN11-STAGE1-V2-20261019' in old:raise RuntimeError('candidate already published; review subsequent status update instead')
paragraphs=old.split('\n\n',2)
if len(paragraphs)!=3:raise RuntimeError('unexpected current status format')
current=f'''# 当前研究状态

更新：{now}。当前 task=WS-V82-R15-GN11-STAGE1-V2-20261019；本地候选与15项CPU检查完成，等待用户开机后绑定真实输入及执行CUDA零更新检查。新版GPU训练0步，尚无效果结论。

用户授权相同五身份a6/a37/a5/a26/a18、相同49帧raw89–137、相同原始输入初始化和seed=0，重新训练第一阶段30K；第二阶段0，不从旧30K继续追加30K。归属与道路重建分为两套目标：静态关闭时间运动/显隐并限制车身监督通路；道路以raw89车后Query Patch查询raw131–137同世界位置，必要时检查同窗其余帧，无合格真实候选才使用已确认左右沥青供体，真实/先验/未知分开记录。3D数据集标注框作为新增归属体积/遮挡约束输入明确披露。两目标分别计算梯度，每步一次原生更新，不新增预算。

左右供体坐标以480×320原图为准：(49,225)–(185,284)与(352,227)–(398,298)。已生成全幅定位与标线过滤候选预览，未执行真实世界Query Patch融合或补全。真实LiDAR/相机绑定、全部遮挡核验、同输入fresh初始化、CUDA前后向及磁盘准入仍待执行；本地检查不等于修复质量通过。开机后通过这些检查才进入已授权30K，完整状态每5K保存，终点完整49帧零更新评估并实际看静态/删除图和动态保持情况。

[候选实现、简单组件图与全幅供体预览]({DOC.relative_to('docs').as_posix()}/index.html) · [执行范围]({DOC.relative_to('docs').as_posix()}/task_record.json)。failure_ledger_refs=[V82-F06]，未新增失败ID或科学结论，human_verdict=null。AutoDL上轮已关机；本轮未连接或启动GPU，旧automation-2保持暂停，不恢复旧队列。

## 已完成的原A/B基线与历史证据

'''
write(status,current+paragraphs[2]);paths.append('docs/RESEARCH_STATUS.md')
exp=REPO/'docs/EXPERIMENTS.md';text=exp.read_text(encoding='utf-8')
anchor='|---|---|---|'
row=f'| WS-V82-R15-GN11-STAGE1-V2-20261019 / gn11_stage1_v2_seed0 | 五身份第一阶段v2：静动态归属与同世界道路查询；候选代码、全幅供体预览与固定30K协议 | [组件与证据]({DOC.relative_to("docs").as_posix()}/index.html) |'
if row in text:raise RuntimeError('duplicate experiment registration')
write(exp,text.replace(anchor,anchor+'\n'+row,1));paths.append('docs/EXPERIMENTS.md')
card=REPO/'docs/research_failures/entries/V82-F06.md';text=card.read_text(encoding='utf-8')
heading='## 2026-10-19：GN11第一阶段v2双模块候选与CPU检查'
if heading in text:raise RuntimeError('duplicate failure card section')
addition=f'''

{heading}

针对已确认静态表示/监督通路漏洞与道路观测缺口，用户授权同五身份、同49帧和初始化fresh第一阶段30K。候选将静态位置/时间可见性固定，车身RGB/深度等监督路由到动态；选定3D标注车框体积内排除静态不透明度并保护道路薄层。道路模块另以世界曲面查询后续真实曝光，缺真实候选才使用用户确认的raw89左右供体；先剔标线，再按道路坐标重采样，不整框复制、不称异位置先验为真值。两套目标独立计算与记录，限制道路梯度只进静态，每步一次更新。动态完整mask alpha MSE保留。

15项CPU行为检查通过，源码覆盖和动态类方法AST核验完成；真实raw89世界投影/曝光比例、CUDA实际前后向与训练效果均尚未验证，新增GPU更新0。全幅输入预览的颜色候选不是实际几何合格供体。3D中心体积项不能保证大splat尾部/错误深度车形已消失，先验可能出现纹理重复/接缝；需实际30K后的49帧分层/删除图检验，不能凭实现宣称漏洞已完全修复。原A/B历史资产和终态报告保持。

[候选组件、精确供体坐标与检查凭据](../../{DOC.relative_to('docs').as_posix()}/index.html)。human_verdict=null；failure_ledger_delta=candidate_and_CPU_evidence_only_no_new_scientific_verdict。
'''
write(card,text+addition);paths.append('docs/research_failures/entries/V82-F06.md')
(W/'publish_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(paths),state=record['state'],training_updates=0)))
