"""r37–r41轻量证据、状态与同一失败卡；不把数据检查记作模型收益。"""
from pathlib import Path
import json,shutil,subprocess
P=Path('/root/autodl-tmp/motion_proj_v77');T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    backup=T/'r37/before_status_changes';backup.mkdir(exist_ok=True)
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
        if not (backup/Path(rel).name).exists():shutil.copy2(P/rel,backup/Path(rel).name)
    a=read(T/'r38/instance_quality.json');b=read(T/'r39/instance_quality.json');paired=read(T/'r40/paired_quality.json') if (T/'r40/paired_quality.json').exists() else None
    knowledge={
        'r37':'新18scene预登记后15scene/450帧可用；有序抽样与多地图分派修复，17条SAM轨迹15条技术通过',
        'r38':'21空间轨迹/6scene，补齐39实例评价标签后仅3条同scene普通背景技术候选；冻结代表S016独立QA0，未准入',
        'r39':'真实过去2秒位姿作A的20次候选尝试，2条空间可行，分别缺完整几何/其他帧证据，0准入',
        'r40':('完成固定23例提示帧对照：'+json.dumps(paired['after_counts'],ensure_ascii=False)) if paired else '固定23例仅比较SAM提示帧；修正6条复用标签的实际提示帧/JPEG溯源；26变更任务运行中',
        'r41':'旧440候选池中静止清楚B的7条可解析窗口，相机三秒行程均不足1m；固定视差过程筛选0来源，不提RGB/不放宽阈值'}
    files={
        'r37':['run.json','source_summary.json','source_ready.json','infra_validation.json','map_provenance.json','ordered_sampling_comparison.json','exposure_engineering_fix.json'],
        'r38':['run.json','review_selection.json','independent_quality.json','S016_map_diagnosis.json','local_delivery_validation.json'],
        'r39':['run.json','proposal_audit.json'],
        'r40':['run.json','paired_quality.json','reuse_provenance_validation.json'],
        'r41':['run.json','summary.json']}
    for run,names in files.items():
        (E/run).mkdir(exist_ok=True,parents=True)
        for name in names:
            if (T/run/name).exists():
                if name.endswith('.json'):(E/run/name).write_text((T/run/name).read_text())
                else:shutil.copy2(T/run/name,E/run/name)
        close={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':run,'stage':'in_progress' if run=='r40' and not paired else 'bounded_control_complete',
            'new_knowledge':knowledge[run],'raw_evidence':str(T/run),'host':'wm-3090-1001','training_steps':0,
            'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: '+knowledge[run],
            'new_training_admission':0,'human_verdict':None,'real_DELETE_benefit_demonstrated':False,'shutdown_condition_met':False}
        dump(T/run/'closeout.json',close);dump(E/run/'closeout.json',close)
    for run in ['r38','r39']:
        result=read(T/run/'instance_quality.json');summary={k:v for k,v in result.items() if k!='cases'}
        summary['cases']=[{k:r[k] for k in ['case_id','source_id','scene','split','technical_candidate','reason','identity_flags','scope_flags']} for r in result['cases']]
        summary['full_metrics']=str(T/run/'instance_quality.json');dump(E/run/'input_quality_summary.json',summary)
    prompt=read(T/'r40/prompt_comparison.json');prompt['jobs']=[{k:v for k,v in j.items() if k!='scores'} for j in prompt['jobs']]
    prompt['full_scores']=str(T/'r40/prompt_comparison.json');dump(E/'r40/prompt_comparison.json',prompt)
    result40=('原分母：'+json.dumps(paired['before_counts'],ensure_ascii=False)+'；新分母：'+json.dumps(paired['after_counts'],ensure_ascii=False)+'。身份疑点7→6，但该例仍缺显露依据；无新增技术候选或训练准入，关闭提示帧对照，不推广为全局策略。') if paired else '26条需重新执行SAM，13条复用同样输入和实际提示帧的标签；当前运行中，尚无质量改善结论。'
    result41='在旧440候选池中排除旧scene后203个actor窗口，固定静止B、清楚尺寸、至少16m深度等要求筛至9条，7条有合法30曝光，2条抽样失败。7条相机三秒累计行程仅0.002–0.375m，均不满足预定1–8m运动要求，最终0来源。没有提新RGB、运行SAM或降低阈值；这是既有候选池的缺口，不能推断整个nuScenes不存在该过程。'
    report=f'''# r37–r41：新来源与完整实例标签的有界检查

task `WS-V77-TARGET-PROTECTED-20260929`，wm-3090-1001，v77。旧Q060合法条件QA2保留；本轮没有新增DriveEditor训练或真实DELETE效果结论。

```mermaid
flowchart LR
 M[新scene元数据\n预先划分train/DEV] --> R[30真实曝光\n按location载入地图]
 R --> A[生成A轨迹\n地图/LiDAR/碰撞检查]
 A --> H[最终删除洞H\n原像素冻结]
 H --> Q[完整实例Y-SAM\n仅离线质检]
 Q --> V[独立图像QA\n未合格不准入]
 R --> X[H遮后合法RGB]
 V -.通过后.-> C[条件构建与对照训练]
 X --> C
```

## 来源和工程修复

旧池主要受03/07两个分片限制。本轮事前固定新02/04分片18个scene，按真实B的清晰尺寸、至少16m深度和投影变化筛选，剔除旧83scene/旧r7训练scene；全部为nuScenes官方train，按seed3701划分收集train/DEV。不是最终测试，也不按生成效果挑场景。

逐目标最近曝光可能重复使用同一张图；新入口改为已有有序一对一曝光匹配，维持55ms误差/180ms间隔限制。旧14条已可用窗口像素和时间戳不变，恢复F013；另2条无合法30曝光，F003未过完整几何。最终15scene、450实际解码帧、1028文件含105份LiDAR。提取约169MB，没有整包解压或重新下载数据。

实际来源含Boston、Singapore-onenorth及Hollandvillage。修掉默认Boston地图假设：按source location选择已有v1.3地图，混合城市禁止旧`.road`隐式调用。Holland官方文件有两条道路`[null]`多边形引用，明确记账并不给道路支持；其他悬空引用报错。Boston道路/停车区合并几何与旧版差面积0。空LiDAR返回直接拒绝，不制造地面平面。回归见[r37工程检查](../autoresearch/worldsim_v77/target_protected_20260929/r37/infra_validation.json)。

初始17条真实Y-SAM轨迹15条技术连续性通过。完整Y仅作离线标签与监督，不进入条件；技术mask合格仍不表示新遮挡任务可用。

## r38：空间规则不是图像空间正确性的充分条件

固定旧24位置×0/3/6m/s规则，只换新来源并保留r32主B/邻车角色修复。得到21条空间轨迹、6scene，最终洞H触及39个不同实例标签（6复用、33新增SAM）；逐个检查空标签、两ID同mask、非车辆或不完整位姿及其他帧显露证据。

技术结果：`{json.dumps(a['counts'],ensure_ascii=False)}`。3条普通背景技术候选全部同scene，按事前每scene每family一例选S016。独立QA给0：图像看起来A位于安全岛/草地区域并重叠信号杆，未准入；S017/18没有独立QA，也不放行。

追加空间诊断保留矛盾：S016脚印约96.7%在地图lane内、100%在一条connector内，地图walkway交叠0；真实车框投影与RGB基本对齐。不能据此覆盖图像中的落点疑点，也不能草率断言全局相机错误。地图/稀疏点云通过不等于完整道路语义正确。原图、A脚印投影和独立意见全部保留。

审核页本地`outputs/v77-target-protected-r38/index.html`，三列是真实Y、A与H位置、实际带洞X。灰色是输入，不是补景失败，也不是新模型输出。独立AI分与空白human verdict分开。

## r39：唯一过去位姿对照，停止该无效尝试

不追加地图位置/速度网格，只用当前主B在2秒前真实标注过的位姿提出A：一条世界静止、一条按过去轨迹回放。历史GT仅用于被删A的合成位置，不读取历史RGB，不作为保护车状态证据；无外推/补位姿，Z仍来自当前真实地面。

15来源中5缺前置地面或主分割，10来源共20次位姿检查，仅F006两条空间可行。复用3条完全同源Y评价标签，无新GPU分割。两条分别因不完整保护几何、其他帧证据不足被拒绝：`{json.dumps(b['counts'],ensure_ascii=False)}`。本次唯一2秒对照关闭，不继续搜索时间差。

## r40：提示帧与标签实际溯源

r38中多例不同GT实例的SAM结果大幅重叠。固定r38+r39共23条保存像素/轨迹/H/质量门，仅将提示帧改为GT框光线first-return可见像素最多的帧（256×144排名，0.05m容差；其次可见比例、中点距离、帧序），仍只box prompt，不加点、不调阈值、不后处理mask。几何可见性是选帧代理，不能代替视觉身份判断。

对照前发现6条复用Y-SAM的元数据写的是“新队列计划提示帧”，实际标签却来自旧源f15，而且旧源JPEG质量96，新评价队列质量98。修复复用标签的实际frame/box/输入路径/权重记录，并使新对照读取同一批JPEG字节；未改动旧mask，未把元数据修复计为模型改善。错误计划与第一次队列备份在r40/before_provenance_fix。

{result40}

旧mask的逐像素溯源检查覆盖6条轨迹180帧，均与原r37标签一致；仅修正元数据，证据见r40/reuse_provenance_validation.json。

## r41：先按过程找来源

{result41}

## 当前边界

新数据准入仍0，约50条完整配方未形成。正式A/B/C新增训练0步，身份分支/完整surfel未启动，既有真实DELETE对照保持。用户已允许清可恢复旧物，但盘余约44–45GB，未删除原始数据、权重、失败对照或其他文件。真实DELETE两侧跨scene收益仍未证明，未执行关机。
'''
    (P/'docs/v77/TARGET_PROTECTED_FRESH_SOURCES_R37_R40.md').write_text(report)
    state=f'''# 当前研究状态

2026-10-02，wm-3090-1001，v77。Temporal reveal＋合法projected actor-state；surfel后置。数据盘余约44–45GB，用户允许必要时清可恢复旧物，当前未删除。

r37新增固定15scene/450连续曝光；修复有序抽样、多地图分派/显式null道路及空LiDAR处理，Boston旧几何不变。r38的21轨迹完整实例检查仅3条同scene普通背景技术候选，冻结代表S016独立QA0（安全岛落点疑点）；其余未独立QA，不准入。r39唯一过去2秒真实位姿对照两条可行，但均缺几何或显露证据，关闭该尝试。

r40只在固定23例上比较SAM提示帧。6条复用标签的实际提示帧/JPEG溯源已修复，180旧mask帧逐像素不变。{result40}

r41固定来源过程筛选完成。{result41}

旧Q060输入/条件QA2保留，Q046条件仍1；r36可见一致性过滤未推广。新数据准入0，50条配方未齐，正式A/B/C新增训练0步。不能将来源增加、地图通过或条件指标当成真实DELETE收益；下一步检查来源池每scene只保留3条旧方案是否提前丢掉了所需过程，保留全部质量要求做一次无截断元数据对照。

模型入口仍r21 sam_full_v2，r26裁洞失败不推广。GT相机/轨迹/LiDAR明示POC辅助，Y只监督/质检，human verdict空。没有新增定时任务；真实DELETE保护车和无车背景两侧跨scene收益未达到，关机条件未满足。

报告及组件图：[r37–r41](v77/TARGET_PROTECTED_FRESH_SOURCES_R37_R40.md)；本地`outputs/v77-target-protected-r38/index.html`。继续更新同一[V77-F02](research_failures/entries/V77-F02.md)。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(state)
    path=P/'docs/EXPERIMENTS.md';lines=path.read_text().splitlines()
    for run,know in knowledge.items():
        key=f'| WS-V77-TARGET-PROTECTED-20260929 / {run} |';line=f'{key} {know}；新增训练0 | [收口/状态](autoresearch/worldsim_v77/target_protected_20260929/{run}/closeout.json) |'
        if any(l.startswith(key) for l in lines):lines=[line if l.startswith(key) else l for l in lines]
        else:lines.insert(next(i for i,l in enumerate(lines) if l.startswith('|---'))+1,line)
    path.write_text('\n'.join(lines)+'\n')
    path=P/'docs/research_failures/entries/V77-F02.md';body=path.read_text();title='## r37–r40：新来源空间与实例标签边界'
    if title in body:body=body.split(title)[0].rstrip()
    body+='\n\n'+title+'\n\n新15scene/450帧修复多地图和曝光匹配后仍无新增训练准入。21空间候选中仅3条同scene普通背景技术通过，冻结代表S016独立QA0（安全岛落点疑点）；地图lane覆盖不等于图像空间语义正确。唯一过去2秒位姿对照两例也缺几何/显露依据，不继续时间差搜索。\n\n固定23例提示帧对照前发现复用标签元数据混淆计划frame与实际f15，并可能改变JPEG输入；已修复溯源、保证原JPEG与180帧旧mask不变。'+result40+'\n\nr41：'+result41+'\n\n这是数据/标签/工程问题的有界检查，不是模型路线否定；新正式训练0，没有真实DELETE新收益。详见[报告与组件图](../../v77/TARGET_PROTECTED_FRESH_SOURCES_R37_R40.md)。failure_ledger_delta: updated V77-F02。\n'
    path.write_text(body)
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python',str(P/'scripts/build_research_failure_index.py')],cwd=P,check=True)
    print('RECORDED r37-r41',bool(paired))


if __name__=='__main__':main()
