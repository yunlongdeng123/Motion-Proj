"""按职责收口r31–r36，完整失败资产仍在原run目录。"""
from pathlib import Path
import json,shutil,subprocess
P=Path('/root/autodl-tmp/motion_proj_v77');T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    backup=T/'r36/before_document_changes';backup.mkdir(exist_ok=True)
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
        if not (backup/Path(rel).name).exists():shutil.copy2(P/rel,backup/Path(rel).name)
    names={
        'r31':['run.json','controller_state.json','gate_detail.json'],
        'r32':['run.json','controller_state.json','comparison.json','role_regression.json','r32_independent_quality.json'],
        'r33':['run.json','controller_state.json','r33_independent_quality.json','engineering_error.json'],
        'r34':['run.json','controller_state.json','condition_evaluation.json','actor_identity_evaluation.json','r34_independent_quality.json','r34_initial_review_pack_quality.json','review_provenance_check.json','old_sheet_crop_source_match.json','local_delivery_validation.json'],
        'r35':['run.json','controller_state.json'],
        'r36':['run.json','controller_state.json','condition_evaluation.json','actor_identity_evaluation.json','paired_comparison.json','r36_independent_quality.json','Q060_review_provenance_check.json','Q046_review_provenance_check.json','local_delivery_validation.json']}
    summaries={
        'r31':('complete_role_gate_diagnosis','同87候选拆分主显露B与轻触邻车，定位旧门槛语义错误'),
        'r32':('complete_input_role_fix','同87轨迹技术1→3；Q046输入QA2，Q035因小而模糊QA0；旧Q060保留'),
        'r33':('complete_bounded_negative','372固定静止A解只得同scene P001，独立QA0，停止同批位置搜索'),
        'r34':('complete_condition_grade1','Q046主B身份可用、邻车边缘溢出，条件1；修复审核图纵横比'),
        'r35':('complete_do_not_relax_visibility','150投影拒绝中0例仅由水平截边导致；不放宽ego/尺度约束'),
        'r36':('complete_do_not_promote_globally','合法轮廓反证数值改善，但Q046仍QA1且损失正确投影；不推广、不调网格')}
    for run,(stage,knowledge) in summaries.items():
        close={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':run,'stage':stage,'new_knowledge':knowledge,
               'raw_evidence':str(T/run),'host':'wm-3090-1001','training_steps':0,'surfel':False,
               'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: '+knowledge,
               'human_verdict':None,'real_DELETE_benefit_demonstrated':False,'shutdown_condition_met':False}
        dump(T/run/'closeout.json',close);dump(E/run/'closeout.json',close)
        for name in names[run]:
            src=T/run/name
            if src.exists():
                dest=E/run/name;dest.parent.mkdir(exist_ok=True,parents=True)
                if name=='gate_detail.json':
                    data=read(src);data['cases']=[{k:v for k,v in c.items() if k!='process'} for c in data['cases']]
                    data['full_evidence']=str(src);dump(dest,data)
                else:shutil.copy2(src,dest)
    for run,name,omit in [('r31','alignment.json',['cases']),('r33','proposal_audit.json',['attempts']),('r35','projection_diagnosis.json',['cases'])]:
        data=read(T/run/name)
        for k in omit:data.pop(k,None)
        # 完整逐解参数留原run；提交固定计数及证据地址，不复制重资产。
        if run=='r33':data={k:v for k,v in data.items() if k in ['counts','accepted','attempted','sources','count']}
        data['full_evidence']=str(T/run/name);dump(E/run/name,data)
    report='''# r31–r36：保护车角色、投影边缘与可见证据控制

task `WS-V77-TARGET-PROTECTED-20260929`，wm-3090-1001，v77；failure：V77-F02。

本轮修复一项数据准入语义错误和审核图显示错误。增加一例输入QA2的Q046，但其条件仍QA1，不能把技术通过当训练通过。Q060条件保持QA2。**没有新DriveEditor推理或正式A/B/C训练，没有真实DELETE新收益，surfel未启动。**

```mermaid
flowchart LR
 Y[真实30帧Y] --> Q[输入质量检查\n主B／仅保留邻车]
 Y --> H[合成A与最终洞H]
 H --> X[H遮后RGB]
 X --> S[可见SAM＋冻结位姿]
 S --> C[局部代理状态\nO / N / U / F / Q]
 C --> R[跨帧可见轮廓反证\n一次固定控制]
 Y -.仅评价.-> V[逐身份／逐帧检查]
 C --> V
 R --> V
 V -.配方不足／条件未全合格.-> A[后续同预算A/B/C训练]
```

## r31/r32：主显露对象不能与轻触邻车混为一谈

同87条候选中39条有足够的主车遮挡量、24条同时有遮挡变化；另有身份与第一阶段非车对象问题，不是排序一改就都能用。旧门槛要求每辆被H触及的B都至少30%遮挡且发生显露，误拦轻触邻车。修正为至少一个主B满足原显露条件，所有受影响B继续接受身份、深度及其他帧证据检查。

位置、30帧、所有像素与H不改；旧默认回归Q035/Q046/Q060五组主要结果完全一致，新角色规则在12个过身份/空间门的例上检查，Q066/Q067/Q068仍因其他B缺证据拒绝。87例技术候选1→3：旧Q060保留；新增Q035由独立QA判0（实际主B约57×38像素，模糊，不能借邻白车代替证据）；Q046输入2分，进入合法条件检查。其余29身份不确定、37非车或几何缺失、5无显露、3尺度、2深度、3静态前景、5证据不足。

## r33/r35：静止A求解不能靠放宽空间约束救活

r33围绕原指定清楚B，在f0/15/29左右边各一次反解A深度和遮挡边缘；静止A世界位姿固定，最多六个解/B，不增加速度网格。372个尝试仅1条P001技术通过，仍在Q046同scene。独立输入QA0：两个主B中一个只在早期看到部分车身，f8后全遮；58.1%框面格支持不能证明隐藏纹理真的见过。另一个清楚B不能补足第一个B的证据。停止本次位置求解，不据此否定模型。

保存P001审核图时因新增主B不在旧curated列表触发StopIteration；已使用完整保留实例框查找，备份错误脚本与日志后断点续跑，没有重复已完成source。

r35只诊断原150条size/border/ego拒绝，同位置不重新搜索。134条触ego底部、133条尺度/面积不合适、72条接近/穿越相机平面、98条可见比例不足，原因重叠。**只有水平截边失败的0条**，因此放宽入画/出画要求不能解决这批位置，原空间要求保留。

## r34：来源正确，显示图错误；条件边缘确有风险

Q046从最终H遮后RGB重新SAM，再用固定GT相机/轨迹与窗口内LiDAR构建cuboid代理。全Y只离线评价。主B在30帧保持同一身份，洞内同实例精度95.30%、隐藏部分覆盖54.67%；两辆轻触邻车却只有26.54%／57.75%边缘精度，联合90.83%掩盖了问题。地面正证据不足，N保持0，洞内U94.06%，未把未知填成背景。

旧审核条带将多车宽crop强行拉正方形，独立QA一度疑似源错配。核验30帧H外X与Y逐像素相等、H内127；旧Y格与对应正确source裁剪仅JPEG误差（MAE1.06–1.28）。已保存旧条带/初始QA，修为保比例逐actor条带及全景定位；条件与指标未改变。独立重审条件1分：主身份正确，但邻车末段有边缘过投影，不能计入2分训练条件。

## r36：一次合法轮廓反证控制，有收益也有代价

针对框面代理误投，借鉴[Space Carving一手论文](https://homes.cs.washington.edu/~seitz/papers/kutu-ijcv00.pdf)的可见观测一致性思路：将已有actor局部点投向30帧，只有点未落在H及其2px边缘、未被保留对象遮挡时才判断；落到本实例SAM的2px容差外则剔除。冻结0容许反证、0.15m深度容差；无Y输入、无新增表面、无填U，背景N逐帧保持不变。这只是动态代理点的诊断，不是该论文算法的完整实现或保证。

| case | O洞内精度：原→过滤 | 隐藏B覆盖：原→过滤 | 独立条件评分 |
|---|---|---|---|
| Q060 / scene-0256 | 95.30%→96.44% | 61.26%→61.00% | 2→2 |
| Q046 / scene-0714 | 90.83%→93.78% | 55.80%→51.69% | 1→1 |

Q046两邻车精度分别26.54%→76.46%、57.75%→66.39%，但仍有误投。主B覆盖54.67%→50.49%；被删2600个主B投影像素中，2397原本落在近似Y-SAM B上。预登记的精度不降/覆盖保留至少90%数值门通过，**独立质量并未两例同时2分，因此不推广全局默认，不追加halo/票数阈值网格**。Q060这个单例仍是合法条件正证据；不能把UNKNOWN增加和小规模精度提升当成补景任务成功。

## 交付、验证和剩余边界

- 本地`outputs/v77-target-protected-r34/index.html`：Q035/Q046/P001明确主B、邻车、A与H，12视频360帧；输入和条件不冒充生成结果。
- 本地`outputs/v77-target-protected-r36/index.html`：两例真实Y/带洞输入/原条件/过滤后条件，8视频240帧；保留全部逐actor30帧与独立评分。人工verdict为空。
- 两页文件链接、JS语法和视频实际解码检查完成；没有声明浏览器播放验证。方法合同覆盖H未知、他车遮挡不作负证据、原样本不改；60帧N逐像素不变，O/N/U互斥完整。默认状态构建保留，过滤仅显式参数启用。

当前合格池仍不足约50条配方。旧高运动来源偏向近车与快速接近，静止A全窗可用空间有限；后续应先补有物理空间且保护车真实可见的来源窗口，再生成遮挡，不能用更密的旧位置网格补数量。真实r21入口、r26反例、原/r7权重和所有失败完整保留。数据盘约46GB可用，无文件删除；GPU无新训练。未达到真实DELETE两侧跨scene收益，关机条件不满足。

轻量登记与收口：[r32](../autoresearch/worldsim_v77/target_protected_20260929/r32/closeout.json)、[r34](../autoresearch/worldsim_v77/target_protected_20260929/r34/closeout.json)、[r36配对](../autoresearch/worldsim_v77/target_protected_20260929/r36/paired_comparison.json)。完整图像/NPZ/逐解日志在对应远端run，不以轻量索引冒充备份。
'''
    (P/'docs/v77/TARGET_PROTECTED_ROLES_CONSISTENCY_R31_R36.md').write_text(report)
    status='''# 当前研究状态

2026-10-02，wm-3090-1001，v77。推进Temporal reveal＋合法projected actor-state，surfel后置。用户允许清可恢复旧缓存；盘余约46GB，未删除文件。

r31–r36已完成有界工程/条件控制：修复主显露B与轻触邻车混淆，原87轨迹不变技术1→3；Q035主车小模糊QA0，Q046输入QA2。Q046最终H遮后条件主B身份稳定，但两邻车边缘误投，条件QA1；保比例审核图修复并证明无源错配，旧图/初始QA保留。旧Q060输入与条件QA2仍保留。

r33围绕清楚B求解372个静止A位置，唯一P001仍缺真实显露证据，QA0；r35证明150投影拒绝中0例仅水平截边，不放宽ego/尺度，不再追加同批位置或速度网格。r36只用可见SAM作一致性反证，两例O精度提高且总覆盖保留≥90%，但Q046仍QA1并损失正确主B投影，故不推广过滤器，不做阈值网格。

50条配方仍不足，A/B/C正式训练0步，尚未验证模型利用条件的收益。下一步补有物理空间、保护车真实可见的来源窗口，先联合检查空间与显露；保持原scene split、输入隔离、30帧同一次调用和逐例独立QA，不能用proxy或单例结果替代跨scene收益。

当前模型入口仍r21 sam_full_v2；r26邻车洞裁减不推广；r24稀疏状态重采样修复保留。GT相机/轨迹/LiDAR明示POC辅助，Y只监督和评价，人工verdict空。真实DELETE保护车与无车背景两侧跨scene收益未证实，关机条件未满足；没有新增定时任务。

交付：[本轮报告与组件图](v77/TARGET_PROTECTED_ROLES_CONSISTENCY_R31_R36.md)，本地outputs/v77-target-protected-r34/index.html（12视频360帧）与r36/index.html（8视频240帧），链接/JS/实际解码通过。failure更新同一[V77-F02](research_failures/entries/V77-F02.md)。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(status)
    path=P/'docs/EXPERIMENTS.md';lines=path.read_text().splitlines()
    for run,(_,knowledge) in summaries.items():
        key=f'| WS-V77-TARGET-PROTECTED-20260929 / {run} |'
        line=f'{key} {knowledge}；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/{run}/closeout.json) |'
        if any(l.startswith(key) for l in lines):lines=[line if l.startswith(key) else l for l in lines]
        else:lines.insert(next(i for i,l in enumerate(lines) if l.startswith('|---'))+1,line)
    path.write_text('\n'.join(lines)+'\n')
    path=P/'docs/research_failures/entries/V77-F02.md';body=path.read_text();title='## r31–r36：保护角色门槛修复，条件精度与覆盖不能互相代替'
    if title not in body:
        body+='\n\n'+title+'\n\n同87轨迹把主显露B与轻触邻车分开，技术1→3；Q035主车小模糊QA0，Q046输入2但条件1。Q046审核图宽crop拉方形造成假源错配观感；30帧逐像素输入核验与旧crop匹配排除实际源错配，保比例重审确认真实问题为邻车边缘误投。旧图/初始QA完整保留。\n\n372个静止A解仅P001技术通过，独立检查发现一辆主B只早期部分可见，QA0；几何格支持不能当隐藏纹理观测。150投影拒绝中0例只水平截边，不放宽ego/尺度、不加旧搜索网格。\n\nr36合法可见轮廓反证提高两例投影精度，但Q046主B覆盖54.67%→50.49%且条件仍1，Q060保持2；不以数值门通过替代逐例质量，不推广全局过滤、不调阈值。仅条件控制，正式训练0步、surfel未启、没有真实DELETE新收益。应补物理可用来源与真实显露过程，不继续无限解释同批来源。详见[报告与组件图](../../v77/TARGET_PROTECTED_ROLES_CONSISTENCY_R31_R36.md)。failure_ledger_delta: updated V77-F02；关机条件未满足。\n'
        path.write_text(body)
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python',str(P/'scripts/build_research_failure_index.py')],cwd=P,check=True)
    print('RECORDED r31-r36')


if __name__=='__main__':main()
