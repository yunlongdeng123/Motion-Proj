"""r42/r43的分母、工程续跑和空间先行数据工厂记录。"""
from pathlib import Path
import json,shutil,subprocess
P=Path('/root/autodl-tmp/motion_proj_v77');T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'


def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    backup=T/'r43/before_changes';backup.mkdir(exist_ok=True,parents=True)
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
        out=backup/(Path(rel).name+'.before_source_record')
        if not out.exists():shutil.copy2(P/rel,out)
    r42=read(T/'r42/index_comparison.json');r43=read(T/'r43/source_summary.json')
    state=read(T/'r43/controller_state.json');space=read(T/'r43/space_summary.json') if (T/'r43/space_summary.json').exists() else None
    labels=read(T/'r43/instance_quality.json') if (T/'r43/instance_quality.json').exists() else None
    qa=read(T/'r43/independent_quality.json') if (T/'r43/independent_quality.json').exists() else None
    complete=labels is not None and qa is not None
    phase='有界来源对照已完成；4候选独立QA为0/0/1/0，均未准入' if complete else (('实例技术检查完成，独立QA尚未执行') if labels else ('空间检查完成，完整实例QA尚未执行' if space else '公共归档按冻结清单提取中'))
    new42='取消每scene前三截断后440→612窗口；原440及r41过滤逐条重现，但固定过程筛选仍0来源；停止将截断作为主因'
    new43='取消固定16m距离代理后6个新scene满足元数据过程；保留全部实际空间门槛并前置到SAM之前；'+phase
    for run,knowledge in [('r42',new42),('r43',new43)]:
        folder=E/run;folder.mkdir(exist_ok=True,parents=True)
        for name in ['run.json','summary.json','index_comparison.json','source_summary.json','source_ready.json','map_provenance.json','space_summary.json','extraction_amendment.json','extraction_regression.json','gpu_plan.json','independent_quality.json','local_delivery_validation.json']:
            src=T/run/name
            if src.exists():(folder/name).write_text(src.read_text())
        record={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':run,'stage':'bounded_control_complete' if run=='r42' or complete else state['stage'],
            'new_knowledge':knowledge,'evidence':str(T/run),'new_training_admission':0,'formal_A_B_C_steps':0,
            'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: '+knowledge,
            'human_verdict':None,'real_DELETE_gain_demonstrated':False,'shutdown_condition_met':False}
        dump(folder/'closeout.json' if run=='r42' else folder/'progress.json',record)
        dump(T/run/('closeout.json' if run=='r42' else 'progress.json'),record)
    if labels:
        dump(E/'r43/input_quality_summary.json',{'counts':labels['counts'],'cases':[{k:r[k] for k in ['case_id','source_id','scene','split','technical_candidate','reason','identity_flags','scope_flags']} for r in labels['cases']],
            'full_evidence':str(T/'r43/instance_quality.json'),'training_admission':0,'human_verdict':None})
    space_text=json.dumps(space,ensure_ascii=False) if space else '尚未完成，不能将6来源写成6个可训练样本。'
    label_text=json.dumps(labels['counts'],ensure_ascii=False) if labels else '尚无完整实例分割结果。'
    qa_text='；'.join(f"{r['case_id']}={r['assistant_grade']}：{r['reason']}" for r in qa['cases']) if qa else '尚未独立复核。'
    report=f'''# r42–r43：按实际空间判断来源，避免过早的距离代理

任务`WS-V77-TARGET-PROTECTED-20260929`，wm-3090-1001，v77。此处是数据工厂控制；未证明条件或真实DELETE模型有新增收益。

```mermaid
flowchart LR
 M[nuScenes train元数据] --> T[清楚B＋静止B／运动ego\n固定30连续曝光]
 T --> F[固定scene split\n按清单提取RGB／LiDAR]
 F --> G[真实道路与贴地\n逐帧A碰撞／ego／尺度]
 G --> Q[仅可行轨迹执行SAM\n完整Y只作离线QA]
 Q --> I[独立输入QA2]
 I --> X[最终H遮后可见输入]
 X --> C[合法actor-state条件\n条件QA2后才对照训练]
```

## 截断控制：保留负结果

{new42}。无截断索引仍保持原每actor可见性、尺寸与遮挡过滤，未扩大到val/test。取消截断新增172窗口，但不是静止清楚B＋运动ego缺口的直接解；不继续排列旧候选次序。

## 用实际空间检查替代距离预筛选

16m是之前为合成A预留空间的启发式代理，不是用户规定的清晰度标准。r43整体移除这一距离代理，不搜索新的距离阈值；B全8关键帧visibility4、100×60最小尺寸、真实30曝光几何、静止位移≤0.5m、相机三秒行程1–8m保持。候选89窗口中67可解析曝光、22抽样失败，最终6scene：预先按scene与location固定代表及5train/1DEV，完整Y只质检和监督。它们均为训练来源，不是最终测试。

下一层不放宽物理门槛。只试每来源旧规则最多24个车道中点和世界静止A，不追加速度／offset网格；所有帧保留地图、真实LiDAR贴地、GT间距、ego保护、尺寸、连续性等检查。现在将这些检查放在昂贵SAM之前，避免先分割物理上无法合成的来源。

空间结果：`{space_text}`

实例结果：`{label_text}`

阶段：{phase}。技术通过不是独立QA2，更不是训练或真实DELETE收益。

独立输入QA：{qa_text}

新知识：碰撞可行与所需遮挡过程不是同一约束。G001/G002没有建立camera→A→指定B顺序，G004洞与来源B分离，G003缺乏同一B的互补显露。静止A、静止B与运动ego本身并不保证reveal。关闭本次固定来源对照；下一步修生成入口的目标关系与显露约束，不在这六个来源上增加位置／速度网格。技术失败仍有效，独立QA1也不能放行。

审核页保留全部4候选、12段三列视频、360个可解码帧。第三列是灰洞输入而不是模型补景；人工verdict均空。Y只作监督／离线质检。页面：本地`outputs/v77-target-protected-r43/index.html`，完整抽帧意见见`independent_quality.json`。

## 工程与资源

旧提取器写死最多两个归档，遇到新6来源对应的04/05/06/09停止；已增加显式允许的冻结分片清单，旧调用默认两分片限制保留。公共包只流式读取所需member，不整体落盘。单路读取较慢，在04完成的记录边界续跑后，剩余最多两路读取；中断时刚开始的下一个分片移到备份，不把部分文件当完成数据。来源、质量规则和模型均未改动。原错误和续跑检查点保存于r43/before_changes，新增CPU预算见extraction_amendment.json。

最终409个必要RGB/LiDAR文件全部找到，无缺失与损坏，落盘70,316,986字节，180个真实RGB帧完成解码。跨05/06归档的成员查询返回码不作为完整性判断，最终逐文件核对为准。提取器回归验证覆盖默认两分片保护、显式分片范围、两路读取、已完成归档续跑。

空间仍约44GiB可用，尚未删除任何历史数据、权重或对照。旧r37–r41报告和S016失败视频保留。此对照的CPU/GPU作业已结束。正式A/B/C训练0步，surfel后置；真实DELETE两侧跨scene收益关机条件没有满足。
'''
    (P/'docs/v77/TARGET_PROTECTED_SPACE_FIRST_R42_R43.md').write_text(report)
    status=f'''# 当前研究状态

2026-10-02，wm-3090-1001，v77。Temporal reveal＋合法projected actor-state，先条件后surfel。

r37–r40已修曝光匹配、多地图/空LiDAR和SAM实际提示帧/JPEG溯源；15新来源尚无新增训练准入。S016独立QA0；r40提示帧改善身份疑点但未新增可用样本，控制关闭。旧Q060输入/条件2保留，Q046条件仍1，r36过滤未推广。

{new42}。r43用实际空间检查替代固定B距离预筛选：6scene来源已冻结，阶段为{phase}。控制器状态`{state['stage']}`；运行目录`{T/'r43'}`。空间统计：{space_text}。实例统计：{label_text}。没有重复作业；按真实进程/状态断点续跑。

当前新训练准入0，50条完整配方未齐，正式A/B/C训练0步；身份分支与完整surfel未启动。GT camera/track/LiDAR明示POC辅助，完整Y仅监督/离线QA，条件仍必须从最终H遮后RGB构造。模型入口保留r21 sam_full_v2，r26裁洞失败不推广。人工verdict空。

下一步先修生成入口中camera→A→指定B及显露关系的前置检查，不再扩本次六来源的位置／速度网格。四例独立QA为0/0/1/0，全部拒绝。报告：[r42–r43与组件图](v77/TARGET_PROTECTED_SPACE_FIRST_R42_R43.md)；当前审核页本地`outputs/v77-target-protected-r43/index.html`，旧r38页面保留。

数据盘约44–45GiB可用，用户允许必要时清不重要旧物，本轮未删除。没有新增定时任务。真实DELETE保护车与无车背景两侧跨scene收益未达到，未满足关机条件。继续更新同一[V77-F02](research_failures/entries/V77-F02.md)。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(status)
    path=P/'docs/EXPERIMENTS.md';lines=path.read_text().splitlines()
    for run,knowledge in [('r42',new42),('r43',new43)]:
        key=f'| WS-V77-TARGET-PROTECTED-20260929 / {run} |';target='closeout' if run=='r42' else 'progress'
        line=key+f' {knowledge}；新训练0 | [记录](autoresearch/worldsim_v77/target_protected_20260929/{run}/{target}.json) |'
        if any(x.startswith(key) for x in lines):lines=[line if x.startswith(key) else x for x in lines]
        else:lines.insert(next(i for i,x in enumerate(lines) if x.startswith('|---'))+1,line)
    path.write_text('\n'.join(lines)+'\n')
    path=P/'docs/research_failures/entries/V77-F02.md';body=path.read_text();title='## r42–r43：来源预筛选与实际空间检查'
    if title in body:body=body.split(title)[0].rstrip()
    body+='\n\n'+title+'\n\n'+new42+'。\n\n'+new43+'。空间结果：'+space_text+'；实例结果：'+label_text+'。独立QA：'+qa_text+'。\n\n边界：无碰撞不能代替camera→A→B遮挡顺序；静止A/B＋运动ego不能保证互补显露。固定六来源已收口，不继续位置／速度网格。禁止将元数据新增来源记成训练准入或真实收益。完整结果及组件图见[报告](../../v77/TARGET_PROTECTED_SPACE_FIRST_R42_R43.md)。failure_ledger_delta: updated V77-F02。\n'
    path.write_text(body)
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python',str(P/'scripts/build_research_failure_index.py')],cwd=P,check=True)
    print('SOURCE_CONTROLS_RECORDED',phase,flush=True)


if __name__=='__main__':main()
