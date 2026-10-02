"""r28–r30轻量证据收口；保留原始注册和全部失败资产。"""
from pathlib import Path
import json,shutil,subprocess
P=Path('/root/autodl-tmp/motion_proj_v77')
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    backup=T/'r29/before_document_changes';backup.mkdir(exist_ok=True)
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
        dest=backup/Path(rel).name
        if not dest.exists():shutil.copy2(P/rel,dest)
    for run,names in {
        'r28':['run.json','selection_amendment.json','rejection_order_audit.json','unprojected_pose_audit.json','pose_metadata_repair.json','source_split_audit.json','r28_independent_quality.json','instance_quality.json','controller_state.json','local_delivery_validation.json'],
        'r29':['run.json','controller_state.json','condition_evaluation.json','r29_independent_quality.json'],
        'r30':['run.json','controller_state.json','window_comparison.json']}.items():
        for name in names:
            dest=E/run/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(T/run/name,dest)
    for name in ['input_validation.json','candidate_roster.json']:
        full=read(T/'r28'/name)
        if name=='input_validation.json':full.pop('cases')
        else:
            full['cases']=[{k:v for k,v in c.items() if k!='trajectory'}|{'trajectory_key':{k:c['trajectory'][k] for k in ['source_id','lane_token','lane_midpoint_distance_m','speed_mps']}} for c in full['cases']]
        full['full_evidence']=str(T/'r28'/name);dump(E/'r28'/name,full)
    r28={'stage':'complete_one_data_candidate','candidates':87,'sources':30,'new_Y_SAM_jobs':194,'reused_Y_SAM_jobs':19,
         'pixel_contract_frames':2610,'data_QA_2_cases':['Q060'],'training_steps':0,
         'local_review':'outputs/v77-target-protected-r28/index.html','failure_ledger_refs':['V77-F02'],
         'failure_ledger_delta':'updated V77-F02: deterministic roster, unprojected pose metadata, low qualified yield',
         'human_verdict':None,'shutdown_condition_met':False}
    r29={'stage':'complete_qualified_single_case_condition','case_id':'Q060','condition_assistant_grade':2,
         'actual_frames_independently_reviewed':30,'eligible_for_later_training_pool':True,'frozen_dataset_complete':False,
         'training_steps':0,'surfel':False,'failure_ledger_refs':['V77-F02'],
         'failure_ledger_delta':'updated V77-F02: single-case positive legal condition evidence, not model benefit',
         'human_verdict':None,'shutdown_condition_met':False}
    r30={'stage':'complete_do_not_promote_short_window_as_general_fix','new_technical_candidate':['Q046'],
         'independent_QA':'not_admitted_or_reviewed_for_training','training_steps':0,'default_window_frames':30,
         'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: 20-frame quality control does not fill recipe',
         'human_verdict':None,'shutdown_condition_met':False}
    for run,data in [('r28',r28),('r29',r29),('r30',r30)]:
        dump(T/run/'closeout.json',data);dump(E/run/'closeout.json',data)
    report='''# r28–r30：候选复现、单例合法条件与短窗控制

task WS-V77-TARGET-PROTECTED-20260929，wm-3090-1001，v77，failure_ledger_refs [V77-F02]。

本轮修复两个数据工厂工程错误，得到1条经过输入和条件独立质检的reveal候选。**尚未形成50条配方，A/B/C正式训练0步，不能据此宣称真实DELETE改善。**r21完整目标入口继续保留；r26可见邻车洞裁减仍不推广；未启用surfel。

```mermaid
flowchart LR
 Y[真实30帧Y] --> G[固定87条轨迹\n轮廓A与最终H]
 Y --> Q[Y实例标签\n仅离线质量参考]
 G --> Q
 Q --> V[1条输入独立QA2]
 V --> X[最终H遮后RGB]
 X --> S[SAM2可见B]
 X --> B[可见背景与窗口内LiDAR]
 S --> C[actor局部代理\nO / N / U / Q / F]
 B --> C
 C --> R[30帧独立条件QA2]
 R -. 配方仍缺额 .-> A[同预算A / B / C训练]
```

## 修复与原始证据

旧r23只保存首个拒绝原因计数，不能从“73条unreviewed”还原稳定候选。GT对象通过集合遍历，顺序在进程间变化；固定相同几何、只反转对象顺序，两个来源共5条轨迹改变首因。已排序对象并按场景、车道距离和速度冻结语义清单，检查所有帧的未核验实例。清单实际为87条、30scene：已有72条图像逐像素复用，另外15条来自同一原始轨迹搜索域；位置、速度和质量门槛不变。原73条登记和失败日志保留，修订另记selection_amendment。

另有元数据将“二维投影为空”错误写成“三维位姿不存在”。恢复16例中19条case-instance记录的已知位姿，其中7条车辆轨迹本来是完整的；没有外推真正缺失的GT，也没有重跑图像或SAM。此bug属于新r28工厂，不能追认为旧r7/r14模型失败原因。

213个Y评价实例轨迹中194个新跑、19个复用。所有87×30=2610帧检查通过：A影响区被guard覆盖、guard被最终H覆盖、guard外RGB与Y相同、guard内已擦除、H不进ego底部保护带。工程通过不能替代身份/空间/显露质量。

## 数据质量结果

| 判定 | 数量 |
|---|---:|
| 实例身份或可见性不确定 | 29 |
| 碰到第一阶段外对象，或缺少完整保护几何 | 37 |
| 实际遮挡不足 | 4 |
| 实际轮廓过小 | 3 |
| 保护车深度顺序不成立 | 2 |
| 静态前景深度顺序不成立 | 3 |
| 缺少显露过程 | 4 |
| 其他帧证据不足 | 4 |
| protected reveal技术候选 | 1 |

这些是候选质量门槛的判定，尤其“身份不确定”并非已经证明SAM一定错，更不是模型补景失败。唯一Q060 / N171 / scene-0256经gpt-6-sol xhigh独立检查f0/15/29，输入2分；没有启用fast。Y真实小箱车由轻遮逐步变重遮，矩形H会额外遮去部分可见B和道路，后续必须保留未知。

## r29：合法条件的单例正证据

方法从**最终H完整遮后的RGB**重新跑SAM。完整Y及Y-SAM只在独立评价脚本中读取。相机、车辆轨迹、窗口内LiDAR是明示的POC几何辅助，不能宣称全自动估计。当前为cuboid局部表面代理，不是surfel，也没有给隐藏表面自由可学习embedding。

| 相对完整Y-SAM的参考指标 | Q060 |
|---|---:|
| 遮后可见SAM精度 / 召回 | 99.52% / 97.97% |
| 洞内O精度 | 95.30% |
| O覆盖被遮B | 61.26% |
| 有证据N占洞面积 | 4.85% |
| N落在参考B上的像素 | 0 |
| U占洞面积 | 87.21% |

另一次独立复核覆盖全部30帧条带：仍为同一辆小箱车，未见明显串向大货车/树/道路，投影保留稀疏局部外观和大量未知，条件质量2分。允许这个单例进入后续候选池；**Y-SAM不是人工像素GT，局部框面不能证明完整纹理，条件可用不是扩散模型已经获益**。

## r30：缩短窗口没有解决配方缺额

同87轨迹，只取既有f5–24的20个真实曝光（1.9秒），不新搜位置/速度、不改mask或门槛。30帧默认代码先在Q001/Q002/Q023/Q060做真实数据回归，结果保持一致。20帧技术reveal由1变2（新增Q046，Q060保留），但身份不确定由29变36、无显露过程由4变7；缩窗既可能去掉缺失几何，也会失去观测证据。Q046未独立准入，不把它混入已合格池；不将20帧作为全局修复，保留30帧主配置，停止该有界对照。

评价用的Y标签仍由30帧上下文生成，明确只是离线参考；如果后续使用20帧训练，必须重建仅20帧可用的输入SAM和条件，不能携带额外10帧RGB。

## 交付与限制

本地outputs/v77-target-protected-r28/index.html保留全部87例抽帧、按原因确定性选出的9例三列视频，并补Q060三列条件视频。共30视频/900帧解码、所有文件链接与JS语法通过；没有声称浏览器播放已验证。人工评分留空。输入页不是新DriveEditor推理效果页。

来源仍是nuScenes train域工程train/DEV-val，场景split不交叉；这些来源在旧候选目录出现过，不称最终未曝光留出集。原资产、权重、失败记录全部保留；数据盘约46GB可用，本轮无删除。没有自动训练、没有新定时任务；真实跨场景模型收益关机条件尚未满足。

详细证据：[r28收口](../autoresearch/worldsim_v77/target_protected_20260929/r28/closeout.json)、[r29逐帧评价](../autoresearch/worldsim_v77/target_protected_20260929/r29/condition_evaluation.json)、[r30对照](../autoresearch/worldsim_v77/target_protected_20260929/r30/window_comparison.json)。
'''
    path=P/'docs/v77/TARGET_PROTECTED_INSTANCE_STATE_R28_R30.md';path.write_text(report)
    status='''# 当前研究状态

2026-10-02，wm-3090-1001，v77。用户授权修工程问题并推进Temporal reveal＋projected actor-state，surfel后置；允许清可恢复旧缓存。盘余约46GB，本轮无删除。

r28/r29已收口：固定87候选、30scene，复用72图像并补15原搜索域轨迹。修复集合遍历首因不稳定与无2D投影时丢3D位姿。213实例标签、2610帧像素合同完成；86例未过质量门槛，Q060输入独立QA2，并由最终H遮后RGB重新SAM、构建合法条件。全30帧独立条件QA2；相对Y-SAM洞内O精度95.30%、被遮B覆盖61.26%、N占洞4.85%、U占洞87.21%。只有1条可进后续候选池，不是50条配方完成，不是模型增益。

r30同轨迹20帧有界控制完成：reveal技术候选1→2，但身份不确定29→36、无显露4→7；Q046未独立准入，不将短窗全局推广。保留30帧，停止该对照。下一步优先定位proposal为什么偏离已清楚观测的保护车，再补跨场景配方；不重复同组速度网格凑数。

当前模型入口仍r21 sam_full_v2；r26可见邻车洞裁减在A048/A034/A061退化，不推广。r24稀疏条件重采样修复保留，零初始化真实网络前后向已验证。新A/B/C训练0步，身份分支收益未验证，surfel未启动。相机/轨迹/LiDAR为明示POC辅助，Y只质量与监督，人工verdict留空。

交付：[r28–r30报告与组件图](v77/TARGET_PROTECTED_INSTANCE_STATE_R28_R30.md)，本地outputs/v77-target-protected-r28/index.html，87抽帧、30视频/900帧解码与链接/JS通过。r26真实DELETE对照仍保留。相关failure同一[V77-F02](research_failures/entries/V77-F02.md)。没有新定时任务；未达真实DELETE跨场景模型收益，关机条件不满足。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(status)
    path=P/'docs/EXPERIMENTS.md';lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r28 |'):
            lines[i]='| WS-V77-TARGET-PROTECTED-20260929 / r28 | 固定87候选补实例标签，1条输入QA2；修复首因与位姿元数据 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r28/closeout.json) |'
    for run,text in [('r29','Q060最终H遮后合法状态，30帧独立条件QA2；训练0步'),('r30','同87轨迹20帧质量控制，仅增1未准入候选，不全局推广')]:
        if not any(line.startswith(f'| WS-V77-TARGET-PROTECTED-20260929 / {run} |') for line in lines):
            i=next(i for i,s in enumerate(lines) if s.startswith('|---'));lines.insert(i+1,f'| WS-V77-TARGET-PROTECTED-20260929 / {run} | {text} | [收口](autoresearch/worldsim_v77/target_protected_20260929/{run}/closeout.json) |')
    path.write_text('\n'.join(lines)+'\n')
    path=P/'docs/research_failures/entries/V77-F02.md';text=path.read_text()
    title='## r28–r30：候选复现错误修复与单例合法条件'
    if title not in text:
        text+='\n\n'+title+'\n\n集合遍历使同轨迹首拒原因在进程间不稳定；已排序并按所有帧未知实例冻结87条语义清单，保留旧73计数和72素材。无二维投影被误当无三维位姿，恢复16例19条case-instance已知元数据（7条车辆轨迹本来完整），不外推缺失GT。两项属于新工厂工程错误，不据此归因旧微调失败。\n\n213实例标签与2610帧全覆盖/隔离合同完成，质量门槛只留Q060；其余29身份/可见性不确定、37非车或缺几何、其余遮挡/尺寸/顺序/显露不足。Q060输入QA2后从最终H遮后RGB重新构建条件，30帧独立QA2；参考Y-SAM洞内O精度95.30%、隐藏B覆盖61.26%、N压B=0、U仍87.21%。这是单例条件正证据，非真实DELETE收益，也未完成50条配方。\n\n同87轨迹20帧控制技术候选仅1→2，身份不确定29→36、无显露4→7，不作全局修复；新增Q046未独立准入。A/B/C训练0步，surfel未启。保持r21默认与r26反例，人工null，全部原资产保留。见[报告与组件图](../../v77/TARGET_PROTECTED_INSTANCE_STATE_R28_R30.md)。failure_ledger_delta: updated V77-F02；真实收益关机条件未满足。\n'
        path.write_text(text)
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python',str(P/'scripts/build_research_failure_index.py')],cwd=P,check=True)
    print('RECORDED r28/r29/r30')

if __name__=='__main__':main()
