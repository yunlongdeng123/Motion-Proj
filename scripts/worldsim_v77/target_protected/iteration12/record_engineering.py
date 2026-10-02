"""保存实际完成的工程与条件结果；不代填人工分数，不提前宣称新模型有效。"""
from pathlib import Path
import json,shutil
P=Path('/root/autodl-tmp/motion_proj_v77');T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    notes={
    'A022':'固定f5：旧入口原模型/r7均出现车，新入口两权重都显示空道路与墙面。此为输入工程收益，不能记成微调收益；尚非整视频人工通过率。',
    'A041_w10':'固定f5：新旧入口均有右边缘深色车形/涂抹残留，未解决。',
    'A013':'固定f5：后方白巴士前部仍有箱状几何，未见明确入口改善。',
    'A048':'固定f5：左侧后方车仍有拉伸/涂抹，新旧差别有限。',
    'A007':'固定f5：目标位置为平滑背景，四臂差别有限；隐藏实体是否正确不由此单帧证明。',
    'A034':'固定f5：后车尾部仍不完整/模糊，四臂差别有限。',
    'A042':'固定f5：车形涂抹仍在，输入首帧还有SAM身份不确定；不可当已解决。',
    'A061_w08':'固定f5：r7白车外观仍比原模型较完整，新旧入口差别有限；保留旧人工r7优先排序，不外推整视频。'}
    r21=T/'r21';r22=T/'r22'
    assert len(read(r21/'baseline_state.json')['completed'])==16
    dump(r21/'assistant_output_review.json',{'review_frame':5,'cases':[{'case_id':k,'observation':v,'human_verdict':None} for k,v in notes.items()],'scope':'fixed_frame_observation_not_video_verdict'})
    report='''# r21–r22：修复真实入口，检查条件能提供的实际证据

2026-10-02，wm-3090-1001，v77。task `WS-V77-TARGET-PROTECTED-20260929`，runs r21/r22，failure_ledger_refs [V77-F02]。

**A022的固定帧反例在修复入口后消失：原模型和r7都能删除目标。其余关键保护车/残留问题仍在。**这次没有新微调，不能把入口收益记给网络。

```mermaid
flowchart LR
 R[真实RGB＋SAM] --> H[去掉GT硬裁／完整H与alpha]
 H --> D[原模型与固定r7同条件DELETE]
 X[固定30帧遮后窗口] --> S[重新跑SAM2可见保护车]
 S --> A[局部框表面代理／身份分开]
 X --> N[同窗可见LiDAR背景证据]
 A --> C[删除A后投影O／N／U／Q]
 N --> C
 C --> Q[条件精度／覆盖与独立QA]
 Y[真实Y及完整视频SAM] -.仅评价与未来训练监督.-> Q
 Q --> F[后续补Temporal reveal数据]
```

## 真实入口与整窗支持

另建8例208帧，不覆盖旧70例。保留完整原SAM，不再以GT包络硬裁；相对于SAM，洞外遗漏及目标alpha非1均为0。GT邻车包络只作保守写回与诊断；真实可见保护mask冲突必须拒绝。全SAM覆盖不是SAM正确性证明，A042首帧立柱旁碎片仍不确定。空SAM帧严格零写回。

8例原模型/r7各跑1窗，共16新窗口；同旧固定10帧、576×1024、seed42、25步，保存native与alpha写回。旧视频原样保留。固定f5观察：A022两权重旧入口都生成车，新入口都没有该车；A013巴士结构、A041边缘车形、A048后车涂抹仍在。A061保留用户所见r7局部正例，未发现新入口明确再改善。所有人类0/1/2留空，单帧不能证明视频稳定。

单次30真实曝光、2.9秒、320×576训练前后向通过，80个时间self-attention张量梯度有限非零，峰值14.185GiB；优化0步。整窗3步推理六次网络调用均含30帧，峰值21.453GiB。3步只证明入口可运行，不是效果评估；没有把短窗拼接冒充长窗。继承r19 FP32初始化修复。

按真实Y、H、遮后X及保护标签逐值去重：旧50训练任务实际46个，旧11合成DEV验证实际10个。所有重复原目录保留；共享旧形状的目录不冒称新独立测试。

## 合法状态条件的最小检查

三个旧DEV长窗L001/L007/L009，固定30帧。重新在已擦除的RGB上运行SAM2，4条保护轨迹120帧。状态构建API不接受Y：可见颜色经光线与GT cuboid表面相交进入actor局部坐标；按身份和深度投影，目标A不参与遮挡。不同身份深度冲突进入U，不平均。背景只取同窗实测近地LiDAR的可见RGB，N不能设成1−O。GT相机、轨迹、框、道路地图与LiDAR是明确的POC几何辅助。

遮后输入边界、无证据为U、跨帧显露、移除A后可见性、身份冲突、正证据N六类合同通过。背景平面也只用窗口内扫描重新拟合；缺失扫描没有用窗外数据代替。完整Y与旧全视频SAM只由独立评价程序读取，后者是质量标签而非像素级真值。

| DEV条件 | 洞内O的同身份标签精度 | 被遮B覆盖 | N覆盖全洞：点／局部片 |
|---|---:|---:|---:|
| L001 | 95.93% | 17.80% | 1.28% / 6.50% |
| L007（未遮B） | 不适用 | 不适用 | 0.033% / 0.019% |
| L009 | 99.26% | 36.90% | 2.89% / 6.99% |

局部片对照只改变背景采样：实测地面三角片内取可见RGB，不跨越1.5m世界边长/32px图像边长，不向无支持区外推，可信度从0.8降为0.4。它增加了一些覆盖，但L007仍几乎无洞内证据，没有各例共同收益，因此保留为诊断、不替代默认点条件，不扫阈值。不能把这些覆盖提升当成DELETE收益。

独立gpt-6-sol xhigh、无fast，检查全部3例固定0/15/29帧：L001/L009 uncertain，L007 pass仅针对可见部分；未见明显串车，但所有洞都没有足够证据支持完整补全。人工分数仍为空。简单条件接口与实际模型增益尚未验证，完整surfel未启动。

## 下一步与交付

r23已冻结最多80个scene各一条真实过程来源；不再要求其拥有旧一秒虚拟锚点。当前61条通过完整长窗几何输入检查，正在CPU按需补603个文件；这是来源准备，不是50个已经AI2的数据case。保持背景和mask来源分隔，先补实际显露和合法条件覆盖，再登记A/B/C同预算微调。

本地HTML：`outputs/v77-target-protected-r21/index.html`（8例、48视频、480帧实际解码）；`outputs/v77-target-protected-r22/index.html`（3例、15视频、450帧实际解码）。页面都有简单组件图、同步播放和逐帧定位。Codex文件浏览策略限制此前已记录，未宣称UI播放已验证。

新知识：**A022存在可被统一入口修复消除的固定帧反例；而合法可见条件的主要限制已可量化为证据覆盖，而非仅看投影精度。**这没有否定条件方案，也没有证明新模型有效。资源约50GB空余，没有清理数据或权重；真实保护车与无车背景的跨场景收益/整段人工验收仍未满足，不触发关机。
'''
    (P/'docs/v77/TARGET_PROTECTED_INPUT_STATE_R21_R22.md').write_text(report)
    for run,files in [('r21',['run.json','input_fix_summary.json','assistant_input_review.json','assistant_output_review.json','deduplicated_catalog.json','baseline_plan.json','baseline_state.json','train_30/result.json','infer_30/result.json','review/media_validation.json']),('r22',['run.json','observed_queue.json','condition_quality.json','condition_patches_quality.json','ground_patch_control.json','r22_independent_quality.json','review/media_validation.json'])]:
        E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'/run
        for name in files:
            dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(T/run/name,dest)
        dump(E/'closeout.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':run,'stage':'engineering_or_condition_availability_complete','new_training_steps':0,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02','human_verdict':None,'surfel_started':False,'real_cross_scene_model_gain':False})
    card=P/'docs/research_failures/entries/V77-F02.md';backup=r21/'before_changes/V77-F02_before_r21_r22.md'
    if not backup.exists():shutil.copy2(card,backup)
    addition='''\n\n## r21/r22：入口修复有效，但条件覆盖仍不足

8例208帧另建完整SAM入口，相对SAM的H/alpha遗漏均0；同旧窗口/权重/seed重跑原模型与r7共16窗。固定f5 A022旧两臂补车、新两臂均删除，明确是入口工程收益，不能算微调。A013巴士结构、A041边缘车形、A048后车涂抹仍在；A042首帧mask身份不确定保留。30帧2.9秒单次训练前后向及3步推理通过，优化0步。

遮后重新SAM的4轨迹120帧用于合法actor局部框条件；完整Y只评价。L001/L009洞内O身份代理精度95.93%/99.26%，但被遮B覆盖仅17.80%/36.90%；默认N洞内覆盖1.28%/0.033%/2.89%。局部地面片仅在有支持区域采样后覆盖6.50%/0.019%/6.99%，没有一致收益，保留诊断不推广/不扫阈值。不能用空点当无车或GT隐藏表面补条件。独立6-sol xhigh三例L001/L009 uncertain、L007 pass仅可见部分；不是整洞补全验收。

下一步补实际显露来源，r23冻结最多80scene、已有61长窗输入候选，正式A/B/C与surfel尚未启动。见[报告与组件图](../../v77/TARGET_PROTECTED_INPUT_STATE_R21_R22.md)。failure_ledger_delta: updated V77-F02（入口反例修复与合法条件覆盖边界），不新增ID。\n'''
    if '## r21/r22：' not in card.read_text():card.write_text(card.read_text()+addition)
    status='''# 当前研究状态

2026-10-02，wm-3090-1001，v77。用户授权Temporal reveal＋projected actor-state先验证简单条件价值，surfel后置；允许清可恢复的不重要旧物，目前约50GB空余，无删除。当前实际任务r23为CPU新长窗来源准备：冻结80scene各一条、61例通过长窗输入门槛，按需补603文件；尚非50个已质检训练case。输入源ID别名缺上下文已按八个真实keyframe token修复，不重新抽样。

r21完成8例完整SAM入口/16个同条件原模型-r7对照，A022固定f5旧两臂补车、新两臂均删车，收益属于输入修复。A013/A041/A048等仍失败，A042首帧身份不确定。30帧2.9秒单次前后向和3步推理通过、训练0步，峰值14.19/21.45GiB。旧目录去重为46train/10合成DEV-val，原目录保留。

r22完成遮后4轨迹SAM与3例合法状态条件，L001/L009 O精度高但被遮B覆盖18%/37%，N很稀疏；独立QA uncertain/pass-visible-only/uncertain。局部地面片采样没有各例一致收益，保持默认点条件。当前优先补真正显露过程；新条件尚未接模型训练，A/B/C同预算训练与surfel都未启动。原模型、r7/r14/r18、失败资产全部保留，final未用，无新自动化，当前未满足关机收口条件。

交付与证据：[r21/r22报告](v77/TARGET_PROTECTED_INPUT_STATE_R21_R22.md)、[r23预案](autoresearch/worldsim_v77/target_protected_20260929/r23/plan.md)，failure_ledger_refs [V77-F02]。人工分数空。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(status)
    # HTML固定帧观察明示：不依赖用户展开之前的消息。
    page=r21/'review/index.html';txt=page.read_text();marker='<button id="export">'
    if '固定帧新结果' not in txt:txt=txt.replace(marker,'<p class="warning">固定帧新结果：A022新入口的原模型与r7均删除了目标，旧入口均补出车。A013/A041/A048等仍有问题；这不是新模型收益，也不是视频人工通过结论。</p><p><a href="../v77-target-protected-r22/index.html">查看合法actor-state条件预检</a></p>'+marker);page.write_text(txt)
    print('RECORDED_R21_R22')

if __name__=='__main__':main()
