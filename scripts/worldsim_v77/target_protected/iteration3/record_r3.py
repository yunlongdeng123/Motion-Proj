"""归档本轮真实来源、失败窗口和新造数结果，更新同一V77-F02。"""
import argparse,json,shutil
from pathlib import Path

def read(p):return json.loads(p.read_text())
def main(root,repo):
    s=read(root/'summary.json');ev=repo/'docs/autoresearch/worldsim_v77/target_protected_20260929/r3';ev.mkdir(parents=True,exist_ok=True)
    for src,name in [('summary.json','summary.json'),('run_config.json','run_config.json'),('sam2_runtime.json','sam2_runtime.json'),('still_mask_reviews.json','still_mask_reviews.json'),('temporal/preparation.json','centered_window_control.json'),('temporal_windows/preparation.json','window_preparation.json'),('temporal_windows/independent_mask_reviews.json','temporal_mask_reviews.json'),('temporal_windows/mask_review/mask_audit.json','temporal_mask_audit.json'),('review/independent_synthetic_reviews.json','synthetic_reviews.json'),('review/delivery_validation.json','delivery_validation.json'),('factory/synthesis_config.json','synthesis_config.json'),('factory/assembly.json','assembly.json'),('placement_control.json','placement_control.json'),('coupled_source_control.json','coupled_source_control.json')]:
        (ev/name).write_text((root/src).read_text())
    p=read(root/'factory/pair_candidates.json');(ev/'proposal_summary.json').write_text(json.dumps({k:v for k,v in p.items() if k!='selected'},ensure_ascii=False,indent=2)+'\n')
    clips=read(root/'temporal_windows/source_manifest.json')['clips'];(ev/'source_windows.json').write_text(json.dumps([{'source_id':c['source_id'],'geometry_pass':c['geometry_pass'],'geometry_flags':c['geometry_flags'],'window_provenance':c['window_provenance'],'camera':c['camera'],'instance_token':c['actors'][0]['instance_token'],'frame_filenames':[f['filename'] for f in c['frames']]} for c in clips],ensure_ascii=False,indent=2)+'\n')
    (ev/'review_link.md').write_text('[本轮报告](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r3/index.html) · [新数据逐帧全检](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r3/data_review.html)\n\n完整无损对、来源RGB与所有候选保存于 '+str(root)+'\n')
    report=f'''# v77 r3：连续多视角供体与新造遮挡

`WS-V77-TARGET-PROTECTED-20260929/r3`。2026-09-30用户明确开GPU继续造数据；保持真实Y监督、原DriveEditor架构，训练0、DriveEditor前向0。本轮SAM2为同一`sam2.1_hiera_large`，seed42；独立QA沿用单个`gpt-6-sol/xhigh`、未启用fast。

```mermaid
flowchart LR
 D[新多视角来源试产] --> W[整段曝光与几何选窗]
 W --> S[冻结SAM2连续mask]
 S --> F[本轮0合法放置 / 保存证据]
 L[已有干净来源与连续mask] --> A[几何配对 / 连续3D放置 / 洞H]
 Y[真实视频Y含保护车] --> X[只新增遮挡A的X]
 A --> X
 X --> C[先清零H / masked-X条件]
 Y --> G[真实GT不改动]
 C --> Q[逐帧工程检查 / 独立抽帧]
 G --> Q
 Q --> H[用户逐帧确认]
```

## 实际执行与输入关卡

4张静帧SAM2轮廓独立通过；但直接以该帧居中的固定10曝光窗口只有1/4满足整段约束。原提案与失败逐帧原因保留在`r3/temporal`。同一4个种子改用统一、有界的10种包含种子的窗口，先检查完整曝光/尺寸/边界/潜在前景遮挡，再按最小目标面积选一个；没有按mask输出挑窗口。最终3/4窗口通过，V002仍因整段可用性不足退出。

真实RGB只从原nuScenes train公共分片提取，没有生成视角或复制曝光。3段30帧SAM2从已审核种子mask双向传播；逐帧数值通过{s['temporal_numeric_pass']}段，独立抽帧通过{s['temporal_independent_pass']}段。官方可选CUDA `_C`扩展不可用，日志明确跳过微小孔洞填补后处理；原推理mask保留，未用自定义裁框/最大连通域补救，也不声称运行了该可选步骤。

## 新合成结果

首先仅把本轮3个连续供体加入旧receiver池：相机相对回放与既有固定世界位移两个有界控制均0个可行提案。没有以门槛放宽强行放置，也不据此否定SAM2或补景模型。随后按几何可配对性重新使用此前已审过的干净来源，污染来源黑名单、hole/ego检查和原物理门槛保留；选中32个提案中8个与旧结果完全同一来源/位移/轨迹，排除不重渲染。实际24个新样本中使用本轮新多视角供体为0；不能宣称“多视角让造数效果更好”。这明确暴露“独立挑清晰供体，再找放置”的流程问题：今后先用几何/目标视角匹配receiver，再提取与分割新增来源。

原Y与保护车mask复用。候选先过地图/真实LiDAR支持、米制净距、整段视角、尺度与位置/yaw连续、A在B前的深度次序、真实mask遮挡比例。整段使用一条同实例camera-track，禁止逐帧换图；原同一实例多相机是异步真实观测，不宣称同步多目联合世界，也没有向神经网络添加多视角通道。

新增{s['new_cases']}例 / {s['receiver_scenes']}个receiver scene，类型{s['new_case_types']}；独立结果{s['new_case_status']}。人工verdict仍空，训练准入0。现有旧27个技术候选不因本轮新造数而自动训练准入；case和scene分母分别报告，约50个合格且覆盖多类型的目标不能用重复短窗凑数。

## 造数入口修正

已知D006/D009是空间错误。新增可配置的最终hole底部64px禁入门槛，覆盖所有类别，在候选评估与最终渲染入口各查一次；边缘扩张侵入也拒绝。这个保守区域不是精确ego分割，也不能认证全部道路合法性。两个正反控制测试通过，保持旧默认接口供历史实验复现；本轮r3显式启用。

五项训练输入标准沿用：silhouette、尺度/位置、时序、A→B次序、全部synthetic影响被H移除。新的无损Y逐像素等于真实源RGB重新解码；X变化只在记录影响域；清零H后的X与Y一致，隐藏Y扰动不会泄漏进条件。为响应用户减少周额度消耗，本轮新合成抽帧规则在独立QA前固定为每例一张：纯背景用中帧，单保护车用最大遮挡帧；机器仍检查全部帧。这是对旧r1多帧抽样成本的明确修订，独立一帧绝不认证全视频或替代人工逐帧审核。车身RGB外观完全被遮住时不单独拒绝，明显悬浮/错误hole与邻车误遮仍拒绝。

全部新视频解码、资源与逐帧合同结果见[交付验证](../autoresearch/worldsim_v77/target_protected_20260929/r3/delivery_validation.json)。本地HTML保留真实Y／合成X／身份与遮挡标注／实际masked-X四栏；它们是数据审核，不是DriveEditor生成结果。原评分和r2人工进度不覆盖，r3独立评分存储。浏览器交互未实测。

GPU/CPU本轮作业已结束，无训练或后台自动启动控制器；没有创建定时任务或执行电源操作。完整run：`{root}`。轻量[证据](../autoresearch/worldsim_v77/target_protected_20260929/r3/summary.json)与[HTML入口](../autoresearch/worldsim_v77/target_protected_20260929/r3/review_link.md)。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
'''
    (repo/'docs/v77/TARGET_PROTECTED_GPU_R3.md').write_text(report)
    status=f'''# 当前研究状态

2026-09-30，v77；`WS-V77-TARGET-PROTECTED-20260929/r3`。用户已开GPU继续造数，本轮冻结SAM2与新合成完成；原DriveEditor架构不变，训练0、模型补景前向0。[当前报告](v77/TARGET_PROTECTED_GPU_R3.md)。

4张真实多视角静帧轮廓通过，但居中窗口仅1/4合格；保留控制后，按全段几何/曝光有界选窗得到3/4，V002仍退出。3段30帧SAM2，独立连续mask通过{s['temporal_independent_pass']}段。使用原官方缺少可选_C孔洞后处理的实际输出，未隐藏该运行边界。

新合成{s['new_cases']}例 / {s['receiver_scenes']}个receiver scene；类型{s['new_case_types']}；独立{s['new_case_status']}，人工未填、训练准入0。五项训练输入标准生效；全类别洞/边缘扩张不得进入底部64px自车保守区，所有新增影响编码前被mask清除。旧37个1分重评中的27个技术候选继续单独保留，D006/D009不升级。

新多视角供体在两种放置控制均无可行提案；实际新合成来自此前已审干净来源的几何配对，排除8个旧重复提案，不能把结果算作新多视角素材收益。本地`outputs/v77-target-protected-r3/index.html`与`data_review.html`提供本轮报告和新逐帧页，旧r2页面/原AI分数不覆盖。GPU与CPU本轮作业结束，未启动训练、自动化或关机；不继承旧审计电源授权。后续来源必须先验证与receiver的空间/视角可配对性，再补RGB/SAM2；补密集类型及场景缺额，技术通过仍需用户全帧确认。subagent默认gpt-6-sol/xhigh，禁fast。

run：`{root}`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
'''
    (repo/'docs/RESEARCH_STATUS.md').write_text(status)
    pth=repo/'docs/EXPERIMENTS.md';t=pth.read_text();line=f'| WS-V77-TARGET-PROTECTED-20260929 / r3 | 连续来源3段；新合成{s["new_cases"]}例/{s["receiver_scenes"]}场景；独立{s["new_case_status"]}；训练0 | [GPU造数报告](v77/TARGET_PROTECTED_GPU_R3.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r3/summary.json) |'
    assert line not in t;pth.write_text(t.replace('|---|---|---|','|---|---|---|\n'+line,1))
    pth=repo/'docs/research_failures/entries/V77-F02.md';t=pth.read_text();marker='## 连续供体窗口与自车像素禁入：r3造数入口'
    assert marker not in t
    pth.write_text(t.rstrip()+f'''\n\n{marker}

`WS-V77-TARGET-PROTECTED-20260929/r3`。4张静帧mask通过不代表10帧窗口可用：居中窗口3例因边界/曝光退出；统一有界全段筛选后3段进入SAM2，V002保持退出，不按下游成功调窗口。原始失败与新条件保留。全部合成类型增加最终hole与底部64px自车区域检查，并在渲染入口复核，避免D006/D009类空间错被GT无碰撞掩盖。

新3个多视角供体的两种既有位姿算子均0可行提案，物理/视角门槛未放宽。实际新合成{s['new_cases']}例/{s['receiver_scenes']}场景来自已有干净来源的几何配对，8个旧重复排除；不将收益归于新多视角。下一次采样改为先验证source/receiver可配对，再做提取/分割。独立{s['new_case_status']}，仅固定单帧粗检+全帧机器合同，人工全检之前训练准入0。外观完全隐藏不单独否决，hole形状/空间/遮挡仍须真实证据。SAM2缺少官方可选_C后处理的实际运行情况记录，不做静默裁框或后处理救结果。尚不证明同步共享世界的多相机编辑或模型改进。[报告](../../v77/TARGET_PROTECTED_GPU_R3.md)、[证据](../../autoresearch/worldsim_v77/target_protected_20260929/r3/summary.json)。failure_ledger_delta: updated V77-F02。
''')
    print('RECORDED_R3',s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.repo)
