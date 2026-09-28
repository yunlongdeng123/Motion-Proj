"""只以已收齐的实际证据生成收口报告。"""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.output;S=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
summary=read(R/'evidence/actual_summary.json');check=read(R/'validation.json');obs=read(R/'observations.json');tests=read(S/'current_checks.json');n=summary['counts']
assert len(obs['scenes'])==n['scenes']==9 and check['total_decoded_frames']==n['video_frames']
doc='''# v77九例统一配置完整DELETE视频交付

`WS-V77-NINE-FULL-20260928/r1`。本轮九例均完成同一协议的SAM2、DriveEditor、冻结Ω、Hunyuan与原位/DELETE读出；人工verdict为空。按照用户最新要求，本轮交付统一配置视频，不继续归因或加入逐场景补丁。

```mermaid
flowchart LR
 X[9例多相机RGB] --> M[SAM2实例mask]
 M --> D[DriveEditor补景与固定写回]
 D --> O[冻结Ω背景B_t]
 G[GT相机/框与LiDAR尺度] --> M
 G --> O
 X --> A[冻结Hunyuan显式GLB]
 A --> F[B_t加回原位GLB]
 O --> F
 O --> E[DELETE只关闭GLB]
 X --> V[原视频/原位重建/DELETE]
 F --> V
 E --> V
```

## 结果如何读

'''+obs['overview']+'''

默认三列中间是补景后B_t加回同一目标的新GLB，右侧严格复用B_t，只把visible改为false；不是三次独立视频生成，也不是直接把原RGB充当重建。默认纯点几何；输入RGB仅无几何命中位置回填是可选诊断，不能证明几何完整。可切六相机、同范围固定目标裁剪与0.25倍回放。每例30帧六相机、10Hz约3秒。原图黄框为目标GT位置。

九例全部已参与选择或观察，不是独立泛化评测。旧r18/r21/r30和上一轮产物完整保留，没有当成本轮输出。本轮无训练，未把失败例从分母移除；输入/补景不合格仍只运行一次完整诊断链，不能当准入成功。

## 九例目标

| scene / actor | 目标 |
|---|---|
'''
for r in summary['scenes']:
    o=obs['scenes'][r['scene']];doc+=f"| {r['scene']} / {r['actor']} | {o['identity']} |\n"
doc+='''
## 实际执行与检查

'''+f"共{n['source_rgb_viewtimes']}个RGB视角时刻、{n['sam_streams']}个SAM流；DriveEditor记录{n['drive_window_records']}窗，其中{n['drive_generated_windows']}窗实际生成、{n['drive_empty_windows']}窗全空mask保留RGB；Ω实际{n['omega_forwards']}次六相机前向；Hunyuan各9次shape与PBR；最终原位渲染层{n['actor_layers']}张，方向控制18张。默认27个三列视频，含六相机、放大和其他诊断共{n['video_files']}段视频，实际解码{check['total_decoded_frames']}帧。\n\n"
doc+=f"本轮15项已有几何/编辑测试通过；背景写回、保护区不变、原位投影/尺度、DELETE仅visibility变化和所有帧数由实际产物验证。页面{check['local_refs']}个静态引用、{check['dynamic_paths_verified']}个动态视频路径、JS语法和图像文件检查通过。浏览器交互未实际验证。上述检查不替代视觉验收。\n\n"
doc+='| scene | GT有投影视角时刻 | 其中空mask | Ω次数 | GLB层 | 几何覆盖中位数 | 资产通过深度测试中位比例 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for r in summary['scenes']:doc+=f"| {r['scene']} | {r['visible_gt_viewtimes']} | {r['empty_mask_visible_viewtimes']} | {r['omega_forwards']} | {r['actor_layers']} | {r['geometry_coverage_median']:.1%} | {r['asset_depth_pass_median']:.1%} |\n"
doc+='''
覆盖和深度通过比例只是工程描述，不是正确几何/正确遮挡率。GT有投影不保证表面可见；空mask需要区分遮挡、出视野和分割缺失。真实邻车包络审计不能当像素可见性真值，更不能把所有删除区车形都判成幻觉。

## 共同协议与工程修正

完整参数见[协议](NINE_FULL_PROTOCOL.md)。同一seed、提示/选帧/写回/资产规则施于全部九例。首例固定Rz-90转换暴露长轴错放，错误GLB与局部渲染保留；九例共同改为水平PCA长轴X，再用真实参考yaw0/180的固定maskIoU−0.25×RGB_MAE选择方向，差小于0.01固定0并标不确定。这是所有例共同的BUILD修正，未根据编辑效果人工逐例调参数。初始登记与工程修正/effective登记均保留，模型不因该修正重跑。

不做光照拟合/接触阴影/逐例贴地补丁，GT位姿、尺寸、高度和固定world+sun规则相同。B_t按时刻独立重建，其他车辆多仍在背景，尚非持久一致世界；GT/LiDAR辅助、隐藏表面无GT、官方权重训练重叠未核实均为边界。没有执行MOVE、训练或新增模型家族。

## 资源与证据

'''
res=summary['resources'];doc+=f"单卡{res['gpu']}串行模型队列；DriveEditor阶段{res['drive_seconds']/60:.1f}分钟、峰值allocated {res['drive_peak_gib']:.2f}GiB，Ω阶段{res['omega_seconds']/60:.1f}分钟、峰值{res['omega_peak_gib']:.2f}GiB。Hunyuan逐作业耗时见actual_summary，含加载/CPU处理，不把推理显存当训练显存。Blender5.2.2本地CPU4线程。\n\n"
doc+='''唯一run根 `/root/autodl-tmp/runs/worldsim_v77/WS-V77-NINE-FULL-20260928/r1`，完整原图/mask/生成图/Ω预测点云/GLB/RGBA深度/错误控制原地保留。本地审核 `outputs/v77-nine-full/index.html`。轻量登记、资源和验证见[证据索引](../autoresearch/worldsim_v77/nine_full_20260928/review_link.md)。没有定时任务、自动重试或电源操作。

`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02（共同坐标转换工程修正与本轮边界，无新增质量归因）`；`human_verdict: null`。本轮停止在九例同配置完整对比。用户要求：后续策略需在固定九例中逐例确认收益；如果只有部分场景受益、其他场景没有收益或退化，先不接入，再考虑可信数据与训练动态处理差异。本轮不启动训练，不逐例换seed/参数。
'''
(S/'NINE_FULL_RESULTS.md').write_text(doc,encoding='utf-8')
close=dict(task_id=summary['task_id'],run_id='r1',state='complete_pending_user_visual_review',counts=n,checks=tests,local_media_validation={k:v for k,v in check.items() if k!='videos'},role=summary['role'],failure_ledger_refs=['V77-F02'],failure_ledger_delta='updated V77-F02',human_verdict=None)
(R/'evidence/closeout.json').write_text(json.dumps(close,ensure_ascii=False,indent=2),encoding='utf-8');print('NINE_REPORT_BUILT',flush=True)
