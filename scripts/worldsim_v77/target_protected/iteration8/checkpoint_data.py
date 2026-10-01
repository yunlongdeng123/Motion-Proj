"""轻量数据检查点；训练/生成结果未知时不提前填写效果。"""
from pathlib import Path
import sys,shutil,json
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,T,read,dump
R=Path('/root/autodl-tmp/motion_proj_v77')

def main():
    evidence=R/'docs/autoresearch/worldsim_v77/target_protected_20260929/r8';evidence.mkdir(parents=True,exist_ok=True)
    for name in ['run_config.json','factory_amendment.json','pose_sampling_amendment.json','source_pool_summary.json','short_merge.json','asset_selected.json','evaluation_source_split.json','technical_checks.json','static_occupancy.json','independent_data_reviews.json','dataset_catalog.json','admission_result.json','evaluation_plan.json','real_evaluation_plan.json']:
        if (O/name).exists():shutil.copy2(O/name,evidence/name)
    s=read(O/'dataset_catalog.json')['summary'];text=f'''# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r8`已准入50train/25个receiver场景，单scene最多3例；24背景、23单保护、3密集（密集训练仅scene-0240）。7合成评测/4个隔离场景，8个已曝光真实DELETE开发窗口。固定原/r7/新数据三臂，final不用。

68候选全10帧RGB/覆盖/metric放置/depth/连续性复核，gpt-6-sol xhigh独立固定帧61pass/5uncertain/2reject；50train和7val均技术通过且AI2，人工空。M053/M064视觉树带/人行道拒绝，即使三扫描静态占据无正证据也不放行。5待定、4合格备用完整保留。

数据改为已有DEV sedan/SUV显式mesh在真实camera/metric SE3下投影，完整synthetic RGB擦除；Y永远真实。train/val共享两种形状模板，只隔离真实世界来源，不声称未见形状/跨域。Boston白天数据，真实DELETE无去车GT。原图、全部PNG/视频与旧规则保留。

下一步同r7有效初始化、80空间attention、320×576/160步、原loss/seed6201、从原DriveEditor开始；不延续r7、不扩大模块/加权/加步数。真实原/r7对照已完成，等待本轮训练及合成/真实三臂完整采样和HTML。无电源/自动化操作。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02（数据准入，模型结论待定）。
'''
    (R/'docs/RESEARCH_STATUS.md').write_text(text)
    report='''# Target + Protected Actors r8：数据覆盖控制

任务 `WS-V77-TARGET-PROTECTED-20260929/r8`。用户要求只改变数据，暂保持当前模块范围和首轮预算。模型效果尚待完整采样。

```mermaid
flowchart LR
 Y[真实nuScenes Y] --> G[GT相机/轨迹/地图/LiDAR]
 G --> A[已有显式车辆mesh\n连续SE3轮廓A]
 Y --> P[SAM2真实保护车B/C]
 A --> H[全影响洞H\n擦除X后编码]
 P --> Q[全帧机器+独立固定帧QA]
 H --> Q
 Q --> D[50train/25worlds\n7val/4worlds]
 Y --> T[原结构DriveEditor\n同80张量160步]
 D --> T
 T --> E[原/r7/新数据\n合成GT与真实DELETE]
```

## 数据准入

68候选来自三维轮廓工厂；全部680帧实际磁盘X/Y/影响/H重载，验证影响完全遮盖、编码前X和Y遮洞条件相同、真实Y重新解码一致、ego禁入、metric位姿/尺度/yaw/轮廓连续和A前B后的深度合同。所有模板先在真实地面、地图可行驶/停车范围、GT包络间距至少0.3m与地面支撑2.5m内筛选，未知动态包络不允许被洞吞入；真实后方静态道路设施可作为恢复背景。

独立gpt-6-sol xhigh、不用fast，每例固定一帧实看：61pass、5uncertain、2reject。M053/M064落树带/人行道，视觉拒绝保留，不用地图/GT无碰撞或稀疏LiDAR零点抵消。另对所有候选检查3独立扫描的静态占据（每扫至少3个20cm voxels、距拟合地面0.15m以上，GT对象已移除）；未发现满足该保守门槛的正例，零证据不证明空或完整道路合法。

最终50train/25scene，最多3/scene；24背景、23单保护、3密集，密集训练仅scene-0240，未达到原14例密集提案。7val/4scene：3背景、2单保护、2密集。5待定和2拒绝不准入，另4合格备用未采入。所有最终案例技术pass且AI2，human null。固定帧QA不认证时序视觉；机器仅覆盖一秒10帧连续输入。

## 数据改变与边界

旧仿射供体入口在精确mask/物理门槛下不足20场景；有界来源扩展与粗/细metric位置采样后，改用两种已保留DEV显式网格投影轮廓，不调用新生成网络。真实Y从未改写，灰色完整X仅供轮廓/位置示意，其RGB全部被H遮掉，外观假不作为拒绝理由。train/val共用形状，不能声称未见资产泛化；所有真实来源是Boston白天官方train，原模型预训练场景重叠未知。

新增40窗口候选，37有序唯一曝光与几何通过，36完全提取合并，1窗口13成员缺失保留清单；公共RGB/LiDAR仍足够本轮，不补造缺帧。数据覆盖与3D silhouette工厂同时改变，不能把后续差异仅归于scene数量。

## 冻结对照

原DriveEditor、r7修复encoder的160步checkpoint、新数据版三臂；新训练从原权重和官方106目标encoder开始，80空间自attention/49,574,080参数，320×576，160步，AdamW lr1e-5、wd0.01、clip1、seed6201，原StandardDiffusionLoss，无保护加权。推理576×1024/10帧/seed42/25步，三臂同RGB/H/alpha与previous条件关闭；native和局部写回分开。

合成来源场景与所有r7/r8训练receiver/donor隔离，选窗和QA在新数据训练前冻结；真实A022/A041_w10/A013/A048/A007/A034/A042/A061_w08为曝光DEV，不计final。真实无去车GT，不报告恢复MAE；主要验收保住后车/邻车与正确去目标，不能用合成误差代替。旧r6三个真实原模型输出在RGB/H、窗口/seed/步数/previous条件逐项一致后复用原生图，重新以当前alpha写回，保留来源。

下一步训练后完成两套三臂采样、量化和人工HTML。若真实无稳定收益，保留原权重默认，不机械增加训练步数或模块。failure_ledger_refs: [V77-F02]；human_verdict: null。
'''
    (R/'docs/v77/TARGET_PROTECTED_DATA_CONTROL_R8.md').write_text(report)
    p=R/'docs/EXPERIMENTS.md';txt=p.read_text();row='| WS-V77-TARGET-PROTECTED-20260929 / r8 | 数据控制：50train/25scene/max3，7val/4scene；同80张量160步，两套三臂评测待完成 | [报告](v77/TARGET_PROTECTED_DATA_CONTROL_R8.md) · [准入](autoresearch/worldsim_v77/target_protected_20260929/r8/admission_result.json) |\n'
    if ' / r8 |' not in txt:txt=txt.replace('| WS-V77-TARGET-PROTECTED-20260929 / r7',row+'| WS-V77-TARGET-PROTECTED-20260929 / r7',1);p.write_text(txt)
    p=R/'docs/research_failures/entries/V77-F02.md';txt=p.read_text();marker='## r8：数据覆盖控制与静态空间视觉拒绝'
    if marker not in txt:
        p.write_text(txt+'\n\n'+marker+'\n\n`WS-V77-TARGET-PROTECTED-20260929/r8`旧仿射轮廓入口在原精确mask/metric门槛下覆盖不足；改用两种保留DEV显式mesh做真实相机投影，真实Y不改，合成RGB全擦除。68候选全帧机器合同、独立固定帧61pass/5uncertain/2reject；50train/25worlds、单scene最多3（24背景/23单/3密集），7val/4隔离worlds。密集train仍仅1scene，不宣称三类均充分覆盖。\n\nM053/M064视觉位于树带/人行道，保留拒绝；地图/GT间距和三扫描稀疏静态零正证据不能证明可放置，独立图像质量检查仍必要。非活跃保护标签的洞外细枝不进入模型/加权loss，记录诊断，不等同H泄漏。没有因而否定补景网络。\n\n本轮模型更新范围/预算冻结r7有效配方；原/r7真实对照已运行，新数据训练及完整合成/真实三臂仍待完成，效果不提前填写。train/val共享已曝光形状、Boston白天、GT/LiDAR辅助、1秒窗口、无真实去车GT、final未用均保留。见[数据控制与组件图](../../v77/TARGET_PROTECTED_DATA_CONTROL_R8.md)。failure_ledger_delta: updated V77-F02（数据准入与空间门控边界，科学效果待定）。\n')
    print('DATA_CHECKPOINT',s)
if __name__=='__main__':main()
