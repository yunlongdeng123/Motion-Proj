"""发布一次有限补证结果；保留前阶段报告、输入和所有原生输出。"""
import json,datetime,shutil,subprocess,html,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path('/root/autodl-tmp/motion_proj');O=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01');P=O.parent/'WS-V81-GPU-P2-01';A=O/'av2_atlas';F=O/'figures';D=R/'docs/autoresearch/worldsim_v81/bounded_closeout';RF=R/'docs/figures/worldsim_v81_closeout';RF.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text());res=read(O/'residual_metrics.json');refs=read(O/'residual_reference.json');new=read(O/'new_intervention_metrics.json');pairs=read(O/'new_intervention_pairs.json');summary=read(O/'new_intervention_summary.json');reg=read(O/'registration.json')
def write(p,t):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t.rstrip()+'\n')
def fmt(v,n=3):return '不可评' if v is None else f'{v:.{n}f}'
# 模块图：HELDOUT仅接评价；控制输入单独标示。
fig,ax=plt.subplots(figsize=(13,5));ax.set_xlim(0,13);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,w,h,label,color='#e8f0f4'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=color,edgecolor='#45616d'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
def arrow(x1,y1,x2,y2):ax.annotate('',(x2,y2),(x1,y1),arrowprops={'arrowstyle':'->','lw':1.6,'color':'#45616d'})
box(.1,3.4,2.3,.8,'Frozen RGB\n& visible surfaces');box(3.0,3.4,2.1,.8,'Official frozen\nDVGT / VGGT');box(5.7,3.4,2.1,.8,'Native output\n& frame audit');box(8.5,2.0,2.1,1.0,'Residual / view\nrecovery evaluation');box(11.2,2.0,1.6,1.0,'Four-link\ndecision','#f5e4cc');box(.1,1.7,2.3,.8,'Current INPUT\nLiDAR only','#f5e4cc');box(3.0,1.7,4.8,.8,'Outside-target scale / local plane\nExtra information disclosed','#f5e4cc');box(5.7,.2,2.1,.8,'HELDOUT\nreference scans','#d9ebe2');arrow(2.45,3.8,2.95,3.8);arrow(5.15,3.8,5.65,3.8);arrow(7.85,3.8,8.5,2.9);arrow(2.45,2.1,2.95,2.1);arrow(7.85,2.1,8.45,2.4);arrow(7.8,.6,9.2,1.95);arrow(10.65,2.5,11.15,2.5);ax.text(.1,.4,'8 discovery logs / 3 real-view interventions\n18 new forwards; 440 prior forwards reused\nNo training / no synthetic escalation',fontsize=10,color='#45616d');fig.tight_layout();fig.savefig(F/'four_link_architecture.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.5,4));labels=['Gray column','Brick wall','Stucco wall'];ids=list(dict.fromkeys(r['roi_id'] for r in new));x=np.arange(3)
for ax,key,ylabel in [(axs[0],'mae_m','Depth MAE (m)'),(axs[1],'normal_error_deg','Normal error (deg)')]:
 for j,var in enumerate(['rich','removed','restored']):
  vals=[next(r for r in new if r['roi_id']==rid and r['method']=='vggt' and r['variant']==var)['after_visual_plus_input_lidar_scale'][key] for rid in ids];ax.bar(x+(j-1)*.23,vals,width=.22,label=var,color=['#3480a3','#dc9855','#74a78e'][j])
 ax.set_xticks(x,labels);ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
axs[0].axhline(.5,color='#a14848',ls='--',lw=1,label='Operational cutoff');fig.suptitle('VGGT + identical outside-target INPUT LiDAR anchors | 3 discovery logs');fig.tight_layout();fig.savefig(F/'new_evidence_summary.png',dpi=160);plt.close(fig)
shutil.copy2(P/'analysis/figures/case_scene-0139_7e27d5c0_CAM_BACK_LEFT_12.png',F/'retained_gray_wall_goodcase.png')
corefigs=['four_link_architecture.png','new_evidence_summary.png','av2_effective_evidence_before_model.png','intervention_av2_04994d08_ring_side_right_11.png','residual_scene-0632_1b3e964e_CAM_FRONT_RIGHT_13.png','retained_gray_wall_goodcase.png']
for f in corefigs:shutil.copy2(F/f,RF/f)
current=[]
for rr in refs:
 a=next(r for r in res if r['roi_id']==rr['roi_id'] and r['method']=='dvgt' and r['output_key']=='full6');b=next(r for r in res if r['roi_id']==rr['roi_id'] and r['method']=='vggt' and r['output_key']=='full6')
 current.append(f'| `{rr["roi_id"]}` | {rr["points"]} / {fmt(rr["leave_scan_normal_spread_deg"],2)}° / {"通过" if rr["reference_pass"] else "未通过"} | {fmt(a["before"].get("mae_m"))}→{fmt(a["after_input_global_scale"].get("mae_m"))} | {fmt(b["before"].get("mae_m"))}→{fmt(b["after_input_global_scale"].get("mae_m"))} | {fmt(b["after_input_global_scale"].get("normal_error_deg"),1)}° | {fmt(rr["input_local_plane"].get("mae_m"))} |')
newtable=[]
for rid,name in zip(ids,labels):
 vals={r['variant']:r for r in new if r['roi_id']==rid and r['method']=='vggt'};q=next(p for p in pairs if p['roi_id']==rid and p['method']=='vggt')
 maes=' / '.join(fmt(vals[v]['after_visual_plus_input_lidar_scale']['mae_m']) for v in ['rich','removed','restored'])
 normals=' / '.join(fmt(vals[v]['after_visual_plus_input_lidar_scale']['normal_error_deg'],2) for v in ['rich','removed','restored'])
 newtable.append(f'| {name} / `{q["log"][:8]}` | {fmt(q["future_parallax_deg"],2)}° | {vals["rich"]["reference_points"]} | {maes} | {normals} | {fmt(vals["rich"]["input_local_plane_extra_information"]["mae_m"])} |')
report=f'''# V8.1 有限补证收口：保留改善候选，降低共同失效立论优先级

`WS-V81-CLOSE-01 / 20260913-bounded-r1`；本轮完成。**V8.2：NO_GO；当前“稀疏视角×低纹理共同失效”子命题降低优先级，停止这轮发现批次。**这不是对整个方向的否定，也不代表 V8.1 方法族实测已完成。DGGT 等直接重建方法仍有方法级证据缺口，已纳入[方法族 badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)，不会用 raw VGGT 替代。

![四环证据架构](figures/worldsim_v81_closeout/four_link_architecture.png)

## 本轮增加了什么

复用旧 440 项预测，完成 10 个既有候选的固定内部复核，共 {len(res)} 条模型/输入配置控制记录；其中 8 个内部参考通过数值稳定性检查，仍需结合可见表面复核。新来源严格限定 8 个 AV2 日志、840 个格子；256 个通过几何筛查，按模型前纹理排序保留 16 个视觉候选。先看 RGB/参考，限定 4 个同一可见面内部，最终 3 个具有可用后一帧的日志做真实观测干预。DVGT/GPU0 与 VGGT/GPU1各 9 项，共 18 项新推理。未训练、未重跑旧 440 项、未追加合成极端条件。

8 个日志都保留在筛查分母中，未换日志补失败；它们是 V8.1 新发现来源，但部分 metadata 曾用于旧研究，因此不是独立确认集。原 10 日志确认 reserve 未读质量、未用于本轮选择。16→4 的内部矩形只依据 RGB 可见面边界，在新模型输出前冻结；完整候选和排除原因一并保存。

## 现有残余复核

固定向内缩 32 像素，报告多扫描点数、逐次去掉一个扫描后的最大法向偏移，以及原输出→区域外 INPUT LiDAR 全局尺度后的 MAE。局部平面只用目标区域当前 INPUT LiDAR 拟合，再在 HELDOUT 上评价；它提供额外信息，不属于纯视觉同预算优势。

| ROI | 内部点数 / 去扫描法向变化 / 参考 | DVGT MAE 原→尺度控制 m | VGGT MAE 原→尺度控制 m | VGGT 控制后法向 | INPUT 局部平面 MAE m |
|---|---|---|---|---|---|
{chr(10).join(current)}

**木板墙 `scene-0632_1b3e964e_CAM_FRONT_RIGHT_13` 不能升格为代表 badcase。**原 ROI 92 点；内部 60 点，去扫描后的法向偏移达 23.44°，未通过参考稳定性检查。不能把它控制后的大法向误差解释成低纹理失效。另一远处墙面也因 5.51° 的变化未过本轮 5° 筛查线。此检查只是参考稳定性，不替代 RGB 与参考同表面的核查。

![木板墙参考与尺度复核](figures/worldsim_v81_closeout/residual_scene-0632_1b3e964e_CAM_FRONT_RIGHT_13.png)

**灰墙 goodcase 原样保留**：旧完整 ROI 有 297 个参考点、DVGT MAE 0.103m；本次内部子区域 206 点，原输出约 0.108m。收缩窗口不覆盖旧证据，不把 206 点写成原例点数。

![保留原灰墙goodcase](figures/worldsim_v81_closeout/retained_gray_wall_goodcase.png)

控制也会新增误差。例如灰墙 `CAM_FRONT_LEFT_02` 的 VGGT 内部 MAE 约 0.055→0.901m。报告同时保留 `residual_axes` 与 `control_created_axes`，只有控制前后同一轴都超过本轮实用量级筛查线才计残余；控制创造的误差不算模型原有失败。区域外全局尺度不能解释所有局部偏差，场景 0919 的墙/门/设施边界候选仍保留可见性疑问，未宣布共同失效。

## 新日志的真实有效证据与恢复

原双侧 ±0.5s 设计在模型前检查中不成立：部分前帧出画、被车挡，另一墙面视差只有约 0.53°/0.64°。因此在所有新 forward 之前记录一次输入方案修订：保留原日志与目标，使用确实可见的 +0.5s 观测。没有依据模型误差改选帧。三个最终目标的新增视差约 12.03°、17.91°、8.29°；可见比例是稀疏遮挡检查加 RGB 平面投影复核，不能冒充稠密可见性真值。

![模型前有效证据核查](figures/worldsim_v81_closeout/av2_effective_evidence_before_model.png)

`rich=[当前,后一帧] → removed=[当前] → restored=[当前,同一后一帧]`。所有变体使用相同当前 LiDAR、相同目标相机、相同区域外点投影集合定尺度；VGGT 的三个目标分别 8795/10155/9178 条投影记录，集合不随变体改变。不是多相机重复点数，也未用 HELDOUT 拟合。单图 raw VGGT 没有可识别米制尺度，控制前绝对 MAE 留空，只报告尺度无关形状/朝向；不能与带 LiDAR 结果混排。

| VGGT＋区域外 INPUT LiDAR | 新增视差 | HELDOUT 点数 | MAE rich / removed / restored（m） | 法向 rich / removed / restored | INPUT 局部平面 MAE（m） |
|---|---|---|---|---|---|
{chr(10).join(newtable)}

![新日志的观测变化](figures/worldsim_v81_closeout/new_evidence_summary.png)

VGGT 在砖墙和灰泥墙上有 **0.429m、0.290m** 的可恢复深度差距；这两条正结果保留，不能写成“加真实观测完全没用”。灰色立柱没有相同退化，单图反而略好。三个单图控制后 MAE 约 0.018/0.472/0.337m，法向 1.81°/5.51°/0.15°；砖墙 AbsRel 约 5.20%，是值得保留的边界候选。恢复输出与 rich 相同，最大差 0；这是相同输入/seed 的确定性恢复，不是额外独立复现。

本轮预设深度残余线为 `max(0.5m, 5%参考深度, 3×参考平面RMS)`，法向为 `max(10°, 3×参考不稳定度)`；观测效应还需 MAE 增量 >0.25m 或法向增量 >5°。这些是有限预算发现的操作标准，**0.472m 接近阈值不等于误差没有研究价值**。当前只有 3 个发现日志，没有纹理因果对照，不能据此声称普遍无失效或稀疏×低纹理交互成立。

![砖墙可恢复候选](figures/worldsim_v81_closeout/intervention_av2_04994d08_ring_side_right_11.png)

DVGT 只有砖墙的这套相机光轴评价合同可比较：rich/removed/restored MAE 为 0.094/0.153/0.094m，法向 7.13°/7.81°/7.13°。两个后向单相机案例在按数据集 ego 系导出时出现非正光轴深度，导致尺度拟合或 ROI 覆盖不可用。原生张量与错误/空缺分母保留，标记 `POINT_FRAME_OR_COVERAGE_CONTRACT_UNRESOLVED`；没有完成单相机点图坐标对应的专门验证，**不把这些数值当作科学 badcase，也不把两个不可评日志当作稳健负例**。完整 18 项 forward 成功不等于 18 项科学评价都成立。

## 一次推进判定

| 四环 | 证据状态 | 本次判定 |
|---|---|---|
| 可靠残余 | 木板墙最强法向候选参考不稳；VGGT 仍有局部可恢复深度候选，但新日志普通锚点后残余低于本轮预设量级线 | 部分证据；没有共同可靠代表失败 |
| 明确因素 | 三日志真实视差变化；两处 VGGT 深度改善、一处不改善；没有低/高纹理干预或匹配 | 支持局部观测改善空间；不支持共同交互归因 |
| 普通控制与恢复空间 | 同 INPUT 点集定尺度、局部 INPUT 平面与真实后一帧均已做，额外信息单列 | 已有强控制；成熟重建参照及完整下游仍缺 |
| 新日志与任务一致 | 8 日志筛查、3 日志发现干预；确认 reserve 未触碰；没有本轮 DGGT/DriveMVS/FocusGS/VGGD 方法级结果 | 未通过 |

**NO_GO / DEPRIORITIZE_CURRENT_JOINT_FAILURE_CLAIM。**停止当前有限批次，不因更极端筛选而续命；没有满足前三环的具体失败触发条件，因此本次不扩展条件下游实验、不消耗独立确认集。DGGT 是对应驾驶重建任务的优先待实测方法，但本次没有新增 core/renderer/diffusion 输出；其他方法的机制、可用性、反例定义与所需图已写入方法族报告。V8.1 的方法族实证范围仍未完成，不能宣布“2026 驾驶重建方法普遍失败”或“这些方法已通过”。

后续若继续，应以可运行直接重建方法的明确任务或新可靠失败定义重新立项，保留本轮好例、边界候选与停止决定；不自动重开当前批次。是否值得研究小于 0.5m 的残余，需要任务需求和新预注册标准，不能对本轮结果事后改线宣称已过准入。

## 执行与证据

新推理18/18结束；DVGT/VGGT峰值分别 {max(r['gpu_peak_gib'] for r in new if r['method']=='dvgt'):.2f}/{max(r['gpu_peak_gib'] for r in new if r['method']=='vggt'):.2f} GiB。2×3090足够；没有训练、控制器、自动恢复或新关机动作。原始输出、尺度点集合、旧/新输入协议、16个模型前视觉评审和所有图保存在 `{O}`。`residual_metrics.json`、`residual_reference.json`、`new_intervention_metrics.json`、`new_intervention_pairs.json` 是复查入口。

AV2 LiDAR 已由数据集补偿到 ego 参考时刻，本轮再用各自 LiDAR/相机 pose 转换；使用[官方数据定义](https://argoverse.github.io/user-guide/datasets/sensor.html)和本地官方 AV2 camera API 核对接口，不把 nuScenes 的扫描级补偿声明直接套用。模型沿用上一阶段官方 checkpoint、Torch2.4.1+cu121/BF16，未声称作者整套评测复现。

failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=V81-F05。人工 verdict=null。
'''
write(R/'docs/WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md',report)
decision='''# V8.1 → V8.2：NO_GO，当前共同失效子命题降优先级

![四环证据](figures/worldsim_v81_closeout/four_link_architecture.png)

WS-V81-CLOSE-01 已完成一次有限补证；没有训练或重复旧440项。准入只保留四环：可靠残余、预指定真实证据变化与恢复、普通控制及可恢复空间、新日志确认及与任务对应的重建方法。自然完整四格不是所有推进的强制先决条件。

10个现有候选内部复核；最强木板墙法向候选参考不稳定（去扫描最大变化23.44°）。灰墙297点、DVGT0.103m的goodcase保留。8个新AV2发现日志筛查后，3日志18项真实观测干预完成；VGGT有两处可恢复深度差距0.429/0.290m，但单图加普通度量锚点后MAE为0.018/0.472/0.337m。保留0.472m、AbsRel5.20%的砖墙边界候选，不以预设0.5m筛查线否定其潜在价值。

当前不足以推进“稀疏视角×低纹理共同失效”方法开发：没有纹理因果证据；DVGT两个后向单相机案例坐标/覆盖评价合同未建立；缺直接重建方法和独立确认。没有把不可评当作稳健，也没有把raw VGGT当VGGD。

停止本批次，状态`BOUNDED_CLOSEOUT_COMPLETE / DEPRIORITIZE_CURRENT_JOINT_FAILURE_CLAIM`。不追加极端合成、不重复本批次、不启动V8.2。条件下游扩展未触发；DGGT、DriveMVS、FocusGS、VGGD等仍按方法族报告保留待实测状态。此为现有子命题收口，不是V8.1整个方法族实测完成或全方向被证伪。

[完整结果与图](WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)；[各方法badcase定义和可执行性](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)。确认reserve继续封存。人工verdict=null；failure_ledger_delta=V81-F05。
'''
write(R/'docs/WORLDSIM_V8_1_TO_V8_2_DECISION.md',decision)
failure='''# V81-F05：有限补证未建立共同失效，保留局部可恢复候选
<!-- metadata: {"title":"参考稳定性、同锚点真实观测干预与有限收口","defined_ids":["V81-F05"],"referenced_ids":["V81-F01","V81-F02","V81-F03","V81-F04"],"versions":["V81"],"topics":["data_evidence","engineering","novelty"]} -->

状态/分类：证据不足与普通控制边界；包含未解决的单相机坐标评价合同，不是模型族科学失败。

命题：可靠可见低纹理表面在真实有效观测减少后出现可重复、有实际量级的残余，且普通控制不能解释全部差距。WS-V81-CLOSE-01 / 20260913-bounded-r1，只补一轮，不重跑440项，不训练。

10个既有候选统一32px内部复核，8个数值参考通过；最强木板墙候选内部60点、去扫描法向最大变化23.44°，不支持代表badcase。原灰墙297点、DVGT MAE0.103m保留。控制可能创造误差，残余必须控制前后同轴均存在。

8个AV2新发现日志模型前冻结：840格→256几何通过→16可视候选→4平面内部→3有效后一帧目标，18项新forward。双侧设计因出画/遮挡在模型输出之前修订为当前+真实后一帧；原协议保留。相同目标区域外当前INPUT LiDAR记录进行全局定尺度，局部INPUT平面单列额外信息。

VGGT三日志单图控制后MAE0.018/0.472/0.337m；新增真实观测后0.031/0.043/0.047m。两处有0.429/0.290m恢复空间，不能说完全没有退化；0.472m/AbsRel5.20%砖墙是阈值附近候选。本轮预设0.5m等操作筛查线未过，不代表论文任务中一定无意义。恢复同输入最大差0，是确定性恢复而非独立确认。

DVGT可比较砖墙MAE0.094/0.153/0.094m；两个后向单相机在数据集ego坐标导出下出现非正光轴深度，尺度或覆盖不可评。未确认是模型失败还是单相机坐标不定/合同不适用；保存原生张量，不当作badcase或稳健负例。旧测试验证过的多相机合同不能自动外推所有新输入配置。

本次决定：NO_GO，降低当前共同失效立论优先级，停止本批。没有纹理因果对照、直接下游重建与独立确认；条件DGGT扩展未触发，不声称整个V8.1方法族完成。DriveMVS/FocusGS/VGGD等不可用或未跑的方法没有方法级failure。

证据：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01`中的registration、residual_metrics/reference、av2_atlas/final_intervention_protocol、new_predictions原生输出、new_intervention_metrics/pairs；[收口报告](../../WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)、[方法族报告](../../WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)。官方AV2输入定义和DVGT导出来源在报告与代码中。

failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=V81-F05。人工verdict=null。
'''
write(R/'docs/research_failures/entries/V81-F05.md',failure)
head='''# 当前：V8.1 有限补证收口（2026-09-13）

WS-V81-CLOSE-01完成：10旧候选内部复核；8新AV2日志840格模型前筛查，3日志18项新推理。旧440项未重跑、无训练。木板墙最强法向候选参考不稳；灰墙297点/0.103m好例保留。VGGT两个新日志有0.429/0.290m可恢复差距，单图度量锚点后MAE0.018/0.472/0.337m；DVGT两个后向单相机坐标/覆盖合同未建立，不计科学failure或稳健负例。

四环证据不足，V8.2 NO_GO；当前共同失效子命题降低优先级，停止本批次。其他8方法已纳入方法族badcase report，DGGT等方法级实证仍缺，条件下游扩展未触发；不能宣称整个V8.1已测完。无自动续跑、确认reserve封存、未关机。人工verdict=null；failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=V81-F05。

[有限收口报告](WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)；[方法族badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)；[推进判定](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。以下为历史。

'''
for f in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/RESEARCH_FAILURES.md','docs/WORLDSIM_V8_1_SCIENTIFIC_REPORT.md','docs/WORLDSIM_V8_1_FAILURE_ATLAS.md','docs/WORLDSIM_V8_1_METHOD_AUDIT.md','docs/WORLDSIM_V8_1_PLAN.md']:
 p=R/f;write(p,head+p.read_text())
p=R/'AGENTS.md';write(p,head.replace('(WORLDSIM_', '(docs/WORLDSIM_')+p.read_text())
p=R/'README.md';write(p,'当前 V8.1：[有限补证收口](docs/WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)、[方法族 badcase report](docs/WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)。当前批次已停，V8.2 NO_GO；不是整个方法族已测完。\n\n'+p.read_text())
write(R/'docs/WORLDSIM_V8_1_GPU_HANDOFF.md',head+'''## 接续边界

本批任务已结束，2×3090充足，不自动重跑。当前入口为有限收口报告及方法族badcase report。后续只有明确新任务才扩展；其他方法的机制假说不能填写成实测failure。

原始GPU P2：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01`，440项；本次`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01`，18项新forward。复核：`scripts/audit_worldsim_v81_residual.py`与`scripts/evaluate_worldsim_v81_bounded_intervention.py`，均为CPU读取现有输出。获取、筛查、freeze脚本用于记录本次构建；不可盲目重跑freeze覆盖原登记时间。

新单图relative导出只在显式`--allow-relative-single-view`下使用`*_depth_z_native.npy`，`metric_scale=null`，不声称米制。新队列结果已改写为VIEW_DIAGNOSTIC与实际rich/removed/restored；stdout中底层通用入口的临时full6标签不作为最终登记。旧自然full6结果未修改。

DVGT新rear单相机的point-frame/coverage状态保持未解决，不按当前camera-z评价器推进该科学主张。确认reserve未消耗。不存在继承V74关机的新授权。
''')
p=R/'docs/WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md';t=p.read_text();t=t.replace('```mermaid\nflowchart LR', '```mermaid\nflowchart LR');t=t.replace('## 方法范围','本轮状态：18项新增DVGT/raw VGGT发现推理已完成；条件下游未触发，DGGT本轮没有新增core/Gaussian/render/diffusion输出。其他方法均没有新增方法级badcase。[有限收口结论](WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)。\n\n![架构组件](figures/worldsim_v81_closeout/four_link_architecture.png)\n\n## 方法范围');write(p,t)
reg.update(state='BOUNDED_CLOSEOUT_COMPLETE',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),failure_ledger_delta='V81-F05',new_forwards=18,scientifically_comparable_intervention_logs={'vggt':3,'dvgt':1},promotion='NO_GO',subclaim='DEPRIORITIZE_CURRENT_JOINT_FAILURE_CLAIM',downstream='NOT_TRIGGERED; DGGT method-level evidence outstanding',automatic_resume=False,power_action='NONE')
write(O/'status.json',json.dumps(reg,indent=2));write(O/'completion_audit.json',json.dumps({'task':reg['task'],'actual_result_files':len(list((O/'new_predictions').rglob('result.json'))),'expected_new_forwards':18,'old_forwards_repeated':0,'new_gpu_peaks_gib':{m:max(r['gpu_peak_gib'] for r in new if r['method']==m) for m in ['dvgt','vggt']},'science_status':summary,'training':False,'human_verdict':None},indent=2))
for f in ['registration.json','status.json','completion_audit.json','residual_metrics.json','residual_reference.json','new_intervention_metrics.json','new_intervention_pairs.json','new_intervention_summary.json','new_source_acquisition.json']:shutil.copy2(O/f,D/f)
delivery=O/'delivery';delivery.mkdir(exist_ok=True);shutil.copytree(F,delivery/'figures',dirs_exist_ok=True)
for f in ['WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md','WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md','WORLDSIM_V8_1_TO_V8_2_DECISION.md']:
 write(delivery/f,(R/'docs'/f).read_text().replace('figures/worldsim_v81_closeout/','figures/'))
write(delivery/'V81-F05.md',failure.replace('../../WORLDSIM_','WORLDSIM_'))
shutil.copytree(D,delivery/'evidence',dirs_exist_ok=True)
for f in ['selected_before_model.json','visual_reviews_before_model.json','interventions_before_model.json','final_intervention_protocol.json','registry.json']:shutil.copy2(A/f,delivery/'evidence'/f)
body=['<h1>V8.1 有限补证与方法族 badcase report</h1><p>8 新日志筛查 · 3 有效观测干预 · 18 新推理 · 旧 440 项复用</p><p><b>V8.2 NO_GO；当前共同失效立论降低优先级。</b>保留可恢复深度候选与灰墙良好案例。其他方法尚未实测的机制不写成已证实失败。</p>']
for f,label in [('WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md','完整收口报告'),('WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md','十个方法的 badcase report'),('WORLDSIM_V8_1_TO_V8_2_DECISION.md','四环推进判定')]:body.append(f'<p><a href="{f}">{label}</a></p>')
for f,label in [('four_link_architecture.png','四环证据架构'),('new_evidence_summary.png','VGGT：普通锚点后仍有局部改善空间'),('retained_gray_wall_goodcase.png','保留297点、DVGT0.103m灰墙goodcase'),('av2_effective_evidence_before_model.png','模型输出之前：确认真实有效观测')]:body.append(f'<h2>{label}</h2><img src="figures/{f}">')
for f in sorted(F.glob('*.png')):
 if f.name in ['four_link_architecture.png','new_evidence_summary.png','retained_gray_wall_goodcase.png','av2_effective_evidence_before_model.png']:continue
 body.append(f'<details><summary>{html.escape(f.stem)}</summary><img loading="lazy" src="figures/{f.name}"></details>')
write(delivery/'index.html','<!doctype html><meta charset="utf-8"><title>V8.1 有限收口</title><style>body{font:16px/1.7 system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#213944}img{width:100%;height:auto}details{border-top:1px solid #ccd7db;padding:12px}a{color:#146487}h1{font-size:28px}</style>'+''.join(body))
with tarfile.open(O/'v81_closeout_report.tar','w') as tf:tf.add(delivery,arcname='V81_Closeout_Report')
print(json.dumps({'figures':len(list(F.glob('*.png'))),'delivery':str(delivery),'completion':reg['state']}))
