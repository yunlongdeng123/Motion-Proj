"""用户要求回退后的最终报告；严格区分计划、已完成与已停止。"""
import argparse,pathlib,json
def load(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);o=p.parse_args().output
 ev=load(o/'evidence_summary.json');g={s['scene']:s for s in load(o/'guard_summary.json')['scenes']};s=load(o/'summary.json');st=load(o/'drive_state.json');obs=load(o/'observations.json');v=load(o/'validation.json');rows=[]
 for e in ev['scenes']:
  n=e['scene'];x=g[n];rows.append(f"|{n}|{e['evidence_pixels']}/{e['mask_pixels']} ({e['evidence_pixels']/e['mask_pixels']:.3%})|{x['source_target_detected_frames']}/{x['frames']}|{len(x['arms']['precise']['blocked_frames'])}/{x['frames']}|{len(x['arms']['evidence_first']['blocked_frames'])}/{x['frames']}|")
 observations='\n\n'.join('### '+n+'\n\n'+'\n\n'.join(x['observations']) for n,x in obs.items())
 text=f'''# v77 DELETE：三案例失败记录与方案回退

**已按用户要求停止退化实验，默认方案指回 `WS-V77-DELETE-FULL-20260927/r1`。** 原DriveEditor配置、冻结Ω和原GLB不改；旧版的再生车、模糊与错误遮挡仍公开，回退不代表质量通过。[打开审核HTML](index.html)、[当前配置](rollback.json)、[回退验证](rollback_validation.json)。

```mermaid
flowchart LR
 R[原RGB + 旧mask配置] --> D[DriveEditor trained deletion]
 D --> O[冻结VGGT-Ω背景]
 O --> B[B + 原GLB]
 B --> Q[factual / DELETE]
 R --> T[试验: SAM2精确mask<br/>真实RGB证据warp]
 T --> G[DriveEditor仅补residual]
 G --> C[车辆guard + 视觉检查]
 C --> F[白车/灰白残影<br/>记录V77-F02并停用]
 F --> D
```

## 实际做了什么

唯一task/run：`WS-V77-DELETE-REPAIR-20260927/r1`。0230/actor22/CAM5选f18–47，0255/actor25/CAM3选f65–94，覆盖旧失败后段；新增official_000/actor12/CAM0选f0–29，是右侧黑色MPV，3秒GT平面位移约7.1m。三个已曝光开发scene，非独立测试。目标身份与DELETE语义核对通过；只移除目标，不执行MOVE，不宣称完整交通规则审核。

两组同权重、seed42、25步、1024×576、10帧窗口、stride9与同时间重叠条件：A精确mask直接生成，B先填可信原RGB、只生成residual。0230没有可接受证据，B复用A，不重复推理。模型为官方训练后DriveEditor deletion，沿用已有3090串行CFG适配，未加其他补景模型。

用户要求“差就回退”后，已停止DriveEditor。完成 **{len(st['completed'])}个窗口**，下一窗口中断；两旧scene的A/B各30帧3秒完整保存。第三例A完成30帧3秒，B仅10帧1秒；共同时间对照只用这10帧，A完整3秒另外保留。没有补帧、重复尾帧伪装长度或补跑已失败配置。实际数量见[完成帧表](comparison_plan.json)。

这是补景入口试验，不是新跑三scene完整六相机世界。第三个目标尚无旧DriveEditor/GLB基线，只保留为第三失败诊断案例；不能伪称回退至不存在的旧结果。当前默认工程仍是原来的两个scene。

## mask与真实证据

SAM2.1-large视频实例mask保留原稿；第三例新跑视频传播。box只提示/约束，mask为SAM轮廓∩GT投影凸包+3px，最大连通分量清理后膨胀3px且仍受投影约束。0230编辑范围减少39.92%，0255减少82.96%。它限制了被改写区域，但轮廓不保证完美，不自动涵盖阴影，也可能丢弃被遮挡后分离的小部件。

从整段六相机原RGB的冻结Ω缓存寻找来源，每10帧采样，排除GT身份缺失时刻。使用GT相机和既有框外LiDAR尺度；源射线不能穿过贴地对象包络（向下延60cm），源3D点须有20cm内同帧背景LiDAR支持及低深度梯度。3×3 splat warp后，两来源至少相隔0.5秒，深度差≤25cm、RGB最大差≤25/255，最后腐蚀1px。接受的真实RGB像素及双来源索引保存，生成不能覆写它们。未使用旧生成图作为真实证据。

这些是固定保守规则，未扫阈值。低覆盖也可能由Ω深度误差、稀疏LiDAR或严格遮挡排除导致，不能解释为“背景从未被拍过”。围栏等静态前景也是删除后应该保留的内容。以下覆盖率分母是准备的每scene30帧，并非第三例只完成的B前10帧。

|场景|真实证据/删除像素·帧|原图目标检测|A车辆拦截|B车辆拦截|
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 视觉反例与退化

{observations}

旧/新的初始条件历史不同，旧全段与新短窗不是严格mask单变量控制；A/B才是本轮固定窗口比较。精确mask范围改善，不能被升级为无车背景成功。

## Guard为什么不能单独决定通过

GroundingDINO-tiny检测 `car. truck. bus. van.`，box/text阈值0.25，SAM2取得车辆实例。检测实例与编辑区交叠≥32像素、覆盖原目标核心≥5%，且不匹配原图非目标实例（IoU<0.5）时拦截。源图正控制和原先0230/f25再生车控制同时保存，空路面负控制无命中。

0230清晰白车29/30帧被抓到；0255 A为0/30、B为3/30，第三例两组均0/10，尽管灰白残影肉眼明显。说明“没有检测到车”不能等于背景完成。原图目标检测全部命中也不能消除这种残影漏检。真实露出的后方车辆、分割误差也可能导致误拦截。

原图非目标实例mask中有少量像素变化（例如0255约1.32%，包含分割误差），因此不声称邻车逐实例完全保护。mask之外RGB严格不变是另一个像素合同，不能替代语义完整性。

`repair_admission.approved_paths()`遇检测阳性、原图漏检或未通过助手视觉检查不给Ω返回背景路径。实际三例均无准入路径，[记录](background_admission.json)保留。**本轮零新Ω前向、原GLB不改；不把补景失败归为资产网格失败。** 人工verdict始终null。

## 回退与复核

默认配置写入 `configs/worldsim_v77/delete_pipeline_current.json`：active为旧FULL/r1，细轮廓直接补景试验disabled。旧BUILD/QUERY入口与 `1a2d5189` 比较无改动；完整旧资产、视频与本轮失败原生PNG/掩码/证据均保留，不改写Git历史。

本地HTML上方直接复用旧版5秒/10秒的原视频/factual/DELETE三栏；下方展示三案例退化、同相机旧/新补景、mask、来源、原生输出与guard。没有将失败的第三例加入默认工程。

8项语义/准入测试通过。{v['frames']}帧证据准备来源验证，{v['evidence_copied_pixels']}个候选复制像素可回查原RGB；已生成图检查mask外不变与证据锁定。{s['total_videos']}段本轮视频/{s['total_decoded_frames']}帧实解码通过。页面链接和JS语法检查通过，未冒称执行浏览器交互测试。首窗漏带已有串行CFG导致OOM，零输出；恢复原配置后运行，失败日志保留。最大已分配显存{max(x['peak_gib'] for x in st['completed']):.2f}GiB；已完成窗口计算合计{sum(x['seconds'] for x in st['completed']):.1f}秒，不包含中断窗口与模型加载。

## 本轮留下的边界

精确写回能约束误改范围，但把车形mask直接送入现成DriveEditor并不保证去车；本轮真实证据太少，不能检验“充分真实背景条件能否消除幻觉”。[官方训练代码](https://github.com/yvanliang/DriveEditor/blob/main/sgm/data/nus.py)的get_blank使用矩形洞，官方deletion也用扩大矩形；精确轮廓改变了输入mask分布，但尚未证明这就是失败的唯一原因。

先恢复旧方案，不机械重复这条失败入口。若后续继续修复，可单独验证生成条件范围与最终精确写回范围是否需要分开；那是未执行的后续控制，不能写成已改善结果。当前不需要凭这些失败就自造大训练集，不拿生成图当真实背景GT，也不宣布整个显式资产路线失败。

`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
'''
 (o/'report.md').write_text(text,encoding='utf8')
if __name__=='__main__':main()
