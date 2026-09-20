"""由已保存的真实反馈结果构建中文审阅页；视频和曲线均来自实际执行。"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    data=json.loads((a.evidence/'comparison.json').read_text());rows={r['arm']:r for r in data['cases']}
    assert len(rows)==4 and all(r['reference_overlap_frames']==0 and r['braking_decisions']==0 for r in rows.values())
    for name in ['comparison.json','closed-loop-results.svg','closed-loop-results.png','actual-policy-inputs.jpg','four-arm-feedback.mp4',
                 'real-policy-review.jpg','real-policy-failed.jpg','real-policy-second-pass.jpg','real_policy_result.json',
                 'feedback_protocol.json','source_observability_audit.json']:
        if (a.evidence/name).resolve() != (out/name).resolve():shutil.copy2(a.evidence/name,out/name)
    names={'gt_clean':'GT状态','dvgt_metric':'自然DVGT读出','dvgt_lidar_scaled':'普通全局LiDAR尺度','reference_lidar':'目标LiDAR修复'}
    table=''.join(f'<tr><td>{names[k]}</td><td>{r["initial_target_center_residual_m"]:.3f}</td><td>{r["progress_m"]:.3f}</td><td>{r["end_position_change_vs_gt_m"]:.3f}</td><td>{r["mean_abs_acceleration_change_vs_gt_mps2"]:.3f}</td><td>0/15</td></tr>' for k,r in rows.items())
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 · 首个生成式闭环对照</title>
<style>body{{margin:0;background:#edf1f4;color:#172a3e;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1350px;margin:auto;padding:35px 24px}}h1{{font-size:32px;line-height:1.35}}h2{{font-size:22px}}section{{background:white;border-radius:10px;padding:25px;margin:22px 0}}.tag{{color:#167567;font-weight:700;letter-spacing:2px}}.lead{{font-size:19px}}.chain{{display:flex;flex-wrap:wrap;align-items:center;gap:9px}}.box{{flex:1;min-width:130px;padding:15px;border-radius:7px;background:#173950;color:white}}.loop{{border:2px solid #167567;border-top:0;text-align:center;color:#167567;margin:0 25px;padding:8px}}.note{{background:#fff4da;padding:16px;border-left:4px solid #bb902d}}img,video{{width:100%;height:auto}}a{{color:#176f8c}}small{{color:#617080}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #dfe7eb;text-align:left}}.scroll{{overflow-x:auto}}</style>
<main><div class="tag">V7.5 · ACTUAL CLOSED-LOOP EVIDENCE · 2026-09-20</div>
<h1>哪些重建误差会改变生成状态与闭环任务？</h1>
<p class="lead">第一轮真实反馈已完成：自然DVGT使终点相对GT条件偏移<b>{rows['dvgt_metric']['end_position_change_vs_gt_m']:.2f}米</b>，目标LiDAR修复后为<b>{rows['reference_lidar']['end_position_change_vs_gt_m']:.2f}米</b>。这是小幅、可部分恢复的任务响应；没有出现制动或参考重叠。</p>
<section><h2>这次执行了完整反馈边</h2><div class="chain"><div class="box">重建状态<br>＋普通控制</div>→<div class="box">Ludus条件<br>cuboid / 地图</div>→<div class="box">OmniDreams<br>连续cache → RGB</div>→<div class="box">RGB感知＋IDM<br>实际策略决策</div>→<div class="box">动力学执行<br>ego / 相机</div></div><div class="loop">← 执行状态决定下一段条件 ←</div>
<p>每组117帧、15次决策、seed42。第一条策略输入为真实初帧，其余14次读取上一段生成RGB。四组共468生成帧、56次基于生成画面的反馈决策。参考actors保持独立；没有把错误状态同步写入评价真值。</p>
<p>策略为单前视角感知＋官方nuPlan IDM＋普通路线追踪，使用已知标定、ego、路线和地图地面。它是当前任务的普通控制器，不是端到端驾驶SOTA。</p></section>
<section><h2>四组实际生成视频</h2><video controls preload="metadata" src="four-arm-feedback.mp4"></video><small>左上GT状态，右上自然DVGT，左下普通全局尺度，右下目标LiDAR修复。这里的GT列也是生成闭环；偏离日志轨迹后没有真实像素GT。</small></section>
<section><h2>生成画面 → 当前策略动作</h2><img src="actual-policy-inputs.jpg" alt="实际生成的策略输入，绿框为策略所选前车，标注下一段加速度">
<p>绿框来自实际RGB策略。1.47秒时自然DVGT组选择的前车距离更近，加速度较低；2.8秒附近下一辆远车接近固定40米感知边界，部分组尚未选入。因此动作差包含普通感知范围和跟踪的影响，不解释成长时间放大。</p></section>
<section><h2>几何范数不能直接代替任务影响</h2><div class="scroll"><table><tr><th>条件</th><th>初始中心残余(m)</th><th>执行进度(m)</th><th>终点相对GT差(m)</th><th>平均绝对加速度差(m/s²)</th><th>制动</th></tr>{table}</table></div>
<p>目标LiDAR修复使终点和平均动作更接近GT条件。普通尺度的几何残余更大，但终点差反而较小；本例不支持按中心误差范数直接排列闭环危害。LiDAR控制提供额外度量观测，GT尺寸/朝向和未来actor运动均为共享额外条件。</p>
<img src="closed-loop-results.svg" alt="输入目标轨迹、实际动作、逐帧参考间距及执行差">
<p class="note">四组均无制动，逐帧参考重叠均为0/117；共同最小间距2.246米来自相同初始场景。结果证明了有限的反馈响应与部分恢复，尚未证明任务质量受损、修复改善安全性或普遍SOTA失败。BEV虚线为输入目标条件，不是从生成视频测得的3D几何。</p></section>
<section><h2>先验证真实输入，并保留完整分母</h2><p>先前两条开发日志不满足持续跟车任务。随后固定四条已有日志：三条有路径前车，两条真实前视角基线合格；按冻结顺序选首条生成，没有按重建误差大小挑选。</p>
<table><tr><th>日志</th><th>结果</th></tr><tr><td>05fa5048</td><td>无持续初始前车</td></tr><tr><td>0bae3b5e</td><td>4/5判断正确；距离误差中位0.749米；选中</td></tr><tr><td>0c3bad78</td><td>GT路径前车五时刻均完全位于前视图像之外；输入不可观测，不算检测器失败</td></tr><tr><td>0fb7276f</td><td>4/5判断正确；距离误差中位0.442米；保留，未补跑生成</td></tr></table>
<details><summary>真实输入基线与排除证据</summary><img src="real-policy-review.jpg"><img src="real-policy-failed.jpg"><img src="real-policy-second-pass.jpg"></details></section>
<section><h2>边界与下一项工作</h2><p>这是一条已曝光开发任务、一个seed、约4秒、普通控制器的结果。车辆采用官方demo默认虚拟ego和平地动力学，traffic为非反应式记录；没有覆盖信号灯、完整行人决策或一般驾驶规则。生成相机是否精确遵循请求也未独立确认。</p>
<p>本例收为小幅可恢复响应，不追加seed、扰动或更长窗口将其升级成事故。后续先按真实跟车/制动需求定义任务，研究路径占用、相对距离和进入/离开路径时机，继续判断哪些状态误差值得修。</p>
<p>四组均完整解码117帧；逐帧参考和60次决策因果顺序检查通过。每组约73秒，PyTorch峰值12.97GiB，无OOM。</p>
<p><a href="feedback_protocol.json">冻结反馈协议</a> · <a href="comparison.json">完整比较</a> · <a href="real_policy_result.json">原始真实输入策略结果</a> · <a href="source_observability_audit.json">相机观测支持更正</a> · <a href="../V75_Visible_Cohort_Report/index.html">此前可见性实验正反结果</a></p><small>原始输入、模型输出与日志保留远端；人工verdict：null。当前状态仅维护于 docs/RESEARCH_STATUS.md。</small></section></main></html>'''
    (out/'index.html').write_text(html,encoding='utf-8',newline='\n')
    print(out/'index.html')


if __name__=='__main__':main()
