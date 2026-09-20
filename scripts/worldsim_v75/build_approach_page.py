"""从已完成四组反馈生成本地审阅页，不手填推理结果或补画事故。"""
import argparse,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--evidence',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    src,out=a.evidence,a.output; out.mkdir(parents=True,exist_ok=True)
    data=json.loads((src/'comparison.json').read_text()); assert data['status']=='complete' and len(data['cases'])==4
    for name in ['comparison.json','closed-loop-results.png','closed-loop-results.svg','actual-policy-inputs.jpg','provenance.json']:
        shutil.copy2(src/name,out/name)
    names={'gt_clean':'GT条件','dvgt_metric':'DVGT距离读出','dvgt_lidar_scaled':'普通全局尺度','reference_lidar':'目标LiDAR参照'}
    rows=''.join(f'<tr><td>{names[r["arm"]]}</td><td>{r["initial_target_center_residual_m"]:.3f}</td><td>{r["progress_m"]:.3f}</td><td>{r["progress_change_vs_gt_m"]:+.3f}</td><td>{r["mean_abs_acceleration_change_vs_gt_mps2"]:.3f}</td><td>{r["final_target_reference_clearance_m"]:.3f}</td></tr>' for r in data['cases'])
    diagram='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 190" role="img" aria-label="生成闭环架构">
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#4d738c"/></marker></defs>
    <g fill="#eef5fa" stroke="#89a5b9"><rect x="8" y="30" width="180" height="65" rx="8"/><rect x="238" y="30" width="180" height="65" rx="8"/><rect x="468" y="30" width="180" height="65" rx="8"/><rect x="698" y="30" width="180" height="65" rx="8"/><rect x="928" y="30" width="180" height="65" rx="8"/></g>
    <g font-family="sans-serif" text-anchor="middle" fill="#203c50"><g font-size="17"><text x="98" y="58">目标重建位置</text><text x="328" y="68">OmniDreams RGB</text><text x="558" y="58">检测 / 运动状态</text><text x="788" y="58">IDM → 动作</text><text x="1018" y="58">ego / 相机</text></g><g font-size="13"><text x="98" y="81">GT / DVGT / 尺度 / LiDAR</text><text x="558" y="81">真实历史预热 + 已知地形</text><text x="788" y="81">固定路线与控制参数</text><text x="1018" y="81">官方动力学与地面贴合</text></g></g>
    <g fill="none" stroke="#4d738c" stroke-width="2" marker-end="url(#a)"><path d="M190 62H235"/><path d="M420 62H465"/><path d="M650 62H695"/><path d="M880 62H925"/><path d="M1018 97V136H328V98"/></g>
    <text x="600" y="174" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#4d738c">实际反馈：每组117帧 / 15次决策；固定真实目标用于独立评价</text></svg>'''
    (out/'architecture.svg').write_text(diagram,encoding='utf-8',newline='\n')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 接近车辆的实际生成闭环</title><style>
    body{max-width:1160px;margin:38px auto;padding:0 24px;background:#f5f7fa;color:#203444;font:17px/1.75 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:36px;line-height:1.3}h2{font-size:24px;margin-top:36px}img,video{width:100%;border-radius:10px;background:white}.card{background:white;border:1px solid #dce4eb;border-radius:12px;padding:22px;margin:24px 0}.tag{color:#52798e;font-size:14px}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce4eb}a{color:#00729e}details{margin:24px 0}</style>
    <p class="tag">V7.5 · 2026-09-20 · 单卡全部完成 · 短时接近 / 制动任务</p>
    <h1>重建位置变化，怎样改变<br>生成观测下的制动与执行？</h1>
    <p>本轮完成同一真实接近场景的四组实际反馈：生成RGB驱动策略，动作更新ego与相机，再生成下一段。停止追“长时间放大”，直接观察任务状态和执行响应。</p>
    <img src="architecture.svg" alt="输入、世界模型、策略、动力学与反馈数据流">
    <div class="card"><strong>有明确任务响应；几何修正尚未稳定恢复动作。</strong><p>DVGT中心残余1.67米，对应相对GT条件分支少行进1.35米。目标LiDAR将中心残余降至0.25米，平均动作差却由0.50增至0.68 m/s²；终点差仍约1.30米。四组均无参考重叠。不能据此声称严重驾驶失效、false-safe或跨SOTA普遍缺口。</p></div>
    <h2>四组原始反馈视频</h2><p>相同初帧、seed42、策略和其他条件；只改目标初始平移，并保持这项偏移。每格显示实际施加的加速度。LiDAR两组使用额外度量信息。</p>
    <video controls preload="metadata" src="four-arm-feedback.mp4"></video>
    <h2>状态输入与实际执行</h2><table><tr><th>输入</th><th>中心残余 m</th><th>行进 m</th><th>行进变化 m</th><th>平均动作差 m/s²</th><th>末端真实目标间距 m</th></tr>'''+rows+'''</table>
    <p>动作差相对于GT条件的生成分支，不是相对于最优驾驶。间距是虚拟ego与固定真实目标的二维足迹距离。覆盖0–3.867秒，未声称完成停车或证明长期安全。</p>
    <img src="closed-loop-results.png" alt="实际ego轨迹、制动曲线、真实目标间距和执行差">
    <details><summary>实际策略看到什么：绿色框为当时选中的前车</summary><img src="actual-policy-inputs.jpg" alt="四组实际策略输入"></details>
    <h2>先排除普通基线造成的假问题</h2><p>桥面起伏使单平面在目标处高估地面约0.98米。官方高程栅格把真实RGB距离误差从9.84降到1.19米；使用起点前固定20张真实图像初始化跟踪后，15次真实前车判断全部正确，动作差中位0.26 m/s²。</p>
    <p>首次GT反馈还发现跟踪关联错误：0.51米的正确匹配被无效远距离配对挤掉，引起ID切换和错误加速。将同一10米门控移到Hungarian分配之前修复；原失败保留。修复后的完整GT反馈通过任务检查，最大欠制动差由2.29降至1.07 m/s²。没有调seed、距离阈值或噪声参数。</p>
    <h2>重建与参考的证据边界</h2><p>官方DVGT前向接7张起点前RGB。位置读出额外用已知射线、GT尺寸与朝向，不能冒称原生端到端场景重建排名。原始点图朝向问题单独留档。</p>
    <p>普通全局尺度使用29486个背景LiDAR锚点，反而把本目标残余增至9.48米。目标参照原扫描仅4点，未达既定门槛；补入之前两次扫描后得到42点、目标变化0.008米、参考残余0.253米。所有点严格早于或等于生成起点，不使用GT平移作运动补偿。原4点拒绝记录保留。</p>
    <h2>下一项有判别力的控制</h2><p>对这四份已冻结状态，用相同动力学和普通IDM直接读取三维状态，比较无需生成就可解释的影响，以及生成RGB/状态读出额外引入的偏差。这是额外状态信息的诊断；不追加seed、时长、新模型或训练。</p>
    <p class="tag">一个已暴露开发日志；非反应式记录交通；已知路线/标定/地形；GT尺寸朝向及其他轨迹共享。5次完整生成含一次工程修复前GT，共585帧；最终比较468帧/60决策。另1次DVGT和36次真实检测，均无OOM，人工verdict为null。</p>
    <p><a href="comparison.json">完整结果</a> · <a href="provenance.json">来源与资源</a> · <a href="closed-loop-results.svg">导出SVG图</a> · <a href="../V75_Braking_Task_Research/index.html">前一轮任务筛查</a></p></html>'''
    (out/'index.html').write_text(html,encoding='utf-8',newline='\n'); print(out/'index.html')


if __name__=='__main__': main()
