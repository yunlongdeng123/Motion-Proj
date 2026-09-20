"""用真实记录和已完成策略调用生成静态研究图及本地审阅页。"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    src=args.evidence; out=args.output; out.mkdir(parents=True,exist_ok=True)
    result=json.loads((src/'interface-audit/result.json').read_text())
    control=json.loads((src/'interface-audit/state_control_result.json').read_text())
    source=json.loads((src/'task_sources.json').read_text())
    assert result['status']=='complete' and source['window_count']==48 and not source['cases']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,10),layout='constrained')
    gs=fig.add_gridspec(2,3,height_ratios=[1,1])
    ax=fig.add_subplot(gs[0,:2]); r=result['rows'][1]
    ax.imshow(Image.open(src/'interface-audit/real-015.png'))
    for b,color in [(r['target_projection']['bounds'],'#ffc629'),(r['policy']['lead']['box'],'#10dfa1')]:
        ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,lw=2))
    ax.set(xlim=(400,1050),ylim=(620,230)); ax.axis('off')
    ax.set_title('(a) Real RGB at 0.5 s: yellow GT target / green policy detection',loc='left',fontsize=12)
    ax=fig.add_subplot(gs[0,2]); ax.axis('off')
    ax.text(0,.98,'No world model used here',fontweight='bold',fontsize=14,va='top')
    ax.text(0,.86,'Same visible target\n\nGT gap: 39.72 m\nRGB policy: 26.90 m\n\nGT lead speed: 0.00 m/s\nRGB policy: 5.06 m/s\n\nYet braking:\nGT: -2.008 m/s²\nRGB: -2.071 m/s²',va='top',fontsize=13,linespacing=1.45)
    t=np.array([r['frame']/30 for r in result['rows']])
    ax=fig.add_subplot(gs[1,0]); ax.plot(t,[r['reference_target_gap_without_40m_filter_m'] for r in result['rows']],'-o',label='GT target gap')
    ax.plot(t,[r['policy']['lead']['gap_m'] for r in result['rows']],'-s',label='RGB policy')
    ax.axhline(40,color='#777',ls=':',label='Policy 40 m range')
    ax.set(xlabel='Time (s)',ylabel='Gap (m)',title='(b) Large distance bias'); ax.legend(fontsize=9); ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[1,1]); valid=result['rows'][1:]
    ax.plot(t[1:],[r['oracle']['lead_speed_mps'] for r in valid],'-o',label='GT past-motion speed')
    ax.plot(t[1:],[r['policy']['lead']['lead_speed_mps'] for r in valid],'-s',label='RGB tracking')
    ax.set(xlabel='Time (s)',ylabel='Lead speed (m/s)',title='(c) Speed bias acts oppositely'); ax.legend(fontsize=9); ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[1,2]); keys=['reference','rgb_gap_only','rgb_velocity_only','both_rgb']
    v=control['rows'][0]['acceleration_mps2']; bars=ax.bar(range(4),[v[k] for k in keys],color=['#277da8','#dd7955','#aa71b2','#278c70'])
    for bar,k in zip(bars,keys): ax.text(bar.get_x()+bar.get_width()/2,v[k]-.06,f'{v[k]:.3f}',ha='center',va='top',fontsize=10)
    ax.set_xticks(range(4),['GT state','RGB gap\n+ GT speed','GT gap\n+ RGB speed','Both RGB'])
    ax.set(ylim=(-3.5,.1),ylabel='IDM acceleration (m/s²)',title='(d) Fixed IDM, offline state substitution')
    ax.tick_params(axis='x',labelsize=9); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Similar actions can conceal different state errors',fontsize=18,fontweight='bold')
    for suffix in ['png','svg']: fig.savefig(out/f'state-action-controls.{suffix}',dpi=180)
    plt.close(fig)
    for f in ['task_sources.json','protocol.json']: shutil.copy2(src/f,out/f)
    for f in ['result.json','state_control_result.json','real-policy-review.jpg']: shutil.copy2(src/'interface-audit'/f,out/f)
    diagram='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 170" role="img" aria-label="闭环研究组件图">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#546c80"/></marker></defs>
    <g fill="#edf4fa" stroke="#7796ad"><rect x="8" y="30" width="164" height="60" rx="8"/><rect x="205" y="30" width="164" height="60" rx="8"/><rect x="402" y="30" width="164" height="60" rx="8"/><rect x="599" y="30" width="164" height="60" rx="8"/><rect x="796" y="30" width="194" height="60" rx="8"/></g>
    <g font-family="sans-serif" font-size="16" text-anchor="middle" fill="#203d50"><text x="90" y="66">重建场景状态</text><text x="287" y="66">世界模型生成 RGB</text><text x="484" y="55">任务状态</text><text x="484" y="77" font-size="13">路径占用 / 距离 / 相对速度</text><text x="681" y="66">策略 → 动作</text><text x="893" y="55">ego / 相机更新</text><text x="893" y="77" font-size="13">任务结果与独立参考</text></g>
    <g fill="none" stroke="#546c80" stroke-width="2" marker-end="url(#arrow)"><path d="M172 60H202"/><path d="M369 60H399"/><path d="M566 60H596"/><path d="M763 60H793"/><path d="M893 92V127H287V93"/></g><text x="555" y="151" font-size="14" font-family="sans-serif" text-anchor="middle" fill="#546c80">实际反馈已接通；本轮真实 RGB 检查任务状态 → 策略这一段</text></svg>'''
    (out/'architecture.svg').write_text(diagram,encoding='utf-8')
    rows=''.join(f'<tr><td>{r["log_id"][:8]}</td><td>{r["offset_seconds"]:.1f}</td><td>{r["recorded_speed_drop_mps"]:.2f}</td><td>{r.get("oracle_braking_times",0)}/5</td><td>{"起点在40m外" if r.get("oracle_braking_times",0)>0 else "无匹配的前车制动需求"}</td></tr>' for r in source['rows'] if 'base' in r)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 闭环任务与状态</title><style>
    body{max-width:1120px;margin:40px auto;padding:0 24px;background:#f5f7fa;color:#203242;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:34px;line-height:1.3}h2{font-size:23px;margin-top:38px}img{max-width:100%;background:white;border-radius:10px}.card{background:white;border:1px solid #d8e0e8;border-radius:12px;padding:22px;margin:22px 0}.tag{color:#49768f;font-size:14px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}a{color:#006d9e}details{margin:20px 0}strong{color:#163c51}</style>
    <p class="tag">V7.5 · 2026-09-20 · 任务优先 · 全部当前调用已结束</p><h1>哪些重建误差会改变<br>生成世界模型中的驾驶状态？</h1>
    <p>从真实反馈任务回答，停止追加“长时间放大”。前一轮四组生成反馈已有小幅可恢复响应，尚无严重驾驶 badcase。本轮检查真实制动需求与策略基线。</p>
    <img src="architecture.svg" alt="闭环研究组件、状态和实际反馈">
    <div class="card"><strong>本轮结论：动作相近不代表任务状态正确。</strong><p>真实 RGB 中，距离与前车速度误差在同一个 IDM 中出现抵消。这个问题出现在基线；不能归因于生成模型，也不能用接近参考的动作掩盖它。</p>
    <p>12 条日志 × 4 个固定起点 → 48 个窗口 → 9 个记录减速窗口 → 0 个通过原定持续前车入口。另做一次明示事后的真实 RGB 入口诊断：5 次检测，0 次重建、0 次世界模型生成、无 OOM。</p></div>
    <h2>真实图像 → 状态偏差 → 策略响应</h2><p>示例为 02678d04、起点 +6.5 秒。0.5 秒时，目标真实距离 39.72 米、策略估计 26.90 米；静止目标被估计以 5.06 m/s 前进。距离偏小增加制动，速度偏大减少制动。下图用同一个官方 IDM 离线替换状态分量验证这一点；GT 替换提供额外信息，不是合法输入下的方法收益。</p>
    <img src="state-action-controls.png" alt="真实图像、距离和速度偏差、同一IDM的状态替换结果">
    <p>五次观测的范围内前车判断 4/5；正确匹配的四次距离绝对误差中位 9.84 米；加速度绝对差中位 0.302 m/s²。起点“判断不一致”是同一可见车辆被错误估入 40 米范围内，不能称为漏检。距离未满足既有 3 米基线要求，因此没有进入生成或误差对照。</p>
    <details><summary>查看完整五时刻真实 RGB 与实际策略选择</summary><img src="real-policy-review.jpg" alt="五时刻原图和实际前车检测框"></details>
    <h2>保留筛查分母和入口限制</h2><p>已冻结的入口要求起点就有 40 米内前车；唯一具有明显前车制动需求的片段，起点目标距 45.12 米，随后才进入范围，因此被排除。该限制不等于物理场景没有有价值任务。原始筛查结果保持 0 合格，没有改门槛重算，也没有新增来源补位。</p>
    <table><tr><th>日志</th><th>起点 / s</th><th>记录减速 / m/s</th><th>前车制动时刻</th><th>原入口排除</th></tr>'''+rows+'''</table>
    <h2>与已有闭环证据的关系</h2><p>前一轮真正生成的四组各 117 帧、15 次决策：DVGT、普通全局尺度、目标 LiDAR 相对 GT 条件的终点差分别为 0.353 / 0.242 / 0.104 米，均无制动或参考重叠。几何误差范数无法直接排序任务影响。<a href="../V75_Closed_Loop_Research/index.html">查看实际反馈视频与原报告</a>。</p>
    <p>尚未证明生成世界模型中哪一类自然重建误差会稳定损害制动任务。下一步应解决任务输入与普通状态估计的适用性，再比较路径占用、相对距离/速度及进入路径时机；不用增加 seed、时长或扰动来维持主张。</p>
    <p class="tag">开发来源；非反应式记录交通；已知路线/地图地面与标定；人工 verdict 为 null。此页不是 main-paper 成功结果。</p>
    <p><a href="task_sources.json">48窗口原始结果</a> · <a href="result.json">真实策略记录</a> · <a href="state_control_result.json">离线状态控制</a> · <a href="state-action-controls.svg">可导出 SVG 图</a></p></html>'''
    (out/'index.html').write_text(html,encoding='utf-8')
    # Windows生成的SVG统一LF与行末，保证远端Git检查不把CRLF当作空白错误。
    for name in ['state-action-controls.svg','architecture.svg']:
        target=out/name
        target.write_bytes(('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    print(out/'index.html')


if __name__=='__main__': main()
