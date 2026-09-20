"""生成反事实质量定义审阅页，并完整归档已结束的速度先验控制。"""
import argparse
import json
import shutil
from pathlib import Path
from html import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def architecture(out):
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.set(xlim=(0, 15), ylim=(0, 5.6)); ax.axis('off')
    def box(x, y, w, label, color='#e5f1f3'):
        ax.add_patch(FancyBboxPatch((x, y), w, .84, boxstyle='round,pad=0.08',
                                  facecolor=color, edgecolor='#426070', linewidth=1.2))
        ax.text(x+w/2, y+.42, label, ha='center', va='center', fontsize=11, color='#203c50')
    def arrow(a, b, label=None, rad=0, color='#426070'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                                    connectionstyle=f'arc3,rad={rad}', color=color, linewidth=1.35))
        if label: ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.12, label, ha='center', fontsize=9, color=color)
    box(.2, 3.5, 2, 'Allowed\nobservations H')
    box(2.95, 3.5, 2, 'Reconstructed\nworld state R(H)')
    box(5.75, 3.5, 2.35, 'Edit I +\nstate encoder')
    box(8.95, 3.5, 2.15, 'Generator\n+ history')
    box(12, 3.5, 2.6, 'Generated\nsensor observations')
    for a,b in [((2.3,3.92),(2.85,3.92)),((5.05,3.92),(5.65,3.92)),((8.2,3.92),(8.85,3.92)),((11.2,3.92),(11.9,3.92))]: arrow(a,b)
    box(12, 1.85, 2.6, 'Fixed policy')
    box(8.95, 1.85, 2.15, 'Dynamics /\ntraffic model')
    arrow((13.3,3.4),(13.3,2.79))
    arrow((11.9,2.27),(11.2,2.27),'action')
    arrow((9.1,2.79),(7.25,3.4),'next state')
    box(.2, .4, 3.2, 'Independent reference\nS + same semantic edit I', '#fff0da')
    box(5.15, .4, 5.2, 'State / visibility / task fidelity\ncompare against supported reference', '#fff0da')
    arrow((3.5,.82),(5.05,.82))
    arrow((9.9,1.75),(9.3,1.34))
    ax.plot([14.7,14.92,14.92,10.75],[3.92,3.92,.82,.82],color='#426070',lw=1.35)
    arrow((10.75,.82),(10.45,.82))
    ax.text(.2,5.05,'WorldSim quality = fidelity to the edited world and its consequences',
            fontsize=17, weight='bold', color='#203c50')
    ax.text(.2,4.68,'Reference is for evaluation only. Policy success alone does not establish simulator fidelity.',
            fontsize=11, color='#546c7b')
    fig.tight_layout(pad=.5)
    for ext in ['png','svg','pdf']: fig.savefig(out/f'architecture.{ext}', dpi=170)
    plt.close(fig)


def archive_velocity(src, archive, out):
    r = json.loads((src/'result.json').read_text())
    assert r['status']=='complete' and len(r['cases'])==20 and not r['generation_admitted']
    archive.mkdir(parents=True, exist_ok=True)
    for f in src.rglob('*.json'):
        dest=archive/f.relative_to(src); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dest)
    # 从逐决策记录重算指标，防止展示选择性丢弃结果。
    for c in r['cases']:
        rows=json.loads((src/c['log']/Path(c['detail']).name).read_text())['rows']
        assert len(rows)==15
        for mode in ['old','zero']:
            for key, rs, field in [('physical',rows,'reference'),('condition_after_startup',rows[1:],'condition')]:
                e=np.array([row[f'{mode}_minus_{field}_mps2'] for row in rs])
                expected=[np.mean(abs(e)),np.median(abs(e)),max(0,float(e.max())),max(0,float(-e.min()))]
                actual=[c[key][mode][k] for k in ['mean_abs_error_mps2','median_abs_error_mps2','max_underbraking_mps2','max_overbraking_mps2']]
                assert np.allclose(actual,expected,rtol=0,atol=1e-10)
    logs=list(dict.fromkeys(c['log'] for c in r['cases']))
    labels={'real':'Real','gt_clean':'GT','dvgt_metric':'DV','dvgt_lidar_scaled':'Sc',
            'reference_lidar':'Li','dvgt_class_prior':'Cp','dvgt_visible_extent':'Ex'}
    keys=['mean_abs_error_mps2','max_underbraking_mps2','max_overbraking_mps2']
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey='col')
    for i,log in enumerate(logs):
        cases=[c for c in r['cases'] if c['log']==log]; x=np.arange(len(cases))
        for j,key in enumerate(keys):
            ax=axes[i,j]
            for m,mode in enumerate(['old','zero']):
                ax.bar(x+(m-.5)*.36,[c['physical'][mode][key] for c in cases],width=.36,
                       label=['Original velocity prior','Zero world-velocity prior'][m],color=['#387b8f','#d9774d'][m])
            ax.set_xticks(x,[labels[c['arm']]+('' if c['seed'] is None else str(c['seed'])) for c in cases],rotation=45,ha='right')
            ax.set_title(log[:8]+' | '+['Mean absolute error','Maximum underbraking','Maximum overbraking'][j],fontsize=11)
            ax.set_ylabel('Action discrepancy (m/s²)'); ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            if j: ax.axhline(2,color='#996746',ls=':',lw=1)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.027,'DV: DVGT | Sc: global scale | Li: target LiDAR | Cp: class prior | Ex: visible extent | 42/43: seed',ha='center',fontsize=10)
    fig.text(.5,.006,'20 existing streams / 300 decisions. Fixed observations and ego. CPU replay only; no new closed-loop execution.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.945))
    for ext in ['png','svg','pdf']: fig.savefig(out/f'velocity-prior.{ext}',dpi=170)
    plt.close(fig)
    for ext in ['png','svg','pdf']: shutil.copy2(out/f'velocity-prior.{ext}',archive/f'velocity-prior.{ext}')
    table=''.join('<tr><td>'+c['log'][:8]+'</td><td>'+escape(c['arm'])+'</td><td>'+str(c['seed'])+'</td>'+''.join(
        f'<td>{c["physical"]["old"][k]:.3f} → {c["physical"]["zero"][k]:.3f}</td>' for k in keys)+'</tr>' for c in r['cases'])
    return table


def main():
    p=argparse.ArgumentParser();p.add_argument('--velocity',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    architecture(out);table=archive_velocity(a.velocity,a.archive,out)
    shutil.copy2(a.velocity/'result.json',out/'velocity-result.json')
    shutil.copy2(a.velocity/'protocol.json',out/'velocity-protocol.json')
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5｜WorldSim 做得好意味着什么</title><style>
body{max-width:1150px;margin:40px auto;padding:0 24px;background:#f4f7fa;color:#233e50;font:17px/1.85 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:36px;line-height:1.3}h2{font-size:24px;margin-top:36px}a{color:#08758c}img{width:100%;background:white;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:12px;border-bottom:1px solid #cbd8e0;vertical-align:top}.card{padding:22px;background:white;border:1px solid #cadce2;border-radius:12px;margin:22px 0}.tag{font-size:14px;color:#5e7788}.warn{border-left:5px solid #bc7950}.scroll{overflow:auto}summary{cursor:pointer}</style></head><body>
<p class="tag">V7.5 · 研究定义 · 2026-09-21 · 本页不是新反事实实验结果</p>
<h1>WorldSim 做得好：<br>干预后的世界与后果，是否可信？</h1>
<div class="card"><strong>在声明的场景、干预和策略范围内，执行指定干预，保留不受影响的世界事实，生成一致的观测，并忠实反映策略在该世界中的后果。</strong><p>输入无法确定的事实保留不确定性。我们研究哪些可传入的重建状态能改善这一点，不预设更复杂的重建必然有用。</p></div>
<img src="architecture.png" alt="观测、重建、编辑、生成、策略和动力学反馈；独立编辑后参考仅用于评价">
<p><a href="architecture.svg">矢量架构图</a> · <a href="architecture.pdf">PDF</a></p>
<h2>先分清两种实验</h2><div class="card"><p><strong>编辑世界：</strong>把车 A 移走，参考世界也要移走 A。原日志未来不再是唯一正确答案。</p><p><strong>估错世界：</strong>在同一编辑后参考世界内，比较自然重建、普通修复和额外信息参考。参考不能跟着错误估计一起移动。</p></div>
<h2>五项质量，各自给出证据</h2><div class="scroll"><table><tr><th>轴</th><th>具体问题</th><th>有效参照</th></tr>
<tr><td>重放保真</td><td>未编辑、同动作同相机，是否复现观测？</td><td>真实记录的可支持区域</td></tr>
<tr><td>编辑遵循</td><td>对象是否真的移除/移动，身份和尺寸是否保持？</td><td>明确的编辑命令＋生成观测</td></tr>
<tr><td>无关事实保留</td><td>未受影响的对象和地图是否保留？</td><td>共同可见事实；允许遮挡和合理交互改变</td></tr>
<tr><td>世界与可见性一致</td><td>已知对象在显露、新视角后是否仍一致？</td><td>独立观测/已知场景；未知隐藏内容不冒充真值</td></tr>
<tr><td>决策与后果保真</td><td>同一策略的事件、效用和相对优劣是否符合参考？</td><td>编辑后的可信参考；不奖励贴近未编辑日志</td></tr></table></div>
<div class="card warn"><strong>策略更安全 ≠ 仿真更准确。</strong><p>如果差策略在参考世界会撞车，仿真却让它通过，碰撞率下降反而可能是 false-safe。任务成功、碰撞和进度是策略表现；仿真质量要看这些结果相对参考有多准确。单一 IDM 不能证明策略排名一致性。</p></div>
<h2>三个必须保留的边界</h2><ol><li>其他车辆可以响应干预。应保留不受影响的事实，不要求所有像素或轨迹不变。</li><li>隐藏区域可能有多个合理答案。没有参考时，只评价能验证的约束并报告未知比例。</li><li>当前模型只接初帧、文本、框/地图和历史。没有进入输入的表面细节，不能解释成模型利用了该几何。</li></ol>
<h2>第一轮只做：有参考支持的对象移除</h2><p>先在已有两个开发任务检查一个合格目标：被遮挡内容有真实证据，移除后的初帧与条件一致，重建差异确实进入模型。任一不满足，记录数据或接口缺口，不靠扩大编辑强迫失败。</p><p>资格通过后才冻结最多六段：参考结构、合法输入估计、普通框先验，各有未编辑/移除配对；固定动作与相机、seed42、117帧。先检验编辑、显露与无关事实，再决定是否值得做最多三段实际反馈。GT/LiDAR 单列额外信息；没有合法初态与阈值前不启动该队列。</p>
<p>反事实范围按编辑类型、输入预算、参考覆盖和质量轴分别报告。离散测试点不代表整条连续曲线；遮挡切换也不必平滑或单调。</p>
<h2>已有证据重新归位</h2><p>已有两个任务证明真实生成反馈可运行，也保留了普通控制、负结果和一个可由普通类别先验改善的小效应。它们没有证明重建已经普遍成熟，也没有证明反事实仿真正确。较大形状候选额外 seed 未复现，不重开。</p>
<details><summary>收口：上一轮速度先验普通控制（全部20条已有流）</summary><p>唯一改变是新轨迹的世界速度均值：ego 速度 → 0。复用检测、相机与保存的 ego，非新闭环执行。300次评价选择的目标框不变，原实现700次访问精确复现。固定主项平均误差1.102→0.891，最大欠制动2.420→1.890，但额外制动1.114→2.142 m/s²，未通过；同场景两个GT seed也出现门控退化。未启动新生成，不扩展速度/噪声搜索。</p>
<img src="velocity-prior.png" alt="两日志全部20条流，原速度先验与零速度先验的动作误差比较"><p>图中三项均相对同一保存 ego 的物理参考 IDM；主项门控相对条件状态参考、排除共同初帧，两者不能混用。</p><div class="scroll"><table><tr><th>日志</th><th>分支</th><th>seed</th><th>平均绝对误差</th><th>最大欠制动</th><th>最大额外制动</th></tr>'''+table+'''</table></div><p><a href="velocity-result.json">完整结果</a> · <a href="velocity-protocol.json">事前协议</a> · <a href="velocity-prior.pdf">完整科学图</a></p></details>
<h2>一手依据</h2><p><a href="https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html">NuRec / Li Auto</a>分别评价传感器、感知与闭环一致性。<a href="https://research.nvidia.com/publication/2026-06_nvidia-omnidreams-real-time-generative-world-model-closed-loop-autonomous">OmniDreams</a>提供结构化状态与生成历史通路。<a href="https://arxiv.org/abs/2510.18135">World-in-World</a>说明视觉质量不足以评价任务价值；它不替我们提供反事实真值。</p>
<p><a href="../V75_Multiscene_Closed_Loop/index.html">返回完整历史实证与视频</a></p><p class="tag">human_verdict: null · 不新增科学失败卡 · 定义不等于主张已成立</p></body></html>'''
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps({'report':str(out/'index.html'),'velocity_streams':20,'verified_metrics':320,'new_model_calls':0},ensure_ascii=False))


if __name__=='__main__': main()
