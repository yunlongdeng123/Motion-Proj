"""由实际生成帧与完整轨迹构建状态输入比较图，不绘制想象的失效。"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLORS = ['#237a9b', '#d26b3f']
LABELS = ['Fixed class dimensions', 'Visible-extent fit']


def save(fig, path):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(path.with_suffix('.' + ext), dpi=190, facecolor='white')
    plt.close(fig)
    path = path.with_suffix('.svg')
    path.write_text('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8', newline='\n')


def architecture(out):
    fig, ax = plt.subplots(figsize=(14, 3.3))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    nodes = [(.09, .70, 'Same RGB +\nDVGT range'),
             (.31, .70, 'Two ordinary\n3D state exports'),
             (.55, .70, 'OmniDreams\ngenerated RGB'),
             (.78, .70, 'Detector / tracker\n+ fixed IDM'),
             (.78, .18, 'Ego dynamics\n+ next camera'),
             (.31, .18, 'Direct-state IDM\ncontrol (extra input)')]
    for x, y, label in nodes:
        ax.text(x, y, label, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=.7', facecolor='#eef4f7', edgecolor='#8099a6'))
    for a, b in [((.18,.70),(.21,.70)), ((.42,.70),(.46,.70)),
                 ((.65,.70),(.67,.70)), ((.78,.48),(.78,.37)),
                 ((.31,.48),(.31,.39))]:
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', lw=1.7, color='#355466'))
    ax.annotate('', xy=(.55,.48), xytext=(.68,.18),
                arrowprops=dict(arrowstyle='->', connectionstyle='angle,angleA=180,angleB=-90,rad=15', lw=1.7, color='#355466'))
    ax.text(.96,.44,'Actions',ha='right',fontsize=10,color='#526878')
    ax.text(.48,.19,'Feedback',ha='center',fontsize=10,color='#526878')
    save(fig, out/'components')


def case_figure(src, out, short):
    d = json.loads((src/short/'figure_inputs.json').read_text())
    arms = d['arms']; t = d['selected_time_s']
    summary = {r['arm']: r for r in d['summary']['rows']}
    r = [summary[a['arm']] for a in arms]
    gap = arms[1]['readout']['task_geometry']['unclipped_near_route_gap_m'] - arms[0]['readout']['task_geometry']['unclipped_near_route_gap_m']
    cpu_delta = r[1]['direct_control']['progress_m'] - r[0]['direct_control']['progress_m']
    rgb_delta = r[1]['progress_m'] - r[0]['progress_m']
    fig = plt.figure(figsize=(15, 10.3))
    fig.suptitle(f'{short}  |  Near-equal input gaps, different generated feedback',
                 x=.04, y=.985, ha='left', fontsize=19, fontweight='bold', color='#183e53')
    fig.text(.04, .941, f'Same initial RGB / DVGT range / seed 42   |   Input near-gap difference: {gap*1000:+.1f} mm   |   117 frames, 15 decisions',
             fontsize=11, color='#536978')
    grid = fig.add_gridspec(3, 6, left=.065, right=.98, bottom=.155, top=.89,
                           height_ratios=[1.15,.42,1.07], hspace=.30, wspace=.65)
    for i in range(3):
        ax = fig.add_subplot(grid[0, 2*i:2*i+2]); ax.axis('off')
        file = 'initial.png' if i == 0 else arms[i-1]['arm']+'.jpg'
        ax.imshow(Image.open(src/short/file))
        title = '(a) Recorded initial RGB' if i == 0 else f'({"bc"[i-1]}) {LABELS[i-1]}\nGenerated RGB at {t:.2f} s'
        ax.set_title(title, loc='left', fontsize=12, color='#183e53', pad=10)
    texts = ['Shared initial frame; route, terrain,\nother actors and relative future motion fixed.\nGreen boxes: actual policy detections.']
    for i, a in enumerate(arms):
        dims = a['readout']['dimensions_m']; obs = a['selected_observation']
        texts.append(f'Input L/W/H: {dims[0]:.2f} / {dims[1]:.2f} / {dims[2]:.2f} m\n'
                     f'Applied acceleration: {obs["policy"]["acceleration_mps2"]:+.2f} m/s²\n'
                     f'At-own-ego reference: {obs["oracle_acceleration_mps2"]:+.2f} m/s²')
    for i, txt in enumerate(texts):
        ax = fig.add_subplot(grid[1,2*i:2*i+2]); ax.axis('off')
        ax.text(0, .95, txt, transform=ax.transAxes, va='top', fontsize=11, linespacing=1.6,
                color='#526878' if i == 0 else COLORS[i-1])
    ax1 = fig.add_subplot(grid[2,:3]); ax2 = fig.add_subplot(grid[2,3:])
    for i, a in enumerate(arms):
        times = a['time_s']; error = np.array(a['actual_acceleration_mps2']) - np.array(a['reference_acceleration_mps2'])
        ax1.step(times, error, where='post', color=COLORS[i], label=LABELS[i], lw=2)
        ax1.scatter(times, error, color=COLORS[i], s=14)
        ax2.plot(a['dense_time_s'], a['dense_target_clearance_m'], color=COLORS[i], lw=2)
        ax2.annotate(f'{a["dense_target_clearance_m"][-1]:.2f} m', (a['dense_time_s'][-1],a['dense_target_clearance_m'][-1]),
                     xytext=(-5,12 if i==0 else -16), textcoords='offset points', ha='right', color=COLORS[i],fontsize=10)
    ax1.axhline(0,color='#718796',lw=.8)
    ax1.set(title='(d) All 15 applied-action errors', ylabel='Applied − at-own-ego reference (m/s²)', xlabel='Time (s)')
    ax2.set(title='(e) All 117 reference clearances', ylabel='Ego-to-fixed-target footprint distance (m)', xlabel='Time (s)')
    for ax in [ax1,ax2]:
        ax.axvline(t,color='#718796',ls=':',lw=1); ax.grid(alpha=.17); ax.set_xlim(0,3.867)
    ax1.text(.015,.98,'Positive: less braking than reference',transform=ax1.transAxes,va='top',fontsize=9,color='#596c78')
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.055,.054), ncol=2, frameon=False, fontsize=11)
    fig.text(.51,.076, f'Paired executed progress: RGB {rgb_delta:+.3f} m\nDirect-state control {cpu_delta:+.6f} m', fontsize=11, color='#183e53', linespacing=1.5)
    fig.text(.065,.034,'Both runs: no fixed-reference overlap. Single seed; exposed development log. Shape and center change together.',fontsize=9,color='#536978')
    fig.text(.065,.014,'Snapshot chosen by largest paired action difference after shared startup. Ego paths already differ; images are not matched-pose renders.',fontsize=9,color='#536978')
    save(fig, out/f'{short}-state-feedback')


def build(src, audit, output):
    out = output/'shape-feedback'; out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    data = json.loads((src/'comparison.json').read_text()); assert data['status']=='complete'
    shutil.copy2(src/'comparison.json',out/'comparison.json')
    shutil.copy2(audit/'result.json',out/'audit-result.json')
    architecture(out)
    sections=[];rows=[]
    names={'gt_clean':'GT条件','oracle_shape_dvgt':'DVGT距离＋GT形状','dvgt_class_prior':'固定类别尺寸','dvgt_visible_extent':'可见范围拟合'}
    for case in data['cases']:
        short=case['log_id'][:8];case_figure(src,out,short)
        dest=out/short;dest.mkdir(exist_ok=True)
        for f in (src/short).iterdir():
            if f.is_file():shutil.copy2(f,dest/f.name)
        for tail in ['shape-pair.mp4','observations.jpg']:
            shutil.copy2(src/f'{short}-{tail}',out/f'{short}-{tail}')
        for r in case['rows']:
            rows.append(f'<tr><td>{short}</td><td>{names[r["arm"]]}</td><td>{r["progress_change_vs_gt_m"]:+.3f}</td><td>{r["mean_abs_action_error_at_own_ego_mps2"]:.3f}</td><td>{r["max_underbraking_mps2"]:.3f}</td><td>{r["max_overbraking_mps2"]:.3f}</td><td>{r["final_target_reference_clearance_m"]:.3f}</td></tr>')
        sections.append(f'''<h3>{short}：实际状态输入、生成画面和制动反馈</h3>
        <img src="shape-feedback/{short}-state-feedback.png" alt="真实初帧、两组生成画面与完整动作和间距曲线">
        <video controls preload="metadata" src="shape-feedback/{short}-shape-pair.mp4"></video>
        <p><a href="shape-feedback/{short}-state-feedback.pdf">PDF图</a> · <a href="shape-feedback/{short}-state-feedback.svg">SVG图</a> · <a href="shape-feedback/{short}/figure_inputs.json">逐帧绘图数据与选帧依据</a></p>
        <details><summary>固定四个时刻的实际策略输入</summary><img src="shape-feedback/{short}-observations.jpg"></details>''')
    return '''<section id="shape-feedback"><h2>新证据：相近的近端距离，不保证相同的生成闭环</h2>
    <p>旧实验保留了目标GT尺寸和朝向。本轮复用同一DVGT距离和真实检测框，比较“固定类别尺寸”与“可见范围拟合”两种普通状态构建；移除目标GT形状先验，固定其余输入。两任务各两组、seed42、117帧/15次决策，共468帧，无OOM。</p>
    <img src="shape-feedback/components.svg" alt="重建状态经生成RGB和策略形成真实动作反馈，旁路直接状态控制">
    <div class="card"><strong>02678d04：输入近端间距只差3.94毫米，实际行进相差2.757米。</strong>
    <p>两种状态经直接IDM控制的行进仅差−0.000150米；经实际生成RGB反馈后，可见范围拟合比固定类别尺寸多行进2.757米，最大欠制动偏差3.612 vs 1.906 m/s²。24642607的相应行进差为0.301米，保留这一响应较小的例子。两组都无固定参考碰撞，不能升级为false-safe或事故结论。</p>
    <p>普通类别尺寸在两例都降低平均动作误差，但第一例的最大额外制动从0.529增至1.244 m/s²；它是部分缓解，尚非一致恢复。原GT形状分支保留为额外信息参照，不能与普通读出混成同预算排名。</p></div>'''+''.join(sections)+'''
    <h3>完整正反指标</h3><table><tr><th>日志</th><th>条件</th><th>相对GT行进变化 m</th><th>平均自身ego参考动作误差 m/s²</th><th>最大欠制动 m/s²</th><th>最大额外制动 m/s²</th><th>末端参考间距 m</th></tr>'''+''.join(rows)+'''</table>
    <p>“自身ego参考”是在每条实际执行轨迹上，用固定物理参考状态计算的同一IDM动作；未把两条已经分叉的ego轨迹当成同一姿态。选帧规则为启动之后两组动作差最大的一步，两例规则相同，并同时展示完整曲线。绿色框是策略检测，未作生成相机已被验证的假设。</p>
    <h3>这一比较支持到哪里</h3><p>支持一个开发候选：普通近端几何控制近似等价的状态，经生成观测与策略反馈后仍可能产生不同制动。尺寸与中心同时变化，尚不能将差异独立归因宽度或高度；同样不能将整个生成／检测／跟踪链条的差异全归因生成器。新形状是外部普通适配算法输出，DVGT没有原生对象形状头，此处不是SOTA端到端重建排名。</p>
    <p>类别L/W/H=3.9/1.6/1.56米来自<a href="https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/kitti_models/pointpillar.yaml">OpenPCDet官方Car先验</a>，未运行PointPillars；朝向采用ego方向。可见拟合使用前向表面、检测框边界和固定长度，没有目标GT形状。二维边界拟合准确不等于真实三维形状准确。两例的核心点都只击中一个后表面，未观测长度仍依赖先验。</p>
    <p>同一CPU审计保留五组普通方法：框＋类别先验、DVGT＋类别先验、DVGT可见拟合、背景LiDAR尺度拟合、目标LiDAR拟合。后两项使用额外度量信息；目标掩码也依赖GT。1170帧CPU控制、0次新重建调用；全部结果见下方。共享未来相对轨迹、地形和其他演员仍为提供的额外信息。</p>
    <p>实际生成已结束，4个新run与4个旧参照共936帧按原动作精确回放，相机矩阵最大差0。首个工程预检因旧条件目录引用错误而停止，尚未调用模型；修复为读取原协议的真实目录后另起run，失败记录保留。人工verdict为空；没有新增科学failure卡。</p>
    <p><a href="shape-feedback/comparison.json">完整反馈结果</a> · <a href="shape-feedback/audit-result.json">五组普通控制及读出审计</a> · <a href="shape-feedback/components.pdf">组件图PDF</a></p></section>'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(a.output/'shape-section.html').write_text(build(a.source,a.audit,a.output),encoding='utf-8')
