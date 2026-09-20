"""从已完成直接状态控制生成科学对比图和审阅页；不生成或改画场景。"""
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

NAMES = ['DVGT', 'Global LiDAR scale', 'Target LiDAR anchor']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--recorded', type=Path, required=True); p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    result = json.loads((a.evidence/'result.json').read_text()); assert result['status'] == 'complete'
    for name in ['result.json','protocol.json','replay_verification.json','offline_action_comparison.json']:
        shutil.copy2(a.evidence/name, out/name)
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    rows = result['cases'][1:]; x = np.arange(3)
    fig, axs = plt.subplots(1,3,figsize=(15.3,4.3), gridspec_kw={'width_ratios':[.85,1.2,1.2]})
    colors = ['#386986','#d96c4b']; labels = ['Direct state + IDM','Generated RGB + IDM']
    axs[0].bar(x,[r['initial_geometry_residual_m'] for r in rows],color=['#8baab5','#8baab5','#8baab5'],width=.6)
    for i,r in enumerate(rows): axs[0].text(i,r['initial_geometry_residual_m']+.25,f"{r['initial_geometry_residual_m']:.2f}",ha='center')
    axs[0].set(title='(a) Frozen input geometry',ylabel='Target center residual (m)',ylim=(0,11))
    for j,key in enumerate(['direct_progress_change_vs_gt_m','rgb_progress_change_vs_gt_m']):
        bars = axs[1].bar(x+(j-.5)*.34,[r[key] for r in rows],width=.34,color=colors[j],label=labels[j])
        for b,r in zip(bars,rows):
            v=r[key]; axs[1].text(b.get_x()+b.get_width()/2,v+(.13 if v>=0 else -.15),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
    axs[1].axhline(0,color='#66727e',lw=.8); axs[1].set(title='(b) Executed progress',ylabel='Change from each interface\'s GT arm (m)',ylim=(-5.6,2.05))
    for j,key in enumerate(['direct_mean_abs_action_change_vs_gt_mps2','rgb_mean_abs_action_change_vs_gt_mps2']):
        bars=axs[2].bar(x+(j-.5)*.34,[r[key] for r in rows],width=.34,color=colors[j])
        for b,r in zip(bars,rows): axs[2].text(b.get_x()+b.get_width()/2,r[key]+.02,f'{r[key]:.3f}',ha='center',fontsize=9)
    axs[2].set(title='(c) Applied action response',ylabel='Mean absolute change from GT (m/s²)',ylim=(0,1.06))
    for ax in axs:
        ax.set_xticks(x,['DVGT','Global scale','Target LiDAR']); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.legend(*axs[1].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.57,1.0),frameon=False)
    fig.text(.5,.015,'One development scene • 3.87 s • shared first action • 0 overlaps in all arms • state control has extra information',ha='center',fontsize=10,color='#556677')
    fig.tight_layout(rect=(0,.06,1,.9)); fig.savefig(out/'interface-control.png',dpi=180); fig.savefig(out/'interface-control.svg'); plt.close(fig)

    offline=json.loads((a.evidence/'offline_action_comparison.json').read_text())['rows']
    row=next(r for r in offline if r['arm']=='reference_lidar' and r['frame']==84)
    recorded_path=a.recorded/'reference_lidar.json'
    if not recorded_path.exists(): recorded_path=a.recorded/'reference_lidar/decisions.json'
    recorded=json.loads(recorded_path.read_text())
    decision=next(r for r in recorded if r['observation_frame']==84)
    policy=decision['policy']; image=np.array(Image.open(a.image)); box=policy['lead']['box']
    fig=plt.figure(figsize=(14,7)); gs=fig.add_gridspec(2,2,width_ratios=[1.35,1],height_ratios=[1,.85])
    ax=fig.add_subplot(gs[:,0]); ax.imshow(image); ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='#17d3aa',lw=2))
    ax.set_axis_off(); ax.set_title('(a) Actual generated policy input at 2.80 s\nGreen: detector box selected by the driving policy',loc='left',fontsize=12)
    text_ax=fig.add_subplot(gs[0,1]); text_ax.axis('off')
    text_ax.text(0,.96,'Same ego state, different state sources',weight='bold',fontsize=13,va='top')
    text_ax.text(0,.76,f"Input geometry residual: {result['cases'][3]['initial_geometry_residual_m']:.3f} m\n\nCondition leader: {row['condition_lead']['gap_m']:.2f} m gap, {row['condition_lead']['lead_speed_mps']:.2f} m/s\nRGB policy readout: {policy['lead']['gap_m']:.2f} m gap, {policy['lead']['lead_speed_mps']:.2f} m/s\n\nActual action: {row['actual_rgb_acceleration_mps2']:+.2f} m/s²\nCondition-state IDM: {row['condition_acceleration_at_own_ego_mps2']:+.2f} m/s²",fontsize=12,va='top',linespacing=1.4)
    ax=fig.add_subplot(gs[1,1]); t=np.array([r['observation_frame']/30 for r in recorded]); acc=[r['policy']['acceleration_mps2'] for r in recorded]
    other=[r['condition_acceleration_at_own_ego_mps2'] for r in offline if r['arm']=='reference_lidar']
    ax.plot(t,acc,'o-',color=colors[1],label='Actual RGB action',ms=4); ax.plot(t,other,'s--',color=colors[0],label='Condition at same ego',ms=4)
    ax.axvline(2.8,lw=1,color='#87949b'); ax.axhline(0,lw=.7,color='#87949b'); ax.set(xlabel='Observation time (s)',ylabel='Acceleration (m/s²)')
    ax.legend(fontsize=9,frameon=False); ax.grid(alpha=.15)
    fig.text(.5,.02,'Readout speeds are estimator outputs, not verified object motion in generated pixels. Selected-leader identities are not quantitatively matched.',ha='center',fontsize=9,color='#556677')
    fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(out/'state-action-example.png',dpi=180); fig.savefig(out/'state-action-example.svg'); plt.close(fig)
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 278" role="img" aria-label="同一状态的生成RGB闭环与直接状态控制">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#567085"/></marker></defs>
    <g fill="#fff" stroke="#93abb8"><rect x="10" y="81" width="160" height="70" rx="7"/><rect x="225" y="20" width="160" height="60" rx="7"/><rect x="440" y="20" width="160" height="60" rx="7"/><rect x="225" y="176" width="375" height="60" rx="7"/><rect x="656" y="20" width="160" height="60" rx="7"/><rect x="656" y="176" width="160" height="60" rx="7"/><rect x="930" y="20" width="180" height="60" rx="7"/><rect x="930" y="176" width="180" height="60" rx="7"/></g>
    <g fill="#24435b" text-anchor="middle" font-family="sans-serif" font-size="17"><text x="90" y="110">四份冻结状态</text><text x="90" y="136" font-size="13">GT / DVGT / 两项控制</text><text x="305" y="56">生成 RGB</text><text x="520" y="56">检测与跟踪</text><text x="412" y="212">直接读取当时状态（额外信息）</text><text x="736" y="56">普通 IDM</text><text x="736" y="212">相同 IDM</text><text x="1020" y="56">动力学 → ego</text><text x="1020" y="212">相同动力学 → ego</text></g>
    <g fill="none" stroke="#567085" stroke-width="2" marker-end="url(#arrow)"><path d="M170 100H195V50H220"/><path d="M170 131H195V206H220"/><path d="M387 50H435"/><path d="M602 50H651"/><path d="M818 50H925"/><path d="M602 206H651"/><path d="M818 206H925"/><path d="M1020 82V115H305V84"/><path d="M1020 238V265H410V240"/></g><text x="705" y="143" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#567085">分别反馈各自ego；同一初始动作、时间表、固定参考世界</text></svg>'''
    (out/'architecture.svg').write_text(architecture,encoding='utf-8')
    names={'gt_clean':'GT条件','dvgt_metric':'DVGT','dvgt_lidar_scaled':'全局尺度','reference_lidar':'目标LiDAR'}
    table=''.join(f'<tr><td>{names[r["arm"]]}</td><td>{r["initial_geometry_residual_m"]:.3f}</td><td>{r["direct_progress_change_vs_gt_m"]:+.3f}</td><td>{r["rgb_progress_change_vs_gt_m"]:+.3f}</td><td>{r["direct_mean_abs_action_change_vs_gt_mps2"]:.3f}</td><td>{r["rgb_mean_abs_action_change_vs_gt_mps2"]:.3f}</td></tr>' for r in result['cases'])
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 普通状态控制与生成闭环</title><style>body{max-width:1120px;margin:36px auto;padding:0 24px;background:#f4f7fa;color:#203c50;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:35px;line-height:1.35}h2{margin-top:36px;font-size:24px}img{width:100%;background:white;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px;border-bottom:1px solid #cad9e2;text-align:left}.card{padding:24px;background:white;border:1px solid #ccdce5;border-radius:12px;margin:24px 0}.tag{font-size:14px;color:#608093}a{color:#087a9b}</style>
    <p class="tag">V7.5 · CPU强控制 · 2026-09-20 · 0次新增GPU调用</p><h1>几何误差能解释多少制动差异？<br>先让普通控制直接读状态。</h1>
    <p>复用同一开发场景的四份冻结状态，各117帧、15次决策。新增四条直接状态反馈轨迹；生成RGB结果来自上一轮，未重跑生成、检测或改变时长。</p><img src="architecture.svg" alt="两条反馈支路与共享条件">
    <div class="card"><strong>大尺度组的大部分进度变化，普通几何与控制已能产生；准确目标位置仍未保证RGB闭环恢复。</strong><p>全局尺度的行进变化在直接状态控制中为−4.52米，生成RGB闭环为−4.67米。目标LiDAR组的位置残余仅0.25米，直接状态反馈变化+0.05米，RGB反馈却为+1.30米。后者来自两种信息接口的差异，不能把1.25米的差值全部叫作世界模型误差。</p></div>
    <h2>比较的是各自相对于GT条件的变化</h2><table><tr><th>条件</th><th>位置残余 m</th><th>状态控制进度变化 m</th><th>RGB进度变化 m</th><th>状态控制动作差 m/s²</th><th>RGB动作差 m/s²</th></tr>'''+table+'''</table>
    <p>动作差取15次决策的平均绝对差；进度变化为有符号变化。直接状态GT行进32.975米，RGB的GT行进33.591米，两者基线并不相同。四组均无固定参考重叠；更多制动、较少进度不能直接称为更安全。</p><img src="interface-control.png" alt="输入几何残余、进度变化和动作变化的双接口对比">
    <h2>任务状态反例：位置修正后，策略仍可能错误估计运动</h2><p>目标LiDAR分支2.80秒的条件前车速度为0，RGB策略输出约9.38 m/s；实际动作+0.19 m/s²，同一ego下直接读取条件会给出−2.15 m/s²。图中是原始生成输入和当时真实使用的检测框。速度是跟踪器估计，不是已验证的生成物体运动；检测身份没有真值级匹配，不能单独归因生成器。</p><img src="state-action-example.png" alt="实际生成帧及状态读出和动作差异">
    <h2>接口与验证</h2><p>保留所有分支相同的第一段真实RGB动作，从第4帧后的决策开始查询当前条件状态；速度只用当前与之前条件位置，不读取未来状态。复用同一IDM、40米范围、路线控制、执行器映射、官方动力学和高程贴合。</p><p>先回放四组已保存动作：468帧相机矩阵和全部决策边界状态与原结果完全一致；60次动作映射也完全一致。新增四组反馈468帧，全程禁用CUDA，执行约8.85秒。</p>
    <h2>这轮改变了什么研究决策</h2><p>不能再用本例的大尺度减速作为生成模型独有问题；不能仅按三维中心误差给“修复”排名。下一项强控制只检查普通时序关联：在保存的真实与生成输入上测试是否能合法恢复状态/动作，先排除策略自身的身份切换；它若不能通过真实基线或不能恢复，就关闭该分支，不调阈值或追加本例seed/时长。离线回放不会冒称新的完整闭环改善。</p>
    <p class="tag">一个事后开发场景；额外三维状态、已知路线/地形、GT尺寸朝向与共享非反应式轨迹。没有新训练、模型推理、跨场景确认或科学失败卡，人工verdict为null。</p><p><a href="result.json">结果</a> · <a href="protocol.json">冻结协议</a> · <a href="replay_verification.json">精确回放验证</a> · <a href="interface-control.svg">主图SVG</a> · <a href="../V75_Approach_Closed_Loop/index.html">上一轮四组实际视频</a></p></html>'''
    (out/'index.html').write_text(html,encoding='utf-8',newline='\n')
    # Windows绘图库按系统换行输出SVG；归档使用统一LF。
    for path in out.glob('*.svg'):
        content=path.read_text(encoding='utf-8')
        path.write_text('\n'.join(line.rstrip() for line in content.splitlines())+'\n',encoding='utf-8',newline='\n')
    print(out/'index.html')


if __name__ == '__main__': main()
