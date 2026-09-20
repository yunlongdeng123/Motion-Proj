"""根据真实输入和实际生成结果绘图，三维准入失败时明确留空。"""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES={'gt_clean':'GT condition','dvgt_metric':'DVGT + known rays','ordinary_bbox':'Ordinary box fit','reference_lidar':'Extra target LiDAR'}
COLORS={'gt_clean':'#657284','dvgt_metric':'#c9433a','ordinary_bbox':'#735fb0','reference_lidar':'#168575'}

def value(x):return '未通过' if x is None else f'{x:.2f}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();e=a.evidence;out=a.output;out.mkdir(parents=True,exist_ok=True)
    reads=json.loads((e/'readout_queue_result.json').read_text())
    result=json.loads((e/'assessment_result.json').read_text());assert result['status']=='complete' and len(result['cases'])==1
    case=result['cases'][0];real=case['real_metric_reference'];geometry=np.load(e/'common_geometry.npz')
    for name in ['comparison.mp4','target-comparison.jpg','tracking-comparison.jpg','real-tracking-silver.jpg','real-tracking-black.jpg',
                 'candidates-24642607.jpg','candidates-29a00842.jpg','candidates-2c652f9e.jpg',
                 'assessment_result.json','readout_queue_result.json','observation_selection.json','protocol.json','visual_review.json','real_triangulation_summary.json']:
        shutil.copy2(e/name,out/name)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in NAMES:
        times=[r['frame']/30 for r in case['frames']]
        axes[0].plot(times,[r['arms'][name]['to_real_px'] if r['admitted'] else np.nan for r in case['frames']],marker='o',color=COLORS[name],label=NAMES[name])
        if name!='gt_clean':
            axes[1].plot(times,[r['arms'][name]['projection_shift_px'][0] for r in case['frames']],linestyle='--',color=COLORS[name])
            axes[1].plot(times,[r['arms'][name]['paired_clean_vector_px'][0] if r['admitted'] else np.nan for r in case['frames']],marker='o',color=COLORS[name],label=NAMES[name])
    axes[0].set_title('Tracked points: generated vs real',loc='left',fontsize=11)
    axes[1].set_title('Horizontal response: solid=points, dashed=projection',loc='left',fontsize=10)
    for ax in axes:
        ax.set_xlabel('Time (s)');ax.set_ylabel('Pixels');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7)
    fig.savefig(out/'point-response.png',dpi=180);fig.savefig(out/'point-response.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    keep=geometry['common'];camera=geometry['cameras'][0]
    if case['metric_comparison_admitted']:
        for name in ['real_actor',*NAMES]:
            xyz=(geometry[name][keep]-camera[:3,3])@camera[:3,:3]
            axes[0].scatter(xyz[:,0],xyz[:,2],s=23,label='Real video triangulation' if name=='real_actor' else NAMES[name],color='#162b40' if name=='real_actor' else COLORS[name],alpha=.8)
        axes[0].set_xlabel('Initial camera right (m)');axes[0].set_ylabel('Initial camera forward (m)');axes[0].legend(fontsize=7)
        axes[0].set_title('Common fitted surface points',loc='left',fontsize=11)
    else:
        axes[0].text(.5,.5,'Metric comparison not admitted\nNo depth ranking reported',ha='center',va='center',transform=axes[0].transAxes)
        axes[0].set_title('Common fitted surface points',loc='left',fontsize=11)
        axes[0].set_axis_off()
    axes[1].bar(np.arange(4),[case['metric_point_results'][name]['rigid_fit_accepted_points'] for name in NAMES],color=[COLORS[name] for name in NAMES])
    axes[1].set_xticks(np.arange(4),['GT condition','DVGT','Box fit','Extra LiDAR'],rotation=15)
    axes[1].axhline(case['initial_points'],color='#657284',linestyle='--');axes[1].set_ylim(0,case['initial_points']+2)
    axes[1].set_ylabel('Points passing fixed five-view rigid fit');axes[1].set_title('All original points retained in denominator',loc='left',fontsize=10)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'rigid-points.png',dpi=180);fig.savefig(out/'rigid-points.svg');plt.close(fig)
    for svg in out.glob('*.svg'):svg.write_bytes(('\n'.join(r.rstrip() for r in svg.read_text().splitlines())+'\n').encode())
    readrows=''.join('<tr><td>'+r['log_id'][:8]+'</td>'+''.join(f'<td>{r["readouts"][k]["center_error_m"]:.3f}</td>' for k in ['dvgt_metric','dvgt_lidar_scaled','ordinary_bbox','reference_lidar'])+f'<td>{"通过" if r["admitted"] else "参考读出未通过"}</td></tr>' for r in reads['cases'])
    rows=''
    for name in NAMES:
        metric=case['metric_point_results'][name]
        rows+=f'<tr><td>{NAMES[name]}</td><td>{value(case["detector_means_px"][name])}</td><td>{value(case["flow_means_px"][name])}</td><td>{metric["rigid_fit_accepted_points"]}/{case["initial_points"]}</td><td>{value(metric["median_point_distance_to_real_m"])}</td></tr>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V7.5 可见性前置与生成几何</title>
<style>body{{margin:0;background:#edf1f5;color:#172a3e;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1300px;margin:auto;padding:32px 24px}}section{{background:white;padding:26px;margin:22px 0;border-radius:10px}}h1{{font-size:30px}}h2{{font-size:22px}}img,video{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde5eb}}.note{{padding:15px;background:#fff2d3;border-left:4px solid #bd9026}}.flow{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}.box{{background:#193b55;color:white;padding:12px;border-radius:7px;flex:1;min-width:120px}}small{{color:#536578}}</style>
<main><h1>可见性前置：从位置读出到生成几何</h1><p>V7.5 · 新开发窗口 · 4日志 → 8候选 → 2个真实可测目标 → 1个参考合格目标 → 4组实际生成</p>
<p class="note">本轮未形成一致退化/修复的自然badcase：LiDAR修复改善检测中心，但共同点指标没有同步改善。定位诊断在此收口；后续研究转向哪些状态误差会改变实际闭环决策，不追加长时间放大实验。</p>
<section><h2>输入、条件与观测通路</h2><div class="flow"><div class="box">起点前七视图 RGB</div>→<div class="box">DVGT / 普通控制<br>初始actor位置</div>→<div class="box">官方cuboid条件<br>共享GT未来运动</div>→<div class="box">OmniDreams<br>同初帧 / seed42</div>→<div class="box">生成RGB → RAFT<br>已知相机三角测量</div></div>
<p>真实未来RGB先验证点追踪与几何；额外目标LiDAR用于参考检查，未输入DVGT。尺寸/朝向、相机标定、地图与未来ego/actor轨迹是额外信息。无策略反馈。</p></section>
<section><h2>先在真实数据中选可见、可测目标</h2><p>每日志最多检查6个按初始投影面积排序的几何候选，再按固定顺序选首个可见且可测目标。四日志及排序在查看模型结果前冻结；已关闭的旧来源窗口没有补位。真实银色轿车23点、黑色车辆20点在五个时刻全保留。</p><img src="real-tracking-silver.jpg"><img src="real-tracking-black.jpg">
<details><summary>所有有候选的日志，包括遮挡排除</summary><img src="candidates-24642607.jpg"><img src="candidates-29a00842.jpg"><img src="candidates-2c652f9e.jpg"></details></section>
<section><h2>模型读出与普通尺度控制</h2><table><tr><th>日志</th><th>DVGT中心误差(m)</th><th>全局LiDAR尺度(m)</th><th>普通框拟合(m)</th><th>目标LiDAR读出(m)</th><th>生成准入</th></tr>{readrows}</table>
<p>黑色车辆的参考读出0.663m超过原定0.5m，因此未运行生成，也没有替换目标。银色轿车全局尺度确实改善；本轮冻结四组生成没有包含该全局尺度组，不能声称普通尺度无法解释生成差距。</p></section>
<section><h2>真实多视图测量先过关</h2><p>目标真实平移仅{real['actor_translation_over_window_m']:.3f}m，相机相对目标基线{real['camera_baseline_in_actor_m']:.2f}m。真实{real['accepted_count']}/{real['initial_count']}点通过五视图刚体拟合；{real['lidar_associations']}点与3px内邻近LiDAR投影关联，深度差中位数{real['lidar_neighbor_median_depth_error_m']:.3f}m。</p>
<p>固定规则：五帧均跟踪、正深度、视差角至少2°、重投影中位数≤2px且最大≤4px；真实参考还需至少8点、25%支持及6个LiDAR邻近关联。邻近回波不是精确特征点真值。该追加观察器在生成已启动、但读取生成影像/量化前冻结，不伪称整个研究预注册。</p></section>
<section><h2>实际生成与共同点结果</h2><video src="comparison.mp4" controls preload="metadata"></video><small>真实 / GT条件 / DVGT / 额外目标LiDAR；普通框拟合保留在下表与五列图。</small>
<table><tr><th>输入条件</th><th>检测中心均值(px)</th><th>共同点二维均值(px)</th><th>通过刚体拟合的点</th><th>共同点三维距离中位数(m)</th></tr>{rows}</table>
<p>二维全窗口可测：{case['flow_full_window_admitted']}；三维共同合格点：{case['rigid_geometry_common_points']}/{case['initial_points']}；三维比较准入：{case['metric_comparison_admitted']}。缺失或不合格时不补零、不改窗口。三维列比较同一初始点身份与真实视频三角测量结果，未评估生成车辆框中心。</p>
<img src="point-response.svg"><img src="rigid-points.svg"><img src="target-comparison.jpg"><details><summary>全部点轨迹与支持</summary><img src="tracking-comparison.jpg"></details></section>
<section><h2>证据与边界</h2><p>DVGT的3.423m中心残余没有带来统一的可见退化。额外LiDAR把检测中心均值从3.78降至1.74px，共同点指标则从2.82变为3.37px，不能据此声称统一修复收益。普通框拟合只有4/5次检测、0/23点通过联合刚体条件；四组共同合格点为0，所有三维排名留空。</p>
<p>三角测量使用请求/条件中的相机轨迹，尚未独立验证生成视频遵循该轨迹。因此这里只是条件性的刚体一致性检查，不能当作实际生成车辆的米制状态。点身份错误、外观变化、视差不足或刚体假设不满足均可能导致拟合失败，不能直接算作物理事故。一个实际生成案例、一个seed，不是独立跨场景确认；主方法、普遍性和闭环危害尚未成立，人工verdict为null。</p>
<p><a href="protocol.json">冻结来源协议</a> · <a href="observation_selection.json">重建前锁定记录</a> · <a href="readout_queue_result.json">完整读出</a> · <a href="assessment_result.json">生成观测结果</a> · <a href="real_triangulation_summary.json">真实几何参考检查</a></p></section></main></html>'''
    (out/'index.html').write_text(body,encoding='utf-8',newline='\n');print(out/'index.html')

if __name__=='__main__':main()
