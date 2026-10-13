"""Only build final hub from real complete six-run evidence and reviewed saved reports."""
from pathlib import Path
import html,json,math,zipfile
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs'
four=json.loads((O/'drivingeditor_four_arm_30k/manifest.json').read_text(encoding='utf-8-sig'))
gmc=json.loads((O/'drivingeditor_GMC_GN11_30k/summary.json').read_text(encoding='utf-8-sig'))
red=json.loads((O/'drivingeditor_red_GN11_30k/summary.json').read_text(encoding='utf-8-sig'))
redreview=json.loads((O/'drivingeditor_red_GN11_30k/engineering-visual-review.json').read_text(encoding='utf-8-sig'))
sam=json.loads((W/'work/drivingeditor_resume/heartbeat_20261013_1056/red_SAM_actual_status.json').read_text(encoding='utf-8-sig'))
assert sam['returncode']==0 and sam['nvidia_sampling_errors']==0
assert redreview['review_complete'] is True and redreview['human_verdict'] is None
assert red['training_steps']==gmc['training_steps']==30000 and red['retained_frame_count']==gmc['retained_frame_count']==49
rows=[]
for name in four['arm_order']:
    arm=four['arms'][name];s,m=arm['status'],arm['memory'];assert s['status']=='complete' and m['run_status']=='complete' and m['monitor_status']=='closed'
    assert m['error_count']==m['sampling']['samples_with_errors']==0
    q=four['comparison']['arms'][name]['equal_frame_means']
    rows.append({'scene':'scene17','arm':name,'full_psnr_db':q['full_composite_psnr_db'],'actor_psnr_db':q['actor_composite_psnr_db'],'actor_valid_frames':49,'static_psnr_db':q['static_bg_psnr_db'],
        'wall_minutes':s['wall_seconds']/60,'allocated_GiB':m['pytorch']['peak_allocated_bytes']/2**30,'reserved_GiB':m['pytorch']['peak_reserved_bytes']/2**30,
        'PID_sampled_GiB':m['nvidia_smi']['process_peak_used_bytes']/2**30,'device_sampled_GiB':m['nvidia_smi']['device_peak_used_bytes']/2**30,
        'sampling_errors':0,'source':arm['source'],'static_points':m['extra']['render_static_points'],'dynamic_points':m['extra']['render_dynamic_points']})
for label,s in [('GMC',gmc),('红车',red)]:
    rows.append({'scene':label,'arm':'GN11','full_psnr_db':s['full_composite_psnr_db'],'actor_psnr_db':s['actor_composite_psnr_db'],'actor_valid_frames':s['valid_actor_frame_count'],
        'static_psnr_db':s['background_static_psnr_db'],'wall_minutes':s['wall_minutes'],'allocated_GiB':s['allocated_peak_GiB'],'reserved_GiB':s['reserved_peak_GiB'],
        'PID_sampled_GiB':s['process_peak_GiB'],'device_sampled_GiB':s['device_peak_GiB'],'sampling_errors':0,'static_points':s['static_points'],'dynamic_points':s['dynamic_points']})
maximum=max(max(r[k] for k in ['allocated_GiB','reserved_GiB','PID_sampled_GiB','device_sampled_GiB']) for r in rows)
peak_with_preprocessing=max(maximum,10.279296875)
assert all(math.isfinite(r['full_psnr_db']) for r in rows)
summary={'status':'SIX_REAL_RUNS_COMPLETE_REPORTS_REVIEWED','total_training_steps':180000,'runs':rows,'training_peak_max_GiB':maximum,'including_preprocessing_observed_max_GiB':peak_with_preprocessing,
    '3090_capacity_screen_budget_GiB':22,'3090_initial_screen_passed':peak_with_preprocessing<22,'target_3090_measured':False,'certified_3090_fit':False,'certified_3090_speedup':False,
    'checkpoint_bundles_checked':18,'components_per_bundle':4,'restoration_verified':False,'all_observed_training_views':True,'independent_test':False,'human_GN11_verdict':None,'human_C00_verdict':'用户确认符合预期',
    'red_body_valid_frames':red['valid_actor_frame_count'],'red_empty_body_frames':red['empty_actor_frame_count'],
    'red_SAM_resources':{k:sam[k] for k in ['torch_allocated_peak_bytes','torch_reserved_peak_bytes','nvidia_process_peak_bytes','nvidia_device_peak_bytes','nvidia_sample_count','nvidia_sampling_errors','nvidia_peaks_are_sampled_lower_bounds']},
    'power_status':'Training/report complete; separate power receipt required'}
(O/'DrivingEditor_final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained');fig.patch.set_facecolor('#0b1220')
for ax in axes:
    ax.set_facecolor('#101d30');ax.tick_params(colors='#becddd');ax.spines[['top','right']].set_visible(False)
    for s in ax.spines.values():s.set_color('#34445d')
    ax.title.set_color('#e8f1ff');ax.yaxis.label.set_color('#becddd');ax.grid(axis='y',alpha=.14,color='#8ba8ca')
labels=[r['scene']+' '+r['arm'] for r in rows];x=np.arange(len(rows))
axes[0].bar(x-.23,[r['allocated_GiB'] for r in rows],width=.23,label='PyTorch allocated',color='#7aaff8')
axes[0].bar(x,[r['reserved_GiB'] for r in rows],width=.23,label='PyTorch reserved',color='#bc92ed')
axes[0].bar(x+.23,[r['PID_sampled_GiB'] for r in rows],width=.23,label='PID 采样下界',color='#43c8b0')
axes[0].axhline(22,color='#f2b75b',ls='--',lw=1.2,label='3090 初筛 22 GiB')
axes[0].set_ylim(0,24);axes[0].set_ylabel('GiB');axes[0].set_title('六组完整训练显存 · 目标3090仍需实测');axes[0].set_xticks(x,labels,rotation=25,ha='right')
leg=axes[0].legend(fontsize=8,facecolor='#101d30',edgecolor='#34445d');plt.setp(leg.get_texts(),color='#d8e5f5')
axes[1].bar(x,[r['wall_minutes'] for r in rows],color=['#7aaff8']*4+['#43c8b0','#f2b75b'])
axes[1].set_ylabel('分钟');axes[1].set_title('各30k总耗时 · 含保存、LPIPS与时间控制');axes[1].set_xticks(x,labels,rotation=25,ha='right')
fig.savefig(O/'DrivingEditor_resource_and_speed.png',dpi=180,facecolor=fig.get_facecolor());fig.savefig(O/'DrivingEditor_resource_and_speed.pdf',facecolor=fig.get_facecolor());plt.close(fig)
table=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in [html.escape(r['scene']+' / '+r['arm']),f"{r['full_psnr_db']:.3f}",f"{r['actor_psnr_db']:.3f} ({r['actor_valid_frames']}/49)",f"{r['static_psnr_db']:.3f}",f"{r['wall_minutes']:.2f}",f"{r['allocated_GiB']:.3f}",f"{r['reserved_GiB']:.3f}",f"{r['PID_sampled_GiB']:.3f}",f"{r['device_sampled_GiB']:.3f}"] )+'</tr>' for r in rows)
page='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DrivingEditor · 六组30k结果</title>
<style>*{box-sizing:border-box}body{margin:0;background:#080e18;color:#dbe7f4;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:40px 24px 70px}h1{font-size:34px;line-height:1.2}h2{font-size:21px;color:#eff6ff}a{color:#68d6c3;text-decoration:none}p{color:#adbed2}.eyebrow{letter-spacing:2px;color:#56cdbb;font-size:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{display:block;padding:22px;border:1px solid #263850;border-radius:14px;background:#101c2d}.card strong{display:block;font-size:21px;color:#eff6ff}.card span{display:block;margin-top:12px;color:#a8bed7}.badge{color:#62d9bc}.note{background:#142139;border-left:3px solid #6797d9;padding:16px 20px;border-radius:5px;margin:24px 0;color:#c1d2e7}.warning{border-color:#e7b563}.component{width:100%;display:block;margin:22px auto}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:#0e192a;font-size:13px}th,td{padding:12px 10px;text-align:left;white-space:nowrap;border-bottom:1px solid #24324a}th{color:#b4d0ee}img{max-width:100%;border-radius:10px}.foot{font-size:13px;border-top:1px solid #263850;margin-top:25px;padding-top:16px}@media(max-width:800px){.grid{grid-template-columns:1fr}h1{font-size:28px}}</style>
<main><div class="eyebrow">DRIVINGEDITOR · REPRODUCTION REVIEW</div><h1>六组30k训练与分支检查</h1><p>原四臂120k + 两个追加场景各GN11 30k，共180,000步。全部scene为49 FRONT、seed0、从零训练，每10k保存完整状态。</p>
<div class="grid"><a class="card" href="drivingeditor_four_arm_30k/index.html"><strong>四臂 · C00 / G10 / N01 / GN11</strong><span>相同scene17、相同逐步采样，49帧同步比较与固定相机时间检查。</span><span class="badge">120k完成 · 1764 PNG</span></a><a class="card" href="drivingeditor_GMC_GN11_30k/index.html"><strong>GMC + Jeep · GN11</strong><span>Corolla保留静态，两个动态目标合入一个动态分支；原帧23–71。</span><span class="badge">30k完成 · 全49帧保留</span></a><a class="card" href="drivingeditor_red_GN11_30k/index.html"><strong>横穿红车 · GN11</strong><span>原帧69–117；真实DEVA实例130，末端3帧通过缓存SAM真实推理修复。</span><span class="badge">30k完成 · 全49帧保留</span></a></div>
<div class="note">GN11相对C00：全图 +0.3254 dB、车体 −0.1287 dB、非天空车外静态 +0.3175 dB。完整配对和实际49帧/固定时间复核未触发用户确认的严重回退条件；局部车体代价和静态残影仍保留。追加两个场景没有各自C00，不能用于证明模块增益。</div>
<h2>输入、组件与输出</h2><img class="component" src="drivingeditor_red_GN11_30k/architecture-components.svg" alt="49真实RGB和LiDAR经静态、动态Gaussian与共享Renderer输出分层RGB和alpha">
<p>继承2024活跃机制：完整动态mask opacity、静态RGB、寿命/速度约束、LiDAR深度衰减；SGCM使用修正后的视图支撑与克隆谱系规则，NPS保留原sum/L2公式。支撑判据是代理，NPS不保证绝对静态。</p>
<h2>真实30k重建结果与资源</h2><div class="scroll"><table><thead><tr><th>场景 / 臂</th><th>全图PSNR</th><th>车体PSNR / 有效帧</th><th>静态PSNR</th><th>分钟</th><th>allocated GiB</th><th>reserved GiB</th><th>PID采样 GiB</th><th>device采样 GiB</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p>RGB由保存FP32按原协议clip[0,1]后逐帧计算；静态区域为非天空且车体外。空body分数保留为null、均值只用有效帧，全部49帧仍可查看。不同scene不作配对质量比较。</p><img src="DrivingEditor_resource_and_speed.png" alt="完整训练显存与30k耗时对比">
<div class="note warning">3090容量初筛通过：六组训练与实际预处理的最大观测值为 __PEAK__ GiB（DEVA设备采样），低于预留2 GiB后的22 GiB预算。SAM本次也记录了allocator与11个NV样本，其设备峰6.422 GiB。采样峰是下界，未在目标3090上实际跑过；容量、速度和稳定性仍需目标机实测。sky/DEVA子进程的PyTorch allocated/reserved外部不可读，梯度诊断阶段没有NV采样，不以0替代缺值。</div>
<h2>应保留的质量边界</h2><p>GMC场景存在局部接地阴影不足、静态道路黑残影，以及固定pose下的静态RGB时变。红车观察帧的静态道路保留黑色车底/轮胎残影；固定pose下，黑色条块随模型时间移动，后段还出现右侧天空细绿线和局部浅粉道路斑块。这些反事实pose-time组合没有对应GT；两新场景没有各自C00，不能把绝对缺陷归因给SGCM/NPS的相对回退。全部49帧和固定49时间实际复核记录可查看。</p><p>body-only监督没有完整shadow，M-detector没有作为真实预处理实现。作者两个场景的实际训练窗口未公开，本地统一扩展为49帧。全部指标来自训练观测重建，没有heldout测试或删除后GT；不能称完整论文效果复现。</p>
<p>18个保存节点、每节点4组件及49帧/固定49时间的stat/header合同通过；未实际加载大checkpoint恢复模型、Adam、RNG或谱系，restoration_verified=false。网页和离线包是查看资产，完整大模型仍保留在AutoDL数据盘。</p>
<div class="foot"><a href="DrivingEditor_final_summary.json">结构化结果</a> · <a href="drivingeditor_four_arm_30k/summary.html">四臂详细分析</a> · <a href="drivingeditor_GMC_GN11_30k/results-report.md">GMC工程报告</a> · <a href="drivingeditor_red_GN11_30k/engineering-visual-review.json">红车真实视觉记录</a><p>人工质量评价：C00为用户确认符合预期；GN11与两个新scene仍为null。电源状态见独立关机凭据，不从SSH断开推定。</p></div></main></html>'''
page=page.replace('__TABLE__',table).replace('__PEAK__',f'{peak_with_preprocessing:.3f}')
out=O/'DrivingEditor_final_report.html';assert not out.exists();out.write_text(page,encoding='utf-8')
print(json.dumps({'final_report':str(out),'training_steps':180000,'observed_peak_with_preprocessing_GiB':peak_with_preprocessing}))
