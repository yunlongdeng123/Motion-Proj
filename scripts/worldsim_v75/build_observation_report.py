"""保留观测缺口的曲线和实际点轨迹，供主报告引用。"""
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'dvgt_metric':'#c9433a','ordinary_bbox':'#735fb0','reference_lidar':'#168575'}
NAMES={'dvgt_metric':'DVGT + known rays','ordinary_bbox':'Ordinary box fit','reference_lidar':'Extra target LiDAR'}

def observation_section(evidence,out):
    result=json.loads((evidence/'result.json').read_text(encoding='utf-8'))
    assert result['status']=='complete'
    folder=out/'observation';folder.mkdir(exist_ok=True)
    for name in ['protocol.json','result.json','calibration.json','queue_result.json','tracking-seed42.jpg','tracking-seed43.jpg',
                 'real-targets.jpg','confirmation-source-screen.jpg','confirmation_protocol.json','confirmation_screen_result.json','confirmation_visual_review.json']:
        shutil.copy2(evidence/name,folder/name)
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for row in result['rows']:
        ax.plot([r['frame']/30 for r in row['frames']],[r['common_points'] for r in row['frames']],marker='o',label=f'Seed {row["seed"]}')
    ax.axhline(16,color='#777',linestyle='--',label='Frozen minimum: 16 of 62')
    ax.set_xticks([0,.5,1,1.5,2]);ax.set_ylim(0,65);ax.set_ylabel('Common retained initial point identities')
    ax.set_xlabel('Time (s)');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    fig.savefig(folder/'support.png',dpi=180);fig.savefig(folder/'support.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained',sharey=True)
    for ax,name in zip(axes,COLORS):
        ref=result['rows'][0]['frames'];times=[r['frame']/30 for r in ref]
        ax.plot(times,[r['arms'][name]['projection_shift_px'][0] for r in ref],color='#283d51',linestyle='--',label='Input cuboid projection')
        for row,marker in zip(result['rows'],['o','s']):
            values=[r['arms'][name]['paired_clean_vector_px'][0] if r['admitted'] else np.nan for r in row['frames']]
            ax.plot(times,values,marker=marker,color=COLORS[name],alpha=1 if row['seed']==42 else .6,label=f'RAFT points, seed {row["seed"]}')
        ax.set_title(NAMES[name],loc='left',fontsize=10);ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.axhline(0,color='#bbb',linewidth=.7)
        ax.spines[['top','right']].set_visible(False);ax.set_xticks([0,.5,1,1.5,2])
    axes[0].set_ylabel('Horizontal shift relative to GT condition (px)');axes[1].legend(fontsize=7)
    fig.savefig(folder/'projection-control.png',dpi=180);fig.savefig(folder/'projection-control.svg');plt.close(fig)
    for path in folder.glob('*.svg'):
        path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    rows=''
    for row in result['rows']:
        for f in row['frames'][1:]:
            values=['—' if f['arms'][name]['to_real_distance_px'] is None else f'{f["arms"][name]["to_real_distance_px"]:.2f}' for name in ['gt_clean',*COLORS]]
            rows+=f'<tr><td>{row["seed"]} / {f["frame"]/30:.1f}s</td><td>{f["common_points"]}/62</td>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>'
    return f'''<section><h2>独立点轨迹检查：普通投影响应与测量边界</h2>
<p>使用现有RAFT权重，在真实初帧白车中央选择62个纹理点，随后只按图像光流追踪；未来检测框不参与追踪。±4px已知平移校准的中位残差为0.020 / 0.024px，全部点保留。这仅验证初帧局部平移，不证明后续语义对应总是正确。</p>
<div class="flow"><div class="box">同一真实初帧<br>62个纹理点</div>→<div class="box">已有真实/生成帧<br>RAFT双向追踪</div>→<div class="box">固定共同支持<br>与普通投影比较</div></div>
<img src="observation/support.svg"><p class="note">共同支持需至少8点且不少于初始点的25%，即本例至少16点。seed42末帧仅8点；seed43从1秒起不足，末帧为0。完整两秒的独立均值不成立，缺失值保留为空，不补零、不缩短窗口重新宣布通过。支持流失是观测限制，不是驾驶失效。</p>
<table><tr><th>Seed / 时刻</th><th>共同点</th><th>GT条件(px)</th><th>DVGT(px)</th><th>普通框(px)</th><th>额外LiDAR(px)</th></tr>{rows}</table>
<small>表中为共同初始点“生成−真实”位移的逐坐标中位向量长度；不是车辆中心或米制三维误差。</small>
<img src="observation/projection-control.svg"><p>可测非初始时刻的横向响应与条件投影同号；幅度也相近。seed42的DVGT条件投影为+10.15、+3.26、−7.06px，生成点相对GT条件为+6.61、+2.44、−4.01px。其余控制完整显示。投影框中心与纹理点不是同一观测对象，故不把两者差或比值称为历史放大。</p>
<p>三维误差范数忽略方向和透视：本例2.965m的DVGT读出与0.417m的LiDAR读出，初帧投影框中心分别右移14.92和14.15px。普通框拟合的二维收益也可能包含对GT条件生成基线偏差的补偿，不能由收益单独推出更好的物理状态。</p>
<details><summary>全部实际点轨迹（青色为共同点，黄色为仅本序列保留点）</summary><img src="observation/tracking-seed42.jpg"><img src="observation/tracking-seed43.jpg"></details>
<p class="note">阶段结论：保留同场景条件敏感性与额外观测改善信号；本例不晋级为额外放大、稳定三维偏差或主论文badcase。不继续改本例的追踪阈值、seed、窗口或幅度。下一批来源先验证真实输入的可观测性，再盲看模型输出。</p>
<p><a href="observation/protocol.json">冻结协议</a> · <a href="observation/result.json">全部结果与缺失分母</a> · <a href="https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html">Torchvision官方输入与像素单位契约</a></p></section>
<section><h2>有限新来源窗口：没有合格目标，未运行模型</h2><p>按本地完整性、UUID顺序及既有文档未提及记录冻结4条日志。未被文档提及不保证从未曝光，也不排除预训练重叠。一条没有满足原几何/参考规则的目标；另外三条的预选目标在五个真实时刻存在遮挡或轮廓混叠，无法作为干净目标。没有换目标、换起点或追加日志。</p>
<img src="observation/real-targets.jpg"><p>完整分母：4条日志，3条几何初筛通过，0条真实可观测性通过，0次重建、0次生成。这是采样与测量条件的限制，不是模型科学失败。固定UUID优先的选择规则不保证得到适合运动测量的前景目标；本窗口按停止规则关闭。</p>
<details><summary>四日志初筛全景</summary><img src="observation/confirmation-source-screen.jpg"></details>
<p><a href="observation/confirmation_protocol.json">来源冻结与曝光审计</a> · <a href="observation/confirmation_visual_review.json">逐日志排除记录</a></p></section>'''
