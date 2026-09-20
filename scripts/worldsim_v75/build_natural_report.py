"""只从实际读出和生成结果构建科研图与本地HTML，不合成实验画面。"""
import argparse
import html
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES={'gt_clean':'GT condition','dvgt_metric':'DVGT + known rays','dvgt_lidar_scaled':'+ global LiDAR scale',
       'ordinary_bbox':'RGB box fit + size/yaw','reference_lidar':'Extra target LiDAR'}
COLORS={'gt_clean':'#576578','dvgt_metric':'#c9433a','dvgt_lidar_scaled':'#d69a22','ordinary_bbox':'#735fb0','reference_lidar':'#168575'}

def replication_section(evidence, out):
    result=json.loads((evidence/'replication_result.json').read_text(encoding='utf-8'))
    assert result['status']=='complete' and result['seeds']==[42,43]
    destination=out/'seed43';destination.mkdir(exist_ok=True)
    for name in ['comparison.mp4','target-comparison.jpg','full-comparison.jpg','review_result.json','protocol.json','queue_result.json']:
        shutil.copy2(evidence/name,destination/name)
    shutil.copy2(evidence/'replication_result.json',out/'replication_result.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained',sharey=True)
    rows='';contrasts=''
    for ax,row in zip(axes,result['rows']):
        seed=row['seed']
        rows+=f'<tr><td>{seed}</td>'+''.join(f'<td>{row["means_px"][k]:.2f}</td>' for k in result['variants'])+'</tr>'
        for name,case in row['cases'].items():
            ax.plot([r['frame']/30 for r in case['pairs']],[r['to_real_px'] for r in case['pairs']],
                    color=COLORS[name],marker='o',label=NAMES[name])
        for contrast in row['contrasts']:
            contrasts+=f'<tr><td>{seed}</td><td>{NAMES[contrast["comparator"]]}</td><td>{contrast["dvgt_minus_comparator_mean_px"]:+.2f}</td><td>{contrast["comparator_better_noninitial_frames"]}/4</td></tr>'
        ax.set_title(f'Seed {seed} | same frozen input states',loc='left',fontsize=11)
        ax.set_xlabel('Time (s)');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('2D detector center distance to real RGB (px)');axes[0].legend(fontsize=8)
    fig.savefig(out/'replication-response.png',dpi=180);fig.savefig(out/'replication-response.svg');plt.close(fig)
    headings=''.join(f'<th>{NAMES[name]}</th>' for name in result['variants'])
    return f'''<section><h2>固定 seed 43 复核：改善重复，普通对照同样有效</h2>
<p>同一白车、同一条件/编码/初帧、同一五个评价时刻，仅更换seed。新增四组各237帧全部解码，检测20/20匹配，无OOM，PyTorch峰值12.81GiB。没有重跑重建、改变误差或扩大评价窗口。</p>
<table><tr><th>Seed</th>{headings}</tr>{rows}</table><small>表中为前两秒固定五帧的二维中心距离均值，单位px；相邻时刻不是独立样本。</small>
<img src="replication-response.svg"><p>额外目标LiDAR在两个seed各四个非初始时刻均改善，但它增加了真实观测。普通框拟合的三维误差3.986m，比DVGT的2.965m更大，却在两个seed都具有较好二维均值。GT条件自身也有明显生成偏差，不能把DVGT的全部误差归因于输入位置。</p>
<table><tr><th>Seed</th><th>DVGT 对照项</th><th>DVGT − 对照均值(px)</th><th>对照改善的非初始时刻</th></tr>{contrasts}</table>
<p>正数表示对照更接近真实RGB。三项对比和全部五帧均保留；没有增加事后准入阈值。第2秒行人遮挡的限制继续适用。</p>
<video src="seed43/comparison.mp4" controls preload="metadata"></video><img src="seed43/target-comparison.jpg">
<p class="note">通过的是同一发现案例的随机性复核，不是跨场景确认。现有结果支持位置条件替换的重复响应与改善空间；未证明误差放大、米制生成状态漂移、普遍重建缺口或闭环危害。不追加seed44或更大偏移来维持主张。</p>
<p><a href="replication_result.json">两seed全部对比</a> · <a href="seed43/protocol.json">预先冻结复核协议</a> · <a href="seed43/review_result.json">seed43原始量化</a></p></section>'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replication',type=Path);parser.add_argument('--trace',type=Path)
    a=parser.parse_args();e=a.evidence;out=a.output;out.mkdir(parents=True,exist_ok=True)
    sources=json.loads((e/'readout_queue_result.json').read_text(encoding='utf-8'))
    rollout=json.loads((e/'review_result.json').read_text(encoding='utf-8'))
    assert sources['status']==rollout['status']=='complete'
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained');x=np.arange(4);width=.19
    for j,name in enumerate(['dvgt_metric','dvgt_lidar_scaled','ordinary_bbox','reference_lidar']):
        values=[r['readouts'][name]['center_error_m'] for r in sources['cases']]
        ax.bar(x+(j-1.5)*width,values,width,color=COLORS[name],label=NAMES[name])
    ax.set_xticks(x,[r['log_id'][:8] for r in sources['cases']]);ax.set_ylabel('Initial center error to annotation (m)')
    ax.set_xlabel('Four pre-screened development logs; shared GT size/yaw')
    ax.set_title('A natural residual is not universal; a global scale can worsen it',loc='left',fontsize=11)
    ax.legend(fontsize=8,ncol=2);ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'state-errors.png',dpi=180);fig.savefig(out/'state-errors.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for row in rollout['variants']:
        name=row['variant'];t=[p['frame']/30 for p in row['pairs']]
        axes[0].plot(t,[p['to_real_px'] for p in row['pairs']],marker='o',label=NAMES[name],color=COLORS[name])
        axes[1].plot(t,[p['paired_clean_distance_px'] for p in row['pairs']],marker='o',label=NAMES[name],color=COLORS[name])
    for ax,title in zip(axes,['Generated vs real RGB','Generated vs same-seed GT condition']):
        ax.set_title(title,loc='left',fontsize=11);ax.set_xlabel('Time (s)');ax.set_ylabel('2D detector center distance (px)')
        ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'generation-response.png',dpi=180);fig.savefig(out/'generation-response.svg');plt.close(fig)
    # Windows写入统一LF，避免在Git中给整份SVG引入CRLF噪声。
    for svg in out.glob('*.svg'):
        normalized='\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n'
        svg.write_bytes(normalized.encode('utf-8'))
    for name in ['target-comparison.jpg','full-comparison.jpg','comparison.mp4','selected-crops.jpg','source-screen.jpg']:
        shutil.copy2(e/name,out/name)
    for name in ['readout_queue_result.json','readout_protocol.json','source_protocol.json','initial_readout.json','review_result.json']:
        shutil.copy2(e/name,out/name)
    state_rows=''.join('<tr><td>'+r['log_id'][:8]+'</td>'+''.join(f'<td>{r["readouts"][k]["center_error_m"]:.3f}</td>' for k in ['dvgt_metric','dvgt_lidar_scaled','ordinary_bbox','reference_lidar'])+'</tr>' for r in sources['cases'])
    gen_rows=''.join(f'<tr><td>{NAMES[r["variant"]]}</td><td>{r["valid_matches"]}/{r["scheduled"]}</td><td>{r["mean_distance_to_real_px"]:.2f}</td><td>{r["mean_paired_clean_distance_px"]:.2f}</td></tr>' for r in rollout['variants'])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V7.5 自然位置误差与生成响应</title>
<style>body{{margin:0;background:#eaf0f6;color:#182536;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1300px;margin:auto;padding:36px 24px}}section{{background:white;padding:28px;margin:22px 0;border-radius:12px}}h1{{font-size:30px}}h2{{font-size:23px}}p{{max-width:1100px}}img,video{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dde5eb}}.flow{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.box{{background:#163954;color:white;border-radius:8px;padding:14px;flex:1;min-width:130px}}.note{{background:#fff4d9;padding:14px;border-left:4px solid #c89a34}}small{{color:#526477}}</style>
<main><h1>自然位置误差，是否进入生成状态？</h1><p>V7.5 · 四日志初始状态读出 + 一个发现案例的五组真实生成 · seed 42 · 单张 RTX 3090</p>
<p class="note">单seed发现信号：额外目标LiDAR修复让二维均值25.19→10.50px；但三维误差更大的普通框拟合也达到13.97px。尚未完成重复或独立确认，不能声称三维越准就必然生成越准。</p><section><h2>输入与真实条件通路</h2><div class="flow"><div class="box">起点前七视图 RGB</div>→<div class="box">官方 DVGT-1<br>ego 距离读出</div>→<div class="box">已知相机射线<br>固定尺寸/朝向读出位置</div>→<div class="box">3D actor → 官方 raster</div>→<div class="box">OmniDreams + 生成历史</div>→<div class="box">生成RGB<br>与真实视频配对评价</div></div>
<p class="note">尺寸、朝向、未来 actor/ego 运动和其他场景条件使用额外 GT。实验只替换目标的初始位置，并把这一偏移保持在共同的未来轨迹上。LiDAR 校正与修复分列；这不是纯视觉端到端系统，也没有策略反馈。</p></section>
<section><h2>先找可靠目标，再看模型</h2><p>旧工程目标约110米、网络输入约11×7像素，起点前目标LiDAR为0，因此停止其因果生成。固定八条开发日志后，六条具有足够参考支持；两条因真实RGB遮挡排除，剩余四条做同一读出。没有按模型误差改目标或扩大日志池。</p><details><summary>查看八条日志与六个初选目标（包括排除例）</summary><img src="source-screen.jpg"><img src="selected-crops.jpg"></details>
<table><tr><th>开发日志</th><th>DVGT + 已知射线 (m)</th><th>+ 全局LiDAR尺度 (m)</th><th>普通框拟合 (m)</th><th>额外目标LiDAR (m)</th></tr>{state_rows}</table><img src="state-errors.svg"><p>误差针对标注中心；最后一列是同一读出器使用真实点的校准与额外信息上界。两条低残余案例保留。全局尺度造成的退化，不能冒称模型原始错误。</p></section>
<section><h2>2.96米发现案例：实际生成结果</h2><p>选择 0bae3b5e 的白车进行发现性比较。真实RGB的五个时刻均匹配，±4px校准残差均小于2px。固定前两秒量化，其余完整237帧保留供审阅；没有把相邻帧计为独立案例。</p>
<video src="comparison.mp4" controls preload="metadata"></video><small>真实视频 / GT条件 / DVGT读出 / 额外目标LiDAR修复；预览参考采用20Hz最近帧，指标仅用精确重合时刻。</small>
<img src="full-comparison.jpg"><img src="target-comparison.jpg"><p>目标裁剪以真实检测框为中心，所有栏使用同一裁剪，不跟随预测移动。第2秒出现行人遮挡；该帧保留在预定分母中，但检测中心还包含外观/遮挡敏感性，不能单点推断三维漂移或安全后果。</p>
<table><tr><th>条件来源</th><th>有效检测</th><th>与真实中心差均值 (px)</th><th>与GT条件生成差均值 (px)</th></tr>{gen_rows}</table><img src="generation-response.svg"></section>
<section><h2>证据范围</h2><p>这是一个seed、一个生成发现案例和四个开发读出。二维检测差不等于生成三维中心误差，也不等于碰撞或闭环危害。原始点图朝向未直接通过已知rig投影检查，结果属于明确标注的“ego距离＋已知射线”适配诊断。若GT条件基线已有明显偏差，须和新增条件影响分开解释。</p><p>原始数据、native点图、输入条件、未压缩生成、失败与良好案例均保留在远端。人工verdict仍为null。</p><p><a href="readout_queue_result.json">四日志完整读出</a> · <a href="readout_protocol.json">读出与遮挡记录</a> · <a href="source_protocol.json">八日志冻结协议</a> · <a href="review_result.json">生成量化结果</a></p></section></main></html>'''
    if a.replication:
        section=replication_section(a.replication,out)
        body=body.replace('<section><h2>2.96米发现案例：实际生成结果</h2>',section+'<section><h2>seed42 发现阶段：保留全局尺度在内的五组结果</h2>')
        body=body.replace('五组真实生成 · seed 42','五组发现生成 + 四组固定复核 · seeds 42 / 43')
        body=body.replace('单seed发现信号：额外目标LiDAR修复让二维均值25.19→10.50px；但三维误差更大的普通框拟合也达到13.97px。尚未完成重复或独立确认，不能声称三维越准就必然生成越准。',
                          '同一案例两seed复核：额外目标LiDAR修复让二维均值25.19→10.50px、24.24→11.71px；但三维误差更大的普通框拟合也达到13.97 / 14.33px。改善可重复，几何准确度与生成质量仍不单调；尚无跨场景或闭环确认。')
        body=body.replace('这是一个seed、一个生成发现案例和四个开发读出。','这是两个seed、一个生成发现案例和四个开发读出。')
    if a.trace:
        from build_natural_trace_figure import build_trace
        build_trace(a.trace,out)
        shutil.copy2(a.trace/'trace_data.json',out/'trace_data.json')
        body=body.replace('<section><h2>先找可靠目标，再看模型</h2>',
            '<section><h2>把初始位置误差放回场景</h2><img src="input-state-condition.svg"><p>黄色框是真实初帧中的同一白车。BEV按实际坐标与共享尺寸绘制：DVGT读出沿车前向近2.963m，完整三维中心差为2.965m。右侧为实际送入生成器的条件裁剪，含填充cuboid与地图；固定显示1秒，量化仍保留全部五个时刻。这是actor位置误差图，不是首回波或假表面证据。</p><a href="trace_data.json">图中坐标与来源</a></section><section><h2>先找可靠目标，再看模型</h2>')
    for svg in out.glob('*.svg'):
        normalized='\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n'
        svg.write_bytes(normalized.encode('utf-8'))
    (out/'index.html').write_text(body,encoding='utf-8',newline='\n')
    print(out/'index.html')

if __name__=='__main__':main()
