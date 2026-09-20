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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
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
<p class="note">单seed发现信号：额外目标LiDAR修复让二维均值25.19→10.50px；但三维误差更大的普通框拟合也达到13.97px。尚未完成重复或独立确认，不能声称三维越准就必然生成越准。</p><section><h2>输入与真实条件通路</h2><div class="flow"><div class="box">起点前七视图 RGB</div>→<div class="box">官方 DVGT-1<br>ego 距离读出</div>→<div class="box">已知相机射线<br>固定尺寸/朝向读出位置</div>→<div class="box">3D actor → 官方 raster</div>→<div class="box">OmniDreams + 生成历史</div></div>
<p class="note">尺寸、朝向、未来 actor/ego 运动和其他场景条件使用额外 GT。实验只替换目标的初始位置，并把这一偏移保持在共同的未来轨迹上。LiDAR 校正与修复分列；这不是纯视觉端到端系统，也没有策略反馈。</p></section>
<section><h2>先找可靠目标，再看模型</h2><p>旧工程目标约110米、网络输入约11×7像素，起点前目标LiDAR为0，因此停止其因果生成。固定八条开发日志后，六条具有足够参考支持；两条因真实RGB遮挡排除，剩余四条做同一读出。没有按模型误差改目标或扩大日志池。</p><details><summary>查看八条日志与六个初选目标（包括排除例）</summary><img src="source-screen.jpg"><img src="selected-crops.jpg"></details>
<table><tr><th>开发日志</th><th>DVGT + 已知射线 (m)</th><th>+ 全局LiDAR尺度 (m)</th><th>普通框拟合 (m)</th><th>额外目标LiDAR (m)</th></tr>{state_rows}</table><img src="state-errors.svg"><p>误差针对标注中心；最后一列是同一读出器使用真实点的校准与额外信息上界。两条低残余案例保留。全局尺度造成的退化，不能冒称模型原始错误。</p></section>
<section><h2>2.96米发现案例：实际生成结果</h2><p>选择 0bae3b5e 的白车进行发现性比较。真实RGB的五个时刻均匹配，±4px校准残差均小于2px。固定前两秒量化，其余完整237帧保留供审阅；没有把相邻帧计为独立案例。</p>
<video src="comparison.mp4" controls preload="metadata"></video><small>真实视频 / GT条件 / DVGT读出 / 额外目标LiDAR修复；预览参考采用20Hz最近帧，指标仅用精确重合时刻。</small>
<img src="full-comparison.jpg"><img src="target-comparison.jpg"><p>目标裁剪以真实检测框为中心，所有栏使用同一裁剪，不跟随预测移动。第2秒出现行人遮挡；该帧保留在预定分母中，但检测中心还包含外观/遮挡敏感性，不能单点推断三维漂移或安全后果。</p>
<table><tr><th>条件来源</th><th>有效检测</th><th>与真实中心差均值 (px)</th><th>与GT条件生成差均值 (px)</th></tr>{gen_rows}</table><img src="generation-response.svg"></section>
<section><h2>证据范围</h2><p>这是一个seed、一个生成发现案例和四个开发读出。二维检测差不等于生成三维中心误差，也不等于碰撞或闭环危害。原始点图朝向未直接通过已知rig投影检查，结果属于明确标注的“ego距离＋已知射线”适配诊断。若GT条件基线已有明显偏差，须和新增条件影响分开解释。</p><p>原始数据、native点图、输入条件、未压缩生成、失败与良好案例均保留在远端。人工verdict仍为null。</p><p><a href="readout_queue_result.json">四日志完整读出</a> · <a href="readout_protocol.json">读出与遮挡记录</a> · <a href="source_protocol.json">八日志冻结协议</a> · <a href="review_result.json">生成量化结果</a></p></section></main></html>'''
    (out/'index.html').write_text(body,encoding='utf-8',newline='\n')
    print(out/'index.html')

if __name__=='__main__':main()
