"""从固定轻量结果生成开发基线报告；图像、视频均来自实际执行。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    result=json.loads((a.evidence/'evaluation.json').read_text())
    frames=result['frames'];fig,ax=plt.subplots(figsize=(8,3.2),layout='constrained')
    ax.plot([r['frame']/30 for r in frames],[r['paired_center_distance_px'] for r in frames],marker='o',color='#1864ab')
    ax.set(xlabel='Time (s)',ylabel='2D center difference (px)',ylim=(0,None),
           title='Clean generation versus recorded RGB | one development vehicle')
    ax.grid(alpha=.2)
    for ext in ['png','svg']:fig.savefig(a.output/f'center-response.{ext}',dpi=180)
    svg=a.output/'center-response.svg';svg.write_bytes(('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n').encode('utf-8'))
    table=''.join(f'<tr><td>{r["frame"]/30:.1f}</td><td>{r["real"] is not None}</td><td>{r["generated"] is not None}</td><td>{r["paired_center_distance_px"]:.2f}</td></tr>' for r in frames)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V7.5 · Argoverse 输入桥接与 clean 基线</title><style>
body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#192738;margin:0}main{max-width:1180px;margin:auto;padding:30px 24px 70px}h1{font-size:30px;margin:6px 0 12px}h2{font-size:22px;margin:30px 0 12px}.tag{color:#38617b;font-weight:600}.card{background:white;border:1px solid #dbe3ed;padding:18px;border-radius:12px;margin:18px 0}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.node{background:#eaf2ff;border:1px solid #bad1ed;border-radius:9px;padding:12px;text-align:center;flex:1;min-width:140px}img,video{max-width:100%;display:block;border-radius:6px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #dde4ed;text-align:left}.muted{color:#5c6e81}.notice{border-left:4px solid #ac7512;padding-left:15px}a{color:#175ea7}summary{cursor:pointer}
</style><main><div class="tag">WS-V75-AV2-BRIDGE-01 / 20260920-r1 · 开发资源</div>
<h1>真实视频、条件与生成：单卡 clean 基线</h1>
<p>同一条已曝光 Argoverse 2 日志，保留真实未来 RGB，以标注地图和轨迹构造条件。单张 RTX 3090 完成 237 帧，约 123 秒，无 OOM。</p>
<div class="card flow" aria-label="architecture components"><div class="node">真实初帧<br>固定文本</div><b>→</b><div class="node">官方 OmniDreams 2B<br>生成历史 / cache</div><b>→</b><div class="node">生成视频</div><b>→</b><div class="node">独立二维检测<br>对比真实未来 RGB</div></div>
<div class="card flow"><div class="node">GT 地图 / actor 轨迹<br>记录 ego 轨迹</div><b>→</b><div class="node">自建格式适配<br>官方 Ludus 渲染</div><b>→</b><div class="node">逐帧条件<br>送入上方生成器</div></div>
<p class="notice">本轮是输入与评价工程验证。条件来自额外标注信息，尚无自然重建误差干预，也没有策略反馈；画面或检测差异不能直接归因于重建模型。</p>
<h2>1 · 一眼查看完整对比</h2><p>左：真实 RGB；中：适配后的 GT 条件；右：clean 生成。固定同一时刻，没有按结果重新选裁剪。</p>
<video controls preload="metadata" src="comparison.mp4"></video><p class="muted">视频中真实参考按最近的 20Hz 帧展示；量化只用精确时间重合帧。</p><img src="clean-review.jpg" alt="固定五个时刻的真实RGB、条件、生成对比">
<h2>2 · 目标车与测量</h2><p>目标 94dede… 在生成前由几何可见性规则选定。真实与生成均在 9/9 个固定时刻匹配；检测器 ±4px 平移校准通过。二维框中心差中位数 2.68px，均值 4.89px，末帧 14.24px。这是 clean 与真实视频的差异，不能称作扰动放大。</p>
<img src="center-response.png" alt="clean相对真实视频的检测框中心差"><details><summary>逐时刻匹配与差异</summary><table><tr><th>时间 / s</th><th>真实匹配</th><th>生成匹配</th><th>中心差 / px</th></tr>TABLE</table></details>
<img src="target-review.jpg" alt="目标车辆固定参考位置裁剪：真实、条件、生成"><p class="muted">每行三列使用相同的200×120像素参考裁剪，并等比放大；指标仍按1280×704原分辨率计算。</p>
<h2>3 · 已验证与仍缺失</h2><ul><li>原生相机 1550×2048，等比中央裁剪到 1280×704；RGB 和投影共享变换。</li><li>237 个生成时刻中，79 个与原始 20Hz 视频精确重合。轨迹用世界坐标插值，超过 150ms 的缺口分段，不外推。</li><li>针孔到 renderer 相机多项式最大残差 0.0034px；三处实际渲染框与针孔边界差不超过 1.61px。</li><li>独立检查拦住了 RDF / FLU 接口错误，修正后才启动生成；原始失败检查保留。</li><li>双线简化、可行驶区域边界代理及交通灯等标注缺失均会影响条件语义。当前相机域与官方样例不同；不把这条适配基线作为 SOTA 普遍失效证据。</li></ul>
<details><summary>原始投影与输入检查图</summary><img src="projection-review.jpg" alt="真实图像与GT投影"><img src="condition-review.jpg" alt="真实图像与条件渲染"></details>
<p class="muted">整卡采样峰值 17.41GiB；PyTorch 峰值分配 12.81GiB。原始目录：/root/autodl-tmp/runs/worldsim_v75/WS-V75-AV2-BRIDGE-01/20260920-r1。failure_ledger_delta：none；人工 verdict：null。</p></main></html>'''.replace('TABLE',table)
    (a.output/'index.html').write_text(html,encoding='utf-8')

if __name__=='__main__':main()
