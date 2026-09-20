"""汇总三条原生样例实际clean结果，HTML不含预设失败结论。"""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads((a.evidence/f'scene-{i}-review.json').read_text()) for i in range(3)]
    table=''.join(f'<tr><td>{i+1}</td><td>{r["scene_uuid"]}</td><td>{r["generation_wall_s"]:.1f}s</td><td>{r["peak_allocated_gib"]:.2f}GiB</td><td>{r["decode_counts"]["clean.mp4"]}/237</td></tr>' for i,r in enumerate(rows))
    sections=''.join(f'<section><h2>开发样例 {i+1} · {r["scene_uuid"]}</h2><video controls preload="metadata" src="scene-{i}.mp4"></video><details open><summary>固定 0 / 2 / 4 / 6 / 7.8 秒</summary><img src="scene-{i}.jpg" alt="样例{i+1}：真实RGB、官方条件与生成的固定时刻对比"></details></section>' for i,r in enumerate(rows))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 · 官方原生样例 clean 基线</title>
<style>body{margin:0;background:#f3f6fa;color:#1b2b40;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:30px 24px 70px}h1{font-size:30px;margin:8px 0}h2{font-size:20px;overflow-wrap:anywhere}section{background:white;padding:20px;border:1px solid #dfe6ef;border-radius:12px;margin:24px 0}.flow{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}.box{border:1px solid #b6cee5;background:#e9f3ff;padding:14px;border-radius:8px;flex:1;text-align:center;min-width:130px}video,img{max-width:100%;display:block;margin:12px 0}.tag{color:#38627f;font-weight:600}.note{border-left:4px solid #ac771b;padding-left:14px;color:#586b7c}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dbe4ed;text-align:left}td:nth-child(2){overflow-wrap:anywhere}summary{cursor:pointer}a{color:#175ea7}</style>
<main><div class="tag">WS-V75-NATIVE-COHORT-01 / 20260920-r1</div><h1>三个官方原生样例：真实视频 → 条件 → clean 生成</h1>
<p>单张 RTX 3090；官方 single-view 2B 权重、704×1280、30fps、seed42，每个样例237帧。三条视频均无OOM并完整解码。左列是真实未来RGB，中列是官方提供的HDmap视频，右列是本轮实际生成。</p>
<div class="flow" aria-label="architecture components"><div class="box">官方初帧 / 文本<br>官方 HDmap 固定前缀</div><b>→</b><div class="box">官方 OmniDreams<br>自回归历史 / cache</div><b>→</b><div class="box">生成 RGB</div><b>→</b><div class="box">与真实未来 RGB<br>按原始时间戳对照</div></div>
<p class="note">这是基线资格检查，尚无自然重建误差干预或驾驶后果测量。画面差异本身不证明重建误差放大；不能把三个clean样例称为三个已确认badcase。</p>
<table><tr><th>样例</th><th>固定 UUID</th><th>加载+生成</th><th>PyTorch峰值</th><th>解码检查</th></tr>TABLE</table>
<p>开发/保留划分先按官方UUID字典序冻结：前3个开发，随后3个保留。保留视频未读取。三例官方PNG初帧与RGB视频第0帧在共同预处理后逐像素相同，RGB/HDmap前缀时间戳逐帧一致。</p>
<p>固定读取前237帧，使用官方共用resize函数的INTER_AREA预处理，显式送入底层官方pipeline。80–100秒原始文件只做必要前缀读取；未把部分下载声明为完整资产。</p>
SECTIONS
<section><h2>本轮证据边界</h2><p>samples仓库提供RGB、HDmap、初帧和文本，没有同案例的三维actor轨迹或相机/地图文件。它可以验证原生条件下的生成基线和未来RGB比较；开展真实重建状态→条件因果实验，还需要可编辑、可核验的三维状态接口。</p><p>Argoverse桥接已另行验证，其额外GT状态、视场与地图语义限制单独记录。当前未新增科学失败卡，人工verdict保持null。</p></section>
<p class="note">原始目录：/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATIVE-COHORT-01/20260920-r1。generation保持官方权重、调度器与历史设置；编码器预计算后释放。没有训练、策略反馈或关机队列。</p></main></html>'''.replace('TABLE',table).replace('SECTIONS',sections)
    (a.output/'index.html').write_text(html,encoding='utf-8')

if __name__=='__main__':main()
