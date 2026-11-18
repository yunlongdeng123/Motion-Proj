"""生成深色 CPU 诊断报告；数值表来自冻结的诊断 JSON。"""
from pathlib import Path
import argparse
import html
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.out / 'report_data.json').read_text(encoding='utf-8'))
    esc = html.escape
    rows = ''.join('<tr>' + ''.join('<td>' + esc(str(v)) + '</td>' for v in row) + '</tr>' for row in d['measurements'])
    figures = ''.join(f'<figure><a href="{esc(f[0])}" target="_blank"><img loading="lazy" src="{esc(f[0])}" alt="{esc(f[1])}"></a><figcaption>{esc(f[1])}</figcaption></figure>' for f in d['figures'])
    proof = ''.join(f'<li><a href="{esc(x[0])}">{esc(x[1])}</a></li>' for x in d['evidence'])
    checklist = ''.join(f'<tr><td>{esc(x[0])}</td><td>{esc(x[1])}</td></tr>' for x in d['validation'])
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>二维道路教师：色差与高频细节 CPU 诊断</title><style>
:root{color-scheme:dark;--bg:#0b111a;--panel:#111b27;--ink:#e6edf5;--muted:#9eb0c4;--line:#26374b;--a:#8fcdff;--good:#6bd4b1;--warn:#f1c578}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:42px 28px 70px}h1{font-size:34px;line-height:1.3;letter-spacing:-.5px;margin:14px 0 20px}h2{font-size:23px;margin:38px 0 14px}h3{font-size:18px;margin:0 0 9px}p{margin:10px 0}.eyebrow,.muted,figcaption{color:var(--muted)}.eyebrow{font-size:13px;letter-spacing:1px}a{color:var(--a)}.lead{font-size:19px;max-width:1040px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.card,.notice{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:19px 21px}.status{display:flex;flex-wrap:wrap;gap:9px;margin:20px 0}.tag{border:1px solid var(--line);border-radius:30px;padding:4px 12px;font-size:13px}.tag.good{color:var(--good)}.tag.warn{color:var(--warn)}img{display:block;width:100%;height:auto;border-radius:10px}figure{margin:18px 0 28px}figcaption{font-size:13px;margin:7px 0}.scroll{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{background:#172333;white-space:nowrap}tr:last-child td{border:0}code{font-family:Consolas,monospace;font-size:13px;overflow-wrap:anywhere;color:#b5d8f4}pre{white-space:pre-wrap;background:#070d14;padding:17px;border:1px solid var(--line);border-radius:8px}ol,ul{padding-left:23px}.note{border-left:3px solid var(--warn);padding-left:15px;color:#c1cbd7}.small{font-size:14px}details{margin:15px 0}summary{cursor:pointer;color:var(--a)}footer{margin-top:36px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px}@media(max-width:700px){main{padding:26px 16px}h1{font-size:27px}.cards{grid-template-columns:1fr}th,td{padding:9px;font-size:13px}.lead{font-size:17px}}
</style><main>
<div class="eyebrow">WORLDSIM · 2026-11-19 · CPU 诊断</div>
<h1>教师失真与监督采样损失，需要分别处理</h1>
<p class="lead">__LEAD__</p>
<div class="status"><span class="tag good">CPU 诊断与输入准备完成</span><span class="tag">GPU 推理 0</span><span class="tag">训练更新 0</span><span class="tag">三维渲染修改 0</span><span class="tag warn">GPU 画质与显存尚未验收</span></div>
<img src="architecture.svg" alt="实拍与掩码进入冻结二维教师，原始输出分别进入监督包与硬回填诊断；三维渲染不在本轮范围">
<h2>责任划分</h2><div class="cards">
<article class="card"><h3>教师原始生成：已有失真</h3><p>使用无损原始解码，与同位置干净道路对照。色调与纹理频谱在硬回填前已经偏离，属于二维教师链路；扩散网络、VAE 与输入处理的具体责任仍待 GPU 控制分离。模型尺度高频能量并非普遍不足，伪线条也会抬高统计。</p></article>
<article class="card"><h3>教师 → 学生：有额外损失</h3><p>实际链路是 <code>768×512 → 双线性 1920×1280（uint8）→ 4×4 均值 480×320</code>。放大不能恢复纹理，之后的面积平均会继续去除高频。</p></article>
<article class="card"><h3>黑色接缝：单独问题</h3><p>历史 W 硬回填会保留许可边缘的原图黑圈。已有审计显示，学生孔洞监督的合法完整足迹内，raw 与回填目标差为零。接缝不能解释全部学生颗粒缺失。</p></article>
<article class="card"><h3>三维投影：本轮不判断</h3><p>高斯足迹和水平细纹由其他对话处理。此报告只诊断二维源与采样，不将二维包的 CPU 验证表述为三维画质改善。</p></article></div>
<h2>固定规则下的测量</h2>
<p class="small muted">4 个合成遮挡任务来自 081 街段两个时刻，使用独立 clean RGB 与道路证书；它们是已用过的诊断集，不是盲测。真实 065 车下没有真值。纹理比较排除许可边界与明显结构边缘，高频能量不等同于真实颗粒。</p>
<div class="scroll"><table><thead><tr><th>测量</th><th>结果</th><th>解释与边界</th></tr></thead><tbody>__ROWS__</tbody></table></div>
__FIGURES__
<h2>四个假设的证据</h2><div class="cards">
<article class="card"><h3>1 · 区域大，上下文少：局部支持</h3><p>当前 B 输入整图隐藏占比约 6.87%–12.27%，训练中位数约 34.65%，不能以全图比例认定超出训练。raw110 到声明可见道路的最远距离约 105.5 模型像素，训练中最大约 79；raw86 约 70。训练与查询的掩膜语义不同，此距离仅是上下文支持代理。</p></article>
<article class="card"><h3>2 · 分辨率不足：明确瓶颈</h3><p>1920 原图缩到 768 后，每个模型像素跨 2.5 个原图像素；原图约 5 像素以下周期超出其采样带宽。VAE 又以 8 倍空间压缩运行，但解码器仍能合成像素纹理，不能把 8 倍压缩当作硬性的纹理截止频率。提高分辨率须由固定对照验证。</p></article>
<article class="card"><h3>3 · 注意力 LoRA 不擅长纹理：尚未证实</h3><p>当前 rank-4 只训练注意力投影；这能限制适配自由度，但并不证明无法生成颗粒。历史 600 步在注意力上追加局部卷积 LoRA 的对照，未优于注意力版本；暂不直接切换结构。</p></article>
<article class="card"><h3>4 · 训练步数不足：尚未证实</h3><p>实际为 600 次单样本更新、68 个训练样本，每例约 8–9 次。训练只在伪孔洞区域计算噪声预测 MSE，没有显式颜色或纹理损失。200/400 步检查点保留，但无配对评价曲线；先分离分辨率、视野、LoRA 与 VAE，再决定是否续训。</p></article></div>
<p class="note">训练是原图下方 <code>[0,600,1920,1280]</code> 裁片缩为 <code>768×272</code>，推理整帧为 <code>768×512</code>。两者原图倍率相同；改变的是视野、上下文、latent 高度和输入重采样方式，不能写成道路像素尺度错配。</p>
<h2>已准备的改进入口</h2>
<ol><li><strong>监督保留原生教师。</strong> 新包提供 768 与 480 两档；A 从实拍直接取源，C/S 从原始生成取源。RGB 单次面积采样，来源域按全部非零滤波支持保守准入；K 任意相交即剔除。原始来源许可保留，不能因为是生成图就自动授予高频材质许可。</li>
<li><strong>先做冻结教师的 48 次推理上限对照。</strong> 6 个固定任务 × 2 个 seed（101/202）× 4 组：整帧 768 的 T0、整帧 768 无 LoRA、训练视野 768×272 的 T0、整帧 1152×768 的 T0。prompt、DDIM20、CFG3、eta0 固定，不改 W/K；CPU 检查 6 例 W 均完整落在固定裁片内。</li>
<li><strong>单列 VAE 控制。</strong> 同尺度合法实拍做 latent mode 的 encode/decode，与重采样参考比较；仅进入独立 VAE 控制。实际推理的条件 latent 使用 sample，mode 控制不等价于完整扩散链路。</li>
<li><strong>评价同时看颜色、频谱和结构。</strong> 评分入口报告合成核心 RGB/Lab 与同尺度多频段纹理能量偏差；边界与标线通过原生 PNG 并列复核，不由高频分数自动选优。两 seed、逐 case 保留回退；真实隐藏道路没有真值，两帧真实任务只量化合法可见道路的保持情况。</li></ol>
<p class="small muted">不同画幅的计算量不同，同一数字 seed 也不代表逐像素相同噪声；两 seed 只作稳健性诊断。单次面积采样只能减少重复滤波损失，不能恢复 480 网格已丢失的带宽。768 监督仍是待接入的二维接口，不含高分辨率三维相机、覆盖或 alpha 的验收结果。</p>
<h2>CPU 验证与执行边界</h2><div class="scroll"><table><thead><tr><th>检查</th><th>状态</th></tr></thead><tbody>__CHECKS__</tbody></table></div>
<p>远端工作目录：<code>/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_hf_cpu_20261119_r1</code>。GPU 入口需要显式 <code>--execute-gpu</code>；没有后台队列。当前无卡实例保持开机。</p>
<p class="note">这里完成的是 CPU 诊断与 GPU 前准备。真实 GPU 模型加载、历史重放一致性、高分辨率显存和图像质量尚未验证。后续若训练预计约一小时及以上，先做约十分钟真实 smoke，覆盖保存、评价、完整恢复与下一阶段入口；CPU 单测不能替代。</p>
<details><summary>证据与复现文件</summary><ul>__PROOF__</ul><p class="small">__PUBLICATION__</p></details>
<details><summary>实现依据与来源</summary><ul><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint">Diffusers 官方 Inpainting 文档</a>：局部裁片可提高目标区域的有效分辨率；此处固定视野控制与原生高分辨率控制分别执行。</li><li><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting">SD1.5 Inpainting 模型卡</a>：基础模型以 512 方图训练，不构成提高推理分辨率必然改善的保证。</li></ul><p class="small muted">旧报告和附件仅作为证据；当前执行范围来自本次用户请求。历史结果和其他对话的三维实验目录均未替换。</p></details>
<footer>ROAD-2D-TEACHER-HF-CPU-20261119 · 简体中文 · Asia/Singapore · human_verdict / problem_solved = null</footer></main></html>'''
    for token, value in {'__LEAD__':esc(d['lead']), '__ROWS__':rows, '__FIGURES__':figures, '__CHECKS__':checklist, '__PROOF__':proof, '__PUBLICATION__':esc(d.get('publication', '源码与结果归档将在交付核验后记录。'))}.items():
        content = content.replace(token, value)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'index.html').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
