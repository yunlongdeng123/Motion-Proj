#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import shutil
import imageio_ffmpeg

base = Path('/root/autodl-tmp/drivingeditor_repro')
run = base / 'runs/0017085_demo_000_020_dynamic_30000'
source = run / 'eval/train_30000_render'
output = base / 'reports/paper50k_comparison/history'
output.mkdir(parents=True, exist_ok=True)
metrics = json.loads((source / 'metrics.json').read_text())
frames = sorted(source.glob('*.png'))
assert len(frames) == 21 and all(path.stem == f'{i:03d}' for i, path in enumerate(frames))
target = output / 'baseline30k.mp4'
if not target.exists():
    temporary = output / 'baseline30k.partial.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '10',
                    '-start_number', '0', '-i', str(source / '%03d.png'), '-frames:v', '21',
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-threads', '2',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], check=True)
    temporary.replace(target)
reader = imageio_ffmpeg.read_frames(str(target), pix_fmt='rgb24')
decoded = next(reader)
decoded['decoded_frames'] = sum(1 for frame in reader)
assert decoded['decoded_frames'] == 21 and decoded['fps'] == 10
(output / 'baseline30k_video_validation.json').write_text(json.dumps(decoded, indent=2) + '\n')
shutil.copy2(source / '010.png', output / 'baseline30k_frame010.png')
shutil.copy2(source / 'metrics.json', output / 'baseline30k_metrics.json')
document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>官网场景 · 30k 历史代码基线</title><style>body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#101c2b;color:#e6eff9;margin:0}main{max-width:1100px;margin:auto;padding:32px}a{color:#91c6ff}video,img{width:100%;border-radius:10px;background:#000}.note{padding:14px;border-left:4px solid #d2a05a;background:#283244}.flow{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:24px 0}.flow b{padding:12px;border:1px solid #546c83;border-radius:8px}.labels{display:grid;grid-template-columns:1fr 1fr;gap:8px}code{overflow-wrap:anywhere}</style>
<main><a href="../index.html">← 50k / 每 10k 比较页</a><h1>官网银色轿车场景：30k 历史代码基线</h1>
<p>21 帧 · FRONT · 0–20 帧 · 官方代码默认 30,000 步 · seed 0 · 缩放 4</p>
<div class="flow" role="img" aria-label="数据经过双分支高斯训练生成对比视频"><b>Waymo RGB + LiDAR</b> → <b>静态 / 动态高斯 + 天空</b> → <b>30k 训练</b> → <b>重建与删除视频</b></div>
<p class="note">这是已经完成的旧代码基线，使用 21 帧及不同损失；新实验使用 49 帧、论文方法和 50k 周期，不能将此视频作为新实验的 30k 节点。删除结果仍有暗斑，不能据此宣称阴影处理或论文编辑质量已达标。</p>
<div class="labels"><span>左上：真实图像 GT</span><span>右上：组合重建</span><span>左下：移除动态高斯后的背景</span><span>右下：动态前景（黑背景）</span></div>
<video controls loop muted playsinline preload="metadata" poster="baseline30k_frame010.png" src="baseline30k.mp4"></video>
<p>统一按 Waymo 约 10 Hz 的原始采样速度播放，21 帧 / 2.1 秒。模型内部时间间隔 0.02 与物理播放时间分开记录。</p>
<p>全图训练视角均值：PSNR {PSNR} dB · SSIM {SSIM} · LPIPS {LPIPS}。此处没有留出集，也不是论文四场景均值。</p>
<p><a href="baseline30k_metrics.json">指标原始记录</a> · <a href="baseline30k_video_validation.json">编码与帧数验证</a></p></main></html>'''
document = document.replace('{PSNR}', f"{metrics['psnr']:.4f}").replace('{SSIM}', f"{metrics['ssim']:.4f}").replace('{LPIPS}', f"{metrics['lpips']:.4f}")
(output / 'baseline30k.html').write_text(document, encoding='utf-8')
print(json.dumps({'output': str(output), 'validation': decoded, 'metrics': metrics}, indent=2))
