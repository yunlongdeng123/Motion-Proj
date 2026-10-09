"""CPU 导出实际 P1 输入、训练快照和论文对照；视频等资产不进入 Git。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def video(folder: Path, output: Path) -> None:
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-framerate', '7', '-i',
                    str(folder / '%05d.png'), '-c:v', 'libx264', '-pix_fmt',
                    'yuv420p', '-crf', '20', str(output)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition-diagnostic', type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.is_relative_to(REPO):
        raise ValueError('审核媒体必须位于 Git 仓库外')
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for log in sorted((run / 'logs').glob('train_to_*.log')):
        for line in log.read_text().splitlines():
            try:
                item = json.loads(line)
                if item.get('event') == 'train_step':
                    rows[item['step']] = item
            except (json.JSONDecodeError, TypeError):
                pass
    logs = [rows[k] for k in sorted(rows)]
    tail = logs[-100:]
    speed = float(np.mean([x['wall_seconds'] for x in tail]))
    snapshot = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                'latest_step': logs[-1]['step'], 'training_budget': 100000,
                'mean_last100_seconds': speed,
                'remaining_training_days_estimate': (100000-logs[-1]['step'])*speed/86400,
                'losses_and_gradients_finite_last100': all(np.isfinite(x['loss']) and
                    all(y['finite'] for y in x['gradients'].values()) for x in tail),
                'unique_videos_seen': len({x['video_id'] for x in logs}),
                'controller_state': json.loads((run/'controller_state.json').read_text())}
    (output/'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, keys in zip(axes, [('diffusion',), ('flow_l1', 'ternary_warp')]):
        for key in keys:
            vals = np.array([x[key] for x in logs])
            ax.plot([x['step'] for x in logs], vals, alpha=.12)
            if len(vals) >= 50:
                ax.plot([x['step'] for x in logs][49:], np.convolve(vals, np.ones(50)/50, 'valid'), label=key)
        ax.set_xlabel('optimizer updates'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'loss.png', dpi=140); plt.close(fig)
    from motion_proj.worldsim_v81.p1_data import YouTubeVOSP1Dataset
    dataset = YouTubeVOSP1Dataset(Path('/root/autodl-tmp/data/worldsim_v81/youtube_vos_2019/train/JPEGImages'))
    profile = dataset.profile(); (output/'data_profile.json').write_text(json.dumps(profile, indent=2))
    samples = []
    for row in (logs[0], logs[1], logs[min(299, len(logs)-1)]):
        step = row['step']; sample = dataset[step-1]
        if sample['video_id'] != row['video_id'] or sample['clip_start'] != row['clip_start']:
            raise ValueError('预览未匹配真实训练样本')
        dest = output/'inputs'/f'step{step:06d}'; dest.mkdir(parents=True, exist_ok=True)
        sheets = Image.new('RGB', (5*256, 3*282), 'white'); draw = ImageDraw.Draw(sheets)
        for name in ('target_rgb', 'visible_rgb', 'hole_mask'):
            folder = dest/name; folder.mkdir(exist_ok=True)
            for i, tensor in enumerate(sample[name]):
                arr = tensor.numpy()
                if name == 'hole_mask':
                    arr = np.repeat(arr, 3, axis=0)*255
                else:
                    arr = (arr+1)*127.5
                im = Image.fromarray(np.round(arr.transpose(1,2,0)).clip(0,255).astype('uint8'))
                im.save(folder/f'{i:05d}.png')
                if i in (0,6,12,18,24):
                    x = (0,6,12,18,24).index(i)*256; y = ('target_rgb','visible_rgb','hole_mask').index(name)*282
                    draw.text((x+8,y+5), f'{name} f{i:02d}', fill='black'); sheets.paste(im,(x,y+26))
            if name != 'hole_mask': video(folder, dest/f'{name}.mp4')
        sheets.save(dest/'contact.jpg', quality=92)
        index, start = dataset.choice(step-1)
        with Image.open(dataset.videos[index][1][start]) as source:
            source.convert('RGB').save(dest/'source.jpg', quality=92)
            source_size = list(source.size)
        metadata = {'step':step,'video_id':sample['video_id'],'clip_start':sample['clip_start'],
                    'source_size':source_size,'file_indices':list(range(start,start+25)),
                    'source_filenames':[p.name for p in dataset.videos[index][1][start:start+25]]}
        (dest/'input.json').write_text(json.dumps(metadata, indent=2)); samples.append(metadata)
    cards = []
    for x in samples:
        base = f"inputs/step{x['step']:06d}"
        cards.append(f'''<article><h3>实际训练 step {x['step']} · {x['video_id']} · start {x['clip_start']}</h3>
        <p>原始尺寸 {x['source_size']}；下图依次是监督 RGB、模型可见 RGB、白色洞区。25 张原始 JPEG 按文件名连续取样，无重复帧。播放帧率 7 仅为当前导出设置，不代表原始采集帧率。</p>
        <div class="two"><figure><img src="{base}/source.jpg"><figcaption>原始首帧，展示裁剪前视野</figcaption></figure><figure><video controls loop muted src="{base}/target_rgb.mp4"></video><figcaption>256² 真实监督视频</figcaption></figure><figure><video controls loop muted src="{base}/visible_rgb.mp4"></video><figcaption>模型可见视频：左右各 84px 为洞</figcaption></figure></div>
        <img class="contact" src="{base}/contact.jpg"><p><a href="{base}/input.json">实际采样清单</a></p></article>''')
    validation_cards = []
    results = sorted(run.glob('validation*/step*/*/side_*/run.json'))
    results += sorted(run.glob('sampler_controlled/step*/*/run.json'))
    results += sorted(run.glob('reference_m4/validation*/step*/*/side_*/run.json'))
    results += sorted(run.glob('paper_bidirectional_m4/validation*/step*/*/side_*/run.json'))
    results += sorted(run.glob('paper_bidirectional_m4/validation*/step*/*/full_video/side_*/run.json'))
    for result in results:
        meta = json.loads(result.read_text()); rel = result.parent.relative_to(run)
        dest = output/rel; dest.mkdir(parents=True, exist_ok=True)
        for name in ('gt.mp4','visible.mp4','pred.mp4','comp.mp4','run.json'):
            shutil.copy2(result.parent/name, dest/name)
        labels = [('gt','真实完整 RGB'),('visible','实际可见输入'),('pred','原生模型输出'),('comp','中央可见区硬写回')]
        cells = ''.join(f'<figure><video controls muted loop src="{rel.as_posix()}/{name}.mp4"></video><figcaption>{label}</figcaption></figure>' for name,label in labels)
        phase_label = ('论文双向传播短预检' if rel.parts[0] == 'paper_bidirectional_m4' else
                       '公开参考父链训练' if rel.parts[0] == 'reference_m4' else '旧公开训练 / 诊断')
        validation_cards.append(f'<article><h3>{phase_label} · step {meta["checkpoint_step"]} · {meta["sequence_id"]} · {meta["mode"]}</h3><div class="four">{cells}</div><p>帧数与生成范围见来源记录。原生与写回分开看；硬写回中央正确不代表生成侧边正确。sampler_controlled 目录仅诊断，不进入正式论文指标。<a href="{rel.as_posix()}/run.json">来源记录</a></p></article>')
    gate_path = run/'quality_gates/step001000.json'
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        gate_summary = html.escape(gate.get('summary', '质量门决策见记录'))
        (output/'quality_gate_1000.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2))
        gate_note = f'<div class="panel warn"><b>旧All Frames 1000 步决策：{html.escape(gate["decision"])}</b><p>{gate_summary}</p><a href="quality_gate_1000.json">助手决策与证据</a></div>'
    else:
        gate_note = '<div class="panel warn">1000 步质量门尚未放行；先独立审核固定视频和采样诊断。</div>'
    diagnostic_note = ''
    diagnostic_path = run/'sampler_controlled/step001000/sampler_diagnostic.json'
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        shutil.copy2(diagnostic_path, output/'sampler_diagnostic.json')
        delta = diagnostic['pixel_mae_vs_literal']
        diagnostic_note = f'''<div class="panel"><h2>1000步：同条件采样诊断</h2>
        <p>固定同一 CLIP、RAFT、补全光流、传播潜变量与初始噪声，只切换采样。公开原路径重放与此前断点视频的25帧逐像素一致。</p>
        <p>无反演＋标准CFG相对公开路径的RGB平均差为 {delta['no-inverse-matched-cfg']:.5f}；保留反演首支＋标准CFG为 {delta['inverse-first-matched-cfg']:.5f}（RGB范围0–1）。差值不是恢复质量指标。</p>
        <p>独立助手看三模式原生与写回的全部25帧：三组都缺少地面、树木、人物延展，并保留接缝；无反演模式仍有天空碎片闪烁。当前单窗没有“改采样即可修复画质”的证据。公开反演的参数化风险仍未由此排除。</p>
        <p><a href="sampler_diagnostic.json">数值与实验边界</a> · <a href="sampler_controlled_assistant_review.json">独立目视审核</a> · <a href="smallmask_assistant_review.json">三段小mask审核</a> · <a href="inference_assistant_review.json">三段大mask及现成feedforward审核</a></p></div>'''
    phase_note = ''
    reference_audit_note = ''
    audit = run/'reference_direction_audit'
    if (audit/'diagnostic.json').is_file():
        dest = output/'reference_direction_audit'
        dest.mkdir(exist_ok=True)
        for name in ('diagnostic.json', 'contact.png'):
            shutil.copy2(audit/name, dest/name)
        reference_audit_note = '''<div class="panel warn"><h2>新的CPU反证：参考传播方向与过去证据</h2>
        <p>对固定公开模块做两帧水平平移：f00 的点在 x=4，f01 的点在 x=5。把未来 f01 拉回 f00，应出现在 x=4；公开一支却落在 x=6。反过来，只有过去 f00 可见时，公开两支都不能把内容送到末帧 f01。</p>
        <p>原因：两个分支沿同一条未来父链；其中一支还把 source→target 流用于需要 target→source 的拉取。当前 reference_m4 虽修了选帧和成对监督，仍继承此公开模块缺陷，不能称论文双向参考传播已正确复现。到1000步保存后保持暂停，先短预检。</p>
        <div class="diagram"><div class="box">可见参考latent<br>来源mask</div>→<div class="box">目标→过去 / 未来流<br>参考链组合</div>→<div class="box">一次拉取<br>原有细化与融合</div>→<div class="box">SVD条件<br>短训验证</div></div>
        <img src="reference_direction_audit/contact.png" alt="固定公开模块的方向与末帧证据CPU反证">
        <p>这是工程反证，不是新修复画质或论文指标。新 owned 实现对照论文 Eq.5–6，保留原网络参数结构；FCNet稀疏流序列、显存及生成质量仍需验证。<a href="reference_direction_audit/diagnostic.json">实际数值</a></p></div>'''
    phase = run/'reference_m4'
    if (phase/'controller_state.json').exists():
        phase_state = json.loads((phase/'controller_state.json').read_text())
        phase_rows = {}
        for log in sorted((phase/'logs').glob('*.log')):
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if row.get('event') == 'train_step': phase_rows[row['step']] = row
                except (json.JSONDecodeError, TypeError): pass
        phase_latest = phase_rows[max(phase_rows)] if phase_rows else None
        phase_summary = {'controller':phase_state,'latest_train':phase_latest,
                         'legacy_allframes_steps_not_counted':1000}
        (output/'reference_m4_status.json').write_text(json.dumps(phase_summary,ensure_ascii=False,indent=2))
        phase_note = f'''<div class="panel warn"><h2>保留的公开参考父链阶段</h2>
        <p>新训练完成 {phase_latest['step'] if phase_latest else 0} 步；当前状态 {html.escape(phase_state['status'])}。从原始权重新初始化，未恢复旧All Frames 1000断点；旧结果在下方供回溯。</p>
        <p>已修正选帧、成对RAFT及成对warp监督；传播仍调用固定公开模块。新的CPU反证显示其方向与过去参考覆盖有误。到1000步保留断点并暂停，双向传播修正先做短预检，不放行长训。主推理为Gaussian前向采样，公开literal路径保留为诊断。</p><a href="reference_m4_status.json">活动阶段快照</a></div>'''
    handoff = run/'bidirectional_handoff.json'
    if handoff.is_file():
        status = json.loads(handoff.read_text())
        shutil.copy2(handoff, output/'bidirectional_handoff.json')
        phase_note += f'''<div class="panel"><h2>论文双向候选：限定两步预检</h2>
        <p>状态：{html.escape(status['status'])}。串行测试新协议2步训练＋1个固定验证窗；没有放行1000或100K。原细化与融合参数结构保留，显存、梯度与生成效果需要实测。</p>
        <p><a href="bidirectional_handoff.json">队列与实际状态</a></p></div>'''
    candidate = run/'paper_bidirectional_m4'
    if (candidate/'controller_state.json').is_file():
        steps = {}
        for log in sorted((candidate/'logs').glob('train_to_*.log')):
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if row.get('event') == 'train_step': steps[row['step']] = row
                except (json.JSONDecodeError, TypeError): pass
        summary = {'controller': json.loads((candidate/'controller_state.json').read_text()),
                   'steps': [steps[k] for k in sorted(steps)], 'human_verdict': None,
                   'optimizer': 'Adam', 'weight_decay': 0,
                   'memory_adaptation': 'FCNet activation checkpoint + mask-weighted chunked ternary checkpoint'}
        (output/'paper_bidirectional_status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        phase_note += f'''<div class="panel"><h2>双向候选：实测与资源修复</h2>
        <p>已记录 {len(steps)} 个更新；当前预算 {summary['controller']['training_budget']}，状态 {html.escape(summary['controller']['status'])}。首次第一步通过，第二步因光流损失峰值超过3090显存而停止；保留第一步断点与失败日志后续跑。FCNet完整时轴采用激活重算，ternary loss分块按mask像素数加权；固定官方公式的CPU数值与梯度对照通过，不改变flow对、损失或采样预算。</p>
        <p>此阶段同时对齐论文Adam、零weight decay，旧AdamW断点不混用。因此它是论文配置修复预检，不能将画质差异单独归因传播方向。仍使用bf16和CPU卸载，未声称与两张A6000完全等价。两步画质也不能用来否定论文能力。</p>
        <p><a href="paper_bidirectional_status.json">逐步损失、梯度、显存与实际状态</a></p></div>'''
    independent_reports = sorted((run/'review').glob('assistant_p1*review_*.json'),
                                 key=lambda path: path.stat().st_mtime)
    source = next((path for path in reversed(independent_reports)
                   if 'windows' in json.loads(path.read_text())), None)
    if source:
        qa = json.loads(source.read_text())
        shutil.copy2(source, output/'independent_visual_review.json')
        phase_note += f'''<div class="panel warn"><h2>最新独立全帧审核</h2>
        <p>{html.escape(qa['assistant_verdict'])}。覆盖 {len(qa['windows'])} 窗，每窗四种输出、完整序列。逐窗帧数和结构/时序结论以审核记录为准；工程预检与画质收益分别判断。</p>
        <p>人工verdict留空。<a href="independent_visual_review.json">逐窗备注、合成契约与证据路径</a></p></div>'''
    condition_cards = []
    diagnostic = args.condition_diagnostic
    if diagnostic and (diagnostic/'diagnostic.json').is_file():
        dest = output/'condition_diagnostic'; dest.mkdir(exist_ok=True)
        shutil.copy2(diagnostic/'diagnostic.json', dest/'diagnostic.json')
        condition_cards.append('''<div class="panel"><h2>逐帧条件诊断：编码、传播、去噪</h2>
        <div class="diagram"><div class="box">逐帧可见 RGB<br>正常 / 重复首帧</div>→<div class="box">VAE 编码<br>中间解码</div>→<div class="box">双向传播<br>过去 / 未来 / 融合</div>→<div class="box">相同 CLIP 与噪声<br>SVD 原生输出</div></div>
        <p>两组共享首帧 CLIP、时间条件、增强噪声和初始 Gaussian latent。后续 RGB、RAFT 光流与参考选择分别重算。这里检查是否响应后续帧；差值不能当作质量收益或单组件因果证据。中间 latent 解码已对齐 VAE scaling factor。</p>
        <p>masked_visible 是黑色展示；model_visible 才是洞内归一化 0（中灰）的实际模型输入。独立审核看完整25帧，不以硬写回中心正确代替模型能力。<a href="condition_diagnostic/diagnostic.json">条件、噪声与逐帧响应记录</a></p></div>''')
        stages = [('model_visible_rgb','实际可见输入'),
                  ('vae_reconstruction_no_aug','可见输入 VAE 重建'),
                  ('pre_propagation_augmented','增强后，传播前'),
                  ('propagated_past','过去参考分支'),
                  ('propagated_future','未来参考分支'),
                  ('propagated_fused','传播融合条件'),
                  ('native_final','最终原生输出')]
        for stage, label in stages:
            cells = []
            for branch, title in [('normal_visible_25','正常后续帧'),
                                  ('repeat_first_after_f0','后24帧重复首帧')]:
                folder = dest/branch; folder.mkdir(exist_ok=True)
                video(diagnostic/branch/stage, folder/f'{stage}.mp4')
                shutil.copy2(diagnostic/branch/f'{stage}_contact.png',
                             folder/f'{stage}_contact.png')
                rel = f'condition_diagnostic/{branch}/{stage}'
                cells.append(f'<figure><video controls muted loop src="{rel}.mp4"></video><figcaption>{title} · {label}</figcaption><details><summary>完整25帧联系图</summary><img src="{rel}_contact.png"></details></figure>')
            condition_cards.append(f'<article><h3>条件路径 · {label}</h3><div class="pair">'+''.join(cells)+'</div></article>')
    capacity = run/'capacity_fixed_clip_32'
    if (capacity/'run.json').is_file():
        meta = json.loads((capacity/'run.json').read_text())
        dest = output/'capacity_fixed_clip_32'; dest.mkdir(exist_ok=True)
        shutil.copy2(capacity/'run.json', dest/'run.json')
        condition_cards.append(f'''<div class="panel warn"><h2>单训练片段容量诊断：额外32步</h2>
        <p>状态 {html.escape(meta['status'])}；片段 {html.escape(meta['video_id'])}，start {meta['clip_start']}。这32步属于独立诊断，不计入正式训练进度，不是留出评测。Teacher使用带噪真实latent、GT光流和完整首帧CLIP；QUERY只用可见输入、纯Gaussian初始化。</p>
        <p>Teacher改善不能证明纯噪声生成会恢复输入，更不能证明泛化。两种任务与前后权重分别展示。<a href="capacity_fixed_clip_32/run.json">固定sigma、来源与实测</a></p></div>''')
        for name in ('teacher_before','teacher_after'):
            if (capacity/name/'contact.jpg').is_file():
                folder = dest/name; folder.mkdir(exist_ok=True)
                shutil.copy2(capacity/name/'contact.jpg', folder/'contact.jpg')
                condition_cards.append(f'<article><h3>{name} · 带噪真值去噪，仅容量诊断</h3><img src="capacity_fixed_clip_32/{name}/contact.jpg"></article>')
        for name in ('query_before','query_after'):
            if not (capacity/name/'comp.mp4').is_file(): continue
            folder = dest/name; folder.mkdir(exist_ok=True)
            cells = []
            for key, label in [('gt','真实监督'),('visible','实际可见输入'),('pred','原生QUERY'),('comp','中央硬写回')]:
                shutil.copy2(capacity/name/f'{key}.mp4', folder/f'{key}.mp4')
                cells.append(f'<figure><video controls muted loop src="capacity_fixed_clip_32/{name}/{key}.mp4"></video><figcaption>{label}</figcaption></figure>')
            condition_cards.append(f'<article><h3>固定训练片段 · {name} · 同seed与采样步数</h3><div class="four">'+''.join(cells)+'</div></article>')
    for pattern, label in [('assistant_p1_condition_review_*.json','条件路径独立图像审核'),
                           ('assistant_p1_capacity_review_*.json','固定片段独立图像审核')]:
        reports = sorted((run/'review').glob(pattern))
        if reports:
            name = pattern.split('_*.')[0]+'.json'
            shutil.copy2(reports[-1], output/name)
            condition_cards.append(f'<p><a href="{name}">{label}</a> · human_verdict 留空。</p>')
    comparisons = [
        ('视频与尺寸','25 帧 / 256²','JPEG 连续窗口，缩放中心裁剪','相同；1951 个有效视频、19313 个窗口','基本对齐；100K 是重复采样的更新预算'),
        ('外绘 mask','水平总宽 .25 / .66（评测）','训练双侧各 .33','训练每侧 84px；推理两倍率','对齐公开 mask；当前没有驾驶 DELETE 造数'),
        ('传播路径','m=4，最近过去与未来参考','train走All Frames；test两个分支都沿未来父链','reference_m4修正选帧与成对监督，但继承公开传播缺陷','CPU平移已证方向错误与末帧失证据；先短预检，不放行长训'),
        ('训练条件','传播条件 + 扩散训练','完整 RGB 算 RAFT / 首帧 CLIP；masked RGB 编码','沿公开训练路径；QUERY 仅可见输入','训练／QUERY 条件分布有风险，非新增输入泄漏'),
        ('参数范围','冻结空间层，训练时序层','FCNet + propagation + temporal transformer','相同冻结范围；冻结梯度为 0','基本对齐；非全量 U-Net 微调'),
        ('优化器','Adam / lr 1e−5 / 100K','AdamW / wd .01 / batch per GPU 1','双向候选Adam/wd0；旧协议AdamW；batch1','按论文文字对齐候选；双卡有效 batch 未确认'),
        ('硬件与精度','2×A6000','fp16 配置','单 3090，bf16；冻结编码器 CPU 卸载','资源适配；速度和精度不能宣称完全一致'),
        ('推理采样','Gaussian 初始化，前向去噪','额外 inversion + B1→B2','旧literal保留；新阶段主模式为Gaussian feedforward','同条件3模式未修好画质；公开inverse风险仍记录'),
        ('正式评测','DAVIS90 + 附录 YT60 / 四指标','未公开完整指标实现，默认可用全长滑窗','full-video入口已跑通36帧；历史前25帧仅诊断；指标入口前16帧','尚未跑正式整套；预处理/FVD口径未完全确认，protocol_verified=false'),
    ]
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(y)}</td>' for y in x)+'</tr>' for x in comparisons)
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v8.1 P1 实时进度与论文对照</title>
    <style>body{font:16px/1.65 system-ui;background:#f2f5f8;color:#172536;margin:0}main{max-width:1400px;margin:auto;padding:26px}h1,h2,h3{line-height:1.3}article,.panel{background:white;border:1px solid #d6e0e9;border-radius:10px;padding:20px;margin:20px 0}.warn{background:#fff4dc;border-left:5px solid #c1780a}.diagram{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.box{background:#e9f0fa;border:1px solid #8ba4c5;border-radius:8px;padding:10px;text-align:center}.four,.two{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.two{grid-template-columns:repeat(3,minmax(0,1fr))}figure{margin:0}video,img{max-width:100%;width:100%;background:#111}figcaption{padding:8px 0}.contact{margin-top:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d3dce5;padding:10px;vertical-align:top;text-align:left}th{background:#e8eef5}a{color:#145cad}.scroll{overflow:auto}@media(max-width:850px){.four,.two{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main>
    <h1>v8.1 P1：当前进度、实际数据与原文对照</h1>'''
    if (capacity/'run.json').is_file() and meta['status'] == 'complete':
        document += '''<div class="panel warn"><h2>当前结论：P1仍未通过</h2><p>新双向候选正式训练100步，三项条件/容量诊断已完成。逐帧可见信息在VAE与融合条件仍保留，也会影响最终输出，但native仍丢失结构。固定训练片段额外32步只有弱数值改善，QUERY仍为灰块，不能算学会。没有排队1000或100K。</p><p>本次32步诊断保留了日志、图像、seed和源100断点，未保存末步权重；入口已补保存供后续运行，未为补档重复GPU。后续应隔离训练/QUERY的CLIP与flow条件差距，再决定有界预算；不凭32步否定模型容量。</p></div>'''
    document += ''.join(condition_cards) + reference_audit_note + phase_note + f'''<div class="panel warn"><b>旧公开训练快照：{snapshot['observed_at_utc']}（UTC） · step {snapshot['latest_step']} / 100,000</b>
    <p>最近 100 步均值 {speed:.2f} 秒；按同速估算剩余纯训练 {snapshot['remaining_training_days_estimate']:.2f} 天，另加保存与验证。数值/梯度正常 ≠ 生成质量达标。这里展示的最新生成断点与训练步数分开标注。</p>
    <p>这是已停止的旧训练速度估算，并非正在运行七天训练。旧1000质量门hold；新双向候选限定到100后完成短窗及36帧检查，接着做逐帧条件与固定片段容量诊断。没有排队1000或100K，不以有限loss放行长训。</p></div>
    <div class="panel"><h2>实际组件与监督</h2><div class="diagram"><div class="box">25 帧真实 RGB<br>双侧洞 mask</div>→<div class="box">冻结 RAFT / VAE / CLIP<br>光流与条件编码</div>→<div class="box">可训练 FCNet<br>潜变量传播 / 对齐</div>→<div class="box">SVD 时序层<br>空间层冻结</div>→<div class="box">VAE 解码<br>外绘视频</div></div>
    <p>BUILD：真实完整 RGB 提供扩散目标、光流监督和公开训练中的首帧 CLIP。QUERY：只提供 masked RGB，隐藏 RGB 不进入条件。训练同时优化 diffusion、flow L1、ternary warp。</p>
    <p>这是通用视频外绘 P1，尚未进入驾驶 DELETE 的 P2，也未加入创新。输入检查与短训复现必须先过关。</p></div>
    <div class="panel"><h2>真实训练进度</h2><p>有效训练视频 {profile['videos']}；短视频排除 {profile['rejected_short']}；可用连续窗口 {profile['available_windows']}；已接触 {snapshot['unique_videos_seen']} 个不同视频。</p>
    <img src="loss.png"><p>淡线是逐步损失，实线是 50 步均值。各步来自不同视频，曲线不能代替固定输入的生成质量对照。</p><p>展示三段的 JPEG 文件名每 5 递增；“连续25张”指包内相邻文件，并非原视频逐帧。正式 YouTube 评测另用全帧包。作者实际训练采样率尚未确认，需核对这一时序分布差距，不能凭相同尺寸/mask认定协议完全一致。</p><p><a href="snapshot.json">运行快照</a> · <a href="data_profile.json">数据统计</a></p></div>
    <div class="panel"><h2>论文 / 固定公开源码 / 当前实施</h2><p>来源：<a href="https://arxiv.org/html/2604.14648">论文 §4、§5.1、附录 D.1</a>；<a href="https://github.com/InSeokJeon/Seen_to_Scene/tree/2a9dfc9888e44c7fd00b08af41ef967ae46b6323">固定公开代码</a>。表中“差距”不等于已证明它造成失败。</p>
    <div class="scroll"><table><tr><th>项目</th><th>论文</th><th>公开代码</th><th>当前</th><th>结论</th></tr>'''+table+'''</table></div>
    <p>inversion 的 UNet 输出被当作 epsilon 使用，而 SVD scheduler 为 v_prediction；其 B1→B2 广播还会破坏标准 CFG 的同 latent 配对。现成feedforward与literal另有洞区填值/RNG差异，不可直接因果比较；下方sampler_controlled诊断复用完全相同的条件与初始噪声。</p></div>
    <h2>实际训练输入（按真实日志固定取 step 1、2、300）</h2>'''+''.join(cards)+'''<h2>已完成的生成验证</h2>
    <div class="panel warn">历史第 2 步审核：饱和色块、场景结构丢失。第1000步已能辨认天空、建筑或水下动物，但有形变、重复、近乎冻结和边界断裂，尚未达到可用质量。原生与写回全部25帧由独立助手审核，human verdict留空。</div>'''+gate_note+diagnostic_note+''.join(validation_cards)+'''
    <div class="panel"><h2>短周期决策</h2><ol><li>第 1,000 步：完成三固定验证例、两倍率的原生/写回输出；同断点做 feedforward 诊断，独立 subagent 看完整视频。</li><li>先确认采样、条件和传播协议；若仍是色块，保持断点并查明问题，不无条件堆训练步数。</li><li>只有工程链路可信且生成开始恢复场景结构，才放行到第 5,000 步再审核。正式测试 ID 不用于调参。</li></ol>
    <p>尚无正式 PSNR / SSIM / LPIPS / FVD。论文目标仍待实际计算；当前 assistant_verdict 不代替 human_verdict。</p></div></main>
    <script>document.querySelectorAll('article').forEach(a=>{const vs=[...a.querySelectorAll('video')];if(vs.length<2)return;const b=document.createElement('button');b.textContent='同步播放本组';b.onclick=()=>vs.forEach(v=>{v.currentTime=0;v.play()});a.insertBefore(b,a.children[1]);});</script></html>'''
    document = document.replace('</style>', '.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}</style>')
    (output/'index.html').write_text(document, encoding='utf-8')
    print(json.dumps({'review':str(output/'index.html'),'snapshot':snapshot,'input_clips':len(samples),'validation_clips':len(validation_cards)}))


if __name__ == '__main__':
    main()
