"""利用已同步产物更新条件诊断页，不重新编码历史视频或启动GPU。"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.worldsim_v81.build_p1_progress_review import condition_gap_panel


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def review_note(folder: Path, relative: str) -> str:
    path = folder/'assistant_review.json'
    if not path.is_file():
        return '<p>独立图像审核尚未完成。</p>'
    report=read(path)
    verdict = report.get('assistant_verdict',report.get('verdict',''))
    return f'<p>{html.escape(str(verdict))}</p><p><a href="{relative}/assistant_review.json">完整独立审核与限制</a></p>'


def step1000_reviews(folder: Path, relative: str) -> tuple[int, str, bool]:
    links = []
    verdicts = []
    for name in ('assistant_review_skateboard.json', 'assistant_review_animals.json'):
        path = folder/name
        if not path.is_file():
            continue
        report = read(path)
        scope = report.get('scope')
        step = report.get('checkpoint_step', scope.get('checkpoint_step') if isinstance(scope, dict) else None)
        if step != 1000:
            raise ValueError(f'1000步审核文件断点不匹配: {path}')
        cases = report.get('cases', [])
        ids = sorted({case['sequence_id'] for case in cases if 'sequence_id' in case})
        if not ids and report.get('sequence_id'):
            ids = [report['sequence_id']]
        label = f'{" / ".join(ids)} · {len(cases)}窗'
        verdict = report.get('assistant_verdict', report.get('verdict'))
        verdicts.append(str(verdict or ''))
        suffix = f'：{html.escape(str(verdict))}' if verdict else '：结论见原文'
        links.append(f'<a href="{relative}/{name}">{html.escape(label)}独立审核</a>{suffix}')
    if not links:
        return 0, '两份独立图像审核待完成（0/2）', False
    pending = '；另一份待完成' if len(links) == 1 else ''
    return len(links), f'{len(links)}/2份审核已归档：{" · ".join(links)}{pending}', (
        len(links) == 2 and all('hold' in verdict.lower() for verdict in verdicts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='已同步媒体的Git仓库外目录')
    args = parser.parse_args()
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError('媒体和审核页不进入Git')
    gap = read(output/'condition_gap_teacher/diagnostic.json')
    if gap['status'] != 'complete':
        raise ValueError('只导出实际完成的诊断，不把CPU计划当实测')
    continuation_path = output/'bounded1000_state.json'
    continuation = read(continuation_path) if continuation_path.is_file() else None
    if continuation and (continuation['source_step'] != 500 or continuation['training_budget'] != 1000):
        raise ValueError('正式续训预算与500→1000记录不一致')
    step1000_done = bool(continuation and continuation['new_1000_validation_available'])
    bounded2000_path = output/'bounded2000_state.json'
    bounded2000 = read(bounded2000_path) if bounded2000_path.is_file() else None
    if bounded2000 and (bounded2000.get('source_step') != 1000
                        or bounded2000.get('training_budget', bounded2000.get('target_step')) != 2000
                        or bounded2000.get('propagation_protocol') != 'paper-bidirectional-m4'
                        or Path(bounded2000.get('source_checkpoint', '')).name != 'p1-checkpoint-001000.pt'):
        raise ValueError('正式1000→2000阶段快照的源断点、目标或协议不匹配')
    header = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P1 条件与原始SVD对照</title>
    <style>body{font:16px/1.65 system-ui;background:#eef2f6;color:#182a3a}main{max-width:1400px;margin:auto;padding:24px}.panel,article{background:white;border:1px solid #ccd7e3;padding:20px;margin:20px 0;border-radius:10px}.warn{background:#fff5de}.diagram{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.box{padding:12px;background:#e6effb;border:1px solid #96abc4}.four,.five{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.five{grid-template-columns:repeat(5,minmax(0,1fr))}figure{margin:0}video,img{width:100%;max-width:100%}td,th{border:1px solid #ccd7e3;padding:8px}table{border-collapse:collapse}a{color:#145dad}@media(max-width:800px){.four,.five{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main>
    <h1>P1：固定权重条件诊断与原始SVD对照</h1><p><a href="index.html">返回完整进度、历史输出与数据页</a></p>
    <div class="panel warn"><b>P1正式指标尚未计算：正式训练与诊断分别报告。</b><p>旧条件隔离使用100步权重且更新0；正式500步验证质量hold，当前验证状态见下方。固定片段64步控制另列。原始SVD生成不是外扩任务或论文指标。完整GT条件只作BUILD诊断，不能用于正式QUERY；人工verdict留空。</p></div>'''
    sections = [header, condition_gap_panel(gap),
                review_note(output/'condition_gap_teacher', 'condition_gap_teacher')]
    bounded2000_summary = ''
    bounded2000_panel = None
    if bounded2000:
        processes = bounded2000.get('processes', {})
        process_rows = []
        for key, label in (('controller_pid', '控制器'), ('child_pid', '训练子进程')):
            entry = processes.get(key, bounded2000.get(key))
            if isinstance(entry, dict):
                pid, alive = entry.get('pid'), entry.get('alive')
                command = entry.get('actual_command')
            else:
                pid, alive = entry, None
                command = bounded2000.get('command') if key == 'child_pid' else None
            if pid is not None or command:
                shown = f'PID {html.escape(str(pid))}；存活={html.escape(str(alive))}'
                if command:
                    shown += f'；实际命令：<code>{html.escape(str(command))}</code>'
                process_rows.append(f'<li>{label}：{shown}</li>')
        observed = bounded2000.get('observed_train_step', bounded2000.get('training_step'))
        observed_at = bounded2000.get('observed_at_utc', bounded2000.get('updated_at_utc', '未记录'))
        status = html.escape(str(bounded2000.get('status', '未记录')))
        bounded2000_summary = f'快照 {html.escape(str(observed_at))}：记录步数 {html.escape(str(observed))}，状态 {status}'
        process_html = f'<ul>{"".join(process_rows)}</ul>' if process_rows else '<p>快照未记录可核对的PID或实际命令。</p>'
        bounded2000_panel = f'''<div class="panel" id="bounded2000-snapshot"><h2>同协议有界阶段：正式1000 → 目标2000步</h2><p>本阶段最多新增1000次正式更新；从正式1000断点恢复模型、Adam与RNG，数据、模块、seed、mask和传播协议不改。teacher/oracle诊断权重不进入正式训练，不自动延伸到100K。正式1000六窗质量hold是继续有界研究的起点，不是2000步质量放行。</p><p>{bounded2000_summary}。<b>这是采集时快照，不代表实时步数或进程现状。</b></p>{process_html}<p><a href="bounded2000_state.json">实际状态、PID与命令原文</a></p></div>'''
    if continuation and not step1000_done:
        sections.insert(1,f'''<div class="panel"><h2>正式同协议学习曲线：500 → 总计1000步</h2><p>仅新增500次更新，恢复正式500步模型与Adam状态，传播协议 {html.escape(continuation['propagation_protocol'])}，推理模式 {html.escape(continuation['inference_mode'])}。截至快照 {html.escape(continuation['observed_at_utc'])} 记录到 {continuation['observed_train_step']} 步；这是快照，不代表实时步数。1000步结果尚未生成；到达后按固定三例×两倍率共六窗逐帧审核与质量门处理，不能以这三项64步单片段控制代替正式评价。</p><a href="bounded1000_state.json">正式续训实际命令与状态快照</a></div>''')
    state_path=output/'bounded500_state.json'
    if state_path.is_file():
        state=read(state_path)
        sections.insert(2,f'''<div class="panel"><h2>已完成阶段：100 → 总计500步</h2><p>历史快照 {html.escape(state['observed_at_utc'])}；记录 {state['observed_train_step']} 步，状态 {html.escape(state['status'])}。恢复模型、Adam状态和随机状态；数据、mask、网络、学习率与参数范围不变。三例×两倍率的500步验证和独立全帧审核已完成，质量仍为hold；固定片段控制不计入正式训练。</p><a href="bounded500_state.json">历史PID、命令与预算快照</a></div>''')
    precision_path=output/'unet_precision_full_audit.json'
    if precision_path.is_file():
        precision=read(precision_path)
        if precision.get('all_f32_values_equal') is True and precision.get('all_shapes_equal') is True:
            sections.insert(2,f'''<div class="panel"><h2>UNet 初始化来源复核：数值相同</h2><p>官方训练默认读取完整精度文件，我们读取fp16文件再转FP32。独立下载并校验官方完整版后，CPU逐值比较全部 {precision['total_tensors']} 个张量、{precision['total_values']:,} 个值，实际FP32初始化完全相等。源文件variant差异没有改变本轮UNet初值，不为此另起训练。</p><p>此检查不覆盖AMP、优化器或训练轨迹。<a href="unet_precision_full_audit.json">完整比较记录</a> · <a href="unet_full_download_state.json">下载来源与校验</a> · <a href="unet_initialization_audit.json">实际可训练参数范围</a></p></div>''')
    validation=output/'paper_bidirectional_m4/validation/step000500'
    completed=sorted(path for path in validation.glob('*/side_*/run.json')
                     if read(path).get('status')=='complete')
    if completed:
        sections.append('''<div class="panel"><h2>500步固定验证：原生与写回分开看</h2><p>原固定三例×两倍率，各25帧，seed2026/25步；同协议灰洞RAFT与可见首帧CLIP，不更改数据、mask或采样。可见中心由真实RGB硬写回，不能算作模型原生保持能力。以下只展示已完成窗口，不是正式DAVIS/YT指标。</p><div class="diagram"><div class="box">逐帧可见RGB</div>→<div class="box">光流补全 / 双向参考传播</div>→<div class="box">500步SVD去噪</div>→<div class="box">原生视频 / 可见中心硬写回</div></div>''')
        summary_path=output/'bounded500_training_summary.json'
        if summary_path.is_file():
            summary=read(summary_path)
            sections.append(f'''<p>同协议追加{summary['added_optimizer_updates']}步；涉及{summary['unique_training_videos_in_added_updates']}个训练视频，平均{summary['mean_step_seconds']:.2f}秒/步，记录峰值{summary['peak_logged_gpu_allocated_gib']:.2f}GiB。所有新增步损失和梯度有限；随机训练步的曲线不能替代生成质量审核。</p><img src="bounded500_loss.png" alt="500步短训练损失曲线"><p><a href="bounded500_training_summary.json">训练过程摘要</a></p>''')
        for path in completed:
            meta=read(path)
            relative=path.parent.relative_to(output).as_posix()
            sections.append(f'<article><h3>{html.escape(meta["sequence_id"])} · 每侧{meta["side_ratio_each"]:g} · 实际{meta["num_frames"]}帧</h3><div class="four">')
            for name,title in (('gt','真实视频'),('visible','可见输入'),('pred','500步原生输出'),('comp','500步可见中心硬写回')):
                sections.append(f'<figure><video controls preload="metadata" src="{relative}/{name}.mp4"></video><figcaption>{title}</figcaption></figure>')
            sections.append(f'</div><a href="{relative}/run.json">输入角色、参考与运行配置</a></article>')
        previous='paper_bidirectional_m4/validation/step000100/00f88c4f0a/side_0.33'
        if (output/previous/'run.json').is_file():
            sections.append(f'''<article><h3>同一00f88c4f0a/每侧.33的100步历史对照</h3><p>100步只有这一匹配短窗，不能声称其余五窗都完成100→500配对。</p><div class="four"><figure><video controls preload="metadata" src="{previous}/pred.mp4"></video><figcaption>100步原生</figcaption></figure><figure><video controls preload="metadata" src="{previous}/comp.mp4"></video><figcaption>100步硬写回</figcaption></figure></div></article>''')
        sections.extend([review_note(validation,'paper_bidirectional_m4/validation/step000500'),'</div>'])
    review_count, review_detail, review_hold = 0, '', False
    step1000 = output/'paper_bidirectional_m4/validation/step001000'
    if step1000_done:
        curve = read(step1000/'learning_curve.json')
        if (continuation['observed_train_step'] != 1000
                or continuation.get('complete_validation_windows') != 6
                or curve['source_step'] != 500 or curve['final_step'] != 1000
                or curve['actual_added_updates'] != 500):
            raise ValueError('1000步完成状态、训练过程与六窗数量不一致')
        windows = sorted(step1000.glob('*/side_*/run.json'))
        if len(windows) != 6 or {path.parent.relative_to(step1000) for path in windows} != {
                path.parent.relative_to(validation) for path in completed}:
            raise ValueError('1000步六窗与500步固定验证不匹配')
        review_count, review_detail, review_hold = step1000_reviews(
            step1000, 'paper_bidirectional_m4/validation/step001000')
        quality = '两份独立审核均hold' if review_hold else '质量结论以独立审核为准'
        sections.insert(1,f'''<div class="panel"><h2>正式同协议学习曲线：500 → 总计1000步</h2><p>已从正式500步断点追加{curve['actual_added_updates']}次更新，涉及{curve['unique_training_video_ids']}个训练视频，平均{curve['mean_seconds_per_step']:.2f}秒/步；记录峰值{curve['gpu_peak_allocated_gib']:.2f}GiB，梯度有限且冻结梯度为0。原固定三例×两倍率六窗已完成生成；{quality}，不能从损失或工程完成推定通过。</p><div class="diagram"><div class="box">真实25帧 / 可见RGB</div>→<div class="box">双向参考传播 + SVD<br>正式500 → 1000步</div>→<div class="box">固定六窗原生视频</div>→<div class="box">独立全帧审核 / 有界定位</div></div><p>{review_detail}</p><p><a href="#validation-step1000">查看同步五视频六窗</a> · <a href="paper_bidirectional_m4/validation/step001000/learning_curve.json">500次更新实测</a> · <a href="bounded1000_state.json">完成状态</a></p></div>''')
        sections.append('''<div class="panel" id="validation-step1000"><h2>1000步六窗：匹配500步的原生学习曲线</h2><p>每窗同一GT/可见输入，并排同步播放500与1000步原生预测及1000步真实中心硬写回。只有原生两列可用于观察模型变化；合成中心不能算生成成功。以下为25帧短窗诊断，不是正式DAVIS/YT指标。</p>''')
        for path in windows:
            meta = read(path)
            relative = path.parent.relative_to(output).as_posix()
            previous_relative = (validation/path.parent.relative_to(step1000)).relative_to(output).as_posix()
            if meta.get('status') != 'complete' or meta.get('checkpoint_step') != 1000 or meta.get('num_frames') != 25:
                raise ValueError(f'1000步窗口未完成或元数据不匹配: {path}')
            videos = ((f'{relative}/gt.mp4', '真实视频'),
                      (f'{relative}/visible.mp4', '实际可见输入'),
                      (f'{previous_relative}/pred.mp4', '500步原生输出'),
                      (f'{relative}/pred.mp4', '1000步原生输出'),
                      (f'{relative}/comp.mp4', '1000步可见中心硬写回'))
            missing = [name for name, _ in videos if not (output/name).is_file()]
            if missing:
                raise FileNotFoundError(f'1000步同步视频缺失: {missing}')
            sections.append(f'<article><h3>{html.escape(meta["sequence_id"])} · 每侧{meta["side_ratio_each"]:g} · 25帧</h3><div class="five">')
            for name, label in videos:
                sections.append(f'<figure><video controls preload="metadata" src="{name}"></video><figcaption>{label}</figcaption></figure>')
            sections.append(f'</div><p><a href="{relative}/run.json">1000步窗口配置</a> · <a href="{previous_relative}/run.json">500步匹配配置</a></p></article>')
        sections.extend([f'<p>{review_detail}</p>', '</div>'])
    oracle = output/'full_latent_oracle_step1000'
    oracle_done = (oracle/'run.json').is_file()
    oracle_status = ''
    if oracle_done:
        diagnostic = read(oracle/'run.json')
        delivered = read(oracle/'delivery_check.json')
        shared = diagnostic['pairing_exact']
        required = ('initial_latent', 'condition_vae_input', 'fused_condition', 'clip',
                    'time_ids', 'raw_flow_forward', 'raw_flow_backward', 'unconditional_zero_both')
        if (not step1000_done or diagnostic.get('status') != 'complete'
                or diagnostic.get('checkpoint_step') != 1000
                or diagnostic.get('optimizer_updates') != 0
                or not diagnostic.get('diagnostic_only')
                or not all(shared[key] for key in required)
                or not delivered['normal_replay_all25_exact']
                or delivered['new_videos_fully_decoded'] != 8):
            raise ValueError('非法GT条件诊断的完成、配对或重放证据不一致')
        review_path = oracle/'assistant_review.json'
        if review_path.is_file():
            review = read(review_path)
            verdict = review.get('assistant_verdict', review.get('verdict', '结论见审核原文'))
            oracle_status = f'独立QA已归档：{html.escape(str(verdict))}；<a href="full_latent_oracle_step1000/assistant_review.json">审核与限制</a>'
        else:
            oracle_status = '独立全帧QA待完成；当前仅确认诊断运行及文件完整'
        sections.insert(2,f'''<div class="panel warn"><b>非法GT条件仅用于定位，不能算论文复现或方法收益。</b><p>正式1000步六窗独立审核{'仍hold' if review_hold else '结论见审核'}；额外的完整GT潜变量oracle已完成0次训练更新。普通支25帧与正式1000步原生逐像素一致；{oracle_status}。<a href="#full-latent-oracle">查看两支同步视频与输入边界</a></p></div>''')
        videos = ((f'full_latent_oracle_step1000/{branch}/{kind}.mp4', label)
                  for branch, kind, label in (
                      ('normal', 'pred', '合法可见条件：原生'),
                      ('normal', 'comp', '合法条件：真实中心硬写回'),
                      ('illegal_oracle', 'pred', '非法完整GT条件：原生'),
                      ('illegal_oracle', 'comp', '非法条件：真实中心硬写回'),
                      ('normal', 'gt', '真实视频，仅作对照'),
                      ('normal', 'visible', '合法可见输入')))
        videos = tuple(videos)
        missing = [name for name, _ in videos if not (output/name).is_file()]
        reconstruction = oracle/'gt_vae_reconstruction_contact.png'
        if missing or not reconstruction.is_file():
            raise FileNotFoundError(f'非法GT条件诊断媒体缺失: {missing}, VAE图存在={reconstruction.is_file()}')
        sections.append('''<div class="panel warn" id="full-latent-oracle"><h2>非法GT潜变量oracle：仅定位条件路径</h2><p><b>完整GT读取了被遮蔽的真值像素，绝不作合法QUERY、正式指标、论文复现或方法收益。</b>固定正式1000权重，更新0；两支共享实际可见视频、首帧CLIP、双向flow、time IDs与初始Gaussian。普通支的25帧原生PNG与正式1000验证逐像素相同。只把CFG条件支从可见latent传播结果替换成完整GT经VAE mode编码的未缩放latent；两支无条件分支均为零。</p><div class="diagram"><div class="box">正常可见RGB</div>→<div class="box">VAE / 双向传播</div>→<div class="box">条件latent</div>→<div class="box">同一SVD去噪<br>无条件支=0</div>→<div class="box">原生 / 硬写回</div><div class="box">完整GT → VAE mode、未缩放 → 仅非法诊断条件</div></div><p>下方六视频位于同一同步播放组；前四列分别比较两支原生与硬写回，GT及可见输入保留作参照。任何清晰中心均来自真实写回，须只看原生列。</p><div class="four">''')
        for name, label in videos:
            sections.append(f'<figure><video controls preload="metadata" src="{name}"></video><figcaption>{label}</figcaption></figure>')
        sections.append('''</div><figure><img src="full_latent_oracle_step1000/gt_vae_reconstruction_contact.png" alt="完整GT经VAE mode重建的抽帧联系图"><figcaption>完整GT VAE mode重建，仅检查编码/解码；三帧不能判断时序。</figcaption></figure>''')
        sections.extend([f'<p>{oracle_status}</p>',
                         '<p><a href="full_latent_oracle_step1000/run.json">配对与非法输入说明</a> · <a href="full_latent_oracle_step1000/delivery_check.json">25帧重放及八视频解码核验</a> · <a href="full_latent_oracle_step1000/qa_boards/frames_00_04.jpg">连续帧图板入口</a></p></div>'])
    teacher = output/'paired_teacher_step1000'
    teacher_done = (teacher/'run.json').is_file()
    if teacher_done:
        teacher_run = read(teacher/'run.json')
        paired = teacher_run['paired_exact']
        if (not step1000_done or teacher_run.get('status') != 'complete'
                or teacher_run.get('checkpoint_step') != 1000
                or teacher_run.get('optimizer_updates') != 0
                or teacher_run.get('cfg') is not False
                or not math.isclose(teacher_run['sigma'], math.exp(.7), rel_tol=1e-6)
                or not all(paired[key] for key in ('same_x_t_before_after', 'same_scaled_noisy_unet_input',
                                                   'same_clip', 'same_time_ids', 'same_timestep', 'no_cfg_batch'))):
            raise ValueError('1000步单步teacher的来源、配对或固定sigma记录不一致')
        boards = [teacher/'qa_boards'/f'frames_{start:02d}_{start+4:02d}.jpg'
                  for start in range(0, 25, 5)]
        if not all(path.is_file() for path in boards):
            raise FileNotFoundError('单步teacher五张连续帧图板未齐')
        teacher_review = teacher/'assistant_review.json'
        if teacher_review.is_file():
            report = read(teacher_review)
            verdict = report.get('assistant_verdict', report.get('verdict', '结论见审核原文'))
            teacher_status = f'独立QA已归档：{html.escape(str(verdict))}；<a href="paired_teacher_step1000/assistant_review.json">完整审核</a>'
        else:
            teacher_status = '独立全帧QA待完成；此处不预填视觉结论'
        sections.insert(3,f'''<div class="panel warn"><b>单步teacher诊断已完成，仍不是合法自由采样QUERY。</b><p>正式1000步六窗{'独立审核hold' if review_hold else '结论见审核'}；完整GT oracle仍为非法输入，仅作定位。单步teacher两支都以带噪GT latent为输入，固定σ=exp(0.7)、无CFG、更新0；{teacher_status}。<a href="#paired-teacher-step1000">查看五张连续帧图板</a></p></div>''')
        sections.append('''<div class="panel warn" id="paired-teacher-step1000"><h2>固定σ带GT单步teacher：合法条件与非法完整GT条件</h2><p><b>两支均不是QUERY。</b>两支共享同一个带噪GT x_t、CLIP、时间条件和单步timestep；仅将条件从可见视频传播latent切换为读取隐藏像素的完整GT VAE mode latent。固定σ=exp(0.7)，无CFG、无dropout、零训练更新。完整GT条件非法，只用于定位。c_skip × x_t基线本身含GT；即使单步x0显示动作，也不能等同于从纯Gaussian生成25帧通过。</p><div class="diagram"><div class="box">GT latent + ε → 同一带噪x_t</div>→<div class="box">SVD单步去噪<br>无CFG</div>→<div class="box">合法/非法条件x0</div><div class="box">可见RGB → VAE/传播 → 合法条件</div><div class="box">完整GT → VAE mode → 非法条件</div><div class="box">x_t → c_skip基线</div></div><p>每张图板连续五帧，逐行依次为GT、visible、GT-VAE、合法条件x0、非法完整GT条件x0、c_skip × x_t；完整五板覆盖f00–f24。</p>''')
        for board in boards:
            relative = board.relative_to(output).as_posix()
            sections.append(f'<figure><img loading="lazy" src="{relative}" alt="单步teacher连续五帧，六行GT/visible/GT-VAE/合法x0/非法x0/c_skip基线"><figcaption>{html.escape(board.stem)}：连续五帧六行对照</figcaption></figure>')
        sections.append(f'<p>{teacher_status}</p><p><a href="paired_teacher_step1000/run.json">真实输入、配对、σ与限制</a></p></div>')
    capacity=output/'fixed_input_capacity_step500_64'
    capacity_path=capacity/'run.json'
    if capacity_path.is_file():
        probe=read(capacity_path)
        sections.append('''<div class="panel"><h2>固定输入容量：正式500权重 → 独立64步诊断</h2><p>同一训练片段0fc958cde2/start2。每次更新固定GT latent、噪声、σ、RAFT与CLIP；重算可训练FCNet和传播器。沿用Adam1e-5、原损失与参数范围，独立保存，不恢复到正式训练。</p><div class="diagram"><div class="box">固定片段 / GT flow / 固定噪声</div>→<div class="box">FCNet → 参考传播 → SVD</div>→<div class="box">固定σ单步拟合：BUILD</div><div class="box">仅可见视频 + 同Gaussian → 完整QUERY</div></div>''')
        sections.append(f'<p>实际状态：{html.escape(probe["status"])}。<a href="fixed_input_capacity_step500_64/run.json">完整输入角色与资源记录</a>。固定σ拟合不能证明全部噪声档位、完整生成或泛化；本诊断不计入正式四指标。</p>')
        if probe['status']=='complete':
            before=probe['teacher_before'];after=probe['teacher_after']
            sections.append(f'<p>实际优化更新{probe["actual_updates"]}/{probe["update_attempts"]}；teacher加权latent MSE：{before["weighted_mse"]:.6f} → {after["weighted_mse"]:.6f}。此数值只衡量带噪GT单步去噪。</p>')
            sections.append('<div class="four">')
            for stage,title in (('teacher_before','训练前：固定σ带噪GT去噪'),('teacher_after','训练后：固定σ带噪GT去噪')):
                sections.append(f'<figure><img src="fixed_input_capacity_step500_64/{stage}/contact.jpg" alt="{title}"><figcaption>{title}，f00/f12/f24；不据三帧判断时序。</figcaption></figure>')
            sections.append('</div>')
            for stage,title in (('query_before','诊断前QUERY：正式500权重'),('query_after','诊断后QUERY：500 + 固定输入64步')):
                sections.append(f'<article><h3>{title}</h3><p>只有可见RGB，seed2036/25步，fps6、CFG1→3；GT仅用于对照。</p><div class="four">')
                for name,label in (('gt','真实视频'),('visible','可见输入'),('pred','原生纯噪声生成'),('comp','可见中心硬写回')):
                    sections.append(f'<figure><video controls preload="metadata" src="fixed_input_capacity_step500_64/{stage}/{name}.mp4"></video><figcaption>{label}</figcaption></figure>')
                sections.append('</div></article>')
            sections.append(review_note(capacity,'fixed_input_capacity_step500_64'))
        sections.append('</div>')
    visibleclip=output/'fixed_input_capacity_step500_visibleclip64'
    if (visibleclip/'run.json').is_file():
        variant=read(visibleclip/'run.json')
        if variant['status']=='complete':
            sections.append('''<div class="panel"><h2>单因素训练控制：完整首帧CLIP → 可见首帧CLIP</h2><p>两支均从正式500权重起，分别64次更新；GT latent/噪声/flow、条件VAE、mask、fps7和Adam状态均固定，仅teacher首帧CLIP像素来源改变。使用相同训练helper，不声称完整匹配QUERY的PIL编码路径。新条件不替换正式论文训练协议。</p>''')
            comparison=variant['cache_comparison']
            sections.append(f'<p>共享缓存逐值相同：{all(comparison["shared_key_exact"].values())}；CLIP特征平均绝对差{comparison["clip_mae"]:.6f}。可见CLIP teacher单步加权MSE {variant["teacher_before"]["weighted_mse"]:.6f} → {variant["teacher_after"]["weighted_mse"]:.6f}，实际更新{variant["actual_updates"]}/{variant["update_attempts"]}。改变CLIP后的teacher原始误差不可直接与另一支绝对值比大小来认定QUERY改进。</p>')
            sections.append('<article><h3>同输入与Gaussian：仅看原生生成的变化</h3><div class="four">')
            for relative,label in (('fixed_input_capacity_step500_64/query_before/gt.mp4','真实视频，仅作对照'),('fixed_input_capacity_step500_64/query_before/pred.mp4','正式500：训练诊断前'),('fixed_input_capacity_step500_64/query_after/pred.mp4','完整GT CLIP训练64后的原生'),('fixed_input_capacity_step500_visibleclip64/query_after/pred.mp4','可见CLIP训练64后的原生')):
                sections.append(f'<figure><video controls preload="metadata" src="{relative}"></video><figcaption>{label}</figcaption></figure>')
            sections.append('</div></article><article><h3>可见CLIP训练后的原生与写回</h3><div class="four">')
            for name,label in (('gt','真实视频'),('visible','实际可见输入'),('pred','原生输出'),('comp','可见中心硬写回')):
                sections.append(f'<figure><video controls preload="metadata" src="fixed_input_capacity_step500_visibleclip64/query_after/{name}.mp4"></video><figcaption>{label}</figcaption></figure>')
            sections.extend(['</div></article>',review_note(visibleclip,'fixed_input_capacity_step500_visibleclip64'),'<a href="fixed_input_capacity_step500_visibleclip64/run.json">实际共享条件与单因素记录</a></div>'])
    noise_folder=output/'fixed_clip_resampled_diffusion_step500_64'
    if (noise_folder/'run.json').is_file():
        noise=read(noise_folder/'run.json')
        if noise['status']=='complete':
            name='fixed_clip_resampled_diffusion_step500_64'
            sections.append('''<div class="panel"><h2>固定片段64步：每步重采样 diffusion σ 与 ε</h2><p>从同一正式500步权重与Adam状态起，复用鸟片段、GT flow、VAE后验与条件噪声、完整首帧CLIP、mask和time IDs。相比固定噪声64步支，仅训练时每次更新联合重采样σ~LogNormal(.7,1.6)与ε~N(0,I)。固定teacher继续使用原σ/噪声测量；QUERY仍只用可见RGB、同seed2036/25步Gaussian。σ与ε共同变化，不能单独归因，也不是完整论文训练或泛化评估。</p><div class="diagram"><div class="box">同一鸟片段 / 固定观测条件</div>→<div class="box">固定或每步重采样σ、ε</div>→<div class="box">FCNet / 双向传播 / SVD<br>64次Adam</div>→<div class="box">同seed原生QUERY</div>→<div class="box">另列真实中心硬写回</div></div>''')
            before=noise['teacher_before']; after=noise['teacher_after']
            sections.append(f'<p>实际Adam更新{noise["actual_updates"]}/{noise["update_attempts"]}；固定teacher加权latent MSE {before["weighted_mse"]:.6f} → {after["weighted_mse"]:.6f}，只衡量带噪GT单步BUILD，不判定生成质量。缓存相同指固定teacher基准缓存；逐更新σ与ε另见<a href="{name}/updates.jsonl">64步记录</a>。<a href="{name}/optimizer_update_verification.json">Adam更新核验</a></p>')
            sections.append('<article><h3>同起点比较：只看原生视频</h3><div class="four">')
            for relative,label in (('fixed_input_capacity_step500_64/query_before/gt.mp4','GT：仅作对照'),
                                   ('fixed_input_capacity_step500_64/query_before/pred.mp4','两支相同的500步原生起点'),
                                   ('fixed_input_capacity_step500_64/query_after/pred.mp4','固定σ/ε训练64步后的原生'),
                                   (f'{name}/query_after/pred.mp4','重采样σ/ε训练64步后的原生')):
                sections.append(f'<figure><video controls preload="metadata" src="{relative}"></video><figcaption>{label}</figcaption></figure>')
            sections.append('</div></article><article><h3>重采样分支：原生与真实中心写回分开</h3><div class="four">')
            for kind,label in (('gt','真实视频，仅对照'),('visible','实际可见输入'),
                               ('pred','模型原生预测'),('comp','可见中心真实RGB硬写回，不能算原生')):
                sections.append(f'<figure><video controls preload="metadata" src="{name}/query_after/{kind}.mp4"></video><figcaption>{label}</figcaption></figure>')
            sections.append('</div></article><p>独立审核逐帧查看两支各25帧：重采样后植被和草地纹理有变化，但原生鸟仍模糊且近静止，未跟随GT f07–f14展翼；硬合成的清晰鸟头与翅膀来自真实中心，竖向接缝仍明显。本64步控制未见相对固定噪声的清楚结构或运动收益；不据此判充分容量或终止整个autoresearch。</p>')
            sections.extend([review_note(noise_folder,name),f'<p><a href="{name}/run.json">配置、输入角色与资源</a> · <a href="{name}/assistant_review.json">独立全帧审核</a> · <a href="{name}/query_after/qa_boards/frames_10_14.jpg">关键f10–f14逐帧板</a></p></div>'])
    flow_folder=output/'fixed_clip_visible_flow_step500_64'
    if (flow_folder/'run.json').is_file():
        flow=read(flow_folder/'run.json')
        if flow['status']=='complete':
            name='fixed_clip_visible_flow_step500_64'
            four_video_paths=(
                'fixed_clip_visible_flow_step500_64/query_before/gt.mp4',
                'fixed_clip_visible_flow_step500_64/query_before/pred.mp4',
                'fixed_clip_resampled_diffusion_step500_64/query_after/pred.mp4',
                f'{name}/query_after/pred.mp4',
            )
            four_modalities=tuple(f'{name}/query_after/{kind}.mp4' for kind in ('gt','visible','pred','comp'))
            missing=[relative for relative in (*four_video_paths,*four_modalities) if not (output/relative).is_file()]
            if missing:
                raise FileNotFoundError(f'visible-flow页面缺少视频: {missing}')
            sections.append('''<div class="panel"><h2>固定片段64步：训练输入GT flow → 可见RGB估计flow</h2><p>从同一正式500步权重及Adam状态出发，与上一重采样噪声64步支匹配片段、目标GT、首帧完整GT CLIP、VAE后验/条件噪声、优化预算和QUERY seed2036/25步。训练传播的flow输入改为冻结RAFT从可见RGB估计；GT flow继续只用于监督。逐步训练σ全部64项配对；旧支未留逐步ε，不能声称ε逐值相同。该诊断权重不进入正式训练。</p><div class="diagram"><div class="box">同一可见鸟片段</div>→<div class="box">冻结RAFT：GT或可见RGB flow</div>→<div class="box">FCNet / 双向传播 / SVD<br>64次Adam，GT监督</div>→<div class="box">同seed原生QUERY</div>→<div class="box">另列真实中心硬写回</div></div>''')
            visible_before=flow['visible_flow_teacher_before']['weighted_mse']
            visible_after=flow['visible_flow_teacher_after']['weighted_mse']
            gt_before=flow['teacher_before']['weighted_mse']
            gt_after=flow['teacher_after']['weighted_mse']
            sections.append(f'<p>实际更新{flow["actual_updates"]}/{flow["update_attempts"]}。单步带噪GT BUILD分两个口径：可见flow条件 {visible_before:.6f} → {visible_after:.6f}；固定GT-flow teacher基准 {gt_before:.6f} → {gt_after:.6f}。两者均非纯Gaussian多步QUERY画质。<a href="{name}/optimizer_update_verification.json">Adam、σ与GT监督核验</a> · <a href="{name}/paired_query_verification.json">25帧QUERY同起点核验</a></p>')
            sections.append('<article><h3>四视频配对：GT / 正式500 / GT-flow重采样64 / 可见flow64</h3><p>同一训练片段，同seed完整25帧；下方后三列均为原生预测。正式500视频是两支诊断共同的QUERY-before，不是诊断权重。GT只作视觉参照。</p><div class="four">')
            for relative,label in zip(four_video_paths,('真实GT，仅作对照','正式500步原生QUERY','GT-flow训练 + 重采样噪声64步原生','可见flow训练 + 重采样噪声64步原生')):
                sections.append(f'<figure><video controls preload="metadata" src="{relative}"></video><figcaption>{label}</figcaption></figure>')
            sections.append('</div></article><article><h3>可见flow分支：真实、条件、原生与合成</h3><div class="four">')
            for relative,label in zip(four_modalities,('真实视频，仅作对照','实际可见输入','模型原生预测','真实可见中心硬写回，不计原生能力')):
                sections.append(f'<figure><video controls preload="metadata" src="{relative}"></video><figcaption>{label}</figcaption></figure>')
            sections.append('</div></article><p>独立审核逐帧看过两支全25帧：可见flow分支仍只有模糊浅褐鸟块与树林草地，GT在f07–f14展翼时原生没有对应头翼与主体动作。相对GT-flow重采样64步支仅有局部纹理与形状差异，未见明确结构、运动或接缝收益；合成行的清晰运动鸟来自真实中心硬写回。短64步阴性不能判充分容量或整套方法失败。正式500→1000仅新增500次原论文协议更新，按固定六窗审核，不以此诊断直接放行。</p>')
            sections.extend([review_note(flow_folder,name),f'<p><a href="{name}/run.json">输入角色与两种teacher口径</a> · <a href="{name}/assistant_review.json">全帧独立审核</a> · <a href="{name}/query_after/qa_boards/frames_10_14.jpg">f10–f14逐帧板</a></p></div>'])
    temporal=output/'later_frame_control_step500'
    if (temporal/'run.json').is_file():
        control=read(temporal/'run.json')
        if control['status']!='complete':
            raise ValueError('后续帧控制尚未完成')
        sections.append('''<div class="panel"><h2>500权重后续帧控制：正常视频 / 全部重复首帧</h2><p>从正式500重新加载；固定正常视频选出的refs、同首帧CLIP、mask、Gaussian、seed2036/25步与精度，只有后续可见RGB不同。零训练更新；repeat固定normal refs，是受控反事实，不能冒充重复视频自动选参考帧的标准推理。</p>''')
        sections.append(f'<p>Gaussian/CLIP/时间条件逐值一致：{html.escape(str(control["exact_shared"]))}；正常支与容量诊断前QUERY的25帧uint8逐像素相同：{control["normal_replay_matches_previous_uint8_all25"]}。融合条件平均绝对差{control["fused_condition_mean_abs_difference"]:.6f}，原生图像差{control["native_output_mean_abs_difference"]:.6f}。非零响应不能证明正确使用时序，且变化同时经过RAFT、VAE与传播，不能定位单一模块。</p>')
        for branch,title in (('normal','真实的逐帧可见输入'),('repeat_first','后24帧重复已遮蔽首帧')):
            sections.append(f'<article><h3>{title}</h3><div class="four">')
            for name,label in (('gt','真实视频，仅用于对照'),('visible','实际可见输入'),('pred','原生输出'),('comp','实际可见中心硬写回')):
                sections.append(f'<figure><video controls preload="metadata" src="later_frame_control_step500/{branch}/{name}.mp4"></video><figcaption>{label}</figcaption></figure>')
            sections.append('</div></article>')
        sections.extend([review_note(temporal,'later_frame_control_step500'),'<a href="later_frame_control_step500/run.json">完整控制与限制</a></div>'])
    query = output/'clip_query_control'
    if (query/'diagnostic.json').is_file():
        meta=read(query/'diagnostic.json')
        if meta['status']!='complete':
            raise ValueError('完整QUERY对照未完成，不能展示为结果')
        sections.append('''<div class="panel"><h2>完整首帧CLIP能否单独修复自由生成？</h2>
        <p>固定step100、可见黑洞RAFT、VAE/传播条件、时间参数、CFG与完全相同的初始Gaussian，只切换首帧CLIP；两组均25帧、25步，无带噪GT初始化。</p>
        <div class="diagram"><div class="box">同一可见视频</div>→<div class="box">RAFT / FCNet / 双向传播</div>→<div class="box">同一初始Gaussian<br>SVD去噪</div>→<div class="box">原生 / 硬写回</div><div class="box">完整GT或可见首帧CLIP → 去噪</div></div>''')
        for branch in ('full_gt_oracle','visible_black_hole'):
            label='完整GT首帧：不可部署的oracle' if branch=='full_gt_oracle' else '可见首帧：合法QUERY'
            sections.append(f'<article><h3>{label}</h3><div class="four">')
            for file, title in (('gt','真实视频'),('visible','可见条件展示'),('pred','模型原生输出'),('comp','可见中心硬写回')):
                sections.append(f'<figure><video controls preload="metadata" src="clip_query_control/{branch}/{file}.mp4"></video><figcaption>{title}</figcaption></figure>')
            sections.append('</div></article>')
        sections.extend([review_note(query,'clip_query_control'), '<a href="clip_query_control/diagnostic.json">共享噪声和输入角色记录</a></div>'])
    for name, label in (('pretrained_svd_sanity','256×256，全管线bf16 autocast'),
                        ('pretrained_svd_sanity_fp32','256×256，全FP32'),
                        ('pretrained_svd_sanity_native','1024×576，标准fp16与CPU卸载')):
        folder=output/name
        if not (folder/'diagnostic.json').is_file():
            continue
        if read(folder/'diagnostic.json')['status']!='complete':
            raise ValueError(f'{name}尚未完成')
        sections.append(f'''<div class="panel"><h2>原始SVD sanity：{label}</h2>
        <p>原始预训练SVD，未加载作者编辑权重或P1断点。完整首帧 → 标准Diffusers img2vid → 25帧原生输出。不是逐帧外扩；尺寸/精度/原始画幅同时变化的组不能作为单因素归因。</p>
        <div class="four"><figure><img src="{name}/input_full_first.png"><figcaption>唯一完整首帧条件</figcaption></figure><figure><video controls preload="metadata" src="{name}/pred.mp4"></video><figcaption>原始SVD原生生成，25帧</figcaption></figure></div>
        {review_note(folder,name)}<p><a href="{name}/diagnostic.json">实际调用配置与限制</a></p></div>''')
    if bounded2000_panel:
        sections.insert(1, bounded2000_panel)
    sections.append('</main><script>document.querySelectorAll(".four,.five").forEach(g=>{let vs=[...g.querySelectorAll("video")],busy=false;vs.forEach(v=>{v.addEventListener("play",()=>{if(busy)return;busy=true;vs.forEach(x=>{if(x!==v){x.currentTime=v.currentTime;x.play().catch(()=>{})}});busy=false});v.addEventListener("pause",()=>{if(busy)return;busy=true;vs.forEach(x=>{if(x!==v)x.pause()});busy=false});v.addEventListener("seeked",()=>{if(busy)return;busy=true;vs.forEach(x=>{if(x!==v&&Math.abs(x.currentTime-v.currentTime)>.15)x.currentTime=v.currentTime});busy=false})})});</script></html>')
    (output/'condition_gap_review.html').write_text(''.join(sections),encoding='utf-8')
    index=output/'index.html'
    document=index.read_text(encoding='utf-8')
    # 删除已失效的CPU准备块，保留其余历史产物；按嵌套div定位结束。
    start=document.find('<div class="panel warn" id="condition-gap">')
    if start>=0:
        depth=0
        for match in re.finditer(r'<div\b[^>]*>|</div\s*>',document[start:]):
            depth += -1 if match.group().startswith('</') else 1
            if depth==0:
                document=document[:start]+document[start+match.end():]
                break
    # 首页有旧运行器留下的当前时态关机/100步提示；历史关机证据文件保留，
    # 但首页不再用它描述当前运行状态。
    document=re.sub(r'<div class="panel"><b>历史记录：上轮 GPU 作业结束后已关机</b>.*?</div>',
                    '',document,count=1,flags=re.S)
    document=re.sub(r'<div class="panel warn"><h2>当前结论：P1仍未通过</h2>.*?</div>',
                    '',document,count=1,flags=re.S)
    document=re.sub(r'</html><p[^>]*>本轮GPU计算与独立审核已结束；AutoDL已按授权关闭.*?</p>',
                    '</html>',document,count=1,flags=re.S)
    # 旧快照仍供回溯，但其当时时态不能覆盖页首的正式续训状态。
    for old,new in (
        ('<h2>论文双向候选：限定两步预检</h2>','<h2>历史阶段：论文双向候选两步预检</h2>'),
        ('没有放行1000或100K','该时点尚未开始后续正式训练'),
        ('<h2>双向候选：实测与资源修复</h2>','<h2>历史阶段：双向候选实测与资源修复</h2>'),
        ('已记录 100 个更新；当前预算 100','该阶段记录100个更新，预算100'),
        ('显存而停止；','显存而中断；'),
        ('<h2>最新独立全帧审核</h2>','<h2>100步阶段独立全帧审核</h2>'),
        ('这是已停止的旧训练速度估算，并非正在运行七天训练。旧1000质量门hold；新双向候选限定到100后完成短窗及36帧检查，接着做逐帧条件与固定片段容量诊断。没有排队1000或100K，不以有限loss放行长训。',
         '这是旧公开路径的历史速度估算；旧1000质量门hold。该时点双向候选完成100步和后续条件诊断，现时正式续训状态以页首快照为准。'),
    ):
        document=document.replace(old,new)
    if step1000_done:
        quality = '正式1000步六窗独立审核仍hold' if review_hold else '正式1000步六窗已完成生成'
        if bounded2000:
            quality += '；1000→2000有界阶段已有实际快照'
        oracle_notice = (f'非法完整GT latent oracle已完成零训练定位，{oracle_status}；即使oracle清楚也不算方法通过。'
                         '<a href="condition_gap_review.html#full-latent-oracle">查看非法输入边界和两支对照</a>。') if oracle_done else ''
        teacher_notice = (f'带GT单步teacher诊断已完成，{teacher_status}；不代表自由采样通过。'
                          '<a href="condition_gap_review.html#paired-teacher-step1000">查看五张连续帧图板</a>。') if teacher_done else ''
        bounded_notice = (f'{bounded2000_summary}；这是快照，不代表实时训练进度或2000步质量通过。'
                          '<a href="condition_gap_review.html#bounded2000-snapshot">查看实际PID与命令</a>。') if bounded2000 else ''
        notice=f'<div class="panel" id="gap-result-link"><b>当前P1：{quality}</b><p>{bounded_notice}同协议从500到正式1000已追加500次更新；{review_detail}。{oracle_notice}{teacher_notice}<a href="condition_gap_review.html#validation-step1000">查看六窗500/1000原生同步对照</a>；正式指标未计算，人工verdict留空。下方其余进度段落按产生时点保留为历史记录，当前状态以本段和<a href="bounded1000_state.json">1000步完成快照</a>为准。</p></div>'
    elif continuation:
        notice=f'<div class="panel" id="gap-result-link"><b>当前P1：正式500→1000步同协议续训，仅新增500步</b><p>截至 {html.escape(continuation["observed_at_utc"])} 快照记录 {continuation["observed_train_step"]} 步；1000步验证尚无结果。到达后核查固定三例×两倍率共六窗、原生视频与质量门。<a href="condition_gap_review.html">查看正式500步六窗及固定片段64步CLIP、噪声、可见flow控制</a>；人工verdict留空。下方其余进度段落按产生时点保留为历史记录，当前状态以本段和<a href="bounded1000_state.json">续训快照</a>为准。</p></div>'
    else:
        notice='<div class="panel" id="gap-result-link"><b>当前P1：正式500步质量hold</b><p><a href="condition_gap_review.html">查看六窗验证与固定片段64步控制</a>；正式指标未计算，人工verdict留空。</p></div>'
    if 'id="gap-result-link"' not in document:
        document=document.replace('</h1>','</h1>'+notice,1)
    else:
        document=re.sub(r'<div class="panel" id="gap-result-link">.*?</div>',lambda _:notice,document,count=1,flags=re.S)
    index.write_text(document,encoding='utf-8')


if __name__=='__main__':
    main()
