"""利用已同步产物更新条件诊断页，不重新编码历史视频或启动GPU。"""
from __future__ import annotations

import argparse
import html
import json
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
    header = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P1 条件与原始SVD对照</title>
    <style>body{font:16px/1.65 system-ui;background:#eef2f6;color:#182a3a}main{max-width:1400px;margin:auto;padding:24px}.panel,article{background:white;border:1px solid #ccd7e3;padding:20px;margin:20px 0;border-radius:10px}.warn{background:#fff5de}.diagram{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.box{padding:12px;background:#e6effb;border:1px solid #96abc4}.four{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}video,img{width:100%;max-width:100%}td,th{border:1px solid #ccd7e3;padding:8px}table{border-collapse:collapse}a{color:#145dad}@media(max-width:800px){.four{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main>
    <h1>P1：固定权重条件诊断与原始SVD对照</h1><p><a href="index.html">返回完整进度、历史输出与数据页</a></p>
    <div class="panel warn"><b>P1尚未通过：正式训练与诊断分别报告。</b><p>旧条件隔离使用100步权重且更新0；正式500步验证质量hold，固定片段64步控制另列。原始SVD生成不是外扩任务或论文指标。完整GT条件只作BUILD诊断，不能用于正式QUERY；人工verdict留空。</p></div>'''
    sections = [header, condition_gap_panel(gap),
                review_note(output/'condition_gap_teacher', 'condition_gap_teacher')]
    continuation_path=output/'bounded1000_state.json'
    continuation=None
    if continuation_path.is_file():
        continuation=read(continuation_path)
        if continuation['source_step'] != 500 or continuation['training_budget'] != 1000:
            raise ValueError('正式续训预算与500→1000记录不一致')
        if not continuation['new_1000_validation_available']:
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
    sections.append('</main><script>document.querySelectorAll(".four").forEach(g=>{let vs=[...g.querySelectorAll("video")],busy=false;vs.forEach(v=>{v.addEventListener("play",()=>{if(busy)return;busy=true;vs.forEach(x=>{if(x!==v){x.currentTime=v.currentTime;x.play().catch(()=>{})}});busy=false});v.addEventListener("pause",()=>{if(busy)return;busy=true;vs.forEach(x=>{if(x!==v)x.pause()});busy=false});v.addEventListener("seeked",()=>{if(busy)return;busy=true;vs.forEach(x=>{if(x!==v&&Math.abs(x.currentTime-v.currentTime)>.15)x.currentTime=v.currentTime});busy=false})})});</script></html>')
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
    if continuation and not continuation['new_1000_validation_available']:
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
