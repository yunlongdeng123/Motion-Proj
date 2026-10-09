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
    verdict = read(path).get('assistant_verdict', '')
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
    <div class="panel warn"><b>本页条件诊断使用100步断点，P1尚未通过。</b><p>本页诊断均没有训练更新；原始SVD生成不是外扩任务，也不是论文指标。完整GT条件只作诊断，不能用于正式QUERY。人工verdict留空。</p></div>'''
    sections = [header, condition_gap_panel(gap),
                review_note(output/'condition_gap_teacher', 'condition_gap_teacher')]
    state_path=output/'bounded500_state.json'
    if state_path.is_file():
        state=read(state_path)
        sections.insert(1,f'''<div class="panel"><h2>有界学习曲线：100 → 总计500步</h2><p>快照 {html.escape(state['observed_at_utc'])}；已记录 {state['observed_train_step']} 步，状态 {html.escape(state['status'])}。恢复模型、Adam状态和随机状态；数据、mask、网络、学习率与参数范围不变。500步后生成三个原固定验证样本×两倍率并退出，等待助手全帧审核，不自动追加1000或100K。</p><a href="bounded500_state.json">实际PID、命令与预算快照</a></div>''')
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
    document=document.replace('本轮 GPU 作业已结束，AutoDL 已关机','历史记录：上轮 GPU 作业结束后已关机')
    document=document.replace('正式进度100步，P1仍未通过；旧关机通知为历史记录。','诊断使用100步断点，最新训练状态见条件页；P1仍未通过。旧关机通知为历史记录。')
    notice='<div class="panel" id="gap-result-link"><b>最新条件诊断已完成</b><p><a href="condition_gap_review.html">查看18组单步诊断、2组共享Gaussian完整采样与原始SVD对照</a>。诊断使用100步断点，最新训练状态见条件页；P1仍未通过。旧关机通知为历史记录。</p></div>'
    if 'id="gap-result-link"' not in document:
        document=document.replace('</h1>','</h1>'+notice,1)
    index.write_text(document,encoding='utf-8')


if __name__=='__main__':
    main()
