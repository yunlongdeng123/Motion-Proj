"""将已经完成的 r19/r20 诊断做成离线审核页；不重新推理或训练。"""
from pathlib import Path
from html import escape
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('review_dir', type=Path)
    args = ap.parse_args()
    root = args.review_dir
    manifest = json.loads((root / 'diagnostic_manifest.json').read_text(encoding='utf-8'))
    human = json.loads((root / 'human_review.json').read_text(encoding='utf-8'))
    roles = [
        ('target', '原视频 / 真实 GT', 'P019 是已知真实背景；A061 是原视频目标框'),
        ('condition', '实际遮后条件', '灰色区域 H 被擦除后进入模型'),
        ('base', '原始 DriveEditor', '复用原固定条件结果'),
        ('r7_round_only', '仅 r7 范围舍入', '80 个空间张量；优化 0 步'),
        ('r7', '有效 r7', '历史微调结果，保持原权重'),
        ('r14_round_only', '仅 r14 范围舍入', '80 个时间张量；优化 0 步'),
        ('r14', 'r14 时间 self attention', '历史微调结果，保持原权重'),
    ]
    notes = {
        'temporal_P019': (
            '合成遮挡 · 世界静止 A＋运动 ego',
            '你的判断：r7 / r14 路面更好，但新增了一辆不存在的车。',
            '本轮抽看全部 10 帧：两个“仅舍入”对照仍接近原模型的灰色轮廓 / 块状残留，没有复现实际微调版新增的深色车辆。精度转换是工程偏差，但单独不足以解释这个幻觉。'),
        'A061_w08': (
            '真实 DELETE · 目标删除与道路恢复',
            '你的判断：有效 r7 ＞ r14 ＞ 原始 DriveEditor。这是应保留的真实正例。',
            '本轮零步对照的固定 f5 接近原模型；实际微调版还有道路 / 边缘改动。这里保留全部视频供你看；助手没有根据这一帧补填时序通过或人工分数。'),
    }
    cases_html = []
    for case in manifest['cases']:
        cid = case['id']
        process, ranking, note = notes[cid]
        cards = []
        for role, label, description in roles:
            frame = case['frame_pattern'].format(i=f'{5:03d}', role=role)
            cards.append(f'''<article class="column">
<h4>{escape(label)}</h4><p class="sub">{escape(description)}</p>
<video controls muted playsinline preload="none" poster="{escape(frame)}" src="{escape(case['videos'][role])}" data-role="{role}"></video>
<a class="frame-link" href="{escape(frame)}" target="_blank"><img loading="lazy" class="frame" data-role="{role}" src="{escape(frame)}" alt="{cid} {escape(label)} f5"></a>
<span class="frame-label">固定帧 f5 · 点击图片放大</span></article>''')
        native_links = ' · '.join(
            f'<a href="{escape(case["native"][role])}" target="_blank">{escape(label)}</a>'
            for role, label, _ in roles if role in case['native'])
        cases_html.append(f'''<section class="case" id="{cid}" data-case="{cid}">
<div class="case-title"><div><span class="eyebrow">{escape(case['scene'])} · {escape(process)}</span><h3>{cid}</h3></div><span class="badge">10 帧 / 10 fps · seed 42 · 25 采样步</span></div>
<p class="human">{ranking}</p><p>{note}</p>
<div class="controls"><button data-action="play">同步播放</button><button data-action="pause">暂停</button><button data-action="prev">上一帧</button><input type="range" min="0" max="9" step="1" value="5" aria-label="{cid} 帧号"><button data-action="next">下一帧</button><output>f5 / f9</output><span class="sub">滑块会暂停并定位所有视频、更新下方固定帧</span></div>
<div class="scroll"><div class="columns">{''.join(cards)}</div></div>
<p class="sub playback-status" aria-live="polite">视频是最终 alpha 写回结果。第一、二列为输入，后五列为对照输出。</p>
<details><summary>查看五臂原生输出（alpha 写回前）</summary><p>{native_links}</p><p class="sub">原生 deletion 仍是删除结果，不是 factual 重建。可以用它区分模型生成与写回边界问题。</p></details>
<details><summary>全部 10 帧接触图</summary><a href="contacts/{cid}_first5.jpg" target="_blank"><img class="contact" loading="lazy" src="contacts/{cid}_first5.jpg" alt="{cid} 前五帧"></a><a href="contacts/{cid}_last5.jpg" target="_blank"><img class="contact" loading="lazy" src="contacts/{cid}_last5.jpg" alt="{cid} 后五帧"></a></details>
</section>''')
    rows = ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(row[k]))}</td>' for k in ['eval_id', 'scene', 'type', 'process', 'human_ranking_verbatim']) + '</tr>'
        for row in human['cases'])
    payload = json.dumps(manifest, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v77 · 工程与数据诊断 r19 / r20</title><style>
:root{color-scheme:dark;--bg:#11171e;--panel:#1a232e;--line:#354350;--text:#e8edf3;--muted:#a4b4c5;--yellow:#edca7b;--blue:#8cc5f7;--green:#93d4b5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif}main{max-width:1900px;margin:auto;padding:32px}h1{font-size:clamp(27px,3vw,43px);line-height:1.3;margin:8px 0 16px}h2{font-size:26px}h3{font-size:24px;margin:4px 0}h4{font-size:16px;margin:0}p{margin:9px 0}a{color:var(--blue)}.eyebrow{color:var(--blue);font-size:13px;letter-spacing:1px}.lead{font-size:21px;max-width:1000px}.sub,small{font-size:13px;color:var(--muted)}.badge{border:1px solid var(--line);border-radius:18px;padding:4px 12px;font-size:13px;color:var(--muted)}nav{display:flex;gap:18px;flex-wrap:wrap;padding:12px 0;border-bottom:1px solid var(--line)}section{scroll-margin-top:18px}.block,.case{margin:26px 0;padding:24px;border:1px solid var(--line);background:var(--panel);border-radius:12px}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.summary>div{border-top:3px solid var(--yellow);background:var(--panel);padding:20px}.summary strong{font-size:20px}.warning{color:var(--yellow)}.passed{color:var(--green)}.human{border-left:3px solid var(--yellow);padding:8px 14px;background:#292a26}.case-title{display:flex;justify-content:space-between;align-items:center;gap:15px;flex-wrap:wrap}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;margin:14px 0}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid var(--line)}th{color:var(--muted)}.columns{display:grid;grid-template-columns:repeat(7,minmax(235px,1fr));gap:10px;min-width:1740px}.column{min-width:0;background:#10171e;padding:10px;border:1px solid #2b3744;border-radius:8px}.column>.sub{min-height:46px}.column video,.column img{width:100%;display:block;aspect-ratio:16/9;background:black;object-fit:contain}.frame-link{display:block;margin-top:12px}.frame-label{display:block;font-size:12px;color:var(--muted);margin-top:4px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}button{font:inherit;font-size:14px;padding:7px 12px;background:#283a4a;color:var(--text);border:1px solid #526778;border-radius:6px;cursor:pointer}button:hover{background:#35536e}input[type=range]{width:200px;accent-color:var(--blue)}output{font-size:14px;min-width:68px}.contact{width:100%;height:auto;display:block;margin-top:14px}details{border-top:1px solid var(--line);padding:14px 0;margin-top:8px}summary{cursor:pointer;color:var(--blue)}.diagram{overflow:auto}.diagram svg{min-width:980px;width:100%;max-height:380px}.evidence-links{display:flex;flex-wrap:wrap;gap:12px}.split{display:grid;grid-template-columns:1fr 1fr;gap:25px}code{background:#10151d;padding:2px 5px;border-radius:4px;font-size:.9em}.footer{margin:30px 0;color:var(--muted);font-size:13px}@media(max-width:900px){main{padding:18px}.summary,.split{grid-template-columns:1fr}.block,.case{padding:16px}.badge{font-size:12px}table{min-width:650px}}
</style></head><body><main>
<header><span class="eyebrow">V77 · WS-V77-TARGET-PROTECTED-20260929 / r19＋r20 · 2026-10-02</span>
<h1>先排查：工程错误与任务覆盖都存在问题</h1>
<p class="lead">目前不能把失败主要归因于“微调模块选错”。已确认目标 mask 裁漏、训练初始化舍入、重复任务，以及合成与真实 DELETE 的输入分布差异。</p>
<p>按你最新要求，持续训练与造数迭代已停。收到这次 review 后新增训练 <strong>0 步</strong>；只跑了两例、四个零训练诊断窗口。原有权重、视频和人工排序全部保留。</p>
<nav><a href="#findings">已确认问题</a><a href="#mask">A022 漏洞证据</a><a href="#distribution">数据差异</a><a href="#controls">零训练对照视频</a><a href="#human">你的 14 条 review</a><a href="#next">下一步顺序</a><a href="report.md">完整报告</a></nav></header>
<div class="summary"><div><strong>工程：3 项有证据</strong><p>真实入口硬裁 mask；FP32 初始化先被舍入；模型可见任务重复。</p></div><div><strong>数据：有效 ≠ 覆盖充分</strong><p>合成 RGB 无泄漏，但洞的大小、形状、截边和跨帧证据仍未对齐真实删除。</p></div><div><strong>效果：保留正例，也保留反证</strong><p>A061 有真实收益；P019 同时出现路面改善与幻觉车，MAE 无法替代语义验收。</p></div></div>
<section class="block"><h2>实际链条与本轮检查位置</h2><div class="diagram">
<svg viewBox="0 0 1240 300" role="img" aria-label="真实视频生成合成遮挡，或真实删除经 SAM 与 GT 处理，遮后条件进入 DriveEditor，再经 alpha 写回和人工审核；原始真实视频仅作训练监督。">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#92acc4"/></marker></defs>
<g fill="#203345" stroke="#52738d" stroke-width="1.5"><rect x="10" y="30" width="145" height="64" rx="9"/><rect x="185" y="30" width="190" height="64" rx="9"/><rect x="10" y="164" width="145" height="64" rx="9"/><rect x="185" y="164" width="190" height="64" rx="9"/><rect x="423" y="95" width="175" height="74" rx="9"/><rect x="654" y="95" width="170" height="74" rx="9"/><rect x="870" y="95" width="145" height="74" rx="9"/><rect x="1054" y="95" width="175" height="74" rx="9"/></g>
<g fill="none" stroke="#92acc4" stroke-width="2" marker-end="url(#arrow)"><path d="M155 62H185"/><path d="M155 196H185"/><path d="M375 62H400V117H423"/><path d="M375 196H400V147H423"/><path d="M598 132H654"/><path d="M824 132H870"/><path d="M1015 132H1054"/><path d="M739 47V95"/><path d="M80 30V10H630V80H715V95" stroke-dasharray="5 4"/></g>
<g fill="#e6edf4" text-anchor="middle" font-family="system-ui, Microsoft YaHei, sans-serif" font-size="16"><text x="82" y="66">真实视频 Y</text><text x="280" y="57">合成遮挡 X＋H</text><text x="280" y="80" font-size="13" fill="#a4b4c5">车辆轮廓 / 真实 GT</text><text x="82" y="201">真实 DELETE</text><text x="280" y="190">SAM → GT 硬裁</text><text x="280" y="214" font-size="13" fill="#edca7b">矩形 H / 存在漏车</text><text x="510" y="123">擦除 → resize</text><text x="510" y="149">实际模型条件</text><text x="739" y="123">DriveEditor</text><text x="739" y="149">原架构 / 原生补景</text><text x="942" y="124">alpha 写回</text><text x="942" y="149" font-size="13">目标全覆盖检查</text><text x="1142" y="123">视频 / 身份审核</text><text x="1142" y="149" font-size="13">新增车 · 邻车 · 时序</text><text x="740" y="35" font-size="14" fill="#93d4b5">修复初始化：保留训练参数 FP32</text><text x="405" y="278" font-size="14" fill="#a4b4c5">虚线：Y 仅作训练监督；完整 synthetic-X 不进入当前补景条件</text></g></svg></div></section>
<section id="findings" class="block"><h2>已确认问题，以及目前修到哪一步</h2><div class="scroll"><table><thead><tr><th>问题</th><th>实际证据</th><th>状态与边界</th></tr></thead><tbody>
<tr><td>GT 投影硬裁 SAM</td><td>A022 f0 有 1,314 个原 SAM 像素在模型 H 外，画面能看到漏掉的车头边缘；十帧中 2,693 个 SAM 像素的写回 alpha 不为 1。</td><td class="warning">已定位，旧 mask 未改。SAM 不是像素真值，其他 case 需核准实例；A041 全覆盖仍失败。</td></tr>
<tr><td>训练参数初始化舍入</td><td>全模型先转 BF16，再让选中参数回到 FP32，80 张量在优化前已经变化。</td><td><span class="passed">代码已修，160 个实际 checkpoint 张量原值保留验证通过。</span> 尚未重训；零步对照未复现 P019 的幻觉车。</td></tr>
<tr><td>模型可见任务重复</td><td>50 train → 46 个不同任务；11 合成 val → 10。M006 / M009 的 X、Y、H 和五臂十帧输出完全相同。</td><td>已另报去重统计，旧数据不删除。重复影响采样权重和 case 分母；本次 scene 等权 MAE 恰好未改变。</td></tr>
</tbody></table></div><p>额外的损失路由问题：r17 未把 B 标签传入训练，已在 87 步隔离；r18 修正后完成 160 步，但本次已停在 <strong>0 个新评价窗口</strong>，不能宣称有效。旧 r7 / r14 全图损失仍通过真实 Y 监督 B，不因此作废。</p></section>
<section id="mask" class="block"><h2>看这里：A022 的原车边缘仍在输入里</h2><p>从左到右：原 RGB、模型 H 与漏出的 SAM、实际灰色遮后条件。蓝色是 H，红色是原 SAM 在 H 外的区域。A022 第一行车头边缘有可见残留；A042 的红色像素身份仍不确定；A041 的原 SAM 全被盖住，仍然失败。</p><a href="contacts/real_mask_trace.jpg" target="_blank"><img class="contact" src="contacts/real_mask_trace.jpg" alt="A022 A042 A041 的原图、mask 和实际条件对照"></a><p>源码顺序是 <code>largest(SAM &amp; GT_hull)</code> → 膨胀 → 再裁 GT → 外包矩形 H。这会让 GT 误差变成硬边界。正确整改应先核准目标像素，再保证输入 H 和写回 alpha 都完整覆盖，并单独检查邻车冲突。</p></section>
<section id="distribution" class="block"><h2>合成数据没有 RGB 泄漏，但任务分布有缺口</h2><div class="split"><div><h3>已通过的工程检查</h3><p>重新解码 61 个 case、610 帧：Y 精确等于真实 RGB；synthetic-X 的变化全部在 H 内；先遮再 resize 后，masked-X 与 masked-Y 全部一致。</p><p>八项条件字段与实际 DELETE 入口在同尺寸下相同；106 个目标 encoder 权重严格恢复；r7 / r14 的 80 个选中张量确实更新且有限。</p><p class="passed">当前没有证据指向“模型看到完整贴纸 RGB”或“权重根本没生效”。</p></div><div><h3>覆盖检查还不够</h3><p>扫过保护车只有 4 个训练例、3 个 world；验证只有 1 个例、1 个 world。世界静止 A＋运动 ego 已有 11 例，不能说完全缺失。</p><p>训练窗实际仅 0.85–0.90 秒。连续、无漏 mask、每帧保留一部分 B，并不保证“现在被挡的部分在其他输入帧见过”。</p><p>近似几何代理：P006 的其他帧支持约 60.5%，P019 约 15.8%。这不是纹理可见性真值，只能提示证据覆盖需单独检验。</p></div></div>
<div class="scroll"><table><thead><tr><th>进入模型的输入</th><th>合成训练：50 例 / 500 帧</th><th>真实开发：8 例 / 80 帧</th></tr></thead><tbody><tr><td>H 平均占画面</td><td>1.82%</td><td>5.60%</td></tr><tr><td>H 形状</td><td>车辆轮廓；框填充率 0.814</td><td>矩形；框填充率 1.000</td></tr><tr><td>画面截边</td><td>0 / 500 帧</td><td>39 / 80 帧</td></tr><tr><td>分辨率</td><td>320 × 576 训练</td><td>576 × 1024 推理</td></tr></tbody></table></div><p>训练最大洞为 5.08%，A022 平均洞为 21.57%。但 A061 的洞平均 9.15%，同样超出训练范围，却获得人眼收益。因此覆盖差异是实证缺口，<strong>不是充分因果解释</strong>。</p><p>实际训练步骤中，被遮保护车平均仅占画面 0.344%；全图均匀损失的保护车监督较稀疏。P019 又说明道路 MAE 改善与新增车可以同时发生，需要独立记录新增车和保护身份损坏。</p></section>
<section id="controls" class="block"><h2>零训练控制：仅精度舍入会不会导致幻觉？</h2><p>固定 P019 和 A061，原模型分别只对 r7 范围或 r14 范围的 80 个张量做 FP32 → BF16 → FP32，<strong>优化 0 步</strong>。四个新窗口与旧结果使用相同 RGB、H、seed 42、25 采样步、576 × 1024、previous=false 和写回 alpha。</p><p>只有“仅舍入”两列是本轮新生成；原模型、r7、r14 复用历史结果。像素差只描述输出变化，不能当画面质量分。以下固定帧默认 f5，视频可同步查看。</p></section>
__CASES__
<section id="human" class="block"><h2>你的部分人工 review：14 条原文保存</h2><p>8 个真实 DELETE 中，A061 有明确相对收益，5 例没明显区别，A022 / A041 全失败。这是部分排名，不是整个审计集的通过率，也没有转换成 0 / 1 / 2。</p><div class="scroll"><table><thead><tr><th>Case</th><th>Scene</th><th>类型</th><th>过程</th><th>人工原文</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="sub">M009 的“有效 r9”原样保留。历史页面实际权重是 base / r7 / r8 / r10 / r14；r9 是数据阶段，没有独立 checkpoint，未擅自映射成绩。r7 与 r14 的数据不同，不能单独归因更新模块；同数据模块对照是 r10 与 r14。</p><p><a href="../v77-target-protected-r14/index.html">打开原 r14 完整对照页</a> · <a href="human_review.json">人工原始记录 JSON</a></p></section>
<section id="next" class="block"><h2>接下来应先改什么</h2><ol><li><strong>修真实 DELETE 的定义。</strong>核准目标实例，检查 H 全覆盖、目标 alpha=1 和保护对象冲突；保留旧70例原件。</li><li><strong>去重并补任务覆盖。</strong>按实际 Y / H / 遮后条件去重；对齐模型真正看到的洞形状、尺度、截边和遮挡—显露过程；区分其他帧见过、从未见过、证据未知。</li><li><strong>再用同范围模块做固定对照。</strong>保持原架构，使用保真初始化；分开报告道路恢复、新增车、保护车身份和时序，不继续堆同配方步数。</li></ol><p>本轮停在诊断交付，没有启动下一批训练。r18 权重保留但没有效果结论；r20 是零步诊断。未曝光 final 未用，人工分数未代填。尚未达到此前“稳定真实 DELETE＋补景收益后关机”的完成条件，因此未关机。</p></section>
<section class="block"><h2>可复核文件</h2><div class="evidence-links"><a href="report.md">完整技术报告</a><a href="findings.json">结论与范围</a><a href="audit_result.json">610 帧与真实入口检查</a><a href="process_and_duplicates.json">过程统计 / 重复任务</a><a href="duplicate_output_check.json">M006 / M009 输出核对</a><a href="precision_initialization_audit.json">初始化差值分解</a><a href="precision_fix_validation.json">修复验证</a><a href="rounding_diagnostic.json">零步诊断观察</a><a href="diagnostic_manifest.json">视频与帧清单</a><a href="delivery_validation.json">本页文件验证</a></div></section>
<p class="footer">运行位置：wm-3090-1001 · 分支 v77 · failure V77-F02 · 诊断 r19 / r20。所有视频为离线文件，页面不请求外部服务。单帧观察不替代视频时序判断。</p>
</main><script id="manifest" type="application/json">__MANIFEST__</script><script>
const manifest=JSON.parse(document.querySelector('#manifest').textContent);
for(const item of manifest.cases){
  const section=document.getElementById(item.id), videos=[...section.querySelectorAll('video')];
  const slider=section.querySelector('input[type=range]'), output=section.querySelector('output');
  const status=section.querySelector('.playback-status'); let syncing=false;
  function showFrame(index,seek=true){
    index=Math.max(0,Math.min(9,Math.round(index)));slider.value=index;output.textContent=`f${index} / f9`;
    for(const img of section.querySelectorAll('img.frame')){
      const path=item.frame_pattern.replace('{i}',String(index).padStart(3,'0')).replace('{role}',img.dataset.role);
      img.src=path;img.parentElement.href=path;img.alt=`${item.id} ${img.dataset.role} f${index}`;
      img.parentElement.nextElementSibling.textContent=`固定帧 f${index} · 点击图片放大`;
    }
    if(seek){for(const video of videos){video.pause();if(video.readyState>0)video.currentTime=index/10;else{video.dataset.pendingTime=index/10;video.load();}}}
  }
  for(const video of videos){
    video.addEventListener('loadedmetadata',()=>{if(video.dataset.pendingTime!==undefined){video.currentTime=Number(video.dataset.pendingTime);delete video.dataset.pendingTime;}});
    video.addEventListener('error',()=>{status.textContent='视频加载失败：'+video.getAttribute('src')+'。请保留 effects 文件夹与 HTML 的相对位置。';});
  }
  slider.addEventListener('input',()=>showFrame(Number(slider.value)));
  section.querySelector('[data-action=prev]').onclick=()=>showFrame(Number(slider.value)-1);
  section.querySelector('[data-action=next]').onclick=()=>showFrame(Number(slider.value)+1);
  section.querySelector('[data-action=pause]').onclick=()=>{videos.forEach(v=>v.pause());syncing=false;};
  section.querySelector('[data-action=play]').onclick=async()=>{
    syncing=false;
    await Promise.all(videos.map(v=>v.readyState>=2?Promise.resolve():new Promise(resolve=>{
      let done=false;const finish=()=>{if(done)return;done=true;clearTimeout(timer);v.removeEventListener('loadeddata',finish);v.removeEventListener('error',finish);resolve();};
      const timer=setTimeout(finish,8000);v.addEventListener('loadeddata',finish);v.addEventListener('error',finish);v.load();
    })));
    for(const video of videos){if(video.readyState>0)video.currentTime=0;}
    showFrame(0,false);syncing=true;
    const result=await Promise.allSettled(videos.map(v=>v.play()));
    if(result.some(r=>r.status==='rejected'))status.textContent='部分视频未能启动，请检查媒体文件，或使用各视频自带播放按钮。';
    else status.textContent='同步播放中。下方固定帧跟随第一列；可拖滑块暂停逐帧查看。';
  };
  videos[0].addEventListener('timeupdate',()=>{
    if(!syncing||videos[0].paused)return;
    const t=videos[0].currentTime;
    for(const video of videos.slice(1)){if(video.readyState>1&&Math.abs(video.currentTime-t)>.1)video.currentTime=t;}
    const frame=Math.min(9,Math.floor(t*10));if(frame!==Number(slider.value))showFrame(frame,false);
  });
  videos[0].addEventListener('ended',()=>{syncing=false;videos.forEach(v=>v.pause());showFrame(9,false);});
}
</script></body></html>'''
    page = page.replace('__CASES__', ''.join(cases_html)).replace('__ROWS__', rows).replace('__MANIFEST__', payload)
    (root / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps({'html': str(root / 'index.html'), 'cases': len(manifest['cases']), 'human_rows': len(human['cases'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
