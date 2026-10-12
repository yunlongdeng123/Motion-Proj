"""Build a local evidence report. No remote training or checkpoint mutation."""
import datetime, hashlib, html, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs/drivingeditor_vgpu_debug'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'evidence').mkdir(exist_ok=True)
(OUT/'assets').mkdir(exist_ok=True)
for name in ['static20k_frame10.png','dynamic20k_frame10.png']:
    shutil.copy2(HERE/name,OUT/'assets'/name)
evidence_files={
 'events_loss_trends_cpu.json':HERE/'loss_audit/events_loss_trends_cpu.json',
 'road_observation_coverage.json':HERE/'sgcm_data_audit/road_observation_coverage.json',
 'checkpoint_geometry.json':HERE/'sgcm_data_audit/checkpoint_geometry.json',
 'sgcm_log_summary.json':HERE/'sgcm_data_audit/sgcm_log_summary.json',
 'mask_alignment_49frames.json':HERE/'sgcm_data_audit/mask_alignment_49frames.json',
 'task_manifest.json':HERE/'task_manifest.json',
}
for name,path in evidence_files.items():
    if path.exists():shutil.copy2(path,OUT/'evidence'/name)
for frame in ['0017010','0017017']:
    p=HERE/f'sgcm_data_audit/coverage_evidence_{frame}.jpg'
    if p.exists():shutil.copy2(p,OUT/'assets'/p.name)
events=json.loads(evidence_files['events_loss_trends_cpu.json'].read_text())
psnr=np.asarray(events['scalar_points_step_value_walltime']['train/psnr'])
plt.rcParams.update({'figure.facecolor':'#111a29','axes.facecolor':'#111a29','axes.edgecolor':'#43556b','text.color':'#e9f0fa','axes.labelcolor':'#ced8e6','xtick.color':'#aebdd0','ytick.color':'#aebdd0','grid.color':'#304154','font.size':10})
fig,ax=plt.subplots(figsize=(11,3.7),layout='constrained')
ax.plot(psnr[:,0],psnr[:,1],color='#70d9c8',alpha=.17,lw=.4,label='Per-step train view')
means=np.array([[a[:,0].mean(),a[:,1].mean()] for a in np.array_split(psnr,len(psnr)//50)])
ax.plot(means[:,0],means[:,1],color='#70d9c8',lw=1.5,label='50-step averages (different sampled views)')
for n in range(12000,36001,3000):ax.axvline(n,color='#f6b85a',alpha=.35,lw=.8)
ev=np.array(events['eval_scalars']['train/loss_viewpoint - psnr'])
ax.scatter(ev[:,0],ev[:,1],color='#ff758b',s=50,zorder=5,label='49-view saved checkpoint evaluation')
ax.set(xlabel='Iteration (original 50k schedule)',ylabel='Composite PSNR (dB)',ylim=(5,42))
ax.grid(alpha=.35);ax.legend(loc='lower left',fontsize=8,facecolor='#111a29',labelcolor='#e9f0fa')
fig.savefig(OUT/'assets/reset_trend.png',dpi=170);plt.close(fig)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M +08:00')
links=''.join(f'<a href="evidence/{html.escape(n)}">{html.escape(n)}</a>' for n in evidence_files if (OUT/'evidence'/n).exists())
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DrivingEditor · vGPU 论文版排错</title>
<style>:root{color-scheme:dark;--bg:#080e17;--card:#111a29;--line:#29384c;--text:#e8eef7;--muted:#9caec5;--mint:#70d9c8;--amber:#f6b85a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1220px;margin:auto;padding:38px 28px 70px}header{padding:28px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(135deg,#142338,#101725)}h1{font-size:32px;line-height:1.25;margin:12px 0}h2{margin:0 0 16px;font-size:22px}h3{margin:0 0 12px}p{margin:10px 0}a{color:var(--mint)}small,.muted{color:var(--muted)}.badge{display:inline-block;border:1px solid #8c682e;color:var(--amber);padding:2px 10px;border-radius:20px;font-size:13px}.section{margin-top:24px;padding:24px;background:var(--card);border:1px solid var(--line);border-radius:14px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.metric{padding:15px;background:#0a1421;border:1px solid var(--line);border-radius:10px}.metric strong{font-size:26px;color:var(--mint)}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted)}figure{margin:0}img{max-width:100%;border-radius:8px}figcaption{font-size:13px;color:var(--muted)}.warn{border-left:3px solid var(--amber);padding:10px 16px;background:#241e13}.links{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}svg{width:100%;height:auto}code{font-size:12px;overflow-wrap:anywhere}details{margin-top:12px}@media(max-width:760px){main{padding:18px 12px}.grid{grid-template-columns:1fr}header,.section{padding:18px}table{display:block;overflow:auto}h1{font-size:25px}}</style>
<main><header><span class="badge">排查中 · 尚未通过论文效果验收</span><h1>DrivingEditor：vGPU 论文版排错</h1><p>当前证据指向两个独立问题：<b>30k 重置后的低透明度快照</b>，以及<b>更早发生的静态／动态分支混淆</b>。后者使道路进入动态层，即使整体重建清晰，移除编辑仍不可靠。</p><small>更新时间：__NOW__ · 仅推进首场景排错，其它场景队列已暂停。</small><p><a href="../drivingeditor_30k_edit_comparison/index.html?reference=3090-original">查看 3090 原始 30k 与当前版本的深色对比页</a> · <a href="../drivingeditor_checkpoint_comparison/index.html">10k–30k 同步视频</a></p></header>
<section class="section"><h2>排查的数据流</h2><svg viewBox="0 0 1100 230" role="img" aria-label="RGB、LiDAR、mask输入静态与动态高斯，环境贴图共同合成，分别检查分支和编辑输出"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#70d9c8"/></marker></defs><g stroke="#405471" fill="#142338" stroke-width="1.5"><rect x="8" y="73" width="170" height="90" rx="10"/><rect x="255" y="10" width="215" height="84" rx="10"/><rect x="255" y="128" width="215" height="84" rx="10"/><rect x="539" y="75" width="150" height="88" rx="10"/><rect x="754" y="75" width="142" height="88" rx="10"/><rect x="942" y="75" width="150" height="88" rx="10"/></g><g fill="none" stroke="#70d9c8" stroke-width="2" marker-end="url(#arr)"><path d="M178 108L215 108L215 52L255 52"/><path d="M178 133L215 133L215 172L255 172"/><path d="M470 52L505 52L505 100L539 100"/><path d="M470 172L505 172L505 140L539 140"/><path d="M689 120L754 120"/><path d="M896 120L942 120"/></g><g fill="#e8eef7" text-anchor="middle" font-size="17" font-family="system-ui,Microsoft YaHei"><text x="93" y="107">49 帧 FRONT</text><text x="93" y="137" font-size="13">RGB / LiDAR / mask</text><text x="362" y="41">静态高斯 + SGCM</text><text x="362" y="72" font-size="13">天空监督 / NPS / 时变属性审计</text><text x="362" y="160">动态高斯</text><text x="362" y="191" font-size="13">RGB / 不透明度 / 支撑区域</text><text x="614" y="109">环境贴图</text><text x="614" y="139" font-size="13">分支贡献与版本核对</text><text x="825" y="109">完整合成</text><text x="825" y="139" font-size="13">固定视角 / 完整损失</text><text x="1017" y="109">分层与编辑</text><text x="1017" y="139" font-size="13">背景清晰 / 目标局部性</text></g></svg><p class="muted">整体图像正确不等于分支正确；验收同时检查静态道路、动态层范围及移除／平移结果。</p></section>
<section class="section"><h2>四种假说，分别找证据</h2><table><thead><tr><th>用户提出的方向</th><th>当前证据</th><th>判断范围</th></tr></thead><tbody><tr><td>错误环境贴图或混用高斯 checkpoint</td><td>已保存的新 30k 四件模型、固定相机及 196 张导出 PNG 的一致性检查通过；20k／30k 环境统计和单组件替换诊断进行中。</td><td>尚无错配证据；不把渲染相等误当训练算法正确。</td></tr><tr><td>训练与导出不一致</td><td>新 30k 重渲染与保存时评估逐像素一致。旧清晰 3090 历史评估在 reset 前；旧实际 saved 30k 和新 30k 都在 reset 后。</td><td>已确认对照快照时序不同；训练／推理的共享环境合成语义另查。</td></tr><tr><td>优化退化与分支混淆</td><td>20k 整体 PSNR 35.25 dB，但静态道路已有彩噪，动态层覆盖道路。30k 静态点透明度均 ≤0.01。</td><td>已存在独立的分支分离问题；不能只修 checkpoint 时机。</td></tr><tr><td>道路从未观测</td><td>几何抽查 raw0／10／17 的车体投影地面样本中，93.56%／99.41%／100% 有其它帧可见性及真实地面 LiDAR 支持。</td><td>这些被抽查区域无法用“从未观测”解释；该检查不是逐像素无遮挡真值。</td></tr></tbody></table></section>
<section class="section"><h2>20k 已出现：合成清晰，但分层错误</h2><div class="grid"><figure><img src="assets/static20k_frame10.png" alt="20k raw10 静态层道路为彩色噪点"><figcaption>真实 20k 视频第 10 帧 · 静态层。道路明显退化。</figcaption></figure><figure><img src="assets/dynamic20k_frame10.png" alt="20k raw10 动态层包含道路背景"><figcaption>同一步、同一帧 · 动态层。车外道路和背景进入动态表示。</figcaption></figure></div><p class="warn">这不是只发生在 30k 透明度重置后的现象。以整体 RGB 重建分数替代编辑验收，会漏掉这类问题。</p></section>
<section class="section"><h2>30k 的瞬时低质与重置时机有关</h2><img src="assets/reset_trend.png" alt="原始 TensorBoard 训练PSNR与静态透明度重置、checkpoint评估趋势"><p>黄色竖线是每 3,000 步的静态透明度重置；密集训练数据为随机采样视角，粉色点为保存模型后的完整 49 视角评估。两种口径不同，不能逐点直接比较。</p><div class="grid"><div class="metric"><strong>35.84 → 24.41 → 32.04</strong><br><small>训练 PSNR 均值：29,900–30,000 → 30,001–30,100 → 30,201–30,300</small></div><div class="metric"><strong>10.48 dB</strong><br><small>实际 saved 30k 完整评估，位于 reset 之后；固定视角短续训验证待完成。</small></div></div><p>原始代码的静态 reset 没有随 15k 增密结束而停止。新论文流程在 reset／optimizer 后保存及评估；旧 3090 清晰历史输出取自 reset 前。两者必须明确区分。</p></section>
<section class="section"><h2>道路并非一概不可观测</h2><div class="grid"><figure><img src="assets/coverage_evidence_0017010.jpg" alt="raw10被车遮挡的地面点重投影到其它真实帧"><figcaption>raw10 地面样本及另一真实帧的投影支持。</figcaption></figure><figure><img src="assets/coverage_evidence_0017017.jpg" alt="raw17被车遮挡的地面点在车离开后可见"><figcaption>raw17 对应地面在车离开后的真实道路上可见。</figcaption></figure></div><p class="muted">采用拟合地面、4 像素网格、真实 LiDAR 支持及遮挡筛查。标记是几何审计结果，不是生成的道路纹理，也不构成 FID 或编辑区真值。</p></section>
<section class="section"><h2>下一轮受控实验</h2><table><thead><tr><th>实验</th><th>只改变什么</th><th>要回答的问题</th></tr></thead><tbody><tr><td>20k／30k 固定相机组件诊断</td><td>推理时逐组件检查；交叉组合全部单独标记</td><td>环境亮度、静态透明度、动态遮挡各贡献多少？</td></tr><tr><td>完整 30k 状态短续训</td><td>先不修改训练数学；恢复全部 optimizer、RNG 和 sampler，保持 50k 学习率计划</td><td>固定视角何时恢复？静态／动态分离是否仍坏？</td></tr><tr><td>确定实现 bug 的单变量修复</td><td>正在检查 LiDAR 同像素写入次序；验证后单独对照</td><td>监督目标是否错了，以及其实际影响多大？</td></tr><tr><td>SGCM／NPS／动态监督诊断</td><td>每次一项，保留基准对照及源码／配置版本</td><td>哪一项导致或放任道路进入动态层？</td></tr></tbody></table><p class="warn">尚未证明修复完成。论文未公开的权重、采样邻域、SGCM 调度，以及公式与文字的歧义，需要分别标为实现选择或待证假说；不会把恢复旧损失后更好看的结果直接称为论文复现。</p></section>
<section class="section"><h2>版本、边界与证据</h2><p>诊断对象：官网 silver-car，Waymo 1.4.3，FRONT raw0–48，49 帧；原计划从零 50k，3090 的 10k 完整状态迁移到 32GB vGPU，后来主动停止。当前仅检查该论文版。使用原 FP32 renderer/kernel，compact 与 renderer cleanup 候选未部署。</p><p>目前使用 DEVA/SAM 车身 mask，缺 shadow mask 与 M-detector；尚未达到论文四场景主表、完整消融或完整算法效果复现。没有无遮挡真值的移除区域，不宣称质量已获真值验证。</p><p><a href="../drivingeditor_paper_alignment/index.html">论文协议审计</a> · <a href="file:///C:/Users/dengyunlong/Downloads/DrivingEditor.pdf">本地 DrivingEditor.pdf</a> · <a href="../drivingeditor_30k_edit_comparison/diagnostic/index.html">保存模型间的历史诊断</a></p><details><summary>可下载的原始审计数据</summary><div class="links">__LINKS__</div></details><small>报告图像来自实际模型结果及实际数据。当前图表由原始事件日志生成；浏览器交互播放尚未作为本报告的验证证据。</small></section></main></html>'''.replace('__NOW__',now).replace('__LINKS__',links)
component=HERE/'component_diagnostic'
if (component/'diagnostic_manifest.json').exists() and (component/'cases/baseline20/case_manifest.json').exists():
    cases=[]
    for p in sorted((component/'cases').glob('*/case_manifest.json')):
        d=json.loads(p.read_text())
        vals=[f['regions']['full']['metrics']['composite']['clamped_rgb_psnr_db'] for f in d['frames']]
        cases.append((d['case_id'],float(np.mean(vals))))
        dest=OUT/'evidence/component_cases'/p.parent.name
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest/p.name)
    for name in ['cpu_audit.json','diagnostic_manifest.json','train_eval_semantics.json','no_modification_sha.json']:
        if (component/name).exists():shutil.copy2(component/name,OUT/'evidence'/name)
    for name in ['static_foreground_contribution_20k_0017010.png','static_env_contribution_20k_0017010.png']:
        if (component/name).exists():shutil.copy2(component/name,OUT/'assets'/name)
    labels={'baseline20':'原 20k：S20 + D20 + E20','baseline30':'原 30k：S30 + D30 + E30',
        's20_d20_e30_ENV_ONLY_DIAGNOSTIC':'诊断混配：仅将 20k 环境换为 30k',
        's30_d30_e20_ENV_ONLY_DIAGNOSTIC':'诊断混配：仅将 30k 环境换回 20k',
        's20_d30_e30_BRANCH_MIX_DIAGNOSTIC':'诊断混配：仅将 30k 静态高斯换回 20k',
        's30_d20_e30_BRANCH_MIX_DIAGNOSTIC':'诊断混配：仅将 30k 动态高斯换回 20k'}
    rows=''.join(f'<tr><td>{labels[k]}</td><td>{v:.2f} dB</td></tr>' for k,v in cases)
    section='''<section class="section"><h2>组件替换实验：主要变化来自静态分支</h2><p>实际 6 组 × 5 个固定相机已完成。两组原始基线的四个通道共 40 张 PNG 与保存评估逐像素相同；源文件与 checkpoint 的 64 项 hash／修改时间检查通过。</p><table><tr><th>实际组件</th><th>固定 5 帧 composite PSNR 均值</th></tr>'''+rows+'''</table><p>换回环境贴图不足以修复 30k；换回静态高斯则明显恢复。所有混配仅用于隔离原因，不作为训练结果或论文复现成果。这里是 5 帧诊断口径，不能替换上方 49 帧评估。</p><div class="grid"><figure><img src="assets/static_foreground_contribution_20k_0017010.png" alt="20k静态高斯颜色贡献很暗"><figcaption>20k raw10：静态高斯本身的颜色贡献。</figcaption></figure><figure><img src="assets/static_env_contribution_20k_0017010.png" alt="20k静态层环境贴图残差贡献包含道路彩噪"><figcaption>同一帧：环境贴图乘以 (1 − 静态 alpha) 的贡献。道路彩噪在此项中。</figcaption></figure></div><p>由此将“静态图彩噪”进一步定位到低静态不透明度暴露的环境项，而非仅凭图片归因于高斯 SH 颜色错误。训练／评估固定相同光线方向后，composite、static、alpha 和 depth 一致；动态通道的差别符合共享环境项公式。</p><div class="links"><a href="evidence/diagnostic_manifest.json">GPU 诊断记录</a><a href="evidence/train_eval_semantics.json">训练／推理合成核对</a><a href="evidence/no_modification_sha.json">原文件未修改校验</a></div></section>'''
    page=page.replace('<section class="section"><h2>下一轮受控实验</h2>',section+'<section class="section"><h2>下一轮受控实验</h2>')
    page=page.replace('20k／30k 环境统计和单组件替换诊断进行中。','20k／30k 环境统计和 6 组单组件对照已完成，详见下方。')
    page=page.replace('训练／推理的共享环境合成语义另查。','固定光线下训练／推理的共享环境合成已核对。')

depth=HERE/'loss_audit/sparse_depth_collisions_cpu.json'
if depth.exists():
    shutil.copy2(depth,OUT/'evidence'/depth.name)
    page=page.replace('正在检查 LiDAR 同像素写入次序；验证后单独对照','已确认 LiDAR 同像素写入次序依赖；最近正深度候选需 GPU 单变量对照')
    extra='''<section class="section"><h2>已定位的实现问题与算法风险</h2><table><tr><th>项目</th><th>证据</th><th>当前状态</th></tr><tr><td>LiDAR 深度像素取最后写入点</td><td>raw0／10／48 有效深度像素中，11.52%／11.73%／8.09% 取到比最近表面更远的点；逆序输入会改变目标。</td><td>确定的顺序依赖。最近正深度 z-buffer 候选已准备；其是否解释视觉退化尚未验证。</td></tr><tr><td>SGCM 反复克隆不参与成像的点</td><td>20k 额外重复坐标约 999,266 个；最大簇 158,570 个。部分大簇在整个训练时间范围内 alpha 都低于光栅器 1/255 阈值。</td><td>已证明无效重复与资源浪费；不能据此单独认定道路彩噪来源。独立资格筛查候选尚未部署。</td></tr><tr><td>静态模型实际会随时间变化</td><td>20k 中约 61.3 万静态点在实际时间范围内的位置跨度超过 0.1 米，约 103.3 万发生时间激活阈值切换。</td><td>与论文 Eq.4 的 time-invariant 文字存在差距；论文基础 PVG 公式及 v-map 解释也需一并核对，不能直接断言硬冻结即为作者实现。</td></tr><tr><td>动态掩码外缺少分支约束</td><td>实际损失函数的逐像素反例中，正确分层与“动态负责全图／静态消失”可取得相同极低损失。</td><td>短窗300步真实单变量干预已改善背景分离，但仍有动态背景残留；长窗V待验证。新增mask外项仍是诊断假设，并非已核实的作者原始公式。</td></tr></table><p><a href="evidence/sparse_depth_collisions_cpu.json">深度投影的真实场景测量</a></p></section>'''
    page=page.replace('<section class="section"><h2>版本、边界与证据</h2>',extra+'<section class="section"><h2>版本、边界与证据</h2>')
page=page.replace('<text x="614" y="109">环境贴图</text><text x="614" y="139" font-size="13">分支贡献与版本核对</text>', '<text x="614" y="109">三路光栅化</text><text x="614" y="139" font-size="13">共享 EnvMap</text>')
for source_name, output_name in [('loss_audit/handoff_summary.json','loss_audit_summary.json'),('sgcm_data_audit/audit_findings.json','sgcm_audit_findings.json'),('component_diagnostic/compact_case_summary.json','component_case_summary.json'),('opacity_pair_protocol.json','opacity_pair_protocol.json')]:
    src=HERE/source_name
    if src.exists():shutil.copy2(src,OUT/'evidence'/output_name)
page=page.replace('<details><summary>可下载的原始审计数据</summary>', '<p><a href="evidence/loss_audit_summary.json">损失审计与论文逐条对照</a> · <a href="evidence/sgcm_audit_findings.json">SGCM／数据／时间属性审计</a> · <a href="evidence/component_case_summary.json">分层区域指标</a> · <a href="evidence/opacity_pair_protocol.json">单变量实验的预先约定</a></p><details><summary>可下载的原始审计数据</summary>')
continuation_path=HERE/'continuation_summary.json'
if continuation_path.exists():
    continuation=json.loads(continuation_path.read_text(encoding='utf8'))
    shutil.copy2(continuation_path,OUT/'evidence'/continuation_path.name)
    terminal_rows=[]
    selected={'image_0/0017000','image_0/0017010','image_0/0017020','image_0/0017030','image_0/0017048'}
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    image_blocks=[]
    for case in continuation['cases']:
        label='原样续训' if case['case']=='original30k_to30300' else '增加动态掩码外约束（诊断）'
        label_en='Original objective' if case['case']=='original30k_to30300' else 'Outside-opacity hypothesis'
        nodes=case['nodes']
        for ax,metric,title in zip(axes,['composite_psnr_full','static_psnr_observed_non_sky_outside_actor'],['Composite / fixed5 views','Static / observed non-sky outside actor, fixed5 views']):
            xy=[(node['iteration'],np.mean([f[metric] for f in node['frames'] if f['frame_id'] in selected])) for node in nodes]
            ax.plot([p[0] for p in xy],[p[1] for p in xy],marker='o',ms=3,label=label_en)
            ax.set(xlabel='Iteration',ylabel='PSNR (dB)',title=title);ax.grid(alpha=.3)
            ax.legend(fontsize=7,facecolor='#111a29',labelcolor='#e9f0fa')
        for node in nodes:
            if node['iteration']!=30300 or node['frame_count']!=49:continue
            m=node['means']
            terminal_rows.append(f'<tr><td>{label}</td><td>{m["composite_psnr_full"]:.2f}</td><td>{m["static_psnr_observed_non_sky_outside_actor"]:.2f}</td><td>{m["alpha_dynamic_observed_non_sky_outside_actor"]:.4f}</td><td>{m["alpha_static_observed_non_sky_outside_actor"]:.4f}</td><td>{100*m["dynamic_rgb_support_observed_non_sky_outside_actor"]:.2f}%</td></tr>')
            cards=[]
            for channel,cn in [('composite','整体重建'),('static','静态层'),('dynamic','动态层')]:
                source=HERE/'continuation_diagnostic'/case['case']/'extra_eval/30300'/channel/'0017010.png'
                if source.exists():
                    name=case['case']+'_'+channel+'_raw10.png'
                    shutil.copy2(source,OUT/'assets'/name)
                    cards.append(f'<figure><img src="assets/{name}" alt="{label}30300raw10{cn}"><figcaption>{cn} · 同一真实 raw10</figcaption></figure>')
            if cards:image_blocks.append(f'<h3>{label} · 30,300 步</h3><div class="grid" style="grid-template-columns:repeat(3,minmax(0,1fr))">'+''.join(cards)+'</div>')
    fig.savefig(OUT/'assets/controlled_continuation_trend.png',dpi=170);plt.close(fig)
    if terminal_rows:
        part='''<section class="section"><h2>完整状态短续训：恢复整体画质是否也恢复分层？</h2><p>同一正式 30k 四件 checkpoint，恢复完整 Adam、RNG 与 sampler；学习率 horizon 仍是 50k。原样对照第 30,001 步的 10 个历史标量与此前训练日志完全相同。额外评估前后恢复随机状态，末次检查全部 49 帧。</p><table><tr><th>30,300 步实际结果</th><th>整体 PSNR（dB）</th><th>静态 PSNR（dB）*</th><th>动态 α*</th><th>静态 α*</th></tr>'''+''.join(terminal_rows)+'''</table><p class="muted">* 当前车体掩码以外、非天空的已观测区域；表中各项均为 49 帧逐帧指标的算术均值。整体和静态使用不同区域，不能直接相减解释。没有对车体遮挡区虚构真值。</p><img src="assets/controlled_continuation_trend.png" alt="同5个固定相机的短续训整体和静态质量曲线"><p class="muted">曲线始终取同 5 帧；末端也从 49 帧评估中筛选这 5 帧，不混用不同相机数量。</p>'''+''.join(image_blocks)+'''<p class="warn">掩码外项是单变量机制检验，并非作者确认的公式。原样续训中的整体质量恢复并不代表编辑已修复；新增项即使改善，也需要更长训练与剩余协议核验。</p><a href="evidence/continuation_summary.json">完整逐帧指标及配对差值</a></section>'''
        page=page.replace('<section class="section"><h2>下一轮受控实验</h2>',part+'<section class="section"><h2>下一轮受控实验</h2>')
        page=page.replace('固定视角短续训验证待完成。','原样短续训的全 49 帧验证已完成，详见下方。')
page=page.replace('<th>静态 α*</th></tr>', '<th>静态 α*</th><th>动态非黑像素占比*</th></tr>')
page=page.replace('没有对车体遮挡区虚构真值。</p>', '动态非黑支撑使用原诊断定义：RGB 四舍五入到 8 位后，任一通道大于 2 的像素比例。没有对车体遮挡区虚构真值。</p>')
if continuation_path.exists():
    terminal={c['case']:next((n for n in c['nodes'] if n['iteration']==30300 and n['frame_count']==49),None) for c in continuation['cases']}
    if all(terminal.get(k) for k in ['original30k_to30300','opacity_outside30k_to30300']):
        a=terminal['original30k_to30300']['means'];b=terminal['opacity_outside30k_to30300']['means']
        result=f'''<section class="section"><h2>当前结论：找到改善分层的方向，尚未修复完成</h2><p>同一 30k 状态只加入掩码外约束，300 步后完整 49 帧的已观测背景静态 PSNR 提高 {b['static_psnr_observed_non_sky_outside_actor']-a['static_psnr_observed_non_sky_outside_actor']:.2f} dB；整体 PSNR 下降 {a['composite_psnr_full']-b['composite_psnr_full']:.2f} dB。实际 raw10 图像中静态道路已明显恢复，动态图仍含道路光斑和背景纹理，静态图也仍有模糊与残影。</p><p>这支持当前实现缺少动态掩码外约束的诊断。接下来从相同原始 30k 状态做三组 2,000 步检验：原版、原版重复、同一 λ=0.05 候选；停在 32k，避开下一次 33k reset。两份原版用于测量同初态的数值波动，不是独立随机种子的置信区间。执行进度以实际记录为准。</p><p class="warn">目前没有正常效果或完整论文复现的结论。长实验还需检查组合画质、已观测静态背景、目标车辆保真、车外泄漏和真实移除／平移；不能只用某个指标变好验收。</p></section>'''
        page=page.replace('<section class="section"><h2>版本、边界与证据</h2>',result+'<section class="section"><h2>版本、边界与证据</h2>')
refs=HERE/'loss_audit/long_pair_30k_32000/official_git_refs.json'
if refs.exists():
    shutil.copy2(refs,OUT/'evidence/official_git_refs.json')
    part='''<section class="section"><h2>实现来源：哪些是原代码，哪些仍需验证</h2><p>本次核对官方仓库的全部公开 heads／tags，只找到 2024-12-29 的 main（560d57c…），未找到正式论文版本分支或标签。因此这里的“论文版”指我们按 2026 论文实现的版本，不能视为作者已发布并验证的实现。</p><p>晚期透明度重置、深度投影顺序依赖有直接代码或运行证据；新增掩码外约束是分层退化的单变量假说。论文未写清的求和区域、权重与优化细节仍需单独核验，图像变好本身不能证明公式完全复现。</p><a href="evidence/official_git_refs.json">公开分支、标签及提交时间核对</a></section>'''
    page=page.replace('<section class="section"><h2>版本、边界与证据</h2>',part+'<section class="section"><h2>版本、边界与证据</h2>')
attempt=HERE/'continuation_diagnostic/attempts/opacity_outside_01_runtime_image_utils_import/failure_receipt.json'
if attempt.exists():
    shutil.copy2(attempt,OUT/'evidence/diagnostic_observer_import_failure.json')
    page=page.replace('<details><summary>可下载的原始审计数据</summary>', '<p class="muted">实验记录：掩码外约束诊断的首次启动因 PSNR 工具导入路径错误，在恢复／训练之前退出。已保留失败证据，修正观察代码并独立复核后重试；训练公式、系数及原始模型没有因此修改。<a href="evidence/diagnostic_observer_import_failure.json">失败与修复记录</a></p><details><summary>可下载的原始审计数据</summary>')
short_receipt_links=[]
for short_case in ['original30k_to30300','opacity_outside30k_to30300']:
    src_root=HERE/'continuation_diagnostic'/short_case
    dst_root=OUT/'evidence'/short_case
    for rel in ['receipts/full49_summary.json','first_backward_reference_comparison.json','runtime_import_pins.json','terminal_30300/diagnostic_snapshot_manifest.json']:
        src=src_root/rel
        if src.exists():
            dst=dst_root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            short_receipt_links.append(f'<a href="{dst.relative_to(OUT).as_posix()}">{html.escape(short_case)} / {html.escape(Path(rel).name)}</a>')
if short_receipt_links:
    page=page.replace('<details><summary>可下载的原始审计数据</summary>', '<details><summary>短实验恢复、真实源码、完整节点与四件状态证据</summary><div class="links">'+''.join(short_receipt_links)+'</div></details><details><summary>可下载的原始审计数据</summary>')
paired_video=OUT/'paired_video_validation.json'
if paired_video.exists():
    receipt=json.loads(paired_video.read_text())
    media=OUT/'assets'/receipt['file']
    if receipt['status']=='all49_frames_decoded' and media.exists():
        part=f'''<section class="section"><h2>同一时间轴的真实 49 帧对照</h2><video controls muted playsinline preload="metadata" style="width:100%;background:#000;border-radius:8px" src="assets/{html.escape(receipt['file'])}"></video><p>上排：原样续训；下排：仅增加动态掩码外约束的诊断。每排从左到右为 GT、整体重建、静态层、动态层。两组相同相机帧被编码在同一视频画面内，避免多个播放器不同步。</p><p class="muted">两组均为实际 30,300 步，49 帧、10 fps、4.9 秒，已逐帧解码检查。MP4 仅用于观看；指标来自未压缩的原始浮点结果。浏览器播放本身未验证。</p><a href="paired_video_validation.json">视频与原 PNG 来源校验</a></section>'''
        page=page.replace('<section class="section"><h2>下一轮受控实验</h2>',part+'<section class="section"><h2>下一轮受控实验</h2>')
from render_long_original_report import add_long_original_evidence
page,long_b1_complete=add_long_original_evidence(HERE,OUT,page)
from render_actual_temporal_report import add_actual_temporal_evidence
page=add_actual_temporal_evidence(HERE,OUT,page)
from render_actual_S_report import add_actual_S_evidence
page,long_S_complete=add_actual_S_evidence(HERE,OUT,page)
from render_initialisation_gap_report import add_initialisation_gap_evidence
page=add_initialisation_gap_evidence(HERE,OUT,page)
task_path=HERE/'task_manifest.json'
if task_path.exists():
    task=json.loads(task_path.read_text(encoding='utf8'))
    shutil.copy2(task_path,OUT/'evidence/debug_task_status.json')
    if task.get('current_experiment_status')=='B1_running_actual_restore_and_first_step_exact' and not long_b1_complete:
        snapshot=html.escape(task['updated_utc'])
        progress=f'''<section class="section"><h2>正在运行：32k 三组验证的第一组</h2><p>截至 {snapshot}（UTC），B1 原数学对照已在 wm-vgpu-1011 启动，完整 30k 状态恢复与首步 10 项标量逐项相等核验通过。目标是 32k，学习率 horizon 保持 50k；这不是重启正式 50k 训练。</p><p>随后是相同初态的原版重复 B2，再验证相同掩码外约束 V。两组尚未启动；每次只占用一个 GPU 任务。其他场景暂停，每小时继续排错检查。</p><a href="evidence/debug_task_status.json">运行与恢复记录（此页生成时的快照）</a></section>'''
        page=page.replace('<section class="section"><h2>四种假说，分别找证据</h2>',progress+'<section class="section"><h2>四种假说，分别找证据</h2>')
(OUT/'index.html').write_text(page,encoding='utf-8')
manifest={'updated':now,'status':'diagnosis_not_complete','files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='report_manifest.json'}}
(OUT/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':str(OUT/'index.html'),'files':len(manifest['files']),'status':'diagnosis_not_complete'},ensure_ascii=False))
