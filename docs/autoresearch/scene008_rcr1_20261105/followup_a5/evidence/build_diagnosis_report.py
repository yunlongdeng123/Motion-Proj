from pathlib import Path
import hashlib
import html
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'report_draft'
OUT.mkdir(exist_ok=True)
(OUT/'assets').mkdir(exist_ok=True)
(OUT/'evidence').mkdir(exist_ok=True)

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
current = read(ROOT/'current/receipt.json')
old = read(ROOT/'oldclean/receipt.json')
assert current['status'] == old['status'] == 'PASS_READ_ONLY_STAGE_ABLATION'
for name in ('mask_audit_49.json', 'initial_source_proximity_reconcile_cpu.json',
             'initial_source_proximity_cpu.json', 'stage_ablation_summary_cpu.json',
             'supervision_regularization_audit.json', 'code_inventory.md'):
    if (ROOT/name).exists(): shutil.copy2(ROOT/name, OUT/'evidence'/name)
for name,data in (('current_stage_receipt.json',current),('old_single_stage_receipt.json',old)):
    (OUT/'evidence'/name).write_text(json.dumps(data,indent=2)+'\n')
shutil.copy2(ROOT/'render_stage_ablation.py', OUT/'evidence/render_stage_ablation.py')
shutil.copy2(ROOT/'build_diagnosis_report.py', OUT/'evidence/build_diagnosis_report.py')
if (ROOT/'early_single/cpu_findings.json').exists():
    (OUT/'evidence/early_single').mkdir(exist_ok=True)
    for name in ('cpu_findings.json','remote_cpu_evidence.json','retrieval_manifest.json'):
        shutil.copy2(ROOT/'early_single'/name,OUT/'evidence/early_single'/name)
    shutil.copy2(ROOT/'early_single/raw105_early_roof_grid.png',OUT/'assets/raw105_early_roof_grid.png')
raw = 105
with np.load(ROOT/'current/raw105_native_keep_a5.npz') as bank:
    mask = bank['a5_mask']
    box = bank['xyxy']
assert mask.sum() == 10773
x0,y0,x1,y1 = box
inputs = [
    ('GT', ROOT/'current/raw105_gt.png'),
    ('Full5 native 10K / KEEP', ROOT/'existing_images/raw105_full5_keep.png'),
    ('Single-a37 native 10K / KEEP', ROOT/'oldclean/raw105_native_keep.png'),
    ('Ownership native 10K / KEEP', ROOT/'current/raw105_native_keep.png'),
    ('Native + road fine / DELETE', ROOT/'current/raw105_road_fine_delete.png'),
    ('Current M2 + road / DELETE', ROOT/'current/raw105_M2_delete.png'),
]
rows = []
images = []
target = np.asarray(Image.open(inputs[0][1]).convert('RGB'),np.float32)/255.
for title,path in inputs:
    im = Image.open(path).convert('RGB')
    assert im.size == (480,320)
    array = np.asarray(im,np.float32)/255.
    mae = np.abs(array-target)[y0:y1,x0:x1][mask].mean()
    rows.append(dict(stage=title, raw=105, original_a5_pixels=int(mask.sum()),
                     observed_RGB_PNG_MAE=float(mae), source=str(path.relative_to(ROOT))))
    dest = f'raw105_{len(rows):02d}.png'
    shutil.copy2(path,OUT/'assets'/dest)
    cut = im.crop((0,165,150,300)).resize((450,405),Image.Resampling.NEAREST)
    images.append((title,cut,dest))
canvas = Image.new('RGB',(1350,906),'white')
draw = ImageDraw.Draw(canvas)
for i,(title,im,dest) in enumerate(images):
    x,y = (i%3)*450,(i//3)*453
    draw.text((x+8,y+8),title,fill='black')
    draw.text((x+8,y+24),f'a5 observed-mask PNG MAE {rows[i]["observed_RGB_PNG_MAE"]:.5f}',fill='black')
    canvas.paste(im,(x,y+48))
canvas.save(OUT/'assets/raw105_stage_grid.png')
old_replay = np.asarray(Image.open(ROOT/'oldclean/raw105_native_keep.png'))
old_existing = np.asarray(Image.open(ROOT/'existing_images/raw105_oldclean_keep.png'))
replay_max = int(np.abs(old_replay.astype(int)-old_existing.astype(int)).max())
assert replay_max <= 1
full = {r['stage']:r for r in current['rows'] if r['raw']==105}
assert full['native_delete']['a5_delta_maxabs'] == 0
assert full['M2_delete']['a5_delta_maxabs'] == 0
summary = dict(status='DIAGNOSIS_STAGE_LOCALIZED_CAUSAL_COMPONENTS_NOT_ISOLATED',training_updates=0,
    actor_alias=5, raw105_original_mask_pixels=10773, PNG_metrics=rows,
    prior_old_single_reload_max_uint8_difference=replay_max,
    current_raw105_delete_a37_RGB_change_maxabs_on_a5=full['native_delete']['a5_delta_maxabs'],
    current_raw105_M2_replacement_RGB_change_maxabs_on_a5=full['M2_delete']['a5_delta_maxabs'],
    current_raw105_native_to_full_M2_RGB_MAE_delta=full['M2_delete']['a5_RGB_MAE']-full['native_keep']['a5_RGB_MAE'],
    earliest_bracket='full5 native10K -> single-a37 native10K, before ownership/delete/road/M2/RCR',
    candidate_mechanisms=['a5 independent dynamic representation removed',
       'a5 unique initialized source positions 21358 -> 1442 (1496 static source rows)',
       'RGB branch coefficient5 dynamic ->1 static, global1 retained',
       'static-only duration and velocity-magnitude regularizers now include a5'],
    not_proven='No same-initialization single-factor retraining; unique triggering regularizer or density event is not isolated.',
    a5_visible_raw_frames=[89,112], frames113_137_not_a5=True)
(OUT/'evidence/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
table = ''.join(f'<tr><td>{html.escape(row["stage"])}</td><td>{row["observed_RGB_PNG_MAE"]:.5f}</td></tr>' for row in rows)
stage_table = ''.join(f'<tr><td>{r["raw"]}</td><td>{r["stage"]}</td><td>{r["a5_RGB_MAE"]:.6f}</td><td>{r.get("a5_delta_MAE",0):.6f}</td></tr>' for r in current['rows'])
svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 300" role="img" aria-label="a5排查组件与流程">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#466073"/></marker></defs>
<g font-family="sans-serif" font-size="17" text-anchor="middle" fill="#173549" stroke="#b5c5cf" stroke-width="1.5">
<rect x="10" y="25" width="195" height="90" rx="8" fill="#eef5fa"/><rect x="245" y="25" width="220" height="90" rx="8" fill="#eef5fa"/>
<rect x="505" y="25" width="260" height="90" rx="8" fill="#fff0e4"/><rect x="805" y="25" width="280" height="90" rx="8" fill="#eef5fa"/>
<rect x="505" y="200" width="260" height="80" rx="8" fill="#eef5fa"/><rect x="805" y="200" width="280" height="80" rx="8" fill="#eef5fa"/>
</g><g font-family="sans-serif" font-size="17" text-anchor="middle" fill="#173549">
<text x="108" y="60">原图与原始身份标签</text><text x="108" y="88">同帧 RGB / a5 mask</text>
<text x="355" y="60">Full5 原生 10K</text><text x="355" y="88">a5 独立动态层</text>
<text x="635" y="60">Single-a37 原生 10K</text><text x="635" y="88">a5 → 时间静态层：已缺损</text>
<text x="945" y="60">Ownership 原生 10K</text><text x="945" y="88">保留该缺损</text>
<text x="945" y="230">删 a37 → 门控与道路</text><text x="945" y="256">逐组件渲染，无训练</text>
<text x="635" y="230">完整 M2 替换 a6</text><text x="635" y="256">核对 a5 形状与 RGB</text>
</g><g stroke="#466073" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M205 70H240"/><path d="M465 70H500"/><path d="M765 70H800"/><path d="M945 115V195"/><path d="M805 240H770"/></g></svg>'''
(OUT/'architecture.svg').write_text(svg,encoding='utf-8')
text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>a5 车体完整性诊断</title>
<style>body{{max-width:1200px;margin:36px auto;padding:0 20px;color:#203543;font:17px/1.65 system-ui,sans-serif}}h1{{line-height:1.3}}img{{max-width:100%;height:auto}}.notice{{padding:16px;background:#fff0e4;border-left:4px solid #bc4c12}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #dce3e7;padding:7px;text-align:left}}a{{color:#0966a0}}code{{font-size:.9em}}</style>
<p><a href="../index.html">返回 RCR-1 对照与归档</a></p><h1>a5 的缺损早于删车与 M2</h1>
<p class="notice">已把最早可见回退定位在 <b>Full5 原生 10K → Single-a37 原生 10K</b>。a5 是真实蓝车；当前尚未做修复，也没有把异常蓝斑的处理套到它身上。该诊断新增训练更新为 0。</p>
<p>下面全部是 raw105、同一相机与时刻。Full5 的车身仍模糊，但车顶和挡风玻璃较完整；Single-a37 的原生 keep 图已明显缺损。随后 ownership、道路和 M2 延续了这一问题。</p>
<img src="architecture.svg" alt="输入、原生模型、删车道路与M2输出的排查组件图">
<a href="assets/raw105_stage_grid.png"><img src="assets/raw105_stage_grid.png" alt="raw105 六阶段同坐标放大对照"></a>
<h2>哪些阶段已排除为 raw105 缺损的起点</h2>
<p>本次重新加载旧 single 和当前 ownership 模型，使用原生 CUDA 渲染器逐组件渲染，保持 checkpoint 与模型参数不变。旧 single 的 raw105 重放与原先发布 PNG 的最大 8-bit 差为 {replay_max}。</p>
<ul><li>当前 ownership：native keep → delete a37，在 a5 原始标签域上的 RGB 最大差为 <b>0</b>。</li>
<li>原生 a6 + 道路精调 → 完整 M2 替换 a6，a5 标签域 RGB 最大差为 <b>0</b>。</li>
<li>道路门控/道路层有轻微影响：raw105 原生 a5 浮点 RGB MAE 为 {full['native_keep']['a5_RGB_MAE']:.6f}，最终 M2 为 {full['M2_delete']['a5_RGB_MAE']:.6f}。主要缺损在此前已存在。</li></ul>
<p>这些零差结论仅针对 raw105。raw89 的当前 a37 层在 a5 原始标签域有明显贡献，删除后也有明显变化，不能外推为所有帧都不受删车影响。</p>
<h2>表示变化中已确认的差异</h2>
<p>Full5 将 a5 作为独立动态对象；Single-a37 则只保留 a37 动态身份，把 a5 留在时间静态层。Single 是从零重训，并非 Full5 权重续训。a5 独立动态输入有 21,358 个独立源点；Single 的静态初始化仅命中 1,442 个独立位置（1,496 行，包含 54 个重复）。这些精确匹配点均未被 a37 的 0.1m 排除规则剔除，也没有被后来的 a6 转移拿走。</p>
<p>a5 仍受整图 global RGB 和 static RGB 监督，不能归因为“没有 RGB 监督”。但它的分支 RGB 名义系数从 dynamic 5 变为 static 1，并进入 static 专有的 duration/velocity 幅值正则，分别偏向长持续时间和小运动。分项归一化和 SSIM 的存在，使这个系数变化不能简单当成每像素精确 5 倍的梯度比例；时间静态层也仍有速度参数，并非位置被硬冻结。</p>
<p><b>尚未隔离：</b>稀疏初始化、表示能力、RGB 权重和静态正则各自贡献多少；也未定位某一次 SGD 或增密事件。下一项有意义的验证应针对 a5 的静态表示/初始化做单变量对照，并保留现有 a37 + a6 编辑链。</p>
<details><summary>更早的既存 Single smoke 输出：1K / 2K / 3K</summary><p>同输入、seed0、损失、学习率、初始化和增密配置的 smoke，在保存的 1K 图中尚未重建 a5 上部。这说明不必等到 10K 才能观察问题，但低步数欠拟合不能被称作“完整车体在 1K 被破坏”。额外评价节点未见显式 RNG 保护，未实测其 RNG 消耗，因此不主张后续 smoke 与 formal 的中间轨迹逐位一致；也没有 Full5 B 的同 1K 配对图来单独比较。</p><img src="assets/raw105_early_roof_grid.png" alt="旧 single 同设置 smoke1K、2K、3K 与正式10K的a5上部对照"><p><a href="evidence/early_single/cpu_findings.json">配置与早期输出核验</a> · <a href="evidence/early_single/remote_cpu_evidence.json">源码与 RNG 边界证据</a></p></details>
<h2>数值与身份边界</h2><p>下表统一由 PNG 计算 a5 原始 10,773 像素上的 RGB MAE，含量化误差；它是观测域颜色误差，不能单独衡量车体完整性。上面的同坐标图是形状判断依据。</p><table><thead><tr><th>阶段</th><th>raw105 a5 PNG MAE</th></tr></thead><tbody>{table}</tbody></table>
<p>a5（track <code>3S_GJHyCGGK4ypFPVKOOqA</code>）在 raw89–112 有原始标签，raw113–137 已无此车标签。后段左侧其他车不视作同一 a5。GT 标签不能直接等同每个高斯的永久物体归属。</p>
<details><summary>全部实际消融记录（原始 a5 标签域、浮点指标）</summary><table><tr><th>raw</th><th>stage</th><th>RGB MAE</th><th>相邻阶段 RGB 变化 MAE</th></tr>{stage_table}</table></details>
<p>证据：<a href="evidence/summary.json">诊断摘要</a> · <a href="evidence/current_stage_receipt.json">当前模型消融回执</a> · <a href="evidence/old_single_stage_receipt.json">旧 single 回执</a> · <a href="evidence/mask_audit_49.json">49 帧身份核验</a> · <a href="evidence/initial_source_proximity_reconcile_cpu.json">初始点口径核验</a> · <a href="evidence/supervision_regularization_audit.json">监督与正则核验</a> · <a href="evidence/code_inventory.md">源码入口与行号</a> · <a href="evidence/render_stage_ablation.py">实际渲染脚本</a>。</p>
<p>RCR-1 已独立归档，完整去蓝候选仍未采用；原 M2 保留。未修改任何模型、未新增训练、未关闭 GPU。</p></html>'''
(OUT/'index.html').write_text(text,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
