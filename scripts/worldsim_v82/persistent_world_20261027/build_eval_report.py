"""本地阶段二报告：严格区分准备、执行、质量验证，不把规格数当成绩。"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import json
import shutil
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/4D_Editing_Evaluation_20261027'
DATA = OUT / 'delivery'
manifest = json.loads((DATA / 'manifest.json').read_text())
summary = json.loads((DATA / 'summary.json').read_text())
for name in ['scene0017', 'scene0029', 'scene0008_repair10k']:
    video = DATA / name / 'delete_video.mp4'
    result = video.with_name('delete_h264.mp4')
    if video.exists() and not result.exists():
        subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-y', '-i', str(video), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-an', str(result)], check=True)

failures = [
    dict(scene='scene0008', operation='删除指定车辆 / 移动 / 复用插入', observation='中心静态残影、近处条带；部分动态行归属未决。',
         status='已有贡献追踪与图像证据；跨场景对应机制仍需验证', cause='静态污染已在该场景核实；不能推广所有黑色区域',
         evidence='../Persistent_World_Autoresearch_20261027/operations_raw102.jpg'),
    dict(scene='scene0017', operation='整动态分支删除 / 移动 / 复用插入', observation='删除原生图较干净，固定ProPainter在车原位产生暗色模糊区域；移动与复用车辆有明显蓝色拖影。',
         status='中间帧图像已检查，49帧保留供时序复核', cause='生成改坏与资产表达/编辑算子分别待定位；不直接归为缺道路几何',
         evidence='delivery/scene0017/contact.jpg'),
    dict(scene='scene0029', operation='整动态分支删除 / 移动 / 复用插入', observation='车辆消失后，原位置仍留下大块暗色轮廓；固定视频后端未去掉。插入资产存在重叠和拖影。',
         status='中间帧图像已检查，49帧保留供时序复核', cause='背景中烘焙阴影/污染是候选解释；需要贡献与真实显露对照，尚未确认几何错误',
         evidence='delivery/scene0029/contact.jpg'),
]
(OUT / 'failure_matrix.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
labels = {'previous_method_development': '原方法开发', 'previous_reconstruction_development': '历史重建开发',
          'new_diagnostic_development': '新增诊断开发', 'heldout_reserved': '封存：未跑编辑评测'}
gallery = ''.join(f'<article><img loading="lazy" src="delivery/gallery/{escape(s["context"])}.jpg"><p>{escape(s["context"])}</p><small>{labels[s["split"]]} · {escape(s["location"])} · {escape(s["time_of_day"])}<br>{s["frames"]}原始帧 / 49帧PNG；像素掩码与道路GT待补齐</small></article>' for s in manifest['scenes'])
failure_rows = ''.join(f'<tr><td>{x["scene"]}</td><td>{x["observation"]}</td><td>{x["cause"]}</td><td>{x["status"]}</td></tr>' for x in failures)
body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4D 编辑 · 评测驱动研究</title>
<style>body{{background:#f3f7fa;color:#203c51;font:16px/1.7 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1400px;margin:auto;padding:25px}}section{{background:white;border:1px solid #d8e3eb;border-radius:12px;margin:22px 0;padding:22px}}h1{{font-size:29px}}h2{{font-size:22px}}a{{color:#126c9a}}img,video{{max-width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #dce5ed;padding:12px;vertical-align:top}}.note{{background:#fff1d5;padding:16px;border-left:4px solid #b68c37}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}article{{border:1px solid #dce5ed;padding:10px;border-radius:8px}}article p{{font-size:12px;overflow-wrap:anywhere}}small{{font-size:13px}}video{{width:100%;background:#152536}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}}}</style>
<main><h1>4D 编辑评测：先发现重复失败，再改进背景修复</h1><p>2026-10-27 · 阶段二早期 · 质量待用户评审 · 持续自主推进</p>
<p><a href="../Persistent_World_Autoresearch_20261027/index.html">原开发场景证据</a> · <a href="delivery/EVALUATION_PROTOCOL.md">冻结评测合同</a> · <a href="delivery/manifest.json">场景与案例规格</a> · <a href="failure_matrix.json">失败矩阵</a></p>
<section><h2>已经做了什么，哪些仍未完成</h2><table><tr><th>层次</th><th>实际进展</th><th>不能据此宣称</th></tr>
<tr><td>数据准备</td><td>36个真实context逐记录双CRC核验；1764帧RGB和原始轨迹标注。原方法开发1段、历史重建开发2段、新开发10段、封存23段。</td><td>尚未全部成为高斯编辑案例；context之间物理路段重叠仍待审计。</td></tr>
<tr><td>任务规格</td><td>六类任务，共216份待执行规格，先划场景再选目标/窗口。</td><td>不是216项已完成实验；严格撤源、核心真值与多请求复用尚未完成。</td></tr>
<tr><td>实际运行</td><td>另外两个历史开发路段各6种原生操作×49帧，共12条操作序列；各有固定ProPainter删车视频。已完成保背景10K也补齐49帧原生与视频验收。</td><td>整动态分支操作不能冒充指定实例归属已解决；不同历史基座不能直接算方法增益。</td></tr>
<tr><td>阶段三</td><td>候选几何smoke正常保存，共3985额外更新；正式10K未启动，已撤下。视频后端保持ProPainter。</td><td>不提前选择核心算法，也不切换后端掩盖失败。</td></tr></table>
<p class="note">阶段一闭环已经运行，当前重点是阶段二。持久世界状态仍是框架目标；必须证明三维背景在换视角、反复编辑时正确，并减少③修补需求，才能形成算法贡献。</p>
<img src="delivery/evaluation_architecture.svg" alt="多场景评测、固定编辑和生成后端、三维与视频分开检查、失败驱动算法迭代"></section>
<section><h2>新增路段暴露的问题并不相同</h2><table><tr><th>场景</th><th>实际观察</th><th>候选原因与归因边界</th><th>证据状态</th></tr>{failure_rows}</table>
<p>这说明不能把所有失败统一写成“缺道路几何”。生成覆盖、背景中的对象相关暗色外观、资产换视角表达、实例归属，需要分别量化。下面是固定中间帧，完整49帧序列已保留。</p></section>'''
for name, title in [('scene0017', '路口白色轿车'), ('scene0029', '路口红色轿车')]:
    body += f'<section><h2>{title} · {name}</h2><p>上排：原生完整、删除、删除＋固定ProPainter、移动。下排：复用插入、相机侧移、重复删除。所有操作均未训练新修复算法。</p><img src="delivery/{name}/contact.jpg"><video controls preload="metadata" src="delivery/{name}/delete_h264.mp4"></video><p><a href="delivery/{name}/summary.json">逐帧原生记录</a> · <a href="delivery/{name}/video_summary.json">视频改动/时间/显存记录</a></p></section>'
body += f'''<section><h2>数据清单与接续任务</h2><p>下面只展示输入RGB，封存场景没有运行编辑效果评测。原始三维标签、像素语义、可信道路参考、原生checkpoint的就绪状态分别记录；框不作为像素掩码。</p><div class="grid">{gallery}</div>
<ol><li>补齐新增开发路段的像素掩码、米制几何与高斯输入；封存场景保持方法选择隔离。</li><li>严格撤源先移除全部关联输入再编码/初始化，分别报告边界、过渡、内部核心；做不到的只标module-only。</li><li>按完整目标与参考分母评价污染残留、保留对象、跨视角/请求一致性和视频修补负担；无标签中心仍为未知。</li><li>依据跨场景反复失败再设计②，固定①③消融；保留简单锚点、确定性补全和普通缓存强对照。</li></ol></section>
<section><h2>资源与质量状态</h2><p>同一心跳每30分钟继续。每次先查Codex周额度；剩余严格低于20%时快速保存收尾、取回必要结果并关闭AutoDL，再暂停心跳。在此之前继续自主研究，不因阶段结束或负结果关机。</p><p>当前快照周剩余57%；额度以后续工具读取为准。质量仍为quality_accepted=false、human_verdict=null。重复渲染最大差0仅为工程检查，不是世界记忆收益。</p><p>本页做了图片和文件链接校验，没有声称浏览器交互QA。<a href="delivery/summary.json">执行与指标清单</a></p></section></main></html>'''
(OUT / 'index.html').write_text(body, encoding='utf-8')
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.paths=[]
    def handle_starttag(self, tag, attrs):
        self.paths += [v for k,v in attrs if k in ['src','href'] and v and not v.startswith(('http:','https:','#'))]
parser=Links();parser.feed(body)
broken=[p for p in parser.paths if not (OUT / p).exists()]
for file in DATA.rglob('*.jpg'):
    with Image.open(file) as im:im.verify()
assert not broken,broken
result=dict(local_links=len(parser.paths),broken=broken,actual_independent_heldout_results=0,
            raw_contexts=36,prepared_pngs=1764,operation_sequences=12,all_six_families_complete=False,
            browser_render_qa=False,quality_accepted=False,human_verdict=None)
(OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
