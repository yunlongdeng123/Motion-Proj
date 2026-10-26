import json,sys,html,hashlib
from pathlib import Path
from PIL import Image
p=Path(sys.argv[1]);a=json.loads((p/'result.json').read_text());sp=json.loads((p/'spatial_effects.json').read_text());trace=json.loads((p/'pixel_trace.json').read_text())
assert a['source_variant']=='edit_smoke_preserve/full199'
table=''
for r in sp['rows']:
    for key,label in [('road_zero','无新道路直接贡献'),('road_low','少量道路贡献'),('road_visible','道路参与较多')]:
        d=r['domains'][key];table+=f'<tr><td>{r["raw"]}</td><td>{label}</td><td>{d["pixels"]}</td><td>{d["A_MAE"]:.5f}</td><td>{d["B_MAE"]:.5f}</td><td>{d["C_MAE"]:.5f}</td></tr>'
s='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>用户选中的保背景C：为什么有效</title><style>body{background:#edf2f7;color:#18364c;font:17px/1.7 system-ui;margin:0}main{max-width:1450px;margin:auto;padding:24px}section{background:white;border-radius:12px;border:1px solid #d6e2eb;padding:24px;margin:24px 0}h1{font-size:34px}.note{padding:16px;background:#fff2d8;border-left:5px solid #cc942e}.good{padding:16px;background:#e1f3ec;border-left:5px solid #248b70}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}figure{margin:0}img{width:100%;display:block}figcaption{padding:10px 0}select{font:inherit;padding:7px}td,th{border-bottom:1px solid #d4e0e8;padding:10px;text-align:left}table{width:100%;border-collapse:collapse}a{color:#17649b}code{overflow-wrap:anywhere}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main>
<h1>用户选中的保背景 C：为什么有效</h1><p class="note">身份已纠正：C = <code>edit_smoke_preserve/full199/checkpoint1000_from0.pth</code>。上一份报告错把“强透射版本”叫作C，并转向追求道路贡献。该绑定与研究主线已撤回，原数据保留为另一项对照。</p>
<p class="good"><strong>C有效的机制是互补：部分区域通过调整原静态opacity改善；另一部分由道路修补。</strong>评价目标是最终画面、背景保护和跨视角稳定。道路贡献用于归因，不要求它越大越好。</p><img src="architecture.svg" alt="选中C的原静态编辑、道路补充和最终质量检查组件">
<section><h2>保持你选中的 C 不变</h2><label>查看帧 <select id="raw"><option>102</option><option>129</option><option>133</option><option>137</option></select></label><div class="grid"><figure><img id="A"><figcaption>A 原静态删车</figcaption></figure><figure><img id="B"><figcaption>B 同C保背景抑制，移除道路</figcaption></figure><figure><img id="C"><figcaption>C 用户选定：保背景抑制＋道路</figcaption></figure></div><p id="numbers"></p><p>raw102：A→B→C 的参考MAE为0.1999→0.0554→0.0088。这里是固定训练状态的推理干预，新增训练0；B不是另一个独立重训模型。</p></section>
<section><h2>两种作用分别发生在哪里</h2><div class="grid"><figure><img src="road_role_102.png"><figcaption>蓝：无新道路直接贡献；黄：少量；绿：道路参与较多；灰：无可靠颜色参考</figcaption></figure><figure><img src="static_effect_102.png"><figcaption>A→B：静态抑制的颜色变化，绿为误差降低，红为升高</figcaption></figure><figure><img src="road_effect_102.png"><figcaption>B→C：道路带来的颜色变化，绿为误差降低，红为升高</figcaption></figure></div><table><tr><th>帧</th><th>区域（描述性分组）</th><th>像素</th><th>A MAE</th><th>B MAE</th><th>C MAE</th></tr>'''+table+'''</table><p>raw102道路参与较多的3006个参考像素中，2944个加入道路后变好；另外3485个无道路直接贡献的参考像素，静态抑制已经把误差从0.2904降到0.0113。二者都是C的有效组成，不能只承认其中一类。</p><p>三帧参考参与了优化；raw102总目标16021像素，仅7000有可靠参考，9021仍未知。统计不等于整个车底已正确，也不等于独立泛化通过。</p></section>
<section><h2>两处中心为何变干净</h2><p>原生CUDA贡献追踪确认：两个标记点B与C的RGB逐位相同，静态总权重约0.9999，环境约0.0001。原静态几何和颜色都没改；opacity系数改变了前后高斯的混合权重，压低原残影的重要贡献，其他原静态颜色相对增加。这个过程已改善最终画面，不能因新道路贡献0而否定。</p><p>但仍不能直接称其为正确三维路面：贡献加权高斯中心深度不等于首交面，需要留出几何和换视角检查。道路可能在联合训练时影响了门控的学习，最终推理权重为0不证明训练时道路分支可以省掉。</p><p><a href="pixel_trace.json">8点原始贡献、行号、颜色、深度与重放验证</a></p></section>
<section><h2>后续按正确的 C 推进</h2><p>保留C当前的静态复用与道路补充，先针对近处条带、剩余残影和最坏外围帧定位问题；几何可靠性单独验收。生成外观只服务剩余未知，附着在经验证几何上。不预设中心必须改由新道路负责。</p><p>三帧参考MAE：选中C 0.00760，未选强透射0.01336；49帧外围变化MAE：0.00378与0.00904。提高分支贡献不代表提高这两个质量指标。</p><p>默认完整重建仍使用原V41-B，本轮49帧重复最大差0。质量由用户评审，checkpoint依赖仍明确保留。</p><p><a href="REPORT.md">详细更正与后续设计</a> · <a href="result.json">10种干预×49帧指标</a> · <a href="scene_binding.json">选中C绑定</a> · <a href="../index.html">完整短程报告</a></p></section>
<script>const DATA='''+json.dumps(a,ensure_ascii=False)+''';const raw=document.getElementById('raw');function draw(){const r=Number(raw.value);for(const [id,key] of [['A','B_static'],['B','gate_no_road'],['C','C_full']])document.getElementById(id).src=`${key}_${r}.png`;const keys=['B_static','gate_no_road','C_full'];document.getElementById('numbers').textContent=keys.map((key,i)=>{const x=DATA.rows.find(v=>v.raw===r&&v.variant===key);return `${'ABC'[i]}：`+(x.valid_MAE===undefined?'无跨帧参考':`参考MAE ${x.valid_MAE.toFixed(5)}`)+`，目标外变化 ${x.outside_MAE.toFixed(5)}`}).join('；')}raw.onchange=draw;draw();</script></main></html>'''
(p/'index.html').write_text(s,encoding='utf-8')
root=p.parent
parent=root/'index.html';s=parent.read_text(encoding='utf-8')
if '更正后的主报告：用户选中的保背景C为什么有效' not in s:
    s=s.replace('<h1>', '<p class="good"><a href="c_selected_mechanism/index.html">更正后的主报告：用户选中的保背景C为什么有效</a></p><h1>',1)
s=s.replace('新增：C为什么有效——10种固定状态反事实与后续路线','历史误绑定：强透射版本归因（不是用户选中的C）')
s=s.replace('最新1000步：加入真实路面射线的透射约束后','未选中的对照：加入真实路面射线的透射约束后')
s=s.replace('追加：对已证实道路射线直接打开透射','未选中的强透射对照，不能替换保背景C')
s=s.replace('<option value="visibility49">最新：显式透射约束</option><option value="edit49">前一对照：保守门控</option>','<option value="edit49" selected>用户选中C：保背景版本</option><option value="visibility49">未选中的对照：强透射</option>')
s=s.replace('静态残影：图变干净不等于新道路露出','保背景C：静态复用与道路修补的共同收益')
s=s.replace('中心拟合依靠原静态透射变化。移除道路对照证明道路在其他区域有收益，不能据此宣称两个中心也由道路补好。','中心改善依靠原静态透射变化，这是C的一部分有效机制；道路在其他区域提供额外修补。不能把分支贡献大小替代最终质量目标。')
s=s.replace('本组是上轮颜色捷径之后的机制修订。','这是另一个研究目标下的历史对照，不能据此否定保背景C的实际改善。')
s=s.replace('收益已经进入最终编辑合成，但外围损伤尚未完全守住。','分支贡献增加，但参考误差与外围保护均弱于用户选中的C。')
s=s.replace('保守门控：图像改善，关键中心道路贡献仍0','用户选中C：中心复用静态，其他区域由道路补充')
s=s.replace('在独立编辑副本解决静态遮挡，再验收实际道路贡献和目标外损伤。','沿保背景C检查最终编辑质量、目标外保护和多视角稳定；贡献仅用于解释。')
parent.write_text(s,encoding='utf-8')
old=root/'c_mechanism/index.html';s=old.read_text(encoding='utf-8');s=s.replace('<h1>C 为什么有效</h1>','<h1>历史误绑定：未选中的强透射版本</h1><p class="warn">此页曾把强透射版本错认成用户选定C，归因对象不符合用户要求。保留其测量结果，不再作为研究主线。<a href="../c_selected_mechanism/index.html">打开用户选定C的正确诊断</a></p>');old.write_text(s,encoding='utf-8')
manifest=json.loads((root/'selected_c_manifest.json').read_text());images=0
for name,spec in manifest.items():
    f=root/name;assert f.stat().st_size==spec['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==spec['sha256'],name
    if f.suffix=='.png':
        with Image.open(f) as im:im.verify()
        images+=1
(p/'local_verification.json').write_text(json.dumps(dict(files=len(manifest),images=images,all_received_bytes_verified=True,browser_ui_verified=False,updates=0),indent=2))
print(json.dumps(dict(verified_files=len(manifest),decoded_images=images,selected_checkpoint=a['checkpoint'])))
