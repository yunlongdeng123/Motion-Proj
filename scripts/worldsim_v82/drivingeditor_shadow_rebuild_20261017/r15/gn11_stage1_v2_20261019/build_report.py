"""Full-frame scientific input preview and candidate report; no learned results."""
from pathlib import Path
import datetime,hashlib,json,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from world_road import select_asphalt_donors

W=Path(__file__).resolve().parent
BASE=W.parent.parent
O=BASE/'outputs/gn11_stage1_v2_20261019'
O.mkdir(parents=True,exist_ok=True)
cfg=json.loads((W/'config.json').read_text(encoding='utf-8'))
test=json.loads((W/'cpu_test_receipt.json').read_text(encoding='utf-8'))
if test['status']!='PASSED_CPU_BEHAVIOR_CONTRACTS':raise RuntimeError('CPU checks not passed')
source=BASE/'work/baseline_engineering_20261018/GN11_stage1_full49/0008089/gt.png'
rgb=np.asarray(Image.open(source).convert('RGB'));assert rgb.shape==(320,480,3)
maskpath=BASE/'work/identity_completion/metadata/native_masks_for_review/0008089.npz'
with np.load(maskpath,allow_pickle=False) as z:masks=z['masks']
assert masks.shape==(320,480,2)
query=masks.any(2)
polys=cfg['road']['donor_polygons_raw89']
accepted,regions,excluded=select_asphalt_donors(rgb,polys,query)
Image.fromarray(rgb).save(O/'raw89_original.png')
preview=rgb.copy()
for i,color in enumerate([(240,85,215),(181,117,255)]):
    edge=masks[:,:,i]&~binary_erosion(masks[:,:,i],iterations=1)
    preview[edge]=color
im=Image.fromarray(preview);d=ImageDraw.Draw(im)
for i,poly in enumerate(polys):
    d.line([tuple(p) for p in poly]+[tuple(poly[0])],fill='#ffb43b',width=2)
    x,y=poly[0];d.rectangle((x,y-13,x+58,y-1),fill='#111b28');d.text((x+2,y-13),'DONOR '+str(i+1),fill='#ffb43b')
im.save(O/'raw89_query_and_donors.png')
overlay=np.asarray(im).copy()
overlay[accepted]=(0.6*overlay[accepted]+0.4*np.array([38,195,255])).astype(np.uint8)
reject=excluded&(regions>=0)
overlay[reject]=(0.35*overlay[reject]+0.65*np.array([255,72,72])).astype(np.uint8)
Image.fromarray(overlay).save(O/'raw89_donor_filter_preview.png')
counts=[int(((regions==i)&accepted).sum()) for i in range(2)]
receipt=dict(status='LOCAL_FULLFRAME_DONOR_COLOR_FILTER_PREVIEW_ONLY',image_wh=[480,320],
    input_unannotated_RGB_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    mask_sha256=hashlib.sha256(maskpath.read_bytes()).hexdigest(),
    query_actors=[6,37],query_body_pixels_by_actor=[int(masks[:,:,i].sum()) for i in range(2)],
    donor_polygons=polys,color_filter_candidate_pixels=counts,
    measured_road_support_checked=False,all_occluder_check_complete=False,
    true_world_query_projection_executed=False,actual_cross_frame_fusion_executed=False,
    actual_prior_completion_executed=False,training_started=False,GPU_used=False,
    note='Cyan candidates still require measured geometry/all-occluder verification. Magenta outlines are query rays on body silhouettes, not yet world-projected road patches.')
(O/'donor_preview_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
for name in ['config.json','cpu_test_receipt.json','candidate_manifest.json','evaluation_contract.json']:
    shutil.copyfile(W/name,O/name)
svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 520" role="img" aria-label="GN11 v2 双模块结构图">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="none" stroke="#8fadc9"/></marker></defs>
<rect width="1100" height="520" fill="#101b29"/>
<g fill="#1d3044" stroke="#49728f" stroke-width="1.5">
<rect x="20" y="180" width="185" height="118" rx="12"/><rect x="255" y="35" width="260" height="115" rx="12"/>
<rect x="565" y="35" width="240" height="115" rx="12"/><rect x="255" y="230" width="260" height="105" rx="12"/>
<rect x="565" y="210" width="240" height="100" rx="12"/><rect x="565" y="355" width="240" height="100" rx="12"/>
<rect x="860" y="180" width="220" height="140" rx="12"/>
</g><g stroke="#8fadc9" stroke-width="2" fill="none" marker-end="url(#a)">
<path d="M205 211H230V92H255"/><path d="M205 267H255"/><path d="M515 92H565"/>
<path d="M515 262H565"/><path d="M685 310V355"/><path d="M805 92H833V215H860"/>
<path d="M805 255H860"/><path d="M805 405H833V295H860"/>
</g><g fill="#e6eff7" font-family="Microsoft YaHei, sans-serif" font-size="19" text-anchor="middle">
<text x="112" y="211">相同五身份 / 49 帧</text><text x="112" y="240">RGB、mask、LiDAR</text><text x="112" y="269">位姿 / 标定 / 测量框</text>
<text x="385" y="65">① 静动态归属</text><text x="385" y="96">严格静态 + 原生动态</text><text x="385" y="127">车身监督路由 / 体积排除</text>
<text x="685" y="65">归属目标</text><text x="685" y="96">可见背景 / 动态车身</text><text x="685" y="127">独立计算与记录</text>
<text x="385" y="260">② 世界坐标道路查询</text><text x="385" y="289">raw89 遮挡道路 → 曲面</text><text x="385" y="318">同一位置 → raw131–137</text>
<text x="685" y="240">可见性检查 + 真实 RGB 融合</text><text x="685" y="270">先后段，再检查全 49 帧</text><text x="685" y="297">来源 = 同位置真实观测</text>
<text x="685" y="384">无真实候选 → 沥青供体</text><text x="685" y="414">去标线 / 道路坐标纹理先验</text><text x="685" y="444">来源 = 异位置先验</text>
<text x="970" y="213">道路目标只更新静态</text><text x="970" y="247">每步一次优化器更新</text><text x="970" y="280">30K → 完整 49 帧评估</text>
<text x="550" y="497" font-size="16" fill="#a7bed2">两套目标分别计算、限制梯度接收方；真实观测、先验和未知分别输出。</text>
</g></svg>'''
(O/'architecture.svg').write_text(svg,encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GN11 第一阶段 v2 · 候选实现与道路查询</title>
<style>body{margin:0;background:#0b1420;color:#e5edf6;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.3}h2{font-size:24px;margin-top:42px}h3{font-size:19px}a{color:#7dd3ff}p{max-width:1000px}.status{padding:20px;border:1px solid #b98c3b;background:#302916;border-radius:12px}.muted{color:#a8bdd1}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card,figure{margin:0;padding:18px;background:#132131;border:1px solid #2d4255;border-radius:12px}img{width:100%;height:auto;object-fit:contain;display:block}figure img{max-width:960px;margin:auto}figcaption{font-size:14px;color:#b6c8d8;margin-top:12px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px;text-align:left;border-bottom:1px solid #334659;vertical-align:top}code{color:#9de4db}button{padding:9px 16px;background:#203a51;border:1px solid #587a97;color:white;border-radius:7px;cursor:pointer;margin:0 8px 14px 0}button[aria-pressed=true]{background:#235c7c}.tag{color:#ffcb68}footer{margin-top:45px;color:#a8bdd1}@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:24px 14px}h1{font-size:27px}}</style>
<main><p class="muted">A 新五身份 GN11 · FIRST STAGE v2 · raw89–137</p>
<h1>先解决静态里的车辆解释，<br>再找回同一世界位置的道路</h1>
<div class="status"><strong>本地候选代码与 CPU 检查完成；尚未开始新版训练。</strong><br>
AutoDL 待用户开机。真实 LiDAR / 相机输入绑定、CUDA 零更新前后向检查和完整 30K 训练均待执行。下方是输入与供体预览，不是修复后的渲染结果。</div>
<p>用户已确认：相同五身份、相同 49 帧、相同输入初始化与 seed=0，重新训练第一阶段 30K；第二阶段 0。原版参数与历史资产保留。静态时间自由度的关闭是本版明确的方法变更，不能称为完全相同的模型。</p>
<h2>模块与数据流</h2><figure><img src="architecture.svg" alt="输入、静动态归属、道路世界坐标查询、真实观测融合、缺失区域供体和完整评估的模块箭头图"><figcaption>两套目标分开定义、记录和计算梯度。道路梯度仅累积到静态参数，与归属梯度在一次原生更新前汇合；没有增加隐藏训练步数。</figcaption></figure>
<div class="grid" style="margin-top:20px"><section class="card"><h3>① Occlusion-aware Static–Dynamic Disentanglement</h3>
<p>静态点保持固定位置、固定时间可见性，停止静态时间位移、寿命和速度参数训练；增密、SGCM 后继续施加该约束。动态类方法保持原样。</p>
<p>车身 RGB、深度和相关图像正则通过动态路径；静态使用可见背景监督。选定车辆的数据集 3D 标注框内增加静态不透明度排除，保护道路上方 0.2m 薄层。这些框是明确使用的额外归属 / 遮挡约束输入，不是独立评估真值。没有把车身 2D mask 内的整层静态 alpha 清零。</p>
<p class="muted">体积项是中心测试，无法保证排除大 Gaussian 的尾部、错误深度或框外重复解释。它是待验证修复候选，不是已证明完全分离。</p></section>
<section class="card"><h3>② Temporally Revealed Road Reconstruction</h3>
<p>raw89 中 a6、a37 的车身遮挡射线与 LiDAR 支撑的道路高度场相交，定义世界坐标 Query Patch。先投影到 raw131–137，再检查同窗口其余帧；不加入窗外训练图。</p>
<p>通过视野、遮挡 mask、3D 框射线、稀疏实测深度、路面支撑、角度、分辨率和清晰度检查后，在 linear RGB 中做加权一致性融合，至少两帧确认。</p>
<p class="muted">一个真实候选或候选颜色冲突时保持待定，不用供体盖掉；没有任何合格真实候选的区域才允许供体先验。当前没有得到真实 raw89 查询曝光比例。</p></section></div>
<h2>raw89：Query Patch 与供体框分别标记</h2>
<p>原图 480×320、左上角 (0,0)。紫 / 粉轮廓是 a6、a37 车身遮挡射线的入口；橙框是用户指定的可见沥青供体。世界道路投影、真实曝光与未曝光分类待服务器输入绑定后显示，当前不把车身轮廓伪装成已恢复的道路。</p>
<div role="group" aria-label="全画面输入显示"><button data-image="raw89_query_and_donors.png" aria-pressed="true">Query 与供体</button><button data-image="raw89_donor_filter_preview.png" aria-pressed="false">供体剔线候选</button><button data-image="raw89_original.png" aria-pressed="false">未画框原图</button></div>
<figure><img id="frame" src="raw89_query_and_donors.png" alt="raw89 完整观测画面，紫粉标记车身 query 入口，橙框标记左右供体"><figcaption id="caption">所有显示保留完整 480×320 画面。橙框仅定位，取色输入是未画框的原图。</figcaption></figure>
<table><thead><tr><th>区域</th><th>坐标（左上 → 右下）</th><th>排除</th><th>当前颜色过滤候选</th></tr></thead><tbody>
<tr><td>左供体</td><td>(49,225) → (185,284)</td><td>双黄线、左上白线及边缘</td><td>COUNT_LEFT 像素</td></tr>
<tr><td>右供体</td><td>(352,227) → (398,298)</td><td>斜白线、自行车标记边缘</td><td>COUNT_RIGHT 像素</td></tr></tbody></table>
<p>青色表示颜色 / 边缘过滤后的候选，红色表示剔除的标线与边缘。以上数量<strong>尚未</strong>通过真实道路几何与全部遮挡物核验，不是实际供体用量。服务器准备阶段还需实测路面支撑；无支撑时停止准备，不能凭颜色认定沥青。</p>
<p>供体纹理按道路坐标采样，保留剔线后的内部纹理，不整框粘贴。当前先验采用周期性道路坐标重采样，可能产生重复纹理或接缝；始终标记为异位置先验，权重低于真实观测，不称为隐藏道路真值。</p>
<h2>来源与训练边界</h2><table><thead><tr><th>分类</th><th>证据</th><th>训练和展示</th></tr></thead><tbody>
<tr><td>真实观测</td><td>同一世界点、至少两帧通过可见性与颜色一致性检查</td><td>保留逐点来源帧 / UV / 权重；训练时只约束静态道路</td></tr>
<tr><td>供体先验</td><td>合格真实候选数为 0；用户供体内剔线、几何检查通过</td><td>低权重补全；保留供体位置，不能计入真实恢复率</td></tr>
<tr><td>未知 / 待定</td><td>单一候选、颜色冲突、无法构造有效道路交点或供体不可用</td><td>零道路外观监督；完整画面中单独标记</td></tr></tbody></table>
<p>道路表示为测量高度场：局部地面平面加 LiDAR 残差。缺支撑区域仅有平面几何先验；若使用供体，其几何与纹理都不算实测恢复。稀疏 LiDAR 邻近像素不能充当稠密深度真值。所有 49 帧是训练窗口，评估属于重建检查，不是独立测试集泛化结论。</p>
<h2>已验证与开机后执行</h2><p><strong>TEST_COUNT 项 CPU 行为检查通过</strong>：静态时间不变、学习率与增密后继续冻结、车身监督梯度路由、道路梯度接收方、未知区零梯度、世界投影与曲面、遮挡检查、稳健融合、供体剔线和来源保护。候选源码完成语法检查和替换位置核验；这些检查未调用 CUDA。</p>
<ol><li>开机后先只读绑定原 A 五身份的真实有效配置、输入、mask、坐标和全 49 帧，生成逐点证据与全幅预览；失败则保留证据并定位。</li>
<li>隔离部署候选；核对显存、磁盘及 GPU 锁，执行真实 CUDA 前向 / 反向零更新检查。相机固定、时间改变时静态 RGB 必须保持不变，模型梯度必须有限。</li>
<li>通过后按已授权相同初始化 fresh 30K 执行。保留 5K/10K/15K/20K/25K/30K 完整状态与原生评估，无第二阶段、无额外消融、无自动重放。</li>
<li>最终零更新输出全 49 帧完整 / 动态 / 静态 / 删除图及 alpha，和原 A 第一阶段对比；实际检查静态车形、动态质量、道路真实恢复、先验区域与阴影，报告不能仅凭 loss 宣布通过。</li></ol>
<h2>参考工作与实际迁移范围</h2><ul>
<li><a href="https://arxiv.org/html/2503.13176v2">DeGauss</a>（<a href="https://github.com/BatFaceWayne/DeGauss">官方代码</a>）：参考静态 / 动态表示解耦与遮挡归属思路；本候选不是其完整概率合成与重置策略复现。</li>
<li><a href="https://arxiv.org/html/2502.11801v1">3DGIC</a>（<a href="https://github.com/peterjohnsonhuang/3dgic">官方代码</a>）：参考跨视图深度与遮挡一致性；该工作包含图像补全，本候选的真实观测融合与沥青先验明确区分，未接生成模型。</li>
<li><a href="https://arxiv.org/html/2405.14342v1">RoGS</a>（<a href="https://github.com/fzhiheng/RoGS">官方代码</a>）：参考显式道路表面；当前使用 LiDAR 高度场，不声称复现其完整 2D Gaussian 道路网格。</li></ul>
<h2>代码与凭据</h2><p><a href="candidate_source.zip">候选代码与许可证 ZIP</a> · <a href="config.json">配置</a> · <a href="cpu_test_receipt.json">CPU 检查</a> · <a href="candidate_manifest.json">源码覆盖清单</a> · <a href="donor_preview_receipt.json">供体预览证据边界</a> · <a href="evaluation_contract.json">最终评估合同</a></p>
<p><a href="../baseline_engineering_20261018/index.html">查看已完成的旧 A/B 报告</a>。旧结果与本候选分开保存。旧 automation-2 保持暂停，旧队列不恢复；本版尚未启动训练，human_verdict=null。</p>
<footer>构建 UTC：BUILD_TIME。仅文件、源码与本地 PNG 检查；未宣称浏览器交互验收或训练效果通过。</footer></main>
<script>const captions={"raw89_query_and_donors.png":"紫 / 粉：车身 query 射线入口；橙：可见沥青供体。世界投影尚未执行。","raw89_donor_filter_preview.png":"青：颜色过滤后的沥青候选；红：剔除标线与边缘。真实道路 / 全部遮挡物核验尚待执行。","raw89_original.png":"未画框 raw89 原始观测图。标注图不会进入取色。"};document.querySelectorAll('button[data-image]').forEach(b=>b.addEventListener('click',()=>{document.getElementById('frame').src=b.dataset.image;document.getElementById('frame').alt=captions[b.dataset.image];document.getElementById('caption').textContent=captions[b.dataset.image];document.querySelectorAll('button[data-image]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));}));</script></html>'''
html=html.replace('COUNT_LEFT',str(counts[0])).replace('COUNT_RIGHT',str(counts[1])).replace('TEST_COUNT',str(test['tests'])).replace('BUILD_TIME',now)
(O/'index.html').write_text(html,encoding='utf-8')
with zipfile.ZipFile(O/'candidate_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.write(W.parent/'baseline_engineering_20261018/source/LICENSE_gaussian_splatting.md','upstream_original/LICENSE_gaussian_splatting.md')
    for p in sorted(W.iterdir()):
        if p.suffix in {'.py','.json'}:z.write(p,p.name)
    for p in sorted((W/'candidate').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(W).as_posix())
print(json.dumps(dict(output=str(O/'index.html'),candidate_pixels=counts,CPU_tests=test['tests'],training_started=False)))
