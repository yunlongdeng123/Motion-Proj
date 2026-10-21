"""Fresh GPU release; old CPU releases and original report remain untouched."""
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
from verify import read_js
from run_gpu_export import js

ROOT=Path(__file__).resolve().parent;OUTPUTS=ROOT.parents[1]/'outputs'
GPU=ROOT/'gpu_export_r1';VIEWS=ROOT/'gpu_views_r1'
report=json.loads((GPU/'gpu_receipt.json').read_text());views=json.loads((VIEWS/'view_receipt.json').read_text())
assert report['query_count']==116 and report['counterfactual_count']==12 and len(views['views'])==25

def writejs(path,key,value):
    path.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(value,separators=(',',':'),allow_nan=False)+';',encoding='utf-8')

def materialize(value,out):
    if isinstance(value,str) and value.startswith('data:image/webp;base64,'):
        data=base64.b64decode(value.split(',',1)[1]);name=hashlib.sha256(data).hexdigest()+'.webp'
        path=out/'assets/gpu'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(data)
        return 'assets/gpu/'+name
    if isinstance(value,dict):return {k:materialize(v,out) for k,v in value.items()}
    if isinstance(value,list):return [materialize(v,out) for v in value]
    return value

def report_html():
    rows=[]
    for c in report['counterfactuals']:
        m=c['metrics'];rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [c['arm'],c['pixel'],c['removed'],f"{m['selected_pixel_road_before']:.5f} → {m['selected_pixel_road_after']:.5f}",f"{m['selected_pixel_L1_change']:.5f}",f"{m['outside_ROI_RGB_change']:.5f}"] )+'</tr>')
    maximum=max(max(q['trace_errors'].values()) for q in report['queries'])
    ids=[q['id_weight_validation'] for q in report['queries'] if q['id_weight_validation']]
    max_id=max(v['max_weight_error'] for v in ids)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GPU诊断技术报告</title><link rel="stylesheet" href="style.css"><main style="max-width:1200px"><h1>冻结30K · 可追责诊断系统</h1><p><a href="index.html">返回系统</a> · 2026-10-21 · 四组／49训练帧／零优化更新</p><section class="panel"><h2>Architecture components</h2><svg viewBox="0 0 1000 160" width="100%" role="img" aria-label="冻结模型与测量进入原生前向，排序trace和完整干预分支产生四组联动证据"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0V6L7,3Z" fill="#a7bedb"/></marker></defs><g fill="#172b43" stroke="#6281a4"><rect x="5" y="40" width="200" height="70" rx="8"/><rect x="260" y="40" width="210" height="70" rx="8"/><rect x="525" y="5" width="210" height="65" rx="8"/><rect x="525" y="90" width="210" height="65" rx="8"/><rect x="790" y="40" width="205" height="70" rx="8"/></g><g fill="#e4efff" text-anchor="middle" font-size="16"><text x="105" y="68">30K + 相机 + 测量</text><text x="105" y="92">SHA绑定 / 只读</text><text x="365" y="68">原生全模型前向</text><text x="365" y="92">不创建optimizer</text><text x="630" y="33">实际排序缓冲区Trace</text><text x="630" y="57">Tα / 颜色 / ID核验</text><text x="630" y="117">掩码副本 / 新视角</text><text x="630" y="141">完整重新光栅化</text><text x="892" y="68">四组 / 像素 / BEV</text><text x="892" y="92">椭球 / 干预 / 案例</text></g><path d="M205 75H252 M470 75L517 38 M470 75L517 123 M735 38L782 75 M735 123L782 75" fill="none" stroke="#a7bedb" stroke-width="2" marker-end="url(#arrow)"/></svg></section>
<section class="panel"><h2>实际验证</h2><p>116个原生缓冲区CUDA查询，覆盖四组、五种分支、raw89/102/131/137、残影／路面对照／非空动态像素；RGB、alpha、depth、T最大重放误差 {maximum:.3g}。{len(ids)}个非空查询的Top20（不足20则全部）ID由独立原生one-hot特征通道核验，最大权重误差 {max_id:.3g}。这些是本组固定查询的实测结果，不外推所有像素／硬件逐位相同。</p><p>12个全帧Gaussian移除干预，另有36个跨帧控制；原始浮点与绑定记录在远端独立runs目录。20幅原相机五分支图像（其中ordinary原先没有独立保存基线，不谎称旧数据重放），另5幅真实新视角。16幅有旧基线的原相机全帧RGB均实际重放通过。3个负合同测试拒绝镜像相机、错误checkpoint ID、raw138。CPU8项回归通过；浏览器验收单独记录。</p></section>
<section class="panel"><h2>固定Top20干预：如实保留负结果</h2><p>每个raw102像素按颜色贡献选20个普通背景Gaussian；只改保留mask副本。道路列为该像素实际道路Tα，RGB变化为前后L1；区域外为21×21邻域以外变化。数值变化不是错误清除比例，也不是隐藏纹理质量。没有选择性挑选好看的干预。</p><div class="table-wrap" style="max-height:none"><table><thead><tr><th>实验</th><th>像素</th><th>移除数</th><th>道路贡献 前→后</th><th>像素RGBΔ</th><th>区域外RGBΔ</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>例如full左侧像素道路贡献仅约0.122→0.140；不能据此声称黑影已被修复。prior_only同像素道路贡献增加时，RGB反而更暗。局部移除也可影响大量其他像素，应结合完整图、测量和曝光来源判断，不自动删除模型点。</p></section>
<section class="panel"><h2>方法与限制</h2><p>使用本机已冻结的DrivingEditor光栅器；调试扩展不替换原生依赖，只读取真实投影／conic、tile排序、颜色缓冲区，执行相同alpha门限、上限与提前终止表达式。逐查询聚合重放与独立ID权重核验共同约束归因。<a href="https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu">Graphdeco原始实现</a>与<a href="https://docs.gsplat.studio/main/apis/rasterization.html">gsplat投影／排序说明</a>作技术参考，未替换本项目渲染器。</p><p>所有干预完整重新预处理、排序、光栅化和环境合成，不做RGB相减。中心z不是椭球首交点；正常遮路不当错误。源谱系不是唯一寿命ID。没有训练／梯度历史的旧节点不补造。来源权重／梯度不当因果质量。新视角无GT，正常区域损伤未经GT验证；跨帧相同屏幕像素不等于同世界位置。</p><p>原训练模型、优化器、科学配置、共享PCA和全部评价资产不改动。quality_accepted=false，human_verdict=null。研发曾遇Ninja PATH未加入和误把split路径的行过滤用于dynamic路径，两次均在有效归因输出前失败，零训练更新，现场与修复记录保留。</p></section></main></html>'''

for old,name in [('GS_Reconstruction_Diagnostics_20261021_r1','GS_Reconstruction_Diagnostics_20261021_GPU_r2'),('GS_Reconstruction_Diagnostics_20261021_portable_r2','GS_Reconstruction_Diagnostics_20261021_GPU_portable_r3')]:
    target=OUTPUTS/name
    if target.exists():raise ValueError('fresh release required')
    shutil.copytree(OUTPUTS/old,target,ignore=shutil.ignore_patterns('__pycache__','package_manifest.json'))
    for file in ('index.html','app.js','gpu.js','style.css','layout.css','server.py','start_local.ps1','start_gpu.ps1'):
        shutil.copy2(ROOT/'src'/file,target/file)
    if (target/'portable_receipt.json').exists():
        app=target/'app.js';text=app.read_text(encoding='utf-8').replace('数值预览alpha 1/65535、深度1cm量化','便携热图4×4采样／8bit；精确数值见CPU／CUDA射线');app.write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'src/GPU_README.html',target/'README.html')
    for export in (GPU,VIEWS):
        for file in (export/'data').rglob('*.js'):
            if file.name=='gpu_index.js':continue
            payload=read_js(file);text=file.read_text(encoding='utf-8');key=json.loads(text.split('window.GSDATA[',1)[1].split(']=',1)[0])
            dest=target/file.relative_to(export);dest.parent.mkdir(parents=True,exist_ok=True)
            writejs(dest,key,materialize(payload,target))
    index=read_js(GPU/'data/gpu_index.js');index['views']={v['key']:v['path'] for v in views['views']}
    writejs(target/'data/gpu_index.js','gpu_index',index)
    shutil.copy2(GPU/'gpu_receipt.json',target/'gpu_receipt.json');shutil.copy2(VIEWS/'view_receipt.json',target/'view_receipt.json')
    (target/'GPU_REPORT.html').write_text(report_html(),encoding='utf-8')
    shutil.copytree(ROOT/'gpu',target/'source/gpu',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'src',target/'source/viewer',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'tests',target/'source/tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('run_gpu_export.py','run_gpu_views.py','assemble_gpu.py','verify_gpu.py'):
        if (ROOT/f).exists():shutil.copy2(ROOT/f,target/'source/authoring'/f)
    (target/'source/gpu_failures.json').write_text(json.dumps(dict(failures=[dict(reason='Ninja binary not in PATH',stage='extension loader',optimizer_updates=0),dict(reason='dynamic=True keeps all rows; split=True filtering was mistakenly inferred',stage='row-mapping assertion before trace',optimizer_updates=0)],fixes=['prepend actual conda executable bin and CUDA bin','use actual dynamic row order with marginal mask'],research_assets_modified=False),indent=2),encoding='utf-8')
    print(str(target),flush=True)
