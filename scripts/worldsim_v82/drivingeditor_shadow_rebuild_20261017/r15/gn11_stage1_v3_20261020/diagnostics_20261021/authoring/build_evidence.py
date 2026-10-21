"""新目录发行；旧GPU/CPU报告、原始科学资产不动。"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from verify import read_js,unpack
from portable import pack

ROOT=Path(__file__).resolve().parent;OUTPUTS=ROOT.parents[1]/'outputs'
FULL='GS_Reconstruction_Diagnostics_20261021_Evidence_r1'
PORTABLE='GS_Reconstruction_Diagnostics_20261021_Evidence_portable_r1'

def js(path,key,v):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(v,separators=(',',':'),ensure_ascii=False,allow_nan=False)+';',encoding='utf-8',newline='\n')

def report(e,v1,cases,cfrows):
    counts=Counter(g['label'] for t in e['traces'].values() for g in t['by_id'].values())
    text=''
    for arm,pixel,m in cfrows:
        a,b=m['valid'],m['normal_control']
        text+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (arm,pixel,a['count'],f"{a['delta_L_valid']:.6f}" if a['count'] else 'unknown',b['count'],b.get('RGB_harmed_count','unknown'),f"{b.get('alpha_change',0):.6f}"))+'</tr>'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>几何责任与有效区验收 · V3×V1</title><link rel="stylesheet" href="style.css"><main style="max-width:1200px"><h1>从贡献到可判别证据</h1><p><a href="index.html">返回五实验诊断</a> · 30K冻结 · 优化更新0 · quality_accepted=false / human_verdict=null</p>
<section class="panel"><h2>Architecture components</h2><svg viewBox="0 0 1040 170" width="100%" aria-label="冻结V1V3与测量经过实际CUDA归因、几何判别和固定世界区域反事实评价输出联动证据"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0V6L7 3Z" fill="#9fb6d3"/></marker></defs><g fill="#172b43" stroke="#6281a4"><rect x="5" y="45" width="210" height="75" rx="8"/><rect x="265" y="45" width="210" height="75" rx="8"/><rect x="525" y="5" width="245" height="65" rx="8"/><rect x="525" y="100" width="245" height="65" rx="8"/><rect x="820" y="45" width="215" height="75" rx="8"/></g><g fill="#dfebfa" text-anchor="middle" font-size="16"><text x="110" y="74">V1 / V3 冻结30K</text><text x="110" y="100">相机 / LiDAR / 世界单元</text><text x="370" y="74">原生排序 / Tα / 协方差</text><text x="370" y="100">完整掩码干预重渲染</text><text x="648" y="33">测量 + 条件深度区间</text><text x="648" y="57">冲突 / 保护 / 未知</text><text x="648" y="127">固定无遮挡世界有效区</text><text x="648" y="151">ΔL / 正常区误伤 / 拒绝</text><text x="927" y="74">五实验同帧可追责</text><text x="927" y="100">像素 ↔ 椭球 ↔ BEV</text></g><path d="M215 82H258 M475 82L518 38 M475 82L518 132 M770 38L813 82 M770 132L813 82" stroke="#9fb6d3" fill="none" stroke-width="2" marker-end="url(#a)"/></svg></section>
<section class="panel"><h2>P0 几何判别的确切边界</h2><p>实际Tα与conic沿用原生CUDA缓冲。以3D协方差、当前相机旋转、原生负FOV投影J及真实2D协方差，计算条件z均值/3σ区间；不是中心深度硬分类，也不是体积首交面。测量容差声明为LiDAR 0.20m+2%z、道路0.15m+2%z，属于保守诊断政策，不是经过标定的概率置信度。道路切平面与当前像素射线相交，要求锚点距离≤0.20m、投影≤2px。实际贡献的完整3σ区间与证据带分离才标候选冲突；重叠/稀疏/非目标身份不足均未知。</p><p>车体表面要求目标mask、OBB范围、1.5px内真实LiDAR及颜色相近（L1≤0.08），框面不冒充车体实测面；车后正确道路保护。非目标mask缺独立运动身份/表面核对时仍未知，不自动误删。标签只对应当前帧和射线；不授予永久语义或自动修复。未导出Top权重∪Top颜色之外的贡献不被分类。原116查询导出集合计数：{dict(counts)}。新增{len(cases['queries'])}个由测量独立选取的道路/车表面对照另列，不挑模型残差来决定分母。</p></section>
<section class="panel"><h2>P0 反事实科学验收：道路更可见 ≠ 更正确</h2><p>每次完整干预使用同一世界单元、同一真实来源帧RGB；九硬门限全部通过，至少两来源确认，四个双线性采样像素无遮挡（mask额外外扩1px），附近实测LiDAR匹配道路深度。所有指标使用远端原始浮点，不用便携热图/8bit图。valid在干预前固定，非有限值直接失败，不以after筛掉坏点。等权世界锚点观测，不是独立像素或面积统计。</p><p>ΔL_valid = mean(L1_before − L1_after)，正数才表示该已验证子集误差下降。正常区在干预前要求RGB L1≤0.05、α≥0.95、深度一致且在21×21邻域外；另报颜色/深度误差、α变化和损伤计数。RGB误伤门限1/255，深度误伤门限0.20m+2%z；alpha只报变化/下降，不冒充真实GT。未曝光或当前被挡查询只报覆盖/几何变化，RGB loss=null，绝不把跨帧原车像素当隐藏道路RGB。</p><div class="table-wrap"><table><thead><tr><th>组</th><th>raw102像素</th><th>valid世界数</th><th>ΔL_valid（正改善）</th><th>正常世界数</th><th>RGB误伤数</th><th>正常α变化</th></tr></thead><tbody>{text}</tbody></table></div><p>上述真实负结果保留：多数左侧/路面对照移除在曝光valid子集上反而增大误差，不能据道路贡献提升宣布修复。前车干预对曝光子集几乎无影响，也不代表长期遮挡车后的纹理正确。raw89/131/137控制均按各自同世界曝光重新评价，不再仅比较固定屏幕坐标。</p></section>
<section class="panel"><h2>P1 V1 原生GN11 30K</h2><p>来自{html.escape(v1['binding']['run'])}，历史20K恢复到30K，不称fresh。原五身份、49相机与世界变换已实际对齐；V1未注入V3道路。只读兼容前向在49帧×静态/联合/动态三分支对原训练节点PNG核验147次，最大8bit差{max(r['max_8bit_error'] for r in v1['whole_frame_replays'])}（允许1来自保存舍入，PNG无法验证未量化浮点逐位相同）；新增24条相同像素原生trace，聚合与独立ID权重重放通过。V1的高斯ID只在其自身SHA内有效，不能按行号与V3一一对应。二者配置/恢复/道路表示不同，比较用于定位新增贡献类型，不作为隔离单因素的因果消融。</p></section>
<section class="panel"><h2>来源、限制、复现</h2><p>source/gpu/evidence.py保存政策/条件分布/验收实现；evidence_receipt.json、v1_receipt.json、evidence_cases.json保留来源。界面可离线逐帧拖动；实时查询仍需start_gpu.ps1且只读。原生定义参考<a href="https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu">Graphdeco forward.cu</a>和<a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3DGS原论文/实现</a>；本项目的测量标签/门限是额外诊断，不伪称官方标准。</p><p>本轮两次V1导出工程失败均保存：首次误读cycle属性，在渲染前停止；第二次误将动态raw与填环境语义比较，门限拒绝。修正后新目录重跑；均0优化，未更改模型/原结果。原始资产不改，旧报告/失败现场均保留，不启动训练、不开旧控制器、不自动关机。</p></section></main></html>'''

def build():
    evidence=json.loads((ROOT/'evidence_r2.json').read_text());v1root=ROOT/'v1_export_r3';v1=json.loads((v1root/'v1_receipt.json').read_text());cases=json.loads((ROOT/'evidence_cases_r2/receipt.json').read_text())
    for source,name in [('GS_Reconstruction_Diagnostics_20261021_GPU_r2',FULL),('GS_Reconstruction_Diagnostics_20261021_GPU_portable_r3',PORTABLE)]:
        out=OUTPUTS/name
        if out.exists():raise ValueError('fresh output required')
        shutil.copytree(OUTPUTS/source,out,ignore=shutil.ignore_patterns('__pycache__','package_manifest.json'))
        for filename in ('app.js','gpu.js','evidence.js','server.py'):
            shutil.copy2(ROOT/'src'/filename,out/filename)
        index=read_js(out/'data/gpu_index.js');cfrows=[]
        for key,path in index['queries'].items():
            q=read_js(out/path);addition=evidence['traces'][key];q['geometry_evidence']=addition['geometry_evidence']
            for row in q['top']:row['geometry_evidence']=addition['by_id'][row['id']]
            js(out/path,key,q)
        for key,path in index['counterfactuals'].items():
            q=read_js(out/path);ids=[r['id'] for r in q['removed']]
            def matched(k):return next(v for v in evidence['counterfactuals'][k] if v['removed_ids']==ids)
            match=matched(key)
            assert match['source_float']['sha256']==q['float_asset']['sha256']
            q['scientific_evaluation']=match['scientific_evaluation']
            for t in q['temporal_controls']:
                other=matched(f'cf/{q["arm"]}/{t["raw"]}/{q["pixel"][0]}/{q["pixel"][1]}')
                t['scientific_evaluation']=other['scientific_evaluation'];t['float_asset']=other['source_float']
            js(out/path,key,q);cfrows.append((q['arm'],q['pixel'],q['scientific_evaluation']))
        for directory in (v1root,ROOT/'evidence_cases_r2'):
            for file in directory.rglob('*'):
                if file.is_file() and file.suffix in ('.webp','.js'):
                    target=out/file.relative_to(directory);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        for q in v1['queries']+cases['queries']:index['queries'][q['key']]=q['path']
        js(out/'data/gpu_index.js','gpu_index',index)
        js(out/'data/evidence_cases.js','evidence_cases',cases)
        scene=read_js(out/'data/scene.js');scene['bindings']['gn11_v1']=v1['binding']['checkpoints'];scene['historical_v1_contract']=v1['binding']['contract'];js(out/'data/scene.js','scene',scene)
        for filename,value in [('evidence_receipt.json',dict(status=evidence['status'],policy=evidence['policy'],float_sources=evidence['float_sources'],trace_count=len(evidence['traces']),optimizer_updates=0)),('v1_receipt.json',v1),('evidence_cases.json',cases)]:
            (out/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
        text=(out/'index.html').read_text(encoding='utf-8').replace('<script src="app.js">','<script src="data/evidence_cases.js"></script><script src="app.js">').replace('<script src="gpu.js"></script>','<script src="gpu.js"></script><script src="evidence.js"></script>')
        text=text.replace('四组同帧对照','五实验同帧对照').replace('V3 四组诊断','V3 × GN11 V1 诊断')
        (out/'index.html').write_text(text,encoding='utf-8',newline='\n')
        (out/'EVIDENCE_REPORT.html').write_text(report(evidence,v1,cases,cfrows),encoding='utf-8',newline='\n')
        (out/'README.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>五实验诊断 · 使用说明</title><link rel="stylesheet" href="style.css"><main style="max-width:1000px"><h1>V3四组 × V1原生GN11</h1><p><a href="index.html">打开系统</a> · <a href="EVIDENCE_REPORT.html">几何与反事实报告／组件图</a></p><section class="panel"><h2>离线使用</h2><p>直接打开index.html，不联网。选择五实验、raw89—137逐帧拖动；四组V3加V1同步对照。V1没有专用道路分支，相关项明确不适用；共享BEV是外部测量参照，不是给V1补道路。表中点击Gaussian定位椭球；“几何证据判别”提供当前射线标签、条件深度带和原因。</p><p>固定样例新增两个实测无遮挡道路控制与两个目标车LiDAR表面对照。原raw102三处反事实有完整重渲染和当前/跨帧有效世界区域误差；ΔL正数才表示验证子集改善。正常区域误伤独立列出。未知RGB真值始终为null。</p></section><section class="panel"><h2>实时只读</h2><p>运行start_gpu.ps1并打开其127.0.0.1地址，依赖wm-vgpu-1011和空闲GPU。按钮按需执行冻结模型前向、真实排序归因、选中集合完整重渲染；不训练、不创建optimizer、不改检查点、不启动旧控制器、不自动关机。V1支持实时GPU归因；原CPU参考仍只支持四组V3。Ctrl+C结束本启动器，释放自己的SSH worker。</p></section><section class="panel"><h2>解释边界</h2><p>标签是当前射线的保守候选，不是永久语义或自动删除许可。3σ条件深度是线性代理，不是光栅器首交面。非目标身份/稀疏深度不足时未知。V1历史恢复与V3科研条件不同，不当单因素因果比较。便携包480×320显示WebP质量78、热图4×4/8bit；科学指标、Tα和测量判别均由原始未量化数据计算。quality_accepted=false、human_verdict=null。</p></section></main></html>''',encoding='utf-8',newline='\n')
        for folder in ('gpu','src','tests'):
            shutil.copytree(ROOT/folder,out/'source'/('viewer' if folder=='src' else folder),dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(Path(__file__),out/'source/authoring/build_evidence.py')
        if name==PORTABLE:
            # 限制只作用新便携显示副本，不修改指标输入、模型、原生帧尺寸或旧发行版。
            for raw in range(89,138):
                path=out/f'data/gn11_v1/{raw}.js';arrays=unpack(read_js(path));mapped={};info={}
                for k,a in arrays.items():
                    a=a[::4,::4]
                    if 'depth' in k:
                        valid=a!=65535;d=np.where(valid,a.astype(float)/100,0);mapped[k]=np.where(valid,np.round(np.log1p(d)/np.log(656.34)*254)+1,0).astype(np.uint8);info[k]=dict(log_depth=True,nearest_upsample=4)
                    elif k=='body_mask':mapped[k]=a;info[k]=dict(nearest_upsample=4)
                    else:mapped[k]=np.round(a.astype(float)/257).astype(np.uint8);info[k]=dict(multiplier=257,nearest_upsample=4)
                js(path,f'gn11_v1/{raw}',pack(mapped,info))
            # 980+ native-frame display derivatives keep480x320; original float metrics untouched.
            for directory in ('full','real_only','prior_only','ownership_only','gn11_v1'):
                for path in (out/'assets'/directory).rglob('*.webp'):
                    with Image.open(path) as im:im.convert('RGB').save(path,format='WEBP',quality=78,method=6)
            (out/'display_compression.json').write_text(json.dumps(dict(display_only=True,webp_quality=78,width=480,height=320,
                metrics_use_remote_original_float=True,original_models_and_reports_modified=False,all_five_experiments49frames=True),indent=2))
        print(out,flush=True)

if __name__=='__main__':build()
