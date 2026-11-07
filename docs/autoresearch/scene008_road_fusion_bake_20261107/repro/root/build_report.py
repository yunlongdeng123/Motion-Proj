"""从已完成的目标/写回/保护证据建立离线交互报告。"""
from pathlib import Path
import argparse,json,shutil,html
import numpy as np
from PIL import Image
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')


def read(p):return json.loads(Path(p).read_text())


def png(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    if a.ndim==3 and a.shape[0]==3:a=a.transpose(1,2,0)
    Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).save(p)


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)


def table(rows):
    if not rows:return '<p>无可用分母。</p>'
    return '<div class="table"><table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{html.escape(str(v))}</{"th" if i==0 else "td"}>' for v in r)+'</tr>' for i,r in enumerate(rows))+'</table></div>'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    d=read(RUN/'round_decision.json');ev=read(RUN/'results/full49/receipt.json')
    for raw in range(89,138):
        with np.load(RUN/f'red_review/world_masks49_v1/{raw:03d}.npz') as z:
            m=z['world_patch_clean_road'];yy,xx=np.nonzero(m)
        box=(max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(480,int(xx.max())+9),min(320,int(yy.max())+9)) if len(xx) else (140,170,300,260)
        for state in ('keep','delete'):
            for method in ('baseline','direct','fusion'):
                copy(RUN/f'results/full49/{raw:03d}/{method}_{state}.png',out/f'frames/{method}/{raw}_{state}.png')
                crop=out/f'crops/{method}/{raw}_{state}.png';crop.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(RUN/f'results/full49/{raw:03d}/{method}_{state}.png') as im:im.crop(box).save(crop)
            source=RUN/f'results/edgeblend49/{raw:03d}/edgeblend_{state}.png'
            copy(source,out/f'frames/edgeblend/{raw}_{state}.png')
            crop=out/f'crops/edgeblend/{raw}_{state}.png';crop.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(source) as im:im.crop(box).save(crop)
        copy(RUN/f'results/full49/{raw:03d}/observed.png',out/f'frames/observed/{raw}.png')
        crop=out/f'crops/observed/{raw}.png';crop.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(RUN/f'results/full49/{raw:03d}/observed.png') as im:im.crop(box).save(crop)
        folder='projected_BUILD40' if raw not in (89,95,101,105,113,121,125,130,137) else 'projected_HELD9'
        with np.load(RUN/f'targets/{folder}/raw{raw:03d}.npz',allow_pickle=False) as z:
            for k in ('direct_rgb','fusion_rgb'):
                png(out/f'frames/targets/{k}_{raw}.png',z[k])
            mask=z['fusion_replace_mask'] if 'fusion_replace_mask' in z.files else z['fit_mask']
            png(out/f'frames/targets/domain_{raw}.png',np.repeat(mask[...,None],3,2))
        with np.load(RUN/f'targets_edgeblend_v1/{folder}/raw{raw:03d}.npz',allow_pickle=False) as z:
            png(out/f'frames/targets/edgeblend_rgb_{raw}.png',z['edgeblend_rgb'])
            from scipy.ndimage import distance_transform_edt
            weight=np.clip(distance_transform_edt(z['fusion_replace_mask'])/10,0,1)
            png(out/f'frames/targets/weight_{raw}.png',np.repeat(weight[...,None],3,2))
    evidence=out/'evidence';evidence.mkdir()
    copy(RUN/'round_decision.json',evidence/'round_decision.json')
    copy(RUN/'results/full49/receipt.json',evidence/'full49.json')
    for src in [*list((RUN/'red_review').glob('*.json')),*list((RUN/'targets').glob('*.json')),*list((RUN/'targets_edgeblend_v1').glob('*.json'))]:
        copy(src,evidence/(src.parent.name+'_'+src.name))
    for folder,name in [('targets','target_raw128_source_fusion.png'),('targets_edgeblend_v1','edgeblend_raw128_target.png'),('targets_edgeblend_v1','edgeblend_components.png')]:
        copy(RUN/folder/name,out/'figures'/name)
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 300"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#426078"/></marker></defs><style>text{font:16px sans-serif;fill:#17344d}.box{fill:#eef4fa;stroke:#64849c;stroke-width:1.5}.line{stroke:#426078;stroke-width:2;fill:none;marker-end:url(#a)}</style><rect class="box" x="15" y="20" width="230" height="76" rx="8"/><text x="35" y="50">真实道路 RGB / LiDAR</text><text x="35" y="77">相机、可见性与道路坐标</text><rect class="box" x="295" y="20" width="235" height="76" rx="8"/><text x="313" y="50">共享道路纹理空间</text><text x="313" y="77">实测内域 + 可靠边界融合</text><rect class="box" x="580" y="20" width="230" height="76" rx="8"/><text x="598" y="50">重投影成融合监督</text><text x="598" y="77">所有视图共用同一目标</text><rect class="box" x="860" y="20" width="245" height="76" rx="8"/><text x="878" y="50">三维材质写回</text><text x="878" y="77">共享父格 RGB → 四个子格</text><path class="line" d="M245 58H293M530 58H578M810 58H858"/><rect class="box" x="295" y="180" width="330" height="76" rx="8"/><text x="313" y="210">冻结车辆 / 原场景 / 道路几何</text><text x="313" y="237">固定透明度与滤波，不增大基元</text><rect class="box" x="710" y="180" width="395" height="76" rx="8"/><text x="730" y="210">统一原生深度排序与合成</text><text x="730" y="237">49 帧：保留 a37 / 只删除 a37</text><path class="line" d="M625 218H708M983 98V178"/></svg>'''
    (out/'architecture.svg').write_text(architecture)
    architecture=architecture.replace('重投影成融合监督','共享 atlas 重投影').replace('所有视图共用同一目标','可见域 10px 边缘混合')
    (out/'architecture.svg').write_text(architecture)
    cards=''.join('<article><h2>'+html.escape(c['title'])+'</h2><p>'+html.escape(c['body'])+'</p></article>' for c in d['outcomes'])
    tabs=''.join('<section><h2>'+html.escape(x['title'])+'</h2>'+table(x['rows'])+'<p class="note">'+html.escape(x.get('note',''))+'</p></section>' for x in d.get('tables',[]))
    notes=''.join('<li>'+html.escape(t)+'</li>' for t in d['limitations'])
    artifact_links=''.join('<li><a href="'+html.escape(x['href'])+'">'+html.escape(x['label'])+'</a></li>' for x in d.get('downloads',[]))
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 道路融合与三维回写</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#1c3447;font:16px/1.7 system-ui,sans-serif}main{max-width:1560px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.35}h2{font-size:21px;margin-top:0}p{margin:9px 0}section,article{background:white;padding:22px;border:1px solid #d9e3eb;border-radius:10px;margin:18px 0;min-width:0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}.cards article{margin:0}.note{font-size:14px;color:#5c6e7c}a{color:#1760a0}img{display:block;max-width:100%}.architecture{width:100%}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;background:white;padding:12px 0;z-index:1}input[type=range]{width:230px}button,select{font:inherit;padding:5px 10px;border:1px solid #b6c8d8;border-radius:5px;background:white;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.targets{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}figcaption{font-weight:600;font-size:14px;padding:8px 0}.view{width:100%;cursor:zoom-in}dialog{max-width:95vw;max-height:94vh;border:0;padding:16px;border-radius:10px}dialog img{max-height:83vh;image-rendering:auto}dialog button{float:right}.table{max-width:100%;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #dfE7ed;text-align:left;white-space:nowrap}th{background:#edf4fa}pre{white-space:pre-wrap;overflow-wrap:anywhere}.tag{font-size:13px;display:inline-block;padding:3px 9px;background:#e6eff8;border-radius:20px}li{margin:8px 0}@media(max-width:900px){main{padding:14px}.cards{grid-template-columns:1fr}.grid{grid-template-columns:1fr 1fr}.targets{grid-template-columns:1fr}h1{font-size:24px}section,article{padding:16px}}@media(max-width:500px){.grid{grid-template-columns:1fr}input[type=range]{width:170px}}
    </style><main><span class="tag">Scene008 · 同场景机制验证 · 2026-11-07</span><h1>道路融合目标 → 持久三维写回</h1><p>先在共享道路坐标中构造融合监督，再把结果拟合回同一份道路材质；最终画面仍由三维场景统一排序渲染。</p><div class="cards">CARDS</div><section><h2>本轮组件与数据流</h2><img class="architecture" src="architecture.svg" alt="真实道路、共享融合目标、三维材质写回与冻结场景统一成像"/><p class="note">未融合与融合目标共用同190帧真实道路源、同训练掩码、同参数与预算。不是在最终画面覆盖道路图片。</p></section>
    <section id="viewer"><h2>49 帧配对成像</h2><div class="controls"><label>raw <input id="frame" type="range" min="89" max="137" value="128"></label><b id="rawLabel">128</b><select id="state"><option value="delete">删除 a37</option><option value="keep">保留 a37</option></select><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="128">128 蓝圈</button><button data-raw="130">130 车轮</button><button data-raw="137">137 缺口</button><label><input id="showTargets" type="checkbox">查看融合目标</label></div><div class="grid"><figure><figcaption>前轮 AA 支撑候选</figcaption><img id="baselineImage" class="view"></figure><figure><figcaption>同源未融合目标 → 三维写回</figcaption><img id="directImage" class="view"></figure><figure><figcaption>同源融合目标 → 三维写回</figcaption><img id="fusionImage" class="view"></figure><figure><figcaption>原始实拍（删除后隐藏域无真值）</figcaption><img id="observedImage" class="view"></figure></div><div id="targetPanel" hidden><p class="note">下排是训练/诊断目标，跨编辑状态共享。域外保留原实拍；白色表示融合目标实际替换域。上排是写回三维后的真实渲染。</p><div class="targets"><figure><figcaption>同源未融合投影目标</figcaption><img id="directTarget" class="view"></figure><figure><figcaption>共享融合投影目标</figcaption><img id="fusionTarget" class="view"></figure><figure><figcaption>目标替换域</figcaption><img id="domainTarget" class="view"></figure></div></div></section>
    TABLES
    <section><h2>方法边界与剩余问题</h2><ul>NOTES</ul><p>完整配置、目标/写回误差和分帧指标见下方证据。human_verdict 保持空值；不以单一均值替代道路接缝、车体保护和遮挡的分别验收。</p></section>
    <section><h2>参考方法与实际迁移点</h2><p><a href="https://arxiv.org/html/2509.19937">GS-RoadPatching</a>：借鉴重投影与边缘融合后的图像监督，再优化三维表示。<a href="https://arxiv.org/html/2607.20789">3D-GIMP</a>：借鉴边界约束与融合目标，不引入不必要的单目深度。<a href="https://arxiv.org/html/2608.26656">CoGeo-GS</a>：借鉴先固定几何优化外观的分阶段顺序；本轮没有直接照搬邻近剪枝。<a href="https://arxiv.org/html/2610.11198">3DTexMOR</a>：借鉴共享表面纹理空间向多视图投影监督；本轮使用观测RGB，未复现其完整内禀分解与反射模型。</p></section>
    <section><h2>证据与重放</h2><ul><li><a href="evidence/round_decision.json">本轮分项结论与计算账本</a></li><li><a href="evidence/full49.json">完整49帧双状态与冻结项检查</a></li>LINKS</ul><p class="note">完整现场、原模型和传感输入继续保留在AutoDL。轻量交付不等于全部原始数据备份。AutoDL保持开机。</p></section><footer class="note">子代理仅 gpt-6.1-sol / max。蓝圈与红圈分开评价；无学习型道路生成、无完整不透明首交保证。</footer></main><dialog id="zoom"><button id="closeZoom">关闭</button><img id="zoomImage"></dialog><script>
    const frame=document.getElementById('frame'),state=document.getElementById('state');
    function update(){const raw=+frame.value,s=state.value;document.getElementById('rawLabel').textContent=raw;for(const k of ['baseline','direct','fusion'])document.getElementById(k+'Image').src='frames/'+k+'/'+raw+'_'+s+'.png';document.getElementById('observedImage').src='frames/observed/'+raw+'.png';for(const k of ['direct','fusion'])document.getElementById(k+'Target').src='frames/targets/'+k+'_rgb_'+raw+'.png';document.getElementById('domainTarget').src='frames/targets/domain_'+raw+'.png';}
    frame.addEventListener('input',update);state.addEventListener('change',update);document.querySelectorAll('[data-raw]').forEach(b=>b.onclick=()=>{frame.value=b.dataset.raw;update()});document.getElementById('showTargets').onchange=e=>document.getElementById('targetPanel').hidden=!e.target.checked;
    const zoom=document.getElementById('zoom');document.querySelectorAll('.view').forEach(x=>x.onclick=()=>{document.getElementById('zoomImage').src=x.src;zoom.showModal()});document.getElementById('closeZoom').onclick=()=>zoom.close();update();
    </script></html>'''.replace('CARDS',cards).replace('TABLES',tabs).replace('NOTES',notes).replace('LINKS',artifact_links)
    body=body.replace('<label><input id="showTargets"','<label><input id="crop" type="checkbox">放大世界补片</label><label><input id="showTargets"')
    body=body.replace('<section><h2>方法边界与剩余问题</h2>', '<section><h2>目标构造诊断（不是最终渲染）</h2><details><summary>展开 raw128 目标硬切负例与固定10px控制</summary><p class="note">B的共享chart在投影有效mask边缘仍会硬切；C额外用真实可见RGB锚定，因此目标更贴近原图。目标改善需要上方native渲染与固定边界指标确认。</p><img src="figures/target_raw128_source_fusion.png" alt="原始B目标的raw128边界负例"><img src="figures/edgeblend_raw128_target.png" alt="C目标与10像素权重诊断"><img src="figures/edgeblend_components.png" alt="C中共享atlas、视角掩码与持久三维写回的关系"></details></section><section><h2>方法边界与剩余问题</h2>')
    body=body.replace('grid-template-columns:repeat(4,minmax(0,1fr))','grid-template-columns:repeat(5,minmax(0,1fr))')
    body=body.replace('<figure><figcaption>原始实拍（删除后隐藏域无真值）', '<figure><figcaption>固定10px混合目标 → 三维写回 C</figcaption><img id="edgeblendImage" class="view"></figure><figure><figcaption>原始实拍（删除后隐藏域无真值）')
    body=body.replace('<figcaption>目标替换域</figcaption><img id="domainTarget" class="view"></figure>', '<figcaption>目标替换域</figcaption><img id="domainTarget" class="view"></figure><figure><figcaption>C：真实RGB锚定的10px目标</figcaption><img id="edgeblendTarget" class="view"></figure><figure><figcaption>C的atlas权重（并非全权使用先验）</figcaption><img id="weightTarget" class="view"></figure>')
    body=body.replace('不是在最终画面覆盖道路图片。','C额外用真实RGB做固定10px边缘锚定，视角掩码权重不等于单一世界纹理；推理仍只读取三维资产。')
    body=body.replace("const raw=+frame.value,s=state.value;", "const raw=+frame.value,s=state.value,prefix=document.getElementById('crop').checked?'crops':'frames';")
    body=body.replace(".src='frames/'+k+'/'", ".src=prefix+'/'+k+'/'")
    body=body.replace(".src='frames/observed/'", ".src=prefix+'/observed/'")
    body=body.replace("document.getElementById('domainTarget').src=", "document.getElementById('edgeblendImage').src=prefix+'/edgeblend/'+raw+'_'+s+'.png';document.getElementById('edgeblendTarget').src='frames/targets/edgeblend_rgb_'+raw+'.png';document.getElementById('weightTarget').src='frames/targets/weight_'+raw+'.png';document.getElementById('domainTarget').src=")
    body=body.replace("frame.addEventListener('input',update);", "document.getElementById('crop').addEventListener('change',update);frame.addEventListener('input',update);")
    (out/'index.html').write_text(body)
    (out/'report_data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
