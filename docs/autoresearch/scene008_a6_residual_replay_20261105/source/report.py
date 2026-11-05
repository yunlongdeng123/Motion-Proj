"""Package the a6 residual replay and separate street transfer check."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent
PREV=ROOT.parent/'scene008_a6_local_fusion_20261105'
PILOTS=(105,113,130)
ARCH='''<svg viewBox="0 0 910 165" role="img" aria-label="剩余遮挡复查数据流">
<defs><marker id="end" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#607383"/></marker></defs>
<g fill="none" stroke="#607383" stroke-width="2" marker-end="url(#end)">
<path d="M181 44H213V83H244"/><path d="M181 122H213V83H244"/><path d="M430 83H477"/><path d="M665 83H714"/>
</g><g stroke="#bdccd5">
<rect x="10" y="19" width="171" height="50" rx="8" fill="#edf4f8"/><rect x="10" y="97" width="171" height="50" rx="8" fill="#edf4f8"/>
<rect x="244" y="58" width="186" height="50" rx="8" fill="#e3f1ec"/><rect x="477" y="58" width="188" height="50" rx="8" fill="#f9f1e0"/><rect x="714" y="58" width="184" height="50" rx="8" fill="#edf4f8"/>
</g><g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="95" y="49">固定 PVD 与上一轮融合</text><text x="95" y="127">固定下部与保护像素</text>
<text x="337" y="88">αT 归因、候选原生行</text><text x="571" y="88">49 帧真实渲染保护检查</text><text x="806" y="88">共享退役账本、街景复核</text>
</g></svg>'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pooled(rows):
    out={}
    for name in rows[0]['metrics']:
        n=sum(r['metrics'][name]['count'] for r in rows)
        out[name]=sum(r['metrics'][name]['sum'] for r in rows)/n if n else None
    out['ratio']=out['lower_pvd']/out['isolated_lower_alpha']
    out['failed_frames']=sum(bool(r['violations']) for r in rows)
    return out


def image_grid(paths, titles, box=None, scale=1, columns=2):
    panels=[]
    for p in paths:
        with Image.open(p) as source:
            img=source.convert('RGB')
            if box is not None:
                img=img.crop(box)
            if scale!=1:
                img=img.resize((img.width*scale,img.height*scale),Image.Resampling.NEAREST)
            panels.append(img)
    w=max(panels[0].width,196)
    h=panels[0].height+24
    out=Image.new('RGB',(w*columns,h*(len(paths)//columns)),'#000')
    draw=ImageDraw.Draw(out)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        font=ImageFont.load_default()
    for i,(panel,title) in enumerate(zip(panels,titles)):
        x=i%columns*w
        y=i//columns*h
        out.paste(panel,(x+(w-panel.width)//2,y+24))
        draw.text((x+6,y+4),title,fill='#e2eaf1',font=font)
        if i%columns:
            draw.line((x,y,x,y+h-1),fill='#304653')
    return out


def main():
    result=read(ROOT/'result.json')
    receipt=read(ROOT/'receipt.json')
    street=read(ROOT/'street.json')
    assert receipt['status']=='COMPLETE' and result['frames']==49 and street['status']=='PASS'
    assert all(not r['violations'] for r in result['result'])
    all_rows=dict(result['baselines'],guarded=result['result'])
    scores={k:pooled(v) for k,v in all_rows.items()}
    out=ROOT/'delivery'
    out.mkdir(exist_ok=False)
    frames=out/'frames'
    frames.mkdir()
    for raw in range(89,138):
        src=ROOT/'render'
        with Image.open(PREV/'render'/f'{raw:03d}_pvd_pvd.png') as mask:
            y,x=np.nonzero(np.asarray(mask)[:,:,0]>5)
        box=(max(0,int(x.min())-10),max(0,int(y.min())-11),min(480,int(x.max())+11),min(320,int(y.max())+12))
        order=[src/f'{raw:03d}_{state}{suffix}.png'
               for suffix in ('','_pvd','_blocked') for state in ('previous','guarded')]
        labels=['Previous fusion','Guarded retirement','Previous PVD alpha','Guarded PVD alpha',
                'Previous hidden contribution','Guarded hidden contribution']
        image_grid(order,labels,box=box,scale=3).save(frames/f'{raw:03d}.png')
        if raw in PILOTS:
            paths=[src/f'{raw:03d}_{state}.png' for state in ('previous','guarded','measured_only','all_candidates')]
            image_grid(paths,['Previous fusion','Full protection','Measured-only control','All blockers removed'],
                       box=box,scale=3).save(frames/f'{raw:03d}_controls.png')
            paths=[ROOT/'street'/f'{raw:03d}_{state}_{version}.png'
                   for state in ('keep','delete') for version in ('previous','retired')]
            image_grid(paths,['No edit | previous','No edit | new','Delete a37 | previous','Delete a37 | new']).save(frames/f'{raw:03d}_street.jpg',quality=90)
    for name in ('result.json','street.json','receipt.json','replay.json','candidate_summary.json','shared_retirement.npz'):
        shutil.copyfile(ROOT/name,out/name)
    (out/'summary.json').write_text(json.dumps(dict(scores=scores,street=street['means'],
        retirement=result['controls'],guard=result['protection']),indent=2,allow_nan=False)+'\n')
    with (out/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['raw','variant','reveal','known_rgb','measured_depth_m','outside_rgb_change','outside_alpha_change','violations'])
        for name,rows in all_rows.items():
            for row in rows:
                x=pooled([row])
                writer.writerow([row['raw'],name,x['ratio'],x['known_rgb'],x['measured_depth'],x['outside_rgb_delta'],
                                 x['outside_alpha_delta'],'|'.join(row['violations'])])
    labels=dict(previous='上一轮保留侧拆分',guarded='逐帧双重保护',measured_only='只保护实测的诊断',
                all_candidate_retirement='全候选退役的诊断')
    rows=''
    for name in ('previous','guarded','measured_only','all_candidate_retirement'):
        v=scores[name]
        rows+=f"<tr><th>{labels[name]}</th><td>{v['ratio']:.2%}</td><td>{v['known_rgb']:.5f}</td><td>{v['measured_depth']:.4f}</td><td>{v['outside_rgb_delta']:.5f}</td><td>{v['failed_frames']}</td></tr>"
    s=street['means']
    d=s['delete']; k=s['keep']
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 a6 剩余遮挡复查</title>
<style>
body{{background:#f4f7f9;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1080px;margin:auto;padding:26px 20px}}
section{{background:white;border:1px solid #d6e1e8;border-radius:10px;padding:18px;margin:16px 0;overflow:auto}}
svg,img{{width:100%;display:block}}th,td{{padding:7px 9px;border-bottom:1px solid #dce5ec;text-align:left}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}a{{color:#066674}}input{{width:50%}}
</style><main><h1>Scene008 a6 · 剩余遮挡复查</h1>
<p>固定下部候选的显露率从 {scores['previous']['ratio']:.2%} 提高至 <strong>{scores['guarded']['ratio']:.2%}</strong>。新增共享退役行 {result['controls']['guarded_retired']:,}，训练 0 步。</p>
<section><h2>组件与数据流</h2>{ARCH}
<p>重放上一轮保留侧拆分。对同一批下部射线记录 PVD 前方的原生 αT，并按已观测 RGB、LiDAR 与非目标支撑排序候选。每个候选集合使用同一组行，在所有 49 帧真实渲染中验证。</p>
<p>沿用已观测 RGB、实测深度和覆盖门槛，另外约束原生车身在下部提案域及一像素边缘以外的 RGB 变化 ≤ 0.005、累计透明度变化 ≤ 0.01。完整删除和只保护实测的读出保留为诊断。候选排序及分组是有界的，结论适用于本轮实测和选择范围。</p></section>
<section><h2>49 帧黑底对照</h2><p>左列为上一轮，右列为新共享退役。依次显示 RGB、PVD 贡献和固定下部仍未显露的红色贡献。</p>
<label for="raw">raw<span id="no">113</span></label><input type="range" id="raw" min="89" max="137" value="113"><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button>
<img id="frame" src="frames/113.png" alt="raw113 黑底 a6"></section>
<section><h2>同域配对指标</h2>
<table><tr><th>版本</th><th>下部显露</th><th>已观测 RGB MAE</th><th>深度 MAE (m)</th><th>非目标 RGB 变化</th><th>全保护失败帧</th></tr>{rows}</table>
<p>显露分母为同一批下部 PVD 独立透明度。非目标 RGB 变化相对上一轮；其余 MAE 相对固定观测。隐藏部分缺独立真值。</p>
<p>破坏性对照 <a href="frames/105_controls.png">raw105</a> · <a href="frames/113_controls.png">raw113</a> · <a href="frames/130_controls.png">raw130</a></p></section>
<section><h2>独立街景复核</h2>
<p>删 a37 后，固定下部显露率 {d['lower_fraction_before']:.2%} → {d['lower_fraction_after']:.2%}；显露道路 RGB MAE {d['road_rgb_before']:.5f} → {d['road_rgb_after']:.5f}。无编辑时相对上一轮的可见 a37 RGB 变化为 {k['a37_rgb_delta']:.8f}。同一套 a6 参数和共享门控跨两种编辑状态。</p>
<img src="frames/113_street.jpg" alt="raw113 独立街景检查"></section>
<section><h2>回执与账本</h2><a href="fixed_metrics.csv">逐帧指标</a> · <a href="result.json">候选试跑与保护门槛</a> · <a href="street.json">49 帧街景回执</a> · <a href="shared_retirement.npz">共享退役账本</a></section>
<script>
const raw=document.getElementById('raw'),no=document.getElementById('no'),frame=document.getElementById('frame');
function update(){{const key=raw.value.padStart(3,'0');no.textContent=key;frame.src='frames/'+key+'.png';}}
raw.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{raw.value=b.dataset.raw;update();}}));
</script></main></html>'''
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(scores=scores,street=s),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
