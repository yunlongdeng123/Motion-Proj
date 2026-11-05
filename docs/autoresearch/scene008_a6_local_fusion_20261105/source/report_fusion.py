"""Build the black a6 review, with the street attribution kept in its own panel."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent
PILOTS=(105,113,130)
LABELS={
    'native':'原生 a6',
    'pvd':'独立 PVD',
    'direct':'全量直接叠加',
    'lower_only':'仅接入下部 PVD',
    'fusion':'中心筛选退役',
    'footprint':'足迹筛选退役',
    'footprint_split':'足迹退役＋保留侧拆分',
    'support_gate':'扩展下部证据＋LiDAR 保护门控',
    'support_split':'扩展下部证据＋LiDAR 保护拆分',
}
ARCH='''<svg viewBox="0 0 940 235" role="img" aria-label="a6 局部融合数据流">
<defs><marker id="end" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#627383"/></marker></defs>
<g fill="none" stroke="#627383" stroke-width="2" marker-end="url(#end)">
<path d="M186 53H230"/><path d="M186 113H230"/><path d="M186 173H230"/><path d="M460 53H496V112H525"/><path d="M460 113H525"/><path d="M460 173H496V112H525"/><path d="M713 112H749"/>
</g>
<g stroke="#bac9d5">
<rect x="10" y="28" width="176" height="50" rx="8" fill="#edf3f8"/><rect x="10" y="88" width="176" height="50" rx="8" fill="#edf3f8"/><rect x="10" y="148" width="176" height="50" rx="8" fill="#edf3f8"/>
<rect x="230" y="28" width="230" height="50" rx="8" fill="#e0f1eb"/><rect x="230" y="88" width="230" height="50" rx="8" fill="#e0f1eb"/><rect x="230" y="148" width="230" height="50" rx="8" fill="#e0f1eb"/>
<rect x="525" y="87" width="188" height="50" rx="8" fill="#fbf0df"/><rect x="749" y="87" width="181" height="50" rx="8" fill="#edf3f8"/>
</g>
<g fill="#172938" font-size="16" text-anchor="middle" font-family="sans-serif">
<text x="98" y="59">跨帧 RGB、LiDAR</text><text x="98" y="119">固定 PVD</text><text x="98" y="179">原生 a6</text>
<text x="345" y="59">固定实测保护与缺失域</text><text x="345" y="119">车体坐标选择下部提案</text><text x="345" y="179">候选退役与保留侧拆分</text>
<text x="619" y="118">黑底联合深度排序</text><text x="839" y="118">四列对照与贡献图</text>
</g></svg>'''


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pooled(rows,name):
    metrics=[row['variants'][name] for row in rows]
    sums={key:sum(item[key]['sum'] for item in metrics) for key in metrics[0]}
    counts={key:sum(item[key]['count'] for item in metrics) for key in metrics[0]}
    result={key:sums[key]/counts[key] if counts[key] else None for key in sums}
    result['reveal_fraction']=sums['lower_pvd']/sums['isolated_lower_alpha'] if sums['isolated_lower_alpha'] else None
    return result


def guard(base,now,params):
    tests={
        'known_rgb':now['known_rgb']-base['known_rgb']<=params['rgb_guard'],
        'known_rgb_change':now['known_rgb_change']<=params['rgb_guard'],
        'measured_depth':now['measured_depth']-base['measured_depth']<=params['depth_guard_m'],
        'known_alpha':now['known_alpha']>=base['known_alpha']*params['evidence_guard_fraction'],
    }
    return [key for key,val in tests.items() if not val]


def main():
    base=load(ROOT/'results.json')
    evidence=load(ROOT/'evidence/contract.json')
    ledger=load(ROOT/'ledger.json')
    direct=load(ROOT/'footprint_results.json')
    broad=load(ROOT/'lower_support_results.json')
    rows=base['rows']
    paths={name:ROOT/'render' for name in base['variants']}
    for new,names in ((direct,('footprint','footprint_split')),(broad,('support_gate','support_split'))):
        for old,added in zip(rows,new['rows']):
            assert old['raw']==added['raw']
            old['variants'].update(dict(zip(names,added['variants'].values())))
        paths.update(dict.fromkeys(names,ROOT/new['render_dir']))
    names=list(rows[0]['variants'])
    global_scores={name:pooled(rows,name) for name in names}
    pilot_scores={str(raw):{name:pooled([rows[raw-89]],name) for name in names} for raw in PILOTS}
    params=evidence['policy']
    guard_records={}
    for name in names:
        failures=[]
        for row in rows:
            one=row['variants']['native']['measured_depth']['count']
            if one:
                issues=guard(pooled([row],'native'),pooled([row],name),params)
                if issues:
                    failures.append(dict(raw=row['raw'],metrics=issues))
        guard_records[name]=dict(global_failures=guard(global_scores['native'],global_scores[name],params),frames=failures)
    accepted=[name for name in names[3:] if not any(guard_records[name].values())]
    chosen=max(accepted,key=lambda v:global_scores[v]['reveal_fraction']) if accepted else 'lower_only'
    dest=ROOT/'delivery'
    dest.mkdir(exist_ok=False)
    images=dest/'frames'
    images.mkdir()
    street=dest/'street_diagnostic'
    street.mkdir()
    for path in (ROOT/'pilots').glob('*_street_*.png'):
        shutil.copyfile(path,street/path.name)
    shutil.copyfile(ROOT/'street_diagnostic.json',street/'metrics.json')
    for filename in ('ledger.json','ledger.npz','footprint_ledger.json','footprint_ledger.npz',
                     'lower_support_ledger.json','lower_support_ledger.npz'):
        shutil.copyfile(ROOT/filename,dest/filename)
    shutil.copytree(ROOT/'evidence',dest/'evidence')
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        font=ImageFont.load_default()
    primary=('native','pvd','direct',chosen)
    labels=('Native a6','Fixed PVD alone','Direct overlay','Local fusion')
    def suffix(name):
        return {'support_gate':'footprint','support_split':'footprint_split'}.get(name,name)
    def path(raw,name,marker=''):
        return paths[name]/f'{raw:03d}_{suffix(name)}{marker}.png'
    for row in rows:
        raw=row['raw']
        with Image.open(path(raw,'pvd','_pvd')) as alpha:
            y,x=np.nonzero(np.asarray(alpha)[:,:,0]>5)
        box=(max(0,int(x.min())-10),max(0,int(y.min())-12),min(480,int(x.max())+11),min(320,int(y.max())+13))
        scale=3
        width=max(174,(box[2]-box[0])*scale)
        height=(box[3]-box[1])*scale
        panel=Image.new('RGB',(width*4,2*(height+24)),'#000000')
        painter=ImageDraw.Draw(panel)
        for i,name in enumerate(primary):
            for j,marker in enumerate(('','_pvd')):
                with Image.open(path(raw,name,marker)) as im:
                    crop=im.crop(box).resize(((box[2]-box[0])*scale,height),Image.Resampling.NEAREST)
                yoff=j*(height+24)
                panel.paste(crop,(i*width+(width-crop.width)//2,yoff+24))
                painter.text((i*width+7,yoff+3),labels[i] if j==0 else ('Native alpha' if i==0 else 'PVD visible alpha'),font=font,fill='#e2eaf1')
            if i:
                painter.line((i*width,0,i*width,panel.height),fill='#324656')
        panel.save(images/f'{raw:03d}.png')
        if chosen=='lower_only':
            raise RuntimeError('Fallback deficit render must be provided')
        maps=Image.new('RGB',(2*width,height+24),'#000000')
        draw=ImageDraw.Draw(maps)
        for i,name in enumerate(('direct',chosen)):
            with Image.open(path(raw,name,'_blocked')) as im:
                crop=im.crop(box).resize(((box[2]-box[0])*scale,height),Image.Resampling.NEAREST)
            maps.paste(crop,(i*width+(width-crop.width)//2,24))
            draw.text((i*width+7,3),('Direct overlay','Local fusion')[i],font=font,fill='#e2eaf1')
        maps.save(images/f'{raw:03d}_blocked.png')
    summary=dict(status='PASS_ENGINEERING',method_updates=0,evidence=evidence,ledger=ledger,
        footprints=load(ROOT/'footprint_ledger.json'),support=load(ROOT/'lower_support_ledger.json'),
        chosen=chosen,chosen_meets_protection_guards=chosen in accepted,guard_records=guard_records,
        metrics=global_scores,pilot_metrics=pilot_scores,rows=rows,
        protection_rule='Pass pooled and all 49 frames with depth evidence; increase in RGB MAE and RGB change vs native each <=.003, depth MAE increase <=.03m, alpha >=.99x native.',
        unknown_geometry_gt=False, pvd_source=base['pvd_source'], road_source=base['road_source'])
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (dest/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['raw','variant','reveal_fraction','known_rgb_mae','known_rgb_change','measured_depth_mae_m','pvd_on_known'])
        for row in rows:
            for name in names:
                m=pooled([row],name)
                writer.writerow([row['raw'],name,*[m[k] for k in ('reveal_fraction','known_rgb','known_rgb_change','measured_depth','pvd_on_known')]])
    records=''
    for name in names:
        m=global_scores[name]
        failure=guard_records[name]
        status='通过' if not any(failure.values()) else f"未通过 · {len(failure['frames'])} 帧"
        display=' — ' if name in ('native','pvd','direct') else status
        records+=f"<tr><th>{LABELS[name]}</th><td>{m['reveal_fraction']:.2%}</td><td>{m['known_rgb']:.5f}</td><td>{m['measured_depth']:.4f}</td><td>{m['pvd_on_known']:.5f}</td><td>{display}</td></tr>"
    best=global_scores[chosen]
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 a6 下部局部融合</title>
<style>
body{{background:#f3f7fa;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1190px;margin:auto;padding:28px 22px}}
section{{background:white;border:1px solid #d7e1e8;border-radius:12px;padding:20px;margin:18px 0;overflow:auto}}
h1,h2{{line-height:1.2}} svg,img{{width:100%;display:block}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #dce5ec}}
a{{color:#066471}}.street{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
input{{width:50%}}button{{margin:0 3px}}
</style><main><h1>Scene008 · a6 下部局部融合</h1>
<p>主实验只画黑底 a6。采用 <strong>{LABELS[chosen]}</strong>，下部显露率 {best['reveal_fraction']:.2%}，固定已观测 RGB MAE {best['known_rgb']:.5f}，实测深度 MAE {best['measured_depth']:.4f} m。</p>
<section><h2>组件与数据流</h2>{ARCH}
<p>在 a6 车体坐标里，冻结实测面和缺失侧。后半车、低于最低实测点 3 cm、远离实测锚点的提案共有 {evidence['lower_rows']} 行。融合时只启用这些行，选中行的几何、颜色和透明度保持原值。已有后窗和车肩由原生 a6 继续表达。</p>
<p>固定下部评价域共 {evidence['target_pixels']:,} 个像素，排除了可见 a6 轮廓和 LiDAR 测点。显露率是该域中下部 PVD 行的联合贡献，除以这些下部行独立渲染的贡献。全量独立 PVD 另有提案自身重叠，表中略低于 100%。隐藏面缺独立真值。</p></section>
<section><h2>49 帧固定裁剪</h2><p>上排是 RGB，下排是透明度贡献。第一列下格显示原生透明度，其余列显示 PVD 贡献。三组对照和局部融合使用同一相机、车姿及裁剪。</p>
<label for="raw">raw<span id="frameNo">113</span></label>
<input id="raw" type="range" min="89" max="137" value="113"><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button>
<img id="hero" src="frames/113.png" alt="raw113 四组黑底对照">
<p>下部候选域中，红色表示相对下部 PVD 独立渲染仍缺的可见贡献。左为直接叠加，右为局部融合。</p>
<img id="deficit" src="frames/113_blocked.png" alt="raw113 下部未显露权重"></section>
<section><h2>同域指标</h2><table><tr><th>方案</th><th>下部显露率</th><th>可见 a6 RGB MAE</th><th>实测深度 MAE (m)</th><th>可见 a6 上的 PVD 贡献</th><th>保护检查</th></tr>{records}</table>
<p>保护门槛按原生 a6 计算，RGB MAE 增量和相对原生图的 RGB 变化分别至多 0.003，实测深度 MAE 增量至多 0.03 m，原生透明度至少保留 99%。同时检查总体和每一帧。</p></section>
<section><h2>街景遮挡诊断</h2><p>raw113。左为无编辑，中为删 a37，右为进一步隐藏原生 a6。该诊断保持道路与环境，独立于主实验。</p>
<div class="street"><img src="street_diagnostic/113_street_keep.png" alt="无编辑"><img src="street_diagnostic/113_street_delete.png" alt="删 a37"><img src="street_diagnostic/113_street_hide_native_a6.png" alt="再隐藏原生 a6"></div></section>
<section><h2>可核对文件</h2><a href="summary.json">指标、规则和逐帧保护检查</a> · <a href="fixed_metrics.csv">固定域逐帧 CSV</a> · <a href="evidence/contract.json">冻结证据</a> · <a href="lower_support_ledger.npz">车体坐标门控和拆分账本</a></section>
<script>
const raw=document.getElementById('raw'),no=document.getElementById('frameNo'),hero=document.getElementById('hero'),map=document.getElementById('deficit');
function update(){{const key=raw.value.padStart(3,'0');no.textContent=key;hero.src='frames/'+key+'.png';map.src='frames/'+key+'_blocked.png';}}
raw.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{raw.value=b.dataset.raw;update();}}));
</script></main></html>'''
    (dest/'index.html').write_text(body,encoding='utf-8')
    print(json.dumps(dict(chosen=chosen,metrics=global_scores,guard=guard_records),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
