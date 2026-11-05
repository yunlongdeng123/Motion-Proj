"""Validate and publish the native49 local-PNG comparison, without GPU/network."""
from __future__ import annotations
import base64,hashlib,html,json,re,shutil
from pathlib import Path
from PIL import Image

RAW_IDS=list(range(89,138))
METHODS=('old_formal','c2_matte','surface_lod')
STATES=('keep','delete')
ASSET_SHA={'old_formal':'70826fc7c9c011ee073f7ffea4f22501a57f39ffa2cd0d52cd5ebd9c37b14e53',
           'c2_matte':'0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d',
           'surface_lod':'0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d'}
LABELS={'old_formal':'旧 formal','c2_matte':'C2 + matte','surface_lod':'同 S/F LOD（研究）',
        'reference':'原始实拍：a37仍在，非删车GT'}
BEGIN='<!-- A6_FRAME_COMPARE_BEGIN -->'
END='<!-- A6_FRAME_COMPARE_END -->'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def esc(x):return html.escape(str(x),quote=True)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def load_validate(manifest_path):
    """All 294 render PNGs must be present and exact before any publication."""
    manifest_path=Path(manifest_path).resolve()
    if not manifest_path.is_file():raise FileNotFoundError('Complete frame manifest is required; old report was not overwritten: '+str(manifest_path))
    source=json.loads(manifest_path.read_text(encoding='utf-8-sig'));root=manifest_path.parent
    assert source['status']=='COMPLETE','Only a complete actual49 capture can be published'
    assert source['raw_ids']==RAW_IDS and set(source['states'])==set(STATES)
    assert source['viewport']==dict(width=480,height=320)
    columns=source['columns'];assert [x['id'] for x in columns]==list(METHODS)
    for c in columns:assert c['asset_sha256']==ASSET_SHA[c['id']]
    frames=source['frames'];assert [x['raw'] for x in frames]==RAW_IDS
    assert source['default_raw']==113 and source['default_state']=='delete'
    assert source['no_interpolation_or_2D_frame_synthesis'] is True
    assert source['no_neural_colour_or_geometry_fit_updates'] is True
    renderer=source['renderer']['id'];assert re.fullmatch('[a-f0-9]{64}',renderer)
    assert source['renderer']['native_kernel_or_camera_or_environment_updated'] is False
    refs=[bool(x.get('reference')) for x in frames]
    assert all(refs) or not any(refs),'Original-RGB reference must cover all49 or be omitted'
    has_reference=all(refs);files={};records={};crops=[]
    def descriptor(d,raw,state,method):
        assert d['raw']==raw and d['state']==state and d['method']==method
        assert d['width']==480 and d['height']==320
        rel=Path(d['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix.lower()=='.png'
        expected=Path('frames')/f'{raw:03d}'/('reference.png' if method=='original_RGB' else state+'/'+method+'.png')
        assert rel==expected,'Path/raw/state/method crosswire: '+str(rel)
        p=(root/rel).resolve();assert p.is_relative_to(root)
        assert p.is_file(),'Missing actual frame PNG; old report was not overwritten: '+str(p)
        assert re.fullmatch('[a-f0-9]{64}',d['sha256']) and sha(p)==d['sha256'],str(p)
        with Image.open(p) as im:
            assert im.format=='PNG' and im.size==(480,320)
            im.verify()
        if method!='original_RGB':
            assert d['asset_sha256']==ASSET_SHA[method] and d['renderer_id']==renderer
        outrel=rel if rel.parts[0]=='frames' else Path('frames')/rel
        url=outrel.as_posix()
        if url in files:assert files[url]['sha256']==d['sha256']
        files[url]=dict(source=p,path=url,sha256=d['sha256'],raw=raw,state=state,method=method)
        return dict(path=url,sha256=d['sha256'])
    for f in frames:
        raw=f['raw'];assert f['camera']['width']==480 and f['camera']['height']==320
        row=dict(raw=raw)
        for state in STATES:
            assert set(f[state])==set(METHODS)
            row[state]={m:descriptor(f[state][m],raw,state,m) for m in METHODS}
        if has_reference:row['reference']=descriptor(f['reference'],raw,'observed_a37_present','original_RGB')
        crop=f.get('crop_xyxy')
        if crop is not None:
            assert len(crop)==4 and all(isinstance(x,int) for x in crop)
            x0,y0,x1,y1=crop;assert 0<=x0<x1<=480 and 0<=y0<y1<=320
            provenance=f['crop_provenance'];assert provenance['same_crop_for_all_methods_states_and_reference'] is True
            assert provenance['predicted_alpha_used'] is False and provenance['display_only'] is True
        crops.append(crop is not None);row['crop_xyxy']=crop;records[str(raw)]=row
    assert sum(x['method']!='original_RGB' for x in files.values())==294
    assert sum(x['method']=='original_RGB' for x in files.values())==(49 if has_reference else 0)
    delivered_columns=[dict(id=m,label=LABELS[m]) for m in METHODS]
    if has_reference:delivered_columns.append(dict(id='reference',label=LABELS['reference']))
    data=dict(raw_ids=RAW_IDS,default_raw=113,default_state='delete',columns=delivered_columns,
        frames=records,has_all_crops=all(crops),viewport=dict(width=480,height=320))
    return dict(source_path=manifest_path,source=source,source_sha256=sha(manifest_path),files=files,data=data,
        image_count=294,reference_count=49 if has_reference else 0)


STYLE='''<style id="fc-style">
#frame-compare{margin:24px 0 28px;padding:22px;background:#fff;border:1px solid #bfd2df;border-radius:12px}
#frame-compare h2{margin-bottom:12px}.fc-controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px 14px}
.fc-controls button,.fc-controls select{font:inherit;padding:6px 10px;border:1px solid #9db7c7;border-radius:6px;background:#f3f8fb;color:#17384d}
.fc-controls button:disabled,.fc-controls select:disabled{opacity:.5}.fc-range-row{display:flex;align-items:center;gap:12px;flex:1;min-width:260px}
#fc-range{flex:1;min-width:150px;accent-color:#176f9e}#fc-current{font-weight:700;margin:14px 0 5px}
#fc-status{font-size:14px;margin:5px 0 12px}#fc-status[data-level="error"]{background:#fff0ea;color:#9a351c;padding:9px;border-left:3px solid #bd492a}
.fc-stage{position:relative}.fc-grid{display:grid;grid-template-columns:repeat(var(--fc-columns),minmax(0,1fr));gap:10px;--fc-ratio:1.5}
.fc-card{min-width:0;border:1px solid #d0dde5;background:#f6f9fb;border-radius:6px;overflow:hidden}
.fc-card h3{font-size:14px;line-height:1.4;min-height:42px;padding:8px;margin:0;display:flex;align-items:center}
.fc-window{position:relative;width:100%;aspect-ratio:var(--fc-ratio);overflow:hidden;background:#0b1720}
.fc-window img{display:block;position:absolute;width:100%;height:100%;max-width:none;left:0;top:0;object-fit:fill;border:0}
.fc-cropped .fc-window img{width:var(--fc-image-width);height:var(--fc-image-height);left:var(--fc-left);top:var(--fc-top)}
.fc-card-footer{padding:7px;font-size:12px;line-height:1.4}.fc-file{display:block;overflow-wrap:anywhere;margin-top:4px;font-size:10px;background:transparent;padding:0}
#fc-overlay{position:absolute;inset:0;background:rgba(242,247,250,.94);z-index:2;padding:28px;display:flex;align-items:center;justify-content:center;overflow-wrap:anywhere;font-weight:600;text-align:center;white-space:pre-wrap}
#fc-overlay[hidden]{display:none}.fc-links{font-size:14px;margin:10px 0 0}.fc-note{font-size:13px;color:#526a7a}
@media(max-width:800px){#frame-compare{padding:14px 10px}.fc-card h3{font-size:12px;min-height:54px}.fc-grid{gap:5px}.fc-card-footer{font-size:10px;padding:5px}.fc-file{font-size:9px}}
</style>'''


def module_html(data,controller):
    frame=data['frames']['113'];columns=data['columns'];cards=[]
    for c in columns:
        path=frame['reference']['path'] if c['id']=='reference' else frame['delete'][c['id']]['path']
        label=c['label'];cid=c['id']
        cards.append('<article class="fc-card"><h3>'+esc(label)+'</h3><div class="fc-window"><img id="fc-image-'+cid+'" src="'+esc(path)+'" alt="'+esc('raw113 / delete a37 / '+label+' / '+path)+'" data-raw="113" data-state="delete" data-column="'+cid+'"></div><div class="fc-card-footer"><a id="fc-open-'+cid+'" href="'+esc(path)+'" target="_blank" rel="noopener">打开原尺寸PNG</a><code class="fc-file" id="fc-file-'+cid+'">'+esc(path)+'</code></div></article>')
    cropdisabled='' if data['has_all_crops'] else ' disabled'
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('&','\\u0026')
    section='''<section id="frame-compare" aria-label="49帧同步街景对照">
<h2>49帧同步街景对照</h2>
<p class="fc-note">三种研究表示使用同帧、同相机、同视口的实际480×320无损PNG；keep/delete只切换a37状态。初始raw113无需JavaScript也可见，拖动切换需JavaScript。原始实拍列仍有a37，不能当删车后的道路真值。离线可用，移动报告时请连同旁边frames目录保留。</p>
<div class="fc-controls"><button id="fc-prev" type="button" data-fc-control disabled>上一帧</button><div class="fc-range-row"><label for="fc-range">raw89–137</label><input id="fc-range" type="range" min="89" max="137" step="1" value="113" data-fc-control disabled><output id="fc-requested" for="fc-range">选择 raw113</output></div><button id="fc-next" type="button" data-fc-control disabled>下一帧</button>
<button id="fc-jump-105" type="button" data-fc-raw="105" data-fc-control disabled>105</button><button id="fc-jump-113" type="button" data-fc-raw="113" data-fc-control disabled>113</button><button id="fc-jump-130" type="button" data-fc-raw="130" data-fc-control disabled>130</button>
<label for="fc-state">a37状态 <select id="fc-state" data-fc-control disabled><option value="delete" selected>delete a37</option><option value="keep">keep a37</option></select></label>
<label for="fc-view">显示 <select id="fc-view" data-fc-control disabled><option value="full" selected>全景480×320</option><option value="crop"'''+cropdisabled+'''>同帧统一裁剪</option></select></label></div>
<p id="fc-current">当前显示：raw113 / delete a37 / 全景480×320</p>
<p id="fc-status" role="status" aria-live="polite">静态初始raw113已直接写入图片src；切换控件需JavaScript。若控件未启用，可打开完整静态49帧对照。</p>
<div class="fc-stage"><div id="fc-grid" class="fc-grid" style="--fc-columns:'''+str(len(columns))+'''">'''+''.join(cards)+'''</div><div id="fc-overlay" hidden aria-live="polite"></div></div>
<p id="fc-crop-note" class="fc-note">全景480×320；所有列使用原始PNG，点击“打开原尺寸”可查看480原尺寸。</p>
<p class="fc-links"><a href="frames_static.html#raw113">完整静态49帧对照（无需JavaScript）</a> · <a href="frame_compare_manifest.json">交付帧映射与SHA</a> · <a href="frame_compare_source_manifest.json">原始采集manifest</a> · <a href="frames_sha256.json">全部PNG哈希</a> · <a href="code/report/frame_compare_controller.js">切换脚本源码</a></p>
<noscript><p class="fc-note">当前静态图为raw113/delete。可使用上面的静态49帧页面逐帧查看，两状态均完整。</p></noscript>
'''+STYLE+'<script id="fc-data" type="application/json">'+payload+'</script><script id="fc-controller">'+controller+'</script></section>'
    return BEGIN+section+END


def static_fallback(data):
    nav=' '.join('<a href="#raw'+str(r)+'">'+str(r)+'</a>' for r in RAW_IDS)
    parts=[]
    for raw in RAW_IDS:
        f=data['frames'][str(raw)];body='<section id="raw'+str(raw)+'"><h2>raw'+str(raw)+'</h2>'
        for state in STATES:
            body+='<h3>'+state+' a37 · 全景480×320</h3><div class="grid">'
            for c in data['columns']:
                p=f['reference']['path'] if c['id']=='reference' else f[state][c['id']]['path']
                body+='<figure><figcaption>'+esc(c['label'])+'</figcaption><a href="'+esc(p)+'"><img loading="lazy" src="'+esc(p)+'" alt="'+esc('raw'+str(raw)+' '+state+' '+c['label']+' '+p)+'"></a><code>'+esc(p)+'</code></figure>'
            body+='</div>'
        parts.append(body+'</section>')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>49帧静态街景对照</title><style>body{font:15px/1.6 system-ui,sans-serif;max-width:1700px;margin:20px auto;padding:16px;background:#f4f7fa;color:#173449}nav{display:flex;flex-wrap:wrap;gap:9px}section{padding:18px;background:white;margin:24px 0;scroll-margin-top:12px}.grid{display:grid;grid-template-columns:repeat('+str(len(data['columns']))+',minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;height:auto}code{font-size:10px;overflow-wrap:anywhere}a{color:#145b84}</style><h1>49帧完整静态街景对照</h1><p><a href="index.html#frame-compare">返回报告的同步拖动模块</a>。本页无需JavaScript，全部49帧、两状态、三研究列齐备；实拍仍有a37，不是delete真值。点击图片打开无损480原尺寸PNG。</p><nav>'+nav+'</nav>'+''.join(parts)+'</html>'


def publish(validated,out,controller_path):
    """Copies only lossless original PNG bytes, after complete prevalidation."""
    out=Path(out);data=validated['data'];ledger={}
    for url,d in validated['files'].items():
        target=out/url;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_file() or sha(target)!=d['sha256']:shutil.copy2(d['source'],target)
        assert sha(target)==d['sha256']
        ledger[url]=d['sha256']
    shutil.copy2(validated['source_path'],out/'frame_compare_source_manifest.json')
    normalized=dict(schema='a6.local-frame-compare.delivery.v1',source_manifest_sha256=validated['source_sha256'],
        renderer=validated['source']['renderer'],source_columns=validated['source']['columns'],**data)
    write(out/'frame_compare_manifest.json',normalized);write(out/'frames_sha256.json',ledger)
    (out/'frames_static.html').write_text(static_fallback(data),encoding='utf-8')
    code=out/'code/report';code.mkdir(parents=True,exist_ok=True)
    shutil.copy2(controller_path,code/controller_path.name)
    shutil.copy2(Path(__file__),code/Path(__file__).name)
    receipt=dict(status='PASS_ALL49_FULL_NATIVE_PNG_INPUTS',raw_ids=RAW_IDS,states=list(STATES),methods=list(METHODS),
        render_PNGs=294,reference_PNGs=validated['reference_count'],default_raw=113,default_state='delete',
        initial_view='full480',source_manifest_sha256=validated['source_sha256'],all_images_SHA_dimensions_checked=True,
        all_states_methods_same_recorded_camera=True,per_raw_common_crop_available=data['has_all_crops'],
        frames_as_relative_lossless_PNG=True,frames_embedded_as_base64=False,
        JavaScript_required=dict(interaction=True,initial_images=False,static_fallback=False),
        local_manifest_fetch=False,all_columns_preload_decode_then_atomic_commit=True,
        loading_failure='Old complete group remains with explicit still-showing frame/state and failed file; never partial new/old group',
        no_model_or_metric_or_adoption_change=True)
    write(out/'frame_compare_receipt.json',receipt)
    return module_html(data,Path(controller_path).read_text(encoding='utf-8')),receipt


def scientific_baseline(out,baseline_path):
    """Bind the pre-slider 170c report once, before changing its HTML."""
    baseline_path=Path(baseline_path);known='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191'
    if not baseline_path.is_file():
        current=Path(out)/'index.html';assert sha(current)==known,'Pre-slider scientific baseline is not the reviewed170c report'
        baseline_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(current,baseline_path)
    assert sha(baseline_path)==known
    return baseline_path.read_bytes()


def inject_preserving_baseline(page,module,baseline):
    base=page.encode('utf-8');assert base==baseline,'Static scientific report changed; refuse to overwrite reviewed content'
    marker='<figure>';assert page.count(marker)>=1
    result=page.replace(marker,module+marker,1)
    stripped=re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END),'',result,count=1,flags=re.S).encode('utf-8')
    assert stripped==baseline
    return result
