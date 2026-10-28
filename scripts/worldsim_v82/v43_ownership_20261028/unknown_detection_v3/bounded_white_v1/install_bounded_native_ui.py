"""安装真实蓝框限定原生白底结果；只写bounded_white_v1输出目录。

仅在三帧真实PNG、evidence和receipt全部验证后写入。历史CPU硬填白图保留，
默认展示真实 G_after + (1-alpha_after)*white，域外取同帧existing分支。
用法：python work/v43_unknown_detection_20261028/install_bounded_native_ui.py --source <已确认取回的R2目录>
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
RAWS = (98, 113, 114)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def display(array):
    return np.round(np.clip(array.transpose(1, 2, 0), 0, 1) * 255).astype('uint8')


def arch():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 330" width="1500" height="330"><rect width="1500" height="330" fill="#f0f5f9"/>']
    def box(x, y, w, h, title, cells, color):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="white" stroke="#88a6ba"/><text x="{x+w/2}" y="{y+29}" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="19" fill="#233e52">{title}</text>')
        cw = (w-34-18*(len(cells)-1))/len(cells)
        for i, text in enumerate(cells):
            xx = x+17+i*(cw+18)
            parts.append(f'<rect x="{xx}" y="{y+44}" width="{cw}" height="42" rx="4" fill="{color}"/><text x="{xx+cw/2}" y="{y+70}" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="14" fill="#233e52">{text}</text>')
            if i < len(cells)-1:
                line(xx+cw+2, y+65, xx+cw+15, y+65)
    def line(x1, y1, x2, y2):
        parts.append(f'<path d="M{x1} {y1}H{x2-8}" stroke="#456e89" stroke-width="2"/><polygon points="{x2-8},{y2-5} {x2},{y2} {x2-8},{y2+5}" fill="#456e89"/>')
    box(18, 111, 285, 108, '同帧existing与用户范围', ['既有基线', '蓝框＋局部带'], '#e8f0f7')
    box(365, 20, 720, 108, '分支一：用户要求的全白范围（CPU显示门控）', ['局部显示带', '像素直接覆盖白色', '范围目标预览'], '#f7ebd4')
    box(365, 198, 720, 108, '分支二：真实高斯候选剥离＋R2局部合成', ['蓝框内二维中心', 'alpha×T选行并重算', 'G＋透射×白色'], '#e0eff7')
    box(1150, 111, 330, 108, '共同显示边界与核对', ['域外existing', '行ID／剩余支撑'], '#e6eef1')
    parts.append('<path d="M303 165H330V75H348 M330 165V253H348 M1085 75H1118V165H1134 M1085 253H1118V165" fill="none" stroke="#456e89" stroke-width="2"/><polygon points="348,70 358,75 348,80" fill="#456e89"/><polygon points="348,248 358,253 348,258" fill="#456e89"/><polygon points="1134,160 1144,165 1134,170" fill="#456e89"/>')
    return ''.join(parts)+'</svg>'





HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>蓝框限定：局部全白目标与真实R2</title><style>
*{box-sizing:border-box}body{margin:0;background:#0e1925;color:#dce9f3;font:16px/1.65 "Microsoft YaHei",system-ui,sans-serif}header,main,footer{max-width:1500px;padding:22px;margin:auto}h1{font-size:28px;margin:5px 0 14px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:8px 0}a{color:#8bd4ff}.panel{background:#152638;border:1px solid #35546b;border-radius:9px;padding:18px;margin-bottom:20px}.notice{padding:14px 18px;background:#3b3122;border-left:4px solid #d9a34e}.small,figcaption{font-size:14px;color:#b5cad9}nav,.controls,.links{display:flex;align-items:center;gap:12px 18px;flex-wrap:wrap}select{font:inherit;padding:7px 10px;background:#213c52;color:#e9f5ff;border:1px solid #7194ad;border-radius:5px}.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;min-width:0}img{width:100%;height:auto;display:block;background:#08111a;border:1px solid #3b586e;border-radius:4px}figcaption{margin-top:6px}.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:18px 0}.metric{border:1px solid #42637b;background:#1a3043;padding:9px 11px;border-radius:5px}.metric small{display:block;font-size:13px;color:#bad0df}.metric strong{font:600 21px/1.5 system-ui,sans-serif}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 10px;text-align:left;border-bottom:1px solid #3c566b}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:340px;overflow:auto;background:#102131;padding:12px;font:12px/1.6 Consolas,monospace}details{margin-top:14px}summary{cursor:pointer}code{background:#0c1b29;padding:2px 5px}.pending{outline:2px solid #e7b45c}@media(max-width:900px){.comparison{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.pair{grid-template-columns:1fr}}@media(max-width:600px){header,main,footer{padding:15px}h1{font-size:24px}}
</style></head><body><header><h1>蓝框限定：局部全白范围与真实候选剥离 R2</h1><p class="notice"><strong>第二列是用户要求的局部全白范围目标，仅为CPU显示门控；第三列才是真实候选高斯剥离后的R2，仍未完全变白，车底暗边与细彩纹保留。</strong> 新增行只允许二维投影中心落在蓝框内（不是3D实体box）；真实白底仅补透射，局部带外沿用同帧existing。暗像素阈值计数0不等于视觉纯白或去影通过，也不证明可迁移语义归属。</p><nav><a href="comparison.png">三帧 existing／局部全白目标／真实R2 PNG</a><a href="native_manifest.json">真实结果与验证</a><a href="ai_visual_review.json">AI辅助目视记录（非人工GT）</a><a href="../../../GS_Scene_Diagnostics_20261027/index.html">共享入口</a></nav></header><main>
<section class="panel"><div class="controls"><label>源帧 <select id="raw"><option value="113">raw113</option><option value="98">raw98</option><option value="114">raw114</option></select></label><strong id="status"></strong></div><div class="comparison"><figure><h3>existing</h3><img id="existing" alt="同帧existing基线"><figcaption>原基线；框外继续采用这一分支。</figcaption></figure><figure><h3>用户要求的局部全白范围</h3><img id="goal" alt="用户局部全白范围，CPU显示门控非自动归属"><figcaption>既有 <code>white.png</code>：局部像素硬覆盖，仅作范围目标，非自动归属或算法去影结果。</figcaption></figure><figure><h3>真实候选高斯剥离 R2</h3><img id="result" alt="真实高斯状态按G加透射白底合成的R2"><figcaption id="resultCaption">G＋(1−alpha)×白底；不是纯白硬盖，仍保留细纹。</figcaption></figure></div><div id="metrics" class="metrics"></div><div class="links"><a id="receiptLink" href="#">当前R2 receipt</a><a id="evidenceLink" href="#" download>新增候选行ID／中心／分量 NPZ</a><a href="native_summary.json">原始R2 summary</a></div></section>
<section class="panel"><h2>两条分支</h2><img src="native_architecture.svg" alt="existing进入用户局部硬填白范围预览和真实高斯候选剥离两条分支，域外都保留existing"><details><summary>新增行ID与真实查询记录</summary><p id="rowSummary" class="small"></p><div class="table"><table><thead><tr><th>新增原static行</th><th>二维投影中心 x / y</th><th>中心在蓝框内</th></tr></thead><tbody id="rowIds"></tbody></table></div><p class="small">列前100行，完整ID见NPZ；按帧记录，不写成全局语义标签。R2沿用R1真实光栅得到的G／alpha，只收紧合成局部带，未把剩余G强涂为白。</p><pre id="receipt"></pre></details><details><summary>原图、蓝框与真实G／alpha分量</summary><div class="pair"><figure><h3>原实拍</h3><img id="original" alt="同帧原相机实拍"></figure><figure><h3>蓝框和车底query</h3><img id="scope" alt="蓝框二维中心资格上限和车底query"></figure><figure><h3>真实G分量</h3><img id="gaussian" alt="真实剥离后高斯分量"><figcaption>完整新状态分量；最终域外显示另取existing。</figcaption></figure><figure><h3>真实alpha</h3><img id="alpha" alt="真实剥离后alpha"><figcaption>保留跨界支撑，不据此宣称全部高斯清零。</figcaption></figure></div></details><details><summary>历史CPU范围页面与原始数字</summary><div class="links"><a href="history_cpu_scope.html">历史范围预览页面（非算法结果）</a><a href="history_cpu_comparison.png">历史范围三帧图</a><a href="manifest.json">保留原CPU凭据</a></div></details></section>
</main><footer>三帧同场景局部干预 · 新增训练0步 · 无自动归属或彻底去影成功声明</footer><script src="native_data.js"></script><script src="native_viewer.js"></script></body></html>'''

JS = r'''(function(root){'use strict';const D=root.BOUNDED_NATIVE_DATA;if(!D||typeof document==='undefined')return;const $=id=>document.getElementById(id);const S={raw:113};const fmt=v=>typeof v!=='number'||!Number.isFinite(v)?'未提供':v===0?'0':Number.isInteger(v)?v.toLocaleString('en-US'):Math.abs(v)<.001?v.toExponential(3):v.toFixed(6);function image(id,path,label){const e=$(id);e.alt=label;e.style.visibility='hidden';e.onload=()=>{e.style.visibility='visible';e.classList.remove('pending')};e.onerror=()=>{e.style.visibility='visible';e.classList.add('pending');e.alt='图像未读取：'+label};if(path)e.src=path;else{e.removeAttribute('src');e.style.visibility='visible';e.classList.add('pending')}}function metric(label,v){const a=document.createElement('div');a.className='metric';const t=document.createElement('small'),n=document.createElement('strong');t.textContent=label;n.textContent=fmt(v);a.append(t,n);$('metrics').append(a)}function show(){const f=D.frames[String(S.raw)],r=f.receipt,v=f.validation;$('raw').value=String(S.raw);$('status').textContent='raw'+S.raw+' · 第三列为真实R2';$('metrics').replaceChildren();metric('从existing新增候选行',r.new_unknown_rows);metric('新增中心在蓝框外',r.new_rows_outside_blue_centers);metric('局部带外最大RGB差',v.outside_display_scope_max_delta);metric('仍跨界支撑query的框外中心行',v.residual_rows_centers_outside_blue);metric('指定query暗像素阈值计数',r.remaining_dark_effect_pixels);for(const [id,key,label] of [['existing','existing','existing基线'],['goal','historical_white','CPU局部全白目标，非算法结果'],['result','white','真实R2，细纹仍保留'],['gaussian','gaussian','真实G分量'],['alpha','alpha','真实alpha'],['original','original','原实拍'],['scope','scope','蓝框和车底查询']])image(id,f.images[key],'raw'+S.raw+' '+label);$('resultCaption').textContent='G＋(1−alpha)×白底；仍有跨界高斯支撑，未完全白。'+(S.raw===98?' AI辅助目视另提示：右侧银车左缘可能受影响。':'');$('receiptLink').href=f.receipt_path;$('evidenceLink').href=f.evidence_path;$('receipt').textContent=JSON.stringify(r,null,2);$('rowSummary').textContent='新增 '+fmt(r.new_unknown_rows)+' 行；existing已有剥离 '+fmt(v.existing_removed_rows)+' 行；仍在query贡献 '+fmt(v.residual_query_rows)+' 行，其中中心在蓝框外 '+fmt(v.residual_rows_centers_outside_blue)+' 行。原行空间original_static_rows。';$('rowIds').replaceChildren();for(const row of f.row_examples){const tr=document.createElement('tr');for(const text of [String(row.id),row.screen.map(x=>x.toFixed(3)).join(' / '),row.center_inside_blue?'是':'否']){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('rowIds').append(tr)}}function select(raw){if(!D.frames[String(raw)])return;S.raw=Number(raw);show()}$('raw').onchange=()=>select(Number($('raw').value));show();root.BoundedNative={data:D,state:S,select,show}})(typeof window==='undefined'?globalThis:window);'''


def validate_frame(source, output, raw):
    dest = source / f'{raw:03d}'
    receipt = read(dest / 'receipt.json')
    assert receipt['raw'] == raw and receipt['actual_rasterizer_rerendered'] is True
    assert receipt['training_updates'] == 0 and receipt['labels_for_training'] is False and receipt['environment_checkpoint_unchanged']
    arrays = np.load(dest / 'evidence.npz', allow_pickle=False)
    scope = np.load(output / f'{raw:03d}/diagnostic_gate.npz', allow_pickle=False)
    blue = arrays['allowed'].astype(bool)
    band = arrays['display_scope'].astype(bool) if 'display_scope' in arrays.files else scope['unknown_display_gate'].astype(bool)
    assert blue.shape == band.shape == (320, 480) and not np.any(band & ~blue)
    assert np.array_equal(blue, scope['allowed'])
    assert np.array_equal(band, scope['unknown_display_gate']), 'Final display scope must match the user local band; do not install full-blue R1.'
    existing = arrays['baseline_native_rgb']
    assert np.array_equal(existing, scope['source_native_rgb'])
    white, native, g, a, env = [arrays[k] for k in ('white_rgb', 'native_rgb', 'gaussian_rgb', 'alpha', 'environment_rgb')]
    assert a.shape == (320, 480) and all(x.shape == (3, 320, 480) for x in (existing, white, native, g, env))
    assert all(x.dtype == np.float32 for x in (existing, white, native, g, a, env))
    outside_delta = float(np.max(np.abs(white[:, ~band] - existing[:, ~band])))
    assert outside_delta == 0 and np.array_equal(native[:, ~band], existing[:, ~band])
    white_closure = float(np.max(np.abs(white[:, band] - (g+(1-a)[None])[:, band])))
    native_closure = float(np.max(np.abs(native[:, band] - (g+(1-a)[None]*env)[:, band])))
    assert white_closure < 2e-6 and native_closure < 2e-6
    new, eligible, old = arrays['new_unknown_ids'], arrays['eligible_ids'], arrays['existing_removed_ids']
    assert np.all(np.diff(new) > 0) and np.all((new >= 0) & (new < 918850))
    assert set(map(int, new)).issubset(map(int, eligible)) and not set(map(int, new)).intersection(map(int, old))
    centers = arrays['screen_centers']; assert centers.shape == (len(new), 2) and np.isfinite(centers).all()
    uv = np.rint(centers).astype(int)
    assert np.all((uv[:, 0] >= 0) & (uv[:, 0] < 480) & (uv[:, 1] >= 0) & (uv[:, 1] < 320))
    inside = blue[uv[:, 1], uv[:, 0]]; assert inside.all()
    assert len(new) == receipt['new_unknown_rows'] and receipt['new_rows_outside_blue_centers'] == 0
    assert sum(h['new_rows'] for h in receipt['history'][:-1]) == len(new)
    query = arrays['query'].astype(bool); assert not np.any(query & ~blue)
    assert abs(float(a[query].mean()) - receipt['final_query_alpha_mean']) < 2e-6
    effect = scope['user_contact_query'].astype(bool)
    assert abs(float(a[effect].mean()) - receipt['final_effect_alpha_mean']) < 2e-6
    dark = (np.clip(white, 0, 1)*np.array([.2126, .7152, .0722])[:, None, None]).sum(0) < .2
    assert int((dark & effect).sum()) == receipt['remaining_dark_effect_pixels']
    residual = arrays['residual_ids']; residual_outside = int((~np.isin(residual, eligible)).sum())
    assert residual_outside == receipt['history'][-1]['outside_center_remaining_rows']
    for name, values in [('white', white), ('native', native), ('gaussian', g)]:
        image = np.array(Image.open(dest / f'{name}.png').convert('RGB'))
        assert image.shape == (320, 480, 3) and np.array_equal(image, display(values)), name
    alpha_png = np.array(Image.open(dest / 'alpha.png'))
    assert np.array_equal(alpha_png, np.round(np.clip(a, 0, 1)*255).astype('uint8'))
    assert np.array_equal(np.array(Image.open(output / f'{raw:03d}/existing.png').convert('RGB')), display(existing))
    record = dict(receipt=receipt, receipt_path=f'native/{raw:03d}/receipt.json', evidence_path=f'native/{raw:03d}/evidence.npz', images=dict(existing=f'{raw:03d}/existing.png', scope=f'{raw:03d}/scope.png', original=f'{raw:03d}/original.png', white=f'native/{raw:03d}/white.png', native=f'native/{raw:03d}/native.png', gaussian=f'native/{raw:03d}/gaussian.png', alpha=f'native/{raw:03d}/alpha.png', display_scope=f'native/{raw:03d}/display_scope.png', historical_white=f'{raw:03d}/white.png'), row_examples=[dict(id=int(row), screen=center.tolist(), center_inside_blue=bool(ok)) for row, center, ok in zip(new[:100], centers[:100], inside[:100])], validation=dict(existing_removed_rows=len(old), eligible_rows=len(eligible), display_scope_pixels=int(band.sum()), blue_pixels=int(blue.sum()), outside_display_scope_max_delta=outside_delta, white_formula_max_delta=white_closure, native_formula_max_delta=native_closure, residual_query_rows=len(residual), residual_rows_centers_outside_blue=residual_outside))
    return record, band


def write_comparison(output):
    sheet = Image.new('RGB', (1440, 1056), (12, 24, 37)); draw = ImageDraw.Draw(sheet)
    font_paths = ('C:/Windows/Fonts/msyh.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    font_path = next((value for value in font_paths if Path(value).exists()), None)
    if font_path:
        font = ImageFont.truetype(font_path, 18)
        labels = ('existing 基线', '用户局部全白目标（CPU范围）', '真实 R2（细纹保留，仍未全白）')
    else:
        font = ImageFont.load_default()
        labels = ('existing baseline', 'user local-white goal (CPU fill)', 'actual R2 (fine traces remain)')
    for row, raw in enumerate(RAWS):
        paths = (output / f'{raw:03d}/existing.png', output / f'{raw:03d}/white.png', output / f'native/{raw:03d}/white.png')
        for col, (label, image) in enumerate(zip(labels, paths)):
            x, y = col*480, row*352
            draw.text((x+8, y+3), f'raw{raw} | {label}', font=font, fill='white')
            sheet.paste(Image.open(image).convert('RGB'), (x, y+32))
    sheet.save(output / 'comparison.png')


def install(source, output=OUT):
    summary = read(source / 'summary.json')
    assert summary['training_updates'] == 0
    assert {r['raw'] for r in summary['records']} == set(RAWS)
    # 验证全部三帧后才写输出，不把未到达或scope错误的R1作为默认。
    verified = {raw: validate_frame(source, output, raw) for raw in RAWS}
    assert all((output / f'{raw:03d}/scope.png').exists() and (output / f'{raw:03d}/original.png').exists() for raw in RAWS)
    manifest = dict(schema='bounded-native-white-ui-v1', source=str(source.resolve()), default_mode='white', frames={str(raw): value[0] for raw, value in verified.items()}, center_eligibility='2D projected Gaussian center within user blue polygon; not a 3D entity box', outside_preservation='same-frame existing branch outside the user local display band', white_equation='G_after+(1-alpha_after)*1 inside display band', transferable_semantic_labels=False, training_updates=0, human_verdict=None)
    if not (output / 'history_cpu_comparison.png').exists():
        shutil.copy2(output / 'comparison.png', output / 'history_cpu_comparison.png')
    history = output / 'history_cpu_scope.html'
    if not history.exists():
        old = (output / 'index.html').read_text(encoding='utf8')
        notice = '<div style="padding:18px;background:#58401b;color:#fff3d5"><strong>历史CPU范围预览：这里是纯白硬覆盖，不是原生删行算法。</strong> 原公式为(1-M)×existing+M×白；旧数字仅作范围沟通。<a style="color:white" href="index.html">查看当前真实原生白底结果 →</a></div>'
        old = old.replace('</style>', '</style>'+notice, 1).replace('href="comparison.png"', 'href="history_cpu_comparison.png"')
        history.write_text(old, encoding='utf8')
    output.mkdir(parents=True, exist_ok=True)
    for raw, (_, band) in verified.items():
        target = output / f'native/{raw:03d}'; target.mkdir(parents=True, exist_ok=True)
        for name in ('white.png', 'native.png', 'gaussian.png', 'alpha.png', 'evidence.npz', 'receipt.json'):
            shutil.copy2(source / f'{raw:03d}' / name, target / name)
        Image.fromarray((band*255).astype('uint8')).save(target / 'display_scope.png')
    shutil.copy2(source / 'summary.json', output / 'native_summary.json')
    review_path = HERE / 'bounded_white_v1/ai_visual_review.json'
    if review_path.exists():
        review = read(review_path); assert review['human_verdict'] is None and review['ground_truth'] is False
        shutil.copy2(review_path, output / 'ai_visual_review.json')
    (output / 'native_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    (output / 'native_data.js').write_text('window.BOUNDED_NATIVE_DATA='+json.dumps(manifest, ensure_ascii=False, separators=(',', ':'))+';\n', encoding='utf8')
    (output / 'native_viewer.js').write_text(JS, encoding='utf8')
    (output / 'native_architecture.svg').write_text(arch(), encoding='utf8')
    (output / 'index.html').write_text(HTML, encoding='utf8')
    write_comparison(output)
    validation = dict(schema='bounded-native-ui-validation-v1', real_frames=3, real_pngs_copied=12, center_eligibility_checked=True, display_scope_matches_user_local_band=True, output_outside_scope_unchanged=True, float32_white_and_native_equations_checked=True, retained_G_not_hard_overpainted=True, historical_cpu_preview_separate=True, browserQA=False, shared_entry_modified=False, quality_success_claimed=False, records=[dict(raw=raw, **record['validation']) for raw, (record, _) in verified.items()])
    (output / 'native_ui_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf8')
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='仅使用主线程确认已完整取回且采用最终局部带scope的真实native目录')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(install(args.source.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
