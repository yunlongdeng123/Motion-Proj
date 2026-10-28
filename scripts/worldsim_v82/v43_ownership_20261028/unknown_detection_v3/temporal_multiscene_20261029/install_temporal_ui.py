"""Register genuine 3 x 49 bounded-unknown frames in the shared offline viewer.

CPU only. Default audits inputs without changing outputs; --install performs the
validated registration. Existing experiment groups and the accepted R2 report stay
intact. The existing renderer's original-static row-ID filter is reused directly.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
WORK = Path(__file__).parent
OUT = BASE / 'outputs/GS_Scene_Diagnostics_20261027'
CONTEXTS = ('14705303724557273004_3105_000_3125_000', '4295449061847708198_3769_000_3789_000', '473735159277431842_630_095_650_095')
GID = 'v43_bounded_unknown'
BASE_GROUP = 'v43_actor_10000'
ARM = 'source_body_contact'
VERSION = 'bounded-unknown-20261029-1'
PNG_NAMES = ('white', 'original', 'actor_omitted', 'native', 'gaussian', 'alpha', 'observed', 'ungated_white')
NOTE = '二维只在本帧局部显示范围合成 G_after＋(1−alpha_after)×白底；范围外保留同帧 actor-omitted 分支。右侧三维按同帧原static行ID真实删行，Actor为空；它展示候选删行几何，不表示框外像素保护已变成全局三维修复，也不是可迁移语义标签。'
VISUAL_NOTE = '机械审核通过不等于全帧视觉通过：场景3 raw13仍有暗带，raw37等帧存在邻车边界问题，需逐帧核对。'
REPORT_RELATIVE = '../V43_Ownership_20261027/gpu_v43/temporal_multiscene_20261029/index.html'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment(path):
    source = path.read_text(encoding='utf8')
    return json.JSONDecoder().raw_decode(source[source.index('=') + 1:])[0]


def scalar(arrays, key):
    return np.asarray(arrays[key]).item()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_frame(folder, context, raw, ordinal, frame, static_meta, *, light=False, remote_frame=None):
    receipt = read_json(folder / 'receipt.json')
    prefix = f'{context}/raw{raw}'
    for key, expected in [('context', context), ('raw', raw), ('camera_index', ordinal), ('status', 'ok'), ('arm', ARM), ('query_mode', ARM), ('independent_frame_reset', True), ('final_after_recomputed', True), ('after_attribution_recomputed', True), ('actual_rasterizer_rerendered', True), ('training_updates', 0), ('labels_for_training', False), ('semantic_unknown_claim', False)]:
        check(receipt.get(key) == expected, f'{prefix}: receipt {key} mismatch')
    check(str(receipt['frame_id']) == str(frame['frame_id']), prefix + ': source frame_id mismatch')
    check(abs(float(receipt['model_time']) - frame['model_time']) < 1e-7, prefix + ': query time mismatch')
    check(receipt['static_rows'] == static_meta['total'], prefix + ': static population mismatch')
    expected_namespace = static_meta['checkpoint'].rsplit('/', 1)[0] + '@30000:static'
    check(receipt['row_namespace'] == expected_namespace, prefix + ': original-static namespace mismatch')
    check(receipt['human_verdict'] is None, prefix + ': human verdict must remain unfilled')
    evidence_name = 'ui_evidence.npz' if light else 'evidence.npz'
    if light:
        check(remote_frame and remote_frame.get('verified') is True and remote_frame.get('reset_verified') is True, prefix + ': verified remote full-float audit required')
        for key in ('context', 'raw', 'row_namespace', 'frame_id', 'camera_index', 'model_time', 'checkpoint_ref'):
            check(str(remote_frame.get(key)) == str(receipt.get(key)), prefix + ': remote audit ' + key + ' mismatch')
        check(remote_frame.get('after_payload_present') is True and remote_frame.get('outside_display_max_delta') == 0 and remote_frame.get('inside_white_equation_max_error') is not None and remote_frame['inside_white_equation_max_error'] <= 2e-6, prefix + ': remote full-float composition audit not closed')
    with np.load(folder / evidence_name, allow_pickle=False) as arrays:
        if light:
            check('full_evidence' in arrays and scalar(arrays, 'full_evidence') is False, prefix + ': light evidence must explicitly mark full_evidence=false')
        for key, value in [('context', context), ('raw', raw), ('row_namespace', expected_namespace), ('checkpoint_ref', receipt['checkpoint_ref'])]:
            check(scalar(arrays, key) == value, prefix + ': evidence ' + key + ' mismatch')
        original_ids = np.asarray(arrays['final_removed_ids'])
        check(original_ids.ndim == 1 and original_ids.dtype.kind in 'iu', prefix + ': invalid row IDs')
        ids = original_ids.astype(np.int64)
        check(not len(ids) or (ids[0] >= 0 and ids[-1] < static_meta['total'] and np.all(np.diff(ids) > 0)), prefix + ': row IDs must be ascending unique original-static indices')
        check(len(ids) == receipt['final_removed_rows'], prefix + ': receipt row count mismatch')
        centers = np.asarray(arrays['screen_centers'])
        check(centers.shape == (len(ids), 2) and np.isfinite(centers).all(), prefix + ': removed-row screen centers mismatch')
        masks = {}
        for key in ('allowed', 'query', 'display_scope', 'body', 'contact'):
            mask = np.asarray(arrays[key])
            check(mask.shape == (320, 480) and np.isin(mask, [0, 1]).all(), prefix + ': invalid ' + key + ' mask')
            masks[key] = mask.astype(bool)
        check(not (masks['display_scope'] & ~masks['allowed']).any(), prefix + ': display scope outside allowed region')
        uv = np.rint(centers).astype(np.int64)
        check(not len(ids) or (np.all((uv[:, 0] >= 0) & (uv[:, 0] < 480) & (uv[:, 1] >= 0) & (uv[:, 1] < 320)) and masks['allowed'][uv[:, 1], uv[:, 0]].all()), prefix + ': selected center outside blue eligibility region')
        if 'eligible_ids' in arrays:
            check(np.isin(ids, arrays['eligible_ids']).all(), prefix + ': removed IDs outside actual eligible set')
        if 'starting_removed_ids' in arrays:
            check(len(arrays['starting_removed_ids']) == 0, prefix + ': this arm must reset independently')
        yy, xx = np.where(masks['body'])
        check(len(xx) > 0, prefix + ': no observed body mask')
        bbox = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
        check(np.allclose(bbox, receipt['body_bbox_xyxy'], rtol=0, atol=1e-7), prefix + ': body bbox mismatch')
        domain = masks['display_scope']
        inside_error = outside_error = None
        float_images = {}
        if not light:
            a = np.asarray(arrays['alpha']); g = np.asarray(arrays['gaussian_rgb']); white = np.asarray(arrays['white_rgb']); baseline = np.asarray(arrays['baseline_native_rgb'])
            check(a.shape == (320, 480) and g.shape == white.shape == baseline.shape == (3, 320, 480), prefix + ': float-map shape mismatch')
            check(all(np.isfinite(x).all() for x in (a, g, white, baseline)), prefix + ': nonfinite float maps')
            inside_error = float(np.abs(white[:, domain] - (g + (1 - a)[None])[:, domain]).max(initial=0))
            outside_error = float(np.abs(white[:, ~domain] - baseline[:, ~domain]).max(initial=0))
            check(inside_error <= 2e-6 and outside_error == 0, prefix + ': white composition or outside preservation failed')
            float_images = {'white': white, 'actor_omitted': baseline, 'gaussian': g, 'alpha': a}
        check(receipt['new_rows_outside_blue_centers'] == 0 and receipt['outside_output_max_difference'] == 0, prefix + ': receipt scope guarantee failed')
        check(int(domain.sum()) == receipt['display_pixels'] and int(masks['query'].sum()) == receipt['query_pixels'], prefix + ': mask count mismatch')
        png_hashes = {}
        for name in PNG_NAMES:
            file = folder / (name + '.png')
            check(file.is_file(), prefix + ': missing real ' + name + '.png')
            with Image.open(file) as image:
                check(image.size == (480, 320), prefix + ': PNG dimensions mismatch')
                pixels = np.asarray(image.convert('L' if name == 'alpha' else 'RGB'))
            if name in float_images:
                expected = float_images[name]
                if expected.ndim == 3:
                    expected = expected.transpose(1, 2, 0)
                expected = np.round(np.clip(expected, 0, 1) * 255).astype(np.uint8)
                check(np.abs(pixels.astype(int) - expected.astype(int)).max(initial=0) <= 1, prefix + ': PNG does not match its float evidence')
            png_hashes[name] = sha(file)
        check(np.array_equal(np.asarray(Image.open(folder / 'white.png').convert('RGB'))[~domain], np.asarray(Image.open(folder / 'actor_omitted.png').convert('RGB'))[~domain]), prefix + ': saved PNG outside differs from actor-omitted PNG')
    return dict(source_raw=raw, frame_id=receipt['frame_id'], camera_index=ordinal, model_time=receipt['model_time'], row_namespace=expected_namespace, final_removed_ids=ids.tolist(), screen_centers=centers.tolist(), removed_rows=len(ids), body_bbox_xyxy=bbox, receipt=receipt, png_sha256=png_hashes, evidence_name=evidence_name, evidence_sha256=sha(folder / evidence_name), receipt_sha256=sha(folder / 'receipt.json'), local_full_float_reloaded=not light, local_white_inside_max_error=inside_error, local_outside_max_error=outside_error, remote_full_float_audit=remote_frame if light else None)


def read_remote_audit(path):
    audit = read_json(path)
    check(audit.get('schema') == 'frame-independent-bounded-temporal-audit-v1', 'unsupported remote full audit schema')
    check(audit.get('producer_frame_reset_verified') is True and not audit.get('possible_carryover_detected'), 'remote reset audit not verified')
    frames = {}
    for group in audit['groups']:
        if group['method_arm'] != ARM:
            continue
        check(not group['missing_expected_raws'] and not group['skipped_frames'], 'remote full audit has missing/skipped frames')
        for frame in group['frames']:
            key = (group['context'], int(frame['raw']))
            check(key not in frames, 'duplicate frame in remote audit')
            frames[key] = frame
    check(len(frames) == 147, 'remote full audit must cover 147 source_body_contact frames')
    return frames


def audit_inputs(input_root, *, light=False, remote_audit=None):
    remote_frames = read_remote_audit(remote_audit) if light else {}
    deliveries = []
    for number, context in enumerate(CONTEXTS, 1):
        scene = OUT / context
        data = read_json(scene / 'scene.json')
        v43 = read_json(scene / 'v43_groups_data.json')
        static_meta = assignment(scene / 'cloud_full_manifest.js')['static']
        raws = sorted(map(int, data['frames']))
        check(len(raws) == 49, context + ': expected 49 source frames')
        check(BASE_GROUP in v43['geometry'], context + ': missing actual V43 geometry')
        frames = {}
        for ordinal, raw in enumerate(raws):
            state = v43['geometry'][BASE_GROUP].get(str(raw))
            check(state and state.get('supported') and abs(state['world_time'] - data['frames'][str(raw)]['model_time']) < 1e-7, context + ': missing same-frame geometry')
            frames[str(raw)] = validate_frame(input_root / context / ARM / f'{raw:03d}', context, raw, ordinal, data['frames'][str(raw)], static_meta, light=light, remote_frame=remote_frames.get((context, raw)))
        deliveries.append(dict(schema='bounded-unknown-delivery-v1', context=context, scene_number=number, group_id=GID, geometry_base_group=BASE_GROUP, arm=ARM, static_total=static_meta['total'], static_checkpoint=static_meta['checkpoint'], row_namespace=frames[str(raws[0])]['row_namespace'], light_delivery=light, local_full_float_reloaded=not light, remote_audit_sha256=sha(remote_audit) if light else None, frames=frames))
    return deliveries


def block(source, name, body, *, after=None, before_pattern=None):
    start, end = '<!-- ' + name + '-start -->', '<!-- ' + name + '-end -->'
    value = start + body + end
    if start in source:
        return re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: value, source, count=1, flags=re.S)
    if after is not None:
        check(after in source, 'HTML insertion marker missing: ' + after)
        return source.replace(after, after + value, 1)
    updated, count = re.subn(before_pattern, lambda _: value + _.group(0), source, count=1)
    check(count == 1, 'HTML viewer insertion marker missing')
    return updated


def link(context, raw, text):
    return '<a class="button" style="display:inline-block;padding:7px 11px;border:1px solid #8472b3;border-radius:5px;background:#2c2342" href="' + context + '/index.html#workspace?group=' + GID + '&amp;raw=' + str(raw) + '&amp;channel=rgb">' + html.escape(text) + '</a>'


ARCHITECTURE = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 136" role="img" aria-label="同帧输入、真实删行重算、二维局部合成与三维删行展示"><rect width="1060" height="136" rx="8" fill="#101c2d"/><g fill="#1b2f44" stroke="#667391"><rect x="12" y="20" width="268" height="96" rx="7"/><rect x="320" y="20" width="340" height="96" rx="7"/><rect x="700" y="20" width="348" height="96" rx="7"/></g><g fill="#e7effa" font-family="Microsoft YaHei,sans-serif" font-size="15" text-anchor="middle"><text x="146" y="47">同帧冻结参数＋原body范围</text><text x="146" y="74">Actor omitted → 中心投影约束</text><text x="146" y="99">每帧独立空集合起步</text><text x="490" y="47">真实高斯贡献 → 增加候选行</text><text x="490" y="74">删行 → after重新光栅化</text><text x="490" y="99">再查贡献 → 无新增或预算结束</text><text x="874" y="47">二维：局部G＋T白底／范围外同帧保留</text><text x="874" y="74">三维：原static行ID过滤</text><text x="874" y="99">两种展示边界分别说明</text></g><path d="M280 68H309 M660 68H689" stroke="#a6b8df" stroke-width="2"/><g fill="#a6b8df"><polygon points="309,63 319,68 309,73"/><polygon points="689,63 699,68 689,73"/></g></svg>'''


REGISTRATION = r'''(function(root){
  const M=__MANIFEST__;
  root.BOUNDED_UNKNOWN_DELIVERY=M;
  const originals=root.EDIT_GEOMETRY.groups[M.geometry_base_group],geometry={},frames={};
  for(const [raw,f]of Object.entries(M.frames)){
    const source=originals[raw];if(!source?.supported)throw Error('bounded unknown missing same-frame geometry '+raw);
    const q=JSON.parse(JSON.stringify(source)),target=q.objects.find(o=>o.key==='target');
    q.objects=[];q.static_visible=true;q.static_suppressed_rows=f.final_removed_ids;
    delete q.static_adapter;delete q.semantic_render_filter;delete q.target_detection;
    q.reference_boxes=target?[{target:true,track:root.SCENE_DATA.target_track,label:'原目标参考框（Actor已移除）',corners_local:target.corners_local}]:[];
    q.bounded_unknown={source_raw:Number(raw),row_namespace:M.row_namespace,removed_rows:f.removed_rows,actor_omitted:true,geometry_scope:'selected-row-removal; not pixel-preserved world repair'};
    q.note=M.scope_note;geometry[raw]=q;
    frames[raw]={channels:{rgb:f.images.white},bounded_unknown_receipt:f.receipt_path};
  }
  root.SCENE_DATA.groups=root.SCENE_DATA.groups.filter(g=>g.id!==M.group_id);
  root.SCENE_DATA.groups.push({id:M.group_id,title:'V43 局部unknown原生诊断 · 49帧',category:'v43',kind:'v43_bounded_unknown',frames,summary:'bounded_unknown/install_receipt.json'});
  root.EDIT_GEOMETRY.groups[M.group_id]=geometry;
})(window);
'''


RUNTIME = r'''(function(root){
  'use strict';const M=root.BOUNDED_UNKNOWN_DELIVERY,a=root.SceneViewer;if(!M||!a)return;
  const get=id=>document.getElementById(id);let previousGroup=null;
  function setup(){if(S.group===M.group_id&&previousGroup!==M.group_id){
    get('geometryState').value='current';get('semanticAblation').value='all';get('cloudColor').value='branch';S.cameraView=true;S.cameraReference='current';
  }previousGroup=S.group;}
  function update(){const active=S.group===M.group_id,frame=M.frames[String(S.raw)];get('boundedUnknownFrame').hidden=!active;
    if(!active||!frame)return;get('boundedUnknownFrameStatus').textContent='raw'+S.raw+' · 原static候选删行 '+frame.removed_rows.toLocaleString()+' · '+frame.row_namespace+' · 每帧独立重算，未训练';
    const links=get('boundedUnknownEvidence');links.replaceChildren();
    for(const [name,label]of [['original','同帧V43完整原生RGB'],['actor_omitted','同帧Actor omitted'],['native','实际环境合成'],['gaussian','删行后Gaussian分量'],['alpha','删行后alpha'],['observed','原始实拍']]){const e=document.createElement('a');e.href=frame.images[name];e.target='_blank';e.textContent=label+' ↗';e.style.marginRight='14px';links.append(e);}
    for(const [url,label]of [[frame.receipt_path,'真实receipt'],[frame.row_ids_path,'候选行ID与屏幕中心'],[frame.evidence_path,M.light_delivery?'轻量NPZ证据':'完整NPZ证据']]){const e=document.createElement('a');e.href=url;e.target='_blank';e.textContent=label+' ↗';e.style.marginRight='14px';links.append(e);}
  }
  const beforeDraw=draw3D;draw3D=function(){beforeDraw();if(S.group!==M.group_id)return;
    const current=get('geometryState').value!=='baseline'&&!['gt','sensor_overlay','semantic_2d'].includes(S.channel);
    get('geometryStateInfo').textContent=current?'当前raw实际候选删行几何；Actor omitted；二维框外同帧像素保护不等于全局三维修复。':'当前是原场景参照；未显示本组候选删行状态。';
    if(current)get('semanticFilterStatus').textContent='本组固定过滤final_removed_ids，与全局基础unknown独立。3D只显示高斯中心；白底是二维局部合成，跨界高斯支撑并未被伪装成全局修复。'+(get('semanticAblation').value!=='all'?' 当前还开启了额外的用户语义过滤。':'');
  };
  const beforeShow=show;show=async function(...args){setup();const raw=S.raw,group=S.group,result=beforeShow(...args);update();await result;if(S.raw===raw&&S.group===group)update();};
  Object.assign(a,{show,draw3D});root.BoundedUnknown={manifest:M,update};setup();update();if(S.group===M.group_id)draw3D();
})(window);
'''


def build_products(deliveries, input_root):
    products, copies, scene_manifests = {}, [], []
    for delivery in deliveries:
        context = delivery['context']; scene = OUT / context
        manifest = {key: value for key, value in delivery.items() if key != 'frames'}
        manifest.update(scope_note=NOTE, preferred_raw=(113 if delivery['scene_number'] == 1 else 181 if delivery['scene_number'] == 2 else 45), frames={})
        for raw, frame in delivery['frames'].items():
            prefix = 'assets/v43_bounded_unknown/' + f'{int(raw):03d}'
            evidence_prefix = 'bounded_unknown/' + f'{int(raw):03d}'
            entry = {key: copy.deepcopy(value) for key, value in frame.items() if key not in ('receipt', 'screen_centers')}
            entry.update(images={name: prefix + '/' + name + '.png' for name in PNG_NAMES}, receipt_path=evidence_prefix + '/receipt.json', evidence_path=evidence_prefix + '/' + frame['evidence_name'], row_ids_path=evidence_prefix + '/selected_rows.json')
            manifest['frames'][raw] = entry
            source = input_root / context / ARM / f'{int(raw):03d}'
            for name in PNG_NAMES:
                target = scene / entry['images'][name]
                copies.append((source / (name + '.png'), target))
                products[target.with_name(target.name + '.js')] = 'window.IMAGE_DATA[' + js(entry['images'][name]) + ']=' + js('data:image/png;base64,' + base64.b64encode((source / (name + '.png')).read_bytes()).decode('ascii')) + ';\n'
            copies.extend([(source / 'receipt.json', scene / entry['receipt_path']), (source / frame['evidence_name'], scene / entry['evidence_path'])])
            products[scene / entry['row_ids_path']] = js(dict(context=context, source_raw=int(raw), row_namespace=manifest['row_namespace'], final_removed_ids=frame['final_removed_ids'], screen_centers=frame['screen_centers'], center_rule='np.rint pixel center lies in allowed; not 3D entity box', semantic_label=False))
        products[scene / 'bounded_unknown_data.js'] = REGISTRATION.replace('__MANIFEST__', js(manifest))
        compact = copy.deepcopy(manifest)
        for frame in compact['frames'].values():
            frame.pop('final_removed_ids')
        products[scene / 'bounded_unknown_manifest.json'] = json.dumps(compact, ensure_ascii=False, indent=2)
        receipt = dict(schema='bounded-unknown-ui-install-v1', context=context, frames=len(manifest['frames']), group=GID, actual_native_rgb=True, geometry_filter='original static_suppressed_rows', actor_objects_empty=True, row_namespace=manifest['row_namespace'], local_full_float_reloaded=not delivery['light_delivery'], remote_audit_sha256=delivery['remote_audit_sha256'], semantic_ground_truth=False, browserQA=False, training_updates=0)
        products[scene / 'bounded_unknown/install_receipt.json'] = json.dumps(receipt, ensure_ascii=False, indent=2)
        page = (scene / 'index.html').read_text(encoding='utf8')
        page = block(page, 'bounded-unknown-data', '<script src="bounded_unknown_data.js?v=' + VERSION + '"></script>', before_pattern=r'<script src="\.\./viewer\.js(?:\?[^\"]*)?"></script>')
        page = block(page, 'bounded-unknown-runtime', '<script src="../scene_bounded_unknown.js?v=' + VERSION + '"></script>', before_pattern=r'</body>')
        direct = '#workspace?group=' + GID + '&amp;raw=' + str(manifest['preferred_raw']) + '&amp;channel=rgb'
        validation_note = '<p>浮点方程与after归因由<a href="../bounded_unknown_remote_full_audit.json">远端完整审核</a>核验；本地只读取轻量证据核对行ID、投影、范围、PNG与几何接线。</p>' if delivery['light_delivery'] else '<p>本地已读取完整浮点证据核对G＋T白底方程和范围外同帧保留，并核对行ID、投影、PNG与几何接线。</p>'
        known_issues = '<p class="notice warn">' + VISUAL_NOTE + '</p>'
        if delivery['scene_number'] == 3:
            known_issues += '<p><a href="#workspace?group=' + GID + '&amp;raw=13&amp;channel=rgb">检查raw13暗带 →</a>　<a href="#workspace?group=' + GID + '&amp;raw=37&amp;channel=rgb">检查raw37邻车边界 →</a></p>'
        panel = '<section class="panel notice" id="boundedUnknown"><h2>逐帧局部unknown原生诊断 · 49帧独立重算</h2><p>' + NOTE + '</p><p>范围由原始body框传递蓝色多边形；eligible是二维投影中心约束。此组从空集合逐帧起步，与先前接受的R2初始化不同；保留旧R2报告和全部旧组。</p>' + known_issues + validation_note + '<a class="button" style="display:inline-block;padding:8px 12px;border:1px solid #8a78b5;border-radius:5px" href="' + direct + '">打开本场景49帧真实白底RGB＋删行几何 →</a><details><summary>组件与数据流</summary><img class="architecture" src="../temporal_unknown_architecture.svg" alt="同帧输入、真实删行重算、二维合成与三维删行"></details><div id="boundedUnknownFrame" hidden><p id="boundedUnknownFrameStatus"></p><div id="boundedUnknownEvidence" class="subtools"></div></div><p><a href="bounded_unknown_manifest.json">147帧任务中的本场景实际清单 →</a></p></section>'
        products[scene / 'index.html'] = block(page, 'bounded-unknown-panel', panel, after='<main>')
        scene_manifests.append(receipt)
    shared = (OUT / 'index.html').read_text(encoding='utf8')
    links = ' '.join(link(m['context'], 113 if m['scene_number'] == 1 else 181 if m['scene_number'] == 2 else 45, '场景' + str(m['scene_number']) + ' · 49帧局部unknown') for m in deliveries)
    remote_note = '<p>本地为轻量接线核对；浮点方程与after归因见<a href="bounded_unknown_remote_full_audit.json">远端完整审核</a>。</p>' if deliveries[0]['light_delivery'] else ''
    report_link = '<p><a href="' + REPORT_RELATIVE + '">147帧时序、邻车边界与坏例报告 →</a></p>'
    shared = block(shared, 'bounded-unknown-summary', '<section class="notice"><strong>新增147帧独立原生诊断：</strong>' + links + '<p>默认真实white RGB；右侧按同raw原static行ID删行。范围外保留同帧actor-omitted像素，不宣称全局三维修复或语义标签；旧R2与全部历史组保留。</p><p>' + VISUAL_NOTE + '</p>' + remote_note + report_link + '</section>', after='<main id="workspace">')
    for delivery in deliveries:
        context = delivery['context']; raw = 113 if delivery['scene_number'] == 1 else 181 if delivery['scene_number'] == 2 else 45
        def card(match):
            source = match[0]
            if context not in source:
                return source
            addition = '<p>' + link(context, raw, '新增：49帧局部unknown诊断 →') + '</p>'
            return block(source, 'bounded-unknown-card', addition, before_pattern=r'</article>')
        shared = re.sub(r'<article\b[^>]*>.*?</article>', card, shared, flags=re.S)
    products[OUT / 'index.html'] = shared
    products[OUT / 'scene_bounded_unknown.js'] = RUNTIME
    products[OUT / 'temporal_unknown_architecture.svg'] = ARCHITECTURE
    products[OUT / 'bounded_unknown_install.json'] = json.dumps(dict(schema='bounded-unknown-multiscene-install-v1', frames=147, scenes=scene_manifests, local_full_float_reloaded=not deliveries[0]['light_delivery'], remote_audit_sha256=deliveries[0]['remote_audit_sha256'], accepted_R2_report_modified=False, old_groups_modified=False, browserQA=False, ssh_used=False), ensure_ascii=False, indent=2)
    return products, copies


def install(deliveries, input_root, *, remote_audit=None):
    products, copies = build_products(deliveries, input_root)
    if remote_audit:
        copies.append((remote_audit, OUT / 'bounded_unknown_remote_full_audit.json'))
    # Plan and validate every destination before writing any output.
    for file in [*products, *(target for _, target in copies)]:
        check(file.resolve().is_relative_to(OUT.resolve()), 'destination outside shared workspace')
    protected = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html'
    protected_sha = sha(protected)
    prior_ids = {ctx: [group['id'] for group in read_json(OUT / ctx / 'v43_groups_data.json')['groups']] for ctx in CONTEXTS}
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        check(sha(source) == sha(target), 'copy checksum failed: ' + str(target))
    for file, source in products.items():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding='utf8')
    check(protected_sha == sha(protected), 'accepted R2 page unexpectedly changed')
    check(prior_ids == {ctx: [group['id'] for group in read_json(OUT / ctx / 'v43_groups_data.json')['groups']] for ctx in CONTEXTS}, 'prior V43 manifest unexpectedly changed')
    return dict(schema='bounded-unknown-local-install-qa-v1', total_frames=147, real_pngs_copied=len(deliveries) * 49 * len(PNG_NAMES), output_files_written=len(products) + len(copies), local_full_float_reloaded=not deliveries[0]['light_delivery'], remote_full_audit_sha256=sha(remote_audit) if remote_audit else None, protected_R2_sha256=protected_sha, original_v43_group_ids_preserved=prior_ids, actual_static_row_filter=True, browserQA=False, ssh_used=False)


CANVAS_QA = r'''
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const base=__BASE__,out=path.join(base,'outputs/GS_Scene_Diagnostics_20261027'),contexts=__CONTEXTS__;
const bundledCanvas='C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas';
const {createCanvas,loadImage}=require(process.env.CODEX_CANVAS_MODULE||(fs.existsSync(bundledCanvas)?bundledCanvas:'@napi-rs/canvas'));
function fakeDocument(){const nodes=new Map(),doc={nodes};doc.createElement=tag=>{const e={tagName:tag,children:[],style:{},classList:{toggle(){}},value:'',checked:true,textContent:'',append(...x){this.children.push(...x);},replaceChildren(...x){this.children=[...x];},addEventListener(){}};Object.defineProperty(e,'options',{get(){return this.children;}});return e;};doc.getElementById=id=>{if(nodes.has(id))return nodes.get(id);const e=doc.createElement('div');if(/Canvas$/.test(id)||id==='sceneCanvas'){const c=createCanvas(id==='sceneCanvas'?840:480,id==='sceneCanvas'?560:320);e.width=c.width;e.height=c.height;e.nativeCanvas=c;e.getContext=(...args)=>c.getContext(...args);if(id==='sceneCanvas')e.getContext('2d').drawImage=()=>{throw Error('3D RGB backdrop prohibited');};}e.id=id;e.value=({geometryState:'current',pointSize:'2.5',cloudColor:'branch',semanticProfile:'gpu',semanticBoundary:'0',semanticBoundaryMass:'.2',semanticAblation:'all'})[id]||'';if(id==='lidar3d')e.checked=false;nodes.set(id,e);return e;};return doc;}
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
(async()=>{const results=[];
for(const context of contexts){
 const folder=path.join(out,context),doc=fakeDocument(),sb={console,document:doc,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Uint32Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,URL,location:{hash:''},setTimeout};sb.window=sb;sb.addEventListener=()=>{};vm.createContext(sb);
 for(const file of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js'])vm.runInContext(fs.readFileSync(path.join(folder,file),'utf8'),sb,{filename:file});
 const oldIds=Array.from(sb.SCENE_DATA.groups,g=>g.id);if(fs.existsSync(path.join(folder,'target_detection_data.js')))vm.runInContext(fs.readFileSync(path.join(folder,'target_detection_data.js'),'utf8'),sb);
 vm.runInContext(fs.readFileSync(path.join(folder,'bounded_unknown_data.js'),'utf8'),sb,{filename:'bounded_unknown_data.js'});
 const M=sb.BOUNDED_UNKNOWN_DELIVERY,group=sb.SCENE_DATA.groups.find(g=>g.id===M.group_id);assert.equal(sb.SCENE_DATA.groups.filter(g=>g.id===M.group_id).length,1);assert(oldIds.every(id=>sb.SCENE_DATA.groups.some(g=>g.id===id)));assert.equal(Object.keys(group.frames).length,49);
 const viewer=fs.readFileSync(path.join(out,'viewer.js'),'utf8').replace(/init\(\);\s*$/,'');vm.runInContext(viewer,sb);const a=sb.SceneViewer;a.S.clouds={};
 for(const [key,value]of Object.entries({...sb.CLOUD_DATA,...sb.FOREIGN_CLOUDS}))a.S.clouds[key]={...value,data:await a.unpack(value.data)};
 doc.getElementById('cloudColor').value='branch';a.S.group=M.group_id;a.S.channel='rgb';a.S.cameraView=true;const raws=Object.keys(M.frames).map(Number).sort((x,y)=>x-y);let wrappers=0;const samples=[];
 for(const raw of raws){const f=M.frames[raw],state=sb.EDIT_GEOMETRY.groups[M.group_id][raw],download=JSON.parse(fs.readFileSync(path.join(folder,f.row_ids_path),'utf8'));
  assert.equal(state.objects.length,0);assert.deepEqual(Array.from(state.static_suppressed_rows),download.final_removed_ids);assert.equal(state.world_time,f.model_time);assert(!state.static_adapter&&!state.semantic_render_filter);assert.equal(group.frames[raw].channels.rgb,f.images.white);
  for(const [name,relative]of Object.entries(f.images)){const bytes=fs.readFileSync(path.join(folder,relative)),wrapper=fs.readFileSync(path.join(folder,relative+'.js'),'utf8'),m=wrapper.match(/data:image\/png;base64,([A-Za-z0-9+/=]+)/);assert(m);assert.equal(hash(bytes),hash(Buffer.from(m[1],'base64')));assert.equal(hash(bytes),f.png_sha256[name]);wrappers++;}
  if([raws[0],raws[Math.floor(raws.length/2)],raws.at(-1)].includes(raw)){a.S.raw=raw;a.draw3D();const removed=new Set(download.final_removed_ids),projection=a.S.projection;assert(projection.length>0);assert(projection.every(p=>p.cloud==='static'&&!removed.has(a.S.clouds.static.data.ids[p.j])));assert(a.geometryInstances().every(p=>p.cloud==='static'));
   const expected=sb.IDENTITY_AUDIT.frames[raw].vehicle_boxes.filter(b=>!b.target).length;assert.equal(a.currentBoxes().filter(b=>!b.target).length,expected);
   const im=await loadImage(path.join(folder,f.images.white));assert.equal(im.width,480);assert.equal(im.height,320);samples.push({raw,displayedStaticPoints:projection.length,removedOriginalRows:removed.size,otherVehicleBoxes:expected});}
 }
 // Run the real optional runtime with existing group state and actual metadata.
 a.S.raw=raws[Math.floor(raws.length/2)];vm.runInContext(fs.readFileSync(path.join(out,'scene_bounded_unknown.js'),'utf8'),sb);assert(!doc.getElementById('boundedUnknownFrame').hidden);assert.match(doc.getElementById('geometryStateInfo').textContent,/Actor omitted/);
 doc.getElementById('geometryState').value='baseline';a.draw3D();assert.match(doc.getElementById('geometryStateInfo').textContent,/原场景参照/);
 results.push({context,frames:raws.length,wrappersVerified:wrappers,allFrameGeometryRowsVerified:true,allActorObjectsEmpty:true,actualCanvasSamples:samples});
}
console.log(JSON.stringify({schema:'bounded-unknown-real-canvas-qa-v1',browserQA:false,sshUsed:false,realViewerCanvas:true,scenes:results}));
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
'''


def validate_links():
    class References(HTMLParser):
        def __init__(self, source):
            super().__init__(convert_charrefs=True)
            self.references = []
            self.feed(source)

        def handle_starttag(self, tag, attrs):
            self.references.extend(value for key, value in attrs if key in ('href', 'src') and value)

    local_references, bounded_links = 0, []
    for page in [OUT / 'index.html', *(OUT / context / 'index.html' for context in CONTEXTS)]:
        for relative in References(page.read_text(encoding='utf8')).references:
            parts = urlsplit(relative)
            if parts.scheme or parts.netloc:
                continue
            file = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            check(file.exists(), 'broken local HTML reference: ' + str(page) + ' -> ' + relative)
            local_references += 1
            if not parts.fragment.startswith('workspace?'):
                continue
            params = dict(parse_qsl(parts.fragment.split('?', 1)[1]))
            if params.get('group') != GID:
                continue
            manifest = read_json(file.parent / 'bounded_unknown_manifest.json')
            check(params.get('raw') in manifest['frames'] and params.get('channel') == 'rgb', 'invalid bounded group/raw/channel link')
            bounded_links.append(dict(page=str(page.relative_to(BASE)), context=file.parent.name, raw=int(params['raw'])))
    result = dict(schema='bounded-unknown-static-links-v1', local_references_checked=local_references, bounded_group_links=bounded_links, report_link_exists=(OUT / REPORT_RELATIVE).resolve().is_file(), browserQA=False)
    (WORK / 'ui_link_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return dict(local_references_checked=local_references, bounded_group_links=len(bounded_links), report_link_exists=result['report_link_exists'])


def validate_installed(deliveries):
    references = 0
    for delivery in deliveries:
        scene = OUT / delivery['context']
        manifest = read_json(scene / 'bounded_unknown_manifest.json')
        check(len(manifest['frames']) == 49, 'installed frame count mismatch')
        for raw, frame in manifest['frames'].items():
            original = delivery['frames'][raw]
            for name, relative in frame['images'].items():
                check(sha(scene / relative) == original['png_sha256'][name], 'installed PNG mismatch')
                references += 1
            check(sha(scene / frame['evidence_path']) == original['evidence_sha256'], 'installed NPZ mismatch')
            check(sha(scene / frame['receipt_path']) == original['receipt_sha256'], 'installed receipt mismatch')
            selected = read_json(scene / frame['row_ids_path'])
            check(selected['final_removed_ids'] == original['final_removed_ids'], 'installed row-ID download mismatch')
        page = (scene / 'index.html').read_text(encoding='utf8')
        data_at = page.index('bounded_unknown_data.js')
        viewer_at = page.index('../viewer.js')
        check(data_at < viewer_at, 'group must register before viewer captures groups')
        check(page.count('<!-- bounded-unknown-panel-start -->') == page.count('<!-- bounded-unknown-runtime-start -->') == 1, 'duplicate/missing scene injection')
        check('group=' + GID + '&amp;raw=' in page, 'scene direct group link missing')
    shared = (OUT / 'index.html').read_text(encoding='utf8')
    check(shared.count('<!-- bounded-unknown-card-start -->') == 3, 'shared scene-card links missing')
    for ctx in CONTEXTS:
        check(ctx + '/index.html#workspace?group=' + GID in shared, 'shared direct group link missing')
    code = CANVAS_QA.replace('__BASE__', js(str(BASE))).replace('__CONTEXTS__', js(CONTEXTS))
    result = subprocess.run(['node', '-'], input=code, text=True, encoding='utf8', cwd=BASE, capture_output=True, check=True)
    canvas = json.loads(result.stdout)
    (WORK / 'ui_canvas_validation.json').write_text(json.dumps(canvas, ensure_ascii=False, indent=2), encoding='utf8')
    return dict(installed_png_hashes_checked=references, direct_scene_links=3, card_links=3, real_canvas=True, links=validate_links(), browserQA=False, ssh_used=False)


def main():
    global BASE, OUT, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, default=BASE, help='project root containing outputs/ and work/')
    parser.add_argument('--input', type=Path, help='delivery root (relative paths use workspace root); default is work/v43_temporal_multiscene_20261029/temporal_multiscene_20261029_r1')
    parser.add_argument('--install', action='store_true', help='write outputs after complete 147-frame validation')
    parser.add_argument('--validate-installed', action='store_true', help='verify actual installed PNGs/rows/links with local native Canvas')
    parser.add_argument('--light', action='store_true', help='use ui_evidence.npz; full-float equations are verified by the remote complete audit')
    parser.add_argument('--remote-audit', type=Path, help='complete remote audit_sequence JSON covering all 147 source_body_contact frames')
    args = parser.parse_args()
    BASE = args.workspace_root.resolve()
    OUT = BASE / 'outputs/GS_Scene_Diagnostics_20261027'
    WORK = BASE / 'work/v43_temporal_multiscene_20261029'
    input_root = ((args.input if args.input.is_absolute() else BASE / args.input).resolve() if args.input else WORK / 'temporal_multiscene_20261029_r1')
    remote_audit = ((args.remote_audit if args.remote_audit.is_absolute() else BASE / args.remote_audit).resolve() if args.remote_audit else None)
    check(not args.light or args.remote_audit is not None, '--light requires a remote complete full-float audit')
    deliveries = audit_inputs(input_root, light=args.light, remote_audit=remote_audit)
    summary = dict(schema='bounded-unknown-input-audit-v1', total_frames=147, local_full_float_reloaded=not args.light, remote_full_audit=str(args.remote_audit) if args.light else None, scenes=[dict(context=d['context'], frames=len(d['frames']), min_removed_rows=min(f['removed_rows'] for f in d['frames'].values()), max_removed_rows=max(f['removed_rows'] for f in d['frames'].values()), original_static_total=d['static_total']) for d in deliveries], browserQA=False, ssh_used=False)
    if args.install:
        summary['installation'] = install(deliveries, input_root, remote_audit=remote_audit if args.light else None)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / 'ui_install_validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    if args.validate_installed:
        WORK.mkdir(parents=True, exist_ok=True)
        summary['installed_validation'] = validate_installed(deliveries)
        (WORK / 'ui_installed_validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
