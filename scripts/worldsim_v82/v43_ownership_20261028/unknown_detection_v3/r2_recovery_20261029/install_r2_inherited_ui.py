"""Install the completed, independently audited inherited R2 run into scene 1.

Read-only validation is the default. --install is required for publication.
Missing frames, missing gates, wrong namespaces, or mismatched evidence abort
before any public asset is written. This script never runs a GPU experiment.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

LOCAL_ROOT = Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a')
DEFAULT_ROOT = LOCAL_ROOT if (LOCAL_ROOT / 'outputs').is_dir() else Path.cwd()
WORK = Path(__file__).resolve().parent
C1 = '14705303724557273004_3105_000_3125_000'
GID = 'v43_r2_inherited49'
VERSION = 'r2-inherited49-20261029-1'
RAWS = list(range(77, 126))
ANCHORS = [98, 113, 114]
PNG_NAMES = ('white', 'existing', 'original', 'gaussian', 'native', 'native_env', 'ungated_white', 'alpha', 'observed')
GATES = ('anchor_replay_validation', 'contact_anchor_validation', 'seed_anchor_validation', 'seed_full49_validation', 'extend_anchor_validation')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def assignment(path):
    source = path.read_text(encoding='utf8')
    return json.JSONDecoder().raw_decode(source[source.index('=') + 1:])[0]


def scalar(arrays, key):
    return np.asarray(arrays[key]).item()


def id_array(arrays, key, count, where):
    a = np.asarray(arrays[key])
    check(a.ndim == 1 and a.dtype.kind in 'iu', where + ': invalid ' + key)
    a = a.astype(np.int64)
    check(not len(a) or (a[0] >= 0 and a[-1] < count and np.all(np.diff(a) > 0)), where + ': unordered/out-of-range ' + key)
    return a


def validate(root, inputs, audit_path):
    out = root / 'outputs/GS_Scene_Diagnostics_20261027'
    scene = out / C1
    scene_data = read(scene / 'scene.json')
    static = assignment(scene / 'cloud_full_manifest.js')['static']
    namespace = static['checkpoint'].rsplit('/', 1)[0] + '@30000:static'
    check(sorted(map(int, scene_data['frames'])) == RAWS, 'Unexpected scene 1 frame coverage')
    summary = read(inputs / 'summary.json')
    check(summary.get('status') == 'complete' and summary.get('frames') == 49 and summary.get('training_updates') == 0, 'R2 inherited49 is not complete')
    check(sorted(r['raw'] for r in summary['records']) == RAWS, 'Missing/duplicate summary frames')
    check(read(inputs / 'status.json').get('phase') == 'complete', 'Run status is not complete')
    gates = {}
    for name in GATES:
        gate = read(inputs.parent / (name + '.json'))
        check(gate.get('passed') is True, name + ' did not pass')
        if name == 'seed_full49_validation':
            check(not gate.get('missing_raws'), 'Native seed coverage incomplete')
        else:
            check(sorted(x['raw'] for x in gate['records']) == ANCHORS, name + ': all three anchors required')
            check(all(x.get('passed', True) for x in gate['records']), name + ': a record failed')
        gates[name] = gate
    remote = read(audit_path)
    check(remote.get('passed') is True, 'Independent full-float audit must pass')
    remote_frames = {int(x['raw']): x for x in remote['frames']}
    check(sorted(remote_frames) == RAWS, 'Independent audit must cover all 49 frames')
    result = {}
    for ordinal, raw in enumerate(RAWS):
        where = 'raw' + str(raw)
        folder = inputs / f'{raw:03d}'
        receipt = read(folder / 'receipt.json')
        required = dict(context=C1, raw=raw, camera_index=ordinal, status='ok', method='inherited_R2', init_mode='same_frame_existing', row_namespace=namespace,
                        independent_frame_reset=True, final_after_recomputed=True, after_attribution_recomputed=True, actual_rasterizer_rerendered=True,
                        training_updates=0, labels_for_training=False, new_outside_center_rows=0, outside_output_max_difference=0., converged=True)
        for key, expected in required.items():
            check(receipt.get(key) == expected, where + ': receipt ' + key + ' mismatch')
        source = scene_data['frames'][str(raw)]
        check(str(receipt['frame_id']) == str(source['frame_id']), where + ': source frame ID mismatch')
        check(abs(receipt['model_time'] - source['model_time']) < 1e-7, where + ': model time mismatch')
        check(receipt['static_rows'] == static['total'], where + ': static population mismatch')
        check(receipt.get('human_verdict') is None, where + ': no human verdict may be fabricated')
        after = receipt['final_after_summary']
        check(after['residual_eligible_rows'] == 0 and np.isfinite(after['closure']) and abs(after['closure']) <= 1e-4, where + ': after-attribution closure failed')
        if raw in ANCHORS:
            check(receipt.get('anchor_ids_identical') is True and receipt.get('anchor_white_max_error', 1) <= 1e-5, where + ': anchor regression')
        report = remote_frames[raw]
        check(report.get('passed') is True, where + ': remote full audit did not verify frame')
        check(report.get('checks', {}).get('frame_identity') is True and all(report['checks'].values()), where + ': independent full-float checks failed')
        check(receipt == next(x for x in summary['records'] if x['raw'] == raw), where + ': frame receipt differs from complete summary')
        evidence = folder / ('ui_evidence.npz' if (folder / 'ui_evidence.npz').is_file() else 'evidence.npz')
        check(report['packaged_evidence_sha256'] == sha(evidence), where + ': compact evidence hash mismatch')
        with np.load(evidence, allow_pickle=False) as arrays:
            for key in ('context', 'raw', 'row_namespace', 'checkpoint_ref', 'camera_index'):
                check(scalar(arrays, key) == receipt[key], where + ': evidence ' + key + ' mismatch')
            existing = id_array(arrays, 'existing_removed_ids', static['total'], where)
            new = id_array(arrays, 'new_unknown_ids', static['total'], where)
            final = id_array(arrays, 'final_removed_ids', static['total'], where)
            eligible = id_array(arrays, 'eligible_ids', static['total'], where)
            check(np.array_equal(final, np.union1d(existing, new)), where + ': final IDs do not equal same-frame inherited seeds plus new unknown')
            check(len(existing) > 0 and len(existing) == receipt['existing_rows'], where + ': missing inherited seeds')
            check(len(new) == receipt['new_rows'] and len(final) == receipt['final_removed_rows'], where + ': row count mismatch')
            check(np.isin(new, eligible).all(), where + ': newly selected rows outside bounded eligibility')
            masks = {}
            for key in ('allowed', 'query', 'display_scope'):
                mask = np.asarray(arrays[key])
                check(mask.shape == (320, 480) and np.isin(mask, [0, 1]).all(), where + ': invalid ' + key)
                masks[key] = mask.astype(bool)
            check(not (masks['display_scope'] & ~masks['allowed']).any(), where + ': display scope exceeds allowed region')
            check(not (masks['query'] & ~masks['allowed']).any(), where + ': query exceeds allowed region')
            check(int(masks['query'].sum()) == receipt['query_pixels'] and int(masks['display_scope'].sum()) == receipt['display_pixels'], where + ': mask count mismatch')
        png_hashes = {}
        for name in PNG_NAMES:
            file = folder / (name + '.png')
            check(file.is_file(), where + ': missing PNG ' + name)
            with Image.open(file) as image:
                check(image.size == (480, 320), where + ': invalid PNG dimensions ' + name)
            png_hashes[name] = sha(file)
            check(report['PNG_sha256'][name] == png_hashes[name], where + ': image hash mismatch ' + name)
        white = np.asarray(Image.open(folder / 'white.png').convert('RGB'))
        existing_png = np.asarray(Image.open(folder / 'existing.png').convert('RGB'))
        check(np.array_equal(white[~masks['display_scope']], existing_png[~masks['display_scope']]), where + ': outside PNG preservation failed')
        result[str(raw)] = dict(receipt=receipt, final_removed_ids=final.tolist(), existing_rows=len(existing), new_rows=len(new), png_sha256=png_hashes,
                                receipt_sha256=sha(folder / 'receipt.json'), evidence_name=evidence.name, evidence_sha256=sha(evidence), full_float_audit=report)
    return dict(schema='inherited-r2-ui-input-audit-v1', passed=True, context=C1, raws=RAWS, gates=gates,
                remote_audit_sha256=sha(audit_path), frames=result, training_updates=0)


def write(path, text, root):
    if path.is_file() and path.read_text(encoding='utf8') == text:
        return
    if path.is_file():
        backup = WORK / 'before_inherited49' / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf8')


def block(source, name, content, anchor, before=False):
    text = f'<!-- {name}-start -->{content}<!-- {name}-end -->'
    pattern = rf'<!-- {re.escape(name)}-start -->.*?<!-- {re.escape(name)}-end -->'
    if re.search(pattern, source, re.S):
        return re.sub(pattern, lambda _: text, source, flags=re.S)
    check(anchor in source, 'HTML insertion point missing: ' + anchor)
    return source.replace(anchor, text + anchor if before else anchor + text, 1)


RUNTIME = r'''(function(root){
  'use strict';const M=root.R2_INHERITED_DELIVERY,a=root.SceneViewer;if(!M||!a||M.status!=='complete'||!M.validated)return;
  const get=id=>document.getElementById(id),panel=get('r2InheritedPair'),names=['observed','existing','white'];
  let lastKey='';
  function update(){const active=S.group===M.group_id;panel.hidden=!active;if(!active)return;
    const f=M.frames[String(S.raw)];if(!f){get('r2InheritedFrameStatus').textContent='本帧没有R2交付，不替换为邻帧。';return;}
    const key=S.group+':'+S.raw;if(lastKey!==key){lastKey=key;for(const name of names){get('r2Pair_'+name).src=f.images[name];get('r2PairLink_'+name).href=f.images[name];}}
    get('r2InheritedFrameStatus').textContent='raw'+S.raw+' · '+(S.raw-76)+'/49 · 同帧existing '+f.existing_rows.toLocaleString()+'行＋新增unknown '+f.new_rows.toLocaleString()+'行；右侧3D真实移除最终 '+f.removed_rows.toLocaleString()+'行，Actor为空。';
    get('r2FrameReceipt').href=f.receipt;get('r2FrameEvidence').href=f.evidence;
    get('geometryStateInfo').textContent='R2同帧原static删行几何：继承原existing集合＋逐轮after贡献追加；蓝框外禁止新增unknown。二维局部G_after＋(1−alpha_after)白底、范围外保留existing；三维删行影响不限于像素窗口。';
    get('semanticFilterStatus').textContent='当前显示真实R2剥离后的几何。删除候选是诊断集合，并未写成可靠语义或训练标签；旧GPU语义证据不等于本轮unknown集合。';
  }
  const oldShow=show;show=async function(...args){const result=oldShow(...args);update();await result;update();};
  const oldDraw=draw3D;draw3D=function(){oldDraw();update();};Object.assign(a,{show,draw3D});root.R2Inherited={manifest:M,update};
  const params=new URLSearchParams(location.hash.replace(/^#workspace[?&]?/,''));
  if(!params.has('group')){S.raw=113;S.channel='rgb';S.cameraView=true;S.cameraReference='current';get('geometryState').value='current';get('semanticAblation').value='all';get('cloudColor').value='branch';setCategory('v43',M.group_id);show();}else update();
})(window);
'''


def install(root, inputs, audit_path, audit):
    out = root / 'outputs/GS_Scene_Diagnostics_20261027'
    scene = out / C1
    r2 = root / 'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
    protected = {p: sha(p) for p in [r2/'index.html', r2/'native_data.js', *(r2/f'native/{raw:03d}/white.png' for raw in ANCHORS)]}
    source_geometry = read(scene / 'v43_groups_data.json')['geometry']['v43_actor_10000']
    frames, geometry, manifest_frames = {}, {}, {}
    for raw in RAWS:
        key = str(raw)
        f = audit['frames'][key]
        folder = inputs / f'{raw:03d}'
        dest = scene / 'assets' / GID / f'{raw:03d}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in (*[n+'.png' for n in PNG_NAMES], 'receipt.json', f['evidence_name']):
            shutil.copy2(folder / name, dest / name)
        images = {name: f'assets/{GID}/{raw:03d}/{name}.png' for name in PNG_NAMES}
        image = dest / 'white.png'
        write(image.with_name('white.png.js'), 'window.IMAGE_DATA['+dump(images['white'])+']='+dump('data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode())+';\n', root)
        receipt_path = f'assets/{GID}/{raw:03d}/receipt.json'
        frames[key] = dict(channels={'rgb': images['white']}, receipt=receipt_path)
        q = copy.deepcopy(source_geometry[key])
        target = next(o for o in q['objects'] if o.get('key') == 'target')
        q.update(objects=[], static_visible=True, static_suppressed_rows=f['final_removed_ids'])
        for name in ('static_adapter', 'semantic_render_filter', 'target_detection'):
            q.pop(name, None)
        q['reference_boxes'] = [dict(target=True, label='R2原目标参考框', corners_local=target['corners_local'])]
        q['note'] = '真实继承R2；同帧原static最终删行集合与二维PNG对应。二维局部白底，域外existing。'
        q['r2_inherited'] = dict(source_raw=raw, row_namespace=f['receipt']['row_namespace'], removed_rows=len(f['final_removed_ids']), receipt=receipt_path)
        geometry[key] = q
        manifest_frames[key] = dict(images=images, receipt=receipt_path, evidence=f'assets/{GID}/{raw:03d}/'+f['evidence_name'],
                                    existing_rows=f['existing_rows'], new_rows=f['new_rows'], removed_rows=len(f['final_removed_ids']),
                                    png_sha256=f['png_sha256'], receipt_sha256=f['receipt_sha256'], evidence_sha256=f['evidence_sha256'])
    manifest = dict(schema='inherited-r2-viewer-delivery-v1', context=C1, group_id=GID, status='complete', validated=True, raws=RAWS,
                    frames=manifest_frames, training_updates=0, anchor_regression_passed=True, full49_float_audit_passed=True,
                    source_audit_sha256=audit['remote_audit_sha256'], human_visual_verdict=None)
    write(scene/'r2_inherited_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2), root)
    write(scene/'r2_inherited_input_audit.json', json.dumps({k:v for k,v in audit.items() if k!='frames'}, ensure_ascii=False, indent=2), root)
    shutil.copy2(audit_path, scene/'r2_inherited_remote_audit.json')
    write(scene/'r2_inherited_data.js', 'window.R2_INHERITED_DELIVERY='+dump(manifest)+';\nwindow.SCENE_DATA.groups=window.SCENE_DATA.groups.filter(g=>g.id!=='+dump(GID)+');\nwindow.SCENE_DATA.groups.push('+dump(dict(id=GID,title='真实R2继承 · 全49帧（逐帧复算）',category='v43',kind='r2_inherited',summary='r2_inherited_manifest.json',frames=frames))+');\nwindow.EDIT_GEOMETRY.groups['+dump(GID)+']='+dump(geometry)+';\n', root)
    write(out/'scene_r2_inherited.js', RUNTIME, root)
    # The standard viewer always links this conventional execution record.
    write(scene/'evidence'/f'{GID}_metrics.json', json.dumps(dict(status='complete',frames=49,raws=RAWS,training_updates=0,rgb_score_not_evaluation=True,manifest='../r2_inherited_manifest.json'), ensure_ascii=False,indent=2),root)
    apply_html(root)
    check(all(sha(p)==h for p,h in protected.items()), 'Original accepted R2 source changed')
    node = shutil.which('node') or ('C:/Program Files/nodejs/node.exe' if Path('C:/Program Files/nodejs/node.exe').is_file() else None)
    if node:
        for path in (scene/'r2_inherited_data.js',out/'scene_r2_inherited.js'):
            subprocess.run([node,'--check',str(path)],check=True)
    return manifest


def apply_html(root):
    out = root / 'outputs/GS_Scene_Diagnostics_20261027';scene = out / C1
    scene_html = (scene/'index.html').read_text(encoding='utf8')
    pair = '<section id="r2InheritedPair" class="notice" hidden><strong>真实R2 · 同一滑块 / 同一原始帧</strong><p id="r2InheritedFrameStatus"></p><div style="display:flex;gap:12px;flex-wrap:wrap">'+''.join('<a id="r2PairLink_'+name+'" target="_blank" style="flex:1;min-width:220px;max-width:360px"><img id="r2Pair_'+name+'" alt="'+label+'" style="display:block;width:100%"><span>'+label+'</span></a>' for name,label in [('observed','同帧实拍参照'),('existing','同帧原existing基线'),('white','真实R2局部白底')])+'</div><p>白底是已选集合剥离后的局部空洞诊断，不是补景。<a id="r2FrameReceipt">本帧原生凭据</a> · <a id="r2FrameEvidence">本帧完整行ID与范围</a> · <a href="r2_inherited_remote_audit.json">49帧独立浮点审核</a></p></section>'
    scene_html = block(scene_html,'r2-inherited-pair',pair,'<div class="workspace">',before=True)
    scene_html = block(scene_html,'r2-inherited-data',f'<script src="r2_inherited_data.js?v={VERSION}"></script>','<script src="../viewer.js',before=True)
    scene_html = block(scene_html,'r2-inherited-runtime',f'<script src="../scene_r2_inherited.js?v={VERSION}"></script>','</body>',before=True)
    banner = '<section class="notice" id="r2Baseline"><strong>当前：真实R2继承，全49帧已实际复算</strong><p><a class="button" href="#workspace?group='+GID+'&amp;raw=113&amp;channel=rgb">进入R2全49帧 RGB ↔ 3D</a>　'+' '.join(f'<a href="#workspace?group=v43_r2_reference&amp;raw={raw}&amp;channel=rgb">旧R2参考 raw{raw}</a>' for raw in ANCHORS)+'　<a href="../../V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html">原三帧报告</a></p><p>原R2三锚点回放与继承一致性通过；本组raw77—125每帧都从本帧existing集合开始，没有邻帧替代。全49帧数值审核通过，逐帧视觉质量仍需验收。旧空初始化49帧为非R2消融、未采纳。</p></section>'
    scene_html = block(scene_html,'r2-baseline',banner,'<section id="v43Focus"',before=True)
    scene_html = scene_html.replace('车底与阴影</strong> · 按用户要求暂缓','车底与阴影</strong> · 已纳入R2诊断').replace('本轮不优化车底。','该段为旧阶段记录；当前已纳入R2逐帧剥离与贡献来源诊断。')
    write(scene/'index.html',scene_html,root)
    landing = (out/'index.html').read_text(encoding='utf8')
    landing_banner = '<section class="notice"><strong>场景1：真实R2继承49帧已交付</strong><p><a href="'+C1+'/index.html#workspace?group='+GID+'&amp;raw=113&amp;channel=rgb">打开同帧RGB与3D</a> · <a href="'+C1+'/r2_inherited_manifest.json">49帧清单与凭据</a></p><p>原R2三锚点保持一致；本轮为场景1复算，场景2/3未宣称已完成R2迁移。旧147帧空初始化消融保留为未采纳历史。</p></section>'
    landing = block(landing,'r2-baseline',landing_banner,'<main id="workspace">')
    write(out/'index.html',landing,root)


def can_reapply(root):
    """Read-only gate for rebuild hooks; never claim a missing delivery exists."""
    root=Path(root);scene=root/'outputs/GS_Scene_Diagnostics_20261027'/C1
    try:
        m=read(scene/'r2_inherited_manifest.json')
        if not (m.get('status')=='complete' and m.get('validated') is True and m.get('raws')==RAWS and m.get('full49_float_audit_passed') is True):return False
        if sha(scene/'r2_inherited_remote_audit.json')!=m['source_audit_sha256']:return False
        if not (scene/'r2_inherited_data.js').is_file():return False
        for frame in m['frames'].values():
            for name,rel in frame['images'].items():
                if sha(scene/rel)!=frame['png_sha256'][name]:return False
            if sha(scene/frame['receipt'])!=frame['receipt_sha256'] or sha(scene/frame['evidence'])!=frame['evidence_sha256']:return False
        return len(m['frames'])==49
    except (OSError,KeyError,ValueError):return False


def reapply_installed(root):
    root=Path(root).resolve()
    check(can_reapply(root),'Verified installed R2 assets are missing; cannot restore UI')
    write(root/'outputs/GS_Scene_Diagnostics_20261027/scene_r2_inherited.js',RUNTIME,root)
    apply_html(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--input-dir',type=Path)
    p.add_argument('--remote-audit',type=Path)
    p.add_argument('--install',action='store_true')
    a=p.parse_args();root=a.workspace_root.resolve()
    inputs=a.input_dir or root/'work/v43_r2_regression_20261029/r2_delivery/r2_inherited49'
    remote=a.remote_audit or inputs.parent/'local_audit.json'
    audit=validate(root,inputs,remote)
    if a.install:
        manifest=install(root,inputs,remote,audit)
        print(json.dumps(dict(installed=True,group=GID,frames=len(manifest['raws']),full_float_audit=True),ensure_ascii=False))
    else:
        print(json.dumps(dict(installed=False,validated=True,frames=len(audit['frames']),next='--install'),ensure_ascii=False))


if __name__=='__main__':
    main()
