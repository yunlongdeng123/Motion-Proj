"""Publish saved RGB as an optional image-only candidate, preserving v44 default.

No model loading, rendering, optimization, or external service is involved.
The root's actual full49 visual release is a required input.
"""
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VIEWER = Path('C:/Users/dengyunlong/Documents/Codex/2026-10-20/file-c-users-dengyunlong-documents-codex-2/outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1')
ID = 'v44_full5_20261107'
LABEL = 'v44＋a5车顶恢复（候选，细节偏软）'
NOTE = '仅保存RGB的结构修复候选。21/25严格门通过，raw94四项光度FAIL保留；12/24可见帧RGB误差回退，非无损、非人工验收。旧v44仍默认，可切回；本组没有对应Gaussian或depth面板。'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate_path = ROOT/'evidence/full49_semantic_actor5_all49fit/root_visual_review.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    assert gate['candidate_report_and_optional_central_viewer_release'] is True
    assert gate['central_viewer_original_v44_default_must_remain'] is True
    index_path = VIEWER/'index.html'
    before = index_path.read_text(encoding='utf-8')
    backup = ROOT/'evidence/viewer_backup'
    backup.mkdir(exist_ok=True)
    backup_index = backup/'index.before_optional_candidate.html'
    if not backup_index.exists():
        shutil.copyfile(index_path, backup_index)
    protected = ['app.js', 'data/v44_experiment.js']
    protected_before = {p: sha(VIEWER/p) for p in protected}
    for p in protected:
        target = backup/Path(p).name
        if not target.exists():
            shutil.copyfile(VIEWER/p, target)
    all_old_assets = {p.relative_to(VIEWER).as_posix(): sha(p)
                      for p in (VIEWER/'assets/v44_20261107').rglob('*.png')}
    assert len(all_old_assets) >= 147

    copied = []
    image_paths = {key: {} for key in ('deleted_five', 'joint', 'gt')}
    for raw in range(89, 138):
        for channel, state, source in [
                ('deleted_five', 'delete', ROOT/f'evidence/full49_semantic_actor5_all49fit/frames/{raw:03d}/delete.png'),
                ('joint', 'keep', ROOT/f'evidence/full49_semantic_actor5_all49fit/frames/{raw:03d}/keep.png'),
                ('gt', 'observed', ROOT/f'evidence/v44_all49_reference/frames/{raw:03d}/observed.png')]:
            with Image.open(source) as im:
                assert im.size == (480, 320)
                im.verify()
            relative = f'assets/{ID}/{state}/{raw:03d}.png'
            dest = VIEWER/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert sha(dest) == sha(source), 'Never overwrite a different published candidate'
            else:
                shutil.copyfile(source, dest)
            assert sha(dest) == sha(source)
            copied.append(dict(raw=raw, state=state, path=relative,
                               source=source.relative_to(ROOT).as_posix(), sha256=sha(dest), bytes=dest.stat().st_size))
            image_paths[channel][str(raw)] = relative
    entry = dict(id=ID, name=LABEL, label=LABEL, available=True,
                 viewer_scope='image_only', channels=['deleted_five', 'joint', 'gt'],
                 image_channels=['deleted_five', 'joint', 'gt'], images=image_paths,
                 actual_delete_operation='delete_a37', delete_five=False,
                 channel_labels={'deleted_five': '仅删除a37（a5结构修复候选）', 'joint': '完整KEEP（候选）', 'gt': '真实观测 RGB'},
                 operations={'deleted_five': 'delete_a37', 'joint': 'keep_a37', 'gt': 'observed_a37_present'},
                 channel_notes={'deleted_five': NOTE, 'joint': NOTE,
                                'gt': '真实观测中a37仍在，不是删除后隐藏道路真值。'},
                 comparison_note=NOTE, raw_ids=list(range(89,138)),
                 original_v44_default_preserved=True, quality_accepted=False, human_verdict=None,
                 source_comparison_sha256='8201fa883df9ac43274d52984c750f92513118c1152628fa0518e69cf66042e3')
    js = "/* Optional RGB-only candidate; deliberately does not change defaults. */\n(function(){'use strict';const g=window.GSDATA;if(!g?.scene)throw Error('scene metadata missing');const entry="+json.dumps(entry, ensure_ascii=False, separators=(',', ':'))+";g.scene.experiments=(g.scene.experiments||[]).filter(e=>e.id!==entry.id).concat(entry);})();\n"
    js_relative = f'data/{ID}_experiment.js'
    (VIEWER/js_relative).write_text(js, encoding='utf-8')
    (ROOT/'source/published_optional_experiment.js').write_text(js, encoding='utf-8')

    marker = 'const groups='
    start = before.index(marker)+len(marker)
    groups, length = json.JSONDecoder().raw_decode(before[start:])
    original_groups = json.loads(json.dumps(groups))
    new_group = dict(id=ID, title=LABEL, files=image_paths['deleted_five'],
                     reference_title='正式v44 · 同帧仅删a37（原默认版本）',
                     reference_files={str(raw): f'assets/v44_20261107/v44/{raw:03d}.png' for raw in range(89,138)},
                     viewer_scope='image_only', note=NOTE)
    if not any(group.get('id') == ID for group in groups):
        groups.append(new_group)
    else:
        assert next(group for group in groups if group.get('id') == ID) == new_group
    for old in original_groups:
        if old.get('id') != ID:
            assert old in groups
    updated = before[:start]+json.dumps(groups, ensure_ascii=False, separators=(',', ':'))+before[start+length:]
    for select_id, value in [('arm', ID), ('fullclipGroup', str(next(i for i,g in enumerate(groups) if g.get('id') == ID)))]:
        pattern = r'(<select id="'+select_id+r'">)(.*?)(</select>)'
        match = re.search(pattern, updated, flags=re.S)
        assert match
        body = match.group(2)
        if f'value="{value}"' not in body:
            body += '<option value="'+value+'">'+LABEL+'</option>'
        updated = updated[:match.start(2)]+body+updated[match.end(2):]
    script_tag = '<script src="'+js_relative+'?v=semantic49candidate1"></script>'
    if script_tag not in updated:
        hook = '<script src="app.js?v=v44top1"></script>'
        assert updated.count(hook) == 1
        updated = updated.replace(hook, script_tag+hook)
    if 'id="fullclipVersionNote"' not in updated:
        hook = '<strong id="fullclipRaw">raw102 · 14/49</strong><button id="fullclipNext">→</button></div>'
        assert updated.count(hook) == 1
        updated = updated.replace(hook, hook+'<p id="fullclipVersionNote" class="note"></p>')
        draw_hook = "$('fullclipBaseTitle').textContent=g.reference_title||'同帧原静态层参照';"
        assert updated.count(draw_hook) == 1
        updated = updated.replace(draw_hook, draw_hook+"$('fullclipVersionNote').textContent=g.note||'本节只显示已保存RGB；其他面板仍属于其所选原实验。';")
    assert '<option value="v44_20261107" selected>' in updated
    assert '<option value="5" selected>' in updated
    assert not re.search(r'<option value="'+re.escape(ID)+r'"\s+selected', updated)
    assert 'defaultExperiment' not in js and 'defaultChannel' not in js
    assert all(sha(VIEWER/p)==v for p,v in protected_before.items())
    assert all(sha(VIEWER/p)==v for p,v in all_old_assets.items())
    index_path.write_text(updated, encoding='utf-8')
    manifest = dict(version=ID, label=LABEL, viewer_scope='image_only',
                    generated_utc=datetime.now(timezone.utc).isoformat(), images=copied,
                    default_experiment='v44_20261107', default_fullclip_index=5,
                    quality_accepted=False, human_verdict=None,
                    strict_gates='21/25; raw94 four FAIL retained', note=NOTE)
    (VIEWER/f'assets/{ID}/publication_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    receipt = dict(status='PASS_OPTIONAL_PUBLICATION_STATIC_AND_BYTE_QA_RUNTIME_QA_SEPARATE',
                   viewer=str(VIEWER), new_version=ID, label=LABEL,
                   authorization=gate_path.relative_to(ROOT).as_posix(), authorization_sha256=sha(gate_path),
                   source_index_backup=backup_index.relative_to(ROOT).as_posix(),
                   source_index_sha256=sha(backup_index), published_index_sha256=sha(index_path),
                   original_v44_default_preserved=True, original_fullclip_default_preserved=True,
                   old_v44_assets_unchanged=len(all_old_assets), protected_files_unchanged=protected_before,
                   candidate_image_files=len(copied), native_size=[480,320], candidate_images=copied,
                   new_script=js_relative, new_script_sha256=sha(VIEWER/js_relative),
                   image_only=True, geometry_or_depth_reused=False, quality_accepted=False, human_verdict=None,
                   publication_manifest=f'assets/{ID}/publication_manifest.json')
    (ROOT/'evidence/viewer_publication_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('candidate_images','protected_files_unchanged')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
