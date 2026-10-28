"""只按本地真实文件修复共享入口与case2/3快捷入口；不训练、不SSH、不改case1。

可在collect_actor.py / collect_actor_motion.py接入后重跑；pending会随真实交付自动更新。
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs/GS_Scene_Diagnostics_20261027'
CONTEXTS = ('14705303724557273004_3105_000_3125_000', '4295449061847708198_3769_000_3789_000', '473735159277431842_630_095_650_095')
PLANS = [('v43_actor_10000', 'V43 10K结构', 49), ('v43_motion_no_edit', 'V43无编辑', 31), ('v43_motion_delete_target', 'V43删目标', 31), ('v43_motion_translate_lateral_1m', 'V43横移1米', 31), ('v43_motion_turn_left15', 'V43左转15°', 31)]
LEGACY = [('motion_no_edit', '原30K无编辑'), ('motion_delete_target', '原30K删目标'), ('motion_translate_lateral_1m', '原30K横移'), ('motion_turn_left15', '原30K转向')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def assignment(path, key):
    source = path.read_text(encoding='utf-8-sig')
    match = re.search(r'(?:window\.)?' + re.escape(key) + r'\s*=\s*', source)
    assert match, path
    return json.JSONDecoder().raw_decode(source[match.end():])[0]


def audit_scene(context, number):
    folder = OUT / context
    original = read(folder / 'scene.json')
    v43 = read(folder / 'v43_groups_data.json')
    groups = original['groups'] + v43.get('groups', [])
    if (folder / 'target_detection_data.js').exists():
        groups += assignment(folder / 'target_detection_data.js', 'TARGET_DETECTION_DELIVERY').get('groups', [])
    records = []
    for group in groups:
        counts, missing = {}, []
        complete = []
        for raw, frame in group.get('frames', {}).items():
            for channel, relative in frame.get('channels', {}).items():
                file = folder / relative
                counts[channel] = counts.get(channel, 0) + 1
                if not file.is_file(): missing.append(relative)
                elif str(relative).endswith('.png') and not (folder / (relative + '.js')).is_file(): missing.append(relative + '.js')
            rgb = frame.get('channels', {}).get('rgb')
            if rgb and (folder / rgb).is_file() and (folder / (rgb + '.js')).is_file(): complete.append(int(raw))
            if frame.get('maps') and not (folder / frame['maps']).is_file(): missing.append(frame['maps'])
        geometry = v43.get('geometry', {}).get(group['id'], {})
        summary = group.get('summary')
        if summary and not (folder / summary).is_file(): missing.append(summary)
        records.append(dict(id=group['id'], title=group['title'], category=group['category'], kind=group.get('kind'), declared_frames=len(group.get('frames', {})), actual_rgb_frames=len(complete), complete_raws=sorted(complete), channels=counts, missing_referenced_files=missing, geometry_frames=len(geometry), geometry_raws=sorted(map(int, geometry)), summary=summary))
    by_id = {group['id']: group for group in records}
    availability = []
    for gid, title, expected in PLANS:
        group = by_id.get(gid)
        count = group['actual_rgb_frames'] if group else 0
        ready = bool(group and count == expected and not group['missing_referenced_files'] and group['geometry_raws'] == group['complete_raws'] and v43.get('clouds', {}).get('v43:10000', {}).get('sample', 0) > 0)
        availability.append(dict(id=gid, title=title, expected_frames=expected, actual_rgb_frames=count, status='available' if ready else 'pending_delivery', complete_raws=group['complete_raws'] if group else []))
    stress = sorted(int(file.stem) for file in (folder / 'assets/v43/10000/turn15_stress').glob('*.png'))
    return dict(schema='scene-entry-availability-v1', scene_number=number, context=context, base_group_count=len(original['groups']), v43_group_count=len(v43.get('groups', [])), runtime_group_count=len(groups), groups=records, v43_delivery=availability, unregistered_turn_stress_raws=stress, execution_authorization='resumed_by_current_user' if number == 3 else 'authorized', old_hold_is_current=False if number == 3 else None, current_execution='local_delivery_available' if all(item['status'] == 'available' for item in availability) else 'awaiting_local_delivery_and_review', new_training_started_by_patch=False, browserQA=False), original, v43


def normalize_url(url):
    value = html.unescape(url)
    if '#workspace' not in value: return value
    prefix, fragment = value.split('#workspace', 1)
    if not fragment or fragment[0] not in '?&': return value
    pairs = parse_qsl(fragment.lstrip('?&').replace('?', '&'), keep_blank_values=True)
    unique = {}
    for key, item in pairs:
        if key not in unique: unique[key] = item
    return prefix + '#workspace?' + urlencode(unique)


def normalize_links(source):
    return re.sub(r'href="([^"]*)"', lambda match: 'href="' + html.escape(normalize_url(match[1]), quote=True) + '"', source)


def put_block(source, name, body, after):
    start, end = f'<!-- {name}-start -->', f'<!-- {name}-end -->'
    value = start + body + end
    if start in source: return re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: value, source, count=1, flags=re.S)
    assert after in source, name
    return source.replace(after, after + value, 1)


def preferred(status):
    entries = {entry['id']: entry for entry in status['v43_delivery']}
    for gid in ('v43_motion_no_edit', 'v43_actor_10000'):
        entry = entries[gid]
        if entry['status'] == 'available':
            target = 181 if status['scene_number'] == 2 else 45
            return gid, target if target in entry['complete_raws'] else entry['complete_raws'][len(entry['complete_raws'])//2]
    return 'reconstruction_30000', 181 if status['scene_number'] == 2 else 45


def group_link(gid, raw, title, prefix=''):
    return '<a class="button" style="display:inline-block;padding:6px 10px;border:1px solid #3c6286;border-radius:5px;background:#152f48" href="' + prefix + '#workspace?group=' + html.escape(gid) + '&amp;raw=' + str(raw) + '&amp;channel=rgb">' + html.escape(title) + '</a>'


def panel(status):
    entries = []
    for entry in status['v43_delivery']:
        if entry['status'] == 'available':
            target = 181 if status['scene_number'] == 2 else 45
            raw = target if target in entry['complete_raws'] else entry['complete_raws'][len(entry['complete_raws'])//2]
            entries.append(group_link(entry['id'], raw, entry['title'] + ' · ' + str(entry['actual_rgb_frames']) + '帧'))
        else:
            entries.append('<span class="badge">' + entry['title'] + '：待交付接入</span>')
    ready = status['current_execution'] == 'local_delivery_available'
    title = 'V43实际结果：结构与四组编辑' if ready else 'V43入口：按真实已到文件显示'
    state = '本地原生PNG、离线包装与同帧几何均已核对；可直接切换。' if ready else '完整序列已进入取回接入队列；缺项不以旧RGB或邻帧替代。'
    authorization = '第三场景已恢复授权，旧停止记录不再作为当前执行状态。' if status['scene_number'] == 3 else ''
    raw = preferred(status)[1]
    legacy = ' '.join(group_link(gid, raw, text) for gid, text in LEGACY)
    limitations = '这些V43编辑仍带原静态残影，尚未应用新的跨场景unknown诊断；该诊断将单独接入。默认显示原生RGB；generated通道来自既有ProPainter输出。'
    return '<section class="panel notice" id="v43SceneEntry"><h2>' + title + '</h2><p>' + authorization + state + '</p><div class="subtools">' + ' '.join(entries) + '</div><p>' + limitations + '</p><details><summary>原30K编辑（独立历史基线）</summary><div class="subtools">' + legacy + '</div></details><p class="note">分组只链接本场景实际资产；V43新Actor与原30K编辑分别记录。<a href="v43_entry_status.json">本地可用清单 →</a></p></section>'


def patch_case(status):
    folder = OUT / status['context']; file = folder / 'index.html'
    source = normalize_links(file.read_text(encoding='utf8'))
    source = put_block(source, 'scene-v43-entry', panel(status), '<main>')
    # 无hash直开时显式显示真实V43类别；带诊断hash的请求一律保留。
    default = '''<script>(function(){const p=new URLSearchParams(location.hash.replace(/^#workspace[?&]?/,''));if(p.has('group'))return;const a=window.SceneViewer;if(!a)return;const ids=['v43_motion_no_edit','v43_actor_10000'];const g=ids.map(id=>a.D.groups.find(x=>x.id===id&&Object.values(x.frames||{}).some(f=>f.channels?.rgb))).find(Boolean);if(!g)return;const raws=Object.keys(g.frames).map(Number).sort((x,y)=>x-y);const target=TARGET_RAW;a.S.raw=raws.includes(target)?target:raws[Math.floor(raws.length/2)];a.S.channel='rgb';a.selectGroup(g.id);})();</script>'''.replace('TARGET_RAW', str(preferred(status)[1]))
    start, end = '<!-- scene-v43-default-start -->', '<!-- scene-v43-default-end -->'
    if start in source: source = re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: start + default + end, source, flags=re.S, count=1)
    else: source = source.replace('</body>', start + default + end + '</body>', 1)
    file.write_text(source, encoding='utf8')
    (folder / 'v43_entry_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')


def patch_shared(statuses, originals, deliveries):
    file = OUT / 'index.html'; source = normalize_links(file.read_text(encoding='utf8'))
    for status in statuses:
        if status['scene_number'] not in (2, 3): continue
        ctx = status['context']; gid, raw = preferred(status)
        direct = ctx + '/index.html#workspace?group=' + gid + '&raw=' + str(raw) + '&channel=rgb'
        complete = status['current_execution'] == 'local_delivery_available'
        label = 'V43 10K与四组实际编辑已接入' if complete else ('V43 10K结构可看 · 完整编辑待接入' if status['scene_number'] == 2 else '授权恢复 · V43结果待接入')
        def replace_card(match):
            card = match[0]
            if ctx not in card: return card
            card = re.sub(r'<h2>.*?</h2>', '<h2>场景 ' + str(status['scene_number']) + ' · ' + label + '</h2>', card, count=1, flags=re.S)
            card = re.sub(r'href="' + re.escape(ctx) + r'/index\.html[^"]*"', lambda _: 'href="' + html.escape(direct, quote=True) + '"', card)
            groups = {g['id']: g for g in originals[ctx]['groups'] + deliveries[ctx].get('groups', [])}
            rgb = groups[gid]['frames'][str(raw)]['channels']['rgb']
            image = '<img src="' + ctx + '/' + rgb + '" alt="场景' + str(status['scene_number']) + '实际' + ('V43无编辑' if gid.startswith('v43_motion_') else 'V43 10K结构' if gid.startswith('v43_actor_') else '原30K') + ' raw' + str(raw) + '">'
            card = re.sub(r'<img\b[^>]*>', lambda _: image, card, count=1)
            card = card.replace('完整图重建 PSNR', '原30K完整图重建 PSNR') if '原30K完整图重建 PSNR' not in card else card
            note = '<p class="note">' + ('V43结构49帧；无编辑、删除、横移、转向各31帧。' if complete else '按本地交付清单显示，缺项仍待接入。') + '</p>'
            card = re.sub(r'<!-- scene-v43-card-status-start -->.*?<!-- scene-v43-card-status-end -->', '', card, flags=re.S)
            return card.replace('</article>', '<!-- scene-v43-card-status-start -->' + note + '<!-- scene-v43-card-status-end --></article>')
        source = re.sub(r'<article\b[^>]*>.*?</article>', replace_card, source, flags=re.S)
    state_links = ' '.join(group_link(*preferred(item), '场景' + str(item['scene_number']) + '实际V43入口', item['context'] + '/index.html') for item in statuses if item['scene_number'] in (2, 3))
    source = put_block(source, 'scene-v43-status', '<section class="notice"><strong>第二、第三场景V43接入：</strong>' + state_links + '。旧停止范围已由当前授权恢复；入口只展示真实已到产物。当前编辑仍带原静态残影，新的跨场景unknown诊断尚未应用，将单独接入；默认原生RGB。</section>', '<main id="workspace">')
    source = source.replace('<h2>V43 阶段收尾：', '<h2>历史阶段收尾：').replace('新实验已暂停；三候选仍有残影。', '该历史阶段当时暂停；三候选仍有残影。')
    file.write_text(source, encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    statuses, originals, deliveries = [], {}, {}
    for number, ctx in enumerate(CONTEXTS, 1):
        status, original, delivery = audit_scene(ctx, number)
        statuses.append(status); originals[ctx] = original; deliveries[ctx] = delivery
    assert all(not group['missing_referenced_files'] for status in statuses for group in status['groups'])
    if not args.audit_only:
        protected = [OUT / CONTEXTS[0] / 'index.html', BASE / 'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html']
        before = {str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        for status in statuses:
            if status['scene_number'] in (2, 3): patch_case(status)
        patch_shared(statuses, originals, deliveries)
        assert before == {str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        report = dict(schema='shared-scene-entry-audit-v1', scenes=statuses, case1_files_written=False, bounded_report_written=False, protected_sha256=before, browserQA=False, ssh_used=False)
        (OUT / 'scene_entry_audit_20261029.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps([dict(scene=s['scene_number'], base_groups=s['base_group_count'], v43_groups=s['v43_group_count'], runtime_groups=s['runtime_group_count'], delivery=[{k: v for k, v in entry.items() if k != 'complete_raws'} for entry in s['v43_delivery']], execution=s['current_execution']) for s in statuses], ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
