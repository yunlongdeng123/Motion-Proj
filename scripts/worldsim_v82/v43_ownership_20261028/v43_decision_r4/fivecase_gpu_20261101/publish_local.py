from pathlib import Path
import json, shutil, hashlib

W = Path(__file__).resolve().parent
OUT = W.parent.parent / 'outputs/V43_R4_FiveCase_GPU_20261101'
ROOT = Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027')
TARGETS = [ROOT / 'actor_candidates', OUT.parent / 'V43_R4_GPU_20261031/actor_candidates']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root_sha = sha(ROOT / 'index.html')
    data_path = OUT / 'review/data.js'
    data = json.loads(data_path.read_text(encoding='utf-8')[len('window.FIVE_DATA='):-1])
    for c in data:
        folder = OUT / 'evidence' / c['case']
        m = json.loads((folder / 'parent_candidate_manifest.json').read_text(encoding='utf-8'))
        context = json.loads((folder / 'parent_camera_track_context.json').read_text(encoding='utf-8'))
        c['raw_context'] = m['scene_ancestor'].removeprefix('waymo_context_')
        c['raw_track'] = context['target_track']
    data_path.write_text('window.FIVE_DATA=' + json.dumps(data, ensure_ascii=False) + ';', encoding='utf-8')
    p = OUT / 'review/index.html'
    html = p.read_text(encoding='utf-8')
    html = html.replace("+' / '+c.actor", "+'（训练集） / '+c.actor")
    html = html.replace("current.scene_key+' / '+current.actor+'；3个场景中的5辆开发车辆，尚未作独立泛化检验。'", "current.scene_key+' / '+current.actor+'；原始路段 ID：'+current.raw_context+'；原始轨迹 ID：'+current.raw_track+'。3个场景中的5辆开发车辆，尚未作独立泛化检验。'")
    html = html.replace('完整视频覆盖这两条固定方案', '视频覆盖这两条固定方案')
    p.write_text(html, encoding='utf-8')
    script = '''(() => {
  'use strict';
  const cases = new Set(['waymo-train-008_a37','waymo-train-212_a0','waymo-train-212_a35','waymo-train-564_a0','waymo-train-564_a22']);
  const panel = document.createElement('section');
  panel.className = 'panel'; panel.id = 'editComparisonEntry'; panel.hidden = true;
  const heading = document.createElement('h2'); heading.textContent = '编辑对照';
  const info = document.createElement('p'); info.textContent = '查看同一相机下的无编辑、删除、横移和旋转；视频可显示原位置与请求位置框。';
  const link = document.createElement('a'); link.textContent = '打开这辆车的编辑对照';
  panel.append(heading, info, link);
  const media = document.getElementById('actorMediaPanel'); media.before(panel);
  function update() {
    const params = new URLSearchParams(location.hash.slice(1));
    const title = document.getElementById('sceneTitle').textContent;
    const actorTitle = document.getElementById('actorTitle').textContent;
    const scene = title.match(/scene-(\\d+)/);
    const actor = actorTitle.match(/\\ba\\d+\\b/);
    const key = (params.get('scene') || (scene ? 'waymo-train-' + scene[1].padStart(3, '0') : '')) + '_' + (params.get('actor') || (actor ? actor[0] : ''));
    panel.hidden = !cases.has(key);
    if (!panel.hidden) link.href = 'edit_comparisons/index.html#case=' + encodeURIComponent(key);
  }
  for (const id of ['sceneTitle','actorTitle']) new MutationObserver(update).observe(document.getElementById(id), {childList:true,subtree:true,characterData:true});
  for (const id of ['scene','actor']) document.getElementById(id).addEventListener('change', () => queueMicrotask(update));
  window.addEventListener('hashchange', update); update();
})();
'''
    (OUT / 'source/edit_comparison_entry.js').write_text(script, encoding='utf-8')
    receipts = []
    for i, target in enumerate(TARGETS):
        index = target / 'index.html'
        before = index.read_bytes()
        backup = W / f'candidate_index_before_{i}.html'
        if not backup.exists(): backup.write_bytes(before)
        dest = target / 'edit_comparisons'
        shutil.copytree(OUT / 'review', dest, dirs_exist_ok=True)
        v = dest / 'index.html'
        s = v.read_text(encoding='utf-8').replace('../../V43_R4_GPU_20261031/actor_candidates/index.html', '../index.html')
        v.write_text(s, encoding='utf-8')
        (target / 'edit_comparison_entry.js').write_text(script, encoding='utf-8')
        s = before.decode('utf-8')
        if 'src="edit_comparison_entry.js"' not in s:
            s = s.replace('</body>', '<script src="edit_comparison_entry.js"></script></body>')
        index.write_text(s, encoding='utf-8')
        assert s.count('src="edit_comparison_entry.js"') == 1
        receipts.append(dict(path=str(target), index_before=hashlib.sha256(before).hexdigest(), index_after=sha(index), visual_files=sum(f.is_file() for f in dest.rglob('*')), report_published=False))
    assert sha(ROOT / 'index.html') == root_sha
    (OUT / 'publishing_receipt.json').write_text(json.dumps(dict(shared_root_unchanged=True, shared_root_sha256=root_sha, entries=receipts, browser_interaction_tested=False), ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(__file__, OUT / 'source/publish_local.py')
    print('PUBLISHED', len(receipts), 'root unchanged')

if __name__ == '__main__': main()
