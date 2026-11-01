(() => {
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
    const scene = title.match(/scene-(\d+)/);
    const actor = actorTitle.match(/\ba\d+\b/);
    const key = (params.get('scene') || (scene ? 'waymo-train-' + scene[1].padStart(3, '0') : '')) + '_' + (params.get('actor') || (actor ? actor[0] : ''));
    panel.hidden = !cases.has(key);
    if (!panel.hidden) link.href = 'edit_comparisons/index.html#case=' + encodeURIComponent(key);
  }
  for (const id of ['sceneTitle','actorTitle']) new MutationObserver(update).observe(document.getElementById(id), {childList:true,subtree:true,characterData:true});
  for (const id of ['scene','actor']) document.getElementById(id).addEventListener('change', () => queueMicrotask(update));
  window.addEventListener('hashchange', update); update();
})();
