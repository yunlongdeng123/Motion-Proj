/* Actor-local real measured LiDAR. Canvas2D only; never renders Gaussian samples. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const panel = $('lidar3DPanel'), canvas = $('lidar3DCanvas');
  if (!panel || !canvas) return;
  const ctx = canvas.getContext('2d');
  let payload = null, points = [], dimensions = [4.5, 2, 1.8], actorKey = '', dragging = null, queued = false;
  let orbit = { yaw: .78, pitch: .38, distance: 9, panX: 0, panY: 0 };
  const finite3 = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  function fit() {
    orbit = { yaw: .78, pitch: .38, distance: Math.max(3, Math.hypot(...dimensions) * 1.75), panX: 0, panY: 0 };
    $('lidar3DView').value = 'orbit';
  }
  function project(p) {
    const [x, y, z] = p;
    const right = Math.sin(orbit.yaw) * x - Math.cos(orbit.yaw) * y;
    const forward = Math.cos(orbit.yaw) * x + Math.sin(orbit.yaw) * y;
    const depth = forward * Math.cos(orbit.pitch) - z * Math.sin(orbit.pitch) + orbit.distance;
    if (depth <= .05) return null;
    const down = -forward * Math.sin(orbit.pitch) - z * Math.cos(orbit.pitch);
    const f = canvas.width * .72;
    return [canvas.width / 2 + right * f / depth + orbit.panX,
      canvas.height / 2 + down * f / depth + orbit.panY, depth];
  }
  function line(a, b, color, width) {
    const p = project(a), q = project(b);
    if (!p || !q || !p.concat(q).every(Number.isFinite)) return;
    ctx.strokeStyle = color; ctx.lineWidth = width || 1;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
  }
  function draw() {
    queued = false;
    ctx.fillStyle = '#08131e'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!payload) return;
    const floor = -dimensions[2] / 2, extent = Math.ceil(Math.max(dimensions[0], dimensions[1]) / 2) + 1;
    if ($('lidar3DGrid').checked) {
      for (let v = -extent; v <= extent; v += .5) {
        line([v, -extent, floor], [v, extent, floor], '#183044');
        line([-extent, v, floor], [extent, v, floor], '#183044');
      }
    }
    if ($('lidar3DBox').checked) {
      const corners = Array.from({ length: 8 }, (_, i) => dimensions.map((d, k) => (i & (1 << k) ? 1 : -1) * d / 2));
      for (let i = 0; i < 8; i++) for (let k = 0; k < 3; k++) {
        if (!(i & (1 << k))) line(corners[i], corners[i | (1 << k)], '#bf984d', 1.2);
      }
    }
    const size = Number($('lidar3DPointSize').value) || 2;
    const projected = points.map(p => project(p)).filter(p => p && p[0] >= -size && p[0] <= canvas.width + size && p[1] >= -size && p[1] <= canvas.height + size);
    projected.sort((a, b) => b[2] - a[2]);
    ctx.fillStyle = '#58dfe8';
    for (const p of projected) { ctx.globalAlpha = clamp(orbit.distance / p[2], .55, 1); ctx.fillRect(p[0] - size / 2, p[1] - size / 2, size, size); }
    ctx.globalAlpha = 1;
    const axisLength = Math.max(.6, Math.min(1.2, dimensions[0] / 4));
    for (const [end, color, label] of [
      [[axisLength, 0, 0], '#fa9a7a', 'x 前'], [[0, axisLength, 0], '#8fdf91', 'y 左'], [[0, 0, axisLength], '#9bbfff', 'z 上']
    ]) {
      line([0, 0, 0], end, color, 2); const p = project(end);
      if (p) { ctx.fillStyle = color; ctx.font = '14px "Microsoft YaHei", sans-serif'; ctx.fillText(label, p[0] + 5, p[1] - 4); }
    }
    ctx.fillStyle = '#b2c8d9'; ctx.font = '14px "Microsoft YaHei", sans-serif';
    ctx.fillText('实测 LiDAR · 车辆局部坐标 [m] · '+($('lidar3DGrid').checked?'网格 0.5 m':'网格隐藏'), 16, canvas.height - 18);
    if (!points.length) { ctx.fillStyle = '#c3d2df'; ctx.font = '18px "Microsoft YaHei", sans-serif'; ctx.fillText(payload.actualCount === 0 ? '当前范围实测 0 点' : '当前范围没有可显示采样点', 24, 36); }
    $('lidar3DPointValue').textContent = size.toFixed(1) + ' px';
  }
  function schedule() { if (!queued) { queued = true; window.requestAnimationFrame(draw); } }
  function setData(data) {
    const key = data.contextId + '|' + data.trackId;
    const nextDimensions = finite3(data.dimensions) && data.dimensions.every(v => v > 0) ? data.dimensions : dimensions;
    const changed = key !== actorKey; actorKey = key; payload = data; dimensions = nextDimensions.slice();
    // Filter source kinds twice: mixed GS diagnostic mode must never leak Gaussians into this viewer.
    points = (data.points || []).filter(p => p.kind === 'lidar' && finite3(p.xyz)).map(p => p.xyz);
    if (changed) fit();
    panel.hidden = false;
    $('lidar3DIdentity').textContent = data.sceneName + ' / ' + data.actorAlias + (data.mode === 'all' ? ' · 全片汇聚（不是当前帧）' : ' · raw ' + data.raw);
    $('lidar3DInfo').textContent = '3D显示 ' + points.length.toLocaleString() + ' 个采样点；采样前框内实测 ' + (Number.isFinite(data.actualCount) ? data.actualCount.toLocaleString() : '未知') + ' 点。' + (data.mode === 'gs' ? '此3D视图只显示实测LiDAR，高斯中心在下方三视图对照。' : data.mode === 'all' ? '全片在车辆局部坐标汇聚，不能当作单帧密度。' : '与当前已显示的影像帧同步；显示采样不等于原始回波密度。');
    draw();
  }
  function clear(message) {
    payload = null; points = []; panel.hidden = true; dragging = null;
    $('lidar3DInfo').textContent = message || '尚无当前车辆的实测点。';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  $('lidar3DFit').addEventListener('click', () => { fit(); draw(); });
  const views = { orbit: [.78, .38], top: [0, 1.53], side: [Math.PI / 2, .02], front: [0, .02] };
  $('lidar3DView').addEventListener('change', () => { const view = views[$('lidar3DView').value] || views.orbit; orbit.yaw = view[0]; orbit.pitch = view[1]; orbit.panX = orbit.panY = 0; draw(); });
  for (const id of ['lidar3DBox', 'lidar3DGrid']) $(id).addEventListener('change', draw);
  $('lidar3DPointSize').addEventListener('input', draw);
  canvas.addEventListener('pointerdown', e => { dragging = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return; const rect = canvas.getBoundingClientRect(), scale = canvas.width / rect.width;
    const dx = (e.clientX - dragging.x) * scale, dy = (e.clientY - dragging.y) * scale;
    dragging = { x: e.clientX, y: e.clientY };
    if (e.shiftKey || e.buttons === 2) { orbit.panX += dx; orbit.panY += dy; }
    else { orbit.yaw += dx * .008; orbit.pitch = clamp(orbit.pitch + dy * .006, -1.53, 1.53); $('lidar3DView').value = 'orbit'; }
    schedule();
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { dragging = null; });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('wheel', e => { e.preventDefault(); orbit.distance = clamp(orbit.distance * Math.exp(e.deltaY * .001), .8, 150); schedule(); }, { passive: false });
  canvas.addEventListener('dblclick', () => { fit(); draw(); });
  window.ActorLidar3D = { setData, clear };
})();
