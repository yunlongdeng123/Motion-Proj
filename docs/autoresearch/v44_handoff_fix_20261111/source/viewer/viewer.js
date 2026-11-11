(() => {
  'use strict';
  const D = window.HANDOFF_FIX_DATA;
  const R = window.HANDOFF_FIX_ROOT;
  const $ = id => document.getElementById(id);
  const F = new Map(D.frames.map(row => [row.raw, row]));
  const roles = new Map(D.roles.map(role => [role.id, role]));
  const cache = new Map();
  const columns = [
    ['observed', 'observed'], ['native30', 'native30'],
    ['deleteBefore', 'delete_before'], ['fixedKeep', 'fixed_keep'],
    ['fixedDelete', 'fixed_delete']
  ];
  const S = {
    raw: D.default_raw, begin: D.default_start, end: D.frames.at(-1).raw,
    view: 'full', overlay: 'none', sampling: 'smooth', pixel: [240, 240],
    epoch: 0, playing: false, timer: null, pictures: null, errors: []
  };
  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

  function picture(path) {
    if (!cache.has(path)) {
      if (cache.size >= 40) cache.delete(cache.keys().next().value);
      cache.set(path, new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = () => { cache.delete(path); reject(Error('图像加载失败')); };
        im.src = R + path;
      }));
    }
    return cache.get(path);
  }

  function rect() {
    if (S.view === 'full') return [0, 0, D.width, D.height];
    if (S.view === 'pixel') return [clamp(S.pixel[0] - 80, 0, 320), clamp(S.pixel[1] - 53, 0, 214), 160, 106];
    const points = F.get(S.raw).boxes.flatMap(box => box.points).filter(p => p.every(Number.isFinite));
    if (!points.length) return [0, 0, D.width, D.height];
    const x = clamp(Math.min(...points.map(p => p[0])) - 14, 0, 479);
    const y = clamp(Math.min(...points.map(p => p[1])) - 18, 0, 319);
    const x2 = clamp(Math.max(...points.map(p => p[0])) + 14, x + 1, 480);
    const y2 = clamp(Math.max(...points.map(p => p[1])) + 12, y + 1, 320);
    return [x, y, x2 - x, y2 - y];
  }

  function viewport(canvas) {
    const r = rect(), scale = Math.min(canvas.width / r[2], canvas.height / r[3]);
    return { r, scale, x: (canvas.width - r[2] * scale) / 2, y: (canvas.height - r[3] * scale) / 2 };
  }

  function blank(id, message) {
    const c = $(id), g = c.getContext('2d');
    g.fillStyle = '#08101b'; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#a7bbce'; g.font = '24px Microsoft YaHei,sans-serif';
    g.fillText(message, 24, 48);
  }

  function paint(canvas, im) {
    const g = canvas.getContext('2d'), v = viewport(canvas);
    canvas.style.imageRendering = S.sampling === 'nearest' ? 'pixelated' : 'auto';
    g.fillStyle = '#08101b'; g.fillRect(0, 0, canvas.width, canvas.height);
    g.imageSmoothingEnabled = S.sampling === 'smooth';
    g.drawImage(im, ...v.r, v.x, v.y, v.r[2] * v.scale, v.r[3] * v.scale);
    const point = p => [v.x + (p[0] - v.r[0]) * v.scale, v.y + (p[1] - v.r[1]) * v.scale];
    g.save(); g.beginPath(); g.rect(v.x, v.y, v.r[2] * v.scale, v.r[3] * v.scale); g.clip();
    if (S.overlay === 'roles') for (const box of F.get(S.raw).boxes) {
      const role = roles.get(box.id), p = box.points.filter(q => q.every(Number.isFinite));
      if (!role || !p.length) continue;
      const lo = point([Math.min(...p.map(q => q[0])), Math.min(...p.map(q => q[1]))]);
      const hi = point([Math.max(...p.map(q => q[0])), Math.max(...p.map(q => q[1]))]);
      g.strokeStyle = role.colour; g.lineWidth = 2; g.setLineDash([5, 4]);
      g.strokeRect(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]); g.setLineDash([]);
      g.font = '20px Microsoft YaHei,sans-serif'; g.fillStyle = role.colour;
      g.fillText(role.label, lo[0] + 3, Math.max(24, lo[1] - 7));
    }
    const q = point(S.pixel);
    g.strokeStyle = '#ffe5a0'; g.lineWidth = 1; g.beginPath();
    g.moveTo(q[0] - 8, q[1]); g.lineTo(q[0] + 8, q[1]);
    g.moveTo(q[0], q[1] - 8); g.lineTo(q[0], q[1] + 8); g.stroke(); g.restore();
  }

  function draw() {
    if (S.pictures) columns.forEach(([id], i) => S.pictures[i] ? paint($(id), S.pictures[i]) : blank(id, S.errors[i]));
    $('pixel').textContent = `选中像素 (${S.pixel.join(', ')}) · 同帧网格 ${D.width}×${D.height}`;
  }

  async function render() {
    const epoch = ++S.epoch, raw = S.raw, row = F.get(raw);
    S.pictures = null;
    columns.forEach(([id]) => blank(id, `正在加载第 ${raw} 帧…`));
    $('status').textContent = `正在加载第 ${raw} 帧…`;
    const results = await Promise.allSettled(columns.map(([, key]) => picture(row[key])));
    if (epoch !== S.epoch) return;
    S.pictures = results.map(result => result.status === 'fulfilled' ? result.value : null);
    S.errors = results.map(result => result.status === 'fulfilled' ? '' : '此帧图像加载失败');
    draw();
    $('status').textContent = `第 ${raw} 帧 · ${row.time_s.toFixed(2)} 秒${results.some(result => result.status === 'rejected') ? ' · 部分图像暂不可用' : ''}`;
  }

  function stop() {
    S.playing = false; clearInterval(S.timer); S.timer = null; $('play').textContent = '播放';
  }

  function setRaw(value) {
    const number = Number(value);
    S.raw = clamp(Math.round(Number.isFinite(number) ? number : D.default_raw), S.begin, S.end);
    $('frame').value = S.raw; $('rawNumber').value = S.raw;
    $('frameLabel').textContent = `帧 ${S.raw} / ${S.end}`;
    $('prev').disabled = S.raw === S.begin; $('next').disabled = S.raw === S.end;
    const hash = new URLSearchParams(location.hash.slice(1)); hash.set('raw', S.raw);
    history.replaceState(null, '', '#' + hash.toString());
    render();
  }

  function play() {
    if (S.playing) { stop(); return; }
    S.playing = true; $('play').textContent = '暂停';
    S.timer = setInterval(() => setRaw(S.raw === S.end ? S.begin : S.raw + 1), 1000 / Number($('fps').value));
  }

  function range() {
    for (const id of ['frame', 'rawNumber']) { $(id).min = S.begin; $(id).max = S.end; }
    $('window').textContent = S.begin === 0 ? '只看车辆事件片段' : '查看全片';
    $('window').hidden = D.default_start === 0;
  }

  for (const [id] of columns) $(id).onclick = event => {
    const canvas = $(id), b = canvas.getBoundingClientRect(), v = viewport(canvas);
    const x = (event.clientX - b.left) * canvas.width / b.width;
    const y = (event.clientY - b.top) * canvas.height / b.height;
    S.pixel = [clamp(Math.floor((x - v.x) / v.scale + v.r[0]), 0, 479), clamp(Math.floor((y - v.y) / v.scale + v.r[1]), 0, 319)];
    draw();
  };
  $('frame').oninput = event => { stop(); setRaw(event.target.value); };
  $('rawNumber').onchange = event => { stop(); setRaw(event.target.value); };
  $('prev').onclick = () => { stop(); setRaw(S.raw - 1); };
  $('next').onclick = () => { stop(); setRaw(S.raw + 1); };
  $('play').onclick = play;
  $('fps').onchange = () => { if (S.playing) { stop(); play(); } };
  for (const key of ['view', 'overlay', 'sampling']) $(key).onchange = () => { S[key] = $(key).value; draw(); };
  $('window').onclick = () => { stop(); S.begin = S.begin === 0 ? D.default_start : 0; range(); setRaw(S.raw); };

  for (const version of D.versions) {
    const option = document.createElement('option'); option.value = version.id; option.textContent = version.label;
    $('experiment').append(option);
  }
  $('experiment').value = 'fixed';
  $('experiment').onchange = event => {
    const version = D.versions.find(v => v.id === event.target.value);
    if (!version || !version.url) return;
    stop();
    const hash = new URLSearchParams({ raw: String(S.raw), ...(version.params || {}) });
    location.href = version.url + '#' + hash.toString();
  };
  for (const raw of D.jump_raws) {
    if (!F.has(raw)) continue;
    const button = document.createElement('button'); button.textContent = '帧 ' + raw;
    button.onclick = () => { stop(); if (raw < S.begin) { S.begin = 0; range(); } setRaw(raw); };
    $('jumps').append(button);
  }
  $('keep-video').href = R + D.videos.keep;
  $('delete-video').href = R + D.videos.delete_white;
  $('origin').textContent = `${D.frame_count} 帧同步对照 · 全部图像参与训练，未设置留出。修复后两列来自同一编辑场景。`;
  const table = document.createElement('table');
  for (const values of [['车辆', '原始轨迹编号'], ...D.roles.map(role => [role.label, role.track])]) {
    const tr = document.createElement('tr');
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    table.append(tr);
  }
  $('identity').append(table);

  $('save').onclick = () => {
    const record = {
      schema: 'v44.handoff_fix.review.v1', context: D.context,
      experiment_id: D.experiment_id, baseline_experiment_id: D.baseline_experiment_id,
      raw: S.raw, pixel: S.pixel, view: S.view, overlay: S.overlay, sampling: S.sampling,
      note: $('note').value.slice(0, 5000), human_verdict: null
    };
    const anchor = document.createElement('a');
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
    anchor.href = url; anchor.download = `scene-${D.scene}-handoff-fix-${S.raw}.json`;
    anchor.click(); URL.revokeObjectURL(url);
    $('recordStatus').textContent = '已导出当前帧、像素和备注。';
  };
  $('load').onclick = () => $('recordFile').click();
  $('recordFile').onchange = async event => {
    try {
      const file = event.target.files[0];
      if (!file || file.size > 500000) throw Error('请选择小于500KB的当前视图记录。');
      const record = JSON.parse(await file.text());
      if (record.schema === 'v44.delete30k.review.v1') throw Error('这是修复前的记录，请在“修复前 · 30K 白车删除”载入。');
      if (record.schema === 'v44.native_continuation.review.v1') throw Error('这是原生续训记录，请在“原生重建 · 20K”或“原生重建 · 30K”载入。');
      if (record.schema === 'v44.native_quality.review.v1') throw Error('这是原生10K记录，请在“原生重建 · 10K”载入。');
      if (record.schema !== 'v44.handoff_fix.review.v1') throw Error('记录格式不属于当前比较；旧编辑记录请在“历史重建与编辑”载入。');
      if (record.context !== D.context || record.experiment_id !== D.experiment_id || record.baseline_experiment_id !== D.baseline_experiment_id) throw Error('这份记录来自其他场景或实验。');
      if (!Number.isInteger(record.raw) || !F.has(record.raw) || !Array.isArray(record.pixel) || record.pixel.length !== 2 || !record.pixel.every(Number.isFinite)) throw Error('记录帧号或像素无效。');
      stop();
      if (record.raw < S.begin) { S.begin = 0; range(); }
      S.pixel = record.pixel.map((value, i) => clamp(Math.floor(value), 0, i ? 319 : 479));
      for (const [key, allowed] of [['view', ['full', 'roi', 'pixel']], ['overlay', ['none', 'roles']], ['sampling', ['smooth', 'nearest']]]) {
        if (allowed.includes(record[key])) { S[key] = record[key]; $(key).value = record[key]; }
      }
      $('note').value = String(record.note || '').slice(0, 5000);
      setRaw(record.raw); $('recordStatus').textContent = '已载入同实验的观察记录。';
    } catch (error) { $('recordStatus').textContent = error.message; }
    event.target.value = '';
  };
  document.addEventListener('keydown', event => {
    if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(event.target.tagName) || event.target.isContentEditable || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault(); stop(); setRaw(S.raw + (event.key === 'ArrowRight' ? 1 : -1) * (event.shiftKey ? 10 : 1));
    }
    if (event.code === 'Space') { event.preventDefault(); play(); }
  });
  range();
  const hash = new URLSearchParams(location.hash.slice(1)), requested = Number(hash.get('raw'));
  if (hash.has('raw') && F.has(requested) && requested < S.begin) { S.begin = 0; range(); }
  setRaw(hash.has('raw') ? requested : D.default_raw);
})();
