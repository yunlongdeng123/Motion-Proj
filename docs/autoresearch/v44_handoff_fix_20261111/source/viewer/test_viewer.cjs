// Runs the real viewer against DOM/image substitutes. This is not browser QA.
const fs = require('fs'), path = require('path'), vm = require('vm'), assert = require('assert');
const root = path.resolve(process.argv[2]), asset = 'native_handoff_fix_20261111';
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(root, asset, 'viewer.js'), 'utf8');
const assignment = fs.readFileSync(path.join(root, asset, 'data.js'), 'utf8').trim();
const data = JSON.parse(assignment.slice(assignment.indexOf('=') + 1, -1));
const nodes = new Map(), events = {}, pending = new Map(), checks = [];
const state = { auto: true, saved: null, timer: null, intervals: 0 };
const columns = [
  ['observed', 'observed'], ['native30', 'native30'], ['deleteBefore', 'delete_before'],
  ['fixedKeep', 'fixed_keep'], ['fixedDelete', 'fixed_delete']
];
const ids = columns.map(([id]) => id);
function check(name, condition) { assert(condition, name); checks.push({ name, pass: true }); }
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.style = {}; this.children = [];
    this.width = 960; this.height = 640; this.value = ''; this.ctx = { last: null, boxes: 0 };
    for (const key of ['save', 'restore', 'beginPath', 'rect', 'clip', 'setLineDash', 'fillText', 'moveTo', 'lineTo', 'stroke']) this.ctx[key] = () => {};
    this.ctx.strokeRect = () => this.ctx.boxes++;
    this.ctx.fillRect = () => { this.ctx.last = null; this.ctx.boxes = 0; };
    this.ctx.drawImage = (image, ...args) => { this.ctx.last = { src: image.src, args }; };
  }
  append(...items) { this.children.push(...items); }
  getContext() { return this.ctx; }
  getBoundingClientRect() { return { left: 10, top: 20, width: 480, height: 320 }; }
  click() { this.onclick?.({ target: this }); }
}
for (const match of html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
  assert(!nodes.has(match[2]), 'Duplicate ID: ' + match[2]);
  nodes.set(match[2], new Element(match[1]));
}
for (const [key, value] of Object.entries({ view: 'full', overlay: 'none', sampling: 'smooth', fps: '5', experiment: 'fixed' })) nodes.get(key).value = value;
const ctx = {
  window: null, Blob, URLSearchParams, console, HANDOFF_FIX_ROOT: asset + '/', HANDOFF_FIX_DATA: data,
  location: { hash: '', href: '' }, history: { replaceState: (_a, _b, value) => { ctx.location.hash = value; } },
  document: {
    getElementById: key => { if (!nodes.has(key)) throw Error('Missing ID: ' + key); return nodes.get(key); },
    createElement: tag => new Element(tag), addEventListener: (name, fn) => { events[name] = fn; }
  },
  URL: { createObjectURL: blob => { state.saved = blob; return 'blob:mock'; }, revokeObjectURL: () => {} },
  setInterval: fn => { state.timer = fn; state.intervals++; return state.intervals; },
  clearInterval: () => { state.timer = null; }
};
ctx.window = ctx;
ctx.Image = class {
  set src(value) { this._src = value; pending.set(value, this); if (state.auto) queueMicrotask(() => this.onload?.()); }
  get src() { return this._src; }
};
const tick = async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); };
const jump = raw => nodes.get('rawNumber').onchange({ target: { value: raw } });
function finish(raw, missing) {
  const row = data.frames[raw];
  for (const [, key] of columns) {
    const image = pending.get(asset + '/' + row[key]);
    if (image) (missing === key ? image.onerror : image.onload)?.();
  }
}
const allAt = raw => ids.every(id => nodes.get(id).ctx.last?.src.endsWith('/' + raw + '.png'));
const importRecord = async record => {
  const file = nodes.get('recordFile');
  file.files = [{ size: 200, text: async () => JSON.stringify(record) }];
  await file.onchange({ target: file }); await tick();
};
const chooseVersion = id => nodes.get('experiment').onchange({ target: { value: id } });
vm.createContext(ctx); vm.runInContext(source, ctx);
(async () => {
  await tick();
  check('New schema and distinct experiment identify 199 complete scene-065 frames',
    data.schema === 'v44.handoff_fix.viewer.v1' && data.experiment_id !== data.baseline_experiment_id &&
    data.scene === '065' && data.frames.length === 199 && data.frames.every((frame, i) => frame.raw === i));
  check('Default frame 119 fills five synchronized canvases', ids.length === 5 && allAt(119));
  check('Main column order is observed/native30/before/fixedKEEP/fixedDELETE', columns.every(([id, key]) => nodes.get(id).ctx.last.src === asset + '/' + data.frames[119][key]));
  check('Prior deletion points to the preserved old asset directory', nodes.get('deleteBefore').ctx.last.src.includes('/../native_delete30k_20261109/delete_white/119.png'));
  check('Full 0..198 timeline and no held-out frames', data.heldout === 0 && Number(nodes.get('frame').min) === 0 && Number(nodes.get('frame').max) === 198);
  check('Track boxes default off in all five columns', nodes.get('overlay').value === 'none' && ids.every(id => nodes.get(id).ctx.boxes === 0));
  check('Two videos point to their actual independent assets', nodes.get('keep-video').href === asset + '/keep_30k.mp4' && nodes.get('delete-video').href === asset + '/delete_white_30k.mp4');
  check('All requested history choices are present', ['fixed', 'before30', 'native10', 'native20', 'native30', 'previous'].every(id => nodes.get('experiment').children.some(option => option.value === id)));

  state.auto = false; jump(120);
  check('A new requested frame clears every old image', ids.every(id => nodes.get(id).ctx.last === null));
  finish(120, 'fixed_keep'); await tick();
  check('One missing image preserves the four other current-frame images', ids.filter(id => id !== 'fixedKeep').every(id => nodes.get(id).ctx.last?.src.endsWith('/120.png')) && nodes.get('fixedKeep').ctx.last === null && nodes.get('status').textContent.includes('暂不可用'));
  jump(121); jump(122); finish(122); await tick(); finish(121); await tick();
  check('Out-of-order image loads cannot repaint an older frame', allAt(122));
  state.auto = true;
  nodes.get('fixedDelete').onclick({ clientX: 310, clientY: 220 });
  nodes.get('view').value = 'pixel'; nodes.get('view').onchange();
  check('Clicking any column shares pixel crop across all five', ids.every(id => nodes.get(id).ctx.last.args[0] === 220 && nodes.get(id).ctx.last.args[1] === 147));
  nodes.get('view').value = 'roi'; nodes.get('view').onchange();
  check('Vehicle ROI uses one shared viewport for all five columns', ids.every(id => JSON.stringify(nodes.get(id).ctx.last.args) === JSON.stringify(nodes.get(ids[0]).ctx.last.args)));
  nodes.get('sampling').value = 'nearest'; nodes.get('sampling').onchange();
  check('Pixel resampling applies to all five columns', ids.every(id => nodes.get(id).style.imageRendering === 'pixelated'));
  nodes.get('overlay').value = 'roles'; nodes.get('overlay').onchange();
  check('Optional track projection is synchronized across all five columns', ids.every(id => nodes.get(id).ctx.boxes > 0));

  nodes.get('play').click(); check('Play starts one shared frame timer', !!state.timer);
  state.timer(); await tick(); check('Shared playback advances all five together', allAt(123));
  nodes.get('fps').value = '10'; const oldIntervals = state.intervals; nodes.get('fps').onchange();
  check('Speed change restarts the same shared playback control', state.intervals === oldIntervals + 1 && !!state.timer);
  nodes.get('play').click(); check('Pause stops the shared timer', state.timer === null);
  jump(198); await tick(); nodes.get('play').click(); state.timer(); await tick(); nodes.get('play').click();
  check('Playback wraps the entire synchronized clip to frame zero', allAt(0));
  jump(-200); await tick(); check('Negative frame input clamps to frame zero', allAt(0));
  jump(999); await tick(); check('Out-of-range frame input clamps to frame 198', allAt(198));
  jump('invalid'); await tick(); check('Invalid frame input restores default frame 119', allAt(119));

  nodes.get('note').value = 'review'; nodes.get('save').click();
  const saved = JSON.parse(await state.saved.text());
  check('Export has new schema, both experiment identities and no implied verdict', saved.schema === 'v44.handoff_fix.review.v1' && saved.experiment_id === data.experiment_id && saved.baseline_experiment_id === data.baseline_experiment_id && saved.human_verdict === null && saved.note === 'review');
  await importRecord({ ...saved, raw: 150, pixel: [100, 200], view: 'pixel', overlay: 'none' });
  check('Current-experiment record restores frame, pixel, view and overlay', allAt(150) && nodes.get('view').value === 'pixel' && nodes.get('overlay').value === 'none' && nodes.get('pixel').textContent.includes('(100, 200)'));
  for (const [schema, expected] of [
    ['v44.delete30k.review.v1', '修复前'],
    ['v44.native_continuation.review.v1', '原生续训'],
    ['v44.native_quality.review.v1', '原生10K'],
    ['old-unrelated-schema', '历史重建与编辑']
  ]) {
    await importRecord({ ...saved, schema, raw: 173 });
    check('Old record is rejected with the correct route: ' + schema, nodes.get('recordStatus').textContent.includes(expected) && allAt(150));
  }
  await importRecord({ ...saved, baseline_experiment_id: 'different-baseline', raw: 173 });
  check('Record from a different comparison baseline is rejected', nodes.get('recordStatus').textContent.includes('其他场景或实验') && allAt(150));
  await importRecord({ ...saved, pixel: ['bad', 2] });
  check('Malformed record cannot modify the current comparison', nodes.get('recordStatus').textContent.includes('无效') && allAt(150));

  chooseVersion('before30');
  check('Four-column history receives current frame and original controls', ctx.location.href === 'native_delete30k.html#raw=150&step=30000&edit=delete');
  chooseVersion('native20'); check('Native 20K history receives the correct step', ctx.location.href === 'native_continuation.html#raw=150&step=20000');
  chooseVersion('native30'); check('Native 30K history receives the correct step', ctx.location.href === 'native_continuation.html#raw=150&step=30000');
  chooseVersion('native10'); check('Native 10K history remains available', ctx.location.href === 'native_10k.html#raw=150');
  chooseVersion('previous'); check('Older editing history remains available', ctx.location.href === 'history_previous.html#raw=150');
  events.keydown({ target: { tagName: 'DIV' }, key: 'ArrowRight', preventDefault: () => {} }); await tick();
  check('Keyboard navigation advances all five columns', allAt(151));
  events.keydown({ target: { tagName: 'TEXTAREA' }, key: 'ArrowRight', preventDefault: () => { throw Error('Should preserve typing'); } });
  check('Keyboard navigation does not consume text editing keys', allAt(151));
  check('Viewer performs no fetch or XMLHttpRequest', !/\bfetch\s*\(|XMLHttpRequest/.test(source));
  const result = { status: 'PASS_CPU_DOM_LOGIC', checks, browser_qa: false, images_validated: false };
  fs.writeFileSync(path.join(root, 'logic_checks.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({ status: result.status, passed: checks.length, browser_qa: false }));
})().catch(error => { console.error(error); process.exitCode = 1; });
