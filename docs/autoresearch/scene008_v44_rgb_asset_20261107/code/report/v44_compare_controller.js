/* Local-file viewer: real relative PNGs, decoded groups, no network requests. */
(function () {
  'use strict';
  const root = document.getElementById('v44-compare');
  if (!root) return;
  const byId = id => document.getElementById(id);
  const controls = Array.from(root.querySelectorAll('[data-v44-control]'));
  const status = byId('v44-status');
  const overlay = byId('v44-overlay');
  try {
    const data = JSON.parse(byId('v44-viewer-data').textContent);
    const range = byId('v44-range');
    const stateSelect = byId('v44-state');
    const viewSelect = byId('v44-view');
    const observedCheckbox = byId('v44-observed');
    const grid = byId('v44-grid');
    const raws = data.raw_ids;
    const isPilot = data.mode === 'pilot';
    const labels = {current: 'current：当前车体＋旧背景', v44: 'v44：同一车体＋RCR-1 背景',
      observed: '原始实拍：a37 仍在，非删车 GT'};
    const viewLabels = {full: '全景 480×320', crop: '同帧共用车体裁剪',
      blue: '固定蓝 ROI 框', silver: '固定银 ROI 框'};
    if (!Array.isArray(raws) || !raws.includes(113) ||
        !data.frames['113'] || raws.some((x, i) => !Number.isInteger(x) ||
          (i > 0 && x <= raws[i - 1]))) throw new Error('真实帧映射无效');
    let displayed = {raw: 113, state: 'delete', view: 'full', observed: true};
    let requested = {...displayed};
    let sequence = 0;
    let pending = false;
    let errors = [];

    function describe(item) {
      return 'raw' + item.raw + ' / ' + item.state + ' a37 / ' + viewLabels[item.view];
    }
    function columns(item) { return item.observed ? ['current', 'v44', 'observed'] : ['current', 'v44']; }
    function entry(item, column) {
      const frame = data.frames[String(item.raw)];
      return column === 'observed' ? frame.observed : frame[column][item.state];
    }
    function normalize(item) {
      if (!raws.includes(item.raw) || !['keep', 'delete'].includes(item.state) ||
          !Object.prototype.hasOwnProperty.call(viewLabels, item.view)) throw new Error('帧或视图无效');
      const result = {...item};
      if (result.view !== 'full' && !data.frames[String(result.raw)].views[result.view]) {
        result.view = 'full';
      }
      return result;
    }
    function syncControls() {
      const index = raws.indexOf(requested.raw);
      range.value = String(isPilot ? index : requested.raw);
      byId('v44-requested').textContent = isPilot
        ? '索引 ' + index + ' / raw' + requested.raw : '选择 raw' + requested.raw;
      stateSelect.value = requested.state;
      viewSelect.value = requested.view;
      observedCheckbox.checked = requested.observed;
      byId('v44-prev').disabled = index === 0;
      byId('v44-next').disabled = index === raws.length - 1;
      Array.from(viewSelect.options).forEach(option => {
        option.disabled = option.value !== 'full' &&
          !data.frames[String(requested.raw)].views[option.value];
      });
      root.querySelectorAll('[data-v44-raw]').forEach(button => {
        button.disabled = !raws.includes(Number(button.dataset.v44Raw));
        button.setAttribute('aria-pressed', String(Number(button.dataset.v44Raw) === requested.raw));
      });
    }
    function applyView(item) {
      const box = data.frames[String(item.raw)].views[item.view];
      grid.style.setProperty('--v44-columns', String(columns(item).length));
      grid.classList.toggle('v44-cropped', Boolean(box));
      if (box) {
        const [x0, y0, x1, y1] = box;
        const width = x1 - x0, height = y1 - y0;
        grid.style.setProperty('--v44-ratio', String(width / height));
        grid.style.setProperty('--v44-image-width', (480 / width * 100) + '%');
        grid.style.setProperty('--v44-image-height', (320 / height * 100) + '%');
        grid.style.setProperty('--v44-left', (-x0 / width * 100) + '%');
        grid.style.setProperty('--v44-top', (-y0 / height * 100) + '%');
        byId('v44-crop-note').textContent = '所有列共用 raw' + item.raw + ' 的 ' +
          viewLabels[item.view] + ' [' + box.join(', ') + ']，坐标为 [x0,y0,x1,y1)。' +
          (item.view === 'blue' ? '蓝 ROI 指标只使用框内固定道路候选掩码；裁剪框本身不是评价分母。' :
            item.view === 'silver' ? '银 ROI 使用完整预登记矩形作描述统计，没有独立隐藏道路真值。' :
              '裁剪继承既有逐帧车体窗口；两版本与 keep/delete 均使用同一窗口。');
      } else {
        grid.style.setProperty('--v44-ratio', '1.5');
        byId('v44-crop-note').textContent = '全景 480×320。所有列均为本次实际结果 PNG；可打开原尺寸查看。';
      }
      byId('v44-current').textContent = '当前显示：' + describe(item) +
        (item.observed ? ' / 含原始实拍列' : ' / 两版本配对');
      byId('v44-static-current').href = data.static_page + '#raw' + item.raw + '-' + item.state;
    }
    function preload(path) {
      return new Promise((resolve, reject) => {
        const im = new Image();
        let done = false;
        const finish = error => {
          if (done) return;
          done = true;
          clearTimeout(timer);
          im.onload = null;
          im.onerror = null;
          if (error) reject({path, reason: String(error)});
          else resolve(im);
        };
        const timer = setTimeout(() => finish('本地图片加载超时'), 12000);
        im.onload = async () => {
          if (im.naturalWidth !== 480 || im.naturalHeight !== 320) {
            finish('尺寸不是 480×320'); return;
          }
          try {
            if (typeof im.decode === 'function') await im.decode();
            finish(null);
          } catch (error) { finish('图片解码失败'); }
        };
        im.onerror = () => finish('无法读取相对路径 PNG');
        im.src = path;
      });
    }
    function buildCard(column, image, item, path) {
      const card = document.createElement('article'); card.className = 'v44-card';
      const heading = document.createElement('h3'); heading.textContent = labels[column];
      const viewport = document.createElement('div'); viewport.className = 'v44-window';
      image.id = 'v44-image-' + column;
      image.width = 480; image.height = 320;
      image.alt = describe(item) + ' / ' + labels[column] + ' / ' + path;
      image.dataset.raw = String(item.raw); image.dataset.state = item.state;
      image.dataset.column = column; viewport.append(image);
      const footer = document.createElement('div'); footer.className = 'v44-card-footer';
      const link = document.createElement('a'); link.href = path;
      link.target = '_blank'; link.rel = 'noopener'; link.textContent = '打开原尺寸 PNG';
      const filename = document.createElement('code'); filename.textContent = path;
      footer.append(link, filename); card.append(heading, viewport, footer);
      return card;
    }
    async function requestGroup(item) {
      const wantedView = item.view;
      const target = normalize(item);
      const fallback = wantedView !== target.view;
      requested = {...target}; syncControls();
      const ticket = ++sequence;
      pending = true; errors = [];
      const selectedColumns = columns(target);
      const paths = selectedColumns.map(column => entry(target, column).path);
      status.dataset.level = 'loading';
      status.textContent = '正在加载整组 ' + describe(target) + '；仍显示 ' + describe(displayed) + '。';
      overlay.hidden = false; overlay.textContent = '正在解码整组；当前图片与标签仍属于 ' + describe(displayed);
      try {
        const loaded = await Promise.all(paths.map(preload));
        if (ticket !== sequence) return {ok: false, superseded: true};
        return await new Promise(resolve => requestAnimationFrame(() => {
          if (ticket !== sequence) { resolve({ok: false, superseded: true}); return; }
          const fragment = document.createDocumentFragment();
          selectedColumns.forEach((column, index) => fragment.append(buildCard(column, loaded[index], target, paths[index])));
          // One DOM commit after every visible PNG has decoded. Earlier requests
          // cannot commit after a newer request; failures leave this DOM intact.
          grid.replaceChildren(fragment);
          displayed = {...target}; requested = {...target}; pending = false; errors = [];
          applyView(displayed); syncControls(); overlay.hidden = true;
          status.dataset.level = 'ready';
          status.textContent = '已同步显示 ' + describe(displayed) + '；全部列同帧、同状态、同窗口。' +
            (fallback ? '该帧没有所选预登记 ROI，已切回全景。' : '');
          root.dispatchEvent(new CustomEvent('v44:display', {detail: {...displayed}}));
          resolve({ok: true, displayed: {...displayed}});
        }));
      } catch (error) {
        if (ticket !== sequence) return {ok: false, superseded: true};
        pending = false; errors = [error.path || '未知文件'];
        requested = {...displayed}; syncControls();
        status.dataset.level = 'error';
        status.textContent = '加载失败：' + errors[0] + '（' + (error.reason || String(error)) +
          '）。整组未更新，仍显示 ' + describe(displayed) + '。';
        overlay.hidden = false; overlay.textContent = '整组未更新；仍显示 ' + describe(displayed) + '。失败：' + errors[0];
        return {ok: false, displayed: {...displayed}, errors: errors.slice()};
      }
    }
    range.addEventListener('input', () => {
      const raw = isPilot ? raws[Number(range.value)] : Number(range.value);
      requestGroup({...requested, raw});
    });
    stateSelect.addEventListener('change', () => requestGroup({...requested, state: stateSelect.value}));
    viewSelect.addEventListener('change', () => requestGroup({...requested, view: viewSelect.value}));
    observedCheckbox.addEventListener('change', () => requestGroup({...requested, observed: observedCheckbox.checked}));
    byId('v44-prev').addEventListener('click', () => {
      const index = raws.indexOf(requested.raw); requestGroup({...requested, raw: raws[Math.max(0, index - 1)]});
    });
    byId('v44-next').addEventListener('click', () => {
      const index = raws.indexOf(requested.raw); requestGroup({...requested, raw: raws[Math.min(raws.length - 1, index + 1)]});
    });
    root.querySelectorAll('[data-v44-raw]').forEach(button => button.addEventListener('click',
      () => requestGroup({...requested, raw: Number(button.dataset.v44Raw)})));

    const initialImages = Array.from(grid.querySelectorAll('img'));
    function inspectInitial() {
      if (sequence !== 0) return;
      const broken = initialImages.filter(im => im.complete &&
        (im.naturalWidth !== 480 || im.naturalHeight !== 320));
      if (broken.length) {
        errors = broken.map(im => im.getAttribute('src'));
        status.dataset.level = 'error'; overlay.hidden = false;
        status.textContent = '初始 raw113 图片读取失败：' + errors.join('；') + '。可打开静态逐帧页面。';
        overlay.textContent = status.textContent;
      } else if (initialImages.some(im => !im.complete)) {
        status.dataset.level = 'loading';
        status.textContent = '初始 raw113 的真实 PNG 正在加载；路径已列在图片下方。';
      } else {
        errors = []; status.dataset.level = 'ready'; overlay.hidden = true;
        status.textContent = '初始 raw113 / delete 全部图片已加载。拖动后整组预解码，再同步切换。';
      }
    }
    initialImages.forEach(im => {
      im.addEventListener('load', inspectInitial); im.addEventListener('error', inspectInitial);
    });
    controls.forEach(control => { control.disabled = false; });
    syncControls(); applyView(displayed); inspectInitial();
    window.v44FrameCompare = Object.freeze({
      select: (raw, state = requested.state, view = requested.view, observed = requested.observed) =>
        requestGroup({raw: Number(raw), state, view, observed: Boolean(observed)}),
      getState: () => ({displayed: {...displayed}, requested: {...requested}, pending,
        sequence, errors: errors.slice(), paths: columns(displayed).map(column => entry(displayed, column).path)})
    });
  } catch (error) {
    controls.forEach(control => { control.disabled = true; });
    status.dataset.level = 'error';
    status.textContent = '切换脚本未启动；仍保留初始 raw113 / delete 的真实图片。可打开静态逐帧页面。' +
      String(error.message || error);
  }
}());
