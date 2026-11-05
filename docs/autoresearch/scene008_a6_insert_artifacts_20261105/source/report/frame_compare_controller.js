/* Local-file compatible, atomic frame comparison. No fetch or external library. */
(function () {
  'use strict';
  const root = document.getElementById('frame-compare');
  if (!root) return;
  const byId = id => document.getElementById(id);
  const controls = Array.from(root.querySelectorAll('[data-fc-control]'));
  const status = byId('fc-status');
  const overlay = byId('fc-overlay');
  function failBoot(error) {
    status.textContent = '切换脚本未启动；仍显示初始 raw113 / delete 全景。' +
      '可打开完整静态49帧对照。错误：' + String(error.message || error);
    status.dataset.level = 'error';
    controls.forEach(x => { x.disabled = true; });
  }
  try {
    const data = JSON.parse(byId('fc-data').textContent);
    const range = byId('fc-range');
    const stateSelect = byId('fc-state');
    const viewSelect = byId('fc-view');
    const currentLabel = byId('fc-current');
    const requestedLabel = byId('fc-requested');
    const grid = byId('fc-grid');
    const columns = data.columns;
    const first = data.default_raw;
    let displayed = {raw:first, state:data.default_state};
    let requested = {raw:first, state:data.default_state};
    let sequence = 0;
    let pending = false;
    let errors = [];
    let activeView = 'full';
    if (data.raw_ids.length !== 49 || !data.frames[String(first)]) {
      throw new Error('帧映射不完整');
    }
    const groupPaths = (raw, state) => columns.map(col => {
      const frame = data.frames[String(raw)];
      return col.id === 'reference' ? frame.reference.path : frame[state][col.id].path;
    });
    function description(item) {
      return 'raw' + item.raw + ' / ' +
        (item.state === 'delete' ? 'delete a37' : 'keep a37');
    }
    function viewDescription() {
      return activeView === 'crop' ? '同帧统一裁剪' : '全景480×320';
    }
    function applyView() {
      const frame = data.frames[String(displayed.raw)];
      const crop = frame.crop_xyxy;
      if (activeView === 'crop' && crop) {
        const [x0,y0,x1,y1] = crop;
        const w=x1-x0, h=y1-y0;
        grid.style.setProperty('--fc-ratio', String(w/h));
        grid.style.setProperty('--fc-image-width', (480/w*100)+'%');
        grid.style.setProperty('--fc-image-height', (320/h*100)+'%');
        grid.style.setProperty('--fc-left', (-x0/w*100)+'%');
        grid.style.setProperty('--fc-top', (-y0/h*100)+'%');
        grid.classList.add('fc-cropped');
        byId('fc-crop-note').textContent = '所有列共用 raw' + displayed.raw +
          ' crop [' + crop.join(', ') + ']；keep/delete 使用同一窗口。';
      } else {
        grid.classList.remove('fc-cropped');
        grid.style.setProperty('--fc-ratio', '1.5');
        byId('fc-crop-note').textContent = '全景480×320；点击每列“打开原尺寸”查看无损PNG。';
      }
      currentLabel.textContent = '当前显示：' + description(displayed) + ' / ' + viewDescription();
    }
    function updateControls() {
      range.value = String(requested.raw);
      requestedLabel.textContent = '选择 raw' + requested.raw;
      stateSelect.value = requested.state;
      byId('fc-prev').disabled = requested.raw <= data.raw_ids[0];
      byId('fc-next').disabled = requested.raw >= data.raw_ids[data.raw_ids.length-1];
    }
    function preload(path) {
      return new Promise((resolve,reject) => {
        const im = new Image();
        let done = false;
        const finish = (error) => {
          if (done) return;
          done = true;
          clearTimeout(timer);
          if (error) reject({path:path, reason:String(error)});
          else resolve(im);
        };
        const timer = setTimeout(() => finish('加载超时'), 10000);
        im.onload = async () => {
          if (im.naturalWidth !== 480 || im.naturalHeight !== 320) {
            finish('图像尺寸不是480×320'); return;
          }
          try {
            if (typeof im.decode === 'function') await im.decode();
            finish(null);
          } catch (e) { finish('图像解码失败'); }
        };
        im.onerror = () => finish('无法加载本地文件');
        im.src = path;
      });
    }
    async function requestGroup(raw, state) {
      if (!Number.isInteger(raw) || !data.frames[String(raw)] ||
          (state !== 'keep' && state !== 'delete')) {
        throw new Error('无效帧或状态');
      }
      requested = {raw:raw, state:state};
      updateControls();
      const ticket = ++sequence;
      pending = true; errors = [];
      const target = {raw:raw,state:state};
      const paths = groupPaths(raw,state);
      status.dataset.level = 'loading';
      status.textContent = '正在加载 ' + description(target) +
        ' 的全部列；当前仍显示 ' + description(displayed) + '。';
      overlay.hidden = false;
      overlay.textContent = '加载中；仍显示 ' + description(displayed);
      try {
        const loaded = await Promise.all(paths.map(preload));
        if (ticket !== sequence) return {ok:false, superseded:true};
        return await new Promise(resolve => requestAnimationFrame(() => {
          if (ticket !== sequence) { resolve({ok:false,superseded:true}); return; }
          // Already decoded Image nodes replace the whole group in one JS
          // turn/paint. No column shows an earlier frame under a new label.
          columns.forEach((col,index) => {
            const old = byId('fc-image-'+col.id);
            const im = loaded[index];
            im.id = old.id;
            im.alt = description(target) + ' / ' + col.label + ' / ' + paths[index];
            im.dataset.raw = String(raw);im.dataset.state = state;
            im.dataset.column = col.id;
            old.replaceWith(im);
            byId('fc-file-'+col.id).textContent = paths[index];
            byId('fc-open-'+col.id).href = paths[index];
          });
          displayed = target;
          pending = false; errors = [];
          applyView();
          overlay.hidden = true;
          status.dataset.level = 'ready';
          status.textContent = '已同步显示 ' + description(displayed) +
            '；全部列同帧、同相机、同裁剪。';
          root.dispatchEvent(new CustomEvent('fc:display', {detail:{raw:raw,state:state}}));
          resolve({ok:true,raw:raw,state:state});
        }));
      } catch (error) {
        if (ticket !== sequence) return {ok:false,superseded:true};
        pending = false; errors = [error.path || '(未知文件)'];
        status.dataset.level = 'error';
        status.textContent = '加载失败：' + errors[0] + '（' +
          (error.reason || '未知错误') + '）。整组未更新，仍显示 ' + description(displayed) + '。';
        overlay.hidden = false;
        overlay.textContent = '整组未更新，仍显示 ' + description(displayed) +
          '；失败文件：' + errors[0];
        return {ok:false,displayed:{...displayed},errors:errors.slice()};
      }
    }
    range.addEventListener('input', () => { requestGroup(Number(range.value), stateSelect.value); });
    stateSelect.addEventListener('change', () => { requestGroup(Number(range.value),stateSelect.value); });
    viewSelect.addEventListener('change', () => { activeView=viewSelect.value;applyView(); });
    byId('fc-prev').addEventListener('click', () => requestGroup(Math.max(89,requested.raw-1),requested.state));
    byId('fc-next').addEventListener('click', () => requestGroup(Math.min(137,requested.raw+1),requested.state));
    root.querySelectorAll('[data-fc-raw]').forEach(button => {
      button.addEventListener('click', () => requestGroup(Number(button.dataset.fcRaw),requested.state));
    });
    const initialErrors = new Set();
    function inspectInitial(explicitFailure) {
      if (sequence !== 0) return;
      if (explicitFailure) initialErrors.add(explicitFailure);
      let waiting = false;
      columns.forEach(col => {
        const im=byId('fc-image-'+col.id);
        if (!im.complete) waiting=true;
        else if (im.naturalWidth!==480 || im.naturalHeight!==320) {
          initialErrors.add(byId('fc-file-'+col.id).textContent);
        }
      });
      if (initialErrors.size) {
        errors=Array.from(initialErrors);
        status.dataset.level='error';
        status.textContent='初始 raw113 图片加载失败：'+errors.join('；')+'。可打开该文件或完整静态49帧对照。';
        overlay.hidden=false;overlay.textContent='初始raw113当前文件加载失败：'+errors.join('；');
      } else if (waiting) {
        status.dataset.level='loading';
        status.textContent='初始 raw113 / delete 图片仍在加载；文件路径已列在各列下方。';
        overlay.hidden=false;overlay.textContent='正在加载初始 raw113 / delete 整组图片';
      } else {
        overlay.hidden=true;status.dataset.level='ready';
        status.textContent='初始 raw113 / delete 全部图已加载。拖动后，新图全部解码完成才同步更新。';
      }
    }
    columns.forEach(col => {
      const im = byId('fc-image-'+col.id);
      im.addEventListener('load', () => { if (im.isConnected) inspectInitial(); });
      im.addEventListener('error', () => {
        if (im.isConnected) inspectInitial(byId('fc-file-'+col.id).textContent);
      });
    });
    controls.forEach(x => { x.disabled=false; });
    if (!data.has_all_crops) viewSelect.querySelector('[value="crop"]').disabled=true;
    updateControls();applyView();
    inspectInitial();
    window.a6FrameCompare = Object.freeze({
      select:(raw,state='delete',view=activeView) => {
        if (view==='crop' && !data.has_all_crops) throw new Error('没有完整可靠裁剪');
        activeView=view;viewSelect.value=view;return requestGroup(Number(raw),state);
      },
      getState:() => ({displayed:{...displayed},requested:{...requested},view:activeView,
        pending:pending,errors:errors.slice(),sequence:sequence,
        paths:groupPaths(displayed.raw,displayed.state)})
    });
  } catch (error) { failBoot(error); }
}());
