/* 目标关系 / effect 的独立诊断层。原语义、原分支、冻结检查点均不改写。 */
(function (root) {
  'use strict';
  const ROLE_NAMES = ['本目标外 / 未纳入', 'car候选', '本目标U', '本目标阴影U', '混合支撑U'];
  const PIXEL_NAMES = ['目标候选范围外', '车体证据像素', '轮廓候选像素', '车底阴影候选像素', '混合证据像素'];
  const COLORS = ['#465568', '#35dc98', '#d4d9df', '#ad83f5', '#f4a64b'];
  const WHOLE_ROW_RISK = '整颗删除风险：mixed表示同一颗高斯同时承担目标和背景；当前删除对照会整颗去除，可能损伤路面并形成背景空洞。';
  const CASES = ['baseline', 'remove_car', 'remove_u', 'remove_car_u'];
  const TITLES = { baseline: '冻结Actor10K基线', remove_car: '只去car', remove_u: '只去本目标U', remove_car_u: '去car＋本目标U' };
  const PREFIX = 'v43_target_detection_';
  const isMode = mode => mode === 'target_relation' || mode === 'effect';
  const rgb = color => [1, 3, 5].map(k => parseInt(color.slice(k, k + 2), 16));
  const ROLE_RGB = COLORS.map(rgb), EFFECT_RGB = ['#253244', '#245c4d', '#69727e', COLORS[3], '#635a65'].map(rgb);
  const fmtNumber = value => Number.isFinite(value) ? Number(value).toPrecision(5) : '未导出';

  function indexOf(ids, id) {
    if (!ids) return -1;
    let low = 0, high = ids.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (ids[mid] === id) return mid;
      if (ids[mid] < id) low = mid + 1; else high = mid - 1;
    }
    return -1;
  }

  function sourceRaws(evidence, i, frame, manifest) {
    if (i >= 0 && evidence?.source_raw_offsets && evidence?.source_raw_ids) {
      return { scope: 'row', raws: Array.from(evidence.source_raw_ids.slice(evidence.source_raw_offsets[i], evidence.source_raw_offsets[i + 1])) };
    }
    return { scope: 'global', raws: Array.from(frame?.globalsource_frames || manifest.globalsource_frames || evidence?.globalsource_frames || []) };
  }

  function rowRecord(manifest, frame, evidence, cloud, row) {
    if (cloud === manifest.actor?.cloud) {
      return { role: manifest.actor.role_default, name: ROLE_NAMES[manifest.actor.role_default], row, cloud, row_namespace: manifest.actor.row_namespace, effect: null, source_raw: frame?.source_raw, source: { scope: 'checkpoint', raws: Array.from(manifest.actor.source_raws || []) }, reason: '冻结V43 Actor10K基线的显式身份；此命名空间不复用原dynamic行号，身份不是GT。', evidence_index: -1 };
    }
    if (cloud !== 'static') {
      return { role: 0, name: '此cloud不在本目标检测命名空间', row, cloud, effect: null, source_raw: frame?.source_raw, source: { scope: 'none', raws: [] }, evidence_index: -1, reason: '不因动态分支或基础unknown自动归入此目标。' };
    }
    const i = indexOf(evidence?.ids, row);
    const role = i >= 0 ? Number(evidence.static_role[i]) : 0;
    const explicitShadow = i >= 0 && evidence.shadow_effect ? evidence.shadow_effect[i] === true || evidence.shadow_effect[i] === 1 : null;
    const hasShadow = role === 3 || explicitShadow === true;
    const result = { role, name: ROLE_NAMES[role] || '未识别角色', row, cloud, row_namespace: 'original_static', effect: hasShadow ? 'vehicle_shadow_candidate' : null, effect_source: explicitShadow === true ? 'shadow_effect_array' : role === 3 ? 'shadow_U_role' : null, shadow_effect: explicitShadow, mixed_support: role === 4, whole_row_removal_risk: role === 4 ? WHOLE_ROW_RISK : null, source_raw: frame?.source_raw, source: sourceRaws(evidence, i, frame, manifest), evidence_index: i, bg_protected: i >= 0 && evidence.bg_protected ? !!evidence.bg_protected[i] : null };
    for (const key of ['reason_bits', 'body_mass', 'shadow_mass', 'bg_mass', 'total_mass', 'target_fraction', 'shadow_fraction', 'body_frames', 'shadow_frames', 'bg_frames']) if (i >= 0 && evidence[key]) result[key] = Number(evidence[key][i]);
    const reasons = [];
    for (const [bit, label] of Object.entries(manifest.reason_labels || {})) if ((result.reason_bits || 0) & Number(bit)) reasons.push(label);
    result.reason = reasons.join('；') || ({ 0: i < 0 ? '本raw没有此行的目标专属证据；不能据基础unknown纳入U，也不能据此确认背景。' : '本目标外或未纳入；role0不代表已确认背景。', 1: '目标车体贡献支持的car候选；与原19类语义分别记录。', 2: '与此目标有关，但车体/背景证据不足或冲突。', 3: '车底阴影证据支持的effect候选；不把路面几何改成car。', 4: '同一原static行存在混合或冲突支撑；保留混合U，具体车体/阴影/背景证据见数值。' })[role];
    if (explicitShadow === true) result.reason += ' 显式shadow_effect=true，独立记录vehicle_shadow_candidate；不改变此行目标角色或原语义。';
    return result;
  }

  function roleColor(record, mode) {
    if (mode === 'effect') {
      if (record.effect) return record.role === 4 ? COLORS[4] : COLORS[3];
      return record.role === 1 ? '#245c4d' : record.role === 2 ? '#69727e' : record.role === 4 ? '#635a65' : '#253244';
    }
    return COLORS[record.role] || COLORS[0];
  }

  function pixelRecord(evidence, index) {
    const role = evidence?.pixel_role?.[index], flag = evidence?.pixel_shadow_effect?.[index];
    const shadow = flag === undefined ? null : flag === true || flag === 1;
    return { pixel_role: role === undefined ? null : Number(role), pixel_role_name: role === undefined ? '未提供底层pixel_role' : PIXEL_NAMES[role], pixel_shadow_effect: shadow, effect: shadow === true ? 'vehicle_shadow_candidate' : null, effect_scope: 'pixel', weak_candidate: shadow === true };
  }

  function overlayRGBA(base, roles, width, height, alpha = 0.48, pixelShadowEffect = null) {
    const count = width * height;
    if ((!roles && !pixelShadowEffect) || (roles && roles.length !== count)) throw Error('候选像素图缺失或尺寸不符');
    if (pixelShadowEffect && pixelShadowEffect.length !== count) throw Error('pixel_shadow_effect尺寸不符');
    const output = new Uint8ClampedArray(base);
    for (let i = 0; i < count; i++) {
      const role = roles ? Number(roles[i]) : 0;
      if (!role) continue;
      const color = role === 4 ? ((i % width + Math.floor(i / width)) % 8 < 4 ? COLORS[4] : COLORS[3]) : COLORS[role];
      const colorRGB = rgb(color);
      for (let k = 0; k < 3; k++) output[4 * i + k] = Math.round(base[4 * i + k] * (1 - alpha) + colorRGB[k] * alpha);
      output[4 * i + 3] = 255;
    }
    // 独立effect覆盖在候选色上，仅命中显式mask；不重写pixel_role或车体类别。
    if (pixelShadowEffect) {
      const colorRGB = rgb(COLORS[3]), shadowAlpha = 0.6;
      for (let i = 0; i < count; i++) {
        if (pixelShadowEffect[i] !== true && pixelShadowEffect[i] !== 1) continue;
        for (let k = 0; k < 3; k++) output[4 * i + k] = Math.round(output[4 * i + k] * (1 - shadowAlpha) + colorRGB[k] * shadowAlpha);
        output[4 * i + 3] = 255;
      }
    }
    return output;
  }

  function drawOverlay(canvas, image, evidence, width, height) {
    const x = canvas.getContext('2d', { willReadFrequently: true });
    x.drawImage(image, 0, 0, width, height);
    const pixels = x.getImageData(0, 0, width, height);
    pixels.data.set(overlayRGBA(pixels.data, evidence.pixel_role, width, height, 0.48, evidence.pixel_shadow_effect));
    x.putImageData(pixels, 0, 0);
  }

  async function unpackEvidence(payload) {
    if (!payload.arrays) return payload;
    const compressed = Uint8Array.from(atob(payload.data), ch => ch.charCodeAt(0));
    const buffer = await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
    const values = {}, types = { '<f4': Float32Array, '<f8': Float64Array, '<i4': Int32Array, '<u4': Uint32Array, '<u2': Uint16Array, '|u1': Uint8Array, '|b1': Uint8Array };
    for (const [key, meta] of Object.entries(payload.arrays)) {
      const part = buffer.slice(meta.offset, meta.offset + meta.bytes);
      if (meta.dtype === '<f2') {
        values[key] = Float32Array.from(new Uint16Array(part), bits => {
          const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >> 10) & 31, mantissa = bits & 1023;
          return exponent === 31 ? (mantissa ? NaN : sign * Infinity) : sign * (exponent ? Math.pow(2, exponent - 15) * (1 + mantissa / 1024) : mantissa * Math.pow(2, -24));
        });
      } else {
        const Type = types[meta.dtype];
        if (!Type) throw Error('检测证据不支持dtype ' + meta.dtype);
        values[key] = new Type(part);
      }
    }
    return values;
  }

  const core = { ROLE_NAMES, PIXEL_NAMES, COLORS, WHOLE_ROW_RISK, CASES, TITLES, indexOf, sourceRaws, rowRecord, pixelRecord, roleColor, overlayRGBA, drawOverlay, unpackEvidence, isMode };
  root.TargetDetectionCore = core;
  if (typeof document === 'undefined' || !root.TARGET_DETECTION_DELIVERY || typeof S === 'undefined') return;
  const manifest = root.TARGET_DETECTION_DELIVERY;
  if (manifest.context !== D.context) return;
  if (root.TargetDetection?.manifest === manifest) return; // 通用模板与旧局部挂载并存时只初始化一次。
  const state = { cache: new Map(), errors: new Map(), activeMode: null, fullTargetRows: false, panelEpoch: 0 };
  S.targetDetectionFrames = state.cache;
  const get = id => document.getElementById(id);
  const detectionCase = () => S.group.startsWith(PREFIX) ? S.group.slice(PREFIX.length) : null;
  const currentFrame = () => manifest.frames[String(S.raw)];
  const currentEvidence = () => state.cache.get(S.raw)?.value || null;
  const record = (cloud, row) => rowRecord(manifest, currentFrame(), currentEvidence(), cloud, row);
  const roleFor = (cloud, row) => cloud === manifest.actor?.cloud ? manifest.actor.role_default : cloud === 'static' ? state.cache.get(S.raw)?.roleTable?.[row] || 0 : 0;
  const effectFor = (cloud, row) => cloud === 'static' && (roleFor(cloud, row) === 3 || !!state.cache.get(S.raw)?.shadowTable?.[row]);

  async function evidenceFor(raw) {
    const frame = manifest.frames[String(raw)];
    if (!frame?.evidence_path) return null;
    if (state.cache.has(raw)) return state.cache.get(raw).promise;
    const entry = {};
    entry.promise = (async () => {
      await loadScript(frame.evidence_path);
      const payload = root.TARGET_DETECTION_FRAMES?.[String(raw)];
      if (!payload) throw Error('缺少本raw的检测证据包');
      const value = await unpackEvidence(payload);
      if (!value.ids || !value.static_role || value.ids.length !== value.static_role.length) throw Error('检测ids/static_role长度不符');
      entry.value = value;
      entry.roleTable = new Uint8Array(manifest.static_total || (value.ids.length ? value.ids[value.ids.length - 1] + 1 : 0));
      entry.shadowTable = new Uint8Array(entry.roleTable.length);
      for (let i = 0; i < value.ids.length; i++) {
        entry.roleTable[value.ids[i]] = value.static_role[i];
        entry.shadowTable[value.ids[i]] = value.shadow_effect?.[i] === true || value.shadow_effect?.[i] === 1 ? 1 : 0;
      }
      entry.pixelShadowCount = value.pixel_shadow_effect ? Array.from(value.pixel_shadow_effect).reduce((total, flag) => total + (flag === true || flag === 1 ? 1 : 0), 0) : null;
      state.errors.delete(raw);
      return value;
    })();
    state.cache.set(raw, entry);
    try { return await entry.promise; }
    catch (error) { state.cache.delete(raw); state.errors.set(raw, error.message); throw error; }
    finally {
      while (state.cache.size > 6) {
        const oldRaw = state.cache.keys().next().value;
        if (oldRaw === S.raw) break;
        const oldFrame = manifest.frames[String(oldRaw)];
        state.cache.delete(oldRaw);
        if (root.TARGET_DETECTION_FRAMES) delete root.TARGET_DETECTION_FRAMES[String(oldRaw)];
        if (typeof scriptCache !== 'undefined') scriptCache.delete(oldFrame?.evidence_path);
      }
    }
  }

  function add(tag, attributes = {}, text = '') {
    const element = document.createElement(tag);
    Object.assign(element, attributes);
    if (text) element.textContent = text;
    return element;
  }

  function makePanel() {
    const style = add('style');
    style.textContent = '#targetDetection .td-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}#targetDetection figure{margin:0;min-width:0}#targetDetection canvas{width:100%;height:auto;cursor:crosshair}#targetDetection figcaption{font-weight:650;margin:7px 0}#targetDetection .td-active{outline:2px solid #ffbf47;outline-offset:5px}#targetDetection .td-swatch{display:inline-block;width:11px;height:11px;margin:0 4px 0 12px;vertical-align:middle}#targetDetection .td-raw{font:13px monospace}#targetDetectionSelected{white-space:pre-wrap;overflow-wrap:anywhere}#targetDetection .td-buttons{display:flex;flex-wrap:wrap;gap:8px}#targetDetection .td-source{max-height:130px;overflow:auto}';
    document.head.append(style);
    const panel = add('details', { id: 'targetDetection', className: 'panel' });
    panel.append(add('summary', {}, '历史三帧检测（98 / 113 / 114）· 未通过 · 展开查看'));
    panel.ontoggle = () => { if (panel.open) evidenceFor(S.raw).catch(() => null).then(() => drawPanels(S.raw, S.epoch)); };
    panel.append(add('h2', {}, '历史检测对照 · 仅3帧已执行，非R2、非当前逐帧队列' + (manifest.quality_accepted === false ? ' · 未通过' : '')));
    panel.append(add('p', { className: 'note' }, '此历史实验仅执行raw98/113/114，其余帧未执行且没有后台排队。当前R2参考请使用上方入口。原static语义与19类编码、分支和owner分别保留；浅灰U只限此目标。阴影为 vehicle_shadow_candidate effect，身份与语义均非GT。'));
    panel.append(add('p', { id: 'targetDetectionQuality', className: 'notice bad' }, (manifest.quality_accepted === false ? '未通过：' : '质量尚未验收：') + (manifest.source_note || '仅显示实际执行的检测与删除对照；不把渲染完成当成干净剥离。')));
    panel.append(add('p', { id: 'targetDetectionRemovalRisk', className: 'notice' }, WHOLE_ROW_RISK));
    const buttons = add('div', { className: 'td-buttons' });
    const rawLabel = add('label', {}, '共享源帧 '), rawSelect = add('select', { id: 'targetDetectionRaw', ariaLabel: '检测诊断共享源帧' });
    for (const raw of Object.keys(manifest.frames).map(Number).filter(raw => manifest.frames[String(raw)].evidence_path).sort((a,b)=>a-b)) rawSelect.append(add('option', { value: String(raw) }, 'raw' + raw));
    rawSelect.append(add('option', { id: 'targetDetectionMissingRaw', value: '', disabled: true, hidden: true }, '当前帧未执行历史对照'));
    rawSelect.onchange = () => { S.raw = Number(rawSelect.value); show(); };
    rawLabel.append(rawSelect); buttons.append(rawLabel);
    for (const key of CASES) {
      const button = add('button', { id: 'targetDetectionCase_' + key }, TITLES[key] + ' ↔ 3D');
      button.onclick = () => activate(key);
      buttons.append(button);
    }
    const load = add('button', { id: 'targetDetectionLoadRows' }, '载入完整模型并显示此目标证据行');
    load.onclick = async () => {
      load.disabled = true;
      try {
        if (!detectionCase()) await activate('baseline');
        await evidenceFor(S.raw);
        state.fullTargetRows = true;
        get('targetBoost').value = 'none';
        await applySampling();
        draw3D();
      } finally { load.disabled = false; }
    };
    buttons.append(load);
    panel.append(buttons, add('p', { id: 'targetDetectionStatus', className: 'notice' }));
    const legend = add('p', { className: 'legend' });
    for (const [color, text] of [[COLORS[1], 'car候选'], [COLORS[2], '本目标U / 轮廓像素候选'], [COLORS[3], '车底阴影effect候选 / 紫色外边'], ['linear-gradient(90deg,' + COLORS[4] + ' 50%,' + COLORS[3] + ' 50%)', '橙紫双色=混合U；effect另记'], [COLORS[0], '本目标外 / 未纳入']]) {
      const swatch = add('span', { className: 'td-swatch' });
      swatch.style.background = color;
      legend.append(swatch, document.createTextNode(text));
    }
    panel.append(legend);
    const grid = add('div', { className: 'td-grid' });
    for (const key of ['observed', 'overlay', ...CASES]) {
      const figure = add('figure', { id: 'targetDetectionFigure_' + key });
      const label = key === 'observed' ? '同帧原相机实拍 RGB' : key === 'overlay' ? '原实拍＋检测证据候选叠加' : TITLES[key] + ' · 实际原生RGB' + (key !== 'baseline' && manifest.quality_accepted === false ? ' · 未通过' : '');
      const canvas = add('canvas', { id: 'targetDetectionCanvas_' + key, width: W, height: H, ariaLabel: label });
      canvas.onclick = event => pickPixel(event, canvas);
      figure.append(add('figcaption', {}, label), canvas, add('p', { id: 'targetDetectionNote_' + key, className: 'note td-raw' }));
      if (CASES.includes(key)) {
        const button = add('button', {}, '右侧3D同步此组');
        button.onclick = () => activate(key);
        figure.append(button);
      }
      grid.append(figure);
    }
    panel.append(grid, add('p', { id: 'targetDetectionPixel', className: 'notice' }));
    const details = add('details');
    details.append(add('summary', {}, '源raw、冻结检查点与检测口径'));
    details.append(add('p', { className: 'note' }, '2D候选来自导出的像素证据，3D来自同raw的原static行角色。pixel_shadow_effect是独立的二维弱候选mask，仅在显式标记处附加紫色覆盖，可与pixel_role=1车体像素并存；底层pixel_role和车身语义不变。轮廓像素不等于整行U，基础低置信unknown不自动进入目标U；bg_protected与行级shadow_effect是独立证据。mixed_U只有显式shadow_effect=true才增加阴影effect，role4保持不变。只去U默认去role2/3/4；各组使用交付中的实际removal_roles。'));
    details.append(add('pre', { id: 'targetDetectionSource', className: 'td-source note' }));
    panel.append(details);
    const status = get('status');
    status.parentNode.insertBefore(panel, status.nextSibling);
    const select = get('cloudColor');
    for (const [value, text] of [['target_relation', '此目标关系 · car / U / 阴影 / 混合'], ['effect', 'effect · vehicle_shadow_candidate']]) if (![...select.options].some(o => o.value === value)) select.append(add('option', { value }, text));
    const selected = add('div', { id: 'targetDetectionSelected', className: 'notice', hidden: true });
    const exportButton = add('button', { id: 'targetDetectionExport' }, '导出此点检测证据');
    exportButton.onclick = exportSelected;
    get('gaussianInfo').parentNode.insertBefore(selected, get('gaussianInfo').nextSibling);
    selected.parentNode.insertBefore(exportButton, selected.nextSibling);
  }

  async function activate(key = 'baseline', initialEntry = false) {
    if (!allGroups[PREFIX + key]) return;
    S.channel = 'rgb';
    S.cameraView = true;
    S.cameraReference = 'current';
    get('geometryState').value = 'current';
    get('static3d').checked = true;
    get('dynamic3d').checked = true;
    get('boxes3d').checked = true;
    get('otherBoxes3d').checked = true;
    get('semanticAblation').value = 'all';
    get('cloudColor').value = 'target_relation';
    setCategory('v43', PREFIX + key);
    const valid = Object.keys(manifest.frames).map(Number);
    const ready = valid.filter(raw => manifest.frames[String(raw)].evidence_path);
    if (!valid.includes(S.raw) || (initialEntry && !manifest.frames[String(S.raw)]?.evidence_path)) {
      const choices = ready.length ? ready : valid;
      S.raw = choices.includes(Number(manifest.preferred_raw)) ? Number(manifest.preferred_raw) : choices.includes(109) ? 109 : choices[0];
    }
    await show();
  }

  function drawObservedBoxes(canvas) {
    const x = canvas.getContext('2d');
    if (!IA?.frames[S.raw]) return;
    for (const box of IA.frames[S.raw].vehicle_boxes || []) {
      if (!box.in_front_image || !box.uv_sensor) continue;
      x.strokeStyle = box.target ? '#ffbf47' : '#59a9ff';
      x.lineWidth = box.target ? 1.5 : 0.7;
      x.beginPath();
      for (let n = 0; n < 8; n++) for (let k = 0; k < 3; k++) if (!(n & (1 << k))) { x.moveTo(...box.uv_sensor[n]); x.lineTo(...box.uv_sensor[n | (1 << k)]); }
      x.stroke();
    }
  }

  async function drawPanels(raw, epoch) {
    const panelEpoch = ++state.panelEpoch, frame = manifest.frames[String(raw)];
    const alive = () => raw === S.raw && epoch === S.epoch && panelEpoch === state.panelEpoch;
    const evidence = currentEvidence();
    const missingOption=get('targetDetectionMissingRaw');missingOption.hidden=!!frame?.evidence_path;missingOption.value=String(raw);missingOption.textContent='raw'+raw+' · 不在历史三帧内（未执行）';
    get('targetDetectionRaw').value = String(raw);
    const shadowCount = state.cache.get(raw)?.pixelShadowCount;
    get('targetDetectionStatus').textContent = 'raw' + raw + ' · ' + (frame?.evidence_path ? evidence ? evidence.ids.length.toLocaleString() + '行static目标证据已载入' : '检测证据加载中' : '历史对照未执行：本实验只有raw98/113/114') + (shadowCount == null ? '' : ' · 独立二维shadow effect ' + shadowCount + '像素（弱候选，未验收）') + (state.errors.has(raw) ? ' · ' + state.errors.get(raw) : '') + '；所有卡片与右侧共用此raw，不替换邻帧。';
    const source = { query_raw: raw, source_checkpoint: manifest.source_checkpoint, checkpoint_frozen: manifest.checkpoint_frozen, actor: manifest.actor, globalsource_frames: frame?.globalsource_frames || manifest.globalsource_frames || [], removal_roles: manifest.removal_roles, thresholds: frame?.thresholds || manifest.thresholds || null };
    get('targetDetectionSource').textContent = JSON.stringify(source, null, 2);
    const point = S.pixel[1] * W + S.pixel[0];
    const pixel = pixelRecord(evidence, point);
    get('targetDetectionPixel').textContent = 'raw' + raw + ' · 像素(' + S.pixel.join(', ') + ')：' + pixel.pixel_role_name + (pixel.pixel_role == null ? '' : '（pixel_role=' + pixel.pixel_role + '）') + '；独立pixel_shadow_effect=' + (pixel.pixel_shadow_effect == null ? '未导出' : String(pixel.pixel_shadow_effect)) + (pixel.effect ? '，vehicle_shadow_candidate弱候选' : '') + '。effect可与车体像素并存；底层pixel_role、车身语义和整行目标关系分别保留。';
    await Promise.all(['observed', 'overlay', ...CASES].map(async key => {
      const canvas = get('targetDetectionCanvas_' + key), note = get('targetDetectionNote_' + key);
      get('targetDetectionFigure_' + key).classList.toggle('td-active', detectionCase() === key);
      if (key === 'overlay' && !evidence?.pixel_role && !evidence?.pixel_shadow_effect) { empty(canvas, '历史对照未执行：此raw没有像素候选证据'); note.textContent = 'raw' + raw + ' · 不以3D中心投影冒充像素贡献'; return; }
      const path = key === 'observed' || key === 'overlay' ? D.gt[raw] : frame?.images?.[key];
      if (!path) { empty(canvas, '历史对照未执行：此raw没有该组原生RGB'); note.textContent = 'raw' + raw + ' · ' + (frame?.render_status?.[key] || 'pending · 未导出') + '；3D筛行预览不能代替原生重渲染'; return; }
      try {
        const image = await loadImage(path);
        if (!alive()) return;
        if (key === 'overlay') { drawOverlay(canvas, image, evidence, W, H); drawObservedBoxes(canvas); cross(canvas.getContext('2d'), W, H); }
        else imageOn(canvas, image);
        note.textContent = 'raw' + raw + ' · ' + (key === 'observed' ? '原始实拍，作为观测参照' : key === 'overlay' ? '导出像素证据叠加；独立紫色effect仅按pixel_shadow_effect，弱候选不改车身类别；橙框目标 / 蓝框其他车辆' : 'complete · 同帧native_gaussian执行凭据；' + (manifest.removal_roles[key] || []).join('/') + (key !== 'baseline' && manifest.quality_accepted === false ? '；未通过：整行剥离可能损伤背景，参照本轮验收记录。' : ''));
      } catch (error) { if (alive()) { empty(canvas, 'pending：图像读取失败'); note.textContent = 'raw' + raw + ' · ' + error.message; } }
    }));
  }

  function selectedEvidence() {
    const point = S.selected;
    if (!point || !S.clouds?.[point.cloud]) return null;
    const row = S.clouds[point.cloud].data.ids[point.j];
    const instance = geometryInstances().find(item => item.key === point.instance);
    let semantic = null;
    try { semantic = instance ? semanticRow(instance, point.j) : null; } catch (_) { /* 原语义尚未加载时保留空值。 */ }
    return { schema: 'target-detection-point-v1', context: D.context, target_track: manifest.target_track, group: S.group, query_raw: S.raw, color_mode: get('cloudColor').value, original_semantic: semantic, target_relation: record(point.cloud, row), source_checkpoint: manifest.source_checkpoint, checkpoint_frozen: true, human_verdict: null };
  }

  function updateSelected() {
    const box = get('targetDetectionSelected'), value = selectedEvidence();
    box.hidden = !value || !isMode(get('cloudColor').value);
    if (box.hidden) return;
    const r = value.target_relation;
    const scope = r.source.scope === 'row' ? '此行源raw' : r.source.scope === 'global' ? '全局证据源raw（未导出此行细分）' : r.source.scope === 'checkpoint' ? '冻结Actor checkpoint来源raw（如已导出）' : '此cloud无本目标来源证据';
    const list = r.source.raws;
    const source = list.length <= 24 ? list.join(', ') : list.slice(0, 12).join(', ') + ' … 共' + list.length + '帧（导出JSON含完整列表）';
    const fields = ['body_mass', 'shadow_mass', 'bg_mass', 'total_mass', 'target_fraction', 'shadow_fraction'].filter(key => r[key] !== undefined).map(key => key + '=' + fmtNumber(r[key])).join('；');
    box.textContent = '查询raw' + S.raw + ' · ' + r.row_namespace + '行' + r.row + '\n目标关系：' + r.name + '（role ' + r.role + '）；effect：' + (r.effect || '无已登记effect候选') + '；shadow_effect：' + (r.shadow_effect == null ? '未导出' : String(r.shadow_effect)) + '；bg_protected：' + (r.bg_protected === null ? '未导出' : String(r.bg_protected)) + '\n原因：' + r.reason + (r.whole_row_removal_risk ? '\n' + r.whole_row_removal_risk : '') + '\n' + fields + '\n' + scope + '：' + (source || '未导出') + '\n原语义（独立记录）：' + (value.original_semantic ? JSON.stringify(value.original_semantic) : '未加载') + '\nrole0不确认背景；阴影候选不改car/road语义。';
  }

  function exportSelected() {
    const value = selectedEvidence();
    if (!value) { get('targetDetectionSelected').hidden = false; get('targetDetectionSelected').textContent = '先点选右侧真实高斯中心。'; return; }
    const link = add('a', { href: URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })), download: 'target_detection_raw' + S.raw + '_' + value.target_relation.cloud.replace(/[^a-z0-9]/gi, '_') + '_' + value.target_relation.row + '.json' });
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  const originalEditedPoint = editedPoint;
  editedPoint = function (instance, j) {
    const point = originalEditedPoint(instance, j);
    if (state.activeMode) {
      const row = S.clouds[instance.cloud].data.ids[j], role = roleFor(instance.cloud, row);
      point.rgb = state.activeMode === 'effect' ? effectFor(instance.cloud, row) ? ROLE_RGB[role === 4 ? 4 : 3] : EFFECT_RGB[role] : ROLE_RGB[role];
    }
    return point;
  };
  const originalSampleBase = sampleBase;
  sampleBase = function (cloud, part, j) {
    if (state.activeMode && state.fullTargetRows) {
      const key = cloud === S.clouds.static ? 'static' : cloud === S.clouds[manifest.actor.cloud] ? manifest.actor.cloud : null;
      if (key && roleFor(key, cloud.data.ids[j]) > 0) return true;
    }
    return originalSampleBase(cloud, part, j);
  };
  const originalDraw3D = draw3D;
  draw3D = function () {
    const mode = get('cloudColor').value;
    if (!isMode(mode)) { originalDraw3D(); updateSelected(); return; }
    const ablation = get('semanticAblation').value;
    state.activeMode = mode;
    get('cloudColor').value = 'rgb';
    if (detectionCase()) get('semanticAblation').value = 'all';
    try { originalDraw3D(); }
    finally { state.activeMode = null; get('cloudColor').value = mode; get('semanticAblation').value = ablation; }
    const x = get('sceneCanvas').getContext('2d'), size = Number(get('pointSize').value) || 2.5;
    for (const point of S.projection) {
      const row = S.clouds[point.cloud].data.ids[point.j], role = roleFor(point.cloud, row), shadow = effectFor(point.cloud, row);
      if (role !== 4 && !shadow) continue;
      x.globalAlpha = get('opaquePoints').checked ? 1 : Math.max(0, Math.min(1, point.opacity));
      if (role === 4) {
        x.fillStyle = mode === 'effect' && !shadow ? '#685d7c' : COLORS[3];
        x.fillRect(point.x, point.y - size / 2, size / 2, size);
      }
      if (shadow) {
        x.strokeStyle = COLORS[3]; x.lineWidth = 1.25;
        x.strokeRect(point.x - size / 2 - 2, point.y - size / 2 - 2, size + 4, size + 4);
      }
    }
    x.globalAlpha = 1;
    get('geometryColorLegend').textContent = (mode === 'effect' ? 'effect模式：紫罗兰突出vehicle_shadow_candidate；没有effect的角色暗显。' : '') + ' 此目标关系：绿=car候选；浅灰=本目标U；紫罗兰=阴影U；橙紫双色=混合U。紫色外边=阴影effect（role3或显式shadow_effect=true）；mixed仍为role4，不自动推阴影。灰蓝=本目标外/未纳入。bg_protected独立，19类语义与原分支未改。橙框=目标原位，蓝框=其他车辆。';
    get('ownershipStatus').textContent = '检测仅使用本raw导出的目标专属角色；基础unknown不自动加入U。阴影是effect候选，不把道路几何标成car。' + (state.fullTargetRows ? '已请求全模型中此目标证据行完整显示；仍有时间/opacity/视野门限。' : '当前为真实checkpoint采样；载入完整模型可补齐此目标证据行。');
    const frame = currentFrame();
    if (!frame?.evidence_path || !currentEvidence()) get('geometryStateInfo').textContent += ' · 历史对照未执行：此raw检测证据不可用，灰蓝不代表已确认背景。';
    const key = detectionCase();
    if (key) {
      get('geometryStateInfo').textContent += ' · ' + (frame?.images?.[key] ? '本组同raw原生RGB已到达' : '本组原生RGB pending；右侧仅为真实行筛选几何预览');
      get('semanticFilterStatus').textContent = '本组按固定目标角色筛行：' + (manifest.removal_roles[key] || []).join('/') + '；独立于旧全局unknown开关。只去U保留显式Actor；只去car保留本目标U。原生RGB状态见同帧卡片。' + ((manifest.removal_roles[key] || []).includes(4) ? ' ' + WHOLE_ROW_RISK : '');
    }
    updateSelected();
  };
  const originalShow = show;
  show = async function (...args) {
    const result = originalShow(...args), raw = S.raw, epoch = S.epoch;
    if (detectionCase()) get('targetDetection').open = true;
    if (!get('targetDetection').open && !isMode(get('cloudColor').value)) return await result;
    await Promise.allSettled([result, evidenceFor(raw)]);
    if (raw !== S.raw || epoch !== S.epoch) return;
    await drawPanels(raw, epoch);
    if (raw === S.raw && epoch === S.epoch) draw3D();
  };

  makePanel();
  root.TargetDetection = { manifest, state, core, activate, evidenceFor, drawPanels, record, selectedEvidence, exportSelected };
  if (root.SceneViewer) Object.assign(root.SceneViewer, { show, draw3D, editedPoint, sampleBase });
  root.addEventListener('hashchange', () => { if (location.hash.startsWith('#targetDetection')) activate('baseline', true); });
  if (location.hash.startsWith('#targetDetection')) activate('baseline', true);
  else if (get('targetDetection').open) {
    const raw = S.raw, epoch = S.epoch;
    evidenceFor(raw).catch(() => null).then(() => { if (raw === S.raw && epoch === S.epoch) { drawPanels(raw, epoch); draw3D(); } });
  }
})(typeof window === 'undefined' ? globalThis : window);
