(function (root) {
  'use strict';
  const data = root.RECALL_CEILING_DATA;
  if (!data || typeof document === 'undefined') return;
  const get = id => document.getElementById(id);
  const stages = [
    ['actor_omitted', '只移除 Actor', '只移除既有 Actor，原 static 全部保留；这是初始贡献参照。'],
    ['existing', '旧候选剥离', '复放原目标候选 U 的整行剥离，作为召回缺口对照。'],
    ['gates_open_same_domain', '同域门槛全开', '固定旧域，按初始真实贡献放开历史、深度与比例门槛。'],
    ['broad_once', '扩域一次', '固定扩大域，一次去除初始可见支撑行；道路误选记录为成本。'],
    ['peel_01', 'after 重算一轮', '在相同扩大域内，以剥离后的真实透射重算贡献，并补删新显露行。']
  ];
  const state = { raw: data.raws.includes(113) ? 113 : data.raws[0], stage: 'existing' };
  const format = value => value === null || value === undefined || !Number.isFinite(value) ? 'pending' : value === 0 ? '0' : Math.abs(value) < .001 ? value.toExponential(3) : Number.isInteger(value) ? value.toLocaleString('en-US') : value.toFixed(6);
  function node(tag, text, attributes = {}) { const value = document.createElement(tag); Object.assign(value, attributes); if (text !== undefined) value.textContent = text; return value; }
  function alpha(frame, stage, region = 'body') { return frame?.stages?.[stage]?.receipt?.regions?.[region]?.alpha_mean; }
  function setImage(id, path, description) {
    const image = get(id); image.alt = description;
    image.style.visibility = 'hidden';
    image.classList.remove('pending');
    image.onload = () => { image.style.visibility = 'visible'; };
    image.onerror = () => { image.style.visibility = 'visible'; image.classList.add('pending'); image.alt = 'pending：此图文件未能读取 · ' + description; };
    if (path) image.src = path; else { image.style.visibility = 'visible'; image.removeAttribute('src'); image.classList.add('pending'); image.alt = 'pending：此raw／阶段未交付 · ' + description; }
  }
  function metric(label, value, emphasis = false) { const item = node('div', undefined, { className: 'metric' + (emphasis ? ' emphasis' : '') }); item.append(node('small', label), node('strong', format(value))); get('metrics').append(item); }
  const vector = values => values ? values.map(format).join(' / ') : 'pending';
  const categories = [[1, '旧域没覆盖'], [2, '入域但gate拦下'], [4, 'after越过ε'], [8, '旧probe未导出'], [16, '删除后才进入旧域']];
  function renderDark(entry) {
    const dark = entry?.dark_diagnostic;
    get('darkRows').replaceChildren();
    get('darkStatus').textContent = dark ? '当前 raw' + state.raw + ' / ' + state.stage + ' 的实际暗像素查询。body／effect／扩大域的积分分别核对；颜色积分相等说明数值来源，不赋予车辆身份。' : '此阶段尚无暗像素附加查询；只移除Actor控制未执行该查询，不以其他阶段替代。';
    get('darkSummaryLink').hidden = !data.dark_source;
    get('darkReceiptLink').hidden = !dark; get('darkReceiptLink').href = dark?.receipt_path || '#';
    setImage('darkMaskImage', dark?.mask_path, 'raw' + state.raw + ' ' + state.stage + ' 暗像素诊断mask');
    if (!dark) return;
    for (const [key, title] of [['body', '原 body 内'], ['effect', 'effect 域内'], ['broad', '扩大域内']]) {
      const value = dark.receipt.regions[key], row = node('tr');
      for (const item of [title, ['0.1', '0.2', '0.3'].map(t => format(value.dark_threshold_counts[t])).join(' / '), format(value.alpha_mass), format(value.gaussian_rows_above_1e4), vector(value.native_rgb_integral), vector(value.gaussian_rgb_integral), vector(value.environment_rgb_integral)]) row.append(node('td', item));
      get('darkRows').append(row);
    }
  }
  function renderLedger(entry, frame) {
    const ledger = entry?.ledger;
    get('ledgerSection').hidden = !data.ledger_source;
    get('ledgerMetrics').replaceChildren(); get('gateRows').replaceChildren(); get('rowExamples').replaceChildren();
    if (!ledger) return;
    for (const [label, value] of [['扩大域剩余贡献行', ledger.remaining_broad_contributors], ['旧域范围没覆盖', ledger.scope_missed], ['入域但gate拦下', ledger.inside_but_gated], ['after超过ε而initial未超过', ledger.newly_visible_vs_initial]]) { const item = node('div', undefined, { className: 'metric' }); item.append(node('small', label), node('strong', format(value))); get('ledgerMetrics').append(item); }
    const visibility = entry.visibility?.regions?.broad_domain;
    get('visibilityNote').textContent = visibility ? '上述 after 越过 ε 的行中：initial 数值恰为0的 ' + format(visibility.initial_mass_numerically_exact_zero) + ' 行；initial 正且≤ε的 ' + format(visibility.initial_positive_mass_at_or_below_epsilon) + ' 行。数值0不自动证明物理完全不可见；阈值穿越不等于全新几何。各条件可重叠，非真车影GT。' : '该阶段阈值分拆尚未提供。';
    get('missRowsLink').href = entry.miss_rows_path || '#'; get('relaxationsLink').href = frame.relaxations_path || '#';
    const mode = get('gateView').value || 'marginal';
    const gates = entry.gate_mass?.[mode === 'marginal' ? 'single_gate_marginal_sorted_by_broad_mass' : 'failed_gate_multilabel_sorted_by_broad_mass'] || [];
    for (const value of gates) { const row = node('tr'); for (const cell of [value.gate, format(value.rows), format(value.broad_alpha_mass), format(value.body_alpha_mass), format(value.effect_alpha_mass)]) row.append(node('td', cell)); get('gateRows').append(row); }
    const filter = Number(get('rowFilter').value) || 0;
    const rows = (entry.row_examples || []).filter(row => !filter || (row.category_bits & filter));
    for (const value of rows) { const row = node('tr'), cell = node('td'), button = node('button', String(value.row)); button.onclick = () => { get('rowDetail').textContent = JSON.stringify({ query_raw: state.raw, stage: state.stage, row_namespace: 'original_static_rows', ...value }, null, 2); }; cell.append(button); row.append(cell); for (const text of [categories.filter(([bit]) => value.category_bits & bit).map(([, label]) => label).join('；') || '未分入这些类别', format(value.initial_broad_mass) + ' / ' + format(value.after_broad_mass), String(value.failed_gates.length)]) row.append(node('td', text)); get('rowExamples').append(row); }
    if (!rows.length) { const row = node('tr'), cell = node('td', ledger.remaining_broad_contributors === 0 ? '当前扩大域无剩余贡献行；此阶段没有可列的遗漏行。' : '该类别没有代表行；完整集合以NPZ为准。'); cell.colSpan = 4; row.append(cell); get('rowExamples').append(row); }
    get('rowDetail').textContent = '选择代表原行查看其真实权重和gate；不代填身份判断。';
  }
  function render(updateHash = true) {
    const frame = data.frames[String(state.raw)], entry = frame?.stages?.[state.stage], receipt = entry?.receipt;
    for (const button of get('rawButtons').children) button.classList.toggle('active', Number(button.dataset.raw) === state.raw);
    get('stageSelect').value = state.stage;
    const stage = stages.find(item => item[0] === state.stage);
    get('frameStatus').textContent = 'raw' + state.raw + ' · ' + (receipt ? '实际产物' : 'pending：没有此阶段产物');
    get('stageNote').textContent = stage?.[2] || '未登记阶段';
    setImage('originalImage', frame?.images?.original, 'raw' + state.raw + ' 原相机实拍');
    setImage('domainImage', frame?.images?.domain, 'raw' + state.raw + ' 固定扩大域与紫色effect');
    setImage('environmentSourceImage', frame?.images?.environment_source, 'raw' + state.raw + ' 原环境查询');
    for (const [id, key, label] of [['nativeImage', 'native', '完整原生RGB'], ['gaussianImage', 'gaussian', '高斯分量'], ['environmentImage', 'environment', '环境回填分量'], ['alphaImage', 'alpha', '高斯alpha']]) setImage(id, entry?.images?.[key], 'raw' + state.raw + ' ' + state.stage + ' ' + label);
    get('metrics').replaceChildren();
    metric('body 平均 alpha', receipt?.regions?.body?.alpha_mean, true);
    metric('effect 平均 alpha', receipt?.regions?.effect?.alpha_mean, true);
    metric('扩大域平均 alpha', receipt?.regions?.broad_domain?.alpha_mean);
    metric('已移除 static 原行', receipt?.removed_static);
    metric('扩大域剩余贡献行', receipt?.contributing_static_in_broad);
    metric('body 内负环境像素', receipt?.regions?.body?.negative_environment_pixels);
    metric('body 内负原生像素', receipt?.regions?.body?.negative_native_pixels);
    metric('域外变化像素（成本）', receipt?.regions?.outside?.changed_pixels);
    get('environmentRange').textContent = '冻结环境 E 全图原值范围 ' + format(frame?.environment_stats?.minimum) + ' ～ ' + format(frame?.environment_stats?.maximum) + '；当前原生 RGB 原值 ' + format(entry?.float32_stats?.native_minimum) + ' ～ ' + format(entry?.float32_stats?.native_maximum) + '，全图显示裁剪像素 ' + format(entry?.float32_stats?.clipped_native_pixels) + '。来源：原 float32 数组；范围不表示车辆身份。';
    get('closureNote').textContent = '同状态 after 伴随闭合误差 ' + format(receipt?.adjoint_closure) + '；实际视角 SH 积分相对误差 ' + format(receipt?.SH_integral_relative_error) + '。';
    get('receiptLink').href = entry?.receipt_path || '#'; get('finalLink').href = frame?.final_path || '#';
    get('receiptText').textContent = receipt ? JSON.stringify(receipt, null, 2) : 'pending：不替换为其他raw或阶段';
    renderDark(entry); renderLedger(entry, frame);
    get('stageRows').replaceChildren();
    for (const [id, title] of stages) {
      const value = frame?.stages?.[id]?.receipt, row = node('tr', undefined, { className: state.stage === id ? 'current' : '' }); row.dataset.stage = id;
      for (const cell of [title, format(value?.removed_static), format(value?.regions?.body?.alpha_mean), format(value?.regions?.effect?.alpha_mean), format(value?.regions?.broad_domain?.alpha_mean), format(value?.contributing_static_in_broad), format(value?.regions?.outside?.changed_pixels)]) row.append(node('td', cell));
      row.onclick = () => select(state.raw, id); get('stageRows').append(row);
    }
    if (updateHash && typeof history !== 'undefined') history.replaceState(null, '', '#raw=' + state.raw + '&stage=' + state.stage);
  }
  function select(raw, stage) { if (!data.raws.includes(Number(raw)) || !stages.some(item => item[0] === stage)) return; state.raw = Number(raw); state.stage = stage; render(); }
  function parseHash() { const params = new URLSearchParams((root.location?.hash || '').replace(/^#/, '')); const raw = Number(params.get('raw')); const stage = params.get('stage'); if (data.raws.includes(raw)) state.raw = raw; if (stages.some(item => item[0] === stage)) state.stage = stage; }
  get('populationBadge').textContent = data.raws.length + ' 个原始时刻 × ' + stages.length + ' 个实际阶段';
  get('visualReview').hidden = !data.visual_review;
  get('finding').textContent = '旧候选仍留下高斯覆盖；同域门槛放开后明显降低，扩域并按 after 状态补删后，三帧 body／effect 的高斯 alpha 均为 0。完整原生 RGB 仍含冻结环境纹理，来源与车辆影身份须分开判断。';
  for (const raw of data.raws) {
    const button = node('button', 'raw' + raw, { type: 'button' }); button.dataset.raw = String(raw); button.onclick = () => select(raw, state.stage); get('rawButtons').append(button);
    const frame = data.frames[String(raw)], row = node('tr');
    const final = frame.stages[frame.final.final_stage]?.receipt;
    for (const value of ['raw' + raw, format(alpha(frame, 'existing')), format(alpha(frame, 'gates_open_same_domain')), format(alpha(frame, 'broad_once')), format(alpha(frame, 'peel_01')), format(final?.regions?.body?.negative_environment_pixels)]) row.append(node('td', value));
    get('overviewRows').append(row);
  }
  for (const [value, title] of stages) get('stageSelect').append(node('option', title, { value }));
  get('stageSelect').onchange = () => select(state.raw, get('stageSelect').value);
  get('gateView').onchange = get('rowFilter').onchange = () => { const frame = data.frames[String(state.raw)]; renderLedger(frame.stages[state.stage], frame); };
  const env = data.environment_source;
  get('environmentSourceLink').hidden = !env;
  get('environmentClosure').textContent = env ? '原环境为可学习六面 cubemap，线性采样没有输出非负约束。独立 texel 伴随积分的三帧相对闭合误差：' + env.receipts.map(item => 'raw' + item.raw + ' ' + format(item.environment_integral_relative_error)).join('；') + '。负 texel／像素不直接标为车辆阴影。' : 'pending：尚无独立环境 texel 归因产物。';
  get('sourceText').textContent = JSON.stringify(data.provenance, null, 2);
  parseHash(); render(false);
  root.addEventListener?.('hashchange', () => { parseHash(); render(false); });
  root.RecallCeiling = { data, state, stages, select, render, format };
})(typeof window === 'undefined' ? globalThis : window);
