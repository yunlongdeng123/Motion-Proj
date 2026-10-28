// 仅文件、DOM shim与原生Canvas；不启动浏览器或HTTP服务。
'use strict';
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm'), assert = require('node:assert/strict');
const { createCanvas, loadImage } = require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const base = path.resolve(__dirname, '../../..'), report = path.join(base, 'outputs/V43_Ownership_20261027/gpu_v43/recall_ceiling_v1');
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.dataset = {}; this.style = {}; this.value = ''; this.classList = { add() {}, remove() {}, toggle() {} }; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  removeAttribute(name) { delete this[name]; }
}
const nodes = new Map(), doc = { createElement: tag => new Element(tag), getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element('div')); return nodes.get(id); } };
const sb = { document: doc, URLSearchParams, location: { hash: '' }, addEventListener() {} }; sb.window = sb;
vm.createContext(sb);
vm.runInContext(fs.readFileSync(path.join(report, 'data.js'), 'utf8'), sb);
vm.runInContext(fs.readFileSync(path.join(report, 'viewer.js'), 'utf8'), sb);
const api = sb.RecallCeiling;

async function main() {
  assert(api); assert.equal(api.data.raws.length, 3); assert.equal(api.stages.length, 5);
  const paths = new Set(), checks = [];
  for (const raw of api.data.raws) {
    const frame = api.data.frames[raw]; Object.values(frame.images).forEach(value => paths.add(value));
    for (const [stage] of api.stages) {
      api.select(raw, stage);
      const entry = frame.stages[stage];
      assert.equal(api.state.raw, raw); assert.equal(api.state.stage, stage);
      assert.equal(doc.getElementById('nativeImage').src, entry.images.native);
      assert.equal(doc.getElementById('gaussianImage').src, entry.images.gaussian);
      assert.equal(doc.getElementById('environmentImage').src, entry.images.environment);
      assert.equal(doc.getElementById('alphaImage').src, entry.images.alpha);
      assert.equal(doc.getElementById('metrics').children[0].children[1].textContent, api.format(entry.receipt.regions.body.alpha_mean));
      assert.equal(doc.getElementById('receiptLink').href, entry.receipt_path);
      assert.match(doc.getElementById('environmentRange').textContent, /环境 E.*原值范围/);
      assert.equal(doc.getElementById('ledgerMetrics').children[0].children[1].textContent, api.format(entry.ledger.remaining_broad_contributors));
      const receipt = JSON.parse(fs.readFileSync(path.join(report, entry.receipt_path), 'utf8'));
      assert.equal(JSON.stringify(receipt), JSON.stringify(entry.receipt));
      Object.values(entry.images).forEach(value => paths.add(value));
      assert(fs.existsSync(path.join(report, entry.miss_rows_path))); assert(fs.existsSync(path.join(report, frame.relaxations_path)));
      if (entry.dark_diagnostic) {
        assert.equal(doc.getElementById('darkRows').children.length, 3);
        paths.add(entry.dark_diagnostic.mask_path);
        if (stage === 'peel_01') for (const value of Object.values(entry.dark_diagnostic.receipt.regions)) { assert.equal(value.alpha_mass, 0); assert.equal(value.gaussian_rgb_integral.join('/'), '0/0/0'); assert.deepEqual(value.native_rgb_integral, value.environment_rgb_integral); }
      } else assert.match(doc.getElementById('darkStatus').textContent, /未做|未执行|尚无/);
      if (entry.row_examples.length) {
        doc.getElementById('rowExamples').children[0].children[0].children[0].onclick();
        const selected = JSON.parse(doc.getElementById('rowDetail').textContent);
        assert.equal(selected.row_namespace, 'original_static_rows'); assert.equal(selected.row, entry.row_examples[0].row);
      }
      checks.push({ raw, stage, body_alpha: entry.receipt.regions.body.alpha_mean, ledger_remaining: entry.ledger.remaining_broad_contributors, dark_query_present: !!entry.dark_diagnostic, image_sources_exact: true });
    }
  }
  api.select(113, 'existing');
  doc.getElementById('gateView').value = 'failed'; doc.getElementById('gateView').onchange();
  assert.equal(doc.getElementById('gateRows').children[0].children[0].textContent, api.data.frames[113].stages.existing.gate_mass.failed_gate_multilabel_sorted_by_broad_mass[0].gate);
  doc.getElementById('rowFilter').value = '1'; doc.getElementById('rowFilter').onchange();
  assert(doc.getElementById('rowExamples').children.length > 0);
  const sheet = createCanvas(1920, 1120), x = sheet.getContext('2d'); x.fillStyle = '#08111c'; x.fillRect(0, 0, 1920, 1120); x.font = '18px Microsoft YaHei'; x.fillStyle = '#e8eef6';
  x.fillText('最终 peel_01 · 原生RGB / 预乘高斯G / 环境T×E / alpha · 分量不作为修复图', 12, 24);
  for (let row = 0; row < api.data.raws.length; row++) {
    const raw = api.data.raws[row], entry = api.data.frames[raw].stages.peel_01, y = 42 + row * 360;
    x.fillStyle = '#e8eef6'; x.fillText('raw' + raw + ' · body/effect alpha=' + entry.receipt.regions.body.alpha_mean + '/' + entry.receipt.regions.effect.alpha_mean + '；车辆影身份仍需独立判断', 10, y + 18);
    let col = 0;
    for (const key of ['native', 'gaussian', 'environment', 'alpha']) { const image = await loadImage(path.join(report, entry.images[key])); x.drawImage(image, col++ * 480, y + 30, 480, 320); }
  }
  for (const relative of paths) { const image = await loadImage(path.join(report, relative)); assert.equal(image.width, 480); assert.equal(image.height, 320); }
  const html = fs.readFileSync(path.join(report, 'index.html'), 'utf8'), localLinks = [];
  for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
    const value = match[1].split('#')[0].split('?')[0]; if (!value || /^https?:|^data:/.test(value)) continue;
    assert(fs.existsSync(path.resolve(report, value)), value); localLinks.push(value);
  }
  fs.writeFileSync(path.join(report, 'canvas_final_components.png'), sheet.toBuffer('image/png'));
  const validation = JSON.parse(fs.readFileSync(path.join(report, 'ui_validation.json'), 'utf8'));
  Object.assign(validation, { browserQA: false, browser_verification_used_for_acceptance: false, native_canvas: true, decoded_actual_pngs: paths.size, javascript_state_logic: 'DOM shim; not browser QA', javascript_stage_checks: checks.length, row_id_and_gate_controls_checked: true, receipt_links_checked: localLinks.length, component_preview: 'canvas_final_components.png', javascript_checks: checks });
  fs.writeFileSync(path.join(report, 'ui_validation.json'), JSON.stringify(validation, null, 2));
  console.log(JSON.stringify({ native_canvas: true, real_png_decodes: paths.size, receipt_links_checked: localLinks.length, javascript_stage_logic_checks: checks.length, row_and_gate_controls_checked: true, browserQA: false, no_http_server_started: true }));
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
