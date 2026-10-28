// 合成fixture，只在内存运行真实Canvas；没有生成或标注任何实验图片。
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const zlib = require('node:zlib');
const { createCanvas } = require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const here = __dirname;
const runtime = fs.readFileSync(path.join(here, 'scene_target_detection.js'), 'utf8');
const pure = { console, Uint8ClampedArray, Uint8Array, Uint16Array, Uint32Array, Int32Array, Float32Array, Float64Array, Response, Blob, DecompressionStream, atob };
vm.createContext(pure);
vm.runInContext(runtime, pure);
const core = pure.TargetDetectionCore;

class Element {
  constructor(tag, owner) { this.tagName = tag; this.owner = owner; this.children = []; this.style = {}; this.classList = { toggle() {} }; this.value = ''; this.checked = true; this.textContent = ''; }
  set id(value) { this._id = value; this.owner.nodes.set(value, this); }
  get id() { return this._id; }
  get options() { return this.children; }
  get nextSibling() { if (!this.parentNode) return null; return this.parentNode.children[this.parentNode.children.indexOf(this) + 1] || null; }
  append(...items) { for (const item of items) { if (typeof item === 'object') item.parentNode = this; this.children.push(item); } }
  insertBefore(child, before) { child.parentNode = this; const i = this.children.indexOf(before); this.children.splice(i < 0 ? this.children.length : i, 0, child); }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  addEventListener() {}
  click() {}
}

function fakeDocument() {
  const owner = { nodes: new Map() };
  owner.createElement = tag => {
    const element = new Element(tag, owner);
    if (tag === 'canvas') {
      const canvas = createCanvas(480, 320);
      Object.defineProperty(element, 'width', { get: () => canvas.width, set: value => { canvas.width = value; } });
      Object.defineProperty(element, 'height', { get: () => canvas.height, set: value => { canvas.height = value; } });
      element.getContext = (...args) => canvas.getContext(...args);
      element.nativeCanvas = canvas;
    }
    return element;
  };
  owner.root = owner.createElement('main'); owner.head = owner.createElement('head');
  owner.getElementById = id => {
    if (owner.nodes.has(id)) return owner.nodes.get(id);
    const element = owner.createElement(/Canvas$/.test(id) || id === 'sceneCanvas' ? 'canvas' : ['cloudColor', 'category', 'arm', 'channel'].includes(id) ? 'select' : 'div');
    element.id = id;
    owner.root.append(element);
    if (id === 'sceneCanvas') { element.width = 840; element.height = 560; element.getContext('2d').drawImage = () => { throw Error('3D RGB backdrop prohibited'); }; }
    if (id === 'pointSize') element.value = '5';
    if (id === 'geometryState') element.value = 'current';
    if (id === 'cloudColor') element.value = 'target_relation';
    if (id === 'semanticProfile') element.value = 'gpu';
    if (id === 'semanticAblation') element.value = 'all';
    if (id === 'lidar3d') element.checked = false;
    return element;
  };
  owner.createTextNode = text => ({ textContent: text });
  return owner;
}

function cloud(ids, xyz) {
  const n = ids.length;
  return { total: n, sample: n, cycle: 1, data: { ids: Int32Array.from(ids), xyz: Float32Array.from(xyz.flat()), velocity: new Float32Array(n * 3), t: new Float32Array(n), sigma_t: Float32Array.from(ids, () => 100), opacity: Float32Array.from(ids, () => 1), rgb: Uint8Array.from(ids.flatMap(() => [10, 15, 20])), scales: Float32Array.from(ids.flatMap(() => [.05, .05, .05])), rotation: Float32Array.from(ids.flatMap(() => [1, 0, 0, 0, 1, 0, 0, 0, 1])) } };
}

async function test() {
  const manifest = { actor: { cloud: 'v43:synthetic_actor', role_default: 1, row_namespace: 'synthetic_actor' }, globalsource_frames: [1, 2, 3] };
  const frame = { source_raw: 109 };
  const evidence = { ids: Int32Array.from([10, 20, 30, 40, 50]), static_role: Uint8Array.from([0, 1, 2, 3, 4]), bg_protected: Uint8Array.from([1, 0, 0, 0, 1]), reason_bits: Uint32Array.from([0, 1, 2, 4, 8]), pixel_role: new Uint8Array(480 * 320) };
  evidence.pixel_role.set([0, 1, 2, 3, 4]);
  assert.equal(core.rowRecord(manifest, frame, evidence, 'static', 999).role, 0, '基础unknown不混入目标U');
  assert.equal(core.rowRecord(manifest, frame, evidence, 'dynamic', 20).role, 0, '原dynamic行号不继承新Actor身份');
  assert.equal(core.rowRecord(manifest, frame, evidence, 'v43:synthetic_actor', 20).role, 1, '新Actor仅在显式命名空间为car');
  assert.equal(core.rowRecord(manifest, frame, evidence, 'static', 40).effect, 'vehicle_shadow_candidate');
  assert.equal(core.rowRecord(manifest, frame, evidence, 'static', 50).bg_protected, true);
  assert.equal(core.rowRecord(manifest, frame, evidence, 'static', 50).effect, null, '不由role4推断阴影effect');
  evidence.shadow_effect = Uint8Array.from([0, 0, 0, 1, 1]);
  const shadowMixed = core.rowRecord(manifest, frame, evidence, 'static', 50);
  assert.equal(shadowMixed.role, 4, '显式阴影effect不改mixed_U角色');
  assert.equal(shadowMixed.mixed_support, true);
  assert.equal(shadowMixed.effect, 'vehicle_shadow_candidate');
  assert.equal(shadowMixed.effect_source, 'shadow_effect_array');
  assert.match(shadowMixed.whole_row_removal_risk, /整颗去除.*损伤路面/);
  evidence.shadow_effect[4] = 0;
  assert.equal(core.rowRecord(manifest, frame, evidence, 'static', 50).effect, null, '显式false的mixed_U没有阴影effect');
  evidence.shadow_effect[4] = 1;
  assert.equal(core.rowRecord(manifest, frame, evidence, 'static', 10).name, '本目标外 / 未纳入');

  const background = createCanvas(480, 320), x = background.getContext('2d'); x.fillStyle = '#203040'; x.fillRect(0, 0, 480, 320);
  const output = createCanvas(480, 320);
  core.drawOverlay(output, background, evidence, 480, 320);
  const result = output.getContext('2d').getImageData(0, 0, 5, 1).data;
  assert.deepEqual(Array.from(result.slice(0, 4)), [32, 48, 64, 255], 'outside不着色');
  assert(result[5] > result[4], 'body为绿色');
  assert(result[12 + 2] > result[12 + 1], 'shadow为紫罗兰');
  assert.throws(() => core.overlayRGBA(result, [0], 5, 1), /尺寸不符/);
  const rolesBeforeEffect = Array.from(evidence.pixel_role);
  evidence.pixel_shadow_effect = new Uint8Array(480 * 320);
  evidence.pixel_shadow_effect[0] = 1; // 原role0上的独立effect。
  evidence.pixel_shadow_effect[1] = 1; // 允许与原role1车体像素并存。
  const withoutPixelEffect = new Uint8ClampedArray(output.getContext('2d').getImageData(0, 0, 480, 320).data);
  core.drawOverlay(output, background, evidence, 480, 320);
  const withPixelEffect = output.getContext('2d').getImageData(0, 0, 480, 320).data;
  assert(withPixelEffect[2] > withPixelEffect[1], 'role0上的显式effect为紫色覆盖');
  assert(withPixelEffect[6] > withPixelEffect[5], 'role1车体像素上的独立effect为紫色覆盖');
  assert.deepEqual(Array.from(withPixelEffect.slice(8)), Array.from(withoutPixelEffect.slice(8)), '额外紫色覆盖严格只改mask标记像素');
  assert.deepEqual(Array.from(evidence.pixel_role), rolesBeforeEffect, '不重写底层pixel_role');
  assert.equal(core.pixelRecord(evidence, 1).pixel_role, 1);
  assert.equal(core.pixelRecord(evidence, 1).effect, 'vehicle_shadow_candidate');
  assert.equal(core.pixelRecord(evidence, 2).effect, null);
  assert.throws(() => core.overlayRGBA(result, [0, 1, 2, 3, 4], 5, 1, .48, [1]), /pixel_shadow_effect尺寸不符/);
  const effectOnly = core.overlayRGBA(Uint8ClampedArray.from([32, 48, 64, 255, 32, 48, 64, 255]), null, 2, 1, .48, [1, 0]);
  assert(effectOnly[2] > effectOnly[1]);
  assert.deepEqual(Array.from(effectOnly.slice(4)), [32, 48, 64, 255], '独立pixel effect不要求先有pixel_role');

  const binary = Buffer.concat([Buffer.from(new Uint32Array([1, 4]).buffer), Buffer.from([1, 0])]);
  const decoded = await core.unpackEvidence({ arrays: { reason_bits: { dtype: '<u4', offset: 0, bytes: 8 }, bg_protected: { dtype: '|b1', offset: 8, bytes: 2 } }, data: zlib.deflateSync(binary).toString('base64') });
  assert.deepEqual(Array.from(decoded.reason_bits), [1, 4]); assert.deepEqual(Array.from(decoded.bg_protected), [1, 0]);

  // 使用实际scene_viewer与实际draw3D；仅输入是显式合成fixture。
  const doc = fakeDocument(), identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const sourceState = { supported: true, static_visible: true, world_time: 0, camera: identity, objects: [{ key: 'target', cloud: manifest.actor.cloud, query_time: 0, label: 'SYNTHETIC ACTOR', render_only: true }, { key: 'original_dynamic', cloud: 'dynamic', query_time: 0, label: 'SYNTHETIC OTHER', render_only: true }], reference_boxes: [], target_detection: { case: 'baseline' } };
  const groups = ['baseline', 'remove_car', 'remove_u', 'remove_car_u'].map(key => ({ id: 'v43_target_detection_' + key, category: 'v43', title: 'SYNTHETIC ' + key, frames: { 109: { channels: {} }, 110: { channels: {} } } }));
  const delivered = { ...manifest, context: 'synthetic_context', target_track: 'synthetic_track', source_checkpoint: 'synthetic_frozen.pt', checkpoint_frozen: true, frames: { 109: { source_raw: 109, evidence_path: 'synthetic109.js', render_status: {}, images: {} }, 110: { source_raw: 110, images: {}, render_status: {} } }, groups, removal_roles: { baseline: [], remove_car: [1], remove_u: [2, 3, 4], remove_car_u: [1, 2, 3, 4] } };
  const sb = { console, document: doc, ...pure, setTimeout, URL, location: { hash: '' } }; sb.window = sb; sb.addEventListener = () => {};
  sb.SCENE_DATA = { context: delivered.context, target_track: delivered.target_track, frames: { 109: { c2w: identity, K: [200, 200, 240, 160], model_time: 0, target_center: [0, 0, 10] }, 110: { c2w: identity, K: [200, 200, 240, 160], model_time: 0, target_center: [0, 0, 10] } }, gt: { 109: 'synthetic109.png', 110: 'synthetic110.png' }, groups };
  sb.IDENTITY_AUDIT = null;
  sb.EDIT_GEOMETRY = { groups: Object.fromEntries(groups.map(group => [group.id, { 109: sourceState }])) };
  sb.TARGET_DETECTION_DELIVERY = delivered;
  sb.TARGET_DETECTION_FRAMES = { 109: evidence };
  vm.createContext(sb);
  let viewer = fs.readFileSync(path.join(here, '../persistent_world_20261027/scene_viewer.js'), 'utf8');
  for (const [marker, filename] of [['EDIT_GEOMETRY', 'scene_edit_geometry.js'], ['SAMPLING', 'scene_sampling.js'], ['SEMANTIC', 'scene_semantics.js']]) viewer = viewer.replace('/* ' + marker + '_RUNTIME */', fs.readFileSync(path.join(here, '../persistent_world_20261027/' + filename), 'utf8'));
  viewer = viewer.replace('/* BOUNDARY_RUNTIME */', fs.readFileSync(path.join(here, '../persistent_world_20261027/scene_boundary_semantics.js'), 'utf8')).replace(/;init\(\);\s*$/, ';');
  vm.runInContext(viewer, sb);
  sb.syntheticClouds = { static: cloud([10, 20, 30, 40, 50], [[-2, 0, 10], [-1, 0, 10], [0, 0, 10], [1, 0, 10], [2, 0, 10]]), dynamic: cloud([20], [[-1, 1, 10]]), [manifest.actor.cloud]: cloud([20], [[1, 1, 10]]) };
  sb.syntheticImage = background;
  vm.runInContext("S.clouds=syntheticClouds;S.group='v43_target_detection_baseline';S.raw=109;S.cameraView=true;S.cameraReference='current';loadScript=async()=>{};loadImage=async()=>syntheticImage;show=async()=>{S.epoch++;draw3D();};", sb);
  vm.runInContext(runtime, sb);
  await sb.TargetDetection.evidenceFor(109);
  sb.SceneViewer.draw3D();
  assert.equal(sb.SceneViewer.S.projection.length, 7, '真实draw3D显示7个合成点，不用RGB底图');
  const semanticColorsBefore = vm.runInContext('JSON.stringify(GPU_SEMANTIC_COLORS)', sb);
  const pixels3D = doc.getElementById('sceneCanvas').getContext('2d');
  const projected = sb.SceneViewer.S.projection;
  const sample = (cloudName, row) => { const p = projected.find(p => p.cloud === cloudName && sb.syntheticClouds[cloudName].data.ids[p.j] === row); return Array.from(pixels3D.getImageData(Math.floor(p.x), Math.floor(p.y), 1, 1).data).slice(0, 3); };
  assert.deepEqual(sample('static', 20), [53, 220, 152]);
  assert.deepEqual(sample('static', 30), [212, 217, 223]);
  assert.deepEqual(sample('static', 40), [173, 131, 245]);
  assert.deepEqual(sample(manifest.actor.cloud, 20), [53, 220, 152]);
  assert.deepEqual(sample('dynamic', 20), [70, 85, 104]);
  const mixedPoint = projected.find(p => p.cloud === 'static' && sb.syntheticClouds.static.data.ids[p.j] === 50);
  const mixedBorder = () => Array.from(pixels3D.getImageData(Math.floor(mixedPoint.x - 4.5), Math.floor(mixedPoint.y), 1, 1).data).slice(0, 3);
  assert.deepEqual(mixedBorder(), [173, 131, 245], '显式shadow_effect的mixed_U有独立紫色外边');
  assert.deepEqual(Array.from(pixels3D.getImageData(Math.floor(mixedPoint.x - 1), Math.floor(mixedPoint.y), 1, 1).data).slice(0, 3), [244, 166, 75], 'mixed仍保留橙色角色底色');
  evidence.shadow_effect[4] = 0; sb.TargetDetection.state.cache.get(109).shadowTable[50] = 0;
  sb.SceneViewer.draw3D();
  assert.deepEqual(mixedBorder(), [8, 17, 28], '没有显式shadow_effect的mixed_U不画阴影外边');
  assert.equal(sb.TargetDetection.record('static', 50).role, 4);
  assert.equal(sb.TargetDetection.record('static', 50).effect, null);
  evidence.shadow_effect[4] = 1; sb.TargetDetection.state.cache.get(109).shadowTable[50] = 1;
  doc.getElementById('cloudColor').value = 'effect'; sb.SceneViewer.draw3D();
  assert.deepEqual(mixedBorder(), [173, 131, 245], 'effect模式保留显式mixed阴影紫色外边');
  assert.equal(vm.runInContext('JSON.stringify(GPU_SEMANTIC_COLORS)', sb), semanticColorsBefore, 'effect不改19类编码/颜色表');
  await sb.TargetDetection.drawPanels(109, sb.SceneViewer.S.epoch);
  assert.match(doc.getElementById('targetDetectionNote_remove_car').textContent, /pending/);
  assert.match(doc.getElementById('targetDetectionNote_overlay').textContent, /pixel_shadow_effect.*不改车身类别/);
  vm.runInContext('S.raw=110;S.epoch++;', sb);
  await sb.TargetDetection.drawPanels(110, sb.SceneViewer.S.epoch);
  assert.match(doc.getElementById('targetDetectionNote_remove_car').textContent, /raw110.*pending/);
  assert.equal(doc.getElementById('cloudColor').value, 'effect', '临时RGB着色已恢复到effect模式');
  assert.match(doc.getElementById('targetDetectionRemovalRisk').textContent, /mixed.*目标和背景.*整颗去除/);
  console.log(JSON.stringify({ synthetic_fixture_only: true, native_canvas: true, real_viewer_draw3d: true, namespace_separation: true, shadow_violet: true, explicit_mixed_shadow_effect: true, independent_pixel_shadow_effect: true, body_pixel_role_preserved: true, only_marked_pixels_changed: true, mixed_role_preserved: true, no_role4_to_shadow_inference: true, whole_row_removal_risk_visible: true, target_u_lightgrey: true, original_semantic_map_unchanged: true, missing_frame_pending: true, tests: 'passed' }));
}
module.exports = { fakeDocument };
if (require.main === module) test().catch(error => { console.error(error); process.exitCode = 1; });
