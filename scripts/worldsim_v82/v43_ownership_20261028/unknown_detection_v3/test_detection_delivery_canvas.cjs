// 实际已交付PNG、证据、原始蓝框与checkpoint采样的Canvas验收；不是浏览器QA。
'use strict';
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm'), assert = require('node:assert/strict');
const { createCanvas, loadImage: nativeLoadImage } = require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const { fakeDocument } = require('./test_detection_canvas.cjs');
const base = path.resolve(__dirname, '../..'), output = path.join(base, 'outputs/GS_Scene_Diagnostics_20261027');
const folder = path.join(output, '14705303724557273004_3105_000_3125_000');
const qa = path.join(folder, 'target_detection');
const delivery = JSON.parse(fs.readFileSync(path.join(__dirname, 'delivery.json'), 'utf8'));
const doc = fakeDocument();
doc.getElementById('cloudColor').value = 'branch';
doc.getElementById('pointSize').value = '2.5';
const sb = { console, document: doc, Float32Array, Float64Array, Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array, Int32Array, ArrayBuffer, Map, Set, Blob, Response, DecompressionStream, atob, URL, setTimeout, location: { hash: '' } };
sb.window = sb; sb.addEventListener = () => {};
vm.createContext(sb);
function run(file) { return vm.runInContext(fs.readFileSync(path.resolve(folder, file), 'utf8'), sb, { filename: file }); }
for (const file of ['scene.js', 'cloud.js', 'identity.js', 'edit_geometry.js', 'foreign_clouds.js', 'v43_groups.js']) run(file);
const historicalIds = sb.SCENE_DATA.groups.map(group => group.id);
run('target_detection_data.js');
const source = fs.readFileSync(path.join(output, 'viewer.js'), 'utf8').replace(/;init\(\);\s*$/, ';');
vm.runInContext(source, sb);
const a = sb.SceneViewer;
sb.nativeLoadScript = async relative => run(relative);
sb.nativeLoadPng = async relative => nativeLoadImage(path.resolve(folder, relative));
vm.runInContext('loadScript=nativeLoadScript;loadImage=nativeLoadPng;', sb);

async function main() {
  a.S.clouds = {};
  for (const [key, meta] of Object.entries({ ...sb.CLOUD_DATA, [delivery.actor.cloud]: sb.FOREIGN_CLOUDS[delivery.actor.cloud] })) a.S.clouds[key] = { ...meta, data: await a.unpack(meta.data) };
  const semanticMapBefore = vm.runInContext('JSON.stringify(GPU_SEMANTIC_COLORS)', sb);
  const defaultGroup = a.S.group;
  vm.runInContext(fs.readFileSync(path.join(output, 'scene_target_detection.js'), 'utf8'), sb);
  assert.equal(a.S.group, defaultGroup); assert.equal(defaultGroup, 'reconstruction_30000');
  assert(historicalIds.every(id => a.D.groups.some(group => group.id === id)));
  assert.equal(sb.TargetDetection.manifest.quality_accepted, false);
  assert.match(doc.getElementById('targetDetectionQuality').textContent, /未通过/);
  const mount = sb.TargetDetection;
  vm.runInContext(fs.readFileSync(path.join(output, 'scene_target_detection.js'), 'utf8'), sb);
  assert.equal(sb.TargetDetection, mount, '模板/局部脚本并存只挂载一次');

  const cases = ['baseline', 'remove_car', 'remove_u', 'remove_car_u'], manifest = sb.TARGET_DETECTION_DELIVERY;
  const installedRaws = Object.keys(manifest.frames).map(Number);
  assert.equal(installedRaws.length, 49);
  for (const key of cases) {
    assert.equal(installedRaws.filter(raw => manifest.frames[raw].images[key]).length, 3);
    assert.equal(installedRaws.filter(raw => !manifest.frames[raw].images[key]).length, 46);
  }
  const checks = [], fullSheet = createCanvas(1320, 2720), sx = fullSheet.getContext('2d');
  sx.fillStyle = '#08111c'; sx.fillRect(0, 0, fullSheet.width, fullSheet.height);
  sx.font = '18px Microsoft YaHei'; sx.fillStyle = '#f9c3ae';
  sx.fillText('raw113 · 实际原生RGB / 同帧真实Gaussian中心 · 未通过：整行剥离误伤背景', 12, 26);
  for (const raw of [98, 113, 114]) {
    a.S.raw = raw;
    const evidence = await mount.evidenceFor(raw);
    const bodyEffect = Array.from(evidence.pixel_shadow_effect).reduce((sum, flag, i) => sum + (flag && evidence.pixel_role[i] === 1 ? 1 : 0), 0);
    assert.equal(bodyEffect, { 98: 5, 113: 1, 114: 7 }[raw]);
    const mixedShadow = Array.from(evidence.static_role).reduce((sum, role, i) => sum + (role === 4 && evidence.shadow_effect[i] ? 1 : 0), 0);
    assert.equal(mixedShadow, { 98: 1263, 113: 964, 114: 868 }[raw]);
    const observed = await nativeLoadImage(path.join(folder, a.D.gt[raw]));
    const baseCanvas = createCanvas(480, 320), bx = baseCanvas.getContext('2d'); bx.drawImage(observed, 0, 0);
    const basePixels = bx.getImageData(0, 0, 480, 320).data;
    const withoutEffect = sb.TargetDetectionCore.overlayRGBA(basePixels, evidence.pixel_role, 480, 320);
    const withEffect = sb.TargetDetectionCore.overlayRGBA(basePixels, evidence.pixel_role, 480, 320, .48, evidence.pixel_shadow_effect);
    for (let i = 0; i < 480 * 320; i++) if (!evidence.pixel_shadow_effect[i]) for (let k = 0; k < 4; k++) assert.equal(withEffect[4 * i + k], withoutEffect[4 * i + k]);
    const overlapPixel = evidence.pixel_shadow_effect.findIndex((flag, i) => flag && evidence.pixel_role[i] === 1);
    assert(overlapPixel >= 0);
    a.S.pixel = [overlapPixel % 480, Math.floor(overlapPixel / 480)];
    const frame = manifest.frames[raw], expectedBlue = sb.IDENTITY_AUDIT.frames[raw].vehicle_boxes.filter(box => !box.target);
    const expectedVisibleBlue = expectedBlue.filter(box => box.in_front_image && box.uv_sensor).length;
    for (let index = 0; index < cases.length; index++) {
      const key = cases[index];
      await mount.activate(key);
      const id = 'v43_target_detection_' + key, q = a.editState(), roles = manifest.removal_roles[key];
      const expectedRemoved = Array.from(evidence.ids).filter((row, i) => roles.includes(evidence.static_role[i]));
      assert.deepEqual(Array.from(q.static_suppressed_rows), expectedRemoved);
      const metric = frame.metrics.find(metric => metric.mode === key);
      assert.equal(expectedRemoved.length, metric.removed_static_rows);
      const instances = a.geometryInstances(), actorRetained = instances.some(instance => instance.cloud === delivery.actor.cloud);
      assert.equal(actorRetained, !roles.includes(1));
      assert.equal(metric.removed_actor_rows, actorRetained ? 0 : a.S.clouds[delivery.actor.cloud].total);
      const blue = a.currentBoxes().filter(box => !box.target);
      assert.equal(blue.length, expectedBlue.length);
      assert.equal(JSON.stringify(blue), JSON.stringify(expectedBlue));
      assert(a.currentBoxes().some(box => box.target && box.label.includes('参考')));
      const suppressed = new Set(expectedRemoved);
      for (const point of a.S.projection) if (point.cloud === 'static') assert(!suppressed.has(a.S.clouds.static.data.ids[point.j]));
      assert.equal(a.S.projection.some(point => point.cloud === delivery.actor.cloud), actorRetained);
      assert.equal(frame.render_receipts[key].renderer, 'native_gaussian');
      assert.equal(frame.render_receipts[key].source_raw, raw);
      assert.equal(frame.render_receipts[key].source_checkpoint_unchanged, true);
      assert.equal(frame.render_receipts[key].training_updates, 0);
      const installedPng = path.join(folder, frame.images[key]), originalPng = delivery.frames[raw].images[key];
      assert(fs.readFileSync(installedPng).equals(fs.readFileSync(originalPng)), '原PNG字节保持一致');
      const png = await nativeLoadImage(installedPng); assert.equal(png.width, 480); assert.equal(png.height, 320);
      const wrapperBox = { window: { IMAGE_DATA: {} } }; vm.createContext(wrapperBox);
      vm.runInContext(fs.readFileSync(installedPng + '.js', 'utf8'), wrapperBox);
      const wrappedPng = await nativeLoadImage(wrapperBox.window.IMAGE_DATA[frame.images[key]]); assert.equal(wrappedPng.width, 480);
      const overlayContext = doc.getElementById('targetDetectionCanvas_overlay').getContext('2d'), stroke = overlayContext.stroke, colors = [];
      overlayContext.stroke = function (...args) { colors.push(this.strokeStyle); return stroke.apply(this, args); };
      try { await mount.drawPanels(raw, a.S.epoch); } finally { overlayContext.stroke = stroke; }
      assert.equal(colors.filter(color => color === '#59a9ff').length, expectedVisibleBlue);
      assert.match(doc.getElementById('targetDetectionPixel').textContent, /pixel_role=1.*pixel_shadow_effect=true/);
      assert.match(doc.getElementById('targetDetectionNote_' + key).textContent, key === 'baseline' ? /complete/ : /complete.*未通过/);
      checks.push({ raw, case: key, native_rgb_verified: true, receipt_verified: true, removed_static_rows: expectedRemoved.length, removed_actor_rows: metric.removed_actor_rows, displayed_centers: a.S.projection.length, original_other_boxes_3d: blue.length, original_other_boxes_2d: expectedVisibleBlue, independent_pixel_shadow_pixels: evidence.pixel_shadow_effect.reduce((sum, value) => sum + value, 0), pixel_effect_on_body: bodyEffect, mixed_shadow_rows: mixedShadow });
      if (raw === 113) {
        const y = 360 + index * 590;
        sx.fillStyle = '#f9c3ae'; sx.fillText(key + ' · native RGB / Gaussian centers · failed acceptance', 12, y + 20);
        sx.drawImage(png, 0, y + 30, 480, 320);
        sx.drawImage(doc.getElementById('sceneCanvas').nativeCanvas, 480, y + 30);
        if (index === 0) { sx.drawImage(observed, 0, 36); sx.drawImage(doc.getElementById('targetDetectionCanvas_overlay').nativeCanvas, 480, 36); }
      }
    }
  }
  a.S.raw = 109; await mount.activate('remove_u');
  assert.equal(a.S.raw, 109, 'pending raw不替换为已渲染邻帧');
  assert.equal(a.S.projection.length, 0);
  assert.match(doc.getElementById('targetDetectionNote_remove_u').textContent, /raw109.*pending/);
  assert(!manifest.frames[109].images.remove_u);
  assert.equal(vm.runInContext('JSON.stringify(GPU_SEMANTIC_COLORS)', sb), semanticMapBefore);
  // 没有delivery的独立场景只载入通用core，不挂载UI。
  const optional = { window: {}, Uint8ClampedArray }; vm.createContext(optional);
  vm.runInContext(fs.readFileSync(path.join(output, 'scene_target_detection.js'), 'utf8'), optional);
  assert.equal(optional.window.TargetDetection, undefined);
  const otherScenes = [];
  for (const scene of JSON.parse(fs.readFileSync(path.join(output, 'scenes.json'), 'utf8')).filter(scene => scene.context !== manifest.context)) {
    const other = { window: {}, document: { createElement() { throw Error('无delivery不应挂载UI'); } } };
    vm.createContext(other);
    vm.runInContext(fs.readFileSync(path.join(output, scene.context, 'scene.js'), 'utf8') + '\nconst S={};const D=window.SCENE_DATA;', other);
    vm.runInContext(fs.readFileSync(path.join(output, 'scene_target_detection.js'), 'utf8'), other);
    assert.equal(other.window.TargetDetection, undefined); otherScenes.push(scene.context);
  }
  assert(fs.readFileSync(path.join(__dirname, 'scene_target_detection.js')).equals(fs.readFileSync(path.join(base, 'work/persistent_world_20261027/scene_target_detection.js'))));
  assert(fs.readFileSync(path.join(__dirname, 'scene_target_detection.js')).equals(fs.readFileSync(path.join(output, 'scene_target_detection.js'))));
  const html = fs.readFileSync(path.join(folder, 'index.html'), 'utf8');
  assert.equal((html.match(/target_detection_data\.js\?/g) || []).length, 1);
  assert.equal((html.match(/\.\.\/scene_target_detection\.js\?/g) || []).length, 1);
  fs.writeFileSync(path.join(qa, 'raw113_native_geometry_canvas_review.png'), fullSheet.toBuffer('image/png'));
  const report = { schema: 'target-detection-ui-validation-v1', real_canvas_runtime: true, actual_delivery: 'native_target_contact_smoke_r1', browserQA: false, all_native_receipts: checks.length, frame_population: 49, complete_per_case: 3, pending_per_case: 46, historical_groups_retained: historicalIds.length, default_group_preserved: defaultGroup, original_19class_map_unchanged: true, original_all_vehicle_boxes_retained: true, independent_pixel_and_row_effects: true, all_suppressed_static_ids_match_native_metrics: true, optional_mount_without_data_safe: true, other_scenes_without_data_safe: otherScenes, duplicate_mount_safe: true, canonical_template_runtime_synced: true, quality_accepted: false, human_verdict: null, checks };
  fs.writeFileSync(path.join(qa, 'ui_validation.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, checks: checks.length }));
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
