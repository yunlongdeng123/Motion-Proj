// Local CPU only: actual PNG + real viewer Canvas; no browser, server, or network.
'use strict';
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict'), crypto = require('node:crypto');
const { createCanvas, loadImage } = require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const { fakeDocument } = require('../v43_unknown_detection_20261028/test_detection_canvas.cjs');
const base = path.resolve(__dirname, '../..'), output = path.join(base, 'outputs/GS_Scene_Diagnostics_20261027');
const ids = ['v43_actor_10000', 'v43_motion_no_edit', 'v43_motion_delete_target', 'v43_motion_translate_lateral_1m', 'v43_motion_turn_left15'];
const scenes = ['4295449061847708198_3769_000_3789_000', '473735159277431842_630_095_650_095'];
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function testScene(context, sceneNumber) {
  const folder = path.join(output, context), doc = fakeDocument();
  doc.getElementById('cloudColor').value = 'branch';
  const sb = { console, document: doc, Float32Array, Float64Array, Uint8Array, Uint8ClampedArray, Uint16Array, Int32Array, ArrayBuffer, TextDecoder, Map, Set, Blob, Response, DecompressionStream, atob, URL, location: { hash: '' }, setTimeout };
  sb.window = sb; sb.addEventListener = () => {};
  vm.createContext(sb);
  for (const file of ['scene.js', 'cloud.js', 'identity.js', 'edit_geometry.js', 'foreign_clouds.js', 'v43_groups.js']) vm.runInContext(fs.readFileSync(path.join(folder, file), 'utf8'), sb, { filename: file });
  let source = fs.readFileSync(path.join(output, 'viewer.js'), 'utf8').replace(/init\(\);\s*$/, '');
  vm.runInContext(source, sb, { filename: 'actual-viewer.js' });
  const a = sb.SceneViewer, result = { context, sceneNumber, groups: [], nativePngsDecoded: 0, offlineWrapperHashesVerified: 0, canvasSamples: [] };
  a.S.clouds = {};
  for (const [key, value] of Object.entries({ ...sb.CLOUD_DATA, ...sb.FOREIGN_CLOUDS })) a.S.clouds[key] = { ...value, data: await a.unpack(value.data) };
  assert(a.S.clouds['v43:10000'].sample > 0);
  for (const id of ids) {
    const group = a.D.groups.find(g => g.id === id); assert(group, id + ' must be installed');
    const raws = Object.keys(group.frames).map(Number).sort((x, y) => x - y);
    assert.equal(raws.length, id === 'v43_actor_10000' ? 49 : 31);
    const geometry = sb.EDIT_GEOMETRY.groups[id]; assert.deepEqual(Object.keys(geometry).map(Number).sort((x, y) => x - y), raws);
    const evidence = JSON.parse(fs.readFileSync(path.join(folder, group.summary), 'utf8'));
    if (id.startsWith('v43_motion_')) assert(evidence.geometry_request_max_abs < 1e-7);
    for (const raw of raws) {
      const frame = group.frames[raw], filename = path.join(folder, frame.channels.rgb), bytes = fs.readFileSync(filename);
      const image = await loadImage(bytes); assert.equal(image.width, 480); assert.equal(image.height, 320); result.nativePngsDecoded++;
      const wrapper = fs.readFileSync(filename + '.js', 'utf8'), embedded = wrapper.match(/data:image\/png;base64,([A-Za-z0-9+/=]+)/);
      assert(embedded); assert.equal(digest(Buffer.from(embedded[1], 'base64')), digest(bytes)); result.offlineWrapperHashesVerified++;
      assert(geometry[raw].supported, id + ' / raw' + raw + ' geometry');
      assert(geometry[raw].camera.flat().every(Number.isFinite));
      for (const object of geometry[raw].objects) { assert.equal(object.cloud, 'v43:10000'); assert(object.transform.flat().every(Number.isFinite)); }
    }
    const raw = sceneNumber === 2 ? 181 : 45;
    assert(group.frames[raw]); a.S.raw = raw; a.S.group = id; a.S.cameraView = true; a.S.cameraReference = 'current'; a.draw3D();
    const instances = a.geometryInstances(), hasActor = instances.some(item => item.key === 'target' && item.cloud === 'v43:10000');
    assert.equal(hasActor, id !== 'v43_motion_delete_target');
    assert(a.S.projection.length > 0); assert(a.S.projection.every(point => Number.isFinite(point.x) && Number.isFinite(point.y)));
    const boxes = a.currentBoxes(); assert(boxes.some(box => !box.target), 'original other-vehicle boxes remain');
    const sheet = createCanvas(1320, 600), x = sheet.getContext('2d');
    x.fillStyle = '#0c1727'; x.fillRect(0, 0, sheet.width, sheet.height); x.fillStyle = '#e7f3ff'; x.font = '17px sans-serif';
    x.fillText('Scene ' + sceneNumber + ' / ' + id + ' / raw' + raw + ' / native RGB + actual viewer geometry', 10, 25);
    x.drawImage(await loadImage(path.join(folder, group.frames[raw].channels.rgb)), 0, 45, 480, 320);
    x.drawImage(doc.getElementById('sceneCanvas').nativeCanvas, 480, 40);
    fs.writeFileSync(path.join(__dirname, 'scene' + sceneNumber + '_' + id + '_canvas.png'), sheet.toBuffer('image/png'));
    result.canvasSamples.push({ group: id, raw, projectionPoints: a.S.projection.length, targetActorDisplayed: hasActor, otherVehicleBoxes: boxes.filter(box => !box.target).length });
    result.groups.push({ id, frames: raws.length, geometryFrames: raws.length, geometryRequestMaxAbs: evidence.geometry_request_max_abs ?? null });
  }
  // Exercise the installed default snippet with actual group data, without init/network.
  const html = fs.readFileSync(path.join(folder, 'index.html'), 'utf8');
  const match = html.match(/<!-- scene-v43-default-start --><script>([\s\S]*?)<\/script><!-- scene-v43-default-end -->/); assert(match);
  for (const hash of ['', '#workspace', '#v43SceneEntry', '#workspace?group=motion_no_edit&raw=' + (sceneNumber === 2 ? 181 : 45)]) {
    const selected = [], state = { raw: 999, group: 'sentinel' }, test = { URLSearchParams, location: { hash }, window: { SceneViewer: { D: a.D, S: state, selectGroup(id) { selected.push(id); state.group = id; } } } };
    vm.runInNewContext(match[1], test);
    if (!hash.includes('group=')) { assert.deepEqual(selected, ['v43_motion_no_edit']); assert.equal(state.raw, sceneNumber === 2 ? 181 : 45); assert.equal(state.channel, 'rgb'); }
    else { assert.equal(selected.length, 0); assert.equal(state.raw, 999); }
  }
  result.defaultSelectionActualV43 = true; result.explicitHashPreserved = true;
  return result;
}

(async () => {
  const results = [];
  for (let i = 0; i < scenes.length; i++) results.push(await testScene(scenes[i], i + 2));
  const receipt = { schema: 'scene-entry-cpu-canvas-qa-v1', browserQA: false, sshUsed: false, actualSavedRgbOnly: true, realViewerCanvas: true, scenes: results };
  fs.writeFileSync(path.join(__dirname, 'canvas_qa.json'), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify(receipt, null, 2));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
