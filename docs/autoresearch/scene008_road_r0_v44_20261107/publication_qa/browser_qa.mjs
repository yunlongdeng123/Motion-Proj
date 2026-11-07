import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve('outputs/scene008_road_r0_v44');
const qa = path.join(root, 'qa');
fs.mkdirSync(qa, {recursive: true});
const tabs = await (await fetch('http://127.0.0.1:9328/json/list')).json();
const tab = tabs.find(t => t.type === 'page' && t.url.includes('scene008_road_r0_v44'));
if (!tab) throw new Error('No owned report tab');
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
let counter = 0;
const pending = new Map();
const exceptions = [];
ws.onmessage = e => {
  const obj = JSON.parse(e.data);
  if (obj.id) {
    const handle = pending.get(obj.id);
    pending.delete(obj.id);
    if (obj.error) handle.reject(new Error(JSON.stringify(obj.error)));
    else handle.resolve(obj.result);
  } else if (obj.method === 'Runtime.exceptionThrown') exceptions.push(obj.params);
};
function call(method, params = {}) {
  const id = ++counter;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const res = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (res.exceptionDetails) throw new Error(JSON.stringify(res.exceptionDetails));
  return res.result.value;
}
await call('Runtime.enable');
await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride', {width: 1600, height: 1050, deviceScaleFactor: 1, mobile: false});
const cases = [];
for (const raw of [89, 105, 113, 130, 137]) {
  for (const state of ['keep', 'delete']) {
    const result = await evaluate(`(async()=>{
      document.getElementById('raw').value=${raw};
      document.getElementById('state').value=${JSON.stringify(state)};
      document.getElementById('showWide').checked=true;
      document.getElementById('showObserved').checked=true;
      await draw();
      await Promise.all(Array.from(document.querySelectorAll('#board img')).map(x=>x.decode()));
      return {rawLabel:document.getElementById('rawLabel').textContent,
        status:document.getElementById('loadStatus').textContent,
        images:Array.from(document.querySelectorAll('#board figure')).map(f=>({
          id:f.id,hidden:f.hidden,title:f.querySelector('figcaption').textContent,
          src:f.querySelector('img').getAttribute('src'),
          width:f.querySelector('img').naturalWidth,height:f.querySelector('img').naturalHeight}))};
    })()`);
    if (result.rawLabel !== String(raw) || !result.status.includes('真实PNG')) throw new Error(JSON.stringify(result));
    for (const image of result.images) {
      if (image.hidden || image.width !== 480 || image.height !== 320) throw new Error(JSON.stringify(image));
      if (!image.src.includes('/' + String(raw).padStart(3, '0') + '/')) throw new Error(image.src);
      if (image.id !== 'panel_observed' && !image.src.endsWith('_' + state + '.png')) throw new Error(image.src);
    }
    cases.push({raw, state, ...result});
  }
}
await evaluate(`(async()=>{document.getElementById('raw').value=137;document.getElementById('state').value='delete';document.getElementById('showWide').checked=false;document.getElementById('showObserved').checked=false;await draw();document.getElementById('viewer').scrollIntoView();return true})()`);
const visibility = await evaluate(`({wide:document.getElementById('panel_wide').hidden,observed:document.getElementById('panel_observed').hidden,cols:document.getElementById('board').style.getPropertyValue('--cols')})`);
if (!visibility.wide || !visibility.observed || visibility.cols !== '3') throw new Error(JSON.stringify(visibility));
for (const [target, filename] of [['viewer', 'viewer_raw137_delete.png'], ['body-risk', 'body_ground_risk.png']]) {
  await evaluate(`document.getElementById(${JSON.stringify(target)}).scrollIntoView();true`);
  const shot = await call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  fs.writeFileSync(path.join(qa, filename), Buffer.from(shot.data, 'base64'));
}
await call('Emulation.setDeviceMetricsOverride', {width: 430, height: 900, deviceScaleFactor: 1, mobile: true});
await evaluate(`document.getElementById('viewer').scrollIntoView();true`);
const mobile = await evaluate(`({viewport:innerWidth,scroll:document.documentElement.scrollWidth,columns:getComputedStyle(document.getElementById('board')).gridTemplateColumns})`);
if (mobile.scroll > mobile.viewport) throw new Error('Mobile horizontal overflow: ' + JSON.stringify(mobile));
const shot = await call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
fs.writeFileSync(path.join(qa, 'viewer_mobile.png'), Buffer.from(shot.data, 'base64'));
if (exceptions.length) throw new Error(JSON.stringify(exceptions));
fs.writeFileSync(path.join(qa, 'browser_qa.json'), JSON.stringify({cases, visibility, mobile, exceptions, scientific_inputs_changed: false}, null, 2));
console.log(JSON.stringify({cases: cases.length, visibility, mobile, runtimeExceptions: exceptions.length}));
await call('Browser.close');
ws.close();
