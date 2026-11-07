import fs from 'node:fs';
import path from 'node:path';
const base=path.resolve('outputs/scene008_road_seam_contact');
const qa=path.join(base,'qa');fs.mkdirSync(qa,{recursive:true});
const tabs=await(await fetch('http://127.0.0.1:9329/json/list')).json();
const tab=tabs.find(x=>x.type==='page'&&x.url.includes('scene008_road_seam_contact'));
if(!tab)throw Error('Owned report tab missing');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});
const pending=new Map(),exceptions=[];let seq=0;
ws.onmessage=e=>{const x=JSON.parse(e.data);if(x.id){const h=pending.get(x.id);pending.delete(x.id);x.error?h.reject(Error(JSON.stringify(x.error))):h.resolve(x.result)}else if(x.method==='Runtime.exceptionThrown')exceptions.push(x.params)};
function call(method,params={}){const id=++seq;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
await call('Runtime.enable');await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
const cases=[];
for(const raw of [89,105,113,128,130,137])for(const state of ['keep','delete']){
 const r=await evaluate(`(async()=>{
  document.getElementById('frame').value=${raw};document.getElementById('variant').value='aa';document.getElementById('secondary').value='material';
  document.querySelector('[data-state="${state}"]').click();
  for(let i=0;i<100;i++){
   const a=document.getElementById('candidateImage'),r=document.getElementById('r0Image'),s=document.getElementById('secondaryImage');
   const secondaryExpected=${[105,113,128,130,137].includes(raw)};
   if(!a.hidden&&a.complete&&a.naturalWidth===480&&!r.hidden&&r.complete&&r.naturalWidth===480&&(secondaryExpected?(!s.hidden&&s.complete&&s.naturalWidth===480):(s.hidden&&!s.getAttribute('src'))))break;
   await new Promise(ok=>setTimeout(ok,20));
  }
  return {label:document.getElementById('rawLabel').textContent,title:document.getElementById('candidateTitle').textContent,
   candidate:{src:document.getElementById('candidateImage').getAttribute('src'),hidden:document.getElementById('candidateImage').hidden,width:document.getElementById('candidateImage').naturalWidth},
   parent:{src:document.getElementById('r0Image').getAttribute('src'),width:document.getElementById('r0Image').naturalWidth},
   secondary:{src:document.getElementById('secondaryImage').getAttribute('src'),hidden:document.getElementById('secondaryImage').hidden,text:document.getElementById('secondaryMissing').textContent},
   caption:document.getElementById('candidateCaption').textContent};
 })()`);
 if(r.candidate.src!==`frames/aa/${raw}_${state}.png`||r.parent.src!==`frames/r0/${raw}_${state}.png`||r.candidate.width!==480||r.candidate.hidden)throw Error(JSON.stringify(r));
 if(raw===89){if(!r.secondary.hidden||r.secondary.src!==null||!r.secondary.text.includes('无该对照'))throw Error('Partial control reused stale image: '+JSON.stringify(r));}
 else if(r.secondary.src!==`frames/material/${raw}_${state}.png`||r.secondary.hidden)throw Error(JSON.stringify(r));
 cases.push({raw,state,...r});
}
const controls=await evaluate(`(async()=>{
 document.getElementById('frame').value=128;document.querySelector('[data-state="delete"]').click();
 document.getElementById('toggleReference').click();await new Promise(r=>setTimeout(r,120));
 const refs=['v44Image','observedImage'].map(id=>({id,src:document.getElementById(id).getAttribute('src'),width:document.getElementById(id).naturalWidth}));
 document.getElementById('candidateImage').click();const zoom=document.getElementById('zoom').open;document.getElementById('closeZoom').click();
 document.getElementById('toggleReference').click();
 return {refs,zoom,closed:!document.getElementById('zoom').open};
})()`);
if(!controls.zoom||!controls.closed||controls.refs.some(x=>x.width!==480))throw Error(JSON.stringify(controls));
await evaluate(`document.getElementById('frame').scrollIntoView({block:'start'});true`);
let shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(qa,'viewer_raw128_delete.png'),Buffer.from(shot.data,'base64'));
await evaluate(`window.scrollTo(0,0);true`);
shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(qa,'separate_outcomes_desktop.png'),Buffer.from(shot.data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:430,height:900,deviceScaleFactor:1,mobile:true});
await evaluate(`window.scrollTo(0,0);true`);
const mobile=await evaluate(`({inner:innerWidth,client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})`);
if(mobile.scroll>mobile.client)throw Error('Mobile horizontal overflow: '+JSON.stringify(mobile));
shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(qa,'outcomes_mobile.png'),Buffer.from(shot.data,'base64'));
if(exceptions.length)throw Error(JSON.stringify(exceptions));
fs.writeFileSync(path.join(qa,'browser_qa.json'),JSON.stringify({cases,controls,mobile,exceptions,scientific_inputs_changed:false,partial_control_missing_state_tested:true},null,2));
console.log(JSON.stringify({cases:cases.length,reference_and_zoom:controls,partial_control_stale_image:false,mobile,exceptions:exceptions.length}));
await call('Browser.close');ws.close();
