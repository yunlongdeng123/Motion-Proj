import {spawn} from 'node:child_process';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?:([A-Za-z]:))/, '$1'));
const qaRoot = root;
const index = process.argv[2] ?? path.join(qaRoot, 'stage', 'index.html');
const fixture = index.replaceAll('\\','/').includes('/gaussian_transfer_qa/');
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile = path.join(qaRoot, 'chrome_profile');
await mkdir(profile,{recursive:true});
const child = spawn(chrome,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--allow-file-access-from-files','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],
  {windowsHide:true,stdio:['ignore','ignore','pipe']});
let stderr='';
const endpoint = await new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>reject(new Error('Chrome启动超时: '+stderr.slice(-1500))),15000);
  child.stderr.on('data',data=>{stderr+=data.toString();const match=stderr.match(/DevTools listening on (ws:\/\/\S+)/);if(match){clearTimeout(timeout);resolve(match[1]);}});
  child.on('error',reject);
});
const socket = new WebSocket(endpoint);
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
let id=0;const pending=new Map();const errors=[];
socket.addEventListener('message',event=>{
  const packet=JSON.parse(event.data);
  if(packet.id){const item=pending.get(packet.id);if(item){clearTimeout(item.timeout);pending.delete(packet.id);packet.error?item.reject(new Error(JSON.stringify(packet.error))):item.resolve(packet.result);}}
  else if(packet.method==='Runtime.exceptionThrown') errors.push(packet.params.exceptionDetails.exception?.description??packet.params.exceptionDetails.text);
});
function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const request=++id;const timeout=setTimeout(()=>{pending.delete(request);reject(new Error('CDP超时 '+method));},20000);pending.set(request,{resolve,reject,timeout});socket.send(JSON.stringify({id:request,method,params,...(sessionId?{sessionId}:{})}));});}
const {targetId}=await call('Target.createTarget',{url:'about:blank'});
const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
const page=(method,params={})=>call(method,params,sessionId);
async function evaluate(expression){const result=await page('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw new Error(result.exceptionDetails.exception?.description??result.exceptionDetails.text);return result.result.value;}
const change=(selector,value)=>evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));return e.value})()`);
const images=()=>evaluate(`Promise.all([...document.querySelectorAll('img')].filter(e=>!e.closest('[hidden]')).map(e=>e.complete?Promise.resolve(e.naturalWidth>0):new Promise(r=>{e.onload=()=>r(true);e.onerror=()=>r(false)}))).then(v=>({all:v.every(Boolean),failed:[...document.querySelectorAll('img')].filter(e=>!e.closest('[hidden]')&&e.complete&&!e.naturalWidth).map(e=>e.getAttribute('src'))}))`);
function assert(value,message){if(!value)throw new Error(message);}


let result;
try {
 await page('Runtime.enable');await page('Page.enable');
 await page('Emulation.setDeviceMetricsOverride',{width:1500,height:1050,deviceScaleFactor:1,mobile:false});
 await page('Page.navigate',{url:pathToFileURL(index).href});
 for(let i=0;i<100;i++){if(await evaluate(`document.readyState==='complete'&&Boolean(document.getElementById('source-quality-audit'))`))break;await new Promise(r=>setTimeout(r,100));}
 const initial=await evaluate(`({records:C.models.length,old:C.models.filter(m=>m.batch!=='gaussian_transfer').length,rows:C.rows.length,cases:JSON.parse(document.querySelector('#source-quality-audit-data').textContent).cases.length,zoom:document.querySelector('#sq-zoom').value,raw:document.querySelector('#sq-raw').value,patch:document.querySelector('#sq-patch').value,native_table:document.querySelectorAll('#sq-metrics tbody tr').length,refit_table:document.querySelectorAll('#sq-refit-table tbody tr').length,refit_gate:JSON.parse(document.querySelector('#source-quality-audit-data').textContent).summary.quality_gate_passed})`);
 assert(initial.records===58&&initial.old===55&&initial.rows===744&&initial.cases===9&&initial.native_table===6&&initial.refit_table===5,'catalog and actual cases');
 assert(initial.zoom==='1'&&initial.raw==='110'&&initial.patch==='patch_00'&&initial.refit_gate===false,'native default and failed quality gate');
 let views=0;
 for(const raw of ['86','98','110','137','173'])for(const zoom of ['1','2','4']){
  await change('#sq-raw',raw);await change('#sq-zoom',zoom);assert((await images()).all,'missing actual PNG');
  const measured=await evaluate(`(()=>{const im=[...document.querySelectorAll('#sq-images img')];return im.map(e=>({src:e.getAttribute('src'),natural:[e.naturalWidth,e.naturalHeight],display:[e.getBoundingClientRect().width,e.getBoundingClientRect().height],rendering:getComputedStyle(e).imageRendering,maskMode:getComputedStyle(e).maskMode,mask:getComputedStyle(e).maskImage}))})()`);
  assert(measured.length===3&&measured[2].src.includes('/source_rgb_refit_qualified_r2/refit/'),'only final qualified r2 candidate');
  for(const m of measured){assert(m.display[0]===m.natural[0]*Number(zoom)&&m.display[1]===m.natural[1]*Number(zoom),'pixel exact scale');assert(m.rendering==='pixelated'&&m.maskMode==='luminance'&&m.mask!=='none','nearest and core mask');}
  await evaluate(`document.querySelector('#sq-core').click()`);assert(await evaluate(`[...document.querySelectorAll('#sq-images img')].every(e=>getComputedStyle(e).maskImage==='none')`),'context toggle');await evaluate(`document.querySelector('#sq-core').click()`);views++;
 }
 for(const sigma of ['1.2','1.6666666666666667','3']){await change('#sq-band',sigma);assert(await evaluate(`document.querySelectorAll('#sq-refit-table tbody tr').length===5`),'three actual spectral bands');}
 await change('#sq-band','1.6666666666666667');await change('#sq-raw','110');await change('#sq-zoom','1');await images();
 await evaluate(`document.querySelector('#sq-entry').click()`);await images();
 const source100=await evaluate(`({images:[...document.querySelectorAll('#sq-images img')].map(e=>({src:e.getAttribute('src'),natural:[e.naturalWidth,e.naturalHeight],width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})),metrics:document.querySelector('#sq-refit-table').innerText,verdict:document.querySelector('#sq-verdict').textContent,resources:document.querySelector('#sq-resources').innerText,children:document.querySelectorAll('#sq-images figure').length})`);
 assert(source100.children===3&&source100.images.every(e=>e.natural[0]===298&&e.natural[1]===79&&e.width===298&&e.height===79),'native source00/110');
 assert(source100.resources.includes('1663')&&source100.resources.includes('1624')&&source100.resources.includes('412'),'authoritative cumulative resources');
 const capture=async name=>{const shot=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(path.join(root,name),Buffer.from(shot.data,'base64'));};
 await capture('qa_source_quality_100.png');
 await change('#sq-zoom','4');await images();await evaluate(`document.querySelectorAll('.sq-port').forEach(e=>e.scrollLeft=(e.scrollWidth-e.clientWidth)/2)`);await capture('qa_source_quality_400.png');
 await change('#sq-raw','86');await images();await evaluate(`document.querySelectorAll('.sq-port').forEach(e=>e.scrollLeft=(e.scrollWidth-e.clientWidth)/2)`);await capture('qa_source_quality_query86_400.png');
 await evaluate(`document.querySelector('#sq-summary').scrollIntoView({block:'start'})`);await capture('qa_source_quality_findings.png');
 await page('Emulation.setDeviceMetricsOverride',{width:390,height:1050,deviceScaleFactor:1,mobile:true});await change('#sq-raw','110');await change('#sq-zoom','1');await images();await evaluate(`document.querySelector('#source-quality-audit').scrollIntoView({block:'start'})`);
 const mobile=await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})`);assert(mobile.scrollWidth<=mobile.width+1,'mobile body overflow');await capture('qa_source_quality_mobile.png');
 await change('#sq-zoom','4');assert(await evaluate(`document.documentElement.scrollWidth<=innerWidth+1`),'mobile 400% stays inside scroll containers');
 await change('#sq-patch','patch_02');assert(await evaluate(`document.querySelectorAll('#sq-images img').length===2&&document.querySelectorAll('#sq-raw option').length===2`),'no candidate placeholder for source02');
 await change('#batch','residual');assert(await evaluate(`Boolean(document.querySelector('#source-quality-audit').closest('[hidden]'))&&!document.querySelector('#metrics').parentElement.hidden`),'legacy group restored');
 assert(errors.length===0,JSON.stringify(errors));result={passed:true,fixture:false,index,initial,views,source100,mobile,JavaScript_errors:errors};console.log(JSON.stringify(result));
}catch(error){result={passed:false,error:String(error),JavaScript_errors:errors};console.error(JSON.stringify(result));process.exitCode=1;}
finally {await writeFile(path.join(root,'qa_browser.json'),JSON.stringify(result,null,2));await call('Browser.close').catch(()=>{});socket.close();child.kill();}
