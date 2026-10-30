const fs=require('fs'),path=require('path'),assert=require('assert/strict'),{pathToFileURL}=require('url');
const {chromium}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const out=path.resolve('outputs/V43_Ownership_20261027/gpu_v43/sim_m1m2_20261030');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1500,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(out,'architecture.svg')).href);
 const overflow=await page.evaluate(()=>[...document.querySelectorAll('text')].map(t=>{
   const b=t.getBBox(),x=+t.getAttribute('x'),y=+t.getAttribute('y');
   const r=[...document.querySelectorAll('rect')].map(r=>r.getBBox()).filter(r=>x>=r.x&&x<=r.x+r.width&&y>=r.y&&y<=r.y+r.height).sort((a,b)=>a.width*a.height-b.width*b.height)[0];
   return r&&(b.x<r.x+2||b.x+b.width>r.x+r.width-2)?{text:t.textContent,width:b.width,box:{x:r.x,w:r.width}}:null;
 }).filter(Boolean));
 assert.equal(overflow.length,0,JSON.stringify(overflow));
 await page.locator('svg').screenshot({path:path.join(out,'architecture.png')});
 await page.goto(pathToFileURL(path.join(out,'index.html')).href);
 let combinations=0;
 async function check(id,count){
  const imgs=await page.locator(id).evaluate(async el=>{const a=[...el.querySelectorAll('img')];await Promise.all(a.map(i=>i.decode()));return a.map(i=>({src:i.getAttribute('src'),width:i.naturalWidth,height:i.naturalHeight}));});
  assert.equal(imgs.length,count);assert(imgs.every(i=>i.width===480&&i.height===320));combinations++;
 }
 for(const f of ['000000','000008','000015'])for(const e of ['none','delete','shift','rotate'])for(const c of ['final','white','alpha']){
  await page.selectOption('#frame',f);await page.selectOption('#edit',e);await page.selectOption('#channel',c);await check('#main',6);
 }
 for(const e of ['none','delete','shift','rotate']){
  await page.selectOption('#fit_edit',e);await check('#fit',3);
  for(const s of ['main','fit']){await page.selectOption('#hold_edit',e);await page.selectOption('#hold_set',s);await check('#heldout',s==='main'?6:3);}
 }
 await page.selectOption('#frame','000008');await page.selectOption('#edit','delete');await page.selectOption('#channel','final');await check('#main',6);
 await page.locator('#main').screenshot({path:path.join(out,'preview.png')});
 await page.selectOption('#fit_edit','delete');await check('#fit',3);await page.locator('#fit').screenshot({path:path.join(out,'refit_preview.png')});
 const links=await page.locator('a[href]').evaluateAll(a=>a.map(a=>a.getAttribute('href')).filter(h=>!h.startsWith('http')&&!h.startsWith('#')));
 for(const href of links)assert(fs.existsSync(path.resolve(out,href.split('#')[0])),href);
 const scroll=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 assert(!scroll);assert.equal(errors.length,0);
 const assets=JSON.parse(fs.readFileSync(path.join(out,'assets.json')));
 assert(Object.values(assets).every(p=>fs.existsSync(path.join(out,p))));
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify({passed:true,actual_browser_combinations:combinations,assets:Object.keys(assets).length,links:links.length,svg_text_overflow:overflow,js_errors:errors,horizontal_overflow:scroll},null,2));
 await browser.close();console.log(JSON.stringify({passed:true,combinations,assets:Object.keys(assets).length}));
})().catch(e=>{console.error(e);process.exitCode=1;});
