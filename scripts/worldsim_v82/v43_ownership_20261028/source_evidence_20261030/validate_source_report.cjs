const fs=require('fs'),path=require('path'),assert=require('assert/strict'),{pathToFileURL,fileURLToPath}=require('url');
const {chromium}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const out=path.resolve('outputs/V43_Ownership_20261027/gpu_v43/source_evidence_20261030');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1500,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(out,'architecture.svg')).href);
 const overflow=await page.evaluate(()=>[...document.querySelectorAll('text')].map(t=>{const b=t.getBBox(),x=+t.getAttribute('x'),y=+t.getAttribute('y');const r=[...document.querySelectorAll('rect')].map(r=>r.getBBox()).filter(r=>x>=r.x&&x<=r.x+r.width&&y>=r.y&&y<=r.y+r.height).sort((a,b)=>a.width*a.height-b.width*b.height)[0];return r&&(b.x<r.x+2||b.x+b.width>r.x+r.width-2)?t.textContent:null}).filter(Boolean));
 assert.equal(overflow.length,0,JSON.stringify(overflow));
 await page.locator('svg').screenshot({path:path.join(out,'architecture.png')});
 await page.goto(pathToFileURL(path.join(out,'index.html')).href);
 let combinations=0;
 for(const frame of ['000000','000008','000015'])for(const channel of ['final','white','C','fill']){
  await page.selectOption('#frame',frame);await page.selectOption('#channel',channel);
  const a=await page.locator('#source img').evaluateAll(async imgs=>{await Promise.all(imgs.map(i=>i.decode()));return imgs.map(i=>[i.naturalWidth,i.naturalHeight])});
  assert.equal(a.length,6);assert(a.every(([w,h])=>w===480&&h===320));combinations++;
 }
 await page.selectOption('#frame','000008');await page.selectOption('#channel','final');
 await page.locator('#source').screenshot({path:path.join(out,'preview.png')});
 let richCombinations=0;
 if(await page.locator('#richFrame').count()){
  for(const frame of ['000000','000008','000015'])for(const edit of ['none','delete','shift','rotate'])for(const channel of ['final','white','C','alpha','fill','E']){
   await page.selectOption('#richFrame',frame);await page.selectOption('#richEdit',edit);await page.selectOption('#richChannel',channel);
   const a=await page.locator('#richGrid img').evaluateAll(async imgs=>{await Promise.all(imgs.map(i=>i.decode()));return imgs.map(i=>[i.naturalWidth,i.naturalHeight])});
   assert.equal(a.length,3);assert(a.every(([w,h])=>w===480&&h===320));richCombinations++;
  }
  await page.selectOption('#richFrame','000008');await page.selectOption('#richEdit','delete');await page.selectOption('#richChannel','final');
  await page.locator('#richGrid').screenshot({path:path.join(out,'rich_preview.png')});
 }
 await page.locator('details').evaluateAll(ds=>ds.forEach(d=>d.open=true));
 await page.locator('img').evaluateAll(imgs=>imgs.forEach(i=>i.loading='eager'));
 let freshModes=0;
 if(await page.locator('.fresh-content').count()){
  for(const white of [true,false]){
   await page.evaluate(w=>mode(w),white);
   const fresh=await page.locator('.fresh-content .gallery img').evaluateAll(async imgs=>{await Promise.all(imgs.map(i=>i.decode()));return imgs.map(i=>[i.naturalWidth,i.naturalHeight])});
   assert.equal(fresh.length,15);assert(fresh.every(([w,h])=>w===480&&h===320));freshModes++;
  }
  await page.locator('#frame-000008').screenshot({path:path.join(out,'fresh_source_preview.png')});
 }
 const badImages=await page.locator('img').evaluateAll(async imgs=>{await Promise.all(imgs.map(i=>i.decode()));return imgs.filter(i=>!i.naturalWidth).map(i=>i.src)});
 assert.equal(badImages.length,0);
 const links=await page.locator('a[href]').evaluateAll(a=>a.map(x=>x.getAttribute('href')).filter(h=>!h.startsWith('http')&&!h.startsWith('#')));
 for(const href of links)assert(fs.existsSync(href.startsWith('file:')?fileURLToPath(href):path.resolve(out,href.split('#')[0])),href);
 assert.equal(errors.length,0);assert(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth));
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify({passed:true,combinations,richCombinations,freshModes,links:links.length,svg_text_overflow:overflow,js_errors:errors,bad_images:badImages},null,2));
 await browser.close();console.log(JSON.stringify({passed:true,combinations,richCombinations,freshModes}));
})().catch(e=>{console.error(e);process.exitCode=1;});
