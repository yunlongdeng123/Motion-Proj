const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const {chromium}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const root=path.resolve('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030');
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(root,'index.html')).href);
 const checks=[];
 for(const frame of ['000000','000008','000015'])for(const edit of ['none','delete','shift','rotate']){
  await page.selectOption('#frame',frame);await page.selectOption('#edit',edit);
  const images=await page.evaluate(async()=>{const a=[...document.querySelectorAll('#reference,#final,#white,#C,#alpha,#fill')];await Promise.all(a.map(i=>i.decode()));return a.map(i=>({id:i.id,w:i.naturalWidth,h:i.naturalHeight,src:i.getAttribute('src')}));});
  if(images.length!==6||images.some(i=>i.w!==480||i.h!==320))throw Error(JSON.stringify({frame,edit,images}));
  checks.push({frame,edit,images});
 }
 await page.selectOption('#frame','000008');await page.selectOption('#edit','delete');
 await page.evaluate(async()=>Promise.all([...document.images].map(i=>i.decode())));
 await page.locator('section').nth(2).screenshot({path:path.join(root,'preview.png')});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 await browser.close();
 if(errors.length||overflow)throw Error(JSON.stringify({errors,overflow}));
 fs.writeFileSync(path.join(root,'browser_validation.json'),JSON.stringify({passed:true,checks,errors,horizontal_overflow:overflow},null,2));
 console.log(JSON.stringify({passed:true,combinations:checks.length,images:checks.length*6}));
})().catch(e=>{console.error(e);process.exitCode=1;});
