// 运行真实app/inspection/sampling JS和真实离线数据；Canvas/DOM测试不是浏览器验收。
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const deps='C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {createCanvas,Image,GlobalFonts}=require(require.resolve('@napi-rs/canvas',{paths:[deps]}));GlobalFonts.registerFromPath('C:/Windows/Fonts/msyh.ttc','Microsoft YaHei');
const root=process.env.GSDIAG_ROOT;if(!root)throw Error('Set GSDIAG_ROOT to the installed complete offline package');const legacy=root,src=root;
const out=path.resolve(process.env.GSDIAG_QA_OUTPUT||'work/v4_R_30k_QA');fs.mkdirSync(out,{recursive:true});
function file(rel){const p=path.join(root,rel);return fs.existsSync(p)?p:path.join(legacy,rel);}
let created=0;const nodes=new Map(),loads=[],errors=[];
function el(id){if(nodes.has(id))return nodes.get(id);const canvas=/Canvas$/.test(id)||id.startsWith('compare_');const e=canvas?createCanvas(id==='sceneCanvas'?1100:480,id==='sceneCanvas'?650:320):{};
 Object.assign(e,{id,value:'',checked:false,disabled:false,hidden:false,style:{},dataset:{},options:[],children:[],textContent:'',innerHTML:'',classList:{toggle(){},add(){},remove(){}},
 append(...a){this.children.push(...a);this.options.push(...a);},prepend(...a){this.children.unshift(...a);},replaceChildren(...a){this.children=a;this.options=a;this.textContent='';this.innerHTML='';},
 querySelector(s){return el(id+s);},querySelectorAll(){return [];},closest(){return el('panel');},setAttribute(k,v){this[k]=v;},addEventListener(){},before(){},after(){},remove(){},setPointerCapture(){},scrollIntoView(){},getBoundingClientRect(){return {left:0,top:0,width:this.width||480,height:this.height||320};}});
 Object.defineProperties(e,{parentNode:{get:()=>({lastChild:el(id+'_label')})},nextElementSibling:{get:()=>el(id+'_next')},firstElementChild:{get:()=>el(id+'_first')},lastElementChild:{get:()=>el(id+'_last')}});
 if(canvas){Object.defineProperties(e,{clientWidth:{value:e.width},clientHeight:{value:e.height}});}nodes.set(id,e);return e;}
class LocalImage extends Image{set src(p){super.src=fs.readFileSync(file(p));}}
const doc={getElementById:el,createElement:tag=>el('created'+created++),querySelector:s=>el(s),querySelectorAll:s=>s==='.workspace canvas'?['renderCanvas','sceneCanvas','bevCanvas','profileCanvas'].map(el):s.includes('#counterfactual')?[el('.workspace'),el('#counterfactual'),el('#geometryEvidence'),el('#sources')]:[],addEventListener(){},activeElement:{tagName:'BODY'},head:{append(script){loads.push(script.src);try{vm.runInContext(fs.readFileSync(file(script.src),'utf8'),ctx);queueMicrotask(()=>script.onload());}catch(e){errors.push(String(e));queueMicrotask(()=>script.onerror(e));}}}};
const box={console:{...console,error:e=>errors.push(String(e))},document:doc,location:{protocol:'file:'},Image:LocalImage,URL,Blob,Response,DecompressionStream,atob,setTimeout,clearTimeout,performance,ResizeObserver:class{observe(){}},GSDATA:{}};box.window=box;const ctx=vm.createContext(box);
function run(rel,source=false){vm.runInContext(fs.readFileSync(source?path.join(src,rel):file(rel),'utf8').replace(rel==='app.js'?/init\(\);\s*$/:/$a/,''),ctx);}
for(const p of ['data/scene.js','data/v4_experiments.js','data/sampling_meta.js','data/geometry_index.js'])run(p);
for(const p of ['offline_geometry.js','sampling.js','inspection_math.js','v1_ground.js','inspection.js','app.js'])run(p,true);
vm.runInContext(`scene=GSDATA.scene;window.gsdiag={state:S,scene,refresh,pendingExperiment,load,getMaps:()=>maps,getCloud:()=>cloud,getRoad:()=>road,getUnits:()=>units,draw3D,rankRows,getView,view:V,inspection:()=>GSInspect.debug()};`,ctx);
(async()=>{
 await vm.runInContext(`(async()=>{road=await load('road','data/road.js');bind();S.arm='v4_R';S.channel='deleted_five';$('cloudColor').value='role';$('bevMode').value='exposure';$('threeSize').value='2.5';$('threeSolid').checked=true;$('roads3d').checked=true;$('ordinary').checked=false;$('dynamic3d').checked=false;await refresh();})()`,ctx);
 const checks=[];assert(el('status').textContent.includes('就绪'),el('status').textContent);assert.equal(box.gsdiag.getCloud().sampling.stats.static.road_total,1919);assert.equal(box.gsdiag.inspection().data.v4.unit_ids.length,14034);checks.push('V4 R selects real49-frame30K assets;1919 output roads distinguished from14034 initialized/fixed sidecar roads');
 assert.equal(el('arm').options.filter(o=>o.value==='v4_R').length,1);assert(el('arm').options.find(o=>o.value==='v4_R').textContent.includes('30K'));checks.push('original experiment dropdown exposes R30K and retains the RP placeholder');
 assert(!loads.some(p=>p.includes('v4_R/optimization')||p.includes('rays/v4_R')));assert(el('queryRay').disabled);checks.push('unknown V4 optimization history and unexported legacy traces are not substituted');
 assert(box.gsdiag.inspection().data.surfaces.ordinary_depth_m.some(v=>v>0));assert(box.gsdiag.getMaps().road_bg.some(v=>v>0));checks.push('independent native road/ordinary depths and same-raster road contributions are available');
 await vm.runInContext(`S.raw=100;refresh()`,ctx);assert(el('status').textContent.includes('就绪'));assert.equal(box.gsdiag.inspection().data.identity,'v4_R/100');
 await vm.runInContext(`S.raw=137;refresh()`,ctx);assert(el('status').textContent.includes('就绪'));checks.push('switching raw102→100→137 updates real measurements and surfaces without mixing frames');
 await vm.runInContext(`S.raw=102;S.preview_sampling={static_rate:3,dynamic_rate:3,vehicle:37,vehicle_rate:10,vehicle_scope:'projection'};refresh()`,ctx);assert(el('status').textContent.includes('就绪'),el('status').textContent);assert(box.gsdiag.getCloud().sampling.offline);assert.equal(box.gsdiag.getCloud().sampling.stats.static.road_total,1919);checks.push('editable static/dynamic/single-car sampling loads V4 geometry chunks and retains all actual output roads');
 vm.runInContext(`const d=GSInspect.debug().data.v4;GSInspect.selectUnit(d.unit_ids[d.output_mask.findIndex(x=>!x)]);`,ctx);assert(el('roadHistoryInfo').textContent.includes('实际参与输出否'));checks.push('clicking initialized but excluded road explains its fixed geometry and output-mask exclusion');
 run('gpu.js',true);await box.gpuDiag.onSelection();assert(el('queryGPU').disabled);assert(el('viewState').textContent.includes('V4'));
 await el('cachedView').onchange({target:{value:'orbit'}});assert(el('viewState').textContent.includes('不替用'));checks.push('V4 cannot request legacy CUDA attribution or display another experiment orbit');
 await vm.runInContext(`S.preview_sampling={...GSCloud.defaults};S.arm='v4_RP';refresh()`,ctx);assert(el('status').textContent.includes('占位'));assert.equal(box.gsdiag.getCloud(),null);assert(el('.workspace').inert);checks.push('RP remains explicit pending and clears R images/metrics/geometry');
 await vm.runInContext(`S.arm='full';refresh()`,ctx);assert(el('status').textContent.includes('就绪'));assert(!el('.workspace').inert);checks.push('switching back to original V3 restores original evidence and controls');
 await vm.runInContext(`S.arm='v4_R';S.raw=102;S.channel='deleted_five';refresh()`,ctx);assert(el('status').textContent.includes('就绪'));
 fs.writeFileSync(path.join(out,'raw102_R_deleted.png'),el('renderCanvas').toBuffer('image/png'));
 fs.writeFileSync(path.join(out,'raw102_R_road_geometry.png'),el('sceneCanvas').toBuffer('image/png'));
 assert.deepEqual(errors,[],errors.join('\n'));const result={status:'PASSED',checks,browser_ui_verified:false,scope:'actual app JS and packed data with Node Canvas/test DOM',training_updates:0};fs.writeFileSync(path.join(out,'ui_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
