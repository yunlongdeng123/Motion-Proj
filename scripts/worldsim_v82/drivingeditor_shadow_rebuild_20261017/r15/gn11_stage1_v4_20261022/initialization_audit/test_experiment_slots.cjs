// Execute actual selector/refresh/comparison code; no browser or remote requests.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const app=process.argv[2],registry=process.argv[3],gpu=process.argv[4];
if(!app||!registry||!gpu)throw Error('usage: node test_experiment_slots.cjs app.js v4_experiments.js gpu.js');
const nodes=new Map(),loads=[],messages=[],requests=[];
function element(id){if(nodes.has(id))return nodes.get(id);const e={id,style:{},dataset:{},options:[],children:[],checked:true,value:'',disabled:false,hidden:false,textContent:'',innerHTML:'',width:480,height:320,clientWidth:480,clientHeight:320,
 append(...a){this.children.push(...a);this.options.push(...a);},prepend(...a){this.children.unshift(...a);this.options.unshift(...a);},replaceChildren(...a){this.children=a;this.options=a;this.innerHTML='';this.textContent='';},
 querySelector(){return element(id+'_sub');},querySelectorAll(){return [];},closest(){return element('panel');},setAttribute(k,v){this[k]=v;},getAttribute(k){return this[k];},addEventListener(){},before(){},after(){},setPointerCapture(){},scrollIntoView(){},
 getContext(){return {fillRect(){},fillText(text){messages.push({id,text});},clearRect(){},drawImage(){}}},getBoundingClientRect(){return {left:0,top:0,width:480,height:320}}};e.firstElementChild=e;e.lastElementChild=e;nodes.set(id,e);return e;}
const canvasIds=['renderCanvas','sceneCanvas','bevCanvas','profileCanvas'];
const document={getElementById:element,createElement:tag=>element('created_'+nodes.size),querySelector:s=>element(s),querySelectorAll:s=>s==='.workspace canvas'?canvasIds.map(element):s.includes('#counterfactual')?[element('.workspace'),element('#counterfactual'),element('#geometryEvidence'),element('#sources')]:[],addEventListener(){},activeElement:{tagName:'BODY'},head:{append(){throw Error('unexpected script load');}}};
const sourceScene={experiments:[{id:'full',name:'完整版'},{id:'real_only',name:'仅真实观测'},{id:'prior_only',name:'仅先验'},{id:'ownership_only',name:'仅归属修复'}],bindings:{gn11_v1:'test'},cameras:Array.from({length:49},()=>({})),meter_to_model:.1};
const sandbox={console,document,location:{protocol:'file:'},URL,Blob,Response,DecompressionStream,atob,setTimeout,clearTimeout,performance,
 Map,Set,Math,Number,Array,Object,Promise,JSON,Float32Array,Float64Array,Uint8Array,Uint16Array,Uint32Array,Int32Array,
 ResizeObserver:class{observe(){}},fetch(){requests.push('fetch');throw Error('unexpected fetch');},GSDATA:{scene:sourceScene,gpu_index:{}},
 GSCloud:{defaults:{static_rate:2,dynamic_rate:2},bind(){},load:async()=>{loads.push('cloud');return {static_road_id:new Int32Array([-1])};},valid:p=>p},
 GSInspect:{bind(){},load:async()=>({}),accept(){},nearest:()=>null,values:()=>[]},GSV1Ground:{enabled:()=>false,heat:()=>false}};
sandbox.window=sandbox;const ctx=vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(registry,'utf8'),ctx);vm.runInContext(fs.readFileSync(app,'utf8').replace(/init\(\);\s*$/,''),ctx);
vm.runInContext(`scene=GSDATA.scene;road={provenance:new Uint8Array([1])};window.gsdiag={state:S,pendingExperiment,load,draw3D,rankRows,getView};bind();`,ctx);
vm.runInContext(fs.readFileSync(gpu,'utf8'),ctx);
const checks=[];
assert.deepEqual(element('arm').options.map(v=>v.value),['full','real_only','prior_only','ownership_only','gn11_v1','v4_R','v4_RP']);
assert(element('arm').options.slice(-2).every(v=>!v.disabled));checks.push('existing experiment dropdown contains exactly two selectable V4 slots and five existing groups');
sandbox.testLoad=async(key,p)=>{loads.push(p);if(p.includes('/units_'))return {raster_uv:new Float32Array([90,255]),in_view:new Uint8Array([1])};return {};};
vm.runInContext('load=window.testLoad;S.channel="static_alpha";heat=()=>"test";offlineCase=()=>false;render2D=async()=>{};pixelMetrics=()=>{};draw3D=()=>{};drawBEV=()=>{};unitInfo=()=>{};sourceGrid=()=>{};profile=()=>{};',ctx);
(async()=>{
 for(const arm of ['v4_R','v4_RP']){
  sandbox.testArm=arm;await vm.runInContext('S.arm=testArm;S.ray={old:true};S.selected={old:true};S.checked.add("old");refresh()',ctx);
  assert(vm.runInContext('maps===null&&cloud===null&&units===null&&S.ray===null&&S.selected===null&&S.checked.size===0',ctx));
  assert(element('status').textContent.includes('尚未导出'));assert(element('.workspace').inert);assert(element('queryGPU').disabled);assert(element('nativeViewImage').hidden);
  assert(element('compareMetrics').innerHTML.includes('data-arm="'+arm+'"'));assert(element('compareMetrics').innerHTML.includes('结果待导出'));
 }
 checks.push('selecting either V4 slot clears old images/geometry/numeric/source state and displays explicit pending results');
 assert(!loads.some(v=>v.includes('v4_R')));assert(!loads.includes('cloud'));checks.push('pending slots never load nonexistent V4 maps, geometry or checkpoint assets');
 assert(messages.some(v=>v.id==='compare_v4_R'&&v.text.includes('结果待导出')));assert(messages.some(v=>v.id==='compare_v4_RP'&&v.text.includes('结果待导出')));checks.push('same-frame comparison retains two V4 placeholder cards with unknown metrics');
 const prior=loads.length;element('cachedView').value='orbit';await element('cachedView').onchange({target:element('cachedView')});await element('queryGPU').onclick();assert.equal(loads.length,prior);assert.equal(requests.length,0);assert(element('viewState').textContent.includes('尚未导出'));checks.push('GPU and cached orbit controls cannot substitute a full experiment view into a V4 slot');
 assert.throws(()=>vm.runInContext('request()',ctx),/尚未导出/);checks.push('no checkpoint-bound request can be created from an unexported slot');
 await vm.runInContext('S.arm="full";S.unit=0;refresh()',ctx);assert(!element('.workspace').inert);assert(!element('queryGPU').disabled);assert(vm.runInContext('cloud!==null',ctx));assert(element('status').textContent.includes('就绪'));checks.push('switching back to an existing experiment restores the regular data path and controls');
 console.log(JSON.stringify({status:'PASSED',checks,browser_ui_verified:false,training_updates:0},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
