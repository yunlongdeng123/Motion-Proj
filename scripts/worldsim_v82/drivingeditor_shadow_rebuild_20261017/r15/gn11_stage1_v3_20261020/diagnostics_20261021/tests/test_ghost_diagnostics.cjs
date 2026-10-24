const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const root=process.argv[2],nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{options:[],value:'',append(o){this.options.push(o);},prepend(o){this.options.unshift(o);},replaceChildren(){},getContext(){return new Proxy({}, {get:()=>()=>{}})}});return nodes.get(id);}
const ctx={console,Uint8Array,Uint16Array,Uint32Array,Int32Array,Float32Array,Float64Array,Map,Set,performance,setTimeout,clearTimeout,GSCloud:{defaults:{}},document:{getElementById:node,createElement:()=>({})}};ctx.window=ctx;vm.createContext(ctx);
function run(f){vm.runInContext(fs.readFileSync(path.join(root,f),'utf8'),ctx);}
['data/scene.js','data/v4_experiments.js','data/v41_experiments.js','data/gpu_index.js','v41_ghost_index.js'].forEach(run);
const qi=ctx.GSDATA.gpu_index.queries,ci=ctx.GSDATA.gpu_index.counterfactuals;
assert.equal(Object.keys(qi).filter(x=>x.startsWith('gpu/v4_1_R/')).length,20);
assert.equal(Object.keys(ci).filter(x=>x.startsWith('cf/v4_1_R/')).length,5);
assert.equal(ctx.GSDATA.defaultChannel,'native_v1');
let maxError=0,oneHot=0,assets=0;
for(const [key,file] of Object.entries(qi).filter(([k])=>k.startsWith('gpu/v4_1_R/'))){run(file);const q=ctx.GSDATA[key];assert.equal(q.status,'NATIVE_BUFFER_CUDA_TRACE_VERIFIED');assert.equal(q.parameter_updates,0);maxError=Math.max(maxError,...Object.values(q.replay_errors));oneHot=Math.max(oneHot,q.id_weight_validation.max_weight_error);assert(maxError<=2e-5);assert(oneHot<=2e-6);assert(Math.abs(Object.values(q.groups).reduce((a,b)=>a+b)-q.reference.alpha)<2e-5);}
for(const [key,file] of Object.entries(ci).filter(([k])=>k.startsWith('cf/v4_1_R/'))){run(file);const cf=ctx.GSDATA[key];assert.equal(cf.removed.length,20);assert.equal(cf.parameter_updates,0);assert.equal(cf.checkpoint_parameters_modified,false);for(const frame of [cf,...cf.temporal_controls]){for(const file of Object.values(frame.images)){assert(fs.existsSync(path.join(root,file)));assets++;}}}
let app=fs.readFileSync(path.join(root,'app.js'),'utf8').trimEnd();assert(app.endsWith('init();'));
vm.runInContext(app.slice(0,-7)+'\nwindow.testAPI={S,cachedRay}; profile=()=>{}; showRay=v=>{window.delivered=v;};load=(k,p)=>Promise.resolve(window.GSDATA[k]);',ctx);
ctx.gpuDiag={onSelection(){}};
async function main(){
 for(const key of Object.keys(qi).filter(k=>k.startsWith('gpu/v4_1_R/'))){const [_,arm,raw,x,y,mode]=key.split('/');Object.assign(ctx.testAPI.S,{arm,raw:+raw,x:+x,y:+y,rayMode:mode});ctx.delivered=null;await ctx.testAPI.cachedRay();assert(ctx.delivered);assert.equal(ctx.delivered.mode,mode);assert.deepEqual(Array.from(ctx.delivered.pixel),[+x,+y]);}
 Object.assign(ctx.testAPI.S,{raw:103});ctx.delivered=null;await ctx.testAPI.cachedRay();assert.equal(ctx.delivered,null);assert(node('rayState').textContent.includes('未覆盖'));
 // Exercise the real cached counterfactual selection branch with minimal DOM.
 const gpu=fs.readFileSync(path.join(root,'gpu.js'),'utf8');const selection=gpu.slice(gpu.indexOf('async function selection()'),gpu.indexOf(' function showView('));
 const cfctx={console,generation:0,cf:null,$:node,state:()=>ctx.testAPI.S,showCF(v){this.deliveredCF=v;},window:{gsdiag:{state:ctx.testAPI.S,pendingExperiment:()=>null,load:(k,p)=>Promise.resolve(ctx.GSDATA[k])},GSDATA:ctx.GSDATA}};
 cfctx.showCF=v=>{cfctx.deliveredCF=v;};vm.createContext(cfctx);vm.runInContext(selection+'\nthis.select=selection;',cfctx);
 Object.assign(ctx.testAPI.S,{raw:102,x:255,y:240,rayMode:'deleted_five'});await cfctx.select();assert(cfctx.deliveredCF);assert.equal(cfctx.deliveredCF.pixel[0],255);
 Object.assign(ctx.testAPI.S,{raw:103});cfctx.deliveredCF=null;await cfctx.select();assert.equal(cfctx.deliveredCF,null);assert(node('renderCF').disabled);
 const report=fs.readFileSync(path.join(root,'V41_GHOST_DIAGNOSIS.html'),'utf8');assert(report.includes('0.100078'));assert(report.includes('543,783'));assert(report.includes('926,556'));assert(report.includes('道路独立alpha也为0'));
 console.log(JSON.stringify({status:'PASSED',actual_cached_ray_paths:20,actual_cached_CF_path:true,uncovered_pixel_remains_unknown:true,trace_max_error:maxError,onehot_max_error:oneHot,CF_images:assets,browser_ui_verified:false}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
