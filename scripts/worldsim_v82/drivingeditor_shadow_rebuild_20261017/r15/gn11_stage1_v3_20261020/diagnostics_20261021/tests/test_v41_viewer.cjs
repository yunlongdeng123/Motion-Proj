/* 对实际诊断脚本和资产运行 Node 检查；不冒充浏览器 UI 验收。 */
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict'),zlib=require('zlib');
const root=process.argv[2],nodes=new Map(),requests=[];
const canvasContext=new Proxy({drawImage(im){assert(fs.existsSync(path.join(root,im.src)));}}, {get(t,k){return t[k]||(()=>{});},set(t,k,v){t[k]=v;return true;}});
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',dataset:{},options:[],getContext:()=>canvasContext,append(o){this.options.push(o);},prepend(o){this.options.unshift(o);},replaceChildren(){},querySelectorAll:()=>[]});return nodes.get(id);}
const ctx={console,setTimeout,clearTimeout,Uint8Array,Uint16Array,Uint32Array,Int32Array,Float32Array,Float64Array,TextDecoder,Map,Set,performance,
 document:{getElementById:node,createElement:()=>({}),querySelectorAll:()=>[]},GSCloud:{defaults:{}},Image:class{set src(v){this._src=v;requests.push(v);queueMicrotask(()=>fs.existsSync(path.join(root,v))?this.onload():this.onerror(Error('missing '+v)));}get src(){return this._src;}}};
ctx.window=ctx;vm.createContext(ctx);
function script(name){vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),ctx,{filename:name});}
function unpack(p){const b=zlib.inflateSync(Buffer.from(p.data,'base64')),out={};const types={'<i4':Int32Array,'<u4':Uint32Array,'<f4':Float32Array,'<f8':Float64Array,'|u1':Uint8Array,'<u2':Uint16Array};for(const [k,a] of Object.entries(p.arrays)){out[k]=new types[a.dtype](Uint8Array.from(b.subarray(a.offset,a.offset+a.bytes)).buffer);}return out;}
async function main(){
 script('data/scene.js');script('data/v4_experiments.js');const historical=ctx.GSDATA.scene.experiments.map(e=>e.id);
 script('data/v41_experiments.js');
 assert(historical.every(id=>ctx.GSDATA.scene.experiments.some(e=>e.id===id)));
 assert.equal(ctx.GSDATA.scene.experiments.filter(e=>e.id==='v4_1_R').length,1);
 assert.equal(ctx.GSDATA.defaultExperiment,'v4_1_R');
 assert.equal(node('channel').options.length,2);
 let source=fs.readFileSync(path.join(root,'app.js'),'utf8');assert(source.trimEnd().endsWith('init();'));
 source=source.trimEnd().slice(0,-7)+'\nwindow.check={S,render2D,supportsImage,asset};';
 vm.runInContext(source,ctx,{filename:'app.js'});
 assert.equal(ctx.check.S.arm,'v4_1_R');assert.equal(ctx.check.S.channel,'native_v1');
 for(const raw of [89,102,137])for(const channel of ['joint','deleted_five','native_v1','native_static_delete','dynamic_raw','road_raw','road_surface_color']){
   ctx.check.S.raw=raw;ctx.check.S.channel=channel;await ctx.check.render2D();
   assert.equal(node('renderCanvas').dataset.raw,raw);assert.equal(node('renderCanvas').dataset.channel,channel);
   assert(!node('status').className?.includes('bad'),node('status').textContent);
 }
 const before=requests.length;ctx.check.S.arm='full';ctx.check.S.channel='native_v1';await ctx.check.render2D();
 assert.equal(requests.length,before);assert(node('mapLegend').textContent.includes('仅绑定'));
 for(const raw of Array.from({length:49},(_,i)=>89+i)){
   script(`data/v4_1_R/${raw}.js`);const maps=unpack(ctx.GSDATA[`v4_1_R/${raw}`]);
   for(const key of ['road_alpha','road_bg','road_joint','joint_alpha','static_depth_m','joint_depth_m','residual'])assert.equal(maps[key].length,480*320);
   delete ctx.GSDATA[`v4_1_R/${raw}`];
 }
 console.log(JSON.stringify({status:'PASSED',kind:'Node actual app functions and file assets',historical_experiments_preserved:historical,frames:49,render_paths_checked:21,unsupported_old_channel_no_request:true,browser_ui_verified:false}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
