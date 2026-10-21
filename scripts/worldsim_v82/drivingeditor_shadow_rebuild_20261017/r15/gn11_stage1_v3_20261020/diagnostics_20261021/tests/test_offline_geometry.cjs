/* 实际离线分块读取测试；普通Node数据测试，不启动浏览器或HTTP服务。 */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),zlib=require('node:zlib');
const root=process.argv[2],reference=process.argv[3];
const readJS=p=>JSON.parse(fs.readFileSync(p,'utf8').split(']=')[1].trim().slice(0,-1));
function unpack(p){const data=zlib.inflateSync(Buffer.from(p.data,'base64')),out={};const types={'<i4':Int32Array,'<u4':Uint32Array,'<f4':Float32Array,'<f8':Float64Array,'|u1':Uint8Array,'<u2':Uint16Array};for(const [key,a] of Object.entries(p.arrays)){const b=data.subarray(a.offset,a.offset+a.bytes),copy=Uint8Array.from(b);out[key]=new types[a.dtype](copy.buffer);}if(p.branches)out.branches=p.branches;if(p.sampling)out.sampling=p.sampling;return out;}
global.window={GSDATA:{geometryIndex:readJS(path.join(root,'data/geometry_index.js'))}};
global.document={createElement(){return {remove(){}};},head:{append(script){try{window.GSDATA[readKey(script.src)]=readJS(path.join(root,script.src));queueMicrotask(()=>script.onload());}catch(e){queueMicrotask(()=>script.onerror(e));}}}};
function readKey(relative){const text=fs.readFileSync(path.join(root,relative),'utf8');return JSON.parse(text.slice(text.indexOf('window.GSDATA[')+14,text.indexOf(']=')));}
const source=fs.existsSync(path.join(__dirname,'src/offline_geometry.js'))?path.join(__dirname,'src/offline_geometry.js'):path.join(__dirname,'../src/offline_geometry.js');
vm.runInThisContext(fs.readFileSync(source,'utf8'));
const selection=window.GSOfflineGeometry.selection;
const flags=Uint8Array.from([1,0,1,0,0,1,0,0,0,1]);
assert.deepEqual([...selection(10,20,flags,1,50).selected],[0,1,2]);
assert.equal(selection(10,20,flags,1,50).sampled,2);
assert.equal(selection(10,0,flags,1,100).sampled,4);
assert.equal(selection(10,100,null,0,0).selected.length,10);
async function main(){
 const ref=reference?JSON.parse(fs.readFileSync(reference,'utf8')):null,checks=[];
 for(const arm of Object.keys(window.GSDATA.geometryIndex.arms)){
  const base=unpack(readJS(path.join(root,`data/${arm}/cloud.js`)));
  for(const actor of [5,37]){
   const p={static_rate:2,dynamic_rate:2,vehicle:actor,vehicle_rate:10,vehicle_scope:'projection'};
   const out=await window.GSOfflineGeometry.make(arm,102,p,base,unpack,()=>{});
   for(const name of ['static','dynamic']){
    const rows=[...out[name+'_row']].sort((a,b)=>a-b);
    assert.equal(new Set(rows).size,rows.length);
    if(ref)assert.deepEqual(rows,ref[arm][actor][name]);
    assert.equal(out[name+'_role'].filter(v=>v===1).length,out.sampling.stats[name].road_total);
   }
   checks.push(`${arm}/raw102/a${actor}/10% `+(ref?'matches full CPU sampler':'has unique rows and retains road rows'));
  }
 }
 const arm='full',base=unpack(readJS(path.join(root,`data/${arm}/cloud.js`)));
 const all=await window.GSOfflineGeometry.make(arm,102,{static_rate:100,dynamic_rate:100,vehicle:'none',vehicle_rate:10,vehicle_scope:'projection'},base,unpack,()=>{});
 for(const name of ['static','dynamic']){
  const rows=all[name+'_row'],total=window.GSDATA.geometryIndex.arms[arm].branches[name].total;
  assert.equal(rows.length,total);assert.equal(new Set(rows).size,total);
 }
 checks.push('full/100% loads every Gaussian once, including all road rows');
 console.log(JSON.stringify({status:'PASSED',checks},null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
