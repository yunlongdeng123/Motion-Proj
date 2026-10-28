'use strict';
// Actual viewer/Canvas tests at the three regression anchors. The installer
// verifies every one of the 49 receipts, row sets, masks and real image files.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const canvasPath=process.env.CODEX_NATIVE_CANVAS||'C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas';
const {createCanvas,loadImage}=require(canvasPath);
const base=path.resolve(process.argv[2]||path.join(__dirname,'../..')),out=path.join(base,'outputs/GS_Scene_Diagnostics_20261027'),context='14705303724557273004_3105_000_3125_000',folder=path.join(out,context);
function fakeDocument(){
 const nodes=new Map(),doc={nodes};
 doc.createElement=tag=>{const e={tagName:tag,children:[],style:{},classList:{toggle(){}},value:'',checked:true,textContent:'',open:false,append(...xs){for(const x of xs){this.children.push(x);if(x&&typeof x==='object')x.parentNode=this;}},insertBefore(x){this.append(x);},replaceChildren(...xs){this.children=[];this.append(...xs);},addEventListener(){},setAttribute(k,v){this[k]=v;}};Object.defineProperty(e,'options',{get(){return this.children;}});Object.defineProperty(e,'id',{get(){return this._id;},set(v){this._id=v;nodes.set(v,this);}});return e;};
 doc.head=doc.createElement('head');doc.body=doc.createElement('body');doc.createTextNode=text=>({textContent:text});
 doc.getElementById=id=>{if(nodes.has(id))return nodes.get(id);const e=doc.createElement('div');if(/Canvas$/.test(id)||id==='sceneCanvas'){const c=createCanvas(id==='sceneCanvas'?840:480,id==='sceneCanvas'?560:320);e.width=c.width;e.height=c.height;e.getContext=(...x)=>c.getContext(...x);if(id==='sceneCanvas')e.getContext('2d').drawImage=()=>{throw Error('3D RGB backdrop prohibited');};}e.id=id;e.value=({geometryState:'current',pointSize:'2.5',cloudColor:'branch',semanticProfile:'gpu',semanticBoundary:'0',semanticBoundaryMass:'.2',semanticAblation:'all'})[id]||'';if(id==='lidar3d')e.checked=false;doc.body.append(e);return e;};return doc;
}
(async()=>{
 const doc=fakeDocument(),sb={console,document:doc,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Uint32Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,URL,URLSearchParams,location:{hash:''},setTimeout};sb.window=sb;sb.addEventListener=()=>{};vm.createContext(sb);
 for(const file of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js','target_detection_data.js','bounded_unknown_data.js','r2_reference_data.js','r2_inherited_data.js'])vm.runInContext(fs.readFileSync(path.join(folder,file),'utf8'),sb,{filename:file});
 const viewer=fs.readFileSync(path.join(out,'viewer.js'),'utf8').replace(/init\(\);\s*$/,'');vm.runInContext(viewer,sb);const a=sb.SceneViewer;a.S.clouds={};
 for(const[key,value]of Object.entries({...sb.CLOUD_DATA,...sb.FOREIGN_CLOUDS}))a.S.clouds[key]={...value,data:await a.unpack(value.data)};
 const gid='v43_r2_inherited49',group=sb.SCENE_DATA.groups.find(g=>g.id===gid),manifest=sb.R2_INHERITED_DELIVERY;
 assert.deepEqual(Object.keys(group.frames).map(Number),Array.from({length:49},(_,i)=>77+i));assert.equal(sb.SCENE_DATA.groups.filter(g=>g.id===gid).length,1);
 assert.deepEqual(Object.keys(sb.SCENE_DATA.groups.find(g=>g.id==='v43_r2_reference').frames),['98','113','114']);
 assert(sb.SCENE_DATA.groups.find(g=>g.id==='v43_bounded_unknown').title.includes('未采纳'));
 a.S.group=gid;a.S.channel='rgb';a.S.cameraView=true;const samples=[];
 for(const raw of [98,113,114]){
  a.S.raw=raw;a.draw3D();const state=sb.EDIT_GEOMETRY.groups[gid][raw],removed=new Set(state.static_suppressed_rows),receipt=JSON.parse(fs.readFileSync(path.join(folder,manifest.frames[raw].receipt),'utf8'));
  assert.equal(state.objects.length,0);assert.equal(removed.size,receipt.final_removed_rows);assert.equal(state.r2_inherited.source_raw,raw);assert.equal(state.r2_inherited.row_namespace,receipt.row_namespace);
  assert(a.S.projection.length>0);assert(a.S.projection.every(p=>p.cloud==='static'&&!removed.has(a.S.clouds.static.data.ids[p.j])));assert(a.geometryInstances().every(p=>p.cloud==='static'));
  const other=sb.IDENTITY_AUDIT.frames[raw].vehicle_boxes.filter(b=>!b.target).length;assert.equal(a.currentBoxes().filter(b=>!b.target).length,other);
  const image=await loadImage(path.join(folder,group.frames[raw].channels.rgb));assert.equal(image.width,480);assert.equal(image.height,320);
  samples.push({raw,removedRows:removed.size,actualCanvasStaticPoints:a.S.projection.length,otherVehicleBoxes:other});
 }
 vm.runInContext('show=async function(){S.epoch++};window.SceneViewer.show=show;',sb);
 vm.runInContext(fs.readFileSync(path.join(out,'scene_target_detection.js'),'utf8'),sb);
 assert.equal(doc.getElementById('targetDetection').open,false);assert.equal(sb.TargetDetection.state.cache.size,0);
 assert.deepEqual(doc.getElementById('targetDetectionRaw').options.filter(o=>!o.disabled).map(o=>o.value),['98','113','114']);
 vm.runInContext(fs.readFileSync(path.join(out,'scene_r2_reference.js'),'utf8'),sb);
 vm.runInContext(fs.readFileSync(path.join(out,'scene_r2_inherited.js'),'utf8'),sb);await Promise.resolve();
 assert.equal(a.S.group,gid);assert.equal(a.S.raw,113);assert.equal(doc.getElementById('r2InheritedPair').hidden,false);
 for(const raw of [77,98,113,125]){a.S.raw=raw;sb.R2Inherited.update();for(const name of ['observed','existing','white'])assert.equal(doc.getElementById('r2Pair_'+name).src,manifest.frames[raw].images[name]);}
 sb.location.hash='#workspace?group=v43_r2_reference&raw=98';a.S.group='v43_r2_reference';a.S.raw=98;vm.runInContext(fs.readFileSync(path.join(out,'scene_r2_inherited.js'),'utf8'),sb);assert.equal(a.S.group,'v43_r2_reference');assert.equal(a.S.raw,98);assert.equal(doc.getElementById('r2InheritedPair').hidden,true);
 const result={schema:'r2-inherited-ui-canvas-qa-v1',actualCanvas:true,browserQA:false,validatedFull49FilesSeparately:true,anchorSamples:samples,defaultNoExplicitGroup:'v43_r2_inherited49 raw113',explicitReferencePreserved:true,pairedSameFramePaths:true,legacyPanelCollapsed:true,legacyChoices:[98,113,114]};
 fs.writeFileSync(path.join(__dirname,'r2_inherited_canvas_qa.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
