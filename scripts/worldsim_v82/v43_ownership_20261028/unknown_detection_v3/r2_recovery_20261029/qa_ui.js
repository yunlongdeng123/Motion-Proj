'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {createCanvas,loadImage}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const base=path.resolve(__dirname,'../..'),out=path.join(base,'outputs/GS_Scene_Diagnostics_20261027'),context='14705303724557273004_3105_000_3125_000',folder=path.join(out,context);
function fakeDocument(){
 const nodes=new Map(),doc={nodes};
 doc.createElement=tag=>{const e={tagName:tag,children:[],style:{},classList:{toggle(){}},value:'',checked:true,textContent:'',open:false,append(...xs){for(const x of xs){this.children.push(x);if(x&&typeof x==='object')x.parentNode=this;}},insertBefore(x){this.append(x);},replaceChildren(...xs){this.children=[];this.append(...xs);},addEventListener(){},setAttribute(k,v){this[k]=v;}};Object.defineProperty(e,'options',{get(){return this.children;}});Object.defineProperty(e,'id',{get(){return this._id;},set(v){this._id=v;nodes.set(v,this);}});return e;};
 doc.head=doc.createElement('head');doc.body=doc.createElement('body');doc.createTextNode=text=>({textContent:text});
 doc.getElementById=id=>{if(nodes.has(id))return nodes.get(id);const e=doc.createElement('div');if(/Canvas$/.test(id)||id==='sceneCanvas'){const c=createCanvas(id==='sceneCanvas'?840:480,id==='sceneCanvas'?560:320);e.width=c.width;e.height=c.height;e.getContext=(...x)=>c.getContext(...x);if(id==='sceneCanvas')e.getContext('2d').drawImage=()=>{throw Error('3D RGB backdrop prohibited');};}e.id=id;e.value=({geometryState:'current',pointSize:'2.5',cloudColor:'branch',semanticProfile:'gpu',semanticBoundary:'0',semanticBoundaryMass:'.2',semanticAblation:'all'})[id]||'';if(id==='lidar3d')e.checked=false;doc.body.append(e);return e;};return doc;
}
(async()=>{
 const doc=fakeDocument(),sb={console,document:doc,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Uint32Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,URL,URLSearchParams,location:{hash:''},setTimeout};sb.window=sb;sb.addEventListener=()=>{};vm.createContext(sb);
 for(const file of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js','target_detection_data.js','bounded_unknown_data.js','r2_reference_data.js'])vm.runInContext(fs.readFileSync(path.join(folder,file),'utf8'),sb,{filename:file});
 const viewer=fs.readFileSync(path.join(out,'viewer.js'),'utf8').replace(/init\(\);\s*$/,'');vm.runInContext(viewer,sb);const a=sb.SceneViewer;a.S.clouds={};
 for(const[key,value]of Object.entries({...sb.CLOUD_DATA,...sb.FOREIGN_CLOUDS}))a.S.clouds[key]={...value,data:await a.unpack(value.data)};
 const group=sb.SCENE_DATA.groups.find(g=>g.id==='v43_r2_reference');assert.deepEqual(Object.keys(group.frames),['98','113','114']);assert.equal(sb.SCENE_DATA.groups.filter(g=>g.id===group.id).length,1);
 a.S.group=group.id;a.S.channel='rgb';a.S.cameraView=true;const samples=[];
 for(const raw of [98,113,114]){a.S.raw=raw;a.draw3D();const state=sb.EDIT_GEOMETRY.groups[group.id][raw],removed=new Set(state.static_suppressed_rows);assert.equal(state.objects.length,0);assert(a.S.projection.length>0);assert(a.S.projection.every(p=>p.cloud==='static'&&!removed.has(a.S.clouds.static.data.ids[p.j])));assert(a.geometryInstances().every(p=>p.cloud==='static'));const expected=sb.IDENTITY_AUDIT.frames[raw].vehicle_boxes.filter(b=>!b.target).length;assert.equal(a.currentBoxes().filter(b=>!b.target).length,expected);const image=await loadImage(path.join(folder,group.frames[raw].channels.rgb));assert.equal(image.width,480);assert.equal(image.height,320);samples.push({raw,removedRows:removed.size,actualCanvasStaticPoints:a.S.projection.length,otherVehicleBoxes:expected});}
 // Stub only asynchronous RGB I/O; use actual group selection, wrappers and Canvas renderer.
 vm.runInContext('show=async function(){S.epoch++};window.SceneViewer.show=show;',sb);
 vm.runInContext(fs.readFileSync(path.join(out,'scene_target_detection.js'),'utf8'),sb,{filename:'scene_target_detection.js'});
 assert.equal(doc.getElementById('targetDetection').tagName,'details');assert.equal(doc.getElementById('targetDetection').open,false);assert.equal(sb.TargetDetection.state.cache.size,0);
 assert.deepEqual(doc.getElementById('targetDetectionRaw').options.filter(o=>!o.disabled).map(o=>o.value),['98','113','114']);
 a.S.group='v43_actor_10000';a.S.raw=109;
 vm.runInContext(fs.readFileSync(path.join(out,'scene_r2_reference.js'),'utf8'),sb,{filename:'scene_r2_reference.js'});await Promise.resolve();
 assert.equal(a.S.group,'v43_r2_reference');assert.equal(a.S.raw,113);assert.equal(doc.getElementById('frame').min,0);assert.equal(doc.getElementById('frame').max,2);
 doc.getElementById('frame').value='0';doc.getElementById('frame').oninput.call(doc.getElementById('frame'));await Promise.resolve();assert.equal(a.S.raw,98);
 sb.location.hash='#workspace?group=v43_bounded_unknown&raw=109';a.S.group='v43_bounded_unknown';a.S.raw=109;
 vm.runInContext(fs.readFileSync(path.join(out,'scene_r2_reference.js'),'utf8'),sb);assert.equal(a.S.group,'v43_bounded_unknown');assert.equal(a.S.raw,109);
 assert.equal(sb.TargetDetection.state.cache.size,0);assert.equal(doc.getElementById('targetDetection').open,false);
 const result={schema:'r2-reference-ui-qa-v1',actualCanvas:true,browserQA:false,r2Frames:samples,legacyPanelDefaultCollapsed:true,legacyReadyChoices:[98,113,114],legacyNoDataFetchWhenCollapsed:true,defaultNoExplicitGroup:'v43_r2_reference raw113',explicitGroupPreserved:true,threeFrameSliderIndexMapping:true};
 fs.writeFileSync(path.join(__dirname,'ui_canvas_qa.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
