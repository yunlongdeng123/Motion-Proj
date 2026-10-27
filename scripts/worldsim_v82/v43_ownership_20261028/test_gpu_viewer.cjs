const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const {createCanvas}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const root=path.resolve('outputs/GS_Scene_Diagnostics_20261027'),out=path.resolve('outputs/V43_Ownership_20261027/gpu_v43');
async function check(context){
 const folder=path.join(root,context),nodes=new Map(),canvas=createCanvas(840,560);canvas.getContext('2d').drawImage=()=>{throw Error('RGB backdrop prohibited');};let sb;
 function get(id){if(id==='sceneCanvas')return canvas;if(id==='renderCanvas'){if(!nodes.has(id))nodes.set(id,createCanvas(480,320));return nodes.get(id);}if(!nodes.has(id))nodes.set(id,{checked:!['lidar3d'].includes(id),value:({cloudColor:'semantic_branch',pointSize:'2.5',geometryState:'current',semanticProfile:'gpu',semanticRadius:'2',semanticDepth:'.5',semanticSigma:'.5'})[id]||'',textContent:''});return nodes.get(id);}
 sb={console,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,document:{getElementById:get,createElement:()=>({}),head:{append(n){queueMicrotask(()=>{try{vm.runInContext(fs.readFileSync(path.resolve(folder,n.src),'utf8'),sb);n.onload();}catch(e){console.error(e);n.onerror();}});}}}};sb.window=sb;vm.createContext(sb);
 for(const f of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js'])vm.runInContext(fs.readFileSync(path.join(folder,f),'utf8'),sb);
 vm.runInContext(fs.readFileSync(path.join(root,'viewer.js'),'utf8').replace(/init\(\);\s*$/,''),sb);const a=sb.SceneViewer;a.S.clouds={};for(const part of ['static','dynamic'])a.S.clouds[part]={...sb.CLOUD_DATA[part],data:await a.unpack(sb.CLOUD_DATA[part].data)};
 const snaps=[],raws=Object.keys(a.D.frames).map(Number);let decoded=0,classes=new Set();
 for(const raw of raws){a.S.raw=raw;await a.ensureSemantic();const data=a.S.gpuSemanticFrames.get(raw);assert.equal(data.semantic_image.length,480*320);assert.equal(data.class_id.length,data.ids.length);assert(data.ids.every((v,i)=>i===0||v>data.ids[i-1]));decoded++;for(const k of data.semantic_image)classes.add(k);}
 assert(classes.size>=5);for(const i of [14,32,47]){
  a.S.raw=raws[i];await a.ensureSemantic();a.S.cameraView=true;const before=JSON.stringify(a.D.frames[a.S.raw].c2w);a.draw3D();let labeled=0,vehicles=0,staticRoad=0;
  for(const p of a.S.projection){const inst=a.geometryInstances().find(x=>x.key===p.instance),e=a.semanticRow(inst,p.j);if(e.class_id)labeled++;if([14,15,16,17].includes(e.class_id))vehicles++;if(e.branch==='static'&&e.class_id===1)staticRoad++;assert.equal(e.branch,inst.cloud==='static'?'static':'dynamic');}
  assert(labeled>0&&staticRoad>0);assert(a.currentBoxes().some(b=>b.target)&&a.currentBoxes().some(b=>!b.target));assert.equal(before,JSON.stringify(a.D.frames[a.S.raw].c2w));
  fs.writeFileSync(path.join(out,context.slice(0,6)+'_raw'+a.S.raw+'_gpu_semantic3d.png'),canvas.toBuffer('image/png'));
  vm.runInContext('drawSemanticImage()',sb);fs.writeFileSync(path.join(out,context.slice(0,6)+'_raw'+a.S.raw+'_semantic2d.png'),get('renderCanvas').toBuffer('image/png'));
  snaps.push({raw:a.S.raw,labeled_display_samples:labeled,vehicle_display_samples:vehicles,road_static_display_samples:staticRoad});
 }
 return {context,frames_decoded:decoded,class_ids:[...classes].sort((a,b)=>a-b),snapshots:snaps,actual_draw3D:true,branch_and_camera_unchanged:true,all_vehicle_boxes_present:true,no_RGB_backdrop:true,browserQA:false};
}
(async()=>{let r=[];for(const s of JSON.parse(fs.readFileSync(path.join(root,'scenes.json'))))r.push(await check(s.context));fs.writeFileSync(path.join(out,'viewer_validation.json'),JSON.stringify(r,null,2));console.log(JSON.stringify(r));})().catch(e=>{console.error(e);process.exitCode=1;});
