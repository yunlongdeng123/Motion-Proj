const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const {createCanvas,loadImage}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const folder=path.resolve('outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000'),out=path.resolve('outputs/V43_Ownership_20261027/gpu_v43'),nodes=new Map(),canvas=createCanvas(840,560);let sb;
function get(id){if(id==='sceneCanvas')return canvas;if(!nodes.has(id))nodes.set(id,{checked:id!=='lidar3d',value:({cloudColor:'semantic_branch',pointSize:'2',geometryState:'current',semanticProfile:'gpu',semanticBoundary:'5',semanticBoundaryMass:'.2',semanticAblation:'all'})[id]||'',textContent:''});return nodes.get(id);}
sb={console,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,document:{getElementById:get,createElement(){return {}},head:{append(n){queueMicrotask(()=>{try{vm.runInContext(fs.readFileSync(path.join(folder,n.src),'utf8'),sb,{filename:path.basename(n.src)});n.onload()}catch(e){console.error(e.message);n.onerror()}})}}}};sb.window=sb;vm.createContext(sb);
for(const f of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js'])vm.runInContext(fs.readFileSync(path.join(folder,f),'utf8'),sb,{filename:f});
vm.runInContext(fs.readFileSync(path.resolve('outputs/GS_Scene_Diagnostics_20261027/viewer.js'),'utf8').replace(/init\(\);\s*$/,''),sb);
(async()=>{const a=sb.SceneViewer;a.S.clouds={};for(const [k,v]of Object.entries({...sb.CLOUD_DATA,...sb.FOREIGN_CLOUDS}))a.S.clouds[k]={...v,data:await a.unpack(v.data)};a.S.raw=113;a.S.group='v43_shift_3m';a.S.cameraView=true;await a.ensureSemantic();const records=[];
 const gallery=createCanvas(1680,1200),x=gallery.getContext('2d');x.fillStyle='#08111c';x.fillRect(0,0,1680,1200);
 let slot=0;for(const radius of ['0','5'])for(const mode of ['all','hide_unknown','hide_vehicle','hide_both']){
  get('semanticBoundary').value=radius;get('semanticAblation').value=mode;a.draw3D();const count=a.S.projection.length;
  for(const p of a.S.projection){const inst=a.geometryInstances().find(i=>i.key===p.instance),e=a.semanticRow(inst,p.j);assert(!sb.semanticFilterReject(e));if(mode==='hide_both')assert(e.class_id>0&&(e.class_id<14||e.class_id>19));}
  records.push({radius:Number(radius),mode,displayed_samples:count,...a.S.semanticFilterStats});
  if(radius==='5'){x.drawImage(canvas,(slot%2)*840,Math.floor(slot/2)*600+32);x.fillStyle='white';x.font='17px sans-serif';x.fillText('raw113 / 5px / '+mode+' / geometry centers only',(slot%2)*840+10,Math.floor(slot/2)*600+24);slot++;}
 }
 fs.writeFileSync(path.join(out,'unknown_geometry_ablation.png'),gallery.toBuffer('image/png'));
 assert(records.find(r=>r.radius===0&&r.mode==='all').unknown>0);
 assert(records.find(r=>r.radius===5&&r.mode==='all').unknown>=records.find(r=>r.radius===0&&r.mode==='all').unknown);
 assert(records.filter(r=>r.mode==='hide_both').every(r=>r.hidden>0));
 get('semanticAblation').value='all';get('cloudColor').value='semantic_unknown_reason';a.draw3D();fs.writeFileSync(path.join(out,'unknown_reason_geometry.png'),canvas.toBuffer('image/png'));
 fs.writeFileSync(path.join(out,'unknown_preview_validation.json'),JSON.stringify({gpu_calls:0,training_updates:0,actual_canvas_runtime:true,browserQA:false,rgb_ablation_run:false,source_RGB_unchanged:true,records},null,2));
 console.log(JSON.stringify(records));
})().catch(e=>{console.error(e.message);process.exitCode=1});
