'use strict';
// CPU-only integration checks. The real viewer and native Canvas draw saved
// Gaussian geometry; browser script/image loading is the only simulated layer.
// Images injected below are conspicuously synthetic test fixtures, never a
// native GPU-render delivery or evidence of image quality.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),crypto=require('node:crypto');
const canvasPath=process.env.CODEX_NATIVE_CANVAS||'C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas';
const {createCanvas}=require(canvasPath);
const base=path.resolve(process.argv.slice(2).find(s=>!s.startsWith('--'))||path.join(__dirname,'../..'));
const installed=process.argv.includes('--installed');
const out=path.join(base,'outputs/GS_Scene_Diagnostics_20261027');
const folder=path.join(out,'14705303724557273004_3105_000_3125_000');
const EDITS=['v43_actor_10000','v43_motion_no_edit','v43_motion_delete_target','v43_motion_translate_lateral_1m','v43_motion_turn_left15','v43_shift_3m'];
const EXCLUDED=['v43_actor_1000','v43_actor_5000','v43_main_noedit'];
const hash=value=>crypto.createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
const plain=value=>JSON.parse(JSON.stringify(value));

function fakeDocument(){
 const nodes=new Map(),doc={nodes};
 doc.createElement=tag=>{
  const e={tagName:tag.toUpperCase(),children:[],style:{},dataset:{},classList:{toggle(){},add(){},remove(){}},value:'',checked:true,textContent:'',hidden:false,open:false,
   append(...xs){for(const x of xs){this.children.push(x);if(x&&typeof x==='object')x.parentNode=this;}},
   appendChild(x){this.append(x);return x;},insertBefore(x){this.append(x);},replaceChildren(...xs){this.children=[];this.append(...xs);},
   addEventListener(name,fn){this['on'+name]=fn;},removeEventListener(){},setAttribute(k,v){this[k]=v;},removeAttribute(k){delete this[k];},
   getAttribute(k){return this[k]??null;},getBoundingClientRect(){return {left:0,top:0,width:this.width||480,height:this.height||320};}};
  if(tag.toLowerCase()==='canvas'){
   const c=createCanvas(480,320);e.nativeCanvas=c;
   for(const dim of ['width','height'])Object.defineProperty(e,dim,{get(){return c[dim];},set(v){c[dim]=v;}});
   e.getContext=(...xs)=>c.getContext(...xs);e.toBuffer=(...xs)=>c.toBuffer(...xs);
  }
  Object.defineProperty(e,'options',{get(){return this.children;}});
  Object.defineProperty(e,'id',{get(){return this._id;},set(v){this._id=v;nodes.set(v,this);}});
  return e;
 };
 doc.head=doc.createElement('head');doc.body=doc.createElement('body');doc.createTextNode=text=>({textContent:text});
 doc.getElementById=id=>{
  if(nodes.has(id))return nodes.get(id);
  const e=doc.createElement(/Canvas$/.test(id)?'canvas':'div');e.id=id;
  if(id==='sceneCanvas'){e.width=840;e.height=560;e.getContext('2d').drawImage=()=>{throw Error('3D RGB backdrop prohibited');};}
  e.value=({geometryState:'current',pointSize:'2.5',cloudColor:'branch',semanticProfile:'gpu',semanticBoundary:'0',semanticBoundaryMass:'.2',semanticAblation:'all'})[id]||'';
  if(id==='lidar3d')e.checked=false;
  doc.body.append(e);return e;
 };
 return doc;
}

function projectionSignature(a,cloud){
 return hash(a.S.projection.filter(p=>cloud?p.cloud===cloud:true).map(p=>({cloud:p.cloud,row:a.S.clouds[p.cloud].data.ids[p.j],p:p.p,opacity:p.opacity,x:p.x,y:p.y})));
}
function coloredPixels(canvas){
 const data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data,c={green:0,unknown:0};
 for(let i=0;i<data.length;i+=4){if(data[i]===45&&data[i+1]===229&&data[i+2]===170)c.green++;if(data[i]===217&&data[i+1]===221&&data[i+2]===228)c.unknown++;}
 return c;
}

(async()=>{
 const doc=fakeDocument(),sb={console,document:doc,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Uint32Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,URL,URLSearchParams,setTimeout,clearTimeout,location:{hash:'#workspace?group=v43_actor_10000&raw=113&channel=rgb'}};
 sb.window=sb;sb.addEventListener=()=>{};vm.createContext(sb);
 const run=(file,filename=file)=>vm.runInContext(fs.readFileSync(file,'utf8'),sb,{filename});
 for(const f of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js','target_detection_data.js','bounded_unknown_data.js','r2_reference_data.js','r2_inherited_data.js','r2_inspection_data.js','r2_ownership_data.js'])run(path.join(folder,f),f);
 vm.runInContext(fs.readFileSync(path.join(out,'viewer.js'),'utf8').replace(/init\(\);\s*$/,''),sb,{filename:'viewer.js'});
 const a=sb.SceneViewer;a.S.clouds={};
 for(const[key,value]of Object.entries({...sb.CLOUD_DATA,...sb.FOREIGN_CLOUDS}))a.S.clouds[key]={...value,data:await a.unpack(value.data)};
 const geometryHash=hash(sb.EDIT_GEOMETRY.groups),evidenceHash=hash(sb.R2_OWNERSHIP49),groupsHash=hash(sb.SCENE_DATA.groups);
 const initial=new Map(),select=(id,raw=113)=>{
  a.S.group=id;a.S.raw=raw;a.S.channel='rgb';a.S.cameraView=true;a.S.cameraReference='current';a.S.selected=null;
  doc.getElementById('geometryState').value='current';doc.getElementById('semanticAblation').value='all';doc.getElementById('cloudColor').value='branch';
 };
 // Capture real unwrapped renderer output before either new UI hook is loaded.
 for(const id of EDITS)for(const raw of [98,113,114]){select(id,raw);a.draw3D();initial.set(id+':'+raw,{geometry:hash(a.editState()),projection:projectionSignature(a),actor:projectionSignature(a,'v43:10000')});}
 select('v43_actor_10000');
 vm.runInContext('show=async function(){S.epoch++;draw3D();};window.SceneViewer.show=show;',sb);
 run(path.join(out,'scene_target_detection.js'),'scene_target_detection.js');
 run(path.join(out,'scene_r2_reference.js'),'scene_r2_reference.js');
 const ownershipRuntime=installed?path.join(out,'scene_r2_inherited.js'):path.join(base,'work/v43_r2_regression_20261029/r2_ownership_runtime.js');
 run(ownershipRuntime,'r2_ownership_runtime.js');
 await Promise.resolve();
 const R=sb.R2Inherited;assert(R,'Ownership runtime must install');assert.deepEqual(plain(R.compatibleGroups),EDITS);

 const cycles=[],parameterChecks=[],snapshots=[];
 for(const id of EDITS)for(const raw of [98,113,114]){
  a.S.raw=raw;await R.switchTo(id);const expected=initial.get(id+':'+raw);
  assert.equal(R.inspection(),true,id+' must use same-frame ownership evidence');
  assert.equal(hash(a.editState()),expected.geometry,'Ownership must not change the saved edit state');
  assert.equal(projectionSignature(a),expected.projection,'Recoloring must not change row visibility, positions, opacity, or projection');
  assert.equal(projectionSignature(a,'v43:10000'),expected.actor);
  const p=R.actorPartition(),f=sb.R2_OWNERSHIP49.frames[raw];
  assert.equal(p.counts.green+p.counts.unknown,6241);
  for(let row=0;row<6241;row++){
   const role=R.rowRole('v43:10000',row);assert.equal(role.raw,raw);assert.equal(role.row,row);
   assert.equal(role.role,f.actor.default_role[row]?'vehicle':'unknown','Motion labels must retain their source row identity');
  }
  const projected=a.S.projection.filter(p=>p.cloud==='v43:10000');
  for(const p of projected){const inst=a.editState().objects.find(o=>o.key===p.instance),role=a.semanticRow(inst,p.j);assert.equal(role.role,R.rowRole(p.cloud,a.S.clouds[p.cloud].data.ids[p.j]).role);assert.equal(role.raw,raw);assert.equal(role.labels_for_training,false);}
  if(id==='v43_motion_delete_target'){
   assert.equal(a.editState().objects.length,0);assert.equal(projected.length,0);assert.equal(coloredPixels(doc.getElementById('sceneCanvas')).green,0,'Deleted actor must never be painted as a surviving green asset');
  }else{assert.equal(projected.length,6241);assert(coloredPixels(doc.getElementById('sceneCanvas')).green>0);}
  assert(doc.getElementById('geometryStateInfo').textContent.includes(sb.SCENE_DATA.groups.find(g=>g.id===id).title));
  assert(doc.getElementById('r2PartitionStatus').textContent.includes('按行携带归属'));
  assert.equal(doc.getElementById('r2InheritedPair').hidden,true,'Ordinary V43 groups must retain their own RGB panel');
  assert.equal(doc.getElementById('r2GeometryControls').hidden,false);
  cycles.push({group:id,raw,actorRows:projected.length,sourceGreen:p.counts.green,sourceUnknown:p.counts.unknown});
  if(raw===113){const dest=path.join(__dirname,'ownership_'+id+'_113.png');fs.writeFileSync(dest,doc.getElementById('sceneCanvas').toBuffer('image/png'));snapshots.push(dest);}
 }
 // Every parameter change may repartition the same rows, never alter a transform.
 for(const id of ['v43_motion_translate_lateral_1m','v43_motion_turn_left15']){
  a.S.raw=113;await R.switchTo(id);const sig=projectionSignature(a),counts=[];
  for(const radius of [0,3,5,8])for(const fraction of [.1,.2,.35]){
   doc.getElementById('r2Boundary').value=String(radius);doc.getElementById('r2BoundaryMass').value=String(fraction);doc.getElementById('r2Boundary').onchange();
   assert.equal(projectionSignature(a),sig);const q=R.actorPartition().counts;assert.equal(q.green+q.unknown,6241);
   counts.push({radius,fraction,green:q.green,unknown:q.unknown});
  }
  for(const fraction of [.1,.2,.35]){const seq=counts.filter(c=>c.fraction===fraction);assert(seq.every((c,i)=>!i||c.green<=seq[i-1].green));}
  parameterChecks.push({group:id,counts});
 }
 doc.getElementById('r2Boundary').value='5';doc.getElementById('r2BoundaryMass').value='0.2';
 const exclusions=[];
 for(const id of EXCLUDED){
  select(id);doc.getElementById('cloudColor').value='r2_candidates';a.draw3D();assert.equal(R.inspection(),false);assert.equal(doc.getElementById('cloudColor').value,'branch');assert.equal(doc.getElementById('r2GeometryControls').hidden,true);assert.equal(doc.getElementById('semanticAblation').disabled,false);
  const inst=a.editState().objects[0];assert.notEqual(inst.cloud,'v43:10000');assert.notEqual(a.semanticRow(inst,0)?.source,'Actor10K原观测逐点alpha*T语义与轮廓证据');exclusions.push({group:id,cloud:inst.cloud});
 }
 for(const channel of ['gt','sensor_overlay','semantic_2d']){select('v43_motion_turn_left15');a.S.channel=channel;a.draw3D();assert.equal(R.inspection(),false);assert.equal(a.editState(),sb.EDIT_GEOMETRY.groups.reconstruction_30000[113]);}
 select('v43_motion_turn_left15');doc.getElementById('geometryState').value='baseline';a.draw3D();assert.equal(R.inspection(),false);assert.equal(a.editState(),sb.EDIT_GEOMETRY.groups.reconstruction_30000[113]);

 // The white-panel fixture checks are added against the exported runtime API.
 const whiteRuntime=installed?path.join(out,'scene_unknown_white.js'):path.join(__dirname,'unknown_white_runtime.js');
 assert(fs.existsSync(whiteRuntime),'Unknown-white runtime is required for the integration test');
 const whiteResult=await testUnknownWhite({a,sb,doc,R,run,whiteRuntime,select});
 // Verify that the installed draw wrapper still preserves the exact same
 // projected edit geometry after white-image load/retry/epoch state changes.
 doc.getElementById('r2Boundary').value='5';doc.getElementById('r2BoundaryMass').value='0.2';
 for(const id of EDITS){a.S.raw=113;await R.switchTo(id);assert.equal(projectionSignature(a),initial.get(id+':113').projection,'White-panel hooks must preserve edited geometry and row visibility');}
 const html=fs.readFileSync(path.join(folder,'index.html'),'utf8');
 const panelIndex=html.indexOf('id="unknownWhitePanel"'),rgbIndex=html.indexOf('id="renderCanvas"'),rightIndex=html.indexOf('<span class="number">02</span>');
 assert(rgbIndex>=0&&panelIndex>rgbIndex&&rightIndex>panelIndex,'Unknown-white panel must be below the RGB image in the left panel');
 assert.equal((html.match(/id="unknownWhitePanel"/g)||[]).length,1,'Installer must not duplicate the panel');
 assert(html.indexOf('src="unknown_white_data.js')<html.indexOf('src="../viewer.js'),'Manifest data must load before the viewer');
 assert(html.indexOf('src="../scene_unknown_white.js')>html.indexOf('src="../scene_r2_inherited.js'),'White hooks must install after ownership hooks');
 assert.equal(hash(sb.EDIT_GEOMETRY.groups),geometryHash);assert.equal(hash(sb.R2_OWNERSHIP49),evidenceHash);assert.equal(hash(sb.SCENE_DATA.groups),groupsHash);
 const report={schema:'v43-unknown-white-cpu-ui-qa-v1',passed:true,runtimeVariant:installed?'installed-scene':'source',testedSourceSha256:{viewer:hash(fs.readFileSync(path.join(out,'viewer.js')).toString('utf8')),ownership:hash(fs.readFileSync(ownershipRuntime).toString('utf8')),unknownWhite:hash(fs.readFileSync(whiteRuntime).toString('utf8'))},installedPanelPlacementAndScriptOrder:true,actualCanvas:true,browserQA:false,browserImageScriptLoaderStubbed:true,sshUsed:false,gpuUsed:false,nativeGpuImageDeliveryClaimed:false,syntheticImageFixturesOnly:true,sameFrameLabelsAndEditedProjectionPreserved:true,deleteActorAbsent:true,excludedCheckpointsNotRelabelled:exclusions,sourceDataUnchanged:true,cycles,parameterChecks,snapshots,whitePanel:whiteResult};
 fs.writeFileSync(path.join(__dirname,installed?'unknown_white_installed_canvas_qa.json':'unknown_white_canvas_qa.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({passed:true,runtimeVariant:report.runtimeVariant,actualCanvas:true,cycles:cycles.length,exclusions,whitePanel:whiteResult},null,2));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});

async function testUnknownWhite({a,sb,doc,R,run,whiteRuntime,select}){
 const planPath=installed?path.join(folder,'unknown_white_manifest.json'):path.join(__dirname,'render_inputs/render_manifest.prepared.json');
 assert(fs.existsSync(planPath),'The real CPU-prepared manifest must exist');
 const diskHash=hash(fs.readFileSync(planPath).toString('utf8'));
 const prepared=JSON.parse(fs.readFileSync(planPath,'utf8')),M=plain(prepared);
 assert.equal(M.status,'prepared');assert.equal(M.gpu_executed??M.GPU_executed,false);
 assert.equal(M.verified_group_frame_count??Object.values(M.groups).flatMap(g=>Object.values(g.frames||{})).filter(f=>f.status==='verified').length,0);
 assert.equal(M.required_group_frame_count??EDITS.reduce((sum,id)=>sum+M.groups[id].frame_count,0),204);
 if(installed){run(path.join(folder,'unknown_white_data.js'),'unknown_white_data.js');assert.equal(hash(sb.UNKNOWN_WHITE_DELIVERY),hash(M),'Installed script data must exactly match the installed JSON manifest');}
 // Keep the generated prepared plan immutable on disk. The following manifest
 // changes are confined to this disposable VM and identify synthetic fixtures.
 sb.UNKNOWN_WHITE_DELIVERY=M;
 const requests=[];
 sb.__loadUnknownWhiteQA=imagePath=>{
  let resolve,reject;const promise=new Promise((res,rej)=>{resolve=res;reject=rej;});
  requests.push({path:imagePath,resolve,reject});return promise;
 };
 vm.runInContext('loadImage=function(path){return window.__loadUnknownWhiteQA(path);};',sb);
 select('v43_actor_10000');run(whiteRuntime,'unknown_white_runtime.js');
 const W=sb.UnknownWhite;assert(W,'Unknown-white runtime must install');
 const canvas=doc.getElementById('unknownWhiteCanvas'),get=id=>doc.getElementById(id);
 const alphaIsClear=()=>canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data.every((v,i)=>i%4!==3||v===0);
 const assertClear=message=>{assert.equal(W.displayed,null,message);assert.equal(canvas.hidden,true,message);assert(alphaIsClear(),message);assert.equal(get('unknownWhiteReceipt').hidden,true,message);assert(!get('unknownWhiteReceipt').href,message);};
 const flush=async()=>{await Promise.resolve();await Promise.resolve();};
 const setParams=(radius=5,fraction=.2)=>{get('r2Boundary').value=String(radius);get('r2BoundaryMass').value=String(fraction);};
 const fixtureImage=(raw,color)=>{const c=createCanvas(480,320),x=c.getContext('2d');x.fillStyle=color;x.fillRect(0,0,480,320);x.fillStyle='#000';x.font='22px sans-serif';x.fillText('SYNTHETIC UI QA ONLY',35,100);x.fillText('raw'+raw+' - no GPU image',35,135);return c;};
 const firstPixel=()=>Array.from(canvas.getContext('2d').getImageData(3,3,1,1).data);
 const fixtureEntry=(group,raw,radius=5,fraction=.2)=>({
  group_id:group,raw,status:'verified',radius,fraction,
  images:{baseline:`qa_synthetic/${group}/${raw}/baseline.png`,source:`qa_synthetic/${group}/${raw}/source.png`,unknownwhite:`qa_synthetic/${group}/${raw}/white_${radius}_${fraction}.png`},
  receipt:`qa_synthetic/${group}/${raw}/receipt.json`,floats:`qa_synthetic/${group}/${raw}/native_float.npz`,receipt_sha256:'0'.repeat(64),
  checks:{alpha_identical:true,radii_identical:true,depth_sum_identical:true,extra_identical:true,baseline_saved_png_pixels_identical:true,non_color_inputs_identical:true,known_row_SH_identical:true,row_order_unchanged:true,native_environment_unchanged:true,frozen_model_unchanged:true,actual_native_reference_max_error:0},
  synthetic_ui_fixture:true,
 });
 const addFixture=(group,raw,radius=5,fraction=.2)=>{
  M.status='partial';M.gpu_executed=true;M.synthetic_ui_fixture=true;
  M.groups[group].status='partial';M.groups[group].frames[String(raw)]=fixtureEntry(group,raw,radius,fraction);
  return M.groups[group].frames[String(raw)];
 };
 const preparedChecks=[];
 for(const group of EDITS){
  a.S.group=group;a.S.raw=113;setParams();await W.update(true);assertClear('Prepared plans must never display a pretend native image');
  assert(get('unknownWhiteStatus').textContent.includes('等待GPU'));preparedChecks.push(group);
 }
 for(const group of EXCLUDED){a.S.group=group;await W.update(true);assertClear('Unmatched checkpoint namespaces must not reuse Actor10K images');assert(get('unknownWhiteStatus').textContent.includes(prepared.groups[group].reason));}
 assert.equal(requests.length,0,'CPU-prepared and evidence-needed entries must not invoke the image loader');

 a.S.group='v43_actor_10000';a.S.raw=113;setParams();
 const entry=addFixture(a.S.group,113),q=W.lookup();
 assert.equal(W.validate(q,entry),true,'Accept the exact canonical manifest entry emitted by render_unknown_white.py');
 const rejected=[];
 for(const [label,mutate]of [
  ['group',e=>{e.group_id='v43_motion_turn_left15';}],['raw',e=>{e.raw=114;}],
  ['radius',e=>{e.radius=3;}],['fraction',e=>{e.fraction=.35;}],['entry status',e=>{e.status='prepared';}],
  ['image missing',e=>{delete e.images.unknownwhite;}],['receipt missing',e=>{delete e.receipt;}],
  ['native reference mismatch',e=>{e.checks.actual_native_reference_max_error=2.1e-6;}],['native reference check missing',e=>{delete e.checks.actual_native_reference_max_error;}],
  ...['alpha_identical','radii_identical','depth_sum_identical','extra_identical','baseline_saved_png_pixels_identical','non_color_inputs_identical','known_row_SH_identical','row_order_unchanged','native_environment_unchanged','frozen_model_unchanged'].map(k=>[k,e=>{e.checks[k]=false;}])
 ]){const bad=plain(entry);mutate(bad);assert.equal(W.validate(q,bad),false,'Reject '+label);rejected.push(label);}
 for(const [field,value]of [['status','prepared'],['context','other-context'],['schema','other-render-schema'],['gpu_executed',false]]){
  const old=M[field];M[field]=value;assert.equal(W.validate(q,entry),false,'Reject manifest '+field);M[field]=old;rejected.push('manifest '+field);
 }
 // A wrong raw in the requested frame slot cannot fall back to any other image.
 M.groups[a.S.group].frames['113']={...entry,raw:114};const countBefore=requests.length;
 await W.update(true);assertClear('Wrong raw in current frame slot must remain empty');assert.equal(requests.length,countBefore);
 M.groups[a.S.group].frames['113']=entry;

 let pending=W.update(true);let request=requests.at(-1);assertClear('Clear while loading current synthetic test image');
 request.resolve(fixtureImage(113,'#dd33aa'));await pending;
 assert.equal(canvas.hidden,false);assert.equal(W.displayed.raw,113);assert.equal(W.displayed.group,'v43_actor_10000');assert.deepEqual(firstPixel(),[221,51,170,255]);
 assert.equal(get('unknownWhiteReceipt').href,entry.receipt);
 const dest=path.join(__dirname,'qa_synthetic_unknown_white_canvas.png');fs.writeFileSync(dest,canvas.toBuffer('image/png'));
 // Control handlers draw immediately; the old matching image must disappear
 // before any replacement request can finish, even when the viewer epoch stays.
 const viewerEpoch=a.S.epoch;setParams(3,.2);get('r2Boundary').onchange();
 assert.equal(a.S.epoch,viewerEpoch);assertClear('Parameter change must synchronously clear the old image and receipt');
 assert(get('unknownWhiteStatus').textContent.includes('旧参数图片已隐藏'));
 setParams();get('r2Boundary').onchange();pending=W.pending;request=requests.at(-1);
 // Change to an unavailable parameter while an old image is still in flight.
 setParams(8,.35);get('r2BoundaryMass').onchange();assertClear('Unmatched parameter must clear an in-flight image');
 request.resolve(fixtureImage(113,'#dd33aa'));await pending;assertClear('Stale response cannot restore old parameter pixels');

 // Same-frame source labels are valid for motion, but every displayed native
 // image must keep its own group, raw and parameter tuple across async races.
 setParams();a.S.group='v43_motion_no_edit';a.S.raw=113;addFixture(a.S.group,113);
 const oldRawPromise=W.update(true),oldRawRequest=requests.at(-1);
 a.S.raw=114;addFixture(a.S.group,114);const newRawPromise=W.update(),newRawRequest=requests.at(-1);
 assert.notEqual(oldRawRequest.path,newRawRequest.path);assertClear('Changing raw clears old pixels before replacement');
 newRawRequest.resolve(fixtureImage(114,'#3388ee'));await newRawPromise;assert.equal(W.displayed.raw,114);assert.deepEqual(firstPixel(),[51,136,238,255]);
 oldRawRequest.resolve(fixtureImage(113,'#dd33aa'));await oldRawPromise;assert.equal(W.displayed.raw,114);assert.deepEqual(firstPixel(),[51,136,238,255],'Out-of-order old raw cannot overwrite the current frame');

 a.S.raw=113;const oldGroupPromise=W.update(true),oldGroupRequest=requests.at(-1);
 a.S.group='v43_motion_turn_left15';addFixture(a.S.group,113);const showPromise=a.show();await flush();const newGroupRequest=requests.at(-1);
 assertClear('Show hook must clear the prior edit image synchronously');assert.notEqual(oldGroupRequest.path,newGroupRequest.path);
 oldGroupRequest.resolve(fixtureImage(113,'#dd33aa'));await oldGroupPromise;assertClear('Old group completion cannot paint the newly selected edit');
 newGroupRequest.resolve(fixtureImage(113,'#22bb55'));await showPromise;await W.pending;
 assert.equal(W.displayed.group,'v43_motion_turn_left15');assert.deepEqual(firstPixel(),[34,187,85,255]);

 pending=W.update(true);request=requests.at(-1);setParams(5,.35);addFixture(a.S.group,113,5,.35);
 const parameterPromise=W.update(),parameterRequest=requests.at(-1);request.resolve(fixtureImage(113,'#22bb55'));await pending;assertClear('Old fraction response must not paint while the new fraction is loading');
 parameterRequest.resolve(fixtureImage(113,'#ee9933'));await parameterPromise;assert.equal(W.displayed.fraction,.35);assert.deepEqual(firstPixel(),[238,153,51,255]);

 // Force-retrying the exact same tuple must still give the newest request an
 // independent epoch. Matching only group/frame/parameters would be too weak.
 const reloadOld=W.update(true),reloadOldRequest=requests.at(-1),reloadNew=W.update(true),reloadNewRequest=requests.at(-1);
 reloadNewRequest.resolve(fixtureImage(113,'#3388ee'));await reloadNew;assert.deepEqual(firstPixel(),[51,136,238,255]);
 reloadOldRequest.resolve(fixtureImage(113,'#dd33aa'));await reloadOld;assert.deepEqual(firstPixel(),[51,136,238,255],'An older same-tuple retry cannot overwrite the newest epoch');

 // Decode failures leave no prior image or receipt; an explicit retry can load
 // the same tuple again without retaining a rejected loader state.
 pending=W.update(true);request=requests.at(-1);request.reject(Error('synthetic decoder failure'));await pending;assertClear('Decode failure must preserve an empty panel');assert(get('unknownWhiteStatus').textContent.includes('加载失败'));
 const retryPromise=get('unknownWhiteRetry').onclick();request=requests.at(-1);request.resolve(fixtureImage(113,'#ee9933'));await retryPromise;assert.equal(W.displayed.fraction,.35);
 a.S.channel='gt';await W.update();assertClear('Reference RGB must not borrow an edited native image');
 a.S.channel='rgb';get('geometryState').value='baseline';await W.update();assertClear('Baseline geometry must not borrow an edited native image');
 get('geometryState').value='current';
 for(const group of EXCLUDED){a.S.group=group;await W.update();assertClear('1K, 5K and V1 remain evidence-needed after other groups load');}
 a.S.group='v43_motion_delete_target';a.S.raw=113;await W.update();assertClear('No fallback from a turn or no-edit image into the delete group');
 assert.equal(hash(fs.readFileSync(planPath).toString('utf8')),diskHash,'Synthetic fixture mutation must never change the prepared manifest file');
 for(const key of Object.keys(M))delete M[key];Object.assign(M,plain(prepared));await W.update(true);assertClear('Restore the CPU-only prepared plan after synthetic QA');
 return {preparedOnlyNoImageLoader:true,readyGroups:preparedChecks,canonicalRendererEntryAccepted:true,rejectedManifestCases:rejected,wrongRawSlotNoFallback:true,parameterChangeClearsImageAndReceipt:true,staleRawGroupAndParameterResponsesDiscarded:true,olderSameTupleRetryDiscarded:true,decodeFailureAndRetry:true,excludedNamespacesStayUnavailable:true,syntheticFixturesInMemoryOnly:true,preparedManifestFileUnchanged:true,syntheticCanvasSnapshot:dest};
}
