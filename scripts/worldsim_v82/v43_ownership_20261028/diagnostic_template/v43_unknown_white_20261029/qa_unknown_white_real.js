'use strict';
// Read-only acceptance of the *published real* unknown-white delivery.
// No synthetic images, manifest mutation, SSH, CUDA, or rendering jobs.
// The browser filesystem script transport is adapted; the installed viewer's
// own loadImage/imageOn path decodes the published PNG.js bytes into real Canvas.
// Usage: node qa_unknown_white_real.js --real-manifest <scene/unknown_white_manifest.json> --expect-count 204
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),zlib=require('node:zlib');
const assert=require('node:assert/strict'),crypto=require('node:crypto');
const nativePath=process.env.CODEX_NATIVE_CANVAS||'C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas';
const {createCanvas,loadImage:decodeImage,Image}=require(nativePath);
const CONTEXT='14705303724557273004_3105_000_3125_000';
const GROUPS=['v43_actor_10000','v43_motion_no_edit','v43_motion_delete_target','v43_motion_translate_lateral_1m','v43_motion_turn_left15','v43_shift_3m'];
const EXCLUDED=['v43_actor_1000','v43_actor_5000','v43_main_noedit'];
const CHECKS=['alpha_identical','radii_identical','depth_sum_identical','extra_identical','baseline_saved_png_pixels_identical','non_color_inputs_identical','known_row_SH_identical','row_order_unchanged','native_environment_unchanged','frozen_model_unchanged'];
const argv=process.argv.slice(2),argument=name=>{const i=argv.indexOf(name);return i<0?undefined:argv[i+1]&&!argv[i+1].startsWith('--')?argv[i+1]:'';};
const base=path.resolve(argument('--root')||path.join(__dirname,'../..'));
const out=path.join(base,'outputs/GS_Scene_Diagnostics_20261027'),scene=path.join(out,CONTEXT);
const realArgument=argument('--real-manifest'),manifestPath=path.resolve(realArgument||path.join(scene,'unknown_white_manifest.json'));
const expectCount=Number(argument('--expect-count')||204);
const reportPath=path.resolve(argument('--report')||path.join(__dirname,`real_manifest_canvas_qa_${expectCount}.json`));
const sha=value=>crypto.createHash('sha256').update(value).digest('hex');
const hashFile=file=>sha(fs.readFileSync(file));
const objectHash=value=>sha(JSON.stringify(value));
const plain=value=>JSON.parse(JSON.stringify(value));
const readJson=file=>JSON.parse(fs.readFileSync(file,'utf8').replace(/^\uFEFF/,''));
const range=(a,b)=>Array.from({length:b-a+1},(_,i)=>a+i);

function localFile(root,relative){
 assert.equal(typeof relative,'string','Asset path must be a string');
 assert(relative&&!path.isAbsolute(relative)&&!/^\w+:/.test(relative),'Asset path must be relative to the published scene');
 const resolved=path.resolve(root,relative),rel=path.relative(root,resolved);
 assert(rel&&!rel.startsWith('..')&&!path.isAbsolute(rel),'Asset path escapes the published scene: '+relative);
 assert(!/qa_synthetic|synthetic_ui_fixture/i.test(relative),'Synthetic asset paths cannot pass real delivery QA');
 return resolved;
}

function fakeDocument(){
 const nodes=new Map(),doc={nodes};
 doc.createElement=tag=>{
  const e={tagName:tag.toUpperCase(),children:[],style:{},dataset:{},classList:{toggle(){},add(){},remove(){}},value:'',checked:true,textContent:'',hidden:false,
   append(...xs){for(const x of xs){this.children.push(x);if(x&&typeof x==='object')x.parentNode=this;}},appendChild(x){this.append(x);return x;},
   insertBefore(x){this.append(x);},replaceChildren(...xs){this.children=[];this.append(...xs);},addEventListener(name,fn){this['on'+name]=fn;},removeEventListener(){},
   setAttribute(k,v){this[k]=v;},removeAttribute(k){delete this[k];},getAttribute(k){return this[k]??null;},getBoundingClientRect(){return {left:0,top:0,width:this.width||480,height:this.height||320};}};
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
  if(id==='sceneCanvas'){e.width=840;e.height=560;e.getContext('2d').drawImage=()=>{throw Error('A real 3D projection cannot use a pixel backdrop');};}
  e.value=({geometryState:'current',pointSize:'2.5',cloudColor:'branch',semanticProfile:'gpu',semanticBoundary:'0',semanticBoundaryMass:'.2',semanticAblation:'all'})[id]||'';
  if(id==='lidar3d')e.checked=false;
  doc.body.append(e);return e;
 };
 return doc;
}

function pixelsOf(image){
 assert.equal(image.width,480);assert.equal(image.height,320);
 const c=createCanvas(480,320);c.getContext('2d').drawImage(image,0,0);return c.getContext('2d').getImageData(0,0,480,320).data;
}
function assertCanvasImage(canvas,expectedPixels,pixel){
 assert.equal(canvas.width,480);assert.equal(canvas.height,320);
 const actual=canvas.getContext('2d').getImageData(0,0,480,320).data;
 for(let y=0;y<320;y++)for(let x=0;x<480;x++){
  if(Math.abs(x-pixel[0])<=9&&Math.abs(y-pixel[1])<=9)continue; // actual viewer crosshair
  const i=(y*480+x)*4;
  for(let c=0;c<4;c++)if(actual[i+c]!==expectedPixels[i+c])assert.fail(`Canvas does not display this exact published frame at (${x},${y}), channel ${c}`);
 }
}
function projectedSignature(a){return objectHash(a.S.projection.map(p=>({cloud:p.cloud,row:a.S.clouds[p.cloud].data.ids[p.j],p:p.p,x:p.x,y:p.y,opacity:p.opacity})));}

function verifyPanelSlot(sceneDirectory){
 const html=fs.readFileSync(path.join(sceneDirectory,'index.html'),'utf8');
 const rgb=html.indexOf('id="renderCanvas"'),metrics=html.indexOf('id="frameMetrics"'),panel=html.indexOf('id="unknownWhitePanel"'),canvas=html.indexOf('id="unknownWhiteCanvas"');
 const leftStart=html.lastIndexOf('<section class="panel"',rgb),rightStart=html.indexOf('<section class="panel"',rgb+1),panelEnd=html.indexOf('<!-- unknown-white-panel-end -->',panel);
 assert(leftStart>=0&&leftStart<rgb&&rgb<metrics&&metrics<panel&&panel<canvas&&canvas<panelEnd&&panelEnd<rightStart,'Unknown-white slot must be in the left RGB panel, below its metrics and before the 3D panel');
 assert(html.slice(panelEnd,rightStart).includes('</section>'),'Unknown-white must remain inside the outer RGB panel');
 assert(html.slice(rightStart,rightStart+150).includes('<span class="number">02</span>'),'The following panel must be the existing 3D panel');
 assert.equal((html.match(/id="unknownWhitePanel"/g)||[]).length,1);assert.equal((html.match(/id="unknownWhiteCanvas"/g)||[]).length,1);
 assert(html.indexOf('src="unknown_white_data.js')<html.indexOf('src="../viewer.js'));
 assert(html.indexOf('src="../scene_unknown_white.js')>html.indexOf('src="../scene_r2_inherited.js'));
 return {passed:true,evidence:'Installed HTML structure; no browser layout claim',region:'Left native RGB panel below metrics, before existing 3D panel',singlePanelAndCanvas:true,insideRGBPanel:true,scriptOrderCorrect:true,htmlSha256:sha(html)};
}

// Read only two small ID arrays from NumPy's ZIP, without loading the saved
// multi-megabyte float RGB/radii arrays again. Full NPZ SHA binds this read to
// the native receipt; the publisher separately verifies all float invariants.
function npzIntegerArrays(buffer,wanted){
 let end=-1;for(let i=buffer.length-22;i>=Math.max(0,buffer.length-65557);i--)if(buffer.readUInt32LE(i)===0x06054b50){end=i;break;}
 assert(end>=0,'NPZ ZIP directory is missing');
 const entries=buffer.readUInt16LE(end+10),start=buffer.readUInt32LE(end+16),result={};let at=start;
 for(let n=0;n<entries;n++){
  assert.equal(buffer.readUInt32LE(at),0x02014b50,'Invalid NPZ central directory');
  const method=buffer.readUInt16LE(at+10),compressed=buffer.readUInt32LE(at+20),uncompressed=buffer.readUInt32LE(at+24);
  const nameLength=buffer.readUInt16LE(at+28),extraLength=buffer.readUInt16LE(at+30),commentLength=buffer.readUInt16LE(at+32),offset=buffer.readUInt32LE(at+42);
  const name=buffer.toString('utf8',at+46,at+46+nameLength);at+=46+nameLength+extraLength+commentLength;
  const key=name.replace(/\.npy$/,'');if(!wanted.includes(key))continue;
  assert.equal(buffer.readUInt32LE(offset),0x04034b50,'Invalid NPZ local header');
  const dataOffset=offset+30+buffer.readUInt16LE(offset+26)+buffer.readUInt16LE(offset+28),packed=buffer.subarray(dataOffset,dataOffset+compressed);
  assert(method===0||method===8,'Unsupported NPZ compression');const data=method===0?packed:zlib.inflateRawSync(packed);
  assert.equal(data.length,uncompressed);assert.equal(data.subarray(0,6).toString('binary'),'\x93NUMPY');
  const major=data[6],headerLength=major===1?data.readUInt16LE(8):data.readUInt32LE(8),headOffset=major===1?10:12;
  assert([1,2,3].includes(major),'Unsupported NPY format');
  const header=data.toString('latin1',headOffset,headOffset+headerLength),type=header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/),shape=header.match(/['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*\)/);
  assert(type&&['<i8','|i8','<i4','|i4'].includes(type[1])&&shape,'Unknown row IDs must be a one-dimensional integer array');
  assert(/['"]fortran_order['"]\s*:\s*False/.test(header));
  const count=Number(shape[1]),begin=headOffset+headerLength,width=type[1].endsWith('8')?8:4;assert.equal(data.length,begin+count*width);
  result[key]=Array.from({length:count},(_,i)=>width===8?Number(data.readBigInt64LE(begin+i*width)):data.readInt32LE(begin+i*width));
 }
 for(const key of wanted)assert(result[key], 'Missing native row IDs: '+key);
 return result;
}

async function main(){
 assert.notEqual(realArgument,undefined,'Pass --real-manifest explicitly; prepared/synthetic fixture QA is a separate command');
 assert(Number.isInteger(expectCount)&&expectCount>0&&expectCount<=204,'Invalid --expect-count');
 assert.equal(path.resolve(path.dirname(manifestPath)),path.resolve(scene),'Use the actual published scene manifest after publish_unknown_white.py creates PNG.js wrappers');
 const initialManifestHash=hashFile(manifestPath),M=readJson(manifestPath);
 const installedPanelSlot=verifyPanelSlot(scene);
 assert.equal(M.schema,'v43-native-unknown-white-v1');assert.equal(M.gpu_executed,true);assert.equal(M.final_frozen_state_verified,true);assert.equal(M.training_updates,0);
 assert.equal(M.synthetic_ui_fixture,undefined);assert(['complete','partial'].includes(M.status));assert.equal(M.required_group_frame_count,204);assert.equal(M.verified_group_frame_count,expectCount);
 assert.equal(M.status,expectCount===204?'complete':'partial');
 assert.equal(M.publication?.validated_on_CPU,true,'Actual CPU-validated publication is required');
 assert.equal(M.publication?.no_GPU_or_SSH_used_by_publisher,true);
 assert.equal(M.publication.versioned_asset_prefix,'assets/v43_unknown_white/'+M.publication.source_render_manifest_sha256.slice(0,16));
 assert.equal(hashFile(path.join(scene,'r2_ownership_data.js')),M.evidence_sha256,'Current UI evidence must be the exact evidence used by native rendering');
 assert.equal(M.classification.known_actor_color,'native RGB/SH, never semantic green');
 assert.equal(M.classification.labels_for_training,false);assert.equal(M.classification.radius,5);assert.equal(M.classification.fraction,.2);
 assert.equal(Object.values(M.groups).reduce((sum,g)=>sum+Object.keys(g.frames||{}).length,0),expectCount);
 for(const group of EXCLUDED){assert.equal(M.groups[group].status,'evidence_needed');assert.equal(Object.keys(M.groups[group].frames).length,0);}
 if(expectCount===204)for(const group of GROUPS){assert.equal(M.groups[group].status,'complete');assert.deepEqual(Object.keys(M.groups[group].frames).map(Number).sort((a,b)=>a-b),range(group==='v43_actor_10000'?77:86,group==='v43_actor_10000'?125:116));}
 else for(const group of GROUPS)assert(M.groups[group].frames['113'],'A six-group smoke must include raw113 in every supported group');

 const doc=fakeDocument(),get=id=>doc.getElementById(id),sb={console,document:doc,Image,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Uint32Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,URL,URLSearchParams,setTimeout,clearTimeout,location:{hash:'#workspace?group=v43_actor_10000&raw=113&channel=rgb'}};
 sb.window=sb;sb.addEventListener=()=>{};vm.createContext(sb);
 const run=(file,filename=file)=>vm.runInContext(fs.readFileSync(file,'utf8'),sb,{filename,timeout:10000});
 for(const file of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js','target_detection_data.js','bounded_unknown_data.js','r2_reference_data.js','r2_inherited_data.js','r2_inspection_data.js','r2_ownership_data.js','unknown_white_data.js'])run(path.join(scene,file),file);
 assert.equal(objectHash(sb.UNKNOWN_WHITE_DELIVERY),objectHash(M),'Installed data.js must match the published manifest');
 vm.runInContext(fs.readFileSync(path.join(out,'viewer.js'),'utf8').replace(/init\(\);\s*$/,''),sb,{filename:'viewer.js'});
 const a=sb.SceneViewer,geometryHash=objectHash(sb.EDIT_GEOMETRY.groups),evidenceHash=objectHash(sb.R2_OWNERSHIP49);a.S.clouds={};
 for(const cloud of ['static','v43:10000']){const value=({...sb.CLOUD_DATA,...sb.FOREIGN_CLOUDS})[cloud];a.S.clouds[cloud]={...value,data:await a.unpack(value.data)};}
 const sourceScripts=[];sb.IMAGE_DATA=sb.IMAGE_DATA||{};
 sb.__loadLocalScript=async relative=>{const file=localFile(scene,relative);sourceScripts.push(relative);run(file,relative);};
 vm.runInContext("loadScript=async function(path){return window.__loadLocalScript(path);};show=async function(){const epoch=++S.epoch,f=selected().frames[S.raw];S.selected=null;S.baseImage=null;if(f?.channels.rgb){const im=await loadImage(f.channels.rgb);if(epoch!==S.epoch)return;S.baseImage=im;imageOn($('renderCanvas'),im);}draw3D();};window.SceneViewer.show=show;",sb);
 a.S.group='v43_actor_10000';a.S.raw=113;a.S.channel='rgb';a.S.cameraView=true;a.S.cameraReference='current';
 run(path.join(out,'scene_target_detection.js'),'scene_target_detection.js');run(path.join(out,'scene_r2_reference.js'),'scene_r2_reference.js');run(path.join(out,'scene_r2_inherited.js'),'scene_r2_inherited.js');
 const R=sb.R2Inherited;assert(R);
 const expectedByRaw=new Map(),assetRecords=[],pixelCache=new Map();let totalFiles=0,totalBytes=0;
 function expectedFor(raw){
  if(expectedByRaw.has(raw))return expectedByRaw.get(raw);a.S.raw=raw;
  const partition=R.actorPartition(),actorUnknown=[],actorKnown=[];partition.roles.forEach((r,i)=>(r.role==='unknown'?actorUnknown:actorKnown).push(i));
  const expected={actorUnknown,actorKnown,counts:plain(partition.counts),staticUnknown:plain(sb.R2_OWNERSHIP49.frames[raw].static.final_removed_ids)};
  assert.equal(actorKnown.length+actorUnknown.length,6241);expectedByRaw.set(raw,expected);return expected;
 }
 for(const group of GROUPS){
  let checked=0;
  for(const [rawString,entry]of Object.entries(M.groups[group].frames)){
   const raw=Number(rawString),expected=expectedFor(raw),actorVisible=group!=='v43_motion_delete_target';
   assert.equal(entry.synthetic_ui_fixture,undefined);assert.equal(entry.group_id,group);assert.equal(entry.raw,raw);assert.equal(entry.status,'verified');assert.equal(entry.radius,5);assert.equal(entry.fraction,.2);
   assert(entry.images.unknownwhite.startsWith(M.publication.versioned_asset_prefix+'/'),'Real image path must bind to the published render-manifest version');
   const receiptPath=localFile(scene,entry.receipt);assert.equal(hashFile(receiptPath),entry.receipt_sha256);const receipt=readJson(receiptPath);
   for(const key of ['group_id','raw','radius','fraction'])assert.equal(receipt[key],entry[key]);
   assert.equal(receipt.schema,M.schema);assert.equal(receipt.context,CONTEXT);assert.equal(receipt.status,'verified');assert.equal(receipt.gpu_executed,true);assert.equal(receipt.training_updates,0);assert.equal(receipt.synthetic_ui_fixture,undefined);
   for(const key of ['actor_checkpoint_ref','actor_checkpoint_sha256','static_namespace','actor_namespace'])assert.equal(receipt[key],M[key]);
   assert.equal(receipt.ownership_data_sha256,M.evidence_sha256);
   assert.deepEqual(receipt.classification,M.classification);assert.deepEqual(receipt.checks,entry.checks);
   for(const check of CHECKS)assert.equal(receipt.checks[check],true,'Failed real receipt check: '+check);
   assert(Number.isFinite(receipt.checks.actual_native_reference_max_error)&&receipt.checks.actual_native_reference_max_error>=0&&receipt.checks.actual_native_reference_max_error<=2e-6);
   for(const key of ['pixel_painting','row_deletion','local_scope_clipping'])assert.equal(receipt[key],false);
   assert.equal(receipt.actor_visible,actorVisible);assert.equal(receipt.active_input_rows,918850+(actorVisible?6241:0));
   assert.equal(receipt.static_unknown_rows,expected.staticUnknown.length);assert.equal(receipt.actor_unknown_rows,actorVisible?expected.actorUnknown.length:0);assert.deepEqual(receipt.actor_partition_counts,expected.counts);
   const buffers={},images={};
   for(const filename of ['baseline.png','unknownwhite.png','source.png','native_float.npz']){
    const file=path.join(path.dirname(receiptPath),filename),metadata=receipt.files[filename],buffer=fs.readFileSync(file);
    assert.equal(buffer.length,metadata.bytes);assert.equal(sha(buffer),metadata.sha256,'Published file checksum differs: '+file);buffers[filename]=buffer;totalFiles++;totalBytes+=buffer.length;
    if(filename.endsWith('.png')){
     assert.equal(buffer.subarray(0,8).toString('hex'),'89504e470d0a1a0a');images[filename]=pixelsOf(await decodeImage(buffer));
     const channel=filename.slice(0,-4);assert.equal(localFile(scene,entry.images[channel]),file);
    }
   }
   assert.equal(localFile(scene,entry.floats),path.join(path.dirname(receiptPath),'native_float.npz'));
   assert.deepEqual(images['baseline.png'],images['source.png'],'Baseline pixels must remain the actual original native RGB');
   const sourceRoute=sb.SCENE_DATA.groups.find(g=>g.id===group).frames[raw].channels.rgb,sourceFile=localFile(scene,sourceRoute);
   assert.equal(hashFile(sourceFile),receipt.source_rgb_sha256);assert.equal(receipt.source_rgb_path,sourceRoute);assert.equal(sha(buffers['source.png']),receipt.source_rgb_sha256);
   const wrapperPath=localFile(scene,entry.images.unknownwhite+'.js'),wrapperBytes=fs.readFileSync(wrapperPath),wrapperContext={window:{IMAGE_DATA:{}}};
   vm.runInNewContext(wrapperBytes.toString('utf8'),wrapperContext,{filename:wrapperPath,timeout:2000});
   assert.deepEqual(Object.keys(wrapperContext.window.IMAGE_DATA),[entry.images.unknownwhite]);
   const uri=wrapperContext.window.IMAGE_DATA[entry.images.unknownwhite];assert.equal(typeof uri,'string');assert(uri.startsWith('data:image/png;base64,'));
   assert.deepEqual(Buffer.from(uri.slice('data:image/png;base64,'.length),'base64'),buffers['unknownwhite.png'],'Published PNG.js must contain the exact verified PNG bytes');
   totalFiles++;totalBytes+=wrapperBytes.length;
   const ids=npzIntegerArrays(buffers['native_float.npz'],['static_unknown_ids','actor_unknown_ids']);
   assert.deepEqual(ids.static_unknown_ids,expected.staticUnknown,'Native rendered static unknown IDs must equal current JS gray rows');
   assert.deepEqual(ids.actor_unknown_ids,actorVisible?expected.actorUnknown:[],'Native rendered actor unknown IDs must equal current JS gray rows');
   let changedPixels=0;for(let i=0;i<images['baseline.png'].length;i+=4)if([0,1,2].some(c=>images['baseline.png'][i+c]!==images['unknownwhite.png'][i+c]))changedPixels++;
   assert(changedPixels>0,'Real unknown-white output must contain an observable native color change');
   assetRecords.push({group,raw,changedPixels,staticUnknown:expected.staticUnknown.length,actorKnown:actorVisible?expected.actorKnown.length:0,actorUnknown:ids.actor_unknown_ids.length,staticUnknownIdsSha256:objectHash(ids.static_unknown_ids),actorUnknownIdsSha256:objectHash(ids.actor_unknown_ids),actorKnownIdsSha256:objectHash(actorVisible?expected.actorKnown:[]),imageSha256:sha(buffers['unknownwhite.png']),wrapperSha256:sha(wrapperBytes)});
   // Retain only frames exercised by Canvas navigation, keeping memory bounded.
   const available=Object.keys(M.groups[group].frames).map(Number).sort((a,b)=>a-b);
   if(new Set([available[0],113,114,available.at(-1)]).has(raw))pixelCache.set(group+':'+raw,{white:images['unknownwhite.png'],source:images['source.png']});
   checked++;
  }
  console.log(JSON.stringify({validatedRealAssets:group,frames:checked}));
 }
 assert.equal(assetRecords.length,expectCount);

 a.S.raw=113;a.S.group='v43_actor_10000';get('r2Boundary').value='5';get('r2BoundaryMass').value='0.2';
 run(path.join(out,'scene_unknown_white.js'),'scene_unknown_white.js');const W=sb.UnknownWhite;assert(W);await W.pending;
 const whiteCanvas=get('unknownWhiteCanvas'),navigation=[],parameterChecks=[],snapshots=[];
 function assertClear(){assert.equal(W.displayed,null);assert.equal(whiteCanvas.hidden,true);assert.equal(get('unknownWhiteReceipt').hidden,true);assert(!get('unknownWhiteReceipt').href);assert(whiteCanvas.getContext('2d').getImageData(0,0,480,320).data.every((v,i)=>i%4!==3||v===0));}
 for(const group of GROUPS){
  const available=Object.keys(M.groups[group].frames).map(Number).sort((a,b)=>a-b),samples=[...new Set([available[0],113,114,available.at(-1)])].filter(raw=>available.includes(raw));
  for(const raw of samples){
   a.S.raw=raw;await R.switchTo(group);await W.pending;const entry=M.groups[group].frames[String(raw)],expected=pixelCache.get(group+':'+raw);assert(expected);
   assert.equal(R.inspection(),true);assert.equal(whiteCanvas.hidden,false);assert.equal(W.displayed.group,group);assert.equal(W.displayed.raw,raw);assert.equal(W.displayed.path,entry.images.unknownwhite);assert.equal(W.displayed.radius,5);assert.equal(W.displayed.fraction,.2);
   assertCanvasImage(whiteCanvas,expected.white,a.S.pixel);assertCanvasImage(get('renderCanvas'),expected.source,a.S.pixel);
   assert.equal(a.S.baseImage.evidencePath,sb.SCENE_DATA.groups.find(g=>g.id===group).frames[raw].channels.rgb);assert.equal(get('unknownWhiteReceipt').href,entry.receipt);assert.equal(get('unknownWhiteReceipt').hidden,false);
   assert(get('unknownWhitePairNote').textContent.includes('其余高斯保留原颜色'));assert(get('unknownWhiteStatus').textContent.includes('仅改unknown的SH颜色'));
   const actorRows=a.S.projection.filter(p=>p.cloud==='v43:10000'),unknownSet=new Set(expectedFor(raw).actorUnknown);assert.equal(actorRows.length,group==='v43_motion_delete_target'?0:6241);
   for(const p of actorRows){const id=a.S.clouds[p.cloud].data.ids[p.j];assert.equal(R.rowRole(p.cloud,id).role,unknownSet.has(id)?'unknown':'vehicle');}
   navigation.push({group,raw,path:entry.images.unknownwhite,exactRealPixels:true,projectedActorRows:actorRows.length});
   if(raw===113){const whitePath=path.join(__dirname,'real_unknownwhite_'+group+'_113.png');fs.writeFileSync(whitePath,whiteCanvas.toBuffer('image/png'));snapshots.push(whitePath);}
  }
  a.S.raw=113;await R.switchTo(group);await W.pending;const before=projectedSignature(a),viewerEpoch=a.S.epoch;
  get('r2Boundary').value='3';get('r2Boundary').onchange();assertClear();assert.equal(a.S.epoch,viewerEpoch);assert.equal(projectedSignature(a),before);
  get('r2Boundary').value='5';get('r2Boundary').onchange();await W.pending;assert.equal(W.displayed.group,group);assert.equal(W.displayed.raw,113);assertCanvasImage(whiteCanvas,pixelCache.get(group+':113').white,a.S.pixel);
  get('r2BoundaryMass').value='0.35';get('r2BoundaryMass').onchange();assertClear();assert.equal(projectedSignature(a),before);
  get('r2BoundaryMass').value='0.2';get('r2BoundaryMass').onchange();await W.pending;assertCanvasImage(whiteCanvas,pixelCache.get(group+':113').white,a.S.pixel);
  parameterChecks.push({group,radiusChangeClearsActualImage:true,fractionChangeClearsActualImage:true,restoredExactPublishedPixels:true,geometryPreserved:true});
 }
 assert.equal(hashFile(manifestPath),initialManifestHash,'Real QA must not mutate delivery manifest');
 assert.equal(objectHash(sb.EDIT_GEOMETRY.groups),geometryHash);assert.equal(objectHash(sb.R2_OWNERSHIP49),evidenceHash);assert.equal(objectHash(W.manifest),objectHash(M));
 const report={schema:'v43-real-unknown-white-canvas-qa-v1',passed:true,actualCanvas:true,browserQA:false,images:'actual published native PNG bytes; no synthetic fixtures',filesystemScriptTransport:true,sshUsed:false,gpuUsedByQA:false,trainingUpdatesByQA:0,manifest:manifestPath,manifestSha256:initialManifestHash,status:M.status,verifiedEntries:expectCount,assetFilesChecked:totalFiles,assetBytesChecked:totalBytes,allPNGJsExactBytes:true,allNativeUnknownIDsMatchCurrentJS:true,installedPanelSlot,knownActorColorEvidence:{classification:M.classification.known_actor_color,receiptKnownRowSHChecks:true,baselineSourcePixelsExact:true,scope:'Known-row SH inputs are preserved; composited pixels can change where unknown overlaps.'},testedSourceSha256:{viewer:hashFile(path.join(out,'viewer.js')),ownership:hashFile(path.join(out,'scene_r2_inherited.js')),unknownWhite:hashFile(path.join(out,'scene_unknown_white.js'))},nativeAssets:assetRecords,navigation,parameterChecks,loadedActualImageScripts:sourceScripts,snapshots};
 fs.writeFileSync(reportPath,JSON.stringify(report,null,2));
 console.log(JSON.stringify({passed:true,verifiedEntries:expectCount,navigationCases:navigation.length,parameterGroups:parameterChecks.length,report:reportPath,actualCanvas:true,syntheticFixtures:false},null,2));
}
module.exports={npzIntegerArrays,verifyPanelSlot};
if(require.main===module)main().catch(error=>{console.error(error.stack);process.exitCode=1;});
