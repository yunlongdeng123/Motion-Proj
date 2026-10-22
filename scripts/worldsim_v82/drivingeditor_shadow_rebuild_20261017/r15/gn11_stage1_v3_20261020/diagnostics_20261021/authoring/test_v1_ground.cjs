// 普通Node代码/Canvas验证，不启动浏览器、HTTP或桌面自动化。
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const deps='C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {createCanvas,GlobalFonts}=require(require.resolve('@napi-rs/canvas',{paths:[deps]}));
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyh.ttc','Microsoft YaHei');
const base=process.argv[3]||'C:/Users/dengyunlong/Documents/Codex/2026-10-20/file-c-users-dengyunlong-documents-codex-2/outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1';
const source=fs.existsSync(path.join(__dirname,'src'))?path.join(__dirname,'src'):path.join(__dirname,'../src'),dataRoot=process.argv[2]||path.join(__dirname,'../road_layers/data'),out=process.argv[4]||path.resolve(__dirname,'../../outputs/v1_ground_diagnostics_update');fs.mkdirSync(out,{recursive:true});
function read(p){return JSON.parse(fs.readFileSync(p,'utf8').split(']=')[1].trim().slice(0,-1));}
function unpack(p){const data=zlib.inflateSync(Buffer.from(p.data,'base64')),out={};const types={'<i4':Int32Array,'<u4':Uint32Array,'<f4':Float32Array,'<f8':Float64Array,'|u1':Uint8Array,'<u2':Uint16Array};for(const [key,a] of Object.entries(p.arrays)){const b=Uint8Array.from(data.subarray(a.offset,a.offset+a.bytes));if(a.dtype==='<f2'){out[key]=Float32Array.from(new Uint16Array(b.buffer),h=>{const e=(h>>10)&31,m=h&1023,sign=(h&32768)?-1:1;return sign*(e===0?2**-14*m/1024:e===31?(m?NaN:Infinity):2**(e-15)*(1+m/1024));});}else{assert(types[a.dtype],a.dtype);out[key]=new types[a.dtype](b.buffer);}}for(const k of ['metadata','branches','sampling'])if(p[k])out[k]=p[k];return out;}
const v1Root=process.argv[5]||path.join(__dirname,'data');
const elements=new Map();function el(id){if(elements.has(id))return elements.get(id);let e={id,checked:true,value:'',textContent:'',innerHTML:'',disabled:false,options:[],parentNode:{lastChild:{textContent:''}},style:{},dataset:{},before(){},after(){},prepend(v){this.options.unshift(v);},setAttribute(){},addEventListener(){},classList:{toggle(){}},getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth||1200,height:this.clientHeight||700}}};elements.set(id,e);return e;}
for(const id of ['sceneCanvas','renderCanvas']){const e=createCanvas(id==='sceneCanvas'?1200:480,id==='sceneCanvas'?700:320);Object.assign(e,{id,clientWidth:e.width,clientHeight:e.height,before(){},getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}},setPointerCapture(){}});elements.set(id,e);}
el('cloudColor').value='role';el('pointAlpha').value='effective';el('threeSize').value='2.5';el('v1GroundRate').value='10';
const cloud=unpack(read(path.join(base,'data/full/cloud.js'))),road=unpack(read(path.join(base,'data/road.js'))),maps=unpack(read(path.join(base,'data/full/102.js'))),lidar=unpack(read(path.join(base,'data/lidar/102.js')));
const history=unpack(read(path.join(dataRoot,'road_history.js'))),shared=unpack(read(path.join(dataRoot,'inspection/shared_102.js'))),surfaces=unpack(read(path.join(dataRoot,'inspection/full_102.js')));
const scene=read(path.join(base,'data/scene.js'));
const sandbox={console,Float32Array,Uint8Array,Uint16Array,Int32Array,Map,Set,Math,Number,Array,Object,JSON,Promise,performance,Buffer,atob,Blob,Response,DecompressionStream,
 document:{getElementById:el,createElement:()=>({style:{},setAttribute(){},remove(){}}),head:{append(script){try{const relative=script.src.replace(/^data\//,'');let file=path.join(dataRoot,relative);if(!fs.existsSync(file))file=path.join(base,script.src);const text=fs.readFileSync(file,'utf8'),key=JSON.parse(text.slice(text.indexOf('window.GSDATA[')+14,text.indexOf(']=')));sandbox.GSDATA[key]=read(file);queueMicrotask(()=>script.onload());}catch(e){queueMicrotask(()=>script.onerror(e));}}}},GSDATA:{scene},__cloud:cloud,__road:road,__maps:maps,__lidar:lidar,
 GSCloud:{defaults:{static_rate:2,dynamic_rate:2},begin(){},active(){},drawn(){},alpha:v=>el('pointAlpha').value==='solid'?1:v,legend(){},vehicles(){return [];}}};sandbox.window=sandbox;
const ctx=vm.createContext(sandbox);for(const file of ['inspection_math.js','v1_ground.js','inspection.js'])vm.runInContext(fs.readFileSync(path.join(source,file),'utf8'),ctx);
vm.runInContext(fs.readFileSync(path.join(source,'app.js'),'utf8').replace(/init\(\);\s*$/,''),ctx);
vm.runInContext("scene=GSDATA.scene;cloud=__cloud;road=__road;maps=__maps;lidar=__lidar;S.x=255;S.y=240;GSInspect.bind({state:()=>S,scene:()=>scene,maps:()=>maps,cloud:()=>cloud,project:project3,line:line3,world:worldM,quat,mapValue,heat:rgbHeat,draw:draw3D,selectionChanged:()=>{window.selectionUpdates=(window.selectionUpdates||0)+1;}});",ctx);
sandbox.__inspection={history,shared,surfaces,identity:'full/102'};vm.runInContext('GSInspect.accept(__inspection);',ctx);
const checks=[];
// 在旋转后的多个视角，投影焦平面的点必须严格按手势移动，不交换横纵坐标。
for(const yaw of [-1.2,0,.7,2])for(const pitch of [-.8,0,1.1])for(const [dx,dy] of [[30,0],[0,30],[0,-30],[-30,0]]){
 sandbox.__view={yaw,pitch,distance:12,focus:[2,0,0]};sandbox.__dx=dx;sandbox.__dy=dy;
 const result=vm.runInContext('(()=>{Object.assign(V,__view,{camera:false});let p=[...V.focus],before=project3(p);V.focus=GSInspectionMath.panFocus(V,__dx,__dy,1200);let after=project3(p);return [after[0]-before[0],after[1]-before[1]];})()',ctx);
 assert(Math.abs(result[0]-dx)<1e-8);assert(Math.abs(result[1]-dy)<1e-8);
}
checks.push('48 screen-pan directions across yaw/pitch preserve exact horizontal/vertical gesture');
const values=vm.runInContext('GSInspect.values()',ctx);assert.equal(values.length,3);assert(Math.abs(values[0].z-6.37956095)<.002);assert(Math.abs(values[1].z-12.481876)<.002);assert(Math.abs(values[2].z-12.398)<.01);
for(const r of values){sandbox.__p=r.p;const projection=vm.runInContext('(()=>{V.camera=true;V.cameraPan=[0,0];return project3(__p);})()',ctx);const scale=Math.min(1200/480,700/320);assert(Math.abs(projection[0]-((255+.5)*scale+(1200-480*scale)/2))<.002);assert(Math.abs(projection[1]-(240+.5)*scale)<.002);}
checks.push('LiDAR/road/ordinary 3D markers project to one selected pixel center, within0.002px');
vm.runInContext('Object.assign(V,{yaw:.5,pitch:1.05,distance:5,focus:[1,0,0],camera:false});',ctx);
el('compareRoadLayers').onclick();assert(el('ground49').checked&&el('initialRoad').checked&&el('roads3d').checked&&el('threeSolid').checked);assert(!el('ordinary').checked&&!el('dynamic3d').checked);
assert.equal(vm.runInContext('GSInspect.alpha(.00001)',ctx),1);el('threeSolid').checked=false;assert.equal(vm.runInContext('GSInspect.alpha(.00001)',ctx),.00001);el('threeSolid').checked=true;
checks.push('three layers share point size and solid alpha independently of trained effective opacity');
vm.runInContext('V.focus=[__inspection.history.init_xyz[1283*3],__inspection.history.init_xyz[1283*3+1],__inspection.history.init_xyz[1283*3+2]];V.distance=1.2;draw3D();GSInspect.selectUnit(1283);',ctx);assert(el('roadHistoryInfo').textContent.includes('位移'));assert(el('roadHistoryInfo').textContent.includes('opacity 0.5000'));assert(vm.runInContext('S.selected.id',ctx).startsWith('full:'));assert(vm.runInContext('GSInspect.debug().neighbors.length',ctx)>0);
assert(vm.runInContext('selectionUpdates',ctx)>0);checks.push('actual trained road1283 links to frozen initial location/scale/opacity and ground support, and notifies BEV/source linkage');
fs.writeFileSync(path.join(out,'three_layers_canvas.png'),el('sceneCanvas').toBuffer('image/png'));
const empty=history.supported.findIndex(v=>!v);sandbox.__empty=empty;vm.runInContext('GSInspect.selectUnit(__empty)',ctx);assert(el('roadHistoryInfo').textContent.includes('初始化未创建'));assert.equal(vm.runInContext('S.selected',ctx),null);
checks.push('uncreated cells remain gray grid candidates and have no invented trained Gaussian');
el('regionSelect').onclick();assert(vm.runInContext('GSInspect.pointerDown({clientX:200,clientY:120})',ctx));vm.runInContext('GSInspect.pointerMove({clientX:1050,clientY:650});GSInspect.pointerUp();',ctx);assert(el('regionInfo').textContent.includes('屏幕投影框选'));assert(el('regionInfo').textContent.includes('训练后道路'));
checks.push('screen rectangle reports all ground/init/trained/missing counts regardless of opacity');
const missing=shared.lidar_nearest_m.findIndex(v=>!v);sandbox.__missing=missing;vm.runInContext('S.x=__missing%480;S.y=Math.floor(__missing/480);draw3D();',ctx);assert(el('raySurfaceInfo').textContent.includes('无实测点'));
checks.push('a pixel without a LiDAR return explicitly reports no measured point');
const v1Maps=unpack(read(path.join(base,'data/gn11_v1/102.js'))),v1History=unpack(read(path.join(v1Root,'v1_ground_history.js'))),v1Surfaces=unpack(read(path.join(v1Root,'inspection/gn11_v1_102.js'))),v1Cloud=unpack(read(path.join(base,'data/gn11_v1/cloud.js')));
sandbox.__v1Maps=v1Maps;sandbox.__v1History=v1History;sandbox.__v1Surfaces=v1Surfaces;sandbox.__v1Cloud=v1Cloud;
vm.runInContext("S.arm='gn11_v1';S.x=255;S.y=240;maps=__v1Maps;cloud=__v1Cloud;GSInspect.accept({...__inspection,v1:__v1History,identity:'gn11_v1/102',surfaces:__v1Surfaces});",ctx);
assert(!el('initialRoad').disabled);assert(el('emptyRoad').disabled);assert(el('roads3d').parentNode.lastChild.textContent.includes('V1'));
el('compareRoadLayers').onclick();assert(!el('emptyRoad').checked);assert(!el('ordinary').checked);assert(el('initialRoad').checked);
checks.push('V1 enabled native initial ground and trained candidates, with inapplicable grid disabled');
assert.equal(v1History.metadata.initial_static_count,585943);assert.equal(v1History.metadata.initial_ground_count,71120);assert.equal(v1History.row.length,1278443);
const v1=vm.runInContext('GSInspect.values()',ctx);assert(v1[1].z>0&&v1[2].z>0);assert(v1[1].name.includes('V1'));assert(v1[2].name.includes('其余静态'));assert(Math.abs(v1[0].z-6.37956095)<.002);
for(const r of v1){sandbox.__p=r.p;const q=vm.runInContext('(()=>{V.camera=true;V.cameraPan=[0,0];return project3(__p);})()',ctx),scale=Math.min(1200/480,700/320);assert(Math.abs(q[0]-((255+.5)*scale+(1200-480*scale)/2))<.002);assert(Math.abs(q[1]-(240+.5)*scale)<.002);}
checks.push('V1 full-subset and remaining-static mean depth markers share actual selected pixel center with nearest measured LiDAR');
vm.runInContext("Object.assign(V,{yaw:.5,pitch:1.05,distance:5,focus:[1,0,0],camera:false});draw3D();",ctx);
assert(el('inspectionLegend').textContent.includes('1,278,443'));assert(el('inspectionLegend').textContent.includes('10%'));
const counts=vm.runInContext('GSV1Ground.debug()',ctx);assert(counts.active.length>0);assert(counts.screen>0);assert(counts.screen<counts.active.length);
const first=counts.active.find(i=>v1History.root_ground[i]);sandbox.__first=first;
vm.runInContext("GSInspect.selectedRoad({kind:'v1trained',index:__first})",ctx);assert(el('roadHistoryInfo').textContent.includes('检查点保存谱系'));assert(el('roadHistoryInfo').textContent.includes('0.0100'));assert.equal(vm.runInContext('S.selected.road_id',ctx),-1);assert(vm.runInContext('S.selected.id',ctx).startsWith('gn11_v1:9aafb373d1e2:static:'));
checks.push('V1 click preserves native checkpoint static-row ID and links true initial lineage, support, scale and effective opacity');
fs.writeFileSync(path.join(out,'v1_ground_canvas.png'),el('sceneCanvas').toBuffer('image/png'));
const dead=v1History.init_ground.findIndex((v,i)=>v&&v1History.descendants[i]===0);assert(dead>=0);sandbox.__dead=dead;vm.runInContext('GSInspect.selectedRoad({kind:"v1initial",root:__dead})',ctx);assert.equal(vm.runInContext('S.selected',ctx),null);assert(el('roadHistoryInfo').textContent.includes('无存活后代'));
checks.push('a pruned original V1 ground source explicitly has zero survivors and no invented current point');
sandbox.__rect=[200,120,1050,650];const regionA=vm.runInContext('GSV1Ground.region(__rect)',ctx);el('v1GroundRate').value='1';vm.runInContext('draw3D()',ctx);const regionB=vm.runInContext('GSV1Ground.region(__rect)',ctx);assert.equal(regionA,regionB);el('v1GroundRate').value='10';
checks.push('V1 full-area counts, mean effective opacity/scale/source offset unchanged when display sampling is lowered');
assert(vm.runInContext('GSV1Ground.heat(document.getElementById("renderCanvas"),"v1_ground_alpha",cross)',ctx));assert(vm.runInContext('GSV1Ground.pixelInfo()',ctx).includes('不能直接相加'));
el('ordinary').checked=true;const ordinary=vm.runInContext('cloudPositions().filter(r=>r.role===0).map(r=>cloud[r.prefix+"row"][r.i])',ctx);const candidateSet=new Set(v1History.row);for(const row of ordinary)assert(!candidateSet.has(row));
checks.push('V1 independent heatmap reads full candidates; ordinary sampled display excludes duplicate candidate native rows');
vm.runInContext("S.arm='full';maps=__maps;cloud=__cloud;GSInspect.accept(__inspection);GSInspect.selectUnit(1283);",ctx);
assert(!vm.runInContext('GSV1Ground.enabled()',ctx));assert(!el('emptyRoad').disabled);assert(!el('roads3d').parentNode.lastChild.textContent.includes('V1'));
checks.push('switching back restores V3 fixed-road controls and does not carry V1 geometry across arms');
vm.runInContext("S.arm='full';maps=__maps;GSInspect.accept(__inspection);GSInspect.selectUnit(1283);",ctx);
const ids=new Set(cloud.static_road_id.filter(v=>v>=0));assert.equal(ids.size,history.supported.reduce((s,v)=>s+v,0));for(let i=0;i<history.supported.length;i++)assert.equal(ids.has(i),!!history.supported[i]);
checks.push('all14034 trained road IDs match the initial supported set exactly;12054 uncreated cells preserved');
const appCode=fs.readFileSync(path.join(source,'app.js'),'utf8');const pointerCode=appCode.slice(appCode.indexOf("const c=$('sceneCanvas');c.onpointerdown="),appCode.indexOf("$('fit3d').onclick="));vm.runInContext('(function(){'+pointerCode+'})()',ctx);
vm.runInContext('V.camera=false;V.yaw=.7;V.pitch=1.1;V.distance=5;V.focus=[2,0,0];window.testPanPoint=[...V.focus];window.testPanBefore=project3(testPanPoint);',ctx);
el('sceneCanvas').onpointerdown({clientX:300,clientY:300,pointerId:1});el('sceneCanvas').onpointermove({clientX:300,clientY:340,shiftKey:true});
const pointerDelta=vm.runInContext('(()=>{const p=project3(testPanPoint);return [p[0]-testPanBefore[0],p[1]-testPanBefore[1]];})()',ctx);assert(Math.abs(pointerDelta[0])<1e-7);assert(Math.abs(pointerDelta[1]-40)<1e-7);el('sceneCanvas').onpointerup({clientX:300,clientY:340});
vm.runInContext('V.camera=true;V.cameraPan=[0,0];',ctx);el('sceneCanvas').onpointerdown({clientX:300,clientY:300,pointerId:1});el('sceneCanvas').onpointermove({clientX:340,clientY:300,shiftKey:true});assert.equal(vm.runInContext('V.cameraPan[0]',ctx),40);assert.equal(vm.runInContext('V.cameraPan[1]',ctx),0);el('sceneCanvas').onpointerup({clientX:340,clientY:300});
checks.push('actual app pointer handlers pan vertically in orbit view and horizontally in calibrated-camera view');
(async()=>{
 sandbox.GSDATA.v1GroundHistory=read(path.join(v1Root,'v1_ground_history.js'));
 sandbox.GSDATA['inspection/gn11_v1/102']=read(path.join(v1Root,'inspection/gn11_v1_102.js'));
 vm.runInContext("S.arm='gn11_v1';S.raw=102;cloud=__v1Cloud;",ctx);
 const v1Loaded=await vm.runInContext('GSInspect.load(S,load)',ctx);
 assert.equal(v1Loaded.v1.row.length,1278443);assert.equal(v1Loaded.v1.init_ground.reduce((a,b)=>a+b,0),71120);assert.equal(v1Loaded.surfaces.ground_alpha.length,480*320);
 checks.push('actual large V1 offline loader and DecompressionStream decode full geometry and surfaces without HTTP');
 vm.runInContext("S.arm='full';maps=__maps;cloud=__cloud;GSInspect.accept(__inspection);",ctx);
 const loaded=await vm.runInContext('GSInspect.load(S,load)',ctx);assert.equal(loaded.history.metadata.used_ground_count,49182);assert.equal(loaded.history.supported.length,26088);assert.equal(loaded.shared.lidar_nearest_m.length,480*320);assert.equal(loaded.surfaces.ordinary_depth_m.length,480*320);
 checks.push('actual script-file loader and DecompressionStream decode all new local evidence with metadata, without HTTP');
 const result={status:'PASSED',checks,v1_values:v1.map(v=>({name:v.name,z_m:v.z})),v1_region:regionA,browser_ui_verified:false,rendering_scope:'actual app draw3D and inspection modules executed with Node Canvas and test DOM, not a browser screenshot',current_pixel:values.map(v=>({name:v.name,z_m:v.z})),initial_ground_points:history.metadata.used_ground_count};
 fs.writeFileSync(path.join(out,'interaction_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
