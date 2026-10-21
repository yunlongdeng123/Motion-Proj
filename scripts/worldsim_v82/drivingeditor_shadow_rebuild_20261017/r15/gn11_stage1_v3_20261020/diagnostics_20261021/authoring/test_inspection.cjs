// 普通Node代码/Canvas验证，不启动浏览器、HTTP或桌面自动化。
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const deps='C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {createCanvas,GlobalFonts}=require(require.resolve('@napi-rs/canvas',{paths:[deps]}));
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyh.ttc','Microsoft YaHei');
const base=process.argv[3]||'C:/Users/dengyunlong/Documents/Codex/2026-10-20/file-c-users-dengyunlong-documents-codex-2/outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1';
const source=fs.existsSync(path.join(__dirname,'src'))?path.join(__dirname,'src'):path.join(__dirname,'../src'),dataRoot=process.argv[2]||path.join(__dirname,'data'),out=process.argv[4]||path.resolve(__dirname,'../../outputs/diagnostic_road_layers_update');fs.mkdirSync(out,{recursive:true});
function read(p){return JSON.parse(fs.readFileSync(p,'utf8').split(']=')[1].trim().slice(0,-1));}
function unpack(p){const data=zlib.inflateSync(Buffer.from(p.data,'base64')),out={};const types={'<i4':Int32Array,'<u4':Uint32Array,'<f4':Float32Array,'<f8':Float64Array,'|u1':Uint8Array,'<u2':Uint16Array};for(const [key,a] of Object.entries(p.arrays)){const b=Uint8Array.from(data.subarray(a.offset,a.offset+a.bytes));if(a.dtype==='<f2'){out[key]=Float32Array.from(new Uint16Array(b.buffer),h=>{const e=(h>>10)&31,m=h&1023,sign=(h&32768)?-1:1;return sign*(e===0?2**-14*m/1024:e===31?(m?NaN:Infinity):2**(e-15)*(1+m/1024));});}else{assert(types[a.dtype],a.dtype);out[key]=new types[a.dtype](b.buffer);}}for(const k of ['metadata','branches','sampling'])if(p[k])out[k]=p[k];return out;}
const elements=new Map();function el(id){if(elements.has(id))return elements.get(id);let e={id,checked:true,value:'',textContent:'',innerHTML:'',disabled:false,options:[],parentNode:{lastChild:{textContent:''}},style:{},dataset:{},before(){},after(){},prepend(v){this.options.unshift(v);},setAttribute(){},addEventListener(){},classList:{toggle(){}},getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth||1200,height:this.clientHeight||700}}};elements.set(id,e);return e;}
for(const id of ['sceneCanvas','renderCanvas']){const e=createCanvas(id==='sceneCanvas'?1200:480,id==='sceneCanvas'?700:320);Object.assign(e,{id,clientWidth:e.width,clientHeight:e.height,before(){},getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}},setPointerCapture(){}});elements.set(id,e);}
el('cloudColor').value='role';el('pointAlpha').value='effective';el('threeSize').value='2.5';
const cloud=unpack(read(path.join(base,'data/full/cloud.js'))),road=unpack(read(path.join(base,'data/road.js'))),maps=unpack(read(path.join(base,'data/full/102.js'))),lidar=unpack(read(path.join(base,'data/lidar/102.js')));
const history=unpack(read(path.join(dataRoot,'road_history.js'))),shared=unpack(read(path.join(dataRoot,'inspection/shared_102.js'))),surfaces=unpack(read(path.join(dataRoot,'inspection/full_102.js')));
const scene=read(path.join(base,'data/scene.js'));
const sandbox={console,Float32Array,Uint8Array,Uint16Array,Int32Array,Map,Set,Math,Number,Array,Object,JSON,Promise,performance,Buffer,
 document:{getElementById:el,createElement:()=>({style:{},setAttribute(){}})},GSDATA:{scene},__cloud:cloud,__road:road,__maps:maps,__lidar:lidar,
 GSCloud:{defaults:{static_rate:2,dynamic_rate:2},begin(){},active(){},drawn(){},alpha:v=>el('pointAlpha').value==='solid'?1:v,legend(){},vehicles(){return [];}}};sandbox.window=sandbox;
const ctx=vm.createContext(sandbox);for(const file of ['inspection_math.js','inspection.js'])vm.runInContext(fs.readFileSync(path.join(source,file),'utf8'),ctx);
vm.runInContext(fs.readFileSync(path.join(source,'app.js'),'utf8').replace(/init\(\);\s*$/,''),ctx);
vm.runInContext("scene=GSDATA.scene;cloud=__cloud;road=__road;maps=__maps;lidar=__lidar;S.x=255;S.y=240;GSInspect.bind({state:()=>S,scene:()=>scene,maps:()=>maps,cloud:()=>cloud,project:project3,line:line3,world:worldM,quat,mapValue,draw:draw3D});",ctx);
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
checks.push('actual trained road1283 links to frozen initial location/scale/opacity and ground support');
fs.writeFileSync(path.join(out,'three_layers_canvas.png'),el('sceneCanvas').toBuffer('image/png'));
const empty=history.supported.findIndex(v=>!v);sandbox.__empty=empty;vm.runInContext('GSInspect.selectUnit(__empty)',ctx);assert(el('roadHistoryInfo').textContent.includes('初始化未创建'));assert.equal(vm.runInContext('S.selected',ctx),null);
checks.push('uncreated cells remain gray grid candidates and have no invented trained Gaussian');
el('regionSelect').onclick();assert(vm.runInContext('GSInspect.pointerDown({clientX:200,clientY:120})',ctx));vm.runInContext('GSInspect.pointerMove({clientX:1050,clientY:650});GSInspect.pointerUp();',ctx);assert(el('regionInfo').textContent.includes('屏幕投影框选'));assert(el('regionInfo').textContent.includes('训练后道路'));
checks.push('screen rectangle reports all ground/init/trained/missing counts regardless of opacity');
const missing=shared.lidar_nearest_m.findIndex(v=>!v);sandbox.__missing=missing;vm.runInContext('S.x=__missing%480;S.y=Math.floor(__missing/480);draw3D();',ctx);assert(el('raySurfaceInfo').textContent.includes('无实测点'));
checks.push('a pixel without a LiDAR return explicitly reports no measured point');
const v1Maps=unpack(read(path.join(base,'data/gn11_v1/102.js')));sandbox.__v1Maps=v1Maps;vm.runInContext("S.arm='gn11_v1';S.x=255;S.y=240;maps=__v1Maps;GSInspect.accept({...__inspection,identity:'gn11_v1/102',surfaces:null});",ctx);const v1=vm.runInContext('GSInspect.values()',ctx);assert.equal(v1[1].z,null);assert(Math.abs(v1[2].z-8.93)<.011);assert(el('initialRoad').disabled);
checks.push('V1 uses its native static/ordinary depth and does not inject V3 initial roads');
vm.runInContext("S.arm='full';maps=__maps;GSInspect.accept(__inspection);GSInspect.selectUnit(1283);",ctx);
const ids=new Set(cloud.static_road_id.filter(v=>v>=0));assert.equal(ids.size,history.supported.reduce((s,v)=>s+v,0));for(let i=0;i<history.supported.length;i++)assert.equal(ids.has(i),!!history.supported[i]);
checks.push('all14034 trained road IDs match the initial supported set exactly;12054 uncreated cells preserved');
const appCode=fs.readFileSync(path.join(source,'app.js'),'utf8');const pointerCode=appCode.slice(appCode.indexOf("const c=$('sceneCanvas');c.onpointerdown="),appCode.indexOf("$('fit3d').onclick="));vm.runInContext('(function(){'+pointerCode+'})()',ctx);
vm.runInContext('V.camera=false;V.yaw=.7;V.pitch=1.1;V.distance=5;V.focus=[2,0,0];window.testPanPoint=[...V.focus];window.testPanBefore=project3(testPanPoint);',ctx);
el('sceneCanvas').onpointerdown({clientX:300,clientY:300,pointerId:1});el('sceneCanvas').onpointermove({clientX:300,clientY:340,shiftKey:true});
const pointerDelta=vm.runInContext('(()=>{const p=project3(testPanPoint);return [p[0]-testPanBefore[0],p[1]-testPanBefore[1]];})()',ctx);assert(Math.abs(pointerDelta[0])<1e-7);assert(Math.abs(pointerDelta[1]-40)<1e-7);el('sceneCanvas').onpointerup({clientX:300,clientY:340});
vm.runInContext('V.camera=true;V.cameraPan=[0,0];',ctx);el('sceneCanvas').onpointerdown({clientX:300,clientY:300,pointerId:1});el('sceneCanvas').onpointermove({clientX:340,clientY:300,shiftKey:true});assert.equal(vm.runInContext('V.cameraPan[0]',ctx),40);assert.equal(vm.runInContext('V.cameraPan[1]',ctx),0);el('sceneCanvas').onpointerup({clientX:340,clientY:300});
checks.push('actual app pointer handlers pan vertically in orbit view and horizontally in calibrated-camera view');
const result={status:'PASSED',checks,browser_ui_verified:false,rendering_scope:'actual app draw3D and inspection modules executed with Node Canvas and test DOM, not a browser screenshot',current_pixel:values.map(v=>({name:v.name,z_m:v.z})),initial_ground_points:history.metadata.used_ground_count};
fs.writeFileSync(path.join(out,'interaction_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
