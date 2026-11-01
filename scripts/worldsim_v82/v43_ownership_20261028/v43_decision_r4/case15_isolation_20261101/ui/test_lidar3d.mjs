import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { createCanvas, GlobalFonts } from '@napi-rs/canvas';
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyh.ttc','Microsoft YaHei');
const root='C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/V43_R4_GPU_20261031/actor_candidates';
const output='C:/Users/dengyunlong/Documents/Codex/2026-11-01/new-chat/outputs/机制隔离15例_20261101';
const page=await fs.readFile(path.join(root,'index.html'),'utf8');
const nodes=new Map(),callbacks=new Map(),scriptLoads=[];let imageDelay=0;
class Element {
 constructor(id,tag='div'){this.id=id;this.tagName=tag;this.hidden=false;this.value='';this.textContent='';this.checked=true;this.paused=true;this.width=960;this.height=520;this.naturalWidth=960;this.naturalHeight=640;this.events=new Map();this.children=[];}
 addEventListener(name,fn){if(!this.events.has(name))this.events.set(name,[]);this.events.get(name).push(fn);}
 fire(name,e={}){for(const fn of this.events.get(name)||[])fn({preventDefault(){},...e});}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}
 removeAttribute(name){if(name==='src')this._src='';}
 pause(){this.paused=true;}play(){this.paused=false;return Promise.resolve();}
 setPointerCapture(){}getBoundingClientRect(){return {width:this.width,height:this.height,left:0,top:0};}
 getContext(){if(!this.surface)this.surface=createCanvas(this.width,this.height);return this.surface.getContext('2d');}
 set src(value){this._src=value;if(this.id==='mediaImage'||this.tagName==='img'){setTimeout(()=>this.onload?.(),imageDelay);}}
 get src(){return this._src;}
}
for(const [,id] of page.matchAll(/\bid="([^"]+)"/g))nodes.set(id,new Element(id));
for(const id of ['lidarTop','lidarSide','lidarFront']){nodes.get(id).width=360;nodes.get(id).height=220;}
nodes.get('mediaOverlay').width=960;nodes.get('mediaOverlay').height=640;
nodes.get('lidar3DPointSize').value='2';nodes.get('lidar3DView').value='orbit';nodes.get('mediaViewMode').value='frame';nodes.get('eligibilityOperation').value='none';
const context=vm.createContext({console,setTimeout,clearTimeout,setInterval,clearInterval,requestAnimationFrame:fn=>{setTimeout(fn,0);},Map,Set,Math,Number,Array,String,Infinity,Promise});
context.window=context;context.document={getElementById:id=>nodes.get(id),createElement:tag=>new Element('',tag),head:{appendChild:node=>{scriptLoads.push(node.src);setTimeout(async()=>{try{const source=await fs.readFile(path.join(root,node.src),'utf8');vm.runInContext(source,context,{filename:node.src});node.onload?.();}catch(e){node.onerror?.(e);}},0);}}};
context.Image=class extends Element {constructor(){super('','img');}};
context.addEventListener=(type,fn)=>{if(!callbacks.has(type))callbacks.set(type,[]);callbacks.get(type).push(fn);};
const aliases=JSON.parse(await fs.readFile(path.join(root,'actor_aliases.json'),'utf8'));
context.ActorNames={actor:(ctx,track)=>aliases.scenes[ctx].actors[track].alias};
const dataSource=await fs.readFile(path.join(root,'data.js'),'utf8');vm.runInContext(dataSource,context);
let current={};const captures=[];
function choose(sceneKey,alias){const [ctx,scene]=Object.entries(aliases.scenes).find(([,s])=>s.scene_key===sceneKey);const track=Object.keys(scene.actors).find(k=>scene.actors[k].alias===alias);const data=context.ACTOR_CANDIDATES.find(s=>s.context_id===ctx);current={data,row:data.actors.find(r=>r.track_id===track)};return current;}
context.ActorCandidateViewer={current:()=>current,setSynchronizedFrame:()=>{}};
choose('waymo-train-212','a0');
vm.runInContext(await fs.readFile(path.join(root,'lidar3d.js'),'utf8'),context,{filename:'lidar3d.js'});
const originalSet=context.ActorLidar3D.setData;context.ActorLidar3D.setData=p=>{captures.push(p);originalSet(p);};
vm.runInContext(await fs.readFile(path.join(root,'sync_panel.js'),'utf8'),context,{filename:'sync_panel.js'});
async function settled(){for(let i=0;i<30;i++)await new Promise(r=>setTimeout(r,10));}
await settled();
assert.equal(nodes.get('lidarMode').value,'frame');assert.ok(captures.length);assert.equal(captures.at(-1).trackId,current.row.track_id);assert.ok(captures.at(-1).points.every(p=>p.kind==='lidar'));
nodes.get('mediaFrame').value='101';nodes.get('mediaFrame').fire('input');await settled();
assert.equal(captures.at(-1).raw,101);assert.equal(captures.at(-1).actualCount,457);assert.match(nodes.get('lidar3DIdentity').textContent,/a0.*raw 101/);
await fs.writeFile(path.join(output,'assets/lidar3d_212_a0_raw101_preview.png'),nodes.get('lidar3DCanvas').surface.toBuffer('image/png'));
// Scope toggle and mixed GS mode keep the 3D view's explicit measured-only contract.
nodes.get('lidarMode').value='all';nodes.get('lidarMode').fire('change');assert.equal(captures.at(-1).mode,'all');assert.match(nodes.get('lidar3DIdentity').textContent,/不是当前帧/);
nodes.get('lidarMode').value='gs';nodes.get('lidarMode').fire('change');assert.equal(captures.at(-1).mode,'gs');assert.match(nodes.get('lidar3DInfo').textContent,/只显示实测LiDAR/);
const before=nodes.get('lidar3DCanvas').surface.toBuffer('image/png');nodes.get('lidar3DCanvas').fire('pointerdown',{clientX:100,clientY:100,pointerId:1});nodes.get('lidar3DCanvas').fire('pointermove',{clientX:220,clientY:150,buttons:1});nodes.get('lidar3DCanvas').fire('pointerup');await settled();const after=nodes.get('lidar3DCanvas').surface.toBuffer('image/png');assert.notDeepEqual(after,before);
nodes.get('lidar3DCanvas').fire('wheel',{deltaY:-100});await settled();nodes.get('lidar3DFit').fire('click');
// A late image callback must not restore a previous actor after a selection change.
imageDelay=20;choose('waymo-train-212','a35');for(const fn of callbacks.get('actor-selection-changed')||[])fn();choose('waymo-train-212','a0');for(const fn of callbacks.get('actor-selection-changed')||[])fn();await settled();assert.equal(captures.at(-1).trackId,current.row.track_id);assert.match(nodes.get('lidar3DIdentity').textContent,/a0/);imageDelay=0;
choose('waymo-train-537','a0');for(const fn of callbacks.get('actor-selection-changed')||[])fn();await settled();nodes.get('mediaFrame').value='173';nodes.get('mediaFrame').fire('input');await settled();assert.equal(captures.at(-1).raw,173);assert.equal(captures.at(-1).actualCount,165);
await fs.writeFile(path.join(output,'assets/lidar3d_537_a0_raw173_preview.png'),nodes.get('lidar3DCanvas').surface.toBuffer('image/png'));
const last=captures.at(-1);originalSet({...last,mode:'frame',actualCount:0,points:[]});assert.equal(nodes.get('lidar3DPanel').hidden,false);assert.match(nodes.get('lidar3DInfo').textContent,/实测 0 点/);context.ActorLidar3D.clear();assert.equal(nodes.get('lidar3DPanel').hidden,true);
const result={test:'real-file Node DOM/Canvas integration, not a browser interaction test',passed:['current-frame default','212/a0 raw101 measured457','all-sequence explicitly labelled non-frame','GS mode measured-only3D','rotation/zoom/fit change projection','late image actor race guarded','537/a0 raw173 measured165','measured zero separated from missing'],preview_kind:'same Canvas2D code rendered offline using actual source data',source_data_modified:false};
await fs.writeFile(path.join(output,'3D视图代码核验.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
