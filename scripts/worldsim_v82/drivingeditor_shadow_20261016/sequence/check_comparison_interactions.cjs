// Code-level DOM fixture only. Does not automate or bypass any browser policy.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {pathToFileURL,fileURLToPath}=require('node:url');
const page=path.resolve(process.argv[2]);const html=fs.readFileSync(page,'utf8');const dataText=html.match(/<script id="comparison-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
class Element{
 constructor(tag='DIV'){this.tagName=tag;this.children=[];this.dataset={};this.style={};this.attributes={};this.listeners={};this.textContent='';this.value='';this.currentTime=0;this.paused=true;this.src='';const s=new Set();this.classList={add:v=>s.add(v),toggle:(v,on)=>on?s.add(v):s.delete(v),contains:v=>s.has(v)};}
 append(...items){this.children.push(...items)} replaceChildren(...items){this.children=items} setAttribute(k,v){this.attributes[k]=v} addEventListener(k,v){(this.listeners[k]??=[]).push(v)} scrollIntoView(){} pause(){this.paused=true}play(){this.paused=false;return Promise.resolve()}load(){}
}
const ids=new Map();for(const id of [...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]))ids.set(id,new Element(id.startsWith('video-')? 'VIDEO':'DIV'));
ids.get('comparison-data').textContent=dataText;ids.get('video-mode').value='delete';ids.get('view').value='full';
const modes=['delete','observed'].map(x=>{const e=new Element('BUTTON');e.dataset.mode=x;return e});const jumps=[89,101,106,125,137].map(x=>{const e=new Element('BUTTON');e.dataset.raw=String(x);return e});
const doc={getElementById:id=>{assert(ids.has(id),'Unknown id '+id);return ids.get(id)},createElement:tag=>new Element(tag.toUpperCase()),querySelectorAll:query=>query==='[data-mode]'?modes:query==='[data-raw]'?jumps:[],activeElement:new Element('BODY'),addEventListener(){},hidden:false};
vm.runInNewContext(script,{document:doc,Promise,console,setInterval:()=>1,clearInterval(){}});
function images(el){return (el.tagName==='IMG'?[el]:[]).concat(el.children.flatMap(images))}
let states=0;const files=new Set();
for(const mode of modes){mode.onclick();for(let i=0;i<49;i++){
 ids.get('timeline').oninput({target:{value:String(i)}});assert.equal(ids.get('gallery').children.length,4);assert(ids.get('frame-label').textContent.includes('raw'+(89+i)));
 for(const im of images(ids.get('gallery'))){const url=new URL(im.src,pathToFileURL(page));assert.equal(url.protocol,'file:');const filename=fileURLToPath(url);assert(fs.existsSync(filename),filename);files.add(filename)}states++;
}}
ids.get('view').onchange({target:{value:'road'}});assert(ids.get('gallery').classList.contains('road-crop'));ids.get('view').onchange({target:{value:'full'}});assert(!ids.get('gallery').classList.contains('road-crop'));
for(const b of jumps){b.onclick();assert(ids.get('frame-label').textContent.includes('raw'+b.dataset.raw))}
const videos=new Set();for(const key of ['delete','shadow','fixed','translate','yaw15','yaw30','anchor']){ids.get('video-mode').value=key;ids.get('video-mode').onchange();for(const id of ['video-left','video-right']){const p=fileURLToPath(new URL(ids.get(id).src,pathToFileURL(page)));assert(fs.existsSync(p),p);videos.add(p)}ids.get('video-reset').onclick();assert.equal(ids.get('video-left').currentTime,0)}
ids.get('play').onclick();assert.equal(ids.get('play').textContent,'暂停');ids.get('play').onclick();assert.equal(ids.get('play').textContent,'播放');
console.log(JSON.stringify({interaction_fixture:'PASS',browser_rendering_verified:false,frame_mode_states_checked:states,image_files_referenced:files.size,video_groups_checked:7,video_files:[...videos],crop_and_jump_controls_checked:true}));
