// Pure transport test with staggered decoders, not a browser/UI acceptance test.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(__dirname+'/sync_player_template.html','utf8');
const source=html.slice(html.indexOf('function pause(){'),html.indexOf('function tile('));
const playback=html.slice(html.indexOf('async function play(){'),html.indexOf("$('play').onclick="));
const elements=Object.fromEntries(['play','seek','frame','time','status','zoom','actor','speed'].map(k=>[k,{value:k==='actor'?'6':k==='speed'?'1':'',checked:false,textContent:'',classList:{add(){}}}]));
const paints=[];
class Decoder extends EventTarget{
  constructor(name,delay){super();this.name=name;this.delay=delay;this.readyState=4;this.duration=4.9;this._t=0;this.decoded=0;this.request=0;}
  get currentTime(){return this._t;}
  set currentTime(t){this._t=t;const r=++this.request;setTimeout(()=>{if(this.request!==r)return;this.decoded=Math.floor(t*10+1e-6);this.dispatchEvent(new Event('seeked'));},this.delay);}
  pause(){}
}
const entries=[2,9,23].map((delay,i)=>({video:new Decoder('decoder'+i,delay),wait:{style:{}},canvas:{width:960,height:640,getContext:()=>({fillRect(){},drawImage(v){paints.push({name:v.name,decoded:v.decoded});}})}}));
const context={console,Promise,performance,setTimeout,clearTimeout,entries,frame:113,playing:false,epoch:0,raf:0,busy:false,anchor:null,
  $:id=>elements[id],DATA:{rois:{'6':Array(49).fill([0,0,480,320])}},
  requestAnimationFrame:cb=>setTimeout(()=>cb(performance.now()),5),cancelAnimationFrame:clearTimeout};
vm.createContext(context);vm.runInContext(source+playback,context);
for(const e of entries)e.video.addEventListener('seeked',()=>context.draw(e));
(async()=>{
  await context.seek(113);
  assert.equal(paints.length,3,'No panel may paint before all decoders reach the target.');
  assert(paints.every(p=>p.decoded===24));
  paints.length=0;
  await Promise.all([context.seek(90),context.seek(120),context.seek(137)]);
  assert.equal(context.frame,137);assert(paints.every(p=>p.decoded===48),'Rapid dragging must only paint the latest complete frame.');
  paints.length=0;await context.seek(89);paints.length=0;
  await context.play();await new Promise(r=>setTimeout(r,260));context.pause();
  assert(context.frame>=90&&context.frame<=92);
  assert.equal(paints.length%3,0);
  for(let i=0;i<paints.length;i+=3)assert.equal(new Set(paints.slice(i,i+3).map(p=>p.decoded)).size,1,'Playback groups must be frame-identical.');
  await context.seek(121);assert(entries.every(e=>e.video.decoded===32));
  console.log(JSON.stringify({status:'PASS',tests:['all-panel decoding barrier','rapid-drag latest frame','single-clock playback','pause and exact frame seek'],scope:'simulated asynchronous video decoders; browser file:// not tested'}));
})().catch(e=>{console.error(e);process.exitCode=1;});
