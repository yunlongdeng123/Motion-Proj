// Actual generated handler code with a DOM/media fixture; this is not browser UI testing.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(page.match(/<script type="application\/json" id="data">(.*?)<\/script>/s)[1]);
const code=page.match(/<script>\s*('use strict';.*?)<\/script>/s)[1];
const elements={},buttons=[],frames=new Map();let next=0;
function element(tag='DIV'){
 return {tagName:tag,children:[],attrs:{},dataset:{},value:'',textContent:'',disabled:false,readyState:4,currentTime:0,
  paused:true,seeking:false,listeners:{},append(x){this.children.push(x);},setAttribute(k,v){this.attrs[k]=v;},
  addEventListener(k,fn,opts){(this.listeners[k]??=[]).push({fn,once:opts?.once});},
  removeEventListener(k,fn){this.listeners[k]=(this.listeners[k]||[]).filter(x=>x.fn!==fn);},
  dispatch(k,e={}){for(const x of [...this.listeners[k]||[]]){x.fn({target:this,...e});if(x.once)this.removeEventListener(k,x.fn);}},
  play(){this.paused=false;return this.rejectPlay?Promise.reject(new Error('fixture play failure')):Promise.resolve();},
  pause(){this.paused=true;},load(){this.currentTime=0;}};
}
for(const id of ['data','left','right','play','previous','next','speed','loop','seek','frameLabel','status','error','channels','anchors','leftCaption','rightCaption','downloads'])elements[id]=element();
elements.data.textContent=JSON.stringify(data);elements.speed.value='1';elements.loop.checked=true;
const document={getElementById:id=>elements[id],createElement:tag=>{const e=element(tag.toUpperCase());if(tag==='button')buttons.push(e);return e;},
 querySelectorAll:selector=>selector==='[data-channel]'?buttons.filter(x=>x.dataset.channel):[],addEventListener(){}};
const context={document,window:{},Promise,Math,Number,String,Error,setTimeout,clearTimeout,
 requestAnimationFrame:fn=>{const id=++next;frames.set(id,fn);return id;},cancelAnimationFrame:id=>frames.delete(id)};
vm.createContext(context);vm.runInContext(code,context);
const flush=async()=>{await Promise.resolve();await Promise.resolve();await Promise.resolve();};
(async()=>{
 await flush();const api=context.window.NATIVE30K_PLAYER;assert(!api.getState().loading);
 assert.equal(api.getState().channel,'static');assert.equal(api.getState().frame,0);
 let seeks=0;
 for(const channel of Object.keys(data.channels)){
  await api.changeChannel(channel);await flush();assert.equal(api.getState().channel,channel);
  assert.equal(elements.left.src,data.videos['full_'+channel].path);assert.equal(elements.right.src,data.videos['real_only_'+channel].path);
  for(let f=0;f<49;f++){api.setFrame(f);assert.equal(api.getState().frame,f);assert.equal(elements.left.currentTime,elements.right.currentTime);assert(Math.abs(elements.left.currentTime-(f/10+.0001))<1e-9);seeks++;}
 }
 api.setFrame(0);elements.seek.value='40';elements.seek.dispatch('input');assert.equal(api.getState().frame,40);assert(elements.frameLabel.textContent.includes('raw129'));
 elements.next.dispatch('click');assert.equal(api.getState().frame,41);elements.previous.dispatch('click');assert.equal(api.getState().frame,40);
 elements.speed.value='.25';elements.speed.dispatch('change');assert.equal(elements.left.playbackRate,.25);assert.equal(elements.right.playbackRate,.25);
 await api.play();assert(api.getState().playing);elements.left.currentTime=1.205;elements.right.currentTime=.5;
 const fn=[...frames.values()].at(-1);fn();assert.equal(elements.right.currentTime,1.205);assert.equal(api.getState().frame,12);
 api.stop();assert(!api.getState().playing);assert(elements.left.paused&&elements.right.paused);
 elements.loop.checked=false;elements.left.dispatch('ended');assert.equal(api.getState().frame,48);assert.equal(elements.right.currentTime,4.8001);
 elements.loop.checked=true;elements.left.dispatch('ended');await flush();assert(api.getState().playing);assert.equal(api.getState().frame,0);api.stop();
 elements.left.rejectPlay=true;await api.play();await flush();assert(!api.getState().playing);assert(elements.error.textContent.includes('fixture play failure'));
 console.log(JSON.stringify({status:'PASSED_GENERATED_PLAYER_DOM_MEDIA_FIXTURE',channels:4,matched_frame_seeks:seeks,
  raw129_seek:true,step_buttons:true,speed_both:true,drift_correction:true,loop_and_end:true,play_failure_visible:true,browser_UI_tested:false}));
})().catch(e=>{console.error(e);process.exitCode=1;});
