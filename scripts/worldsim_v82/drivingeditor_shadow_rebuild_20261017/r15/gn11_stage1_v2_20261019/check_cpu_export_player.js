// Runs actual generated player handlers in a DOM/media fixture; no browser UI claim.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/<script type="application\/json" id="data">(.*?)<\/script>/s)[1]);
const code=html.match(/<script>\s*('use strict';.*?)<\/script>/s)[1];
const elements={},buttons=[],frames=new Map();let next=0;
function element(tag='DIV'){return {tagName:tag,children:[],dataset:{},attrs:{},value:'',textContent:'',readyState:4,currentTime:0,paused:true,seeking:false,listeners:{},
 append(...children){this.children.push(...children);},replaceChildren(...children){this.children=[...children];},setAttribute(k,v){this.attrs[k]=v;},
 addEventListener(k,fn,opts){(this.listeners[k]??=[]).push({fn,once:opts?.once});},removeEventListener(k,fn){this.listeners[k]=(this.listeners[k]||[]).filter(x=>x.fn!==fn);},
 dispatch(k,e={}){for(const x of [...this.listeners[k]||[]]){x.fn({target:this,...e});if(x.once)this.removeEventListener(k,x.fn);}},
 play(){this.paused=false;return this.rejectPlay?Promise.reject(new Error('fixture play error')):Promise.resolve();},pause(){this.paused=true;},load(){this.currentTime=0;}};}
for(const id of ['data','play','previous','next','speed','loop','seek','frameLabel','status','error','channels','anchors','modelGrid','comparison','downloads','downloadHead'])elements[id]=element();
elements.data.textContent=JSON.stringify(data);elements.speed.value='1';elements.loop.checked=true;
const document={getElementById:id=>elements[id],createElement:tag=>{const e=element(tag.toUpperCase());if(tag==='button')buttons.push(e);return e;},
 querySelectorAll:s=>s==='[data-channel]'?buttons.filter(x=>x.dataset.channel):[],addEventListener(){}};
const context={document,window:{},Math,Number,String,Error,Promise,setTimeout,clearTimeout,requestAnimationFrame:fn=>{const n=++next;frames.set(n,fn);return n;},cancelAnimationFrame:id=>frames.delete(id)};
vm.createContext(context);vm.runInContext(code,context);
const flush=async()=>{await Promise.resolve();await Promise.resolve();await Promise.resolve();};
const currentVideos=()=>elements.modelGrid.children.map(section=>section.children.find(x=>x.tagName==='VIDEO'));
(async()=>{
 await flush();const api=context.window.FINAL49_PLAYER;assert(!api.getState().loading);assert.equal(api.getState().frame,0);
 let seeks=0,videoSeeks=0;
 for(const choice of data.comparisons.map(c=>c[0])){
  elements.comparison.value=choice;
  for(const channel of Object.keys(data.channels)){
   await api.change(channel);await flush();const active=api.getState().active,vs=currentVideos();
   assert.equal(vs.length,choice==='all4'?4:choice==='all5'?5:2);
   active.forEach((name,i)=>assert.equal(vs[i].src,data.models[name].videos[channel].path));
   for(let frame=0;frame<49;frame++){api.setFrame(frame);assert.equal(api.getState().frame,frame);for(const v of vs)assert(Math.abs(v.currentTime-(frame/10+.0001))<1e-9);seeks++;videoSeeks+=vs.length;}
  }
 }
 elements.comparison.value=data.default_comparison;await api.change('static');elements.seek.value='40';elements.seek.dispatch('input');assert.equal(api.getState().frame,40);assert(elements.frameLabel.textContent.includes('raw129'));
 elements.next.dispatch('click');assert.equal(api.getState().frame,41);elements.previous.dispatch('click');assert.equal(api.getState().frame,40);
 elements.speed.value='.5';elements.speed.dispatch('change');for(const v of currentVideos())assert.equal(v.playbackRate,.5);
 api.setFrame(0);await api.play();assert(api.getState().playing);const vs=currentVideos();vs[0].currentTime=2.05;vs.slice(1).forEach(v=>v.currentTime=0);[...frames.values()].at(-1)();for(const v of vs)assert.equal(v.currentTime,2.05);assert.equal(api.getState().frame,20);api.stop();
 elements.loop.checked=false;vs[0].dispatch('ended');assert.equal(api.getState().frame,48);assert(!api.getState().playing);
 elements.loop.checked=true;vs[0].dispatch('ended');await flush();assert.equal(api.getState().frame,0);assert(api.getState().playing);api.stop();
 vs[0].rejectPlay=true;await api.play();await flush();assert(!api.getState().playing);assert(elements.error.textContent.includes('fixture play error'));
 assert.equal(seeks,data.comparisons.length*Object.keys(data.channels).length*49);
 console.log(JSON.stringify({status:'PASSED_ACTUAL_CPU_EXPORT_PLAYER_DOM_MEDIA_FIXTURE',comparison_modes:data.comparisons.length,channels:Object.keys(data.channels).length,global_frame_seeks:seeks,video_frame_seeks:videoSeeks,
  synchronized_models_max:Math.max(...data.comparisons.map(c=>c[0]==='all5'?5:c[0]==='all4'?4:2)),raw129_seek:true,step_buttons:true,speed_all:true,drift_correction:true,loop_and_end:true,error_visible:true,browser_UI_tested:false}));
})().catch(e=>{console.error(e);process.exitCode=1;});
