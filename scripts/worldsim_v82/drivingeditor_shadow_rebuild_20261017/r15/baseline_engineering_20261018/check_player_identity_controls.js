// Exercise the generated page's real handlers; no browser or media decoder is claimed.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../../outputs/baseline_engineering_20261018');
const html=fs.readFileSync(path.join(root,'results.html'),'utf8');
const data=JSON.parse(html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1]);
const js=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{
 constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.style={};this.listeners={};this.value='';this.checked=false;this.disabled=false;this.hidden=false;this.textContent='';this.classList={add(){}};this.width=960;this.height=640;}
 append(e){this.children.push(e);if(this.tagName==='SELECT'&&this.children.length===1)this.value=e.value;}
 replaceChildren(){this.children=[];this.value='';}
 get options(){return this.children;}
 set innerHTML(s){this.markup=s;this.children=[];this.parts={};}
 querySelector(selector){return this.parts[selector]??=new Element(selector==='canvas'?'canvas':'div');}
 getContext(){return {fillRect(){},drawImage(){}};}
 addEventListener(t,f,opts={}){(this.listeners[t]??=[]).push({f,once:opts.once});}
 emit(t){const list=this.listeners[t]||[];this.listeners[t]=list.filter(x=>!x.once);for(const x of list)x.f();}
 pause(){} load(){} removeAttribute(k){delete this[k];}
 get currentTime(){return this.time??0;}
 set currentTime(t){this.time=t;queueMicrotask(()=>this.emit('seeked'));}
}
const elements={};for(const m of html.matchAll(/\bid="([^"]+)"/g))elements[m[1]]=new Element(m[1]==='actor'||m[1]==='models'?'select':'div');
elements.data.textContent=JSON.stringify(data);elements.asset.checked=true;elements.speed.value='1';
const document={getElementById:id=>elements[id],activeElement:{tagName:'BODY'},addEventListener(){},createElement(tag){const e=new Element(tag);if(tag==='video')Object.assign(e,{readyState:2,videoWidth:480,videoHeight:320,duration:4.9});return e;}};
const context=vm.createContext({document,console,performance,queueMicrotask,cancelAnimationFrame(){},requestAnimationFrame(){return 1;},setTimeout});
vm.runInContext(js,context);const run=s=>vm.runInContext(s,context),flush=()=>new Promise(r=>setImmediate(r));
(async()=>{
 await flush();
 const initial=data.initial_model||'gn11five';assert.equal(elements.models.value,initial);
 if(data.models[initial].all_vehicle_stage1){assert.equal(elements.actor.value,'all');assert.equal(elements.actor.options.length,1);assert.equal(elements.actor.disabled,true);assert.equal(elements['asset-control'].hidden,true);assert.ok(!elements.actor.options[0].textContent.includes('a6'));}
 else{assert.equal(elements.actor.options.length,5);assert.equal(elements.actor.disabled,false);assert.equal(elements['asset-control'].hidden,false);}
 const checks=[];
 for(const model of ['old5','r15','repair4k','matched5'].filter(m=>data.models[m]?.ready)){
  elements.models.value=model;await elements.models.onchange();
  assert.deepEqual(elements.actor.options.map(x=>x.value),['6','37','5','26','18']);assert.equal(elements.actor.disabled,false);
  for(const a of ['6','37','5','26','18']){
   await run('seek(128)');elements.actor.value=a;await elements.actor.onchange();await flush();
   const rendered=run('entries.map(e=>e.video&&e.video.src).filter(Boolean)');
   const actor=data.models[model].actors[a];for(const rel of [actor.isolated,actor.delete_no_shadow||actor.delete,actor.dynamic_delete])assert.ok(rendered.includes(rel),`${model}/a${a}/${rel}`);
   for(const rel of rendered)assert.ok(fs.statSync(path.resolve(root,rel)).size>0,rel);
   assert.ok(elements.visibility.textContent.startsWith('a'+a+' '));
   if(a==='5')assert.equal(elements.frame.value,data.best_frame[a]);
   checks.push(`${model}/a${a}`);
  }
 }
 elements.actor.value='37';await elements.actor.onchange();
 const collections=[];for(const name of ['gn11five','allvehicles'].filter(m=>data.models[m]?.ready)){
  elements.models.value=name;await elements.models.onchange();assert.equal(elements.actor.value,'all');assert.ok(elements.scope.textContent.includes('整个动态分支'));assert.equal(elements.actor.disabled,true);assert.equal(elements['asset-control'].hidden,true);collections.push(name);
 }
 elements.models.value='gn11five,old5';await elements.models.onchange();assert.equal(elements.actor.value,'37');assert.equal(elements.actor.disabled,false);assert.equal(elements.panels.children.length,2);assert.ok(elements.scope.textContent.includes('只作用于'));
 for(let raw=89;raw<=137;raw++){await run(`seek(${raw})`);assert.equal(elements.frame.value,raw);for(const t of run('entries.map(e=>e.video&&e.video.currentTime).filter(t=>t!==null)'))assert.ok(Math.abs(t-((raw-89)/10+.001))<1e-8);}
 const receipt={status:'PASSED_GENERATED_PAGE_IDENTITY_CONTROL_CHECK',initial_model_checked:initial,identity_cases:checks,collection_models_checked:collections,stage1_collection_label:true,model_switch_preserves_identity:true,mixed_stage_scope_checked:true,synchronized_frames_checked:49,media_paths_exist:true,full_frame:true,browser_interaction_verified:false,media_decoder_verified:false};
 fs.writeFileSync(path.join(root,'player_controls_validation.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
})().catch(e=>{console.error(e);process.exitCode=1;});
