// Actual viewer logic with DOM/image substitutes; this is not browser QA.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(process.argv[2]),asset='native_delete30k_20261109';
const html=fs.readFileSync(path.join(root,'index.html'),'utf8'),source=fs.readFileSync(path.join(root,asset,'viewer.js'),'utf8');
const data=JSON.parse(fs.readFileSync(path.join(root,asset,'data.js'),'utf8').trim().split('=').slice(1).join('=').slice(0,-1));
const nodes=new Map(),events={},pending=new Map(),checks=[],state={auto:true,saved:null,timer:null};
const ids=['observed','native','continued','edited'];
function check(name,ok){assert(ok,name);checks.push({name,pass:true});}
class E{constructor(tag='div'){this.tagName=tag.toUpperCase();this.style={};this.children=[];this.width=960;this.height=640;this.value='';this.ctx={last:null};for(const k of ['save','restore','beginPath','rect','clip','setLineDash','strokeRect','fillText','moveTo','lineTo','stroke'])this.ctx[k]=()=>{};this.ctx.fillRect=()=>this.ctx.last=null;this.ctx.drawImage=(im,...args)=>this.ctx.last={src:im.src,args};}append(...v){this.children.push(...v);}getContext(){return this.ctx;}getBoundingClientRect(){return {left:10,top:20,width:480,height:320};}click(){this.onclick?.({target:this});}}
for(const m of html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)){assert(!nodes.has(m[2]),'duplicate '+m[2]);nodes.set(m[2],new E(m[1]));}
for(const [k,v]of Object.entries({view:'full',overlay:'none',sampling:'smooth',fps:'5',experiment:'native'}))nodes.get(k).value=v;
const ctx={window:null,Blob,URLSearchParams,console,NATIVE_QUALITY_ROOT:asset+'/',NATIVE_QUALITY_DATA:data,
 location:{hash:'#raw=119&step=30000',href:''},history:{replaceState:(_a,_b,v)=>ctx.location.hash=v},
 document:{getElementById:k=>{if(!nodes.has(k))throw Error('missing '+k);return nodes.get(k);},createElement:t=>new E(t),addEventListener:(n,f)=>events[n]=f},
 URL:{createObjectURL:b=>(state.saved=b,'blob:mock'),revokeObjectURL:()=>{}},setInterval:f=>(state.timer=f,1),clearInterval:()=>state.timer=null};
ctx.window=ctx;ctx.Image=class{set src(p){this._src=p;pending.set(p,this);if(state.auto)queueMicrotask(()=>this.onload?.());}get src(){return this._src;}};
const tick=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
const jump=raw=>nodes.get('rawNumber').onchange({target:{value:raw}});
function finish(raw,{step=20000,mode='keep',missing=false}={}){const row=data.frames[raw];for(const k of ['observed','native',step===20000?'native20':'native30',mode==='keep'?'edited_keep':'delete_white']){const im=pending.get(asset+'/'+row[k]);if(im)(missing&&k==='edited_keep'?im.onerror:im.onload)?.();}}
vm.createContext(ctx);vm.runInContext(source,ctx);
(async()=>{await tick();
 check('Correct scene, new deletion experiment and199 complete frames',data.scene==='065'&&data.experiment_id==='WAYMO-065-DELETE-WHITE30K-20261109'&&data.frames.length===199&&data.frames.every((f,i)=>f.raw===i));
 check('Four canvases default to requested119/30K/delete',ids.every(id=>nodes.get(id).ctx.last)&&nodes.get('continued').ctx.last.src.endsWith('native_30000/119.png')&&nodes.get('edited').ctx.last.src.endsWith('delete_white/119.png'));
 check('No holdout and full0..198 timeline',data.heldout===0&&Number(nodes.get('frame').min)===0&&Number(nodes.get('frame').max)===198);
 check('Actual MP4 link initially points to deletion',nodes.get('edit-video').href.endsWith('delete_white_30k.mp4'));
 nodes.get('continuation-step').onchange({target:{value:'20000'}});await tick();
 check('20K switches only native comparison; edit remains30K',nodes.get('continued').ctx.last.src.endsWith('native_20000/119.png')&&nodes.get('edited').ctx.last.src.endsWith('delete_white/119.png')&&nodes.get('status').textContent.includes('第四列 30K'));
 nodes.get('edit-mode').onchange({target:{value:'keep'}});await tick();
 check('Editable KEEP switches fourth column and movie together',nodes.get('edited').ctx.last.src.endsWith('keep/119.png')&&nodes.get('edit-video').href.endsWith('keep_30k.mp4')&&ctx.location.hash.includes('edit=keep'));
 state.auto=false;jump(120);check('Dragging clears all four old images',ids.every(k=>nodes.get(k).ctx.last===null));finish(120,{missing:true});await tick();
 check('Failed edit frame preserves three current native/source views',ids.slice(0,3).every(k=>nodes.get(k).ctx.last?.src.endsWith('/120.png'))&&nodes.get('edited').ctx.last===null&&nodes.get('status').textContent.includes('暂不可用'));
 jump(121);jump(122);finish(122);await tick();finish(121);await tick();
 check('Out-of-order asynchronous results never show older frames',ids.every(k=>nodes.get(k).ctx.last?.src.endsWith('/122.png')));
 state.auto=true;nodes.get('edited').onclick({clientX:310,clientY:220});nodes.get('view').value='pixel';nodes.get('view').onchange();
 check('All four views share selected pixel and crop',ids.every(k=>nodes.get(k).ctx.last.args[0]===220&&nodes.get(k).ctx.last.args[1]===147));
 nodes.get('sampling').value='nearest';nodes.get('sampling').onchange();check('Pixel resampling applies to all four',ids.every(k=>nodes.get(k).style.imageRendering==='pixelated'));
 nodes.get('play').click();check('Play advances all four on same timeline',!!state.timer);state.timer();await tick();check('Playback frame sync',ids.every(k=>nodes.get(k).ctx.last?.src.endsWith('/123.png')));nodes.get('play').click();check('Pause stops shared timer',state.timer===null);
 nodes.get('note').value='review';nodes.get('save').click();const saved=JSON.parse(await state.saved.text());
 check('Export records edit mode and comparison step',saved.schema==='v44.delete30k.review.v1'&&saved.edit_mode==='keep'&&saved.continuation_step===20000&&saved.human_verdict===null);
 nodes.get('recordFile').files=[{size:200,text:async()=>JSON.stringify({...saved,raw:150,edit_mode:'delete',continuation_step:30000})}];await nodes.get('recordFile').onchange({target:nodes.get('recordFile')});await tick();
 check('Record restores frame, edit query and native step',nodes.get('edited').ctx.last.src.endsWith('delete_white/150.png')&&nodes.get('continued').ctx.last.src.endsWith('native_30000/150.png'));
 nodes.get('recordFile').files=[{size:200,text:async()=>JSON.stringify({...saved,schema:'v44.native_continuation.review.v1'})}];await nodes.get('recordFile').onchange({target:nodes.get('recordFile')});
 check('Prior continuation records remain routed to original viewer',nodes.get('recordStatus').textContent.includes('“原生续训对照”'));
 nodes.get('experiment').onchange({target:{value:'continuation'}});check('Previous continuation page preserves frame/step',ctx.location.href==='native_continuation.html#raw=150&step=30000');
 nodes.get('experiment').onchange({target:{value:'previous'}});check('Historical editing remains available',ctx.location.href===data.previous_url+'#raw=150');
 const result={status:'PASS_CPU_DOM_LOGIC',checks,browser_qa:false,images_validated:false};fs.writeFileSync(path.join(root,'logic_checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({status:result.status,passed:checks.length}));
})().catch(e=>{console.error(e);process.exitCode=1;});
