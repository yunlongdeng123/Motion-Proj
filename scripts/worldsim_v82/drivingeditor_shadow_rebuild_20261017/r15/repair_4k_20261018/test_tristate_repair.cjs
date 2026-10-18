// Logic/canvas-call validation only; no browser or file:// acceptance claim.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const file=path.resolve(__dirname,'../../outputs/r15_repair_4k_20261018/tristate/index.html');
const html=fs.readFileSync(file,'utf8'),json=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const source=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]).join('\n');
let paints=0;const options={disabled:false},ctx={scale(){},fillRect(x,y,w,h){assert([x,y,w,h].every(Number.isFinite));paints++;},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){}};
const el={};for(const id of ['data','scope','context','pool','targetmode','pair','left','right','leftcap','rightcap','details','top','side','persp','reset'])
el[id]={value:id==='scope'?'a6':id==='pair'?'original':'',checked:id==='targetmode',textContent:id==='data'?json:'',clientWidth:650,clientHeight:440,
 addEventListener(){},querySelector(){return options;},getContext(){return ctx;},setPointerCapture(){},getBoundingClientRect(){return {left:0,top:0};}};
const states=[0,1,2].map(n=>({checked:true,dataset:{state:String(n)},addEventListener(){}}));
const context={console,atob,Uint8Array,Float32Array,Set,Math,JSON,Number,Array,devicePixelRatio:1,
 document:{getElementById:id=>el[id],querySelectorAll:()=>states.filter(x=>x.checked)},window:{addEventListener(){}}};
vm.createContext(context);vm.runInContext(source,context);
const evaluate=code=>vm.runInContext(code,context);
for(const actor of [6,37,5,26,18]){
 el.scope.value='a'+actor;el.pair.value='original';el.context.checked=false;evaluate('selectPanels()');
 assert.equal(evaluate("chosen.map(x=>x.stage).join(',')"),'4k,repair4k');
 const unknown=evaluate('chosen[1].array.filter((x,i)=>i%12===3&&x===0).length');
 assert.equal(unknown,({6:50,37:20,5:53,26:27,18:0})[actor]);
 el.context.checked=true;evaluate('selectPanels()');assert(evaluate(`chosen[1].array.some((x,i)=>i%12===4&&x===${actor})`),'Context must retain the target.');
 for(const pair of ['initial','old5']){el.pair.value=pair;evaluate('selectPanels()');assert.equal(evaluate('chosen[1].stage'),'repair4k');}
}
for(const id of ['top','side','persp','reset'])el[id].onclick();
el.pair.value='original';el.scope.value='all';el.context.checked=false;evaluate('selectPanels()');
assert.equal(evaluate("chosen.map(x=>x.stage).join(',')"),'4k,repair4k');
assert(paints>1000);const result={status:'PASS',cases:'5actors × original/initial/old5; target-preserving contexts; exact unknown counts; global scene; view controls',browser_acceptance:false,canvas_calls:paints};fs.writeFileSync(path.join(path.dirname(file),'validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
