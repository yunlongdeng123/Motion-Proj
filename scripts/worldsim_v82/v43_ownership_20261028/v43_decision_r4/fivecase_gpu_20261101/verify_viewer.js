const fs=require('fs'),vm=require('vm'),path=require('path');
const root=process.argv[2];let selectedCase='';
const html=fs.readFileSync(path.join(root,'review/index.html'),'utf8');
const elements=new Map();
function el(id){if(!elements.has(id))elements.set(id,{value:'',hidden:false,textContent:'',src:'',options:[],add(o){this.options.push(o);if(this.value==='')this.value=String(o.value)},replaceChildren(){this.options=[];this.value=''},addEventListener(){},pause(){},load(){}});return elements.get(id)}
Object.entries({op:'none',mode:'video',domain:'fullsource',frame:'1'}).forEach(([id,v])=>el(id).value=v);el('showBoxes').checked=true;
const ctx={window:{},document:{getElementById:el},Option:function(text,value){this.text=text;this.value=value},history:{replaceState(a,b,c){selectedCase=c}},location:{hash:'#case=waymo-train-212_a0'},URLSearchParams,console};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(root,'review/data.js'),'utf8'),ctx);
const script=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');vm.runInContext(script,ctx);
for(let i=0;i<ctx.window.FIVE_DATA.length;i++){
 el('case').value=String(i);vm.runInContext('choose()',ctx);const c=ctx.window.FIVE_DATA[i];
 if(selectedCase!=='#case='+encodeURIComponent(c.case))throw Error('wrong case navigation');
 for(const [domain,d] of Object.entries(c.domains)){
  el('domain').value=domain;vm.runInContext('routeOptions()',ctx);
  for(const op of ['none','delete','shift','rotate']){
   el('op').value=op;
   for(const boxed of [false,true]){
    el('showBoxes').checked=boxed;vm.runInContext('video()',ctx);
    if(!fs.existsSync(path.join(root,'review',el('video').src)))throw Error('missing video');
   }
   for(let k=0;k<3;k++)for(let r=0;r<d.routes;r++){
    el('frame').value=String(k);el('left').value='h'+String(r).padStart(3,'0');el('right').value='h'+String(d.routes-1-r).padStart(3,'0');vm.runInContext('paint()',ctx);
    for(const side of ['leftImage','rightImage'])if(!fs.existsSync(path.join(root,'review',el(side).src)))throw Error('blank image navigation '+c.case+' '+domain);
   }
  }
 }
}
console.log('viewer navigation PASS');
