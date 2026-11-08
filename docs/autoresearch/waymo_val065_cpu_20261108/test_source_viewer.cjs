// 执行实际脚本的DOM/Canvas替身；不宣称浏览器视觉验收。
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(__dirname,'../../outputs/waymo_val065_preparation');
const nodes=new Map(),events={};
function node(){const n={width:960,height:640,value:'',dataset:{},draws:[],textContent:'',getContext:()=>new Proxy({drawImage:(im,...args)=>n.draws.push({src:im.src,args})},{get:(t,k)=>k in t?t[k]:()=>{}})};return n;}
for(const id of ['full','crop','prev','next','frame','label','source'])nodes.set(id,node());
const buttons=[30,60,150,198].map(raw=>{const b=node();b.dataset.raw=raw;return b;});
global.window=global;global.document={getElementById:id=>{assert(nodes.has(id));return nodes.get(id);},querySelectorAll:()=>buttons,addEventListener:(event,fn)=>events[event]=fn};
global.Image=class{set src(p){this._src=p;queueMicrotask(()=>{assert(fs.existsSync(path.join(root,p)));this.onload();});}get src(){return this._src;}};
vm.runInThisContext(fs.readFileSync(path.join(root,'assets/source.js'),'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(root,'assets/viewer.js'),'utf8'));
const tick=()=>new Promise(resolve=>setTimeout(resolve,5));
async function main(){await tick();assert.strictEqual(CpuCaseViewer.getRaw(),60);CpuCaseViewer.set(78);await tick();assert.strictEqual(CpuCaseViewer.getRaw(),78);nodes.get('next').onclick();await tick();assert.strictEqual(CpuCaseViewer.getRaw(),119);assert(CPU_CASE.frames.every(f=>f.raw<79||f.raw>118));for(const b of buttons){b.onclick();await tick();assert.strictEqual(CpuCaseViewer.getRaw(),b.dataset.raw);assert(nodes.get('full').draws.at(-1).src.endsWith('/'+b.dataset.raw+'.jpg'));}CpuCaseViewer.set(-100);assert.strictEqual(CpuCaseViewer.getRaw(),0);assert(nodes.get('prev').disabled);CpuCaseViewer.set(999);assert.strictEqual(CpuCaseViewer.getRaw(),198);assert(nodes.get('next').disabled);CpuCaseViewer.set(30);CpuCaseViewer.set(100);await tick();assert.strictEqual(CpuCaseViewer.getRaw(),140);assert(nodes.get('full').draws.at(-1).src.endsWith('/140.jpg'));console.log('PASS source viewer logic, BUILD gap, four pilot frames, bounds and rapid jumps; not browser QA');}
main().catch(error=>{console.error(error);process.exitCode=1;});
