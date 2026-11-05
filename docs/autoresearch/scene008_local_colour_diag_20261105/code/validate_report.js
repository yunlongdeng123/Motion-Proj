const fs=require('fs'),path=require('path'),vm=require('vm');
const root=path.resolve(process.argv[2]);
const page=fs.readFileSync(path.join(root,'index.html'),'utf8');
const script=page.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const loaded=new Set();let draws=0;
const nodes={};
class Element{
  constructor(){this.value='';this.children=[];this.width=480;this.height=320;}
  append(...items){this.children.push(...items);for(const item of items)if(item.id)nodes[item.id]=item;}
  replaceChildren(...items){this.children=items;}
  getContext(){return {drawImage(){draws++;},strokeRect(){}};}
}
const document={getElementById:id=>nodes[id]||(nodes[id]=new Element()),createElement:()=>new Element()};
nodes.frame=new Element();nodes.slider=new Element();nodes.state=new Element();nodes.state.value='keep';
nodes.roi=new Element();nodes.roi.value='full';
class Image{
  set src(value){if(!fs.existsSync(path.join(root,value)))throw Error('Missing '+value);loaded.add(value);queueMicrotask(()=>this.onload());}
}
const context={document,Image};vm.createContext(context);vm.runInContext(script,context);
(async()=>{
  await new Promise(setImmediate);
  let combinations=0;
  for(const state of ['keep','delete']){
    nodes.state.value=state;context.frameOptions();
    const frames=state==='keep'?Array.from({length:49},(_,i)=>89+i):[105,113,130];
    for(const raw of frames)for(const roi of ['full','left_road_patch','right_curb_patch','right_building','left_trees']){
      nodes.frame.value=String(raw);nodes.roi.value=roi;await context.update();combinations++;
      if(nodes.load.textContent!==`raw${raw}`)throw Error('Frame did not finish loading');
      if(!nodes.numbers.innerHTML.includes('原生 10K'))throw Error('Metrics not populated');
    }
  }
  const refs=[...page.matchAll(/(?:src|href)="([^"#]+)"/g)].map(x=>x[1]);
  for(const value of refs)if(!fs.existsSync(path.join(root,value)))throw Error('Missing static reference '+value);
  const result={status:'PASS',actual_report_handlers_executed:true,frame_region_combinations:combinations,
                unique_gallery_images:loaded.size,canvas_draws:draws,static_references:refs.length};
  fs.writeFileSync(path.join(root,'evidence/report_QA.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exitCode=1});
