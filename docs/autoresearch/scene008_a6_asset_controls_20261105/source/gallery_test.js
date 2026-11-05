const fs=require('fs'),path=require('path'),vm=require('vm');
const root=path.resolve(process.argv[2]);const page=fs.readFileSync(path.join(root,'index.html'),'utf8');
const script=page.match(/<script>([\s\S]*?)<\/script>/)[1];const nodes={arm:{value:'m2'},tintRaw:{value:'113'},tintState:{value:'delete'}};
const context={document:{getElementById:key=>nodes[key]||(nodes[key]={value:'',src:'',textContent:''})}};
vm.createContext(context);vm.runInContext(script,context);let count=0;
for(const arm of ['m1','m2','m3'])for(let raw=89;raw<=137;raw++){
  nodes.arm.value=arm;context.show(raw);
  for(const id of ['objects','street']){if(!fs.existsSync(path.join(root,nodes[id].src)))throw Error('Missing '+nodes[id].src);count++;}
}
for(const raw of [105,113,130])for(const state of ['keep','delete']){
  nodes.tintRaw.value=String(raw);nodes.tintState.value=state;context.tint();
  if(!fs.existsSync(path.join(root,nodes.tint.src)))throw Error('Missing '+nodes.tint.src);count++;
}
const result={status:'PASS',all49_all3arms_actual_JS_handlers:true,resolved_images_checked:count};
console.log(JSON.stringify(result));
fs.writeFileSync(path.join(root,'evidence/gallery_UI_QA.json'),JSON.stringify(result,null,2)+'\n');
