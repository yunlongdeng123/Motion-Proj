const fs=require('fs'),vm=require('vm');
const root='outputs/V43_R4_FiveCase_GPU_20261101';let callback,inserted;
const nodes={sceneTitle:{textContent:'Waymo · scene-212（训练集）'},actorTitle:{textContent:'actor a0'},actorMediaPanel:{before(x){inserted=x}},scene:{addEventListener(){}},actor:{addEventListener(){}}};
const context={document:{createElement(){return {append(){}}},getElementById:id=>nodes[id]},location:{hash:'#scene=waymo-train-212&actor=a0'},window:{addEventListener(t,f){callback=f}},URLSearchParams,MutationObserver:function(){this.observe=()=>{}},queueMicrotask:f=>f()};
vm.runInNewContext(fs.readFileSync(root+'/source/edit_comparison_entry.js','utf8'),context);
for(const c of ['waymo-train-008_a37','waymo-train-212_a0','waymo-train-212_a35','waymo-train-564_a0','waymo-train-564_a22','waymo-train-537_a0','waymo-validation-212_a0']){
 const at=c.lastIndexOf('_');context.location.hash='#scene='+c.slice(0,at)+'&actor='+c.slice(at+1);callback();
 const wanted=!c.includes('537')&&!c.includes('validation');if(inserted.hidden===wanted)throw Error('scope failed '+c);
}
console.log('entry scope: 5 cases shown; 2 unrelated scenes hidden PASS');
