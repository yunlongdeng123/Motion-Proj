// Exact checkpoint population, streamed only when the user applies rates.
window.CLOUD_CHUNKS=window.CLOUD_CHUNKS||{};
const fullCloudCache=new Map();
function rowFraction(id){return ((id+1)*0.6180339887498949)%1;} // Stable low-discrepancy row sampling; same subset at the same rate.
async function fullCloud(context,part){
 const key=context+':'+part;if(fullCloudCache.has(key))return fullCloudCache.get(key);
 const promise=(async()=>{
  const prefix=context===D.context?'':'../'+context+'/';
  await loadScript(prefix+'cloud_full_manifest.js');
  const meta=window.FULL_CLOUD_MANIFEST[part],out={};
  for(let i=0;i<meta.chunks.length;i++){
   const chunk=meta.chunks[i];$('samplingStatus').textContent=`加载 ${context.slice(0,6)} ${part}：${i+1}/${meta.chunks.length} 块`;
   await loadScript(prefix+chunk.path);const data=await unpack(window.CLOUD_CHUNKS[chunk.path]);delete window.CLOUD_CHUNKS[chunk.path];
   for(const [k,v] of Object.entries(data)){const stride=v.length/chunk.rows;if(!out[k])out[k]=new v.constructor(meta.total*stride);out[k].set(v,chunk.start*stride);}
  }
  return {...meta,sample:meta.total,data:out,fullPopulation:true};
 })();fullCloudCache.set(key,promise);try{return await promise;}catch(e){fullCloudCache.delete(key);throw e;}
}
async function applySampling(){
 if(S.samplingLoading)return;S.samplingLoading=true;$('applySampling').disabled=true;
 try{
  const settings={static:Number($('staticRate').value),dynamic:Number($('dynamicRate').value),target:Number($('targetRate').value),boost:$('targetBoost').value};
  for(const k of ['static','dynamic','target'])if(!Number.isFinite(settings[k])||settings[k]<0||settings[k]>100)throw Error('采样率须在 0—100% 内');
  for(const part of ['static','dynamic'])S.clouds[part]=await fullCloud(D.context,part);
  for(const instance of geometryInstances())if(instance.cloud.startsWith('foreign:'))S.clouds[instance.cloud]=await fullCloud(instance.source_context,'dynamic');
  S.sampling=settings;S.selected=null;draw3D();
  $('samplingStatus').textContent=`已应用全模型确定性采样：静态 ${settings.static}% · 动态 ${settings.dynamic}% · ${settings.boost==='none'?'不单独加密':'框邻域 '+settings.target+'%'}。100% 不抽行；时间/opacity 门限仍生效。`;
 }catch(e){$('samplingStatus').textContent='未应用：'+e.message;}finally{S.samplingLoading=false;$('applySampling').disabled=false;}
}
function sampleBase(cloud,part,j){
 if(!S.sampling||!cloud.fullPopulation)return true;
 return rowFraction(cloud.data.ids[j])<S.sampling[part]/100;
}
function sampleBoost(pt,id){
 if(!S.sampling||S.sampling.boost==='none'||rowFraction(id)>=S.sampling.target/100)return false;
 // Display-only oriented target-box neighborhood, never ownership or point removal.
 for(const b of currentBoxes().filter(b=>b.target)){
  const q=b.corners_local,delta=pt.map((v,k)=>v-q[0][k]);let inside=true;
  for(const i of [1,2,4]){const e=q[i].map((v,k)=>v-q[0][k]),len=Math.hypot(...e);const d=delta.reduce((s,v,k)=>s+v*e[k],0)/len;if(d< -1||d>len+1){inside=false;break;}}
  if(inside)return true;
 }
 return false;
}
function ownershipRow(instance,id){
 if(!instance)return null;
 if(instance.cloud.startsWith('v43:'))return null; // Different row namespace; never reuse old 30K labels.
 const context=instance.cloud.startsWith('foreign:')?instance.source_context:D.context;
 const part=instance.cloud==='static'?'static':'dynamic',a=S.ownership?.[context]?.[part];
 return a?{state:a[id*3],target_hits:a[id*3+1],other_hits:a[id*3+2],context,row:id}:null;
}
async function loadOwnership(){
 try{
  S.ownership=S.ownership||{};window.OWNERSHIP_DATA=window.OWNERSHIP_DATA||{};
  const contexts=new Set([D.context,...geometryInstances().filter(o=>o.cloud.startsWith('foreign:')).map(o=>o.source_context)]);
  for(const context of contexts){if(S.ownership[context])continue;await loadScript((context===D.context?'':'../'+context+'/')+'ownership_candidates.js');
   const raw=window.OWNERSHIP_DATA[context];S.ownership[context]={};for(const part of ['static','dynamic'])S.ownership[context][part]=(await unpack(raw[part])).values;
  }
  $('cloudColor').value='ownership';await applySampling();draw3D();
 }catch(e){$('ownershipStatus').textContent='候选未加载：'+e.message;}
}
