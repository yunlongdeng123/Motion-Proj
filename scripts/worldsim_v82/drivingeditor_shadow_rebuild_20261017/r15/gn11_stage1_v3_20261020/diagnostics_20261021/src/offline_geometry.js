/* 本地几何分块；file:// 通过同目录 script 读取，不依赖fetch或本地服务。 */
window.GSOfflineGeometry=(()=>{
 'use strict';
 const chunks=new Map();
 const fields=['row','xyz','sigma','q','velocity','t','sigma_t','opacity','color','role','road_id'];
 function available(arm){return Boolean(window.GSDATA.geometryIndex?.arms?.[arm]);}
 async function read(record,unpack){
  if(chunks.has(record.key))return chunks.get(record.key);
  const promise=new Promise((resolve,reject)=>{
   const script=document.createElement('script');script.src=record.path;
   script.onload=()=>{script.remove();const value=window.GSDATA[record.key];delete window.GSDATA[record.key];value?resolve(value):reject(Error('分块缺少数据 '+record.path));};
   script.onerror=()=>{script.remove();reject(Error('缺少本地分块 '+record.path));};
   document.head.append(script);
  }).then(unpack).catch(e=>{chunks.delete(record.key);throw e;});
  chunks.set(record.key,promise);while(chunks.size>3)chunks.delete(chunks.keys().next().value);
  return promise;
 }
 function selection(total,percent,flags,bit,vehiclePercent){
  const base=Math.ceil(total*percent/100);let candidates=0;
  if(flags)for(let i=0;i<flags.length;i++)if(flags[i]&bit)candidates++;
  const wanted=Math.ceil(candidates*Math.max(percent,vehiclePercent)/100),extra=[];
  let seen=0,sampled=0;
  if(flags)for(let i=0;i<flags.length;i++)if(flags[i]&bit){if(i<base||seen<wanted)sampled++;if(i>=base&&seen<wanted)extra.push(i);seen++;}
  const selected=new Uint32Array(base+extra.length);for(let i=0;i<base;i++)selected[i]=i;selected.set(extra,base);
  return {selected,base,candidates,sampled};
 }
 async function make(arm,raw,p,base,unpack,progress,cancelled=()=>false){
  const index=window.GSDATA.geometryIndex,meta=index.arms[arm],out={...base,branches:base.branches.map(b=>({...b}))},stats={};
  const actor=index.actors.indexOf(Number(p.vehicle));
  const members=p.vehicle==='none'?null:await read(meta.members[String(raw)],unpack);
  for(const name of ['static','dynamic']){
   const prefix=name+'_',info=meta.branches[name],percent=name==='static'?p.static_rate:p.dynamic_rate;
   const binding=base.branches.find(b=>b.branch===name);if(info.checkpoint_sha256!==binding.checkpoint_sha256||info.total!==binding.total)throw Error('几何分块与默认样本检查点不一致，请重新导出');
   const flags=members?.[name+'_'+p.vehicle_scope];if(members&&(!flags||flags.length!==info.ordinary_total))throw Error('单车区域索引与几何数量不一致');
   const selected=selection(info.ordinary_total,percent,flags,1<<actor,p.vehicle_rate),ranks=selected.selected;
   const road=[];for(let i=0;i<base[prefix+'row'].length;i++)if(base[prefix+'role'][i]===1)road.push(i);
   if(road.length!==info.road_total)throw Error('道路全量样本与几何索引不一致');
   const count=ranks.length+road.length;
   for(const field of fields){const old=base[prefix+field],stride=old.length/base[prefix+'row'].length;out[prefix+field]=new old.constructor(count*stride);}
   let cursor=0;
   for(const record of info.chunks){
    if(cursor>=ranks.length)break;if(ranks[cursor]>=record.start+record.count)continue;
    if(cancelled())throw Error('采样已切换');
    progress(`本地读取 ${arm} / ${name} 分块 ${Math.floor(record.start/index.chunk_size)+1}…`);
    const chunk=await read(record,unpack),begin=cursor;
    while(cursor<ranks.length&&ranks[cursor]<record.start+record.count)cursor++;
    // 连续基础采样一次复制；车辆区域的离散点按连续段复制。
    for(let pos=begin;pos<cursor;){let end=pos+1;while(end<cursor&&ranks[end]===ranks[end-1]+1)end++;
     for(const field of fields){const value=chunk[field],stride=value.length/record.count;out[prefix+field].set(value.subarray((ranks[pos]-record.start)*stride,(ranks[end-1]-record.start+1)*stride),pos*stride);}pos=end;
    }
   }
   if(cursor!==ranks.length)throw Error('本地几何分块不完整');
   for(const field of fields){const old=base[prefix+field],stride=old.length/base[prefix+'row'].length;for(let i=0;i<road.length;i++)out[prefix+field].set(old.subarray(road[i]*stride,(road[i]+1)*stride),(ranks.length+i)*stride);}
   out.branches.find(b=>b.branch===name).displayed=count;
   stats[name]={total:info.total,ordinary_total:info.ordinary_total,road_total:road.length,base_sampled:selected.base,sampled:count,vehicle_candidates:selected.candidates,vehicle_sampled:selected.sampled};
  }
  out.sampling={...p,raw,stats,roads_all_retained:true,gpu_used:false,optimizer_updates:0,offline:true};
  return out;
 }
 return {available,make,selection};
})();
