/* Geometry-preview controls. Sampling never changes native rendering or attribution. */
window.GSCloud=(()=>{
 'use strict';
 const defaults={static_rate:2,dynamic_rate:2,vehicle:'none',vehicle_rate:10,vehicle_scope:'projection'};
 const cache=new Map();let state,api,cloudEpoch=0,lastGood=null,notice='',counts=null;
 const el=id=>document.getElementById(id);
 function valid(p){const q={...defaults,...p};for(const k of ['static_rate','dynamic_rate','vehicle_rate']){q[k]=Number(q[k]);if(!Number.isFinite(q[k])||q[k]<0||q[k]>100)throw Error('采样率须在0—100%之间');}if(q.vehicle!=='none'){q.vehicle=Number(q.vehicle);if(![5,26,18,6,37].includes(q.vehicle))throw Error('未知车辆');}if(!['projection','box'].includes(q.vehicle_scope))throw Error('未知单车区域');return q;}
 function hash(x){x=(x^(x>>>16))>>>0;x=Math.imul(x,0x7feb352d)>>>0;x=(x^(x>>>15))>>>0;x=Math.imul(x,0x846ca68b)>>>0;return (x^(x>>>16))>>>0;}
 function sync(){for(const k of Object.keys(defaults)){const n=el('sample_'+k);if(n)n.value=state.preview_sampling[k];}el('sample_vehicle_rate').disabled=state.preview_sampling.vehicle==='none';el('sample_vehicle_scope').disabled=state.preview_sampling.vehicle==='none';}
 function message(t,bad=false){notice=t;const n=el('samplingState');if(n){n.textContent=t;n.classList.toggle('bad',bad);}}
 function bind(s,a){state=s;api=a;state.preview_sampling=valid(state.preview_sampling);
  const controls=document.createElement('div');controls.className='sampling-controls';controls.innerHTML=`<label>静态采样率（普通背景）<span><input id="sample_static_rate" type="number" min="0" max="100" step="0.5" value="2" aria-label="静态采样率"> %</span></label><label>动态采样率<span><input id="sample_dynamic_rate" type="number" min="0" max="100" step="0.5" value="2" aria-label="动态采样率"> %</span></label><label>单独加密车辆<select id="sample_vehicle"><option value="none">不单独加密</option>${[5,26,18,6,37].map(i=>`<option value="${i}">a${i}</option>`).join('')}</select></label><label>单车采样率<span><input id="sample_vehicle_rate" type="number" min="0" max="100" step="1" value="10" aria-label="单车采样率"> %</span></label><input type="hidden" id="sample_vehicle_scope" value="projection"><button id="applySampling">应用采样</button><label>点透明度<select id="pointAlpha"><option value="effective">按真实有效 opacity</option><option value="solid">不透明中心 · 看分布</option></select></label><button id="isolateVehicle">聚焦所选车辆</button>`;
  el('cloudColor').closest('.subtools').after(controls);
  const msg=document.createElement('p');msg.id='samplingState';msg.className='note';msg.setAttribute('role','status');controls.after(msg);
  const help=document.createElement('p');help.className='note';help.textContent='V3道路全量显示；V1地面候选全量载入，初始／训练后显示率独立可调。单车加密同时覆盖普通背景与动态，是当前帧区域采样，不是永久车辆归属。投影区域为原始车身掩码外扩5像素，包含车后背景。';msg.after(help);
  el('sceneCanvas').nextElementSibling.id='sceneLegend';
  el('applySampling').onclick=()=>{try{const p={};for(const k of Object.keys(defaults))p[k]=el('sample_'+k).value;state.preview_sampling=valid(p);sync();api.stop();api.refresh();}catch(e){message(e.message,true);}};
  el('sample_vehicle').onchange=()=>{const off=el('sample_vehicle').value==='none';el('sample_vehicle_rate').disabled=off;el('sample_vehicle_scope').disabled=off;};
  el('pointAlpha').onchange=api.draw3D;
  el('isolateVehicle').onclick=()=>{const actor=Number(el('sample_vehicle').value),box=vehicles().find(b=>b.actor===actor);if(box)api.focus(box.center);else message('先选择一辆车，再点击聚焦。');};
  sync();message('默认静态／动态各2%；调高比例或单车加密时读取同目录几何分块，无需启动服务。');
 }
 function vehicles(){return window.GSDATA.samplingMeta?.vehicles?.[String(state.raw)]||[];}
 function vehicleLabels(scene){if(!state)return;const cam=scene.cameras[state.raw-89],W=cam.W;
  for(const b of vehicles()){const p=b.center,c=W.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]),dist=c[2]/scene.meter_to_model,side=c[0]<-.1?'左侧':c[0]>.1?'右侧':'本车道';const o=el('sample_vehicle').querySelector(`option[value="${b.actor}"]`);if(o)o.textContent=`a${b.actor} · ${side} · ${dist.toFixed(1)}m`;}
 }
 function reduced(base,p){const out={...base,branches:base.branches.map(b=>({...b}))};for(const prefix of ['static_','dynamic_']){const rows=base[prefix+'row'],role=base[prefix+'role'],branch=out.branches.find(b=>b.branch===(prefix==='static_'?'static':'dynamic')),old=base.sampling.stats[branch.branch];const percent=prefix==='static_'?p.static_rate:p.dynamic_rate;const normal=[];const keep=[];for(let i=0;i<rows.length;i++)(role[i]===1?keep:normal).push(i);normal.sort((a,b)=>hash(rows[a])-hash(rows[b])||rows[a]-rows[b]);keep.push(...normal.slice(0,Math.ceil(old.ordinary_total*percent/100)));keep.sort((a,b)=>a-b);for(const key of Object.keys(base)){if(!key.startsWith(prefix)||key===prefix+'cycle')continue;const v=base[key];if(!ArrayBuffer.isView(v))continue;const stride=v.length/rows.length;if(!Number.isInteger(stride))continue;const dst=new v.constructor(keep.length*stride);for(let i=0;i<keep.length;i++)dst.set(v.subarray(keep[i]*stride,(keep[i]+1)*stride),i*stride);out[key]=dst;}branch.displayed=keep.length;}
  out.sampling={...base.sampling,...p,stats:Object.fromEntries(out.branches.map(b=>[b.branch,{...base.sampling.stats[b.branch],sampled:b.displayed,base_sampled:Math.ceil(base.sampling.stats[b.branch].ordinary_total*(b.branch==='static'?p.static_rate:p.dynamic_rate)/100)}]))};return out;
 }
 async function load(s,loader,unpack){sync();const epoch=++cloudEpoch,p=valid(s.preview_sampling),arm=s.arm,raw=s.raw;let result;
  try{if(p.static_rate<=2&&p.dynamic_rate<=2&&p.vehicle==='none'){const base=await loader(`${arm}/cloud`, `data/${arm}/cloud.js`);if(!base.sampling)throw Error('缺少新版2%采样数据，请更新数据包');result=p.static_rate===2&&p.dynamic_rate===2?base:reduced(base,p);}
   else{const key=JSON.stringify([arm,p,p.vehicle==='none'?null:raw]);if(cache.has(key))result=cache.get(key);else{
    if(window.GSOfflineGeometry?.available(arm)){const base=await loader(`${arm}/cloud`,`data/${arm}/cloud.js`);result=await GSOfflineGeometry.make(arm,raw,p,base,unpack,message,()=>epoch!==cloudEpoch);}
    else throw Error('当前实验缺少离线几何分块；请运行 export_offline.ps1 更新此目录');
    cache.set(key,result);while(cache.size>2)cache.delete(cache.keys().next().value);
   }}
   if(epoch===cloudEpoch){lastGood={arm,cloud:result,p:{...p}};message(`已应用${result.sampling?.offline?"（离线）":""}：静态 ${p.static_rate}% · 动态 ${p.dynamic_rate}%`+(p.vehicle==='none'?'':` · a${p.vehicle} 单车目标 ${p.vehicle_rate}%（不降低基础采样率）`));}return result;
  }catch(e){if(epoch!==cloudEpoch)throw e;message(e.message+'；仍显示上次成功采样。',true);if(lastGood?.arm===arm){s.preview_sampling={...lastGood.p};sync();return lastGood.cloud;}const base=await loader(`${arm}/cloud`,`data/${arm}/cloud.js`);s.preview_sampling={...defaults};sync();lastGood={arm,cloud:base,p:{...defaults}};return base;}
 }
 function begin(cloud){counts={ordinary:{pool:0,active:0,screen:0},road:{pool:0,active:0,screen:0},dynamic:{pool:0,active:0,screen:0}};for(const prefix of ['static_','dynamic_'])for(let i=0;i<cloud[prefix+'role'].length;i++){if(window.GSV1Ground?.skip(prefix,cloud[prefix+'row'][i]))continue;counts[['ordinary','road','dynamic'][cloud[prefix+'role'][i]]].pool++;}}
 function active(role){if(counts)counts[['ordinary','road','dynamic'][role]].active++;}
 function drawn(role){if(counts)counts[['ordinary','road','dynamic'][role]].screen++;}
 function alpha(opacity){return el('pointAlpha')?.value==='solid'?1:Math.max(0,Math.min(1,opacity));}
 function legend(mode,cloud,scene){vehicleLabels(scene);const bar='<span class="numeric-colorbar"></span>';
  const legends={role:'<span style="color:#80aff0">● 普通背景</span>　<span style="color:#5ed7b1">● 道路</span>　<span style="color:#ffb379">● 动态</span>',rgb:'DC颜色：仅0阶SH预览；完整方向相关SH外观见二维原生渲染。',opacity:bar+'有效 opacity：深蓝 0 → 蓝 0.25 → 青绿 0.50 → 黄 0.75 → 红 1.00。道路=基础opacity；其他分支另乘时间权重。',scale:bar+'最大轴 σ [m]：深蓝 0 → 蓝 0.414 → 青绿 1.000 → 黄 1.828 → 红 ≥3.000（对数映射）。'};
  legends.v1_origin='<span style="color:#5ed7b1">● 初始地面来源·本帧贴地</span>　<span style="color:#ffb05c">● 初始地面来源·已离地</span>　<span style="color:#78aaff">● 其他初始来源·本帧贴地</span>　<span style="color:#95a4b8">● 其他初始来源·本帧不贴地</span>（贴地仅几何筛选）';if(window.GSV1Ground?.enabled()){legends.rgb='DC颜色：每个训练后高斯的0阶SH基础RGB；杂色不是道路类别。紫空心为初始位置。看来源请选“V1 来源／离地”。';legends.role=legends.role.replace('● 道路','● V1地面候选');legends.opacity=bar+'有效 opacity：深蓝0 → 蓝0.25 → 青绿0.50 → 黄0.75 → 红1。V1候选与其余静态均为基础opacity×本帧时间权重。';}el('sceneLegend').innerHTML=legends[mode]+'<span>　<span style="color:#76e1ec">●</span> LiDAR实测深度回投</span>';
  if(el('threeSolid')?.checked)el('sceneLegend').innerHTML+='；三层道路按统一不透明显示，普通背景／动态沿用点透明度设置。';const source=cloud.sampling?.stats;const stat=cloud.branches.find(b=>b.branch==='static'),dynamic=cloud.branches.find(b=>b.branch==='dynamic');
  const totals={ordinary:source?.static?.ordinary_total??stat.total-(source?.static?.road_total||0),road:source?.static?.road_total??counts.road.pool,dynamic:dynamic.total};
  if(window.GSV1Ground?.enabled())totals.ordinary=stat.total-GSV1Ground.total();el('cloudInfo').innerHTML=Object.entries(counts).filter(([k])=>k!=='road'||!window.GSV1Ground?.enabled()).map(([k,c])=>{const name={ordinary:'普通背景',road:'道路',dynamic:'动态'}[k];const rate=totals[k]?100*c.pool/totals[k]:0;return `<div><strong>${name}</strong>：抽样 ${c.pool.toLocaleString()} / ${totals[k].toLocaleString()}（${rate.toFixed(2)}%）；本帧时间有效 ${c.active.toLocaleString()}；画面内已绘制 ${c.screen.toLocaleString()}</div>`;}).join('')+'<div>'+ (el('pointAlpha').value==='solid'?'不透明中心模式：只看分布与颜色数值，点亮度不表示真实透明度。':'真实有效opacity模式：无最低透明度抬升，颜色会与背景混合变暗。')+' 时间权重≤0.05的中心不显示。三层道路点大小可统一；标记不代表高斯真实覆盖或像素贡献。</div>';
  if(window.GSV1Ground?.enabled())el('cloudInfo').innerHTML+='<div>V1地面候选全量载入，按上方显示率和筛选绘制；拖动暂时减点，松开恢复。此处普通背景排除固定候选全集，二维分组与全量统计保持不变。</div>';const st=cloud.sampling?.stats;if(st&&cloud.sampling.vehicle!=='none'){el('cloudInfo').innerHTML+='<div>所选车辆区域：'+['static','dynamic'].map(k=>`${k==='static'?'普通背景':'动态'} ${st[k].vehicle_sampled.toLocaleString()} / ${st[k].vehicle_candidates.toLocaleString()}（${st[k].vehicle_candidates?(100*st[k].vehicle_sampled/st[k].vehicle_candidates).toFixed(2):'0'}%）`).join('；')+'</div>';}
 }
 return {bind,load,begin,active,drawn,alpha,legend,valid,defaults,vehicles};
})();
