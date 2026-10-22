/* V1地面静态候选：仅显示分组，真实初始来源谱系，保留原生模型ID。 */
window.GSV1Ground=(()=>{
 'use strict';
 const $=id=>document.getElementById(id),xyz=(a,i)=>[a[i*3],a[i*3+1],a[i*3+2]],f=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
 let api,h=null,surfaces=null,rows=new Set(),rootIndex=new Map(),selected=null,active=[],screen=0,shown=0;
 function bind(a){api=a;const label=document.createElement('label');label.id='v1GroundRateControl';label.innerHTML='V1 初始／训练后显示率 <input id="v1GroundRate" type="number" min="0" max="100" step="1" value="10"> %（全量统计）';$('compareRoadLayers').before(label);$('v1GroundRate').oninput=()=>api.draw();}
 function rate(){return Math.max(0,Math.min(100,Number($('v1GroundRate').value)||0));}
 function sampled(row,percent){let x=row>>>0;x=(x^(x>>>16))>>>0;x=Math.imul(x,0x7feb352d)>>>0;x=(x^(x>>>15))>>>0;x=Math.imul(x,0x846ca68b)>>>0;x=(x^(x>>>16))>>>0;return x/4294967296<percent/100;}
 function enabled(){return api?.state().arm==='gn11_v1'&&h!==null;}
 function accept(v){h=v.v1||null;surfaces=v.surfaces;rows=new Set(h?Array.from(h.row):[]);rootIndex=new Map();selected=null;active=[];
  if(h){const binding=api.cloud().branches.find(b=>b.branch==='static');if(binding.checkpoint_sha256!==h.metadata.checkpoint_sha256)throw Error('V1地面候选与当前检查点不匹配');for(let i=0;i<h.root.length;i++){if(!rootIndex.has(h.root[i]))rootIndex.set(h.root[i],[]);rootIndex.get(h.root[i]).push(i);}}
  const v1=api.state().arm==='gn11_v1';$('roads3d').parentNode.lastChild.textContent=v1?'V1 地面静态候选':'训练后道路';$('initialRoad').parentNode.lastChild.textContent=v1?'V1 初始地面高斯':'初始化道路高斯';$('ground49').parentNode.lastChild.textContent=v1?'V1 初始化地面输入（49帧）':'初始化使用的49帧地面LiDAR';
  $('emptyRoad').disabled=v1;if(v1)$('emptyRoad').checked=false;$('initialRoad').disabled=false;$('compareRoadLayers').disabled=false;$('v1GroundRateControl').hidden=!v1;
  for(const option of Array.from($('channel').options||[]))if(option.value.startsWith('v1_'))option.disabled=!v1;
  if(!v1&&api.state().channel?.startsWith('v1_')){api.state().channel='deleted_five';$('channel').value='deleted_five';}
 }
 function skip(prefix,row){return enabled()&&prefix==='static_'&&rows.has(row);}
 function position(i){const timestamp=api.scene().cameras[api.state().raw-89].timestamp,delta=timestamp-h.t[i],omega=2*Math.PI/h.metadata.cycle,m=Math.exp(-.5*(delta/h.sigma_t[i])**2),v=Math.sin(delta*omega)/omega;return {p:xyz(h.xyz,i).map((x,k)=>x+h.velocity[3*i+k]*v),m,opacity:h.opacity[i]*m};}
 function draw(ctx,picks){if(!enabled())return;active=[];screen=0;shown=0;
  const timestamp=api.scene().cameras[api.state().raw-89].timestamp,percent=rate(),visible=$('roads3d').checked;
  for(let i=0;i<h.row.length;i++){const delta=timestamp-h.t[i],m=Math.exp(-.5*(delta/h.sigma_t[i])**2);if(m<=.05)continue;active.push(i);if(!visible||!sampled(h.row[i],percent))continue;shown++;const r=position(i),p=api.project(r.p);if(!p||p[0]<-5||p[1]<-5||p[0]>ctx.canvas.width+5||p[1]>ctx.canvas.height+5)continue;screen++;
   const mode=$('cloudColor').value,sigma=xyz(h.sigma,i);let color='#5ed7b1';if(mode==='opacity')color='rgb('+api.heat(r.opacity).join(',')+')';if(mode==='scale')color='rgb('+api.heat(Math.log1p(Math.max(...sigma)/api.scene().meter_to_model)/Math.log(4)).join(',')+')';if(mode==='rgb')color='rgb('+xyz(h.color,i).map(v=>Math.min(255,Math.max(0,Math.round(v*255)))).join(',')+')';
   ctx.fillStyle=color;ctx.globalAlpha=GSInspect.alpha(r.opacity);ctx.beginPath();ctx.arc(p[0],p[1],GSInspect.size()/2,0,2*Math.PI);ctx.fill();picks.push({screen:p,rec:{kind:'v1trained',index:i}});
  }ctx.globalAlpha=1;
 }
 function under(ctx){if(!enabled())return;ctx.globalAlpha=1;if($('ground49').checked){ctx.fillStyle='#62d4ef';for(let i=0;i<h.ground_xyz.length/3;i++){const p=api.project(xyz(h.ground_xyz,i));if(p)ctx.fillRect(p[0]-GSInspect.size()/2,p[1]-GSInspect.size()/2,GSInspect.size(),GSInspect.size());}}}
 function over(ctx,picks){if(!enabled())return;
  if($('initialRoad').checked){const percent=rate();ctx.strokeStyle='#c596ff';ctx.lineWidth=.65;ctx.globalAlpha=$('threeSolid').checked?1:GSCloud.alpha(.01);for(let i=0;i<h.init_ground.length;i++)if(h.init_ground[i]&&sampled(i,percent)){const p=api.project(xyz(h.init_xyz,i));if(p){ctx.strokeRect(p[0]-GSInspect.size()/2,p[1]-GSInspect.size()/2,GSInspect.size(),GSInspect.size());picks.push({screen:p,rec:{kind:'v1initial',root:i}});}}ctx.globalAlpha=1;}
  if(selected){const root=selected.root,p0=xyz(h.init_xyz,root),radius=h.metadata.support_radius_m;ctx.fillStyle='#ffffff';for(let j=0;j<h.ground_xy.length/2;j++)if(Math.hypot(h.ground_xy[j*2]-h.init_xy[root*2],h.ground_xy[j*2+1]-h.init_xy[root*2+1])<=radius){const p=api.project(xyz(h.ground_xyz,j));if(p)ctx.fillRect(p[0]-2,p[1]-2,4,4);}if(selected.index!==null)api.line(ctx,p0,position(selected.index).p,'#f1dc8c',2);const sigma=h.init_sigma[root];for(let axis=0;axis<3;axis++){let old;for(let j=0;j<=40;j++){const p=[...p0];p[(axis+1)%3]+=Math.cos(j*Math.PI/20)*sigma;p[(axis+2)%3]+=Math.sin(j*Math.PI/20)*sigma;if(old)api.line(ctx,old,p,'#c596ff',1.5);old=p;}}}
  $('inspectionLegend').textContent=`V1：青色为实际初始化静态输入中的地面 ${h.metadata.initial_ground_count.toLocaleString()}点；紫空心为这些位置创建的初始高斯；训练后候选全量 ${h.row.length.toLocaleString()}个，本帧时间有效 ${active.length.toLocaleString()}个；初始／训练后统一按 ${rate()}% 确定性显示，训练后抽到 ${shown.toLocaleString()}个，画面绘制 ${screen.toLocaleString()}个。全量已载入，二维alpha／深度及框选统计不采样。此为静态诊断分组，非V3道路分支：初始地面全部后代 ∪ 49帧任一中心在实测平面±0.30m内的静态点。V1无道路网格，灰框不适用。`;
 }
 function select(rec){if(!enabled()||!['v1trained','v1initial'].includes(rec.kind))return false;
  const root=rec.kind==='v1trained'?h.root[rec.index]:rec.root,family=rootIndex.get(root)||[];let i=rec.kind==='v1trained'?rec.index:null;
  if(i===null&&family.length)i=family.reduce((best,j)=>best===null||position(j).opacity>position(best).opacity?j:best,null);
  selected={root,index:i};const p0=xyz(h.init_xyz,root),w0=api.world(p0),count=h.descendants[root];
  let text=`V1 初始来源 ${root}（原始PLY行 ${h.input_row[root]}） · ${h.init_ground[root]?'初始地面点':'初始非地面点，后代进入近地面区域'}；初始世界[m] ${w0.map(v=>f(v)).join(', ')}；0.8m水平半径内实际地面输入 ${h.support_count[root]}点；30K检查点保留同源后代 ${count}个（含不在地面候选组的后代）。`;
  if(i!==null){const r=position(i),sigma=xyz(h.sigma,i).map(v=>v/api.scene().meter_to_model),offset=Math.hypot(...r.p.map((v,k)=>v-p0[k]))/api.scene().meter_to_model,id=`gn11_v1:${h.metadata.checkpoint_sha256.slice(0,12)}:static:${h.row[i]}`;
   api.state().selected={id,branch:'ordinary',road_id:-1,model_xyz:r.p,world_m:api.world(r.p),quaternion:Array.from(h.q.slice(i*4,i*4+4)),sigma_m:sigma,opacity:h.opacity[i]};
   text+=` 当前后代行 ${h.row[i]}，当前世界[m] ${api.world(r.p).map(v=>f(v)).join(', ')}；距初始来源 ${f(offset)}m；σ[m] ${f(h.init_sigma[root]/api.scene().meter_to_model)}（各向同性，3NN公式复原） → ${sigma.map(v=>f(v)).join('/')}；基础opacity 0.0100 → ${f(h.opacity[i],4)}；本帧有效opacity ${f(r.opacity,4)}，时间权重 ${f(r.m,4)}。`;
  }else{api.state().selected=null;text+=' 当前无存活后代；不伪造当前位置。';}
  text+=' 对应来自检查点保存谱系；clone/split可一对多，“距初始来源”不是同一个高斯的连续移动轨迹。紫线初始1σ，黄线当前1σ，白点附近输入。';$('roadHistoryInfo').textContent=text;$('gaussianInfo').textContent=text;api.draw();return true;
 }
 function values(){if(!enabled())return null;const i=api.state().y*480+api.state().x;return [{key:'road',name:'V1地面候选深度',z:surfaces.ground_alpha[i]>1e-8?surfaces.ground_depth_m[i]:null,source:'固定地面候选独立CPU参考期望深度'}, {key:'ordinary',name:'V1其余静态深度',z:surfaces.ordinary_alpha[i]>1e-8?surfaces.ordinary_depth_m[i]:null,source:'排除地面候选后静态独立CPU参考期望深度'}];}
 function region(box){if(!enabled())return null;let init=0,points=0,n=0,opacity=0,sigma=0,offset=0;for(let i=0;i<h.init_ground.length;i++)if(h.init_ground[i]&&GSInspectionMath.within(api.project(xyz(h.init_xyz,i)),box))init++;
  for(const i of active){const r=position(i);if(GSInspectionMath.within(api.project(r.p),box)){n++;opacity+=r.opacity;sigma+=Math.max(...xyz(h.sigma,i))/api.scene().meter_to_model;offset+=Math.hypot(...r.p.map((v,k)=>v-h.init_xyz[h.root[i]*3+k]))/api.scene().meter_to_model;}}
  for(let j=0;j<h.ground_xyz.length/3;j++)if(GSInspectionMath.within(api.project(xyz(h.ground_xyz,j)),box))points++;
  return `V1屏幕投影框选（非三维体积）：实际初始化地面输入 ${points}点；初始地面高斯 ${init}个；本帧时间有效训练后地面候选 ${n}个；平均有效opacity ${n?f(opacity/n,4):'—'}；平均最大σ ${n?f(sigma/n):'—'}m；平均距初始来源 ${n?f(offset/n):'—'}m（含clone/split，非逐点训练位移）。候选全量统计，与显示开关、普通背景采样率无关；V1无道路网格。`;
 }
 function heat(canvas,channel,mark){if(!enabled()||!channel.startsWith('v1_'))return false;const key={v1_ground_alpha:'ground_alpha',v1_ground_depth:'ground_depth_m',v1_other_alpha:'ordinary_alpha',v1_other_depth:'ordinary_depth_m'}[channel];if(!key)return false;const ctx=canvas.getContext('2d'),im=ctx.createImageData(480,320),depth=key.endsWith('_m');for(let i=0;i<480*320;i++){const v=surfaces[key][i],c=depth&&v<=0?[5,9,16]:api.heat(depth?Math.log1p(Math.min(v,150))/Math.log(151):v);im.data.set([...c,255],i*4);}ctx.putImageData(im,0,0);if(mark)mark(ctx);return true;}
 function pixelInfo(){if(!enabled())return '';const i=api.state().y*480+api.state().x;return `V1完整候选CPU参考：地面候选独立α ${f(surfaces.ground_alpha[i],4)}，z ${surfaces.ground_alpha[i]>1e-8?f(surfaces.ground_depth_m[i]):'—'}m；其余静态独立α ${f(surfaces.ordinary_alpha[i],4)}，z ${surfaces.ordinary_alpha[i]>1e-8?f(surfaces.ordinary_depth_m[i]):'—'}m。两者独立渲染，α不能直接相加。`;}
 return {bind,accept,enabled,skip,draw,under,over,select,values,region,heat,pixelInfo,total:()=>h?.row.length||0,debug:()=>({h,surfaces,selected,active,screen})};
})();
