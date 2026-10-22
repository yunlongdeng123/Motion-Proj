/* V1静态候选诊断；缓存只读数据，保留真实来源及全量统计。 */
window.GSV1Ground=(()=>{
 'use strict';
 const $=id=>document.getElementById(id),xyz=(a,i)=>[a[i*3],a[i*3+1],a[i*3+2]],f=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
 const ORIGIN=['#5ed7b1','#ffb05c','#78aaff','#95a4b8'],BUDGET=12000;
 let api,bound=false,h=null,surfaces=null,rowMask=null,initialRows=null,selected=null,neighbors=[],frame=null,samples=null,colors=new Map(),seenV1=false;
 let screen=0,shown=0,preview=false,trainedPickN=0,initialPickN=0,trainedXY=null,initialXY=null,trainedIndices=null,initialIndices=null;
 const counters={historyBuilds:0,frameBuilds:0,sampleBuilds:0};
 function bind(a){api=a;
  const label=document.createElement('label');label.id='v1GroundRateControl';label.innerHTML='V1 初始／训练后显示率 <input id="v1GroundRate" type="number" min="0" max="100" step="1" value="10"> %（全量统计）';$('compareRoadLayers').before(label);
  const filter=document.createElement('label');filter.id='v1GroundFilterControl';filter.innerHTML='V1 训练后筛选 <select id="v1GroundFilter"><option value="all">全部候选</option><option value="near">本帧贴地（±0.30m）</option><option value="root">初始地面全部后代</option><option value="away">初始地面已离地后代</option></select>';$('compareRoadLayers').before(filter);
  const option=document.createElement('option');option.value='v1_origin';option.textContent='V1 来源／离地';$('cloudColor').append(option);
  $('v1GroundRate').oninput=()=>api.draw();$('v1GroundFilter').onchange=()=>api.draw();
 }
 function rate(){return Math.max(0,Math.min(100,Number($('v1GroundRate').value)||0));}
 function sampled(row,percent){let x=row>>>0;x=(x^(x>>>16))>>>0;x=Math.imul(x,0x7feb352d)>>>0;x=(x^(x>>>15))>>>0;x=Math.imul(x,0x846ca68b)>>>0;x=(x^(x>>>16))>>>0;return x/4294967296<percent/100;}
 function enabled(){return bound&&api?.state().arm==='gn11_v1'&&h!==null;}
 function accept(v){const incoming=v.v1||null;bound=!!incoming;
  if(incoming&&api.cloud().branches.find(b=>b.branch==='static').checkpoint_sha256!==incoming.metadata.checkpoint_sha256)throw Error('V1地面候选与当前检查点不匹配');
  if(incoming&&incoming!==h){
   const binding=api.cloud().branches.find(b=>b.branch==='static');if(binding.checkpoint_sha256!==incoming.metadata.checkpoint_sha256)throw Error('V1地面候选与当前检查点不匹配');
   h=incoming;rowMask=new Uint8Array(h.metadata.trained_static_count);for(let i=0;i<h.row.length;i++)rowMask[h.row[i]]=1;
   const ids=new Int32Array(h.metadata.initial_ground_count);let n=0;for(let i=0;i<h.init_ground.length;i++)if(h.init_ground[i])ids[n++]=i;initialRows=ids.subarray(0,n);
   frame=null;samples=null;colors.clear();counters.historyBuilds++;
  }
  surfaces=v.surfaces;selected=null;neighbors=[];trainedPickN=initialPickN=0;
  const v1=api.state().arm==='gn11_v1';$('roads3d').parentNode.lastChild.textContent=v1?'V1 地面静态候选':'训练后道路';$('initialRoad').parentNode.lastChild.textContent=v1?'V1 初始地面高斯':'初始化道路高斯';$('ground49').parentNode.lastChild.textContent=v1?'V1 初始化地面输入（49帧）':'初始化使用的49帧地面LiDAR';
  $('emptyRoad').disabled=v1;if(v1)$('emptyRoad').checked=false;$('initialRoad').disabled=false;$('compareRoadLayers').disabled=false;$('v1GroundRateControl').hidden=!v1;$('v1GroundFilterControl').hidden=!v1;
  for(const option of Array.from($('channel').options||[]))if(option.value.startsWith('v1_'))option.disabled=!v1;
  for(const option of Array.from($('cloudColor').options||[]))if(option.value==='v1_origin')option.disabled=!v1;
  if(v1&&!seenV1){$('cloudColor').value='v1_origin';seenV1=true;}else if(!v1&&$('cloudColor').value==='v1_origin')$('cloudColor').value='role';
  if(!v1&&api.state().channel?.startsWith('v1_')){api.state().channel='deleted_five';$('channel').value='deleted_five';}
 }
 function skip(prefix,row){return enabled()&&prefix==='static_'&&rowMask[row]===1;}
 function mul4(a,b){return a.map(r=>b[0].map((_,j)=>r.reduce((t,v,k)=>t+v*b[k][j],0)));}
 function toAnchor(){const a=h.metadata.coordinate_anchor,T=a.slice(0,3).map(r=>r[3]);return [...[0,1,2].map(i=>[a[0][i],a[1][i],a[2][i],-a.slice(0,3).reduce((t,r,k)=>t+r[i]*T[k],0)]),[0,0,0,1]];}
 function ensureFrame(){const raw=api.state().raw;if(frame?.raw===raw)return frame;
  const timestamp=api.scene().cameras[raw-89].timestamp,n=h.row.length,omega=2*Math.PI/h.metadata.cycle,A=mul4(toAnchor(),api.scene().model_to_world),plane=h.metadata.plane_anchor;
  // 双精度保留原位置/时间权重公式；最多缓存一帧，旋转无需重新算128万次exp/sin。
  const p=frame?.p||new Float64Array(n*3),opacity=frame?.opacity||new Float64Array(n),active=frame?.allActive||new Int32Array(n),category=frame?.category||new Uint8Array(n);let count=0;const groups=[0,0,0,0];
  for(let i=0;i<n;i++){const delta=timestamp-h.t[i],m=Math.exp(-.5*(delta/h.sigma_t[i])**2);opacity[i]=h.opacity[i]*m;if(m<=.05){category[i]=255;continue;}
   const v=Math.sin(delta*omega)/omega,j=i*3,x=h.xyz[j]+h.velocity[j]*v,y=h.xyz[j+1]+h.velocity[j+1]*v,z=h.xyz[j+2]+h.velocity[j+2]*v;p[j]=x;p[j+1]=y;p[j+2]=z;active[count++]=i;
   const ax=A[0][0]*x+A[0][1]*y+A[0][2]*z+A[0][3],ay=A[1][0]*x+A[1][1]*y+A[1][2]*z+A[1][3],az=A[2][0]*x+A[2][1]*y+A[2][2]*z+A[2][3],height=plane[0]*ax+plane[1]*ay+plane[2]*az+plane[3];
   const near=Math.abs(height)<=h.metadata.current_ground_band_m&&ax>=3&&ax<=90&&Math.abs(ay)<=8,c=h.root_ground[i]?(near?0:1):(near?2:3);category[i]=c;groups[c]++;
  }
  frame={raw,p,opacity,allActive:active,active:active.subarray(0,count),category,groups};colors.clear();counters.frameBuilds++;return frame;
 }
 function ensureSamples(){const percent=rate();if(samples?.rate===percent)return samples;
  let n=0;const rows=new Int32Array(h.row.length);for(let i=0;i<h.row.length;i++)if(sampled(h.row[i],percent))rows[n++]=i;
  let k=0;const roots=new Int32Array(initialRows.length);for(const root of initialRows)if(sampled(root,percent))roots[k++]=root;
  samples={rate:percent,trained:rows.slice(0,n),initial:roots.slice(0,k)};trainedXY=new Float32Array(n*2);trainedIndices=new Int32Array(n);initialXY=new Float32Array(k*2);initialIndices=new Int32Array(k);colors.clear();counters.sampleBuilds++;return samples;
 }
 function matches(c){const mode=$('v1GroundFilter').value||'all';return mode==='near'?c===0||c===2:mode==='root'?c===0||c===1:mode==='away'?c===1:true;}
 function timeWeight(i){const delta=api.scene().cameras[api.state().raw-89].timestamp-h.t[i];return Math.exp(-.5*(delta/h.sigma_t[i])**2);}
 function position(i){const g=ensureFrame();if(g.category[i]!==255)return {p:xyz(g.p,i),m:h.opacity[i]>0?g.opacity[i]/h.opacity[i]:timeWeight(i),opacity:g.opacity[i]};const timestamp=api.scene().cameras[api.state().raw-89].timestamp,delta=timestamp-h.t[i],omega=2*Math.PI/h.metadata.cycle,v=Math.sin(delta*omega)/omega,m=timeWeight(i);return {p:xyz(h.xyz,i).map((x,k)=>x+h.velocity[3*i+k]*v),m,opacity:h.opacity[i]*m};}
 function projector(){return api.projector?api.projector():((x,y,z)=>api.project([x,y,z]));}
 function styles(mode){if(colors.has(mode))return colors.get(mode);const ids=ensureSamples().trained,out=new Array(ids.length),g=ensureFrame(),scale=api.scene().meter_to_model;
  for(let k=0;k<ids.length;k++){const i=ids[k],j=i*3;let color;if(mode==='v1_origin')color=ORIGIN[g.category[i]]||ORIGIN[3];else if(mode==='opacity')color='rgb('+api.heat(g.opacity[i]).join(',')+')';else if(mode==='scale')color='rgb('+api.heat(Math.log1p(Math.max(h.sigma[j],h.sigma[j+1],h.sigma[j+2])/scale)/Math.log(4)).join(',')+')';else if(mode==='rgb')color=`rgb(${Math.min(255,Math.max(0,Math.round(h.color[j]*255)))},${Math.min(255,Math.max(0,Math.round(h.color[j+1]*255)))},${Math.min(255,Math.max(0,Math.round(h.color[j+2]*255)))})`;else color=ORIGIN[0];out[k]=color;}
  if(colors.size>=3)colors.delete(colors.keys().next().value);colors.set(mode,out);return out;
 }
 function draw(ctx,picks){trainedPickN=0;screen=shown=0;if(!enabled())return;
  const g=ensureFrame(),s=ensureSamples(),project=projector(),mode=$('cloudColor').value,size=GSInspect.size(),half=size/2,solid=$('threeSolid').checked||$('pointAlpha').value==='solid';preview=!!api.interacting?.();if(!$('roads3d').checked)return;
  const palette=styles(mode),stride=preview?Math.max(1,Math.ceil(s.trained.length/BUDGET)):1;
  for(let k=0;k<s.trained.length;k+=stride){const i=s.trained[k];if(g.category[i]===255||!matches(g.category[i]))continue;shown++;const j=i*3,p=project(g.p[j],g.p[j+1],g.p[j+2]);if(!p||p[0]<-5||p[1]<-5||p[0]>ctx.canvas.width+5||p[1]>ctx.canvas.height+5)continue;screen++;
   ctx.fillStyle=palette[k];ctx.globalAlpha=solid?1:g.opacity[i];ctx.fillRect(p[0]-half,p[1]-half,size,size);trainedXY[trainedPickN*2]=p[0];trainedXY[trainedPickN*2+1]=p[1];trainedIndices[trainedPickN++]=i;
  }ctx.globalAlpha=1;
 }
 function under(ctx){if(!enabled())return;ctx.globalAlpha=1;if(!$('ground49').checked)return;
  const project=projector(),size=GSInspect.size(),half=size/2,a=h.ground_xyz,step=api.interacting?.()?Math.max(1,Math.ceil(a.length/3/BUDGET)):1;ctx.fillStyle='#62d4ef';
  for(let i=0;i<a.length/3;i+=step){const j=i*3,p=project(a[j],a[j+1],a[j+2]);if(p&&p[0]>=-size&&p[1]>=-size&&p[0]<=ctx.canvas.width+size&&p[1]<=ctx.canvas.height+size)ctx.fillRect(p[0]-half,p[1]-half,size,size);}
 }
 function over(ctx,picks){initialPickN=0;if(!enabled())return;const s=ensureSamples(),g=ensureFrame(),project=projector(),size=GSInspect.size(),half=size/2;
  if($('initialRoad').checked){const step=preview?Math.max(1,Math.ceil(s.initial.length/BUDGET)):1;ctx.strokeStyle='#c596ff';ctx.lineWidth=.65;ctx.globalAlpha=$('threeSolid').checked?1:GSCloud.alpha(.01);
   for(let k=0;k<s.initial.length;k+=step){const root=s.initial[k],j=root*3,p=project(h.init_xyz[j],h.init_xyz[j+1],h.init_xyz[j+2]);if(p&&p[0]>=-size&&p[1]>=-size&&p[0]<=ctx.canvas.width+size&&p[1]<=ctx.canvas.height+size){ctx.strokeRect(p[0]-half,p[1]-half,size,size);initialXY[initialPickN*2]=p[0];initialXY[initialPickN*2+1]=p[1];initialIndices[initialPickN++]=root;}}ctx.globalAlpha=1;
  }
  if(selected){const root=selected.root,p0=xyz(h.init_xyz,root);ctx.fillStyle='#ffffff';for(const j of neighbors){const p=project(h.ground_xyz[j*3],h.ground_xyz[j*3+1],h.ground_xyz[j*3+2]);if(p)ctx.fillRect(p[0]-2,p[1]-2,4,4);}if(selected.index!==null)api.line(ctx,p0,position(selected.index).p,'#f1dc8c',2);const sigma=h.init_sigma[root];for(let axis=0;axis<3;axis++){let old;for(let j=0;j<=40;j++){const p=[...p0];p[(axis+1)%3]+=Math.cos(j*Math.PI/20)*sigma;p[(axis+2)%3]+=Math.sin(j*Math.PI/20)*sigma;if(old)api.line(ctx,old,p,'#c596ff',1.5);old=p;}}}
  const group=g.groups,fmt=v=>v.toLocaleString(),desc=preview?'拖动预览已临时减点；松开恢复所设显示率。':'',filter=$('v1GroundFilter').value||'all',name={all:'全部候选',near:'本帧贴地',root:'初始地面全部后代',away:'初始地面已离地后代'}[filter];
  const note=`V1 候选全量 ${fmt(h.row.length)}；本帧有效 ${fmt(g.active.length)}。初始地面后代：贴地 ${fmt(group[0])}、离地 ${fmt(group[1])}；其他初始来源：本帧贴地 ${fmt(group[2])}、本帧不贴地 ${fmt(group[3])}。贴地按实测平面±0.30m与原走廊筛选，非道路语义真值。\n青点=初始地面输入 ${fmt(initialRows.length)}；紫空心=初始高斯。训练后显示率 ${rate()}%，筛选“${name}”，${preview?'预览':'抽到'} ${fmt(shown)}个，画面绘制 ${fmt(screen)}个。${desc}二维alpha／深度与框选统计始终全量，不随筛选或预览减点改变。V1无道路网格。`;
  if($('inspectionLegend').textContent!==note)$('inspectionLegend').textContent=note;
 }
 function pick(x,y,best=64){if(!enabled())return null;let found=null;for(const [xy,ids,n,kind] of [[trainedXY,trainedIndices,trainedPickN,'v1trained'],[initialXY,initialIndices,initialPickN,'v1initial']])for(let k=0;k<n;k++){const d=(xy[k*2]-x)**2+(xy[k*2+1]-y)**2;if(d<best){best=d;found={distance:d,rec:kind==='v1trained'?{kind,index:ids[k]}:{kind,root:ids[k]}};}}return found;}
 function select(rec){if(!enabled()||!['v1trained','v1initial'].includes(rec.kind))return false;
  const root=rec.kind==='v1trained'?h.root[rec.index]:rec.root;let i=rec.kind==='v1trained'?rec.index:null;
  if(i===null)for(let j=0;j<h.root.length;j++)if(h.root[j]===root&&(i===null||position(j).opacity>position(i).opacity))i=j;
  selected={root,index:i};neighbors=[];const radius=h.metadata.support_radius_m;for(let j=0;j<h.ground_xy.length/2;j++)if(Math.hypot(h.ground_xy[j*2]-h.init_xy[root*2],h.ground_xy[j*2+1]-h.init_xy[root*2+1])<=radius)neighbors.push(j);
  const p0=xyz(h.init_xyz,root),w0=api.world(p0),count=h.descendants[root];let text=`V1 初始来源 ${root}（原始PLY行 ${h.input_row[root]}） · ${h.init_ground[root]?'初始地面点':'初始非地面点，后代进入近地面区域'}；初始世界[m] ${w0.map(v=>f(v)).join(', ')}；0.8m水平半径内实际地面输入 ${h.support_count[root]}点；30K检查点保留同源后代 ${count}个（含不在地面候选组的后代）。`;
  if(i!==null){const r=position(i),sigma=xyz(h.sigma,i).map(v=>v/api.scene().meter_to_model),offset=Math.hypot(...r.p.map((v,k)=>v-p0[k]))/api.scene().meter_to_model,id=`gn11_v1:${h.metadata.checkpoint_sha256.slice(0,12)}:static:${h.row[i]}`;
   api.state().selected={id,branch:'ordinary',road_id:-1,model_xyz:r.p,world_m:api.world(r.p),quaternion:Array.from(h.q.slice(i*4,i*4+4)),sigma_m:sigma,opacity:h.opacity[i]};
   text+=` 当前后代行 ${h.row[i]}，当前世界[m] ${api.world(r.p).map(v=>f(v)).join(', ')}；距初始来源 ${f(offset)}m；σ[m] ${f(h.init_sigma[root]/api.scene().meter_to_model)}（各向同性，3NN公式复原） → ${sigma.map(v=>f(v)).join('/')}；基础opacity 0.0100 → ${f(h.opacity[i],4)}；本帧有效opacity ${f(r.opacity,4)}，时间权重 ${f(r.m,4)}。`;
  }else{api.state().selected=null;text+=' 当前无存活后代；不伪造当前位置。';}
  text+=' 对应来自检查点保存谱系；clone/split可一对多，“距初始来源”不是同一个高斯的连续移动轨迹。紫线初始1σ，黄线当前1σ，白点附近输入。';$('roadHistoryInfo').textContent=text;$('gaussianInfo').textContent=text;api.draw();return true;
 }
 function values(){if(!enabled())return null;const i=api.state().y*480+api.state().x;return [{key:'road',name:'V1地面候选深度',z:surfaces.ground_alpha[i]>1e-8?surfaces.ground_depth_m[i]:null,source:'固定地面候选独立CPU参考期望深度'}, {key:'ordinary',name:'V1其余静态深度',z:surfaces.ordinary_alpha[i]>1e-8?surfaces.ordinary_depth_m[i]:null,source:'排除地面候选后静态独立CPU参考期望深度'}];}
 function region(box){if(!enabled())return null;const g=ensureFrame(),project=projector();let init=0,points=0,n=0,opacity=0,sigma=0,offset=0;const groups=[0,0,0,0];
  for(const root of initialRows){const j=root*3;if(GSInspectionMath.within(project(h.init_xyz[j],h.init_xyz[j+1],h.init_xyz[j+2]),box))init++;}
  for(const i of g.active){const j=i*3;if(GSInspectionMath.within(project(g.p[j],g.p[j+1],g.p[j+2]),box)){n++;groups[g.category[i]]++;opacity+=g.opacity[i];sigma+=Math.max(h.sigma[j],h.sigma[j+1],h.sigma[j+2])/api.scene().meter_to_model;const k=h.root[i]*3;offset+=Math.hypot(g.p[j]-h.init_xyz[k],g.p[j+1]-h.init_xyz[k+1],g.p[j+2]-h.init_xyz[k+2])/api.scene().meter_to_model;}}
  for(let j=0;j<h.ground_xyz.length/3;j++)if(GSInspectionMath.within(project(h.ground_xyz[j*3],h.ground_xyz[j*3+1],h.ground_xyz[j*3+2]),box))points++;
  return `V1屏幕投影框选（非三维体积）：实际初始化地面输入 ${points}点；初始地面高斯 ${init}个；本帧时间有效训练后地面候选 ${n}个；初始地面来源贴地/离地 ${groups[0]}/${groups[1]}，其他来源贴地/不贴地 ${groups[2]}/${groups[3]}；平均有效opacity ${n?f(opacity/n,4):'—'}；平均最大σ ${n?f(sigma/n):'—'}m；平均距初始来源 ${n?f(offset/n):'—'}m（含clone/split，非逐点训练位移）。候选全量统计，与显示筛选、减点、开关、普通背景采样率无关；V1无道路网格。`;
 }
 function heat(canvas,channel,mark){if(!enabled()||!channel.startsWith('v1_'))return false;const key={v1_ground_alpha:'ground_alpha',v1_ground_depth:'ground_depth_m',v1_other_alpha:'ordinary_alpha',v1_other_depth:'ordinary_depth_m'}[channel];if(!key)return false;const ctx=canvas.getContext('2d'),im=ctx.createImageData(480,320),depth=key.endsWith('_m');for(let i=0;i<480*320;i++){const v=surfaces[key][i],c=depth&&v<=0?[5,9,16]:api.heat(depth?Math.log1p(Math.min(v,150))/Math.log(151):v);im.data.set([...c,255],i*4);}ctx.putImageData(im,0,0);if(mark)mark(ctx);return true;}
 function pixelInfo(){if(!enabled())return '';const i=api.state().y*480+api.state().x;return `V1完整候选CPU参考：地面候选独立α ${f(surfaces.ground_alpha[i],4)}，z ${surfaces.ground_alpha[i]>1e-8?f(surfaces.ground_depth_m[i]):'—'}m；其余静态独立α ${f(surfaces.ordinary_alpha[i],4)}，z ${surfaces.ordinary_alpha[i]>1e-8?f(surfaces.ordinary_depth_m[i]):'—'}m。两者独立渲染，α不能直接相加；三维筛选不改变此全量分组。`;}
 return {bind,accept,enabled,skip,draw,under,over,pick,select,values,region,heat,pixelInfo,total:()=>h?.row.length||0,debug:()=>({h,surfaces,selected,active:frame?.active||[],screen,stats:frame?.groups,counters:{...counters},preview,trainedPickN,initialPickN,trainedXY,trainedIndices})};
})();
