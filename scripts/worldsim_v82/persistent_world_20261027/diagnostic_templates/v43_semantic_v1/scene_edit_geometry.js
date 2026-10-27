// Saved camera, clock and rigid transforms used by the native edit renderer.
// Loaded after scene_viewer.js declarations and before init (combined at build).
function editState(raw=S.raw){
 const source=$('geometryState').value==='baseline'||['gt','sensor_overlay'].includes(S.channel);
 const key=source?'reconstruction_30000':S.group;
 return window.EDIT_GEOMETRY?.groups[key]?.[raw]||null;
}
function currentCamera(){return S.cameraReference==='original'?D.frames[S.raw].c2w:(editState()?.camera||D.frames[S.raw].c2w);}
function geometryInstances(){
 const st=editState();if(!st?.supported)return [];
 const list=st.static_visible?[{key:'static',cloud:'static',query_time:st.world_time,transform:null,label:'STATIC'}]:[];
 return [...list,...st.objects];
}
function editedPoint(instance,j){
 const cloud=S.clouds[instance.cloud],pt=cloudPoint(cloud.data,j,cloud.cycle,instance.query_time);
 if(instance.transform)pt.p=matp(instance.transform,pt.p);
 return pt;
}
function currentBoxes(){
 const st=editState();if(!st?.supported)return [];
 const source=$('geometryState').value==='baseline'||['gt','sensor_overlay'].includes(S.channel)||S.group==='reconstruction_30000'||S.group==='motion_no_edit';
 if(source)return IA?IA.frames[S.raw].vehicle_boxes:[{target:true,track:D.target_track,label:'TARGET',corners_local:boxPoints(D.frames[S.raw])}];
 const others=IA?IA.frames[S.raw].vehicle_boxes.filter(b=>!b.target):[];
 return [...others,...st.objects.map(o=>({target:true,track:o.key==='target'?D.target_track:o.key,label:o.label,corners_local:o.corners_local,imported:o.cloud.startsWith('foreign:')}))];
}
function draw3D(){
 const canvas=$('sceneCanvas'),x=canvas.getContext('2d'),st=editState();
 x.fillStyle='#08111c';x.fillRect(0,0,canvas.width,canvas.height);S.projection=[];
 const source=$('geometryState').value==='baseline'||['gt','sensor_overlay'].includes(S.channel);
 $('geometryTitle').textContent=source?'30K 原场景参照 · Gaussian / LiDAR':'当前请求 · Gaussian / LiDAR';
 $('geometryStateInfo').textContent=source?'已选择原场景参照；与编辑结果不同。':selected().title+' · 与左侧共用请求、时刻和相机。';
 if(!st?.supported){empty(canvas,st?.note||'本组没有此帧的编辑几何');$('cloudInfo').textContent='未拿原场景替代当前请求。';$('boxInfo').textContent='';$('viewMode').textContent='';return;}
 if(!S.clouds){empty(canvas,'几何加载中');return;}
 if(!S.cameraView)for(let v=-60;v<=100;v+=10){line3(x,[[v,-60,0],[v,60,0]],'#182c40');line3(x,[[-40,v,0],[100,v,0]],'#182c40');}
 const mode=$('cloudColor').value,instances=geometryInstances(),counts={};
 for(const instance of instances){
  const part=instance.cloud==='static'?'static':'dynamic';if(!$(part+'3d').checked)continue;
  const cloud=S.clouds[instance.cloud];if(!cloud){empty(canvas,'外来资产几何加载中');return;}
  const {data}=cloud;let count=0;
  for(let j=0;j<data.ids.length;j++){
   const own=ownershipRow(instance,data.ids[j]);if(mode==='ownership'&&own&&!$('owner'+(own.state===2?'U':own.state)).checked)continue;
   const baseSample=sampleBase(cloud,part,j)||(mode==='ownership'&&own?.state===1&&$('allPositive').checked);if(!baseSample&&(!S.sampling||S.sampling.boost==='none'))continue;
   const pt=editedPoint(instance,j);if(!baseSample&&!sampleBoost(pt.p,data.ids[j]))continue;if(pt.marginal<=.05||pt.opacity<.005)continue;
   const p=projection(pt.p);if(!p||p[0]<0||p[0]>=canvas.width||p[1]<0||p[1]>=canvas.height)continue;
   const foreign=instance.cloud.startsWith('foreign:');
   x.fillStyle=mode.startsWith('semantic')?semanticColor(semanticRow(instance,j),mode):mode==='ownership'?(own?['#63b3ff','#4bea9b','#ffb347'][own.state]:'#787f89'):mode==='rgb'?`rgb(${data.rgb[j*3]},${data.rgb[j*3+1]},${data.rgb[j*3+2]})`:mode==='branch'?(part==='static'?'#78b0f4':foreign?'#68f2bd':'#ffa85d'):`rgb(${heat(pt.opacity).join(',')})`;
   x.globalAlpha=$('opaquePoints').checked?1:Math.max(.25,pt.opacity);
   const size=Number($('pointSize').value)||2.5;x.fillRect(p[0]-size/2,p[1]-size/2,size,size);
   S.projection.push({x:p[0],y:p[1],part,j,p:pt.p,opacity:pt.opacity,instance:instance.key,cloud:instance.cloud});count++;
  }
  counts[instance.key]=count;
 }
 x.globalAlpha=1;
 // Observed LiDAR is never moved/deleted to impersonate a simulated sensor.
 if($('lidar3d').checked&&S.lidar){x.fillStyle='#54e2e9';for(let j=0;j<S.lidar.depth.length;j++){
  const p=projection(Array.from(S.lidar.xyz.slice(j*3,j*3+3)));if(p&&p[0]>=0&&p[0]<canvas.width&&p[1]>=0&&p[1]<canvas.height){const size=Math.max(1,(Number($('pointSize').value)||2.5)*.65);x.fillRect(p[0]-size/2,p[1]-size/2,size,size);}
 }}
 for(const b of currentBoxes()){
  if(!($('boxes3d').checked&&b.target)&&!($('otherBoxes3d').checked&&!b.target))continue;
  const ps=b.corners_local,color=b.imported?'#68f2bd':b.target?'#ffbf47':'#59a9ff';
  for(let n=0;n<8;n++)for(let j=0;j<3;j++)if(!(n&(1<<j)))line3(x,[ps[n],ps[n|(1<<j)]],color,b.target?2:1);
  const projected=ps.map(projection).filter(Boolean);if(projected.length===8){const xx=Math.min(...projected.map(p=>p[0])),yy=Math.min(...projected.map(p=>p[1]));if(xx<canvas.width&&yy<canvas.height&&Math.max(...projected.map(p=>p[0]))>=0&&Math.max(...projected.map(p=>p[1]))>=0){x.fillStyle=color;x.font='13px monospace';x.fillText(b.label,Math.max(0,xx),Math.max(15,yy-3));}}
 }
 const fr=D.frames[S.raw],cam=st.camera;
 if(!S.cameraView){
  line3(x,raws.map(r=>D.frames[r].c2w.slice(0,3).map(row=>row[3])),'#597794',1);
  const key=source?'reconstruction_30000':S.group,states=Object.values(window.EDIT_GEOMETRY.groups[key]||{}).filter(s=>s.supported);
  line3(x,states.map(s=>s.camera.slice(0,3).map(row=>row[3])),'#5fdfb0',2.5);
  function frustum(c,color){const origin=c.slice(0,3).map(r=>r[3]),k=fr.K,corners=[[0,0],[W,0],[W,H],[0,H]].map(([u,v])=>matp(c,[(u-k[2])/k[0]*3,(v-k[3])/k[1]*3,3]));for(let i=0;i<4;i++){line3(x,[origin,corners[i]],color,2);line3(x,[corners[i],corners[(i+1)%4]],color,2);}}
  frustum(fr.c2w,'#597794');frustum(cam,'#5fdfb0');
  for(const instance of st.objects){const ps=states.map(s=>s.objects.find(o=>o.key===instance.key)?.center_local).filter(Boolean);line3(x,ps,instance.cloud.startsWith('foreign:')?'#68f2bd':'#e4ba73',1.3);}
 }
 // Depth rays follow the same edited optical camera as the saved map.
 if(S.maps&&!source&&selected().frames[S.raw]?.maps){const z=S.maps.depth_m?.[S.pixel[1]*W+S.pixel[0]];if(Number.isFinite(z)){const k=fr.K,p=matp(cam,[(S.pixel[0]+.5-k[2])/k[0]*z,(S.pixel[1]+.5-k[3])/k[1]*z,z]);line3(x,[cam.slice(0,3).map(r=>r[3]),p],'#ff92ba',2);}}
 const inst=S.selected&&instances.find(o=>o.key===S.selected.instance);
 if(inst){
  const {j}=S.selected,d=S.clouds[inst.cloud].data,pt=editedPoint(inst,j),p=projection(pt.p);
  if(p){x.strokeStyle='#ff8cbd';x.lineWidth=2;x.beginPath();x.arc(p[0],p[1],7,0,Math.PI*2);x.stroke();}
  for(const axes of [[0,1],[0,2],[1,2]]){const pts=[];for(let t=0;t<=48;t++){const a=t/48*2*Math.PI,offset=[0,1,2].map(k=>d.rotation[j*9+k*3+axes[0]]*d.scales[j*3+axes[0]]*Math.cos(a)+d.rotation[j*9+k*3+axes[1]]*d.scales[j*3+axes[1]]*Math.sin(a));const v=inst.transform?[0,1,2].map(k=>offset.reduce((s,x,i)=>s+inst.transform[k][i]*x,0)):offset;pts.push(pt.p.map((p,k)=>p+v[k]));}line3(x,pts,'#ff8cbd');}
  $('gaussianInfo').textContent=`${inst.label} · ${inst.source_context||D.context} · checkpoint 行 ${d.ids[j]} · 当前编辑位置米 (${pt.p.map(v=>fmt(v,2)).join(', ')}) · opacity ${fmt(pt.opacity)} · 时间 ${fmt(inst.query_time,3)}。椭球方向使用相同刚体旋转；不是像素贡献排名。`+(mode==='ownership'?(()=>{const v=ownershipRow(inst,d.ids[j]);return v?` 候选 ${v.state===2?'unknown':v.state}；目标证据 ${v.target_hits} 帧，非目标证据 ${v.other_hits} 帧（候选需要 ≥3 帧、覆盖 ≥2 时间段且无反证）。`:' 候选未加载';})():'');
  if(mode.startsWith('semantic'))$('gaussianInfo').textContent+=' 语义观测：'+JSON.stringify(semanticRow(inst,j));
 }else $('gaussianInfo').textContent='点击几何点查看真实 checkpoint 行号、编辑后位置和查询时间。';
 x.fillStyle='#9fb5d1';x.font='12px monospace';x.fillText('EDIT STATE · grid 10 m · Gaussian samples · no RGB backdrop',12,canvas.height-13);
 $('pointSizeValue').textContent=Number($('pointSize').value).toFixed(1)+' px';
 $('viewMode').textContent=S.cameraView?(S.cameraReference==='original'?'原相机观察当前几何（显式对照）':'当前请求相机 · 与左侧原生输出同投影'):'三维总览 · 绿线/视锥为当前请求相机 · 蓝灰为原相机';
 if(mode==='ownership')$('ownershipStatus').textContent='1 绿色：目标候选；0 蓝色：非目标候选；unknown 橙色。灰色：尚未加载证据。以当前查询帧显示中心；标签来自全49帧证据。不是像素贡献真值。';
 semanticLegend();const count=IA?.frames[S.raw].visible_count;
 $('boxInfo').textContent=`当前有效资产框 ${st.objects.length} 个${count!==undefined?'；原帧 '+count+' 个车辆标注与前视相交':''}。目标框随资产平移、旋转或删除；其他蓝框是原始观测标注，不代表独立资产。`;
 const sampling=instances.map(o=>{const c=S.clouds[o.cloud];return c?`${o.label}：${c.sample.toLocaleString()}/${c.total.toLocaleString()} 行（${(100*c.sample/c.total).toFixed(2)}%）`:'';}).filter(Boolean).join('；');
 $('cloudInfo').textContent=`30K 几何加载量：${sampling}。加载量不等于显示量；显示采样见上方配置。中心点不是完整高斯表面。当前画内 ${Object.entries(counts).map(([k,n])=>k+': '+n).join(' / ')}。${$('lidar3d').checked?'青色 LiDAR 保持原始实测状态，不是编辑后传感器输出。':''} ${st.note||''}${S.channel==='generated'?' 右侧是视频后处理之前的三维状态。':''}`;
}
