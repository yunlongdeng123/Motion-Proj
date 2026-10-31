(function(){
  'use strict';
  const scenes=Array.isArray(window.ACTOR_CANDIDATES)?window.ACTOR_CANDIDATES:[];
  const naming=window.ActorNames, annotations=window.ACTOR_ANNOTATIONS||{};
  const el=id=>document.getElementById(id), scene=el('scene'), actor=el('actor'), actorFilter=el('actorFilter');
  const isPrepared=(data,row)=>window.ACTOR_ELIGIBILITY?.[data.context_id]?.[row.track_id]?.in_original_preparation45===true;
  let imageEpoch=0, trajectoryProjection=[], synchronizedRaw=null;
  const display=x=>x==null?'暂无':String(x);
  const current=()=>{const data=scenes[Number(scene.value)];return {data,row:data?.actors[Number(actor.value)]};};
  const meta=(data,row)=>annotations[data.context_id]?.[row.track_id]||{};
  const alias=(data,row)=>naming.actor(data.context_id,row.track_id)||'代号待核对';
  function option(select,value,label){const node=document.createElement('option');node.value=value;node.textContent=label;select.append(node);}
  function fact(label,value){const wrap=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;wrap.append(dt,dd);el('facts').append(wrap);}
  function updateHash(data,row){if(window.history?.replaceState){const params=new URLSearchParams({scene:naming.sceneKey(data.context_id),actor:alias(data,row)});window.history.replaceState(null,'','#'+params);}}
  function paintPreview(data,row,m){
    const image=el('preview'),canvas=el('previewOverlay');
    canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
    const code='actor '+alias(data,row),header=naming.scene(data.context_id)+' · '+code+' · 原始帧 '+row.preview_frame;
    ctx.fillStyle='#10202d';ctx.fillRect(0,0,canvas.width,24);ctx.fillStyle='#ffffff';ctx.font='bold 13px sans-serif';ctx.fillText(header,8,17);
    if(m.preview_box_image_xyxy){
      const [x0,y0,x1,y1]=m.preview_box_image_xyxy;
      ctx.strokeStyle='#00ffbb';ctx.lineWidth=3;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
      ctx.font='bold 15px sans-serif';const width=ctx.measureText(code).width+14;
      const x=Math.max(0,Math.min(canvas.width-width,x0)),y=Math.max(25,Math.min(canvas.height-24,y0-25));
      ctx.fillStyle='#00ffbb';ctx.fillRect(x,y,width,24);ctx.fillStyle='#061a19';ctx.fillText(code,x+7,y+17);
    }
  }
  function drawTrajectory(data,row,m){
    const canvas=el('trajectoryCanvas'),ctx=canvas.getContext('2d');
    const media=window.ACTOR_MEDIA?.[data.context_id];const full=media&&Array.isArray(media.frames)&&media.frames.length===media.frame_count&&media.frame_count===media.full_raw_frame_denominator?media.actors?.[row.track_id]?.trajectory:null;
    const hasFull=Array.isArray(full);let pts;
    if(hasFull){const sorted=full.filter(p=>Number.isInteger(p.raw)&&Array.isArray(p.center_world)&&p.center_world.length>=2&&p.center_world.slice(0,2).every(Number.isFinite)).slice().sort((a,b)=>a.raw-b.raw);let segment=0,previous=null;const base=sorted[0]?.center_world;pts=sorted.map(p=>{if(previous!==null&&p.raw!==previous+1)segment++;previous=p.raw;return [p.raw,p.center_world[0]-base[0],p.center_world[1]-base[1],segment];});}
    else pts=m.trajectory_points||[];
    ctx.fillStyle='#0b1522';ctx.fillRect(0,0,canvas.width,canvas.height);trajectoryProjection=[];
    el('trajectorySection').hidden=!pts.length;if(!pts.length)return;
    const xs=pts.map(p=>p[1]),ys=pts.map(p=>p[2]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    const scale=Math.min((canvas.width-120)/Math.max(xmax-xmin,1),(canvas.height-92)/Math.max(ymax-ymin,1));
    const project=p=>[canvas.width/2+(p[1]-(xmin+xmax)/2)*scale,canvas.height/2-(p[2]-(ymin+ymax)/2)*scale+8];
    ctx.strokeStyle='#29415a';ctx.lineWidth=1;ctx.strokeRect(58,42,canvas.width-116,canvas.height-84);
    ctx.strokeStyle='#67d4ff';ctx.lineWidth=2.5;ctx.beginPath();let previous=null;
    pts.forEach(p=>{const q=project(p);if(!previous||p[3]!==previous[3])ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1]);previous=p;trajectoryProjection.push({x:q[0],y:q[1],point:p});});ctx.stroke();
    function dot(point,color,r=4){const q=project(point);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fill();}
    dot(pts[0],'#a6edb3');dot(pts[pts.length-1],'#ffbc6f');
    const selectedRaw=synchronizedRaw==null?row.preview_frame:synchronizedRaw;const preview=pts.find(p=>p[0]===selectedRaw);if(preview)dot(preview,'#ff8ab5',6);
    ctx.font='bold 16px sans-serif';ctx.fillStyle='#e9f3ff';ctx.fillText('actor '+alias(data,row)+' · 原始标注轨迹',20,25);
    ctx.font='12px sans-serif';ctx.fillStyle='#b4c8df';ctx.fillText('起点 raw'+pts[0][0]+' → 终点 raw'+pts[pts.length-1][0],20,canvas.height-12);
    ctx.fillText('横轴：世界 X；纵轴：世界 Y（米，相对首点）',canvas.width-355,canvas.height-12);
    el('trajectoryInfo').textContent='actor '+alias(data,row)+' · 原始帧 '+pts[0][0]+'–'+pts[pts.length-1][0]+' · '+(hasFull?'完整标注轨迹 '+pts.length+' 点':'抽样轨迹 '+pts.length+'/'+m.trajectory_observation_count+' 点（部分标注帧未显示）')+'；缺测段不连线。粉色点对应当前帧（无该帧标注或未包含于抽样时不显示），悬停轨迹可查看帧号。';
  }
  function showActor(){
    const {data,row}=current();if(!data||!row)return;const m=meta(data,row),code=alias(data,row);
    el('sceneTitle').textContent=naming.scene(data.context_id);el('actorTitle').textContent='actor '+code+(isPrepared(data,row)?' · 准备候选':'');
    el('actorReference').textContent=naming.scene(data.context_id)+' / '+code;
    el('rawTrackId').textContent=row.track_id;el('rawContextId').textContent=data.context_id;
    el('nativeAssetIndex').textContent=naming.record(data.context_id,row.track_id)?.native_asset_index==null?'无已绑定资产序号':String(naming.record(data.context_id,row.track_id).native_asset_index);
    const has=!!row.preview_image,epoch=++imageEpoch,image=el('preview');el('previewWrap').hidden=!has;el('missingImage').hidden=has;el('downloadPreview').disabled=!has;
    if(has){image.onload=()=>{if(epoch===imageEpoch)paintPreview(data,row,m);};image.onerror=()=>{if(epoch!==imageEpoch)return;el('previewWrap').hidden=true;el('missingImage').hidden=false;el('missingImage').textContent='\u56fe\u50cf\u65e0\u6cd5\u8bfb\u53d6\uff1b\u8bf7\u68c0\u67e5\u672c\u5730\u9884\u89c8\u6587\u4ef6\u3002';el('downloadPreview').disabled=true;};image.src=row.preview_image;image.alt=naming.scene(data.context_id)+' / actor '+code+' · 原始帧 '+row.preview_frame;el('frameInfo').textContent='actor '+code+' · 原始帧 '+row.preview_frame+' · 绿色框为该轨迹的原始标注；代号为场景内固定沟通标签。';}
    else{image.onload=null;image.onerror=null;image.removeAttribute('src');el('missingImage').textContent='\u6b64\u8f68\u8ff9\u6682\u65e0\u9884\u89c8\u56fe\u3002';el('frameInfo').textContent='actor '+code+' 暂无对应预览帧。';}
    el('facts').replaceChildren();fact('车辆代号',code);fact('全片标注帧数',display(row.labelled_frames));fact('前向相机投影帧数',display(row.front_projected_frames));fact('前向标注框面积中位比例',row.median_front_bbox_area_ratio==null?'暂无':(100*row.median_front_bbox_area_ratio).toFixed(2)+'%');fact('框内激光点数中位值',display(row.median_front_lidar_annotation_count));fact('掩码文件记录',row.body_mask_frames==null?'暂无':row.body_mask_frames+' 帧（正像素待核）');fact('已观察方位',row.front_view_sector_count==null?'暂无':row.front_view_sector_count+'/12 个粗扇区（代理）');fact('背景显露支持','待测');fact('当前模型范围',row.window_label||'暂无');
    const state={'existing_dynamic_target_ready_for_validation':'已有动态目标，可进一步核验','prepared_dynamic_asset_restore_validation_needed':'已有动态资产，需核验恢复','existing_static_gs_candidate_preparation_needed':'有场景模型，需准备该车辆候选','existing_gs_no_front_projection_in_current_window':'当前模型范围内无前向投影','hf_prepare_inputs':'需准备车身掩码、场景模型和逐车候选'};fact('车辆输入',state[row.editable_status]||'待核对');
    synchronizedRaw=null;drawTrajectory(data,row,m);updateHash(data,row);window.dispatchEvent(new CustomEvent('actor-selection-changed'));
  }
  function renderMapping(data){naming.renderMap(el('mappingTable'),data.context_id,track=>{const i=data.actors.findIndex(r=>r.track_id===track);if(i>=0){if(actorFilter.value==='prepared'&&!isPrepared(data,data.actors[i]))showScene(track);else{actor.value=String(i);showActor();}}});}
  function showScene(preferred){
    const data=scenes[Number(scene.value)];if(!data)return;actor.replaceChildren();
    const i=preferred?data.actors.findIndex(row=>row.track_id===preferred||alias(data,row)===preferred):-1;
    if(i>=0&&actorFilter.value==='prepared'&&!isPrepared(data,data.actors[i]))actorFilter.value='all';
    const included=[];data.actors.forEach((row,index)=>{if(actorFilter.value==='prepared'&&!isPrepared(data,row))return;included.push(index);option(actor,String(index),alias(data,row)+(row.preview_image?'':' · 暂无预览'));});
    actor.value=String(i>=0?i:(included[0]??0));
    el('lookupStatus').textContent='';el('actorSearch').value='';renderMapping(data);showActor();
  }
  function findActor(){const {data}=current();if(!data)return;const value=el('actorSearch').value.trim().replace(/^actor\s+/i,'');const i=data.actors.findIndex(row=>alias(data,row).toLowerCase()===value.toLowerCase()||row.track_id===value);if(i<0){el('lookupStatus').textContent='本场景中未找到该代号或轨迹 ID。';return false;}if(actorFilter.value==='prepared'&&!isPrepared(data,data.actors[i]))showScene(data.actors[i].track_id);else{actor.value=String(i);showActor();}el('lookupStatus').textContent='已定位 actor '+alias(data,data.actors[i]);return true;}
  function downloadMapping(){const {data}=current();if(!data)return;const payload=naming.rows(data.context_id);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download=naming.sceneKey(data.context_id)+'_actor_mapping.json';a.click();URL.revokeObjectURL(a.href);}
  function downloadPreview(){const {data,row}=current(),image=el('preview');if(!row?.preview_image||!image.naturalWidth)return;try{const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);ctx.drawImage(el('previewOverlay'),0,0);const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=naming.sceneKey(data.context_id)+'_'+alias(data,row)+'_raw'+row.preview_frame+'.png';a.click();}catch(error){el('lookupStatus').textContent='浏览器限制本地图片导出；可直接截图保留当前代号和标注框。';}}
  scenes.forEach((data,i)=>option(scene,String(i),naming.scene(data.context_id)));
  const params=new URLSearchParams((window.location?.hash||'').replace(/^#/,''));const initial=scenes.findIndex(data=>naming.sceneKey(data.context_id)===params.get('scene')||data.context_id===params.get('scene'));scene.value=String(initial>=0?initial:0);
  scene.addEventListener('change',()=>showScene());actorFilter.addEventListener('change',()=>showScene());actor.addEventListener('change',showActor);el('findActor').addEventListener('click',findActor);el('actorSearch').addEventListener('keydown',event=>{if(event.key==='Enter')findActor();});el('downloadMapping').addEventListener('click',downloadMapping);el('downloadPreview').addEventListener('click',downloadPreview);
  el('trajectoryCanvas').addEventListener('mousemove',event=>{if(!trajectoryProjection.length)return;const rect=el('trajectoryCanvas').getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*el('trajectoryCanvas').width,y=(event.clientY-rect.top)/rect.height*el('trajectoryCanvas').height;let nearest=null;for(const p of trajectoryProjection){const d=(x-p.x)**2+(y-p.y)**2;if(!nearest||d<nearest.d)nearest={...p,d};}if(nearest&&nearest.d<900){const {data,row}=current(),p=nearest.point;el('trajectoryInfo').textContent='actor '+alias(data,row)+' · 原始帧 '+p[0]+' · 相对首点 X '+p[1].toFixed(2)+' m，Y '+p[2].toFixed(2)+' m；播放和滑块控制当前影像帧。';}});
  function setSynchronizedFrame(raw){synchronizedRaw=raw;const {data,row}=current();if(data&&row)drawTrajectory(data,row,meta(data,row));}window.addEventListener('hashchange',()=>{const next=new URLSearchParams((window.location?.hash||'').replace(/^#/,''));const index=scenes.findIndex(data=>naming.sceneKey(data.context_id)===next.get('scene')||data.context_id===next.get('scene'));if(index>=0){scene.value=String(index);showScene(next.get('actor'));}});window.ActorCandidateViewer={showScene,showActor,findActor,paintPreview,current,setSynchronizedFrame};showScene(params.get('actor'));
})();
