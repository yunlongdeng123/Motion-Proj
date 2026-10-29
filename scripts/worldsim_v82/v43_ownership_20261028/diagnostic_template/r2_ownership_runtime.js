(function(root){
  'use strict';
  const M=root.R2_INHERITED_DELIVERY,E=root.R2_OWNERSHIP49,a=root.SceneViewer;
  if(!M||!E||!a||M.status!=='complete'||!M.validated)return;
  const PRE='v43_r2_inspect49',OWN='v43_r2_ownership49',POST=M.group_id,COLOR='r2_candidates';
  const EDITS=new Set(['v43_actor_10000','v43_motion_no_edit','v43_motion_delete_target','v43_motion_translate_lateral_1m','v43_motion_turn_left15','v43_shift_3m']);
  const get=id=>document.getElementById(id),panel=get('r2InheritedPair'),names=['original','existing','white'];
  const option=document.createElement('option');option.value=COLOR;option.textContent='R2归属：车辆核心＋unknown';get('cloudColor').append(option);
  const sets=new Map(),partitions=new Map();
  function evidence(){return E.frames[String(S.raw)];}
  function selectedRows(){if(!sets.has(S.raw)){const f=evidence();sets.set(S.raw,{all:new Set(f.static.final_removed_ids),added:new Set(f.static.new_unknown_ids)});}return sets.get(S.raw);}
  function current(){return get('geometryState').value!=='baseline'&&!['gt','sensor_overlay','semantic_2d'].includes(S.channel);}
  function inspection(){return ([PRE,OWN].includes(S.group)||EDITS.has(S.group))&&current()&&!!evidence();}
  function parameters(){return {radius:Number(get('r2Boundary').value),fraction:Number(get('r2BoundaryMass').value)};}
  function actorPartition(){
    const f=evidence(),p=parameters(),key=S.raw+':'+p.radius+':'+p.fraction;
    if(partitions.has(key))return partitions.get(key);
    const data=f.actor,ix=E.radii.indexOf(p.radius),roles=[],counts={green:0,unknown:0,boundary:0,base:0,missing:0};
    for(let row=0;row<data.class_id.length;row++){
      const missing=!data.present[row],base=data.class_id[row],vehicle=base>=14&&base<=19;
      const band=ix>=0&&data.boundary_fraction[row*3+ix]>=p.fraction;
      const green=!missing&&vehicle&&!band;
      const reasons=[];if(missing)reasons.push('本帧无足够可见贡献');else if(!vehicle)reasons.push(base===0?'基础低置信/混合语义':'车身资产内非车辆语义冲突');if(band)reasons.push('轮廓混合贡献');
      roles.push({role:green?'vehicle':'unknown',reasons,base_class:base,boundary_fraction:ix<0?0:data.boundary_fraction[row*3+ix],target_fraction:data.target_fraction[row],purity:data.purity[row],mass:data.mass[row],present:!missing});
      counts[green?'green':'unknown']++;if(band)counts.boundary++;if(!missing&&!vehicle)counts.base++;if(missing)counts.missing++;
    }
    const result={roles,counts};partitions.set(key,result);return result;
  }
  function rowRole(cloud,row){
    if(cloud==='v43:10000')return {...actorPartition().roles[row],source:'Actor10K原观测逐点alpha*T语义与轮廓证据',raw:S.raw,row,cloud};
    if(cloud==='static'&&selectedRows().all.has(row))return {role:'unknown',reasons:[selectedRows().added.has(row)?'R2剥离后剩余贡献追踪新增':'继承本帧existing候选'],source:'本帧R2实际剥离static行',raw:S.raw,row,cloud};
    return {role:'unselected',reasons:['本轮R2未选中；不等于确认背景'],raw:S.raw,row,cloud};
  }
  function switchTo(id){S.channel='rgb';S.selected=null;get('geometryState').value='current';get('semanticAblation').value='all';get('static3d').checked=true;get('dynamic3d').checked=true;get('cloudColor').value=id===POST?'branch':COLOR;setCategory('v43',id);return show();}
  get('r2InspectOwnership').onclick=()=>switchTo(OWN);get('r2InspectBefore').onclick=()=>switchTo(PRE);get('r2InspectAfter').onclick=()=>switchTo(POST);
  get('r2Boundary').value='5';get('r2BoundaryMass').value='0.2';
  for(const id of ['r2Boundary','r2BoundaryMass'])get(id).onchange=()=>{S.selected=null;draw3D();};
  // Recolor genuine projected Gaussian centers only. No geometry or opacity edit.
  function paintCandidates(){
    if(!inspection()||get('cloudColor').value!==COLOR||!S.clouds)return;
    const ctx=get('sceneCanvas').getContext('2d'),size=Number(get('pointSize').value)||2.5;
    for(const p of S.projection){const id=S.clouds[p.cloud].data.ids[p.j],e=rowRole(p.cloud,id);
      ctx.fillStyle=e.role==='vehicle'?'#2de5aa':e.role==='unknown'?'#d9dde4':'#486780';
      ctx.globalAlpha=get('opaquePoints').checked?1:Math.max(0,Math.min(1,p.opacity));ctx.fillRect(p.x-size/2,p.y-size/2,size,size);}
    ctx.globalAlpha=1;
    if(S.selected){const p=S.selected,id=S.clouds[p.cloud].data.ids[p.j];get('gaussianInfo').textContent+=' R2逐点证据：'+JSON.stringify(rowRole(p.cloud,id))+'。';}
  }
  let lastKey='';
  function update(){
    const paired=[PRE,OWN,POST].includes(S.group),active=paired||EDITS.has(S.group);panel.hidden=!paired;get('r2GeometryControls').hidden=!active;
    const candidate=inspection()&&get('cloudColor').value===COLOR;get('semanticAblation').disabled=candidate;
    for(const id of ['semanticBoundary','semanticBoundaryMass','semanticStaticUnknown']){get(id).disabled=candidate;get(id).title=candidate?'当前R2归属使用下方独立轮廓参数与真实R2集合':'';}
    get('r2PartitionTools').hidden=!inspection();
    if(!active)return;
    const f=M.frames[String(S.raw)];if(!f){get('r2InheritedFrameStatus').textContent='本帧没有R2交付，不替换为邻帧。';return;}
    const key=S.group+':'+S.raw;if(lastKey!==key){lastKey=key;for(const name of names){get('r2Pair_'+name).src=f.images[name];get('r2PairLink_'+name).href=f.images[name];}}
    get('r2InheritedFrameStatus').textContent='raw'+S.raw+' · '+(S.raw-76)+'/49 · 同帧existing '+f.existing_rows.toLocaleString()+'行＋R2新增unknown '+f.new_rows.toLocaleString()+'行。'+(S.group===OWN?'左：已验收方向的真实R2剥离RGB；右：保留处理前几何，检查车辆/unknown归属。':S.group===PRE?'左：处理前原生RGB；右：同帧完整几何及R2归属。':'左/右均为R2剥离后状态。');
    get('r2FrameReceipt').href=f.receipt;get('r2FrameEvidence').href=f.evidence;
    for(const [id,g] of [['r2InspectOwnership',OWN],['r2InspectBefore',PRE],['r2InspectAfter',POST]])get(id).disabled=S.group===g&&current();
    if(!current()){get('r2GeometryStatus').textContent='当前三维是原30K参照，未应用R2归属检查。';get('semanticFilterStatus').textContent='用上方“R2归属检查”恢复同帧对照。';return;}
    if(inspection()){
      const q=actorPartition().counts,shown=S.projection||[],c={vehicle:0,unknown:0,unselected:0};for(const p of shown)c[rowRole(p.cloud,S.clouds[p.cloud].data.ids[p.j]).role]++;
      const edited=EDITS.has(S.group),deleted=S.group==='v43_motion_delete_target';
      get('geometryTitle').textContent=edited?'当前V43几何 · R2行身份着色':'R2归属检查 · 保留处理前完整高斯';
      get('geometryStateInfo').textContent=edited?selected().title+' · 同帧、同编辑变换。10K原观测归属随车辆行移动，static候选留在原位置；左侧原RGB未因着色而重渲染。':S.group===OWN?'与左侧同帧、同相机：左为R2剥离RGB，右保留处理前高斯，专门检查绿/灰归属。不是删除后的点云。':'处理前原生RGB＋完整高斯；使用同帧R2集合检查归属。';
      get('semanticFilterStatus').textContent=candidate?'绿=有车身语义证据且未进入轮廓待定带；浅灰=车身待定点＋本帧R2残影候选；灰蓝=本轮未选中。不再把整辆资产自动涂绿。':'已保留完整几何；切换“R2归属：车辆核心＋unknown”查看逐点划分。';
      if(candidate)get('geometryColorLegend').textContent='绿：车辆核心（现有证据支持）；浅灰：unknown（轮廓/低置信/缺测/R2残影）；灰蓝：未选中。金框目标，蓝框其他车辆。绿色仍需人工核验。';
      get('r2GeometryStatus').textContent='raw'+S.raw+' · 原车身6241颗：绿色核心 '+q.green+'，unknown '+q.unknown+'（轮廓 '+q.boundary+'、基础混合/冲突 '+q.base+'、缺测 '+q.missing+'；原因可重叠）。另有R2 static unknown '+f.removed_rows+'颗。'+(deleted?'当前删除组不显示已删除车身；仅检查剩余static候选。':'')+'当前画内采样：绿 '+c.vehicle+'、灰 '+c.unknown+'。';
      get('r2PartitionStatus').textContent=edited?'当前采用同帧Actor10K原观测证据；编辑后按行携带归属，不冒充编辑后的重新语义归因。轮廓参数仅改变绿/灰；下方unknown涂白图须与参数匹配，并用当前编辑状态原生重渲染。':'绿色＋车身灰色＝完整6241颗Actor10K；再加static灰色，精确对应本帧R2移除的高斯集合。轮廓参数只重新划分车身绿/灰，不改变R2集合或左侧PNG。环境回填不是高斯，R2白底诊断另有局部环境置白。';
      return;
    }
    get('r2GeometryStatus').textContent='剥离后：车辆资产和static候选已移除；要检查内圈绿/外圈灰，请选“R2归属检查”。';
    get('geometryStateInfo').textContent='真实R2剥离后几何：同帧existing＋逐轮剩余贡献追加；蓝框外禁止新增unknown。';
    get('semanticFilterStatus').textContent='当前是实际删除后的点云，不是归属着色。';
  }
  let previousGroup=null;
  const oldShow=show;show=async function(...args){if(previousGroup!==S.group){previousGroup=S.group;if(inspection())get('cloudColor').value=COLOR;}const result=oldShow(...args);update();await result;update();};
  const oldDraw=draw3D;draw3D=function(){option.disabled=!inspection();if(!inspection()&&get('cloudColor').value===COLOR)get('cloudColor').value='branch';if(inspection()&&get('cloudColor').value===COLOR)get('semanticAblation').value='all';oldDraw();paintCandidates();update();};get('pointSize').oninput=draw3D;
  const oldSemanticRow=semanticRow;semanticRow=function(instance,j){if(inspection()&&get('cloudColor').value===COLOR)return {...rowRole(instance.cloud,S.clouds[instance.cloud].data.ids[j]),classification_parameters:parameters(),labels_for_training:false,human_verified:false};return oldSemanticRow(instance,j);};
  const oldLoadSemantic=get('loadSemantic').onclick;get('loadSemantic').onclick=()=>{if(inspection()){get('cloudColor').value=COLOR;draw3D();}else if(oldLoadSemantic)return oldLoadSemantic();};
  Object.assign(a,{show,draw3D,semanticRow});root.R2Inherited={manifest:M,evidence:E,update,switchTo,paintCandidates,rowRole,actorPartition,inspection,parameters,compatibleGroups:Array.from(EDITS)};
  const params=new URLSearchParams(location.hash.replace(/^#workspace[?&]?/,''));
  if(!params.has('group')){S.raw=113;S.cameraView=true;S.cameraReference='current';switchTo(OWN);}else{if(inspection())get('cloudColor').value=COLOR;update();}
})(window);
