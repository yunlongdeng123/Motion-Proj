'use strict';
(()=>{
 const $=id=>document.getElementById(id),fmt=v=>v===null||v===undefined?'未知':Number(v).toFixed(6);
 const names={geometry_conflict:'几何冲突候选',legal_occlusion:'合法遮挡／后方道路',legal_road:'合法道路候选',unknown:'未知'};
 const reasons={insufficient_or_ambiguous_geometry:'证据不足或3σ区间与多个表面重叠',uncertain_projection_support:'投影协方差支持不明确',non_target_occluder_present_identity_or_surface_unverified:'非目标遮挡存在；缺身份／表面匹配，不误删',static_in_verified_free_space_before_visible_road:'位于确认可见道路前、无观测遮挡物支持',road_surface_behind_measured_target_vehicle:'在实测目标车辆后方、匹配道路曲面',within_measured_visible_road_surface_band:'在实测可见道路深度带内',static_near_measured_vehicle_surface_and_vehicle_color:'静态贡献贴合实测车表面且颜色近似车体',dynamic_matches_measured_target_vehicle_surface:'动态贡献匹配实测目标车表面',no_actual_contribution:'无实际贡献'};
 const panel=document.createElement('section');panel.className='panel';panel.id='geometryEvidence';panel.innerHTML='<div class="panel-title"><span class="number">P0</span><h2>几何证据判别 · 不等于删点指令</h2></div><div id="geometrySummary" class="notice">选择有原生CUDA归因的像素。缺测量不强判。</div><div id="geometryDetail" class="note"></div><div class="table-wrap"><table><thead><tr><th>Gaussian</th><th>当前射线标签</th><th>条件深度3σ [m]</th><th>理由 / 证据</th></tr></thead><tbody id="geometryRows"></tbody></table></div><p class="note">实际Tα与conic来自原生光栅器；条件深度区间结合三维协方差与原生投影J，是线性诊断代理，不是首交面或校准后的传感器置信区间。标签只适用于当前帧／射线。非目标mask不等于已确认运动身份；证据不足保留未知。未导出的贡献没有被分类。</p>';
 document.querySelector('.workspace').after(panel);
 const tools=document.createElement('div');tools.className='subtools';tools.innerHTML='<label>证据筛选<select id="geometryFilter"><option value="all">全部导出贡献</option><option value="geometry_conflict">仅几何冲突候选</option><option value="protect">合法道路／遮挡保护</option><option value="unknown">仅未知</option></select></label><button id="selectConflicts">选择当前冲突候选（仅诊断）</button><span id="geometrySelection" class="note"></span>';panel.querySelector('.panel-title').after(tools);
 function showTrace(value){
  const e=value.geometry_evidence;const body=$('geometryRows');body.replaceChildren();
  if(!e){$('geometrySummary').textContent=value.arm==='v4_1_R'?'当前为已核验原生CUDA贡献；逐高斯合法/冲突几何标签尚未完成，保持未知。':'当前为CPU参考或旧数据，没有测量判别；保持未知。';$('geometryDetail').textContent='';return;}
  const s=e.support,r=s.road,v=s.vehicle;
  $('geometrySummary').textContent=Object.entries(e.counts).map(([k,n])=>`${names[k]} ${n}`).join(' · ')+'。这是Top权重∪Top颜色导出集合，不是整条射线总量。';
  $('geometryDetail').textContent=`LiDAR z=${fmt(s.lidar_m)}m，匹配偏移${fmt(s.lidar_offset_px)}px；容差±${fmt(s.lidar_uncertainty_m)}m。道路X${r.unit_id} z=${fmt(r.z_m)}m，±${fmt(r.uncertainty_m)}m，局部曲面${r.supported?'有支撑':'不足'}，当前${r.current_visible?'确认无遮挡曝光':'不能确认无遮挡'}，${r.confirmed_sources}个通过来源。${v?`车辆OBB范围${fmt(v.near_m)}—${fmt(v.far_m)}m（框不是实测表面）。`:'无已核对目标车表面。'}来源帧：${r.source_raws.map(x=>'raw'+x).join(', ')||'无'}。`;
  const filter=$('geometryFilter').value;
  for(const row of [...value.top].sort((a,b)=>b.weight-a.weight)){
   const g=row.geometry_evidence||{label:'unknown'},tr=document.createElement('tr');
   if(filter!=='all'&&!(filter==='protect'?g.severity==='protect':g.label===filter))continue;
   const interval=g.conditional_depth?.interval_m;
   const entries=[`${row.branch}:${row.row} · Tα=${fmt(row.weight)}`,names[g.label],interval?interval.map(fmt).join(' — '):'未知',reasons[g.reason]||g.reason||'未知'];
   for(const [i,t] of entries.entries()){const td=document.createElement('td');td.textContent=t;if(i===1)td.style.color=g.severity==='suspect'?'#ff9ea9':g.severity==='protect'?'#69d5ae':'#bbad82';tr.append(td);}
   tr.onclick=()=>{window.gsdiag.selectGaussian(row);$('gaussianInfo').textContent=`${row.id} · ${names[g.label]} · ${reasons[g.reason]||g.reason}。这是当前射线证据，不是永久语义标签。`;};body.append(tr);
  }
 }
 $('geometryFilter').onchange=()=>{const value=window.gsdiag.state.ray;if(value)showTrace(value);};
 $('selectConflicts').onclick=()=>{const s=window.gsdiag.state;if(!s.ray?.geometry_evidence)return;s.checked.clear();for(const r of s.ray.top)if(r.geometry_evidence?.label==='geometry_conflict')s.checked.add(r.id);window.gsdiag.rankRows();$('geometrySelection').textContent=`已选择${s.checked.size}个当前射线候选；未删模型。需手动完整重渲染并检查valid误差／正常区损伤，标签本身不保证收益。`;};
 function metricTable(e,title){
  const rows=[['验证道路 valid',e.valid],['干预前正常区 / 邻域外',e.normal_control]];
  const h=document.createElement('div');h.className='table-wrap';h.innerHTML='<h3></h3><table><thead><tr><th>固定分母</th><th>世界单元数</th><th>L前</th><th>L后</th><th>ΔL（正=改善）</th><th>RGB误伤数</th><th>深度MAE 前→后 [m]</th><th>α变化</th><th>道路贡献Δ</th></tr></thead><tbody></tbody></table>';h.querySelector('h3').textContent=title;
  for(const [name,m] of rows){const tr=document.createElement('tr');for(const text of [name,m.count,fmt(m.L_before),fmt(m.L_after),fmt(m.delta_L_valid),m.RGB_harmed_count??'未知',`${fmt(m.depth_MAE_before_m)} → ${fmt(m.depth_MAE_after_m)}`,fmt(m.alpha_change),fmt(m.road_contribution_change)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}h.querySelector('tbody').append(tr);}return h;
 }
 const evalPanel=document.createElement('div');evalPanel.id='scientificEvaluation';$('cfMetrics').after(evalPanel);
 function showCF(value){
  evalPanel.replaceChildren();const e=value.scientific_evaluation;
  if(!e){evalPanel.textContent=value.arm==='v4_1_R'?'本次已记录目标车mask外观测损伤，但尚未完成固定世界有效路面验收；不把全图变化当隐藏路面改善。':'该旧干预尚未进行固定世界有效区评价，不用全图变化冒充改善。';return;}
  evalPanel.append(metricTable(e,`raw${value.raw} · 固定同世界验证区域`));
  const damage=document.createElement('p');damage.className='note';const normal=e.normal_control;damage.textContent=`正常区：深度误差增加超过容差 ${normal.depth_harmed_count??'未知'} 单元；α平均下降 ${fmt(normal.alpha_drop)}，下降超过1/255的单元 ${normal.alpha_drop_over_1_255_count??'未知'}。未通过RGB验证的查询投影只报告道路贡献Δ ${fmt(e.unobserved_query.road_contribution_change)}、期望深度变化 ${fmt(e.unobserved_query.depth_change_m)}m、α变化 ${fmt(e.unobserved_query.alpha_change)}；不定义隐藏RGB误差。`;evalPanel.append(damage);
  const note=document.createElement('p');note.className='note';note.textContent=`valid逐单元要求：九来源门限、至少2帧确认、当前四采样像素无遮挡（外扩1px）、1.5px内实测LiDAR且匹配道路z；干预前固定分母。${e.valid.count}个世界单元 / ${e.unique_valid_pixels}个独立屏幕采样位置。正常区还要求干预前RGB L1≤0.05、α≥0.95且深度一致，位于21×21邻域外。未通过有效区的${e.unobserved_query.count}个可见投影查询单元没有隐藏RGB真值。阈值是声明的诊断容差，不是人工科学验收。`;evalPanel.append(note);
  const temporal=value.temporal_controls||[];for(const t of temporal)if(t.scientific_evaluation)evalPanel.append(metricTable(t.scientific_evaluation,`同一Gaussian集合 · raw${t.raw}的已曝光世界单元（不使用同屏幕坐标代替世界点）`));
 }
 window.evidenceUI={showTrace,showCF};
 const link=document.createElement('a');link.href='EVIDENCE_REPORT.html';link.textContent='几何判别／V1证据报告 ↗';document.querySelector('nav').append(link);
 const option=document.createElement('option');option.value='gn11_v1';option.textContent='GN11 V1 · 原生30K（历史恢复）';$('arm').append(option);
 document.querySelector('h1 em').textContent='V3 × V1';
 for(const c of window.GSDATA?.evidence_cases?.cases||[]){const o=document.createElement('option');o.value=`102,${c.pixel[0]},${c.pixel[1]}`;o.textContent='raw102 · '+(c.name==='confirmed_visible_road'?'测量确认无遮挡道路':'实测目标车辆表面')+` (${c.pixel})`;$('preset').append(o);}
 const v1note=document.createElement('p');v1note.className='note';v1note.textContent='V1 原生GN11：历史20K恢复到30K；同49帧、同五身份、只读。每帧三分支已对原训练节点PNG核对，不向V1注入V3道路或重新训练。它是帧对齐诊断参照，不是只改变一个因素的V3因果消融。';$('comparison').before(v1note);
})();
