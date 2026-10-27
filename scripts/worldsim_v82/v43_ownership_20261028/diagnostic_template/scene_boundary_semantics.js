// Same observed pixels, separate original/new row namespaces. Abstention is a prior.
async function ensureBoundarySemantic(){
 if(!window.V43_BADCASE_AVAILABLE)return;
 if(!window.BOUNDARY_INDEX)await loadScript('gpu_semantics/boundary_index.js');
 const raw=S.raw;S.boundaryFrames=S.boundaryFrames||new Map();
 if(!S.boundaryFrames.has(raw)){
  await loadScript(window.BOUNDARY_INDEX.frames[raw]);const b=window.BOUNDARY_FRAMES[raw],value={};
  for(const key of ['original','actor10k','image'])value[key]=await unpack(b[key]);
  delete window.BOUNDARY_FRAMES[raw];S.boundaryFrames.set(raw,value);if(S.boundaryFrames.size>4){const old=S.boundaryFrames.keys().next().value;S.boundaryFrames.delete(old);scriptCache.delete(window.BOUNDARY_INDEX.frames[old]);}
 }
}
function boundarySemanticRow(instance,j){
 const frame=S.boundaryFrames?.get(S.raw);if(!frame)return null;
 const newer=instance.cloud.startsWith('v43:')||(instance.cloud==='static'&&S.group.startsWith('v43_')&&S.group!=='v43_main_noedit');
 const variant=newer?'actor10k':'original',a=frame[variant],ns=window.BOUNDARY_INDEX.branch_sizes[variant][0],part=instance.cloud==='static'?'static':'dynamic';
 if(!['static','dynamic'].includes(instance.cloud)&&!instance.cloud.startsWith('v43:'))return null;
 const row=S.clouds[instance.cloud].data.ids[j],id=row+(part==='dynamic'?ns:0),idx=sparseIndex(a.ids,id);
 if(idx<0)return {class_id:0,branch:part,map_version:window.BOUNDARY_INDEX.version,reason:'当前原始观测帧没有足够可见贡献，保留 unknown'};
 const radius=Number($('semanticBoundary')?.value||0),col=window.BOUNDARY_INDEX.radii.indexOf(radius),fraction=col<0?0:a.boundary_fraction[idx*3+col],threshold=Number($('semanticBoundaryMass')?.value||.2),abstain=col>=0&&fraction>=threshold,base=a.class_id[idx];
 const dep=S.gpuDepthFrames?.get(S.raw),di=dep&&variant==='original'?sparseIndex(dep.ids,id):-1,rms=di>=0?dep.rms_depth_m[di]:null;
 return {class_id:abstain?0:base,base_class_id:base,class_name:GPU_SEMANTIC_NAMES[abstain?0:base],branch:part,motion_model:part==='static'?'world_static':newer?'rigid_ActorTrack':'time_weighted_per_Gaussian_motion',evidence_state:abstain?'boundary_abstention':base===0?'insufficient_or_mixed':'prediction_and_visible_contribution',map_version:window.BOUNDARY_INDEX.version,row_namespace:variant==='actor10k'&&part==='dynamic'?'V43 measured Actor-local':'V42 original static/dynamic',contribution_alpha_mass:a.mass[idx],semantic_evidence_purity:a.purity[idx],target_mask_fraction:a.target_fraction[idx],vehicle_semantic_fraction:a.vehicle_fraction[idx],boundary_px:radius,boundary_contribution_fraction:fraction,boundary_abstained:abstain,assigned_instance_id:part==='dynamic'?D.target_track:'unknown / 世界静态混合支撑',assignment_source:part==='static'?'原static批次，车辆外观候选未确认实例':newer?'核验实测目标点重建资产':'原dynamic批次标签，不等于逐行物理归属证明',target_LiDAR_RMS_depth_m:rms,depth_status:rms===null?'missing':rms<=.5?'compatible':rms>1?'incompatible':'uncertain',conflict_candidate:part==='static'&&a.target_fraction[idx]>.05&&rms!==null&&rms<=.5,vehicle_appearance_candidate:part==='static'&&a.target_fraction[idx]>.05&&a.vehicle_fraction[idx]>.3,reason:abstain?'可见贡献进入车辆轮廓内外带，规则暂缓分类；原owner不变':base===0?'二维置信或高斯跨类别混合支撑不足，保留unknown':'原始实拍语义与当前字段无编辑观测alpha*T支持；随行携带到编辑，不是编辑后新权重或语义GT'};
}
