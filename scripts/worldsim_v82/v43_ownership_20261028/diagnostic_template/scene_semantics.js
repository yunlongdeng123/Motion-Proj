/* BOUNDARY_RUNTIME */
// Read-only observational semantics. Never change renderer branch or owner.
const SEMANTIC_MAP={0:'unknown / 未取得类别证据',1:'vehicle / 车辆（当前仅目标车掩码）',2:'road / 道路',3:'sidewalk / 人行道',4:'building / 建筑',5:'vegetation / 植被',6:'sky / 天空',7:'pole-sign / 杆与标牌',8:'person-cycle / 行人与骑行者'};
const GPU_SEMANTIC_NAMES=['unknown','road / 道路','sidewalk / 人行道','building / 建筑','wall / 墙','fence / 围栏','pole / 杆','traffic light / 信号灯','traffic sign / 标牌','vegetation / 植被','terrain / 地形','sky / 天空','person / 行人','rider / 骑行者','car / 小汽车','truck / 卡车','bus / 客车','train / 轨道车','motorcycle / 摩托车','bicycle / 自行车'];
const GPU_SEMANTIC_COLORS=['#c8cdd4','#804080','#f423e8','#7e7e7e','#66669c','#be9999','#b2b2b2','#faaa1e','#dcdc00','#6b8e23','#98fb98','#4682b4','#dc143c','#ff0000','#2de5aa','#3c42ba','#0084a0','#008064','#0000e6','#770b20'];
async function ensureGPUSemantic(){
 if(!window.GPU_SEMANTIC_INDEX)await loadScript('gpu_semantics/index.js');
 S.gpuSemanticFrames=S.gpuSemanticFrames||new Map();const raw=S.raw;
 if(!S.gpuSemanticFrames.has(raw)){
  const entry=window.GPU_SEMANTIC_INDEX.frames[raw];if(!entry)throw Error('该源帧没有 GPU 贡献证据');
  await loadScript(entry.path);const value=await unpack(window.GPU_SEMANTIC_FRAMES[raw]);
  S.gpuSemanticFrames.set(raw,value);if(S.gpuSemanticFrames.size>6)S.gpuSemanticFrames.delete(S.gpuSemanticFrames.keys().next().value);
 }
 if(window.GPU_SEMANTIC_INDEX.depth_available){
  if(!window.GPU_DEPTH_INDEX)await loadScript('gpu_semantics/depth_index.js');
  S.gpuDepthFrames=S.gpuDepthFrames||new Map();if(!S.gpuDepthFrames.has(raw)){await loadScript(window.GPU_DEPTH_INDEX.frames[raw]);S.gpuDepthFrames.set(raw,await unpack(window.GPU_DEPTH_FRAMES[raw]));if(S.gpuDepthFrames.size>6)S.gpuDepthFrames.delete(S.gpuDepthFrames.keys().next().value);}
 }
}
function sparseIndex(ids,id){let lo=0,hi=ids.length-1;while(lo<=hi){const mid=(lo+hi)>>1;if(ids[mid]===id)return mid;if(ids[mid]<id)lo=mid+1;else hi=mid-1;}return -1;}
function gpuSemanticRow(instance,j){
 const boundary=boundarySemanticRow(instance,j);if(boundary)return boundary;
 const part=instance.cloud==='static'?'static':'dynamic',a=S.gpuSemanticFrames?.get(S.raw),index=window.GPU_SEMANTIC_INDEX;
 if(instance.cloud.startsWith('v43:'))return {class_id:0,class_name:'unknown / 新资产逐行贡献语义尚未加载',branch:'dynamic',map_version:'measured-actor-seed-v1',instance_id:D.target_track,source_status:'原始实测点及已核验目标掩码；局部刚体结构组',reason:'新行空间，原30K unknown标签不复用；不是unknown学习结果'};
 if(!a||!index||!['static','dynamic'].includes(instance.cloud))return {class_id:0,branch:part,reason:'没有此资产的原始 GPU 贡献观测',map_version:'cityscapes19-plus-unknown-v1'};
 const row=S.clouds[instance.cloud].data.ids[j],id=row+(part==='dynamic'?index.branch_sizes[0]:0),found=sparseIndex(a.ids,id);
 if(found<0)return {class_id:0,branch:part,reason:'该源帧没有足够可见贡献；不能当非车辆',map_version:index.version};
 const k=a.class_id[found],target=a.target_fraction[found],dep=S.gpuDepthFrames?.get(S.raw),jdep=dep?sparseIndex(dep.ids,id):-1,rms=jdep>=0?dep.rms_depth_m[jdep]:null;
 return {class_id:k,class_name:GPU_SEMANTIC_NAMES[k],branch:part,map_version:index.version,contribution_alpha_mass:a.mass[found],semantic_evidence_purity:a.purity[found],target_mask_fraction:target,vehicle_semantic_fraction:a.vehicle_fraction[found],target_LiDAR_RMS_depth_m:rms,target_return_alpha_mass:jdep>=0?dep.return_mass[jdep]:null,depth_status:rms===null?'missing':rms<=.5?'compatible':rms>1?'incompatible':'uncertain',conflict_candidate:part==='static'&&target>.05&&rms!==null&&rms<=.5,reason:'原V42 30K、原相机原时刻的 alpha*T 证据；同像素实测深度只作支持，不是三维身份真值或当前编辑权重'};
}
function drawSemanticImage(){
 const a=S.gpuSemanticFrames?.get(S.raw);if(!a){empty($('renderCanvas'),'GPU 语义图未加载');return;}
 const x=$('renderCanvas').getContext('2d'),image=x.createImageData(W,H);
 for(let i=0;i<W*H;i++){const band=S.boundaryFrames?.get(S.raw)?.image,rad=Number($('semanticBoundary')?.value||0),bi=[3,5,8].indexOf(rad),unknown=band&&bi>=0&&band.bands[bi*W*H+i];const color=GPU_SEMANTIC_COLORS[unknown?0:a.semantic_image[i]],rgb=[1,3,5].map(k=>parseInt(color.slice(k,k+2),16));image.data.set([...rgb,255],i*4);}
 x.putImageData(image,0,0);cross(x,W,H);$('mapLegend').textContent='原始实拍的 SegFormer-B0 语义预测；置信门限0.65；标签不是 GT，也不随反事实编辑移动。';
}
async function ensureSemantic(){
 if(S.channel==='semantic_2d'||($('semanticProfile').value==='gpu'&&($('cloudColor').value.startsWith('semantic')||($('semanticAblation')?.value||'all')!=='all'))){await ensureGPUSemantic();await ensureBoundarySemantic();return;}
 if(!['semantic','semantic_branch','semantic_conflict'].includes($('cloudColor').value))return;
 if(!window.SEMANTIC_OBSERVATIONS)await loadScript('semantic_observations.js');
 S.semanticFrames=S.semanticFrames||new Map();
 const raw=S.raw;if(!S.semanticFrames.has(raw)){
  const a=await unpack(window.SEMANTIC_OBSERVATIONS[raw]);
  // Nearest observed return inside a configurable pixel neighborhood, not filled depth GT.
  const nearest=new Float32Array(W*H);nearest.fill(NaN);const distance=new Uint8Array(W*H);distance.fill(255);
  for(let i=0;i<W*H;i++)if(Number.isFinite(a.lidar_z[i])){const u=i%W,v=(i/W)|0;for(let dy=-5;dy<=5;dy++)for(let dx=-5;dx<=5;dx++){const d=dx*dx+dy*dy;if(d>25||u+dx<0||u+dx>=W||v+dy<0||v+dy>=H)continue;const q=i+dy*W+dx;if(d<distance[q]){distance[q]=d;nearest[q]=a.lidar_z[i];}}}
  S.semanticFrames.set(raw,{...a,nearest,distance});
 }
}
function semanticRow(instance,j){
 if(instance.cloud.startsWith('v43:'))return gpuSemanticRow(instance,j);
 if($('semanticProfile').value==='gpu')return gpuSemanticRow(instance,j);
 const part=instance.cloud==='static'?'static':'dynamic',a=S.semanticFrames?.get(S.raw);
 if(!a||instance.cloud.startsWith('foreign:'))return {class_id:0,branch:part,reason:'无此资产的语义观测，保留 unknown'};
 // Evaluate original geometry with original observed camera. Carry the evidence with the row into edits.
 const c=S.clouds[instance.cloud],p=cloudPoint(c.data,j,c.cycle,D.frames[S.raw].model_time),fr=D.frames[S.raw],M=fr.c2w,d=p.p.map((v,k)=>v-M[k][3]);
 const cam=[0,1,2].map(k=>d.reduce((s,v,n)=>s+v*M[n][k],0));
 if(cam[2]<=.1)return {class_id:0,branch:part,reason:'原始观测相机背后'};
 const K=fr.K,u=Math.floor(K[0]*cam[0]/cam[2]+K[2]),v=Math.floor(K[1]*cam[1]/cam[2]+K[3]);
 if(u<0||u>=W||v<0||v>=H)return {class_id:0,branch:part,reason:'原始观测视野外'};
 const i=v*W+u,mask=!!a.mask[i],sigma=Math.max(...c.data.scales.slice(j*3,j*3+3)),radius=Number($('semanticRadius').value),tol=Number($('semanticDepth').value),maxSigma=Number($('semanticSigma').value),profile=$('semanticProfile').value;
 const depth=radius===0?a.lidar_z[i]:(a.distance[i]<=radius*radius?a.nearest[i]:NaN),depthOK=Number.isFinite(depth)&&Math.abs(cam[2]-depth)<=tol;
 const accepted=mask&&(profile==='projection'||(sigma<=maxSigma&&depthOK));
 return {class_id:accepted?1:0,branch:part,mask_hit:mask,depth_consistent:depthOK,depth_delta_m:Number.isFinite(depth)?cam[2]-depth:null,max_sigma_m:sigma,profile,reason:accepted?(profile==='projection'?'目标车掩码中心投影；遮挡与深度未核验':'掩码与稀疏 LiDAR 深度候选一致；贡献仍待 GPU 核验'):(mask?'掩码投影命中但深度/尺度证据不足':'无此点类别证据；不自动归背景'),conflict_candidate:accepted&&part==='static'};
}
function semanticColor(e,mode){
 if(mode==='semantic_unknown_reason'){if(e.class_id)return '#24384a';if(e.static_vehicle_quarantined)return '#ffa64d';if(e.boundary_abstained)return '#f5dc62';return e.base_class_id===0?'#c8cdd4':'#77b7ff';}
 if(mode==='semantic_conflict'){if(e.conflict_candidate)return '#ff4fd2';if(e.branch==='static'&&e.target_mask_fraction>.05)return e.depth_status==='incompatible'?'#56c7e8':'#f0b64b';return '#374658';}
 if(e.map_version){if(!e.class_id)return '#c8cdd4';const c=GPU_SEMANTIC_COLORS[e.class_id];if(mode==='semantic')return c;const rgb=[1,3,5].map(k=>parseInt(c.slice(k,k+2),16)),factor=e.branch==='static'?.65:1;return `rgb(${rgb.map(v=>Math.round(v*factor)).join(',')})`;}
 if(!e.class_id)return '#c8cdd4';
 return mode==='semantic'?'#4df3c4':e.branch==='static'?'#ff4fd2':'#42e8a2';
}
function semanticLegend(){
 const mode=$('cloudColor').value;if(!mode.startsWith('semantic'))return;
 if($('semanticProfile').value==='gpu'){$('semanticStatus').textContent='GPU map v2（与旧 CPU map 编码分开）：'+GPU_SEMANTIC_NAMES.map((n,i)=>i+' '+n).join('；')+'。联合着色：同类别静态较暗、动态较亮；浅灰 unknown；绿色 car，静态car为深绿、动态car为亮绿。同一高斯可以是car语义＋原static分支，不能把分支当语义。目标支持模式：洋红=静态参与目标且实测深度相容（候选）；青=与实测车深差>1m；黄=缺测/未定，不能整颗直接删。证据源为原V42 30K，不是当前编辑重新计算的权重；原 owner 不变。';return;}
 $('semanticStatus').textContent='当前模式：'+($('semanticProfile').value==='projection'?'仅二维掩码中心投影（弱候选）':'掩码＋LiDAR 邻域深度＋尺度（候选）')+'。绿=vehicle×原动态分支；洋红=vehicle×原静态分支，提示目标表达跨分支；灰蓝=unknown×静态；灰白=unknown×动态。静态车辆可为停放车，不能只凭 vehicle×static 判错。当前只有已审阅目标车掩码，其他车辆/道路等尚无密集语义；框不生成语义。';
}
async function enableSemantics(){try{if(($('semanticAblation')?.value||'all')!=='all')$('semanticProfile').value='gpu';await ensureSemantic();S.selected=null;if(S.channel==='semantic_2d')drawSemanticImage();draw3D();semanticLegend();}catch(e){$('semanticStatus').textContent='语义观测未加载：'+e.message;}}
