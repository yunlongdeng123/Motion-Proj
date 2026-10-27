// Read-only observational semantics. Never change renderer branch or owner.
const SEMANTIC_MAP={0:'unknown / 未取得类别证据',1:'vehicle / 车辆（当前仅目标车掩码）',2:'road / 道路',3:'sidewalk / 人行道',4:'building / 建筑',5:'vegetation / 植被',6:'sky / 天空',7:'pole-sign / 杆与标牌',8:'person-cycle / 行人与骑行者'};
async function ensureSemantic(){
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
 if(mode==='semantic_conflict')return e.conflict_candidate?'#ff4fd2':'#374658';
 if(!e.class_id)return e.branch==='static'?'#536d89':'#b6c0ce';
 return mode==='semantic'?'#4df3c4':e.branch==='static'?'#ff4fd2':'#42e8a2';
}
function semanticLegend(){
 const mode=$('cloudColor').value;if(!mode.startsWith('semantic'))return;
 $('semanticStatus').textContent='当前模式：'+($('semanticProfile').value==='projection'?'仅二维掩码中心投影（弱候选）':'掩码＋LiDAR 邻域深度＋尺度（候选）')+'。绿=vehicle×原动态分支；洋红=vehicle×原静态分支，提示目标表达跨分支；灰蓝=unknown×静态；灰白=unknown×动态。静态车辆可为停放车，不能只凭 vehicle×static 判错。当前只有已审阅目标车掩码，其他车辆/道路等尚无密集语义；框不生成语义。';
}
async function enableSemantics(){try{await ensureSemantic();draw3D();semanticLegend();}catch(e){$('semanticStatus').textContent='语义观测未加载：'+e.message;}}
