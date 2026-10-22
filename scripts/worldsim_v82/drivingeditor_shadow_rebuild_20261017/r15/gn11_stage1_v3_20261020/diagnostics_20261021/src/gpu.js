'use strict';
(()=>{
 const $=id=>document.getElementById(id);let generation=0,cf=null;
 const section=document.createElement('section');section.className='panel';section.id='counterfactual';
 section.innerHTML=`<div class="panel-title"><span class="number">GPU</span><h2>原生CUDA归因与反事实重渲染</h2></div><div class="subtools"><button id="queryGPU">实时CUDA像素归因</button><button id="selectTop">选择颜色Top20普通背景</button><button id="clearSelected">清空选择</button><button id="renderCF">完整重渲染选中集合</button><button id="saveCF" disabled>保存干预证据</button></div><div id="cfState" class="notice">读取预计算GPU证据…</div><div id="cfImages" class="cf-images"></div><p id="cfMetrics" class="note"></p><p id="cfTemporal" class="note"></p><p class="note">实时按钮需要运行 start_gpu.ps1，单槽使用远端GPU；首次加载可能约一分钟。仅冻结30K前向，模型不改动，优化步为0。离线预计算结果无需GPU。删除道路后应露出什么颜色没有隐藏纹理GT，不自动判定干预改善。</p>`;
 document.querySelector('.workspace').after(section);
 const nativeView=document.createElement('div');nativeView.className='panel';nativeView.innerHTML='<div class="panel-title"><h3>全模型原生视角渲染</h3><button id="renderView">渲染三维窗口当前视角</button><label>离线视角<select id="cachedView"><option value="native">原相机 · 当前分支</option><option value="right1m">右移1m · 联合 / raw102</option><option value="orbit">full轨道视角 · 删除 / raw102</option></select></label></div><div id="viewState" class="note">点云只是几何预览；点击此按钮用全模型原生光栅化生成真实图像。预计算原相机五分支图像可离线查看。</div><img id="nativeViewImage" alt="完整480×320原生高斯渲染" hidden style="width:100%;max-width:720px;height:auto"><p class="note">非训练视角没有GT；新视角变化不能当重建质量通过。道路和动态独立视图不填环境；联合、删除与普通背景填同一环境。</p>';
 section.prepend(nativeView);
 const badge=document.querySelector('header .badge.warn');if(badge){badge.textContent='CUDA trace / 完整干预已接入';badge.className='badge good';}
 const oldNote=$('topRows').closest('.panel').querySelector('.notice.warn');if(oldNote)oldNote.textContent='原生CUDA归因读取实际排序／投影缓冲区，逐查询重放核验；可在下方完整重渲染选中集合。CPU参考保留为独立对照，不将贡献排行自动判为错误点。';
 $('rayMode').querySelector('option[value="ordinary"]').textContent='普通背景独立';
 const nativeContract=document.querySelector('.contract-grid').firstElementChild.querySelector('p');nativeContract.textContent='四组fresh stage1 30K / stage2 0，raw89—137，seed0。基础RGB、alpha、深度和道路组贡献来自既有R2；本轮新增冻结CUDA诊断与完整重渲染，优化步仍为0。';
 const reportLink=document.createElement('a');reportLink.href='GPU_REPORT.html';reportLink.textContent='GPU诊断报告 ↗';document.querySelector('nav').append(reportLink);
 document.querySelector('footer').textContent='Gaussian Evidence Lab · native CUDA diagnostic release · 科学质量仍待人审';
 const architecture=document.querySelector('.architecture');if(architecture){architecture.setAttribute('aria-label','冻结30K、相机与测量输入，经只读CPU与CUDA诊断、原生排序归因与完整重渲染，输出四组联动证据');for(const t of architecture.querySelectorAll('text')){if(t.textContent==='只读 CPU 适配器')t.textContent='只读 CPU / CUDA 适配器';if(t.textContent==='CPU 参考 / GPU 待采集')t.textContent='排序归因 / 完整干预重渲染';}}
 const contract=document.querySelector('.contract-grid');if(contract){const p=contract.lastElementChild.querySelector('p');p.textContent='已实际执行冻结30K原生缓冲区逐像素归因与选中集合完整重渲染。结果逐查询核验RGB／alpha／depth／T，并用独立one-hot通道核验Top20 ID的Tα。像素责任及固定渲染干预不等于真实纹理或科学质量通过。不训练、不调损失、不自动关机。';contract.lastElementChild.querySelector('h3').textContent='原生GPU证据与限制';}
 const state=()=>window.gsdiag.state,identity=()=>{const s=state();return `${s.arm}/${s.raw}/${s.x}/${s.y}/${s.rayMode}`;};
 const fmt=v=>v===null||v===undefined?'—':Number(v).toFixed(6);
 function showCF(value,label='完整原生重渲染'){
  cf=value;$('cfState').textContent=`${label} · ${value.arm} / raw${value.raw} / ${value.mode} · ${value.removed.length}个Gaussian · 固定检查点／相机／环境 · 非颜色相减。移除造成的图像变化不等于真实质量改善。`;
  $('cfImages').replaceChildren();
  for(const [key,title] of [['before','移除前'],['after','移除后'],['difference','绝对RGB差异 ×5'],['road_before','道路贡献 · 前'],['road_after','道路贡献 · 后']]){
   const card=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src=value.images[key];im.alt=title;cap.textContent=title;card.append(im,cap);$('cfImages').append(card);
  }
  const m=value.metrics;$('cfMetrics').textContent=`选中像素 RGB L1变化 ${fmt(m.selected_pixel_L1_change)}；道路贡献 ${fmt(m.selected_pixel_road_before)} → ${fmt(m.selected_pixel_road_after)}；α ${fmt(m.selected_pixel_alpha_before)} → ${fmt(m.selected_pixel_alpha_after)}。全图平均变化 ${fmt(m.mean_abs_RGB_change)}；21×21像素邻域外变化 ${fmt(m.outside_ROI_RGB_change)}；变化>1/255像素 ${m.changed_pixel_count}。邻域外变化仅是潜在误伤提示，未经正常区域GT验证。`;
  const temporal=value.temporal_controls||[];$('cfTemporal').textContent=temporal.length?'相同Gaussian集合跨帧干预：'+temporal.map(v=>`raw${v.raw} 全图Δ=${fmt(v.metrics.mean_abs_RGB_change)}，区域外Δ=${fmt(v.metrics.outside_ROI_RGB_change)}`).join('；')+'。同一屏幕坐标不等于同一世界表面。':'';
  $('saveCF').disabled=false;
  window.evidenceUI?.showCF(value);
 }
 async function selection(){
  const token=++generation;cf=null;$('cfImages').replaceChildren();$('cfMetrics').textContent='';$('cfTemporal').textContent='';$('saveCF').disabled=true;
  if($('scientificEvaluation'))$('scientificEvaluation').replaceChildren();
  if($('geometrySelection'))$('geometrySelection').textContent='';
  if($('geometryRows')){$('geometryRows').replaceChildren();$('geometrySummary').textContent='当前像素尚未载入带测量判别的归因；保持未知。';$('geometryDetail').textContent='';}
  $('cachedView').value='native';
  $('nativeViewImage').hidden=true;$('viewState').textContent='点击渲染当前三维视角；未用采样几何生成虚假外观。';
  $('cfState').textContent='勾选贡献表Gaussian后，点击完整重渲染。内置raw102三案例已从贡献表中选最多20个普通背景执行移除。';
  const s=state();if(window.gsdiag.pendingExperiment?.()){const text=window.gsdiag.pendingExperiment().message;$('cfState').textContent=text;$('viewState').textContent=text;return;}if(s.arm.startsWith('v4_')){for(const id of ['queryGPU','renderCF','renderView','selectTop'])$(id).disabled=true;$('cfState').textContent='V4：当前可看49帧原生输出、几何和深度；逐Gaussian trace与删除干预尚未导出。';$('viewState').textContent='V4原生结果见左侧二维通道；不调用旧V3/V1的GPU绑定。';return;}for(const id of ['queryGPU','renderCF','renderView','selectTop'])$(id).disabled=false;const key=`cf/${s.arm}/${s.raw}/${s.x}/${s.y}`,path=window.GSDATA?.gpu_index?.counterfactuals?.[key];
  if(path&&s.rayMode==='deleted_five'){try{const value=await window.gsdiag.load(key,path,false);if(token===generation)showCF(value,value.selection_kind==='geometry_conflict_candidates'?'已执行：当前射线3个几何冲突候选（不代表保证受益或当前复选集合）':'已执行预计算：贡献表内颜色较高的最多20个普通背景，不代表当前复选集合');}catch(e){if(token===generation)$('cfState').textContent='预计算结果缺失：'+e.message;}}
  const viewKey=`view/${s.arm}/${s.raw}/${s.rayMode}/native`,viewPath=window.GSDATA?.gpu_index?.views?.[viewKey];
  if(viewPath){try{const value=await window.gsdiag.load(viewKey,viewPath,false);if(token===generation)showView(value,'预计算原相机分支图像');}catch(e){if(token===generation)$('viewState').textContent=e.message;}}
 }
 function showView(value,label){if(window.gsdiag.pendingExperiment?.())return;$('nativeViewImage').src=value.image;$('nativeViewImage').hidden=false;const proof=value.full_frame_saved_replay_max_error!==null?`与原49帧整幅浮点重放最大误差 ${value.full_frame_saved_replay_max_error}`:value.view?'新视角，无GT':'原相机新分支；旧评价无独立分支基线';$('viewState').textContent=`${label} · ${value.arm} / raw${value.raw} / ${value.mode} · 全量${value.gaussian_count.toLocaleString()}个输入Gaussian，原生480×320。${proof}。${value.raw!==state().raw?'注意：此预计算视角与当前主时间条帧不同。':''}`;}
 async function rpc(action){
  if(state().arm.startsWith('v4_'))throw Error('V4逐Gaussian归因/反事实尚未接入；整幅结果请看二维通道。');if(window.gsdiag.pendingExperiment?.())throw Error('V4结果尚未导出，当前实验仅占位。');
  const s=state(),id=identity(),payload={action,arm:s.arm,raw:s.raw,x:s.x,y:s.y,mode:s.rayMode};
  if(action==='counterfactual'){
   payload.ids=[...s.checked];if(!payload.ids.length&&s.selected)payload.ids=[s.selected.id];
   if(!payload.ids.length)throw Error('请先勾选Gaussian或选择一个椭球。');
  }
  if(action==='render_view')payload.view=window.gsdiag.getView();
  if(location.protocol==='file:')throw Error('实时GPU诊断请运行 start_gpu.ps1；离线结果可以直接查看。');
  const r=await fetch('/api/gpu',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),v=await r.json();
  if(v.error)throw Error(v.error);if(!r.ok)throw Error('GPU诊断服务失败');
  if(identity()!==id)return;
  if(action==='trace'){window.gsdiag.showRay(v);window.gsdiag.draw3D();$('cfState').textContent='CUDA归因已完成；请勾选Gaussian后执行干预。';}
  else if(action==='render_view')showView(v,payload.view?'三维窗口视角真实渲染':'原相机真实渲染');
  else showCF(v,'当前所选集合的实时完整重渲染');
 }
 for(const [button,action] of [['queryGPU','trace'],['renderCF','counterfactual'],['renderView','render_view']])$(button).onclick=async()=>{const b=$(button);b.disabled=true;const target=action==='render_view'?'viewState':'cfState';$(target).textContent=action==='trace'?'正在读取原生排序缓冲区，核验CUDA贡献；只做前向…':'正在完整重新预处理／排序／合成；不是采样点云或颜色相减…';try{await rpc(action);}catch(e){$(target).textContent=e.message;}finally{b.disabled=false;}};
 $('selectTop').onclick=()=>{const s=state();if(!s.ray)return;s.checked.clear();[...s.ray.top].filter(v=>v.branch==='ordinary').sort((a,b)=>Math.hypot(...b.color_contribution)-Math.hypot(...a.color_contribution)).slice(0,20).forEach(v=>s.checked.add(v.id));window.gsdiag.rankRows();$('cfState').textContent=`已选择${s.checked.size}个颜色贡献较高的普通背景Gaussian；这只是诊断集合，不是错误点判决。`;};
 $('cachedView').onchange=async e=>{if(window.gsdiag.pendingExperiment?.()){$('nativeViewImage').hidden=true;$('viewState').textContent=window.gsdiag.pendingExperiment().message;return;}const s=state();if(s.arm.startsWith('v4_')){$('nativeViewImage').hidden=true;$('viewState').textContent='V4原生结果请看二维通道，不替用其他实验视角。';return;}const kind=e.target.value,key=kind==='native'?`view/${s.arm}/${s.raw}/${s.rayMode}/native`:kind==='right1m'?`view/${s.arm}/102/joint/right1m`:'view/full/102/deleted_five/orbit',path=window.GSDATA.gpu_index?.views?.[key];$('nativeViewImage').hidden=true;if(!path){$('viewState').textContent='该帧／分支没有预计算原生视角；可实时渲染，不用其他视角替代。';return;}try{const value=await window.gsdiag.load(key,path,false);showView(value,kind==='native'?'预计算原相机':'预计算真实新视角（注意独立raw／分支标签）');}catch(err){$('viewState').textContent=err.message;}};
 $('clearSelected').onclick=()=>{state().checked.clear();state().selected=null;window.gsdiag.rankRows();window.gsdiag.draw3D();};
 $('saveCF').onclick=()=>{if(!cf)return;const u=URL.createObjectURL(new Blob([JSON.stringify(cf,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download=`intervention_${cf.arm}_${cf.raw}_${cf.pixel.join('_')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),2000);};
 window.gpuDiag={onSelection:selection,showCF};
})();
