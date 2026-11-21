const D=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const option=(id,name)=>`<option value="${esc(id)}">${esc(name)}</option>`;
const stages=D.stages, frames=D.frames;
$('subtitle').textContent=D.subtitle;
$('status').textContent=D.status;
if(D.status_attention){$('status').style.borderColor='var(--orange)';$('status').style.background='#3b3025';}
$('stats').innerHTML=(D.stats||[]).map(x=>`<div class="stat"><span>${esc(x.name)}</span><strong>${esc(x.value)}</strong></div>`).join('');
$('frame').innerHTML=frames.map(r=>option(r,`raw ${r}`)).join('');$('frame').value=String(D.default_frame||110);
for(const id of ['left','right'])$(id).innerHTML=stages.map(s=>option(s.id,s.name)).join('');
$('left').value=stages[0].id;$('right').value=stages.at(-1).id;
$('protocol').innerHTML=stages.map(s=>`<tr><td>${esc(s.name)}</td><td>${esc(s.parameters)}</td><td>${esc(s.supervision)}</td><td>${esc(s.frozen)}</td></tr>`).join('');
$('method').textContent=D.method;
$('metricRole').innerHTML=(D.metric_roles||[]).map(x=>option(x.id,x.name)).join('');
$('metricNote').textContent=D.metric_note;
$('checks').innerHTML=(D.checks||[]).map(x=>`<div class="check"><strong class="${x.passed?'':'warning'}">${esc(x.name)} · ${x.passed?'通过':'待解决'}</strong><small>${esc(x.detail)}</small></div>`).join('');
$('findings').innerHTML=(D.findings||[]).map(x=>`<div class="check"><strong class="${x.attention?'warning':''}">${esc(x.name)}</strong><small>${esc(x.detail)}</small></div>`).join('');
$('proofs').textContent=JSON.stringify(D.proofs,null,2);$('config').textContent=JSON.stringify(D.config,null,2);
$('links').innerHTML=(D.links||[]).map(x=>`<a href="${esc(x.path)}">${esc(x.name)}</a>`).join('');
$('disclaimer').textContent=D.disclaimer?'免责声明：'+D.disclaimer:'';
let crop=false,native=false,generation=0;
function showImage(id,path,box,token){
 const el=$(id);el.replaceChildren();
 if(!path){el.innerHTML='<div class="empty">此帧无对应参考</div>';return;}
 const img=new Image();img.alt=id==='refVisual'?'对应监督参考':'实际三维渲染';
 img.onerror=()=>{if(token===generation)el.innerHTML='<div class="empty">图像加载失败</div>';};
 img.onload=()=>{
  if(token!==generation)return;
  if(crop&&box){const c=document.createElement('canvas');c.width=box[2]-box[0];c.height=box[3]-box[1];c.setAttribute('aria-label',img.alt+'局部');c.getContext('2d').drawImage(img,...box.slice(0,2),c.width,c.height,0,0,c.width,c.height);el.replaceChildren(c);}
  else el.replaceChildren(img);
 };img.src=path;
}
function renderMetrics(){
 const role=$('metricRole').value,raw=Number($('frame').value),cols=D.metric_columns||[];
 $('metricHead').innerHTML='<tr><th>阶段</th>'+cols.map(x=>`<th>${esc(x.name)}</th>`).join('')+'</tr>';
 const rows=(D.metrics||[]).filter(x=>Number(x.raw)===raw&&x.role===role);
 $('metricBody').innerHTML=rows.map(r=>'<tr><td>'+esc(stages.find(s=>s.id===r.stage)?.name||r.stage)+'</td>'+cols.map(c=>{let v=r[c.key];return '<td>'+esc(v==null?'—':typeof v==='number'?(c.integer?String(v):v.toFixed(c.digits??5)):v)+'</td>';}).join('')+'</tr>').join('')||`<tr><td colspan="${cols.length+1}">此帧未计算该域指标</td></tr>`;
}
function update(){
 const token=++generation,raw=$('frame').value,view=$('view').value,ref=D.references[raw]||{},box=D.crop_boxes?.[raw];
 const selected=$('reference').value,kind=selected==='auto'?((view==='joint_delete'||view==='road_only')&&ref.road_teacher?'road_teacher':'rgb'):selected;
 $('refTitle').textContent={road_teacher:'删除 a9 的二维道路参考',rgb:'原始真实观测',vehicle_mask:'a20 观测实例掩码'}[kind];
 $('refInfo').textContent=kind==='road_teacher'?'道路损失使用此目标；当前大足迹高斯仍有越界车辆贡献':kind==='vehicle_mask'?'由观测自动估计；被 a9 遮挡区域不等于车体不存在':'a20 车辆损失使用真实 RGB 与可见实例掩码';
 $('refCaption').textContent=kind==='road_teacher'?'道路教师只在合法道路区域提供监督':kind==='vehicle_mask'?'遮挡后的可见实例掩码':'1152 × 768 原始相机观测';
 showImage('refVisual',ref[kind],box,token);
 for(const side of ['left','right']){
  const s=stages.find(s=>s.id===$(side).value);$(side+'Title').textContent=s.name;$(side+'Info').textContent=s.budget;
  showImage(side+'Visual',s.images[raw]?.[view],box,token);
 }
 $('frameInfo').textContent=(D.frame_info?.[raw]||'')+(view==='vehicle_alpha_delete'?' · 白色表示 a20 在联合深度排序后的可见贡献':'')+(view==='road_alpha'?' · 白色表示道路高斯覆盖充分':'')+(crop?' · 按原生像素裁切；可切换原生像素按 1:1 查看':'');
 for(const id of ['refCard','leftCard','rightCard'])$(id).classList.toggle('native',native);
 renderMetrics();
}
for(const id of ['frame','view','reference','left','right'])$(id).addEventListener('change',update);
$('metricRole').addEventListener('change',renderMetrics);
$('crop').onclick=()=>{crop=!crop;$('crop').setAttribute('aria-pressed',crop);update();};
$('native').onclick=()=>{native=!native;$('native').setAttribute('aria-pressed',native);update();};
update();
