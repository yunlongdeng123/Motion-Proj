const D=__DATA__,C=D.catalog,E=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=4)=>x==null?'—':Number(x).toFixed(n), models=new Map(C.models.map(m=>[m.id,m])),batches=new Map(C.batches.map(b=>[b.id,b]));
let detail=false,batch='all';
function table(id,heads,rows){E(id).innerHTML='<tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')}
function selected(){return [E('left').value,E('right').value].map(id=>models.get(id))}
function available(){return C.models.filter(m=>batch==='all'||m.batch===batch)}
function modelName(id){return models.get(id)?.name??id}
function selectOptions(){return C.batches.filter(b=>batch==='all'||b.id===batch).map(b=>`<optgroup label="${esc(b.label)}">${C.models.filter(m=>m.batch===b.id).map(m=>`<option value="${esc(m.id)}">${esc(m.name)}</option>`).join('')}</optgroup>`).join('')}
function box(model,isTeacher=false){
 const raw=E('raw').value,region=E('region').value,view=E('view').value,b=batches.get(model.batch);
 const teacher=b.teacher,refRegion=isTeacher&&model.original_id.endsWith('IDEAL_ORACLE')?'IDEAL':region;
 const f=isTeacher?teacher.frames[raw]:model.frames[raw];
 let path=isTeacher?f?.[refRegion]:f?.[view];
 const crop=isTeacher?f?.[refRegion+'_crop']:f?.[region+'_'+view+'_crop'];
 const useCrop=detail&&Boolean(crop);if(useCrop)path=crop;
 const role=isTeacher?(teacher.roles?.[raw]?.[refRegion]??'teacher'):null;
 const title=isTeacher?(role==='observed'?'原始观测参考':/ideal|synthetic/i.test(role)?'合成控制参考':region==='A'?'真实道路观测':'冻结二维教师'):model.name;
 const subtitle=isTeacher?b.label+' · 绑定右侧实验':b.label+' · '+String(model.budget??model.original_id);
 if(!path)return `<figure><figcaption><strong>${esc(title)}</strong><small>${esc(subtitle)}</small></figcaption><p class="caption">该实验没有此视角／区域的${isTeacher?'参考':'图像'}。</p></figure>`;
 const size=isTeacher?null:model.frame_sizes?.[raw]?.[useCrop?region+'_'+view+'_crop':view];
 const caption=useCrop?'原生裁片，不插值':(size?size.join('×')+' 报告图像':'本组原始参考图');
 return `<figure><figcaption><strong>${esc(title)}</strong><small>${esc(subtitle)}</small></figcaption><div class="imageport ${useCrop?'detail':''}"><a href="${esc(path)}" target="_blank"><img src="${esc(path)}" alt="${esc(title)}"></a></div><div class="caption">${esc(caption)}</div></figure>`;
}
function matching(row,chosen){const raw=+E('raw').value;return chosen.some(m=>m.id===row.id)&&row.source===E('region').value&&(row.raw==null?[86,110].includes(raw):+row.raw===raw)&&(row.scale==null||+row.scale===+E('scale').value)}
function update(){
 const chosen=selected(),right=chosen[1],b=batches.get(right.batch);
 E('gallery').innerHTML=[box(right,true),...chosen.map(m=>box(m))].join('');
 const cross=chosen[0].batch!==right.batch;
 E('scope').textContent=cross?'跨实验组图像对照；教师绑定右侧实验。评价域与分辨率不同的指标不直接排名。':b.scope;
 E('title').textContent=batch==='all'?'全部实验对照':batches.get(batch).label;
 const rows=C.rows.filter(r=>matching(r,chosen));
 table('metrics',['实验／阶段','评价域','完整域预乘MAE ↓','域内颜色MAE ↓','HF对应度 ↑','HF能量比 → 1','边缘F1 ↑','评分域 / HF核心'],rows.map(r=>[modelName(r.id),[r.raw==null?'86/110合并':'raw'+r.raw,r.split??'',r.domain??''].filter(Boolean).join(' · '),fmt(r.full_premult_RGB_MAE,5),fmt(r.RGB_MAE,5),fmt(r.HF_cosine,3),fmt(r.HF_energy_ratio,2),fmt(r.edge_F1,3),`${r.stable_score_n??r.score_n??'—'} / ${r.core_n??'—'}`]));
 E('metric-domain-note').textContent=rows.length?'沿用各实验原始统计。完整域 |P−T| 与稳定域颜色误差分列；历史合并指标及不同训练划分保留原口径。':'此选择没有对应评价数据；不填入其他实验组的指标。';
 const gapRows=C.gaps.filter(r=>chosen.some(m=>m.id===r.id)&&+r.raw===+E('raw').value&&r.source===E('region').value);
 table('gaps',['实验／阶段','完整编辑域','认证道路域','认证无支持','认证薄覆盖','未认证编辑域','完整域无支持','完整域薄覆盖','高覆盖暗色'],gapRows.map(r=>[modelName(r.id),r.denominator??'—',r.certified_road_n??'—',r.certified_road_no_support_n??'—',r.certified_road_low_alpha_n??'—',r.uncertified_editing_n??'—',r.no_support_n??'—',r.low_alpha_n??'—',r.high_alpha_dark_n??'—']));
 E('gap-note').textContent=gapRows.length?'无原始认证或联合可见贡献数据的项目保持缺省；未认证编辑域不等于确认道路。':'该组未提供同口径覆盖统计；不以其他轮次结果替代。';
 E('picture-note').textContent=E('view').value==='alpha'?'仅显示本组实际导出的透明度图；无透明度图的历史实验保持缺省。':'按各实验原始图像显示；原生细节使用该组已有裁片。教师与观测参考不跨实验组回退。';
 const nr=C.newly_covered_rows.filter(r=>matching(r,chosen));
 table('newly-covered',['实验／阶段','新增覆盖','认证道路','未认证编辑域','颜色MAE','HF对应度','HF能量比','HF核心'],nr.map(r=>[modelName(r.id),r.newly_covered_n,r.newly_covered_certified_n,r.newly_covered_uncertified_editing_n,fmt(r.RGB_MAE,5),fmt(r.HF_cosine,3),fmt(r.HF_energy_ratio,2),r.core_n]));
 E('newly-covered').closest('details').hidden=!nr.length;
 const geometry=b.geometry_support?.[E('raw').value];
 E('geometry-view').innerHTML=geometry?`<a href="${esc(geometry)}" target="_blank"><img src="${esc(geometry)}" style="display:block;width:100%;height:auto" alt="该实验道路表面认证"></a>`:'此实验／视角未导出道路表面认证示意。';
}
function registry(){const visible=available();E('registry-summary').textContent='实验记录与参数（'+visible.length+'条）';table('registry',['实验组','实验／阶段','原始ID','预算','渲染／颜色读取','总行数','活动行数','启用旧锚点','真实新增几何','复用既有新增点'],visible.map(m=>{const p=m.point_counts??{};return[batches.get(m.batch).label,m.name,m.original_id,m.budget,m.renderer??'原报告',p.total??p.total_rows??'—',p.active??p.active_rows??'—',p.activated_existing??'—',p.appended_new_geometry??'—',p.reuse_existing_newgroup??'—']}))}
function configure(pair){for(const id of ['left','right'])E(id).innerHTML=selectOptions();const visible=available();const target=pair??(batch==='all'?C.default_pair:batches.get(batch).defaults);E('left').value=target[0];E('right').value=target[1];if(!E('left').value)E('left').value=visible[0].id;if(!E('right').value)E('right').value=visible[visible.length-1].id;registry();update()}
function preset(key){batch=key==='all'?'all':'residual';E('batch').value=batch;const names=key==='geometry'?['activated','coverage']:key==='material'?['coverage','appearance']:['parent','appearance'];configure(key==='all'?C.default_pair:names.map(n=>'residual::'+D.stages[n]));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.preset===key))}
function renderEvidence(){const labels={'report.md':'本轮技术记录','metrics.json':'完整评价数据','parent_freeze_proof.json':'父资产冻结核验','diagnostic_summary.json':'颜色诊断汇总','numeric_receipt.json':'同目标求解记录','mask_inclusion_audit.json':'实际监督覆盖核验'};const seen=new Set(),top=[],rest=[];for(const x of D.evidence){if(labels[x.name]&&!seen.has(x.name)){seen.add(x.name);top.push({...x,name:labels[x.name]})}else rest.push(x)}const links=xs=>xs.map(x=>`<a href="${esc(x.path)}">${esc(x.name)}</a>`).join(' · ');E('evidence').innerHTML=links(top)+(rest.length?`<details><summary>完整溯源文件（${rest.length}项）</summary>${links(rest)}</details>`:'')}

E('pairs').hidden=false;
E('pairs').insertAdjacentHTML('afterbegin','<label>实验组<select id="batch"><option value="all">全部实验（55条）</option>'+C.batches.map(b=>`<option value="${esc(b.id)}">${esc(b.label)}（${b.count}条）</option>`).join('')+'</select></label>');
E('batch').onchange=()=>{batch=E('batch').value;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',batch==='all'&&b.dataset.preset==='all'));configure()};
for(const id of ['raw','view','region','scale','left','right'])E(id).onchange=update;
for(const id of ['full','detail'])E(id).onclick=()=>{detail=id==='detail';E('full').classList.toggle('active',!detail);E('detail').classList.toggle('active',detail);update()};
document.querySelector('nav').insertAdjacentHTML('afterbegin','<button data-preset="all">全部实验</button>');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>preset(b.dataset.preset));
for(const b of document.querySelectorAll('nav button'))if(b.dataset.preset!=='all')b.textContent={final:'最新轮：起点→终态',geometry:'最新轮：锚点→覆盖',material:'最新轮：覆盖→外观'}[b.dataset.preset];
E('metrics').parentElement.insertAdjacentHTML('beforebegin','<p id="metric-domain-note" class="muted"></p>');
const registryWrap=E('registry').parentElement,registryDetails=document.createElement('details');registryWrap.replaceWith(registryDetails);registryDetails.innerHTML='<summary id="registry-summary"></summary>';registryDetails.appendChild(registryWrap);
document.title='道路材质 · 全部实验统一对照';document.querySelector('h1').textContent='道路材质与覆盖 · 全部实验对照';
document.querySelector('main > small').textContent='Scene065 · 8个实验组 · 55条实验／阶段记录';
E('geometry-counts').textContent='保留原三轮30条、1152教师重训15条、两轮增密各5条；数值诊断另列。最新两轮真实新增7,239个高斯，启用3,121个既有锚点。';
const heading=[...document.querySelectorAll('h2')].find(e=>e.textContent==='统一分辨率评价');if(heading)heading.textContent='各实验原始评价';
const graph=E('registry').closest('section').querySelector('h2');if(graph)graph.textContent='组件与完整实验记录';
renderEvidence();preset('all');
if(D.cumulative_baseline?.length){E('title').closest('section').insertAdjacentHTML('beforebegin','<section><h2>最新两轮累计覆盖</h2><div class="scroll"><table id="cumulative"></table></div></section>');table('cumulative',['帧 / 区域','认证道路分母','增密前低覆盖','最终低覆盖','最终充分覆盖'],D.cumulative_baseline.map(b=>{const a=D.gaps.find(r=>r.id===D.stages.appearance&&r.raw===b.raw&&r.source===b.source),ok=a&&a.certified_road_n===b.certified_road_n;return[b.raw+' / '+({C:'孔洞',S:'去影区'}[b.source]??b.source),b.certified_road_n,b.certified_road_low_alpha_total_n,ok?a.certified_road_low_alpha_total_n:'—',ok?fmt(100*(1-a.certified_road_low_alpha_total_n/a.certified_road_n),2)+'%':'—']}))}
if(D.color_diagnostic){const q=D.color_diagnostic;E('evidence').insertAdjacentHTML('beforebegin','<details><summary>独立数值诊断：颜色残差与同目标核验</summary><p>'+esc(q.finding)+'</p><div class="scroll"><table id="color-diagnostic"></table></div><p class="muted">'+esc(q.limitation)+'</p></details>');table('color-diagnostic',['帧 / 孔洞域','新增HF核心','正式颜色MAE','区间放松下界','诊断颜色MAE','正式HF对应度','诊断HF对应度'],q.rows.map(r=>[r.raw,r.core_n,fmt(r.formal_RGB_MAE,5),fmt(r.box_floor_RGB_MAE,5),fmt(r.numeric_RGB_MAE,5),fmt(r.formal_HF_cosine,3),fmt(r.numeric_HF_cosine,3)]))}

for(const p of document.querySelectorAll('main > p')){if(p.textContent.startsWith('教师：'))p.textContent='教师、训练分辨率与评价域按所选实验组绑定；86／110用于拟合，134／179／198用于跨视角诊断。';else if(p.textContent.startsWith('优先启用'))p.textContent='在同一页面选择原三轮、1152教师重训及两轮局部增密，比较道路、联合渲染与原生裁片。'}
const note=document.querySelector('#metrics').closest('section').querySelector('h2+ p.muted');if(note)note.textContent='道路层指标沿用原报告；联合渲染用于观察遮挡与残影。高频能量须结合对应度、边缘及裁片判断。';

if(!document.getElementById('vehicle-road-report-entry')){document.querySelector('main').insertAdjacentHTML('afterbegin','<p id="vehicle-road-report-entry" style="padding:12px 16px;border:1px solid #38546f;border-radius:10px"><a href="../vehicle_road_joint_20261121/index.html">车辆与道路联合恢复 · 删除 a9 / 补全 a20</a> · <a href="../vehicle_road_joint_20261121/architecture.svg">更新后的四模块架构图</a></p>');}
