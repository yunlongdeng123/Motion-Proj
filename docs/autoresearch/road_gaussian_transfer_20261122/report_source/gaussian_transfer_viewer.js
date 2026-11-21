/* 相似路块推断：来源证据与实际高斯贡献分开显示。 */
(() => {
  const group = C.batches.find(item => item.id === 'gaussian_transfer');
  if (!group || E('gaussian-transfer-panel')) return;
  const file = value => typeof value === 'string' ? value : value?.rgb ?? value?.path ?? value?.preview ?? null;
  const labels = {same:'可信实拍支持 · 输入支持',material:'迁入前参考材质',transfer:'迁入高斯实际贡献',effective:'最终来源分层'};
  const css = document.createElement('style');
  css.textContent = `.gt-panel{border-top:1px solid var(--line);padding-top:18px;margin:20px 0}.gt-arch{display:block;width:100%;min-width:900px}.gt-story{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.gt-story figcaption{min-height:0}.gt-story .imageport{aspect-ratio:auto;height:260px}.gt-story .imageport img{width:100%;height:100%;object-fit:contain}.gt-story .imageport.detail{height:270px}.gt-story .imageport.detail img{width:auto;height:auto;object-fit:initial;max-width:none}.gt-layer{max-width:900px}.gt-legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin:10px 0}.gt-legend i{display:inline-block;width:10px;height:10px;margin-right:6px}.gt-table{margin:15px 0}@media(max-width:1050px){.gt-story{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.gt-story{grid-template-columns:1fr}}`;
  document.head.appendChild(css);
  const panel = document.createElement('div');panel.id='gaussian-transfer-panel';panel.className='gt-panel';
  panel.innerHTML=`<h2>相似路面高斯迁移</h2><p id="gt-outcome" class="note"></p><p id="gt-summary" class="muted"></p><div class="scroll"><img class="gt-arch" src="${esc(group.architecture)}" alt="实拍与源高斯块经过保距路块配准、目标替换及局部接缝拟合，输出持久三维道路资产与五帧渲染"></div>
    <div class="controls"><label>参考源块<select id="gt-patch"></select></label><label>参考帧<select id="gt-source-raw"></select></label></div><div id="gt-story" class="gt-story"></div><h2 style="margin-top:22px">阶段与场景对照</h2><div id="gt-main-comparison"></div>
    <div class="scroll gt-table"><table id="gt-source-quality"></table></div>
    <div class="controls"><label>来源 / 贡献<select id="gt-layer">${Object.entries(labels).map(([key,label])=>`<option value="${key}">${label}</option>`).join('')}</select></label></div>
    <div id="gt-layer-legend" class="gt-legend"><span><i style="background:#65d8ab"></i>可信实拍候选（输入）</span><span><i style="background:#79b9ff"></i>迁入实际贡献</span><span><i style="background:#9aabbc"></i>继承原资产</span></div><div id="gt-layer-image" class="gt-layer"></div>
    <h2>覆盖与保留车辆</h2><div class="scroll"><table id="gt-coverage"></table></div><div class="scroll gt-table"><table id="gt-protection"></table></div>
    <div id="gt-boundary-wrap"><h2>真实可见道路边界与接缝</h2><div class="scroll"><table id="gt-boundary"></table></div></div>
    <div id="gt-preservation-wrap"><h2>源纹理保持</h2><div class="scroll"><table id="gt-preservation"></table></div></div>
    <div id="gt-constraints-wrap"><h2>参数约束核验</h2><div class="scroll"><table id="gt-constraints"></table></div></div>
    <div id="gt-visual-wrap"><h2>实际残余</h2><p id="gt-visual-finding"></p><details><summary>五帧拟合前后原生对照</summary><div id="gt-visual-image"></div></details></div>
    <div id="gt-resources-wrap"><h2>计算记录</h2><p class="muted">F为实际光栅前向；B依据执行代码与记录推导。</p><div class="scroll"><table id="gt-resources"></table></div></div>
    <details><summary>来源、变换与验收记录</summary><div id="gt-evidence"></div></details>`;
  E('gallery').before(panel);
  E('gt-outcome').textContent=group.diagnostics?.outcome??'';E('gt-outcome').hidden=!group.diagnostics?.outcome;
  E('gt-summary').textContent=[group.summary?.target_grid_area_m2==null?'':`请求道路表面 ${fmt(group.summary.target_grid_area_m2,2)} m²`,group.summary?.keep_same_position_count==null?'':`目标域保留同位置实拍 ${group.summary.keep_same_position_count} 个输入支持样本`,'本轮新增扩散调用 0'].filter(Boolean).join(' · ');
  const patches=group.source_patches??{};
  E('gt-patch').innerHTML=Object.entries(patches).map(([id,patch])=>`<option value="${esc(id)}">${esc(patch.label??id)}${patch.role==='primary'?' · 主参考':' · 核验对照'}</option>`).join('');
  if(patches[group.summary?.primary_source_patch])E('gt-patch').value=group.summary.primary_source_patch;
  function sourceOptions(){const frames=patches[E('gt-patch').value]?.frames??{},previous=E('gt-source-raw').value;E('gt-source-raw').innerHTML=Object.keys(frames).map(raw=>`<option value="${esc(raw)}">${esc(raw)}</option>`).join('');E('gt-source-raw').value=frames[previous]?previous:frames[E('raw').value]?E('raw').value:Object.keys(frames)[0]??'';}
  sourceOptions();
  function figure(title,path,subtitle='',extra=''){if(!path)return `<figure><figcaption><strong>${esc(title)}</strong><small>${esc(subtitle)}</small></figcaption><p class="caption">此阶段或视角尚未导出。</p></figure>`;return `<figure><figcaption><strong>${esc(title)}</strong><small>${esc(subtitle)}</small></figcaption><div class="imageport ${detail?'detail':''}"><a href="${esc(path)}" target="_blank"><img src="${esc(path)}" alt="${esc(title)}"></a></div>${extra?`<div class="caption">${extra}</div>`:''}</figure>`;}
  const stageName=id=>models.get(id?.startsWith('gaussian_transfer::')?id:'gaussian_transfer::'+id)?.name??id;
  function stageImage(stage){const model=models.get('gaussian_transfer::'+stage),frame=model?.frames?.[E('raw').value];return frame?.[E('region').value+'_'+E('view').value+'_crop']??frame?.[E('view').value];}
  function addPreset(text,pair,key){const button=document.createElement('button');button.textContent=text;button.dataset.preset=key;button.onclick=()=>{batch='gaussian_transfer';E('batch').value=batch;configure(pair);document.querySelectorAll('nav button').forEach(item=>item.classList.toggle('active',item===button));};document.querySelector('nav').appendChild(button);}
  addPreset('迁移：起点→零步',group.defaults,'gaussian-transfer-zero');
  if(models.has('gaussian_transfer::appearance'))addPreset('迁移：零步→接缝',['gaussian_transfer::transfer','gaussian_transfer::appearance'],'gaussian-transfer-fit');
  for(const button of document.querySelectorAll('nav button'))button.textContent=button.textContent.replace('最新轮：','增密②：');
  E('batch').querySelector('option[value="all"]').textContent=`全部实验（${C.models.length}条）`;
  document.querySelector('main > small').textContent=`Scene065 · ${C.batches.length}个实验组 · ${C.models.length}条实验／阶段记录`;
  const introduction=[...document.querySelectorAll('main > p')].find(item=>item.textContent.startsWith('在同一页面选择'));if(introduction)introduction.textContent='在同一页面比较历次道路实验、相似路面高斯迁移与局部接缝拟合。';
  const referenceNote=[...document.querySelectorAll('main > p')].find(item=>item.textContent.startsWith('教师、训练分辨率'));if(referenceNote)referenceNote.textContent='各组参考、分辨率与评价口径随实验记录保留；86／110用于建模，134／179／198用于跨视角检查。';
  for(const heading of document.querySelectorAll('h2'))if(heading.textContent==='最新两轮累计覆盖')heading.textContent='两轮历史增密累计覆盖';
  E('geometry-counts').textContent=`原55条记录保留；新增${group.count}个高斯迁移阶段。`+E('geometry-counts').textContent.replace('最新两轮','历史两轮增密');
  const oldHeading=[...document.querySelectorAll('h2')].find(item=>item.textContent==='各实验原始评价');
  const oldIntro=oldHeading?.nextElementSibling;
  const oldSummary=E('evidence')?.closest('details')?.querySelector(':scope > summary');if(oldSummary)oldSummary.textContent='道路增密② · 历史协议与来源';
  const footer=document.querySelector('footer');if(footer)footer.textContent='原实验、源块证据与持久化记录已保留。';
  if(D.gaussian_transfer_disclaimer){const paragraph=document.createElement('p');paragraph.id='gt-disclaimer';paragraph.className='muted';paragraph.textContent='免责声明：'+D.gaussian_transfer_disclaimer;document.querySelector('main').appendChild(paragraph);}
  function render(){
    const chosen=selected(),active=chosen.some(item=>item.batch==='gaussian_transfer');panel.hidden=!active;
    const onlyNew=chosen.every(item=>item.batch==='gaussian_transfer');
    E('metrics').parentElement.hidden=onlyNew;if(oldHeading)oldHeading.hidden=onlyNew;if(oldIntro?.tagName==='P')oldIntro.hidden=onlyNew;if(E('metric-domain-note'))E('metric-domain-note').hidden=onlyNew;
    if(!active){panel.after(E('gallery'),E('picture-note'));return;}
    E('gt-main-comparison').append(E('gallery'),E('picture-note'));
    const patch=patches[E('gt-patch').value]??{},sourceRaw=E('gt-source-raw').value,source=patch.frames?.[sourceRaw]??{};
    const context=file(source.full_road),sourceCaption=`${patch.label??E('gt-patch').value} · raw${sourceRaw}`,sourceGaussianCaption=sourceCaption+' · 含核验边缘，仅核心参与迁移';
    E('gt-story').innerHTML=[figure('源实拍',file(source.observation??source.photo),sourceCaption),figure('源块原高斯纹理',file(source.isolated??source.gaussians),sourceGaussianCaption,context?`<a href="${esc(context)}" target="_blank">原道路层上下文</a>`:''),figure('目标零步迁移',stageImage('transfer'),`raw${E('raw').value} · ${E('region').value} · 0更新`),figure(models.has('gaussian_transfer::appearance')?'局部接缝拟合':'接缝拟合 · 待执行',stageImage('appearance'),`raw${E('raw').value} · ${E('region').value}`)].join('');
    if(chosen[1].batch==='gaussian_transfer'){const reference=E('gallery').querySelector('figure');if(reference)reference.outerHTML=figure('相似源块 · 原高斯',file(source.isolated??source.gaussians),sourceGaussianCaption);E('picture-note').textContent='当前参考为相似位置源块；parent、零步迁移与接缝拟合均使用实际高斯渲染。';}
    const quality=group.diagnostics?.source_quality_rows??[];
    table('gt-source-quality',['源块 / 帧','原照对照像素','RGB MAE','HF对应度','充分覆盖'],quality.map(row=>[`${String(row.patch).replace(/^patch_/,'源块')} / ${row.raw}`,row.score_n,fmt(row.RGB_MAE,5),fmt(row.HF_cosine,3),row.alpha95_fraction==null?'—':fmt(100*row.alpha95_fraction,2)+'%']));
    const raw=+E('raw').value,region=E('region').value;
    const gap=C.gaps.filter(row=>row.batch==='gaussian_transfer'&&+row.raw===raw&&row.source===region);
    table('gt-coverage',['阶段','固定编辑域','认证道路','无支持','薄覆盖','充分覆盖'],gap.map(row=>[stageName(row.id),row.denominator,row.certified_road_n,row.certified_road_no_support_n,row.certified_road_low_alpha_n,row.certified_road_n?fmt(100*(1-row.certified_road_low_alpha_total_n/row.certified_road_n),2)+'%':'—']));
    const protection=(group.diagnostics?.protection_rows??[]).filter(row=>+row.raw===raw);
    table('gt-protection',['阶段','保留车辆RGB最大差','保留车辆alpha最大差'],protection.map(row=>[stageName(row.id??row.stage),fmt(row.K_RGB_max,8),fmt(row.K_alpha_max,8)]));
    const boundary=(group.diagnostics?.boundary_rows??[]).filter(row=>+row.raw===raw);E('gt-boundary-wrap').hidden=!boundary.length;
    if(boundary.length)table('gt-boundary',['阶段','固定A边界像素','完整预乘MAE','固定A充分覆盖','常稳颜色MAE','边界梯度误差','接缝配对数','接缝RGB差'],boundary.map(row=>[stageName(row.id??row.stage),row.A_fixed_n??row.A_boundary_n,fmt(row.A_fixed_premult_RGB_MAE,5),row.A_fixed_alpha95_fraction==null?'—':fmt(100*row.A_fixed_alpha95_fraction,2)+'%',fmt(row.A_boundary_RGB_MAE,5),fmt(row.A_boundary_gradient_MAE,5),row.seam_pairs_n,fmt(row.seam_RGB_jump,5)]));
    const preservation=(group.diagnostics?.source_texture_rows??[]).filter(row=>row.raw==null||+row.raw===raw);E('gt-preservation-wrap').hidden=!preservation.length;
    if(preservation.length)table('gt-preservation',['阶段','参考','对应像素','RGB保持误差','HF保持对应度'],preservation.map(row=>[stageName(row.id??row.stage),row.reference_label??'源块原高斯',row.score_n??row.pixels,fmt(row.RGB_MAE,5),fmt(row.HF_cosine,3)]));
    const constraints=group.diagnostics?.parameter_preservation_rows??[];E('gt-constraints-wrap').hidden=!constraints.length;
    if(constraints.length)table('gt-constraints',['阶段','约束项','核验结果','数量'],constraints.map(row=>[stageName(row.stage),row.label,row.passed===true?'通过':row.passed===false?'未通过':'未提供',row.value??'—']));
    E('gt-visual-wrap').hidden=!group.diagnostics?.visual_findings;E('gt-visual-finding').textContent=group.diagnostics?.visual_findings??'';
    const contact=group.diagnostics?.visual_contact_sheet;E('gt-visual-image').innerHTML=contact?`<a href="${esc(contact)}" target="_blank"><img src="${esc(contact)}" style="display:block;width:100%;height:auto" alt="五帧道路及联合渲染的零步迁移和接缝拟合残余对照"></a>`:'';
    const totals=group.resources?.totals;E('gt-resources-wrap').hidden=!totals;
    if(totals)table('gt-resources',['项目','次数','口径'],[
      ['原生前向 F',totals.native_raster_forward,`${totals.logged_closed_context_F} 闭合记录 + ${totals.direct_public_reload_F_outside_context_counter} 公共重载直接调用`],
      ['原生反向 B',totals.native_raster_backward,'由执行循环及记录推导'],['正式更新',totals.retained_formal_optimizer_updates,'保留'],
      ['短程更新',totals.discarded_smoke_optimizer_updates,'丢弃'],['独立恢复探针更新',totals.discarded_independent_probe_optimizer_updates,'丢弃'],
      ['完成入口更新',totals.completed_entry_optimizer_updates,'复核'],['实际更新（含丢弃）',totals.executed_optimizer_updates_including_discarded,'正式 + 短程 + 探针'],['本轮新增扩散',totals.new_diffusion_calls,'r2正式组']]);
    const layerName=E('gt-layer').value,layer=group.source_layers?.frames?.[raw]?.[layerName];
    const weightLabel=layerName==='same'?'输入支持掩码':layerName==='material'?'参考覆盖掩码':'实际贡献权重';
    E('gt-layer-legend').hidden=layerName!=='effective';
    const layerContext=layerName==='same'?'输入支持':layerName==='material'?'源RGB · 最终几何/覆盖':layerName==='transfer'?'贡献归一化材质色':'道路层来源分层';
    E('gt-layer-image').innerHTML=figure(labels[layerName],file(layer),`raw${raw} · ${layerContext}${layer?.stage?' · '+stageName(layer.stage):''}`,layer?.alpha?`<a href="${esc(file(layer.alpha))}" target="_blank">${weightLabel}</a>`:'');
    E('gt-evidence').innerHTML=(D.gaussian_transfer_evidence??[]).map(item=>`<a href="${esc(item.path)}">${esc(item.name)}</a>`).join(' · ');
  }
  const previousUpdate=update;update=function(){previousUpdate();render();};
  for(const id of ['raw','view','region','scale','left','right'])E(id).onchange=update;
  E('gt-patch').onchange=()=>{sourceOptions();update();};E('gt-source-raw').onchange=update;E('gt-layer').onchange=update;
  render();
})();
