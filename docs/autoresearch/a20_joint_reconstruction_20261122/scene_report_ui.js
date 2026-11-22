window.SceneReport=(()=>{
  const $=id=>document.getElementById(id);let previous=null,originalFinalLinks=null,originalFinalSummary=null,originalTitle=null,originalSubtitle=null;
  const originalHeaders=[...document.querySelectorAll('#metrics')][0].closest('table').querySelector('thead tr').innerHTML;
  function modes(el,items,fallback){const prior=el.value;el.replaceChildren();for(const [value,label] of Object.entries(items))el.add(new Option(label,value));el.value=Object.hasOwn(items,prior)?prior:fallback}
  function architecture(src){$('architecture-image').src=src;$('architecture-link').href=src}
  function render(g){
    if(originalFinalLinks===null){originalFinalLinks=$('final-links').innerHTML;originalFinalSummary=$('final-summary').textContent;originalTitle=document.querySelector('h1').textContent;originalSubtitle=document.querySelector('h1').nextElementSibling.textContent}
    const scene=g.role==='scene_residual_cleanup';
    if(!scene){
      for(const opt of $('camera').options)opt.disabled=false;
      for(const opt of $('channel').options)opt.disabled=false;
      $('geometry-view').disabled=false;
      if(previous){modes($('left-mode'),{shared:'共享场景',isolated:'单对象',teacher:'纯T'},'shared');modes($('right-mode'),{isolated:'单对象',shared:'共享场景',teacher:'纯T'},'isolated');$('metrics').closest('table').querySelector('thead tr').innerHTML=originalHeaders}
      if($('scope-note'))$('scope-note').hidden=true;
      $('final-links').innerHTML=originalFinalLinks;$('final-summary').textContent=originalFinalSummary;document.querySelector('h1').textContent=originalTitle;document.querySelector('h1').nextElementSibling.textContent=originalSubtitle;
      architecture(D.architecture);previous=null;return false;
    }
    if(previous!==g.id){$('camera').value='raw086';$('channel').value='rgb';modes($('left-mode'),g.mode_labels,g.default_modes[0]);modes($('right-mode'),g.mode_labels,g.default_modes[1]);$('left-mode').value=g.default_modes[0];$('right-mode').value=g.default_modes[1]}
    previous=g.id;
    document.querySelector('h1').textContent='a20：场景遮挡与外露伪影清理';document.querySelector('h1').nextElementSibling.textContent='保留a20候选 · 上下文sidecar清理 · 0参数更新';
    $('final-links').replaceChildren();for(const [i,x] of (D.final_links||[]).entries()){const a=document.createElement('a');a.href=x.href;a.textContent=['保留a20资产','历史几何账目','历史灰模审查','资产验真'][i]||x.label;a.target='_blank';$('final-links').append(a)}
    $('final-summary').textContent=g.scene_summary;
    if($('scope-note')){$('scope-note').hidden=false;$('scope-note').textContent=D.root_scope?.display||'当前任务：场景遮挡与外露伪影优先；内部灰模保留为历史诊断。'}
    for(const opt of $('camera').options)opt.disabled=!g.views[opt.value];
    if(!g.views[$('camera').value])$('camera').value='raw086';
    for(const opt of $('channel').options)opt.disabled=opt.value!=='rgb';
    $('channel').value='rgb';$('geometry-view').disabled=true;
    const cam=$('camera').value;
    $('status').textContent=D.root_scope?.closed?g.scene_status.replace('当前任务最终质量未定','已封存：部分清理，未获最终场景验收'):g.scene_status;$('stats').replaceChildren();
    for(const [label,value,note] of g.scene_stats){const card=document.createElement('div');card.className='stat';for(const [cls,t] of [['label',label],['value',value],['label',note]]){const e=document.createElement('div');e.className=cls;e.textContent=t;card.append(e)}$('stats').append(card)}
    architecture(g.architecture);
    for(const side of ['left','right']){
      const mode=$(side+'-mode').value,src=g.views[cam]?.[mode]?.rgb;
      $(side+'-title').textContent=cam.replace('raw0','真实')+' · '+g.mode_labels[mode]+' · 固定裁切';
      const box=$(side+'-image');box.replaceChildren();
      if(src){const a=document.createElement('a');a.className='scene-crop';a.href=src;a.target='_blank';const im=document.createElement('img');im.src=src;im.alt=g.label+' '+cam+' '+mode;a.append(im);const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 275 260');const circle=document.createElementNS(svg.namespaceURI,'ellipse');const [cx,cy,rx,ry]=g.diagnostic_ellipses[cam];for(const [k,v] of Object.entries({cx,cy,rx,ry,fill:'none',stroke:'#ff5c63','stroke-width':1.5}))circle.setAttribute(k,v);svg.append(circle);a.append(svg);box.append(a)}
    }
    $('receipts').replaceChildren();const count=document.createElement('span');count.className='muted';count.textContent=Object.entries(g.actual_counts).map(([k,v])=>({optimizer_updates:'参数更新',native_calls:'原生调用',feature_VJP_calls:'贡献查询',accepted_context_rows:'接受行',rejected_rows:'拒绝行',whole_set_trials:'整组检验'}[k]||k)+' '+v).join(' · ');$('receipts').append(count);
    for(const x of g.receipts){const a=document.createElement('a');a.href=x.href;a.textContent=x.label;a.target='_blank';$('receipts').append(a)}
    const head=$('metrics').closest('table').querySelector('thead tr');head.replaceChildren();for(const label of g.scene_table.headers){const th=document.createElement('th');th.textContent=label;head.append(th)}
    $('metrics').replaceChildren();for(const row of g.scene_table.rows){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('metrics').append(tr)}
    $('grey-panel').hidden=true;$('grey-overview').replaceChildren();$('extra-figures').replaceChildren();
    if(g.residual_diagnostics){const h=document.createElement('h2');h.textContent='冻结道路参考：暗层减少，亮紫层暴露';const wrap=document.createElement('div');wrap.className='table-scroll';const table=document.createElement('table');const head=document.createElement('tr');for(const x of ['真实帧','暗残差像素','亮残差像素','RGB绝对差P95']){const th=document.createElement('th');th.textContent=x;head.append(th)}table.append(head);for(const row of g.residual_diagnostics){const tr=document.createElement('tr');for(const x of row){const td=document.createElement('td');td.textContent=x;tr.append(td)}table.append(tr)}wrap.append(table);$('extra-figures').append(h,wrap)}
    return true;
  }
  return {render};
})();
