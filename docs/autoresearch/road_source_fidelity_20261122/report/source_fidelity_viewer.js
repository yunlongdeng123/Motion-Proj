/* Independent source-colour chain; keeps D and the prior quality audit intact. */
(() => {
  const data = document.getElementById('source-fidelity-chain-data');
  if (!data || document.getElementById('source-fidelity-chain')) return;
  const result = JSON.parse(data.textContent);
  if (!result.complete) return;
  const q = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (value, n=5) => value == null ? '—' : Number(value).toFixed(n);
  const stageNames = {photo:'原尺寸实拍', sampled_rgb:'取样点颜色 · 投影', loaded_rgb:'保存/加载颜色 · 投影', original_render:'原源高斯渲染', native_render:'固定几何 · 原生渲染'};
  const stageOrder = ['photo','sampled_rgb','loaded_rgb','original_render','native_render'];
  const table = (id, headers, rows) => {q(id).innerHTML='<thead><tr>'+headers.map(v=>`<th>${esc(v)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>`<td>${esc(v)}</td>`).join('')+'</tr>').join('')+'</tbody>';};
  const css = document.createElement('style');
  css.textContent=`.sfc-panel{border-top:1px solid var(--line);margin-top:24px;padding-top:20px}.sfc-panel h3{margin:20px 0 8px}.sfc-arch{display:block;min-width:1040px;width:100%}.sfc-patch{min-width:0;padding:14px 0;border-bottom:1px solid var(--line)}.sfc-images{display:grid;grid-template-columns:repeat(var(--sfc-count,4),minmax(0,1fr));gap:10px}.sfc-images figure{min-width:0}.sfc-images figcaption{min-height:0}.sfc-port{overflow:auto;min-height:110px;max-height:360px;background:#080e16;padding:10px}.sfc-port img{display:block;max-width:none;margin:0 auto;image-rendering:pixelated;mask-mode:luminance;mask-repeat:no-repeat;mask-size:100% 100%}.sfc-scope{font-size:13px;color:var(--muted);margin:8px 0}.sfc-panel .controls{gap:14px}.sfc-panel small{font-variant-numeric:tabular-nums}.sfc-panel table{font-variant-numeric:tabular-nums}@media(max-width:1100px){.sfc-images{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.sfc-images{grid-template-columns:1fr}}`;
  document.head.append(css);
  const panel = document.createElement('section');
  panel.id='source-fidelity-chain';panel.className='sfc-panel';
  panel.innerHTML=`<h2>源材质链路诊断</h2><p class="muted">1920×1280采集栅格 · 固定几何与相机 · 源00 / 02</p><div id="sfc-findings"></div><div class="scroll"><img class="sfc-arch" src="${esc(result.architecture)}" alt="原尺寸实拍经过取样点颜色、保存加载及固定几何原生渲染，输出颜色与纹理核对"></div><p id="sfc-review-note" class="muted" hidden></p><div class="controls"><label>像素显示<select id="sfc-zoom"><option value="1">100%</option><option value="2">200% · 最近邻</option><option value="4">400% · 最近邻</option></select></label><label>高斯范围<select id="sfc-variant"><option value="core">核心</option><option value="halo">核心＋核验边缘</option></select></label></div><div id="sfc-patches"></div><h3>颜色与纹理</h3><div class="scroll"><table id="sfc-metrics"></table></div><p id="sfc-metric-note" class="muted" hidden></p><h3>保存 / 加载与冻结核验</h3><p id="sfc-encoding" class="muted"></p><div class="scroll"><table id="sfc-checks"></table></div><h3>本轮计算</h3><div class="scroll"><table id="sfc-resources"></table></div><details><summary>本轮证据</summary><p id="sfc-evidence"></p></details>`;
  const anchor=q('source-quality-audit');
  if (!anchor) return;
  anchor.after(panel);
  q('sfc-findings').innerHTML=(result.findings??[]).map(v=>`<p>${esc(v)}</p>`).join('');
  for(const [id,value] of [['sfc-review-note',result.review_note],['sfc-metric-note',result.metric_note]])if(value){q(id).hidden=false;q(id).textContent=value;}
  q('sfc-encoding').textContent='颜色编码：'+result.color_encoding;
  q('sfc-patches').innerHTML=result.patches.map(p=>{
    const cases=result.cases.filter(c=>c.patch===p.id);
    return `<div class="sfc-patch" data-sfc-patch="${esc(p.id)}"><div class="controls"><h3>${esc(p.label??p.id.replace('patch_','源'))}</h3><label>实拍帧<select id="sfc-frame-${esc(p.id)}">${cases.map(c=>`<option value="${c.raw}">${c.raw}${c.role?' · '+esc(c.role):''}</option>`).join('')}</select></label></div>${p.review?`<p class="sfc-scope">${esc(p.review)}</p>`:''}<div id="sfc-images-${esc(p.id)}" class="sfc-images"></div><p id="sfc-scope-${esc(p.id)}" class="sfc-scope"></p></div>`;
  }).join('');
  for(const p of result.patches){const select=q('sfc-frame-'+p.id);if(p.preferred_frame!=null&&[...select.options].some(o=>o.value===String(p.preferred_frame)))select.value=String(p.preferred_frame);select.onchange=render;}
  function render(){
    const zoom=+q('sfc-zoom').value,variant=q('sfc-variant').value,metricRows=[];
    for(const patch of result.patches){
      const row=result.cases.find(c=>c.patch===patch.id&&c.raw===+q('sfc-frame-'+patch.id).value),view=row.variants[variant],[width,height]=row.crop_wh;
      const stages=stageOrder.filter(stage=>view.images[stage]);
      const gallery=q('sfc-images-'+patch.id);gallery.style.setProperty('--sfc-count',stages.length);
      gallery.innerHTML=stages.map(stage=>`<figure data-sfc-stage="${stage}"><figcaption><strong>${esc(stageNames[stage])}</strong><small>${width}×${height}px · ${zoom*100}%</small></figcaption><div class="sfc-port"><a href="${esc(view.images[stage])}" target="_blank"><img src="${esc(view.images[stage])}" width="${width}" height="${height}" style="width:${width*zoom}px;height:${height*zoom}px;${view.display_mask?`mask-image:url('${esc(view.display_mask)}');`:''}" alt="${esc(patch.label??patch.id)} 帧${row.raw} ${esc(view.label)} ${esc(stageNames[stage])}"></a></div></figure>`).join('');
      q('sfc-scope-'+patch.id).textContent=`${view.label} · ${view.scope} · 裁剪 [${row.bbox_xyxy.join(', ')}]`;
      for(const m of view.metrics)metricRows.push([`${patch.label??patch.id} / ${row.raw}`,stageNames[m.stage]??m.stage,m.domain,m.score_n,fmt(m.RGB_MAE),m.RGB_bias?m.RGB_bias.map(v=>fmt(v)).join(' / '):'—',m.sigma_native_px==null?'—':fmt(m.sigma_native_px,3),fmt(m.HF_cosine,3),fmt(m.HF_RMS_ratio,3)]);
    }
    table('sfc-metrics',['源块 / 帧','链路阶段','评价域','有效数','RGB MAE ↓','RGB偏差 R/G/B','σ 采集像素','HF对应度 ↑','HF RMS比 → 1'],metricRows);
  }
  q('sfc-zoom').onchange=render;q('sfc-variant').onchange=render;
  table('sfc-checks',['源块','点数','点ID相同','RGB最大差','RGB平均差','几何冻结'],result.checks.map(c=>[c.patch.replace('patch_','源'),c.point_count,c.point_ids_equal?'是':'否',fmt(c.RGB_max_abs,9),fmt(c.RGB_mean_abs,9),c.geometry_unchanged?'通过':'未通过']));
  table('sfc-resources',['过程','F','B','正式更新','额外更新'],result.resources.map(r=>[r.process,r.nativeF,r.nativeB,r.formal_updates,r.extra_updates]));
  const evidence=[...(result.evidence??[]),...result.checks.map(c=>({name:c.patch.replace('patch_','源')+'数值核验',path:c.evidence}))];
  q('sfc-evidence').innerHTML=evidence.map(e=>`<a href="${esc(e.path)}">${esc(e.name)}</a>`).join(' · ');
  const entry=document.createElement('button');entry.id='sfc-entry';entry.textContent='源材质链路';entry.onclick=()=>{document.querySelector('[data-preset="gaussian-transfer-zero"]')?.click();panel.scrollIntoView({block:'start'});};document.querySelector('nav').append(entry);
  render();
})();
