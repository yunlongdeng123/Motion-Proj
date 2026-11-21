/* 1920采集栅格源保真核查；独立追加，不改写既有实验。 */
(() => {
  const node = document.getElementById('source-quality-audit-data');
  if (!node || document.getElementById('source-quality-audit')) return;
  const audit = JSON.parse(node.textContent);
  if (!audit.complete) return;
  const q = id => document.getElementById(id), safe = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const style = document.createElement('style');
  style.textContent = `.sq-panel{border-top:1px solid var(--line);padding-top:20px;margin:22px 0}.sq-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.sq-grid.sq-three{grid-template-columns:repeat(3,minmax(0,1fr))}.sq-port{overflow:auto;max-height:380px;min-height:130px;padding:12px;background:#080e16}.sq-port img{display:block;max-width:none;margin:0 auto;image-rendering:pixelated}.sq-port img.sq-core{mask-mode:luminance;mask-size:100% 100%;mask-repeat:no-repeat}.sq-arch{display:block;width:100%;min-width:1000px}.sq-links{font-size:13px;margin:12px 0}.sq-selected{background:#1e3648}.sq-panel figure small{font-variant-numeric:tabular-nums}.sq-panel figcaption{min-height:0}.sq-flow{margin:12px 0}.sq-panel h3{margin-top:22px}@media(max-width:1000px){.sq-grid.sq-three{grid-template-columns:1fr}}@media(max-width:800px){.sq-grid{grid-template-columns:1fr}}`;
  document.head.append(style);
  const panel = document.createElement('section');panel.id = 'source-quality-audit';panel.className = 'sq-panel';
  panel.innerHTML = `<h2>源块质量核查</h2><p id="sq-verdict" class="note" hidden></p><p class="muted">1920×1280 采集栅格 · 原源高斯核查${audit.source_refit?'与源RGB重拟合':''} · 本轮新增扩散 0</p><div class="scroll sq-flow"><img class="sq-arch" src="${safe(audit.architecture)}" alt="原尺寸原照与现有源高斯经过分辨率、几何和尺度核查，仅RGB重拟合400步后评价查询视角，本轮停止新目标搬运"></div><div class="controls"><label>源块<select id="sq-patch"></select></label><label>帧<select id="sq-raw"></select></label><label>像素显示<select id="sq-zoom"><option value="1">100%</option><option value="2">200% · 最近邻</option><option value="4">400% · 最近邻</option></select></label><label><input type="checkbox" id="sq-core" checked>仅显示核心</label></div><div id="sq-images" class="sq-grid"></div><p class="sq-links" id="sq-links"></p><div id="sq-refit-metrics" hidden><h3>源00 · 仅颜色重拟合</h3><p class="muted">训练：98 / 137；查询：86 / 110 / 173（历史建模视角）。仅优化1598个核心高斯的4794个颜色参数。</p><div class="controls"><label>高频核 σ（采集像素）<select id="sq-band"><option value="1.2">1.2px</option><option value="1.6666666666666667" selected>1.667px</option><option value="3">3.0px</option></select></label></div><div class="scroll"><table id="sq-refit-table"></table></div><p class="muted">相同1920栅格、固定核心内缩5px ∩ 初始 α≥0.95 ∩ 照片合法掩码；位置、法向、尺度和透明度冻结。</p></div><details id="sq-native-details"><summary>原源核查 · 三块两帧</summary><div class="scroll"><table id="sq-metrics"></table></div><p class="muted">高频核：σ=1.2 个采集像素；评价域：物理核心内缩5px且 α≥0.95。各分辨率按各自像素核分表统计。</p></details><div id="sq-summary" hidden></div><h3>本节计算</h3><div class="scroll"><table id="sq-resources"></table></div><p class="muted">源RGB正式400步与旧目标接缝400步分别记录。本轮未执行新目标迁移。</p><details><summary>核查记录</summary><p id="sq-evidence"></p></details>`;
  const anchor = q('gt-source-quality')?.parentElement;
  if (!anchor) return;
  anchor.after(panel);
  const patches = [...new Set(audit.cases.map(item => item.patch))].sort();
  q('sq-patch').innerHTML = patches.map(id => `<option value="${safe(id)}">${safe(id.replace('patch_', '源块'))}</option>`).join('');
  function frameOptions(){const previous=q('sq-raw').value,rows=audit.cases.filter(r=>r.patch===q('sq-patch').value).sort((a,b)=>a.raw-b.raw);q('sq-raw').innerHTML=rows.map(r=>`<option value="${r.raw}">${r.raw}${r.role?(r.role==='TRAIN'?' · 训练':' · 查询'):''}</option>`).join('');q('sq-raw').value=rows.some(r=>String(r.raw)===previous)?previous:rows.some(r=>r.raw===110)?'110':String(rows[0].raw);}
  frameOptions();
  const format = (value, n=5) => value == null ? '—' : Number(value).toFixed(n);
  function render() {
    const row = audit.cases.find(item => item.patch === q('sq-patch').value && item.raw === +q('sq-raw').value), zoom = +q('sq-zoom').value, core = q('sq-core').checked;
    const [width, height] = row.crop_wh, subtitle = `${row.patch.replace('patch_', '源块')} · 帧${row.raw}${row.role?(row.role==='TRAIN'?' · 训练':' · 查询（历史建模）'):''} · ${width}×${height}px · ${zoom*100}%`;
    const image = (title, path) => `<figure><figcaption><strong>${title}</strong><small>${safe(subtitle)}</small></figcaption><div class="sq-port"><a href="${safe(path)}" target="_blank"><img class="${core?'sq-core':''}" style="width:${width*zoom}px;height:${height*zoom}px;${core?`mask-image:url('${safe(row.core_mask)}');`:''}" src="${safe(path)}" width="${width}" height="${height}" alt="${title}"></a></div></figure>`;
    q('sq-images').classList.toggle('sq-three',Boolean(row.refit));
    q('sq-images').innerHTML = image(core?'原始实拍核心':'原始实拍裁片', row.photo) + image(core?'原源高斯核心':'原源高斯裁片', row.gaussians) + (row.refit?image('源RGB重拟合 · 400步',row.refit):'');
    q('sq-links').innerHTML = `<a href="${safe(row.core_mask)}">物理核心掩码</a> · <a href="${safe(row.score_mask)}">评价掩码</a>${row.alpha?` · <a href="${safe(row.alpha)}">高斯透明度</a>`:''}${row.full_road?` · <a href="${safe(row.full_road)}">原道路层上下文</a>`:''}`;
    const headers = ['源块 / 帧','核心像素','评价像素','RGB MAE','HF对应度','HF RMS比（高斯/实拍）'];
    q('sq-metrics').innerHTML = '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' + (audit.native_cases??audit.cases).map(item => `<tr class="${item.raw===row.raw&&item.patch===row.patch?'sq-selected':''}"><td>${safe(item.patch.replace('patch_','源块'))} / ${item.raw}</td><td>${item.core_pixels}</td><td>${item.score_pixels}</td><td>${format(item.RGB_MAE)}</td><td>${format(item.HF_cosine,3)}</td><td>${format(item.HF_RMS_ratio,3)}</td></tr>`).join('') + '</tbody>';
    const refit=audit.source_refit;q('sq-refit-metrics').hidden=!refit||row.patch!=='patch_00';q('sq-native-details').open=!refit||row.patch!=='patch_00';
    if(refit){const sigma=+q('sq-band').value,band=r=>r.spectral.find(s=>Math.abs(s.sigma_native_px-sigma)<1e-6);const heads=['帧 / 角色','固定像素','原RGB MAE','重拟合RGB MAE','原HF对应','重拟合HF对应','原RMS比','重拟合RMS比'];q('sq-refit-table').innerHTML='<thead><tr>'+heads.map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>'+refit.cases.map(c=>{const a=c.metric_original,b=c.metric_refit,sa=band(a),sb=band(b);return `<tr class="${c.raw===row.raw?'sq-selected':''}"><td>${c.raw} / ${c.role==='TRAIN'?'训练':'查询（历史建模）'}</td><td>${a.score_n}</td><td>${format(a.RGB_MAE)}</td><td>${format(b.RGB_MAE)}</td><td>${format(sa.HF_cosine,3)}</td><td>${format(sb.HF_cosine,3)}</td><td>${format(sa.HF_RMS_ratio,3)}</td><td>${format(sb.HF_RMS_ratio,3)}</td></tr>`;}).join('')+'</tbody>';}
  }
  q('sq-patch').addEventListener('change',()=>{frameOptions();render();});
  for (const id of ['sq-raw','sq-zoom','sq-core','sq-band']) q(id).addEventListener('change',render);
  const summary = audit.summary;
  if (summary) {
    if(summary.verdict){q('sq-verdict').hidden=false;q('sq-verdict').textContent=summary.verdict;}
    q('sq-summary').hidden = false;
    q('sq-summary').innerHTML = `<h3>独立审计与结论</h3>${(summary.findings??[]).map(text=>`<p>${safe(text)}</p>`).join('')}${summary.ranking_rows?'<details><summary>199帧参考排序与检索范围</summary><p>'+safe(summary.retrieval_scope)+'</p><div class="scroll"><table><thead><tr><th>源块</th><th>照片资格通过帧</th><th>核心短轴上限</th><th>前列参考帧</th></tr></thead><tbody>'+summary.ranking_rows.map(r=>`<tr><td>${safe(r.patch.replace('patch_','源块'))}</td><td>${r.qualified_views} / 199</td><td>${format(r.max_short_axis_px,2)}px</td><td>${safe(r.top_frames.join(' / '))}</td></tr>`).join('')+'</tbody></table></div><p>'+safe(summary.reference_semantics)+'</p></details>':''}`;
  }
  const resourceRows=audit.summary?.resource_rows??[['原源尺寸核查',audit.resources.nativeF,audit.resources.nativeB,0,0]];if(!audit.summary?.resource_rows&&audit.source_refit){const r=audit.source_refit.resources;resourceRows.push(['源RGB诊断',r.nativeF,r.nativeB,r.formal_optimizer_updates,r.discarded_smoke_and_restore_updates]);}
  q('sq-resources').innerHTML='<thead><tr>'+['过程','F','B','正式更新','丢弃更新'].map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>'+resourceRows.map(r=>'<tr>'+r.map(v=>`<td>${safe(v)}</td>`).join('')+'</tr>').join('')+'</tbody>';
  const evidence=[{name:'1920原尺寸核查记录',path:audit.receipt},...(audit.source_refit?[{name:'源RGB诊断合同与结果',path:audit.source_refit.receipt}]:[]),...(audit.evidence??[]),...(audit.summary_evidence?[{name:'本轮审计摘要',path:audit.summary_evidence}]:[])];q('sq-evidence').innerHTML=evidence.map(e=>`<a href="${safe(e.path)}">${safe(e.name)}</a>`).join(' · ');
  const button = document.createElement('button');button.textContent = '源块质量核查';button.id = 'sq-entry';
  button.onclick = () => {const preset=document.querySelector('[data-preset="gaussian-transfer-zero"]');preset?.click();panel.scrollIntoView({block:'start'});};
  document.querySelector('nav').append(button);
  render();
})();
