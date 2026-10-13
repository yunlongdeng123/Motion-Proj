
'use strict';
(() => {
  const data = JSON.parse(document.getElementById('viewer-data').textContent);
  const frames = Array.isArray(data.frames) ? data.frames : [];
  const timeFrames = Array.isArray(data.static_time_frames) ? data.static_time_frames : [];
  const summary = data.summary || {};
  const timeSummary = data.static_time_summary || {};
  const width = number(data.width) || 480, height = number(data.height) || 320;
  const state = {tab:'reconstruction', indices:{reconstruction:0,time:0}, fps:6, playing:false, timer:null, roi:false, modalKey:'gt', fit:false};
  const cards = {reconstruction:[],time:[]};
  const branches = {
    gt:{label:'GT 真值',tag:'观测 RGB',footer:'同帧原始观测',short:'GT'},
    composite:{label:'合成结果',tag:'Composite',footer:'静态 + 动态 + EnvMap',short:'合成'},
    static:{label:'静态分支',tag:'Static RGB',footer:'含 EnvMap 背景',short:'静态'},
    dynamic:{label:'动态分支',tag:'Dynamic RGB',footer:'原始分支 RGB · 未乘 body mask',short:'动态'},
    alpha_s:{label:'静态透明度',tag:'α static',footer:'原始 α · 黑 0 / 白 1',short:'α 静态'},
    alpha_d:{label:'动态透明度',tag:'α dynamic',footer:'原始 α · 黑 0 / 白 1',short:'α 动态'},
    body_mask:{label:'车体掩码',tag:'Body mask',footer:'车体为白色 · 无阴影',short:'车体 mask'}
  };
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const list = tab => tab === 'time' ? timeFrames : frames;
  const current = tab => list(tab || state.tab)[state.indices[tab || state.tab]] || {};
  const frameId = frame => String(frame.id ?? frame.frame_id ?? frame.source_timestamp_frame ?? '—');
  function number(value){if(value === null || value === undefined || value === '' || typeof value === 'boolean')return null;const n=Number(value);return Number.isFinite(n)?n:null;}
  function fmt(value,digits=2){const n=number(value);return n===null?'—':n.toFixed(digits);}
  function compact(value){const n=number(value);if(n===null)return '—';if(n!==0&&Math.abs(n)<.0001)return n.toExponential(2);return n.toFixed(6);}
  function first(...values){return values.find(v=>v!==null&&v!==undefined);}
  function metricRegion(frame,name){const m=frame.metrics||frame.regions||{};const regions=m.regions||m;return regions[name]||{};}
  function metric(frame,...names){const m=frame.metrics||{};return first(...names.map(name=>m[name]),...names.map(name=>frame[name]));}
  function timestamp(frame){return first(frame.timestamp,frame.timestamp_model);}
  function textElement(tag,className,text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
  function overviewStat(label,value,unit,detail){const el=textElement('div','stat');el.append(textElement('div','label',label));const v=textElement('div','value',value);if(unit)v.append(textElement('span','unit',unit));el.append(v,textElement('div','detail',detail));$('#overview').append(el);}
  function metadataItem(label,value){if(value===undefined||value===null||value==='')return;const el=textElement('span','meta-item',label+' ');el.append(textElement('strong','',String(value)));$('#metadata').append(el);}
  function stepValue(){const cp=data.checkpoint;if(cp&&typeof cp==='object')return first(cp.iteration,cp.steps,summary.training_steps);if(number(summary.training_steps)!==null)return summary.training_steps;if(typeof cp==='string'){const match=cp.match(/(?:chkpnt|iteration[_-]?)(\d+)/i);if(match)return Number(match[1]);}return null;}
  $('#page-title').textContent=data.title||'30k 分支查看器';
  document.title=data.title||'30k 分支查看器';
  const scene=typeof data.scene==='object'?first(data.scene.name,data.scene.id):data.scene;
  $('#page-subtitle').textContent=[scene?('场景 '+scene):null,'静态、动态与合成结果，同一帧同步查看'].filter(Boolean).join(' · ');
  $('#sgcm-badge').textContent='SGCM '+((data.components||{}).SGCM?'开启':'关闭');
  $('#nps-badge').textContent='NPS '+((data.components||{}).NPS?'开启':'关闭');
  metadataItem('控制臂',data.arm||'—');
  const spec=(data.source||{}).followup_scene_spec||{};
  metadataItem('源帧',Array.isArray(spec.raw_frame_range)?spec.raw_frame_range.join('–'):null);
  const steps=number(stepValue());metadataItem('训练',steps!==null?steps.toLocaleString('en-US')+' 步':null);
  const wallMinutes=first(summary.wall_minutes,number(summary.wall_seconds)!==null?Number(summary.wall_seconds)/60:null,number(summary.wall_hours)!==null?Number(summary.wall_hours)*60:null);
  metadataItem('耗时',number(wallMinutes)!==null?fmt(wallMinutes,1)+' 分钟':null);
  const deviceGiB=first(summary.device_peak_GiB,summary.peak_vram_gib,number(summary.peak_vram_mib)!==null?Number(summary.peak_vram_mib)/1024:null);
  metadataItem('显存采样峰',number(deviceGiB)!==null?fmt(deviceGiB,2)+' GiB':null);
  metadataItem('reserved 峰值',number(summary.reserved_peak_GiB)!==null?fmt(summary.reserved_peak_GiB,2)+' GiB':null);
  if(number(summary.static_points)!==null&&number(summary.dynamic_points)!==null)metadataItem('高斯',Number(summary.static_points).toLocaleString('en-US')+' 静态 / '+Number(summary.dynamic_points).toLocaleString('en-US')+' 动态');
  const actualCount=frames.length;
  $('.scope b').textContent=actualCount+' 帧全观测开发重建';
  overviewStat('全图 · 合成 PSNR',fmt(summary.full_composite_psnr_db),'dB','所有观测帧的汇总');
  overviewStat('车体 · 合成 PSNR',fmt(summary.actor_composite_psnr_db),'dB',summary.valid_actor_frame_count+' / '+frames.length+' 有效帧等权');
  overviewStat('背景 · 静态 PSNR',fmt(summary.background_static_psnr_db),'dB','非天空且车体外');
  overviewStat('车体外 · 动态 α 均值',fmt(summary.outside_dynamic_alpha_mean,4),'','原始未裁 alpha');
  $('#reconstruction-count').textContent=actualCount+' 帧';
  $('#frames-list-count').textContent=actualCount+' 帧 · 点击跳转';
  if(timeFrames.length){$('#tab-time').hidden=false;$('#about-time').hidden=false;$('#time-count').textContent=timeFrames.length+' 时刻';$('#time-list-count').textContent=timeFrames.length+' 个模型时间 · 点击跳转';}
  $('#actor-domain-note').textContent='全部 '+frames.length+' 帧均保留。车体有效 '+summary.valid_actor_frame_count+' 帧，空 mask '+summary.empty_actor_frame_count+' 帧；车体均值按有效帧等权，不以 0 补空域。局部核心/边界区域各自使用真实有效计数。';
  const anchor=first(timeSummary.anchor_pose_frame,timeSummary.anchor_frame,spec.anchor_frame,'—');
  const timeNotice=$('#time-notice');timeNotice.append(textElement('strong','', '相机固定在 '+anchor+'。'),' 仅改变 model time，观察静态分支；相对锚点时间 '+compact(first(timeSummary.anchor_timestamp_model,timeSummary.anchor_timestamp))+' 测量差异。本组没有对应 GT，不计算重建 PSNR。');
  const checkpoint=data.checkpoint;const cpText=checkpoint&&typeof checkpoint==='object'?first(checkpoint.path,checkpoint.name,checkpoint.iteration):checkpoint;
  const sourcePath=typeof data.source==='string'?data.source:(data.source||{}).model_path;
  $('#source-info').textContent=['检查点：'+(cpText??'—'),sourcePath?('模型路径：'+sourcePath):null,'评价尺寸：'+width+' × '+height,'固定pose：'+anchor].filter(Boolean).join(' · ');
  function table(target,headers,rows){const head=document.createElement('thead'),tr=document.createElement('tr'),body=document.createElement('tbody');headers.forEach(v=>tr.append(textElement('th','',v)));head.append(tr);rows.forEach(values=>{const r=document.createElement('tr');values.forEach(v=>r.append(textElement('td','',String(v))));body.append(r);});$(target).replaceChildren(head,body);}
  function jsonRecord(title,value){const d=textElement('details','fold'),s=textElement('summary','',title),pre=textElement('pre','json-record');d.append(s,pre);d.addEventListener('toggle',()=>{if(d.open&&!pre.textContent)pre.textContent=JSON.stringify(value,null,2);});$('#full-records').append(d);}
  const resources=summary.full_resource_record||{},memory=resources.memory_summary||{},status=resources.status||{},savedPhoto=summary.saved_overall_photo_metrics||{};
  $('#resource-overview').textContent='已保存全图评价：PSNR '+fmt(savedPhoto.psnr,6)+' dB · SSIM '+fmt(savedPhoto.ssim,6)+' · LPIPS '+fmt(savedPhoto.lpips,6)+'。全部49个观测开发视图；区域FP32指标已CPU复算核对。';
  table('#resource-table',['字段','真实记录'],[['完整 wall',fmt(summary.wall_minutes,3)+' min'],['allocated 峰',fmt(summary.allocated_peak_GiB,3)+' GiB'],['reserved 峰',fmt(summary.reserved_peak_GiB,3)+' GiB'],['进程采样峰',fmt(summary.process_peak_GiB,3)+' GiB'],['设备采样峰',fmt(summary.device_peak_GiB,3)+' GiB'],['FW+BW 最终步',fmt(summary.iter_forward_backward_ms,3)+' ms'],['最终静态 / 动态高斯',summary.static_points+' / '+summary.dynamic_points],['训练速度窗口',fmt((status.speed_window||{}).wall_iterations_per_second,3)+' iter/s · '+(status.speed_window||{}).first_iteration+'–'+(status.speed_window||{}).last_iteration],['采样',memory.sampling?memory.sampling.samples+' 样本 / '+memory.sampling.interval_seconds+' s':'—']]);
  table('#phase-table',['阶段','inclusive s','exclusive s','allocated GiB','reserved GiB','进程采样 GiB','设备采样 GiB'],Object.entries(memory.phases||{}).map(([key,p])=>[key,fmt(p.wall_seconds,3),fmt(p.wall_seconds_exclusive,3),fmt(number(p.peak_allocated_bytes)!==null?p.peak_allocated_bytes/2**30:null,3),fmt(number(p.peak_reserved_bytes)!==null?p.peak_reserved_bytes/2**30:null,3),fmt(number(p.process_peak_used_bytes)!==null?p.process_peak_used_bytes/2**30:null,3),fmt(number(p.device_peak_used_bytes)!==null?p.device_peak_used_bytes/2**30:null,3)]));
  jsonRecord('完整真实摘要与各区域有效 / 空帧统计',summary);
  jsonRecord('原始 memory_summary 与 status',resources);
  jsonRecord('三节点实际 completion_contract',data.completion_contract);
  jsonRecord('CPU FP32 有限性与指标复算',data.verification);
  jsonRecord('固定49模型时间原始摘要',timeSummary);
  jsonRecord('来源与 effective_config',{source:data.source,effective_config:data.effective_config,restoration_verified:data.restoration_verified,human_verdict:data.human_verdict});

  function createCard(key,container,tab){
    const config=branches[key],card=textElement('article','image-card');card.dataset.key=key;
    const heading=textElement('div','card-heading');heading.append(textElement('i','branch-dot'),textElement('h2','',config.label),textElement('span','card-tag',config.tag));
    const button=textElement('button','image-button');button.setAttribute('aria-label','查看'+config.label+'原始大图');
    const surface=textElement('div','image-surface'),img=document.createElement('img'),empty=textElement('span','empty','当前帧没有这张图');empty.hidden=true;img.alt=config.label;img.decoding='async';img.draggable=false;surface.append(img,empty);button.append(surface);
    const footer=textElement('div','card-footer');footer.append(textElement('span','',config.footer),textElement('span','zoom-hint','↗ 查看大图'));
    card.append(heading,button,footer);container.append(card);
    const record={key,card,surface,img,empty,button};cards[tab].push(record);
    img.addEventListener('error',()=>{empty.hidden=false;img.hidden=true;empty.textContent='图片未能载入 · 请保留 assets/ 目录';});
    img.addEventListener('load',()=>{if(img.getAttribute('src')){empty.hidden=true;img.hidden=false;}});
    button.addEventListener('click',()=>{if((current(tab).images||{})[key])openModal(key);});
  }
  ['gt','composite','static','dynamic'].forEach(key=>createCard(key,$('#reconstruction-grid'),'reconstruction'));
  ['alpha_s','alpha_d','body_mask'].forEach(key=>createCard(key,$('#diagnostic-grid'),'reconstruction'));
  ['static','alpha_s'].forEach(key=>createCard(key,$('#time-grid'),'time'));

  function bboxFor(frame){
    let raw=frame.bbox||frame.actor_bbox||frame.roi_bbox;if(!raw)return null;
    let box;if(Array.isArray(raw)&&raw.length>=4)box=raw.slice(0,4).map(number);else if(typeof raw==='object'){
      const x=first(raw.x0,raw.x,raw.left),y=first(raw.y0,raw.y,raw.top);
      box=[number(x),number(y),number(first(raw.x1,raw.right,number(x)!==null&&number(raw.width)!==null?Number(x)+Number(raw.width):null)),number(first(raw.y1,raw.bottom,number(y)!==null&&number(raw.height)!==null?Number(y)+Number(raw.height):null))];
    }else return null;
    if(box.some(v=>v===null)||box[2]<=box[0]||box[3]<=box[1])return null;
    let x0=Math.max(0,Math.min(width,box[0])),y0=Math.max(0,Math.min(height,box[1])),x1=Math.max(0,Math.min(width,box[2])),y1=Math.max(0,Math.min(height,box[3]));
    if(x1<=x0||y1<=y0)return null;
    // 所有面板使用同一裁切；只扩展到原图比例，不收缩已有上下文框。
    const ratio=width/height,cx=(x0+x1)/2,cy=(y0+y1)/2;let w=x1-x0,h=y1-y0;
    if(w/h<ratio)w=h*ratio;else h=w/ratio;
    if(w>width){w=width;h=w/ratio;}if(h>height){h=height;w=h*ratio;}
    x0=Math.max(0,Math.min(width-w,cx-w/2));y0=Math.max(0,Math.min(height-h,cy-h/2));
    return [x0,y0,w,h];
  }
  function applyCrop(img,surface,frame,tab,native=false){
    const box=state.roi&&tab==='reconstruction'?bboxFor(frame):null;
    surface.style.aspectRatio=width+' / '+height;
    if(box){const [x,y,w,h]=box;img.style.width=(width/w*100)+'%';img.style.left=(-x/w*100)+'%';img.style.top=(-y/h*100)+'%';if(native&&!state.fit)surface.style.width=Math.round(w)+'px';}
    else{img.style.width='100%';img.style.left='0';img.style.top='0';if(native&&!state.fit)surface.style.width=width+'px';}
    if(native&&state.fit)surface.style.width='100%';
  }
  function updateImage(img,empty,surface,key,frame,tab,native=false){
    const path=(frame.images||{})[key];applyCrop(img,surface,frame,tab,native);
    img.alt=branches[key].label+' · '+frameId(frame)+(tab==='time'?' · 固定相机 model time '+compact(timestamp(frame)):'');
    if(!path){img.removeAttribute('src');img.hidden=true;empty.textContent='当前帧没有这张图';empty.hidden=false;return;}
    if(img.getAttribute('src')!==String(path)){empty.hidden=true;img.hidden=false;img.src=String(path);}
  }
  function regionCard(title,region,mainKey,mainLabel,secondaryKey,secondaryLabel){
    const empty=region.pixel_count===0;const el=textElement('div','metric-card');el.append(textElement('div','region',title+' · '+(region.pixel_count??'—')+' px'));
    const value=textElement('div','metric-value',fmt(region[mainKey]));if(!empty)value.append(textElement('span','','dB'));el.append(value,textElement('div','metric-description',empty?'无该区域 · 指标为空':mainLabel));
    const pairs=textElement('div','metric-pairs');
    [[secondaryLabel,fmt(region[secondaryKey])+' dB'],['α 动态',fmt(region.alpha_d_mean,4)],['α 静态',fmt(region.alpha_s_mean,4)]].forEach(([label,value])=>{const row=textElement('div');row.append(textElement('span','',label),textElement('strong','',value));pairs.append(row);});el.append(pairs);return el;
  }
  function renderMetrics(frame){
    $('#frame-metrics').replaceChildren(
      regionCard('全图',metricRegion(frame,'full'),'composite_psnr_db','合成 PSNR','static_psnr_db','静态 PSNR'),
      regionCard('车体 · body mask',metricRegion(frame,'actor'),'composite_psnr_db','合成 PSNR','static_psnr_db','静态 PSNR'),
      regionCard('背景 · 非天空且车体外',metricRegion(frame,'non_sky_outside_actor'),'static_psnr_db','静态 PSNR','composite_psnr_db','合成 PSNR')
    );
  }
  function differenceCard(label,value,description){const el=textElement('div','metric-card');el.append(textElement('div','region',label),textElement('div','metric-value',compact(value)),textElement('div','metric-description',description));return el;}
  function renderTimeMetrics(frame){$('#time-metrics').replaceChildren(
    differenceCard('静态 RGB · 平均绝对差',metric(frame,'static_rgb_MAE_from_anchor','rgb_mae'),'FP32 clip[0,1] · 相对锚点时间 MAE'),
    differenceCard('静态 RGB · 最大绝对差',metric(frame,'static_rgb_max_abs_from_anchor','rgb_max_abs'),'FP32 clip[0,1] · 相对锚点 max |Δ|'),
    differenceCard('静态 α · 平均绝对差',metric(frame,'static_alpha_MAE_from_anchor','alpha_mae'),'相对锚点时间的 MAE')
  );}
  function createFrameList(tab,container){list(tab).forEach((frame,index)=>{const button=textElement('button','frame-chip');button.dataset.index=index;button.append(textElement('span','',String(index+1).padStart(2,'0')+' · '+frameId(frame)),textElement('small','','本地 '+frame.local_index+' / 源 '+frame.source_raw_index+' · t '+compact(timestamp(frame))));button.setAttribute('aria-label','跳转到第'+(index+1)+'帧，'+frameId(frame));button.addEventListener('click',()=>setIndex(index,tab));container.append(button);});}
  createFrameList('reconstruction',$('#frames-list'));createFrameList('time',$('#time-list'));
  function updateControls(tab){
    const control=$('[data-controls="'+tab+'"]'),items=list(tab),index=state.indices[tab],frame=items[index]||{};
    control.querySelector('.frame-label').textContent=tab==='time'?('时间 '+(index+1)+' / '+items.length+' · '+frameId(frame)):('帧 '+(index+1)+' / '+items.length+' · '+frameId(frame));
    const rawIndex=frame.local_index!==undefined?' · 本地 index '+frame.local_index+' / 源 raw '+frame.source_raw_index:'';
    control.querySelector('.frame-meta').textContent='model time '+compact(timestamp(frame))+rawIndex;
    const slider=control.querySelector('.slider');slider.max=Math.max(0,items.length-1);slider.value=index;slider.disabled=items.length<2;slider.setAttribute('aria-valuetext',frameId(frame)+'，model time '+compact(timestamp(frame)));
    control.querySelector('.frame-total').textContent=items.length;
    control.querySelector('.prev').disabled=index===0;control.querySelector('.next').disabled=index>=items.length-1;
    control.querySelector('.play').disabled=items.length<2;
    const roi=control.querySelector('[data-view="roi"]');if(roi)roi.disabled=!bboxFor(frame);
    $$(tab==='time'?'#time-list .frame-chip':'#frames-list .frame-chip').forEach(button=>button.setAttribute('aria-current',Number(button.dataset.index)===index?'true':'false'));
  }
  function render(tab=state.tab){
    const frame=current(tab);cards[tab].forEach(record=>updateImage(record.img,record.empty,record.surface,record.key,frame,tab));
    updateControls(tab);if(tab==='time')renderTimeMetrics(frame);else{renderMetrics(frame);const note=$('[data-controls="reconstruction"] .roi-note');note.textContent=state.roi?(bboxFor(frame)?'车体附近同一 ROI · 所有分支与诊断图同步':'当前帧无车体 ROI · 显示原图'):'同帧四卡同步 · 点击图片查看原始大图';}
    if($('#lightbox').open)renderModal();
  }
  function setIndex(index,tab=state.tab){state.indices[tab]=Math.max(0,Math.min(list(tab).length-1,index));render(tab);}
  function step(delta){setIndex(state.indices[state.tab]+delta);}
  function playUI(){$$('.play').forEach(button=>{const active=state.playing&&button.closest('[data-controls]').dataset.controls===state.tab;button.classList.toggle('active',active);button.setAttribute('aria-pressed',active?'true':'false');button.querySelector('span').textContent=active?'暂停':'播放';button.querySelector('path').setAttribute('d',active?'M6 5h4v14H6zm8 0h4v14h-4z':'M8 5v14l11-7z');});}
  function stop(){if(state.timer!==null)clearInterval(state.timer);state.timer=null;state.playing=false;playUI();}
  function start(){stop();if(list(state.tab).length<2)return;state.playing=true;playUI();state.timer=setInterval(()=>{const next=(state.indices[state.tab]+1)%list(state.tab).length;setIndex(next);},1000/state.fps);}
  function togglePlay(){state.playing?stop():start();}
  function setTab(tab){if(tab==='time'&&!timeFrames.length)return;stop();if($('#lightbox').open)$('#lightbox').close();state.tab=tab;$('#reconstruction').hidden=tab!=='reconstruction';$('#time-panel').hidden=tab!=='time';$$('[data-tab]').forEach(button=>button.setAttribute('aria-selected',button.dataset.tab===tab?'true':'false'));render(tab);}
  $$('[data-tab]').forEach(button=>button.addEventListener('click',()=>setTab(button.dataset.tab)));
  $$('[data-controls]').forEach(control=>{const tab=control.dataset.controls;control.querySelector('.prev').addEventListener('click',()=>setIndex(state.indices[tab]-1,tab));control.querySelector('.next').addEventListener('click',()=>setIndex(state.indices[tab]+1,tab));control.querySelector('.play').addEventListener('click',togglePlay);control.querySelector('.slider').addEventListener('input',event=>setIndex(Number(event.target.value),tab));control.querySelector('.fps').addEventListener('change',event=>{state.fps=Number(event.target.value);$$('.fps').forEach(select=>select.value=state.fps);if(state.playing)start();});});
  $$('[data-view]').forEach(button=>button.addEventListener('click',()=>{state.roi=button.dataset.view==='roi';$$('[data-view]').forEach(option=>option.setAttribute('aria-pressed',(option.dataset.view==='roi')===state.roi?'true':'false'));render('reconstruction');}));

  function openModal(key){state.modalKey=key;const available=state.tab==='time'?['static','alpha_s']:Object.keys(branches);$('#modal-branches').replaceChildren();available.forEach(branch=>{const button=textElement('button','',branches[branch].short);button.dataset.branch=branch;button.addEventListener('click',()=>{state.modalKey=branch;renderModal();});$('#modal-branches').append(button);});$('#lightbox').showModal();renderModal();}
  function renderModal(){
    const frame=current(),key=state.modalKey,index=state.indices[state.tab],total=list(state.tab).length;
    $('#lightbox-title').textContent=branches[key].label;
    $('#lightbox-meta').textContent=frameId(frame)+' · '+(index+1)+' / '+total+' · model time '+compact(timestamp(frame));
    $('#modal-branches').querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',button.dataset.branch===key?'true':'false'));
    $('#modal-image-wrap').classList.toggle('fit',state.fit);
    updateImage($('#modal-image'),$('#modal-image-surface .empty'),$('#modal-image-surface'),key,frame,state.tab,true);
    $('#modal-prev').disabled=index===0;$('#modal-next').disabled=index>=total-1;
    const roi=state.roi&&state.tab==='reconstruction'&&bboxFor(frame);$('#modal-caption').textContent=(roi?'同步车体 ROI':'完整原图')+' · '+width+' × '+height+' 源图 · '+branches[key].footer;
  }
  $('#modal-image').addEventListener('error',()=>{$('#modal-image-surface .empty').textContent='图片未能载入 · 请保留 assets/ 目录';$('#modal-image-surface .empty').hidden=false;$('#modal-image').hidden=true;});
  $('#modal-image').addEventListener('load',()=>{$('#modal-image-surface .empty').hidden=true;$('#modal-image').hidden=false;});
  $('#close-lightbox').addEventListener('click',()=>$('#lightbox').close());
  $('#lightbox').addEventListener('click',event=>{if(event.target!==$('#lightbox'))return;const box=$('#lightbox').getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)$('#lightbox').close();});
  $('#modal-prev').addEventListener('click',()=>step(-1));$('#modal-next').addEventListener('click',()=>step(1));
  function zoom(fit){state.fit=fit;$('#zoom-native').setAttribute('aria-pressed',fit?'false':'true');$('#zoom-fit').setAttribute('aria-pressed',fit?'true':'false');renderModal();}
  $('#zoom-native').addEventListener('click',()=>zoom(false));$('#zoom-fit').addEventListener('click',()=>zoom(true));
  document.addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey)return;const tag=(event.target.tagName||'').toLowerCase();if(['input','select','textarea'].includes(tag))return;if(event.key==='ArrowLeft'){event.preventDefault();step(-1);}else if(event.key==='ArrowRight'){event.preventDefault();step(1);}else if(event.code==='Space'&&tag!=='button'){event.preventDefault();togglePlay();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  render('reconstruction');if(timeFrames.length)render('time');
})();
