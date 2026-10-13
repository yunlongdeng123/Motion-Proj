/* GMC 实例区域只读取已保存资产；不 fetch、训练、渲染或代填人工评价。 */
'use strict';
(() => {
  const payload = document.getElementById('gmc-instance-data');
  if (!payload) return;
  const D = JSON.parse(payload.textContent);
  const $ = id => document.getElementById(id);
  const panel = $('gmc-instance-panel');
  const button = $('gmc-instance-open');
  const legacyControls = document.querySelector('section.controls');
  const legacyTable = $('metric-body').closest('.table-wrap');
  const legacyNodes = ['scheme', 'notice', 'gallery', 'metric-title', 'scene-conclusion'].map($).concat(legacyControls, legacyTable);
  const s = {active:false, mode:'observed', frame:0, actor:0, operation:'delete', branch:'paired', crop:false, timer:null};
  let legacyHidden = [], legacyMode = 'observed';
  const number = (value, digits=6) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'null / 空 body';
  const el = (tag, text, cls) => { const node=document.createElement(tag); if(text!==undefined) node.textContent=text; if(cls) node.className=cls; return node; };
  const blackSelected = () => document.querySelector('[data-scene="black"]').getAttribute('aria-pressed') === 'true';
  const frame = () => D.streams[s.mode].frames[s.frame];
  const edit = () => D.edits['actor'+s.actor+'_'+s.operation].frames[s.frame];
  const hasBox = box => Array.isArray(box) && box.length===4 && box[2]>box[0] && box[3]>box[1];

  function stop() {
    if(s.timer!==null) clearInterval(s.timer);
    s.timer=null; $('gmc-play').textContent='播放';
  }
  function close(restoreMode=false) {
    stop();
    if(s.active) {
      legacyNodes.forEach((node,index)=>{node.hidden=legacyHidden[index];});
      if(restoreMode) document.querySelectorAll('[data-mode]').forEach(node=>node.setAttribute('aria-pressed', node.dataset.mode===legacyMode ? 'true':'false'));
    }
    s.active=false; panel.hidden=true; button.setAttribute('aria-pressed','false');
  }
  function syncScene() {
    button.hidden=!blackSelected();
    if(!blackSelected()) close();
  }
  function open() {
    if(s.active) {close(true); return;}
    if(!blackSelected()) return;
    if($('play').textContent==='暂停') $('play').click();
    legacyMode=document.querySelector('[data-mode][aria-pressed="true"]')?.dataset.mode || 'observed';
    legacyHidden=legacyNodes.map(node=>node.hidden);
    legacyNodes.forEach(node=>{node.hidden=true;});
    document.querySelectorAll('[data-mode]').forEach(node=>node.setAttribute('aria-pressed','false'));
    s.active=true; panel.hidden=false; button.setAttribute('aria-pressed','true'); render();
  }
  function card(title, src, note, box) {
    const article=el('article',undefined,'gmc-card');
    article.append(el('header',title));
    const surface=el('div',undefined,'gmc-surface'), image=el('img');
    image.src=src; image.alt=title+' · '+frame().frame_id; image.loading='eager'; image.decoding='async';
    if(s.crop && hasBox(box)) {
      const [x0,y0,x1,y1]=box, width=x1-x0, height=y1-y0;
      surface.style.aspectRatio=width+'/'+height;
      image.style.width=(480/width*100)+'%'; image.style.left=(-x0/width*100)+'%'; image.style.top=(-y0/height*100)+'%';
    }
    image.onerror=()=>{article.append(el('p','实际图片读取失败：'+src,'gmc-missing')); stop();};
    surface.append(image); article.append(surface,el('footer',note)); return article;
  }
  function metric(label,value) {
    const row=el('tr'); row.append(el('td',label),el('td',value)); return row;
  }
  function render() {
    const f=frame(), isFixed=s.mode==='fixed_camera', item=f.instances[s.actor];
    const e=isFixed ? null : edit(), box=e?.bbox;
    if(isFixed || !hasBox(box)) s.crop=false;
    panel.classList.toggle('gmc-fixed',isFixed);
    $('gmc-observed').setAttribute('aria-pressed',String(!isFixed)); $('gmc-fixed').setAttribute('aria-pressed',String(isFixed));
    $('gmc-operation').disabled=isFixed; $('gmc-crop').disabled=isFixed;
    $('gmc-branch').querySelector('[value="paired"]').textContent=isFixed?'完整场景＋静态 RGB（无 GT）':'合成 RGB / GT / static';
    $('gmc-crop').querySelector('[value="roi"]').disabled=!hasBox(box); $('gmc-crop').value=s.crop?'roi':'full';
    $('gmc-actor').value=String(s.actor); $('gmc-operation').value=s.operation; $('gmc-branch').value=s.branch;
    $('gmc-timeline').value=String(s.frame); $('gmc-prev').disabled=s.frame===0; $('gmc-next').disabled=s.frame===48;
    $('gmc-track-id').textContent=D.instance_ids[s.actor];
    $('gmc-frame-title').textContent=(isFixed?'固定相机模型时间 ':'真实相机帧 ')+(s.frame+1)+' / 49 · '+f.frame_id;
    $('gmc-frame-detail').textContent='相机 '+f.camera_frame_id+' · 模型时间 '+number(f.model_time,2)+' · checkpoint '+D.checkpoint.step;
    $('gmc-notice').classList.toggle('fixed',isFixed);
    $('gmc-notice').textContent=isFixed ? '固定相机锁在实际 0008033，只改变模型时间。这里是未编辑诊断，按单列显示，没有 GT、PSNR 或编辑后结果。' : '真实逐帧：before、after、GT、static 使用该帧实际相机；一次只编辑所选 ID。GT 是未编辑原图，after 没有对应编辑后真值。';
    $('gmc-crop-hint').textContent=isFixed ? '固定相机诊断保留完整480×320。' : (hasBox(box) ? '同一真实 body mask 的扩展 ROI 用于 before/after；旋转和平移可能超出原框，可切回完整原图。' : '该帧真实 body mask 为空：保留完整图和空分母/null，不能据此填补可见性。');
    const cards=[];
    if(isFixed) {
      if(s.branch==='paired') cards.push(card('完整场景 RGB · 未编辑',f.images.composite,'固定真实相机，模型时间变化，无对应GT'),card('静态背景 RGB',f.images.static_rgb,'静态诊断，保留实际变化'));
      else if(s.branch==='visible_alpha') cards.push(card('所选 ID · 联合可见 alpha',item.images.visible_alpha,'完整场景排序后的 alpha×T；黑0白1，无逐图归一化'));
      else cards.push(card('所选 ID · 可见 RGB · 未编辑',item.images.visible_rgb,'完整场景排序后的加权颜色贡献；黑画布不补环境'));
      $('gmc-metric-body').replaceChildren(metric('静态RGB MAE · 相对原相机时间',number(f.diagnostic.static_RGB_MAE_vs_original_camera_time)),metric('所选 ID 可见 alpha 总和',number(f.diagnostic.instance_visible_alpha_sum[s.actor],3)),metric('GT / PSNR / 人工质量评价','无 GT / 无 PSNR / null'));
    } else {
      if(s.branch==='paired') cards.push(card('Before · 当前30k完整场景',e.images.before,'编辑前重建',box),card('After · '+$('gmc-operation').selectedOptions[0].textContent,e.images.after,'只改变所选ID；遮挡变化可影响其他可见像素',box),card('GT · 真实未编辑原图',e.images.gt,'包含原始车辆；不是编辑后真值',box),card('Static · 独立静态背景',e.images.static,'保存FP32已验证编辑前后不变；不证明删除补景正确',box));
      else if(s.branch==='visible_alpha') cards.push(card('Before · 所选 ID 可见 alpha',e.images.before_target_visible,'完整场景排序后的目标alpha贡献，无预乘mask',box),card('After · 所选 ID 可见 alpha',e.images.after_target_visible,'实际编辑后目标贡献；删除后保存FP32验证为0',box));
      else cards.push(card('所选 ID · 可见 RGB · 未编辑',item.images.visible_rgb,'完整场景排序后的加权颜色贡献；当前编辑出口没有after单ID可见RGB',box));
      const m=e.metrics;
      $('gmc-metric-body').replaceChildren(metric('目标轨迹标注有效 / 真实mask已知',String(e.target_pose_valid)+' / '+String(e.mask_known)),metric('真实 body 像素数',String(e.body_pixels)),metric('Before · mask MSE（完整HxW）',number(m.before_mask_mse_full_HxW)),metric('Before · body内可见alpha均值',number(m.before_visible_body_mean)),metric('Before · body外可见alpha均值',number(m.before_visible_outside_mean)),metric('Before→After · 全图RGB MAE',number(m.edit_full_RGB_MAE)),metric('After GT / 人工质量评价','无对应编辑后GT / null'));
    }
    $('gmc-gallery').replaceChildren(...cards);
    $('gmc-evidence').textContent=(D.synthetic_fixture?'合成 fixture · 非结果 · ':'')+'已绑定 30k / 两真实 ID / 49观察帧 + 49固定诊断 / 六编辑294对。结构和实际PNG路径经过CPU校验；未执行GUI QA，不替代照片质量评审。';
  }
  function step(delta) {s.frame=Math.max(0,Math.min(48,s.frame+delta));render();}
  function play() {
    if(s.timer!==null) {stop();return;}
    if(s.frame===48) s.frame=0;
    $('gmc-play').textContent='暂停'; render();
    s.timer=setInterval(()=>{if(s.frame===48){stop();return;}step(1);},1000/6);
  }
  button.addEventListener('click',open);
  document.querySelectorAll('[data-scene]').forEach(node=>node.addEventListener('click',()=>{close();syncScene();}));
  document.querySelectorAll('[data-mode]').forEach(node=>node.addEventListener('click',()=>close()));
  $('gmc-observed').addEventListener('click',()=>{stop();s.mode='observed';render();});
  $('gmc-fixed').addEventListener('click',()=>{stop();s.mode='fixed_camera';s.crop=false;render();});
  $('gmc-actor').addEventListener('change',event=>{stop();s.actor=Number(event.target.value);render();});
  $('gmc-operation').addEventListener('change',event=>{stop();s.operation=event.target.value;render();});
  $('gmc-branch').addEventListener('change',event=>{s.branch=event.target.value;render();});
  $('gmc-crop').addEventListener('change',event=>{s.crop=event.target.value==='roi';render();});
  $('gmc-timeline').addEventListener('input',event=>{stop();s.frame=Number(event.target.value);render();});
  $('gmc-prev').addEventListener('click',()=>{stop();step(-1);}); $('gmc-next').addEventListener('click',()=>{stop();step(1);});
  $('gmc-play').addEventListener('click',play);
  // 只在实例模式接管这些键，避免旧闭包的播放器和新播放器同时改变隐藏帧。
  document.addEventListener('keydown',event=>{
    if(!s.active || event.ctrlKey || event.altKey || event.metaKey || event.isComposing || ['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
    if(['ArrowLeft','ArrowRight'].includes(event.key) || (event.code==='Space' && event.target.tagName!=='BUTTON')) {
      event.preventDefault(); event.stopImmediatePropagation();
      if(event.key==='ArrowLeft'){stop();step(-1);}else if(event.key==='ArrowRight'){stop();step(1);}else play();
    }
  },true);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  syncScene();
  if(location.hash==='#instances-preview'){document.querySelector('[data-scene="black"]').click();open();}
})();
