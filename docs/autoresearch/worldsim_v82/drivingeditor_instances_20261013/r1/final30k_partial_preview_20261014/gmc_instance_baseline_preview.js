/* 真实 baseline 预览；没有 after，不执行编辑或 fetch。 */
'use strict';
(() => {
  const payload=document.getElementById('gmc-preview-data');
  if(!payload)return;
  const D=JSON.parse(payload.textContent),$=id=>document.getElementById('gmc-preview-'+id);
  const panel=$('panel'),button=$('open');
  const old=id=>document.getElementById(id);
  const legacyNodes=['scheme','notice','gallery','metric-title','scene-conclusion'].map(old).concat(document.querySelector('section.controls'),old('metric-body').closest('.table-wrap'));
  const s={active:false,mode:'observed',frame:0,actor:0,branch:'paired',crop:false,timer:null};
  let legacyHidden=[],legacyMode='observed';
  const n=(value,digits=6,missing='未保存 / 暂缺')=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):missing;
  const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
  const blackSelected=()=>document.querySelector('[data-scene="black"]').getAttribute('aria-pressed')==='true';
  const frame=()=>D.streams[s.mode].frames[s.frame];
  const count=()=>D.streams[s.mode].frames.length;
  const fixedReady=D.streams.fixed_camera.frames.length>0;
  const hasBox=box=>Array.isArray(box)&&box.length===4&&box[2]>box[0]&&box[3]>box[1];
  function stop(){if(s.timer!==null)clearInterval(s.timer);s.timer=null;$('play').textContent='播放';}
  function close(restoreMode=false){stop();if(s.active){legacyNodes.forEach((node,index)=>{node.hidden=legacyHidden[index];});if(restoreMode)document.querySelectorAll('[data-mode]').forEach(node=>node.setAttribute('aria-pressed',String(node.dataset.mode===legacyMode)));}s.active=false;panel.hidden=true;button.setAttribute('aria-pressed','false');}
  function syncScene(){button.hidden=!blackSelected();if(!blackSelected())close();}
  function open(){if(s.active){close(true);return;}if(!blackSelected())return;if(old('play').textContent==='暂停')old('play').click();legacyMode=document.querySelector('[data-mode][aria-pressed="true"]')?.dataset.mode||'observed';legacyHidden=legacyNodes.map(node=>node.hidden);legacyNodes.forEach(node=>{node.hidden=true;});document.querySelectorAll('[data-mode]').forEach(node=>node.setAttribute('aria-pressed','false'));s.active=true;panel.hidden=false;button.setAttribute('aria-pressed','true');render();}
  function card(title,src,note,box){const article=el('article',undefined,'gmc-card'),surface=el('div',undefined,'gmc-surface'),image=el('img');article.append(el('header',title));image.src=src;image.alt=title+' · '+frame().frame_id;image.decoding='async';
    if(s.crop&&hasBox(box)){const[x0,y0,x1,y1]=box,w=x1-x0,h=y1-y0;surface.style.aspectRatio=w+'/'+h;image.style.width=480/w*100+'%';image.style.left=-x0/w*100+'%';image.style.top=-y0/h*100+'%';}
    image.onerror=()=>{article.append(el('p','实际PNG读取失败：'+src,'gmc-missing'));stop();};surface.append(image);article.append(surface,el('footer',note));return article;}
  function metric(label,value){const row=el('tr');row.append(el('td',label),el('td',value));return row;}
  function render(){const f=frame(),fixed=s.mode==='fixed_camera',item=f.instances[s.actor],box=fixed?null:item.bbox;
    if(fixed||!hasBox(box))s.crop=false;panel.classList.toggle('gmc-fixed',fixed);$('observed').setAttribute('aria-pressed',String(!fixed));$('fixed').setAttribute('aria-pressed',String(fixed));
    $('actor').value=String(s.actor);$('branch').value=s.branch;$('branch').querySelector('[value="paired"]').textContent=fixed?'完整场景＋静态 RGB（无 GT）':'GT / 编辑前合成 / static';$('crop').disabled=fixed;$('crop').querySelector('[value="roi"]').disabled=!hasBox(box);$('crop').value=s.crop?'roi':'full';
    $('timeline').max=String(count()-1);$('timeline').value=String(s.frame);$('prev').disabled=s.frame===0;$('next').disabled=s.frame===count()-1;$('track-id').textContent=D.instance_ids[s.actor];
    $('fixed').disabled=!fixedReady;$('fixed').textContent=fixedReady?'固定相机49时间 · 无 GT':'固定相机 · 尚未导出';
    $('frame-title').textContent=(fixed?'固定相机模型时间 ':'真实相机帧 ')+(s.frame+1)+' / '+count()+' · '+f.frame_id;$('frame-detail').textContent='相机 '+f.camera_frame_id+' · 模型时间 '+n(f.model_time,2)+' · checkpoint '+D.checkpoint.step;
    $('notice').classList.toggle('fixed',fixed);$('notice').textContent=D.partial?'已导出'+D.streams.observed.frames.length+'/49真实逐帧相机重建。资源已优先让给其他任务；固定相机和两车编辑尚未导出，没有after。':fixed?'相机锁在真实0008033，只改变49个模型时间。单列显示未编辑诊断，没有GT或PSNR。':'每帧使用自己的实际相机：GT、完整重建、static均为真实30k保存结果。两车编辑正在导出，尚未提供after。';
    $('crop-hint').textContent=fixed?'固定相机诊断保留完整480×320。':hasBox(box)?'ROI由该ID真实body PNG扩展40像素；所有分支使用相同区域。':'该ID真实body为空：保留完整图和空分母/null。';
    const cards=[];
    if(s.branch==='paired'){if(!fixed)cards.push(card('GT · 真实未编辑原图',f.images.gt,'包含原始车辆；不是删除后的真值',box));cards.push(card('Before · 真实30k完整重建',f.images.composite,'编辑前baseline；after尚未提供',box),card('Static · 独立静态背景',f.images.static_rgb,'真实保存的静态分支；本预览不验收补景和残影',box));}
    else if(s.branch==='visible_rgb')cards.push(card('所选ID · 全场景可见RGB · 未编辑',item.images.visible_rgb,'共同排序后的加权颜色贡献；黑画布不补环境',box));
    else{cards.push(card('所选ID · 全场景可见alpha · 未编辑',item.images.visible_alpha,'共同排序后的alpha×T；黑0白1，无预乘mask或逐图归一化',box));if(!fixed)cards.push(card('真实body mask · 所选ID',item.images.body_mask,'真实reviewed DEVA/SAM绑定，不是3D框合成mask',box));}
    $('gallery').replaceChildren(...cards);
    if(fixed)$('metric-body').replaceChildren(metric('静态RGB MAE · 相对原相机时间',n(f.diagnostic.static_RGB_MAE_vs_original_camera_time)),metric('所选ID可见alpha总和',n(f.diagnostic.instance_visible_alpha_sum[s.actor],3)),metric('GT / PSNR / 人工质量','无GT / 无PSNR / null'));
    else{const m=item.metrics;$('metric-body').replaceChildren(metric('完整重建 PSNR · 原始GT',n(f.metrics.RGB_PSNR_clipped_db,3)),metric('完整重建 SSIM · 原始GT',n(f.metrics.RGB_SSIM_clipped,4)),metric('所选ID真实body像素数',String(m.body_pixels)),metric('所选ID · 完整HxW mask MSE',n(m.joint_mask_MSE_full_HxW)),metric('所选ID · body内可见alpha均值',n(m.joint_visible_alpha_body_mean,6,m.body_pixels===0?'空 body':'未保存 / 暂缺')),metric('所选ID · body外可见alpha均值',n(m.joint_visible_alpha_outside_mean)),metric('After / 人工质量','尚未导出 / null'));}
    $('evidence').textContent=(D.synthetic_fixture?'SYNTHETIC FIXTURE · 非结果 · ':'')+(D.partial?'仅显示已导出的'+D.streams.observed.frames.length+'个观察帧，不提供全49质量均值。':'真实30k评估含49观察帧与49固定相机时间。')+'after未提供；人工质量评价null。';
  }
  function step(delta){s.frame=Math.max(0,Math.min(count()-1,s.frame+delta));render();}
  function play(){if(s.timer!==null){stop();return;}if(s.frame===count()-1)s.frame=0;$('play').textContent='暂停';render();s.timer=setInterval(()=>{if(s.frame===count()-1){stop();return;}step(1);},1000/6);}
  button.addEventListener('click',open);document.querySelectorAll('[data-scene]').forEach(node=>node.addEventListener('click',()=>{close();syncScene();}));document.querySelectorAll('[data-mode]').forEach(node=>node.addEventListener('click',()=>close()));
  $('observed').addEventListener('click',()=>{stop();s.mode='observed';render();});$('fixed').addEventListener('click',()=>{if(!fixedReady)return;stop();s.mode='fixed_camera';s.crop=false;render();});
  $('actor').addEventListener('change',event=>{stop();s.actor=Number(event.target.value);render();});$('branch').addEventListener('change',event=>{s.branch=event.target.value;render();});$('crop').addEventListener('change',event=>{s.crop=event.target.value==='roi';render();});$('timeline').addEventListener('input',event=>{stop();s.frame=Number(event.target.value);render();});$('prev').addEventListener('click',()=>{stop();step(-1);});$('next').addEventListener('click',()=>{stop();step(1);});$('play').addEventListener('click',play);
  document.addEventListener('keydown',event=>{if(!s.active||event.ctrlKey||event.altKey||event.metaKey||event.isComposing||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;if(['ArrowLeft','ArrowRight'].includes(event.key)||(event.code==='Space'&&event.target.tagName!=='BUTTON')){event.preventDefault();event.stopImmediatePropagation();if(event.key==='ArrowLeft'){stop();step(-1);}else if(event.key==='ArrowRight'){stop();step(1);}else play();}},true);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});syncScene();
  if(location.hash==='#instances-preview'){document.querySelector('[data-scene="black"]').click();open();}
})();
