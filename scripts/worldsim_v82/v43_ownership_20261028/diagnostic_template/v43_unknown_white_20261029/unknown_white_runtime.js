(function(root){
  'use strict';
  const a=root.SceneViewer,R=root.R2Inherited,M=root.UNKNOWN_WHITE_DELIVERY;
  if(!a||!R||!M)return;
  const get=id=>document.getElementById(id),panel=get('unknownWhitePanel'),canvas=get('unknownWhiteCanvas');
  const aliases={v43_r2_inspect49:'v43_actor_10000',v43_r2_ownership49:'v43_actor_10000'};
  let epoch=0,lastKey='',displayed=null,promise=Promise.resolve();
  function lookup(){
    const group=aliases[S.group]||S.group,p=R.parameters();
    return {group,raw:S.raw,radius:p.radius,fraction:p.fraction,record:M.groups?.[group],alias:group!==S.group};
  }
  function validate(q,entry){
    if(M.schema!=='v43-native-unknown-white-v1'||M.gpu_executed!==true)return false;
    if(!['complete','partial'].includes(M.status)||entry?.status!=='verified')return false;
    if(M.context!==D.context||entry.group_id!==q.group||Number(entry.raw)!==q.raw)return false;
    if(Number(entry.radius)!==q.radius||Number(entry.fraction)!==q.fraction)return false;
    const c=entry.checks||{},keys=['alpha_identical','radii_identical','depth_sum_identical','extra_identical','baseline_saved_png_pixels_identical','non_color_inputs_identical','known_row_SH_identical','row_order_unchanged','native_environment_unchanged','frozen_model_unchanged'];
    return keys.every(k=>c[k]===true)&&Number.isFinite(c.actual_native_reference_max_error)&&c.actual_native_reference_max_error<=2e-6&&!!entry.images?.unknownwhite&&!!entry.receipt;
  }
  function clear(){
    epoch++;displayed=null;canvas.hidden=true;
    canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
    get('unknownWhiteReceipt').hidden=true;get('unknownWhiteReceipt').removeAttribute?.('href');
  }
  async function update(force=false){
    const q=lookup(),key=[S.group,q.raw,q.radius,q.fraction,S.channel,get('geometryState').value].join(':');
    if(!force&&key===lastKey)return promise;
    lastKey=key;clear();const ticket=epoch;
    panel.hidden=!q.record&&!S.group.startsWith('v43');
    if(panel.hidden)return;
    const isReference=get('geometryState').value==='baseline'||S.channel!=='rgb';
    get('unknownWhiteFrame').textContent=(q.record?.title||selected().title)+' · raw'+q.raw+' · '+q.radius+'px / '+Math.round(q.fraction*100)+'%';
    get('unknownWhitePairNote').textContent=q.alias?'本图对应右侧完整Actor10K几何；上方R2剥离RGB是另一诊断阶段。':'与上方原RGB共用当前组、帧、相机和编辑变换；右侧灰色unknown在本图中使用白色材质，其余高斯保留原颜色。';
    if(isReference){get('unknownWhiteStatus').textContent='当前为原场景/实拍参照通道。切回“当前实验/编辑状态”和原生RGB后查看匹配的unknown涂白对照。';return;}
    if(!q.record){get('unknownWhiteStatus').textContent='此历史实验尚未适配最新R2归属，本面板不借用其他组的结果。';return;}
    if(q.record.status==='evidence_needed'){get('unknownWhiteStatus').textContent=q.record.reason;return;}
    const entry=q.record.frames?.[String(q.raw)];
    if(!validate(q,entry)){
      get('unknownWhiteStatus').textContent=entry?.status==='verified'?'当前轮廓参数没有匹配的原生涂白图。旧参数图片已隐藏；需要按新参数重渲染。':'CPU接线已完成，等待GPU原生渲染。本组尚无unknown涂白RGB；上方原图和右侧真实几何可继续查看。';
      return;
    }
    get('unknownWhiteStatus').textContent='正在读取同帧原生unknown涂白结果…';
    promise=(async()=>{
      try{
        const im=await loadImage(entry.images.unknownwhite);if(ticket!==epoch)return;
        imageOn(canvas,im);canvas.hidden=false;displayed={...q,path:entry.images.unknownwhite};
        get('unknownWhiteReceipt').href=entry.receipt;get('unknownWhiteReceipt').hidden=false;
        get('unknownWhiteStatus').textContent='已核验原生渲染 · 仅改unknown的SH颜色；opacity、尺度、位置、旋转、数量和环境均保持本组原值。白色会受原遮挡与透明合成影响，不保证每个相关像素纯白。';
      }catch(error){if(ticket!==epoch)return;get('unknownWhiteStatus').textContent='本帧涂白图加载失败：'+error.message;canvas.hidden=true;}
    })();return promise;
  }
  const previousShow=show;show=async function(...args){const p=previousShow(...args);update();await p;await update();};
  const previousDraw=draw3D;draw3D=function(...args){const v=previousDraw(...args);update();return v;};
  const previousPixels=drawPixels;drawPixels=async function(...args){const v=await previousPixels(...args);if(displayed)update(true);return v;};
  get('unknownWhiteRetry').onclick=()=>update(true);
  canvas.onclick=event=>{const r=canvas.getBoundingClientRect();S.pixel=[Math.min(479,Math.max(0,Math.floor((event.clientX-r.left)/r.width*480))),Math.min(319,Math.max(0,Math.floor((event.clientY-r.top)/r.height*320)))];drawPixels(S.epoch);};
  Object.assign(a,{show,draw3D,drawPixels});
  root.UnknownWhite={manifest:M,lookup,validate,update,clear,get displayed(){return displayed;},get pending(){return promise;}};
  update();
})(window);
