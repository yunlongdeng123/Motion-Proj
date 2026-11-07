(() => {
  const slider=document.getElementById('clip-frame');
  const status=document.getElementById('clip-status');
  const label=document.getElementById('clip-label');
  const controls=[...document.querySelectorAll('[data-clip-state]')];
  const play=document.getElementById('clip-play');
  const cards=[['clip-latest','frames/latest'],['clip-warm','evidence/control_final49_2000/Cwarm0/frames'],['clip-trained','evidence/control_final49_2000/material2000/frames']];
  let state='delete',frame=128,token=0,playing=false,timer;
  async function show(raw,nextState=state){
    const id=++token;raw=Math.max(89,Math.min(137,Number(raw)));
    const paths=cards.map(([,prefix])=>`${prefix}/${String(raw).padStart(3,'0')}/${nextState}.png`);
    status.textContent='加载同一帧的三份原图…';
    try {
      await Promise.all(paths.map(src=>new Promise((resolve,reject)=>{
        const im=new Image();im.onload=resolve;im.onerror=reject;im.src=src;
      })));
      if(id!==token)return;
      cards.forEach(([key],i)=>{const im=document.getElementById(key);im.src=paths[i];im.parentElement.href=paths[i];im.alt=`raw${raw} ${nextState} ${key}`;});
      frame=raw;state=nextState;slider.value=String(frame);
      label.textContent=`raw${String(frame).padStart(3,'0')} · ${state==='keep'?'KEEP':'DELETE a37'} · ${frame-88}/49`;
      controls.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.clipState===state)));
      status.textContent='三列已同步；点击原图查看完整480×320。';
    } catch(e) {if(id===token){status.textContent='原图未成功加载，请核对取回清单。';playing=false;play.textContent='播放';}}
  }
  async function tick(){
    if(!playing)return;
    await show(frame===137?89:frame+1,state);
    if(playing)timer=setTimeout(tick,200);
  }
  slider.addEventListener('input',()=>show(slider.value,state));
  controls.forEach(b=>b.addEventListener('click',()=>show(slider.value,b.dataset.clipState)));
  play.addEventListener('click',()=>{playing=!playing;play.textContent=playing?'暂停':'播放';clearTimeout(timer);if(playing)tick();});
  document.getElementById('clip-prev').addEventListener('click',()=>show(frame===89?137:frame-1,state));
  document.getElementById('clip-next').addEventListener('click',()=>show(frame===137?89:frame+1,state));
  window.scene008Control49={show,getState:()=>({frame,state,playing})};
  show(frame,state);
})();
