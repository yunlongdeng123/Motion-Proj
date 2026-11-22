const $=id=>document.getElementById(id);
const fmt=(x,n=3)=>x===null||x===undefined?'—':Number(x).toFixed(n);
const pct=x=>x===null||x===undefined?'—':(100*x).toFixed(2)+'%';
const names={rgb:'RGB',alpha:'目标Alpha',depth:'期望深度灰模',normal:'期望深度法向',grey:'灰影（辅助）'};
D.groups.filter(g=>!['pending','not_started'].includes(g.state)||g.role==='post_geometry_ablation').forEach(g=>$('group').add(new Option(g.label,g.id)));
D.cameras.forEach(c=>$('camera').add(new Option(c.label,c.id)));
$('group').value=D.groups.some(g=>g.id===location.hash.slice(1))?location.hash.slice(1):D.current_group;$('camera').value='raw110';
function selected(){return D.groups.find(g=>g.id===$('group').value)}
function textCell(row,value,cls=''){const td=document.createElement('td');td.textContent=value;if(cls)td.className=cls;row.append(td)}
function stageStatus(g){
  if(g.state==='pending')return g.label+' · '+g.summary;
  const eng=g.engineering_gate===false?'；灰模／场景验收未过':'';
  return g.label+' · '+(g.range_gate_passed?'渲染期望距离数值满足':'期望距离按实际收据')+(g.gate_passed===false?'；覆盖／场景验收未过':g.gate_passed===true?'；收据验收通过，场景准入另审':'')+eng;
}
function draw(){
  const g=selected();if(window.SceneReport?.render(g))return;
  const cam=$('camera').value,channel=$('channel').value,novel=cam.startsWith('novel_');
  const teacherAvailable=Object.values(g.views?.[cam]?.teacher||{}).some(Boolean);
  for(const id of ['left-mode','right-mode']){
    [...$(id).options].find(x=>x.value==='shared').disabled=novel;
    [...$(id).options].find(x=>x.value==='teacher').disabled=!teacherAvailable;
    if(novel&&$(id).value==='shared')$(id).value=id==='left-mode'&&teacherAvailable?'teacher':'isolated';
    if(!teacherAvailable&&$(id).value==='teacher')$(id).value='isolated';
  }
  $('status').textContent=stageStatus(g)+(g.teacher_bg_alpha_max!==null&&g.teacher_bg_alpha_max!==undefined?'；教师外域alpha max约'+g.teacher_bg_alpha_max:'');
  const range=g.range_summary,real=g.real||[],a=real.find(x=>x.raw===86),b=real.find(x=>x.raw===110);
  const stats=[[(range?range.count+'个不透明面返回点：':'')+'渲染期望距离 MAE/P90',range?fmt(range.mae_m,6)+' / '+fmt(range.p90_m,6)+' m':'—','期望距离与灰模验收分开'],['单对象可见区覆盖',a||b?pct(a?.isolated_visible_alpha95_fraction)+' / '+pct(b?.isolated_visible_alpha95_fraction):'—','raw86 / raw110'],['场景内可见贡献',a||b?pct(a?.visible_alpha95_fraction)+' / '+pct(b?.visible_alpha95_fraction):'—','固定真实V域'],['其他车辆变化像素',a||b?(a?.retained_K.RGB_changed_pixels_gt_1over255??'—')+' / '+(b?.retained_K.RGB_changed_pixels_gt_1over255??'—'):'—','变化阈值 >1/255']];
  $('stats').replaceChildren();for(const [label,value,note] of stats){const el=document.createElement('div');el.className='stat';for(const [c,t] of [['label',label],['value',value],['label',note]]){const e=document.createElement('div');e.className=c;e.textContent=t;el.append(e)}$('stats').append(el)}
  for(const side of ['left','right']){
    const mode=$(side+'-mode').value;let actualChannel=channel;
    if(novel&&side==='right'&&$('left-mode').value!=='teacher'&&$('right-mode').value!=='teacher'){actualChannel=channel==='depth'?'normal':'depth';if(!g.views?.[cam]?.[mode]?.[actualChannel]&&g.views?.[cam]?.[mode]?.grey)actualChannel='grey'}
    $(side+'-title').textContent=(D.cameras.find(x=>x.id===cam)?.label||cam)+' · '+({shared:'共享',isolated:'统一a20场',teacher:'纯T'}[mode])+' · '+names[actualChannel];
    const target=$(side+'-image');target.replaceChildren();const src=g.views?.[cam]?.[mode]?.[actualChannel];
    if(src){const link=document.createElement('a');link.href=src;link.target='_blank';const im=document.createElement('img');im.src=src;im.alt=g.label+' '+cam+' '+mode+' '+actualChannel;link.append(im);target.append(link)}
    else{const e=document.createElement('div');e.className='empty';e.textContent=g.state==='pending'?'暂无实际导出':mode==='shared'&&actualChannel!=='rgb'&&actualChannel!=='alpha'?'未导出此共享几何通道':'此通道未导出';target.append(e)}
  }
  $('receipts').replaceChildren();if(g.smoke_counts){const t=document.createElement('span');t.className=g.smoke_passed?'good':'muted';t.textContent='smoke '+g.smoke_counts.main+' +重放'+g.smoke_counts.replay+' +丢弃探针'+g.smoke_counts.discarded+'；工程'+(g.smoke_passed?'通过':'待确认')+'；重载'+(g.reload_passed?'通过':'待确认');$('receipts').append(t)}
  for(const x of g.receipts||[]){const a=document.createElement('a');a.href=x.href;a.textContent=x.label;a.target='_blank';$('receipts').append(a)}
  $('metrics').replaceChildren();for(const x of real){const tr=document.createElement('tr');textCell(tr,'raw'+x.raw);textCell(tr,pct(x.isolated_visible_alpha95_fraction));textCell(tr,pct(x.visible_alpha95_fraction));textCell(tr,x.retained_K.RGB_changed_pixels_gt_1over255);textCell(tr,fmt(x.real_rgb_L1,6));$('metrics').append(tr)}
  if(!real.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=5;td.textContent='尚无实际评价收据';td.className='muted';tr.append(td);$('metrics').append(tr)}
  $('grey-overview').replaceChildren();$('grey-panel').hidden=!g.grey_overview;if(g.grey_overview){const im=document.createElement('img');im.src=g.grey_overview;im.alt=g.label+' 辅助灰影六视';$('grey-overview').append(im)}
  $('extra-figures').replaceChildren();for(const f of g.extra_figures||[]){const box=document.createElement('div');box.className='architecture';const h=document.createElement('h3');h.textContent=f.title;const a=document.createElement('a');a.href=f.src;a.target='_blank';const im=document.createElement('img');im.src=f.src;im.alt=f.title;a.append(im);box.append(h,a);$('extra-figures').append(box)}
}
for(const id of ['group','camera','channel','left-mode','right-mode'])$(id).addEventListener('change',draw);
$('native-view').onclick=()=>{$('camera').value='raw110';$('channel').value='rgb';$('left-mode').value='shared';$('right-mode').value='isolated';draw()};
$('geometry-view').onclick=()=>{$('camera').value='novel_00';$('channel').value='depth';$('left-mode').value='teacher';$('right-mode').value='isolated';draw()};
for(const g of D.groups){const tr=document.createElement('tr');textCell(tr,g.label);textCell(tr,g.planned_steps??'—');textCell(tr,g.actual_steps??'—');textCell(tr,g.role==='scene_residual_cleanup'?'实际完成；视觉结论另列':g.state==='not_started'?'未启动':g.state==='pending'?'待实际结果':g.state==='exported'?(g.role==='initial'?'初始导出；0更新':'实际检验；0更新'+(g.engineering_gate===false?'；灰模未过':'')):g.engineering_gate===false?'实际完成；灰模／场景未过':g.gate_passed===false?'实际完成；数值验收未过':'实际完成；工程结论另审',g.engineering_gate===false?'badge':'muted');textCell(tr,g.summary||'');$('ledger').append(tr)}
for(const x of D.final_links||[]){const a=document.createElement('a');a.href=x.href;a.textContent=x.label;a.target='_blank';$('final-links').append(a)}
if(D.final_iteration){const f=D.final_iteration;$('final-summary').textContent='最终有效高斯 '+f.final_active_rows+'行（原始源'+f.final_active_by_source.old+'＋教师源'+f.final_active_by_source.TRELLIS+'）；累计退休原始源'+f.total_global_retired_old_rows+'。几何训练40步；移除检验0步；材质与联合精修未启动。单对象灰模失败与冻结场遮挡分别记录。'}
draw();
