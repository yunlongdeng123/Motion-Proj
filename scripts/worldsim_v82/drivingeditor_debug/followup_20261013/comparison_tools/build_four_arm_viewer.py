"""离线深色四臂查看页；正式清单必须完整，fixture必须显式允许。"""
import argparse
import html
import json
import math
from pathlib import Path
import re
from _legacy_c00_viewer import PAGE as LEGACY_PAGE

ARMS = ("C00", "G10", "N01", "GN11")


def validate_manifest(data, allow_fixture=False):
    fixture = data.get("mode") == "fixture_ui_only"
    if fixture and not allow_fixture:
        raise ValueError("Fixture is not a real four-arm result; explicit --allow-fixture required")
    if not fixture and (data.get("mode") != "real_four_arm_terminal" or data.get("status") != "PASS_READONLY_FOUR_ARM30K_CONTRACT"):
        raise ValueError("A real complete four-arm terminal manifest is required")
    if data.get("checkpoint") != 30000 or data.get("frame_count") != 49 or tuple(data.get("arm_order", [])) != ARMS:
        raise ValueError("Wrong checkpoint/frame/arm contract")
    if data.get("gate_evaluation_performed") is not False or data.get("human_verdict") is not None:
        raise ValueError("Viewer cannot write or claim gate/human acceptance")
    if not fixture and data.get("pairing", {}).get("status") != "PASS_ALL30000_FRAME_TIME_SHIFT_LR_ROWS":
        raise ValueError("Full paired sampler receipt is missing")
    baseline = data["arms"]["C00"]
    expected_ids = ["image_0/0017%03d" % index for index in range(49)]
    if data.get("scene") != "scene0017_000_048" or [frame["id"] for frame in baseline["frames"]] != expected_ids:
        raise ValueError("Baseline must use the fixed49 silver-scene FRONT identities")
    if any(not math.isclose(frame["timestamp"], -.48 + .02 * index, abs_tol=1e-12) for index, frame in enumerate(baseline["frames"])):
        raise ValueError("Baseline model timestamps do not match the fixed49 sequence")
    for arm in ARMS:
        record = data["arms"][arm]
        if len(record["frames"]) != 49 or len(record["static_time_frames"]) != 49:
            raise ValueError("All49 frames and all49 static-time controls must remain")
        if not fixture and (record["status"].get("status") != "complete" or record["status"].get("iteration") != 30000 or len(record["node_contracts"]) != 3):
            raise ValueError("Arm is not a complete three-node terminal result")
        for index, frame in enumerate(record["frames"]):
            other = baseline["frames"][index]
            if frame["id"] != other["id"] or not math.isclose(frame["timestamp"], other["timestamp"], abs_tol=1e-12) or frame["bbox"] != other["bbox"]:
                raise ValueError("Wrong observed frame/timestamp/sharedROI pairing")
            time = record["static_time_frames"][index]
            if time["id"] != frame["id"].split("/")[-1] or not math.isclose(time["timestamp"], frame["timestamp"], abs_tol=1e-12):
                raise ValueError("Wrong actual static-time/frame mapping")
        if record["static_time_summary"].get("anchor_pose_frame") != "0017010":
            raise ValueError("Actual static-time anchor must remain0017010")


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>__TITLE__</title><style>__LEGACY_CSS__
.shell{max-width:1300px}.grid.withgt{grid-template-columns:repeat(3,minmax(0,1fr))}.control-select{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}.control-select select{min-width:105px}.warning{padding:15px;border:1px solid #d5983d;background:#302416;color:#ffd68d;border-radius:12px;margin:15px 0}.numbers{font-variant-numeric:tabular-nums}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:11px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #223247}th{background:#18263a;color:#b8cae2;position:sticky;top:0}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#172b3c}.current-row{background:#183c37!important}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:10px;max-height:360px;overflow:auto;background:#09121d;border-radius:8px;padding:11px;color:#9fb3cd}.resource-note,.tiny{font-size:11px;color:var(--muted)}.fold .fold{margin-top:9px}.stat .value{font-size:19px}.nav-tabs{display:flex;gap:8px;margin:18px 0 12px}.nav-tabs button{background:#142235;border:1px solid #31445e;border-radius:8px;padding:8px 12px;color:#a9bbd3}.nav-tabs button[aria-pressed=true]{color:var(--accent);border-color:#448477;background:#1a3734}.metric-card .metric-value{font-size:20px}.arch-four{display:grid;grid-template-columns:repeat(2,1fr);gap:5px}.arch-four span{font-size:10px;border:1px solid #405872;border-radius:6px;padding:6px;text-align:center}.image-card img{cursor:zoom-in}.module-flags{font-size:11px;color:var(--muted);margin-top:6px}.frame-table-row{cursor:pointer}.source-label{font-size:10px;color:var(--faint);margin-left:auto}.table-tools{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:10px 0}.metric-pairs div{font-size:11px}@media(max-width:800px){.grid,.grid.withgt{grid-template-columns:1fr}.overview{grid-template-columns:repeat(2,1fr)}.metrics-grid{grid-template-columns:1fr}.architecture{grid-template-columns:1fr}.arch-arrow{transform:rotate(90deg)}}
.architecture.model-architecture{grid-template-columns:1fr 22px 1.25fr 22px .75fr 22px 1fr 22px 1fr;padding:12px;gap:6px}.model-architecture .arch-branch small{display:block;font-size:9px;color:var(--muted)}@media(max-width:800px){.architecture.model-architecture{grid-template-columns:1fr}}
</style></head><body><main class="shell">
<div class="eyebrow">DrivingEditor · 同合同完整四臂</div><div class="header"><div><h1 id="title"></h1><p class="subtitle">C00 · G10 · N01 · GN11 / 30k / 全49帧同步比较</p></div><div class="status"><i></i><span id="status-label">CPU 导出 · 只读合同核验</span></div></div>
<div id="fixture-warning" class="warning" hidden>FIXTURE · 仅以已有 C00 的素材重复四个 UI 槽位。不是四臂实验结果，不做 gate 或 human verdict。下面差值只用于验证交互。</div>
<p class="scope"><b>全49帧已观测开发重建。</b> RGB 固定 clip[0,1] 转8bit；动态 RGB/alpha 未预乘 body mask，alpha 固定灰度0–1。body-only mask 缺阴影；分支和背景分数不等同于编辑质量验收。</p>
<div class="overview" id="overview"></div>
<div class="nav-tabs"><button data-tab="observed" aria-pressed="true">49帧重建</button><button data-tab="time" aria-pressed="false">固定相机 · 49模型时间</button></div>
<div id="time-notice" class="time-notice" hidden>固定真实 pose 0017010、timestamp -.28，只替换49个实际 model timestamp。反事实组合没有对应GT，MAE变化不作为重建PSNR；速度与位移为模型单位。</div>
<div class="controls"><div class="control-row"><div class="navigation"><button class="button" id="prev">←</button><button class="button play" id="play">播放</button><button class="button" id="next">→</button></div><div class="frame-info"><div id="frame-label" class="frame-label"></div><div id="frame-meta" class="frame-meta"></div></div><span class="spacer"></span><label class="control-select">分支 <select id="branch"></select></label><label class="control-select">视野 <select id="crop"><option value="full">完整原图</option><option value="roi">共享车体 ROI</option></select></label><label class="control-select">速度 <select id="fps"><option value="6">6 FPS</option><option value="10">10 FPS</option></select></label><label class="control-select" id="gt-label"><input type="checkbox" id="show-gt">显示GT</label></div><div class="scrub-row"><span class="range-caption">1</span><input id="slider" type="range" min="0" max="48" value="0" aria-label="全部49帧"><span class="range-caption">49</span></div><div class="control-hint"><span>左右箭头逐帧 · 空格播放 · 四臂共享帧、时间和ROI</span><span id="display-rule"></span></div></div>
<div class="grid" id="panes"><article class="image-card" id="left-card"><div class="card-heading"><i class="branch-dot"></i><select id="left-arm" aria-label="左侧控制臂"></select><h2 id="left-branch"></h2><span class="source-label" id="left-module"></span></div><div class="image-surface"><img id="left-image" alt=""><span class="empty" hidden>图片未能载入</span></div><div class="card-footer" id="left-footer"></div></article><article class="image-card" id="right-card"><div class="card-heading"><i class="branch-dot"></i><select id="right-arm" aria-label="右侧控制臂"></select><h2 id="right-branch"></h2><span class="source-label" id="right-module"></span></div><div class="image-surface"><img id="right-image" alt=""><span class="empty" hidden>图片未能载入</span></div><div class="card-footer" id="right-footer"></div></article><article class="image-card" id="gt-card" hidden><div class="card-heading"><i class="branch-dot"></i><h2>同帧 GT</h2><span class="source-label">四臂原数组一致</span></div><div class="image-surface"><img id="gt-image" alt="同帧 GT"><span class="empty" hidden>图片未能载入</span></div><div class="card-footer">原始观测 · 同帧共享 ROI</div></article></div>
<div class="section-title">当前帧配对指标 <small>四臂同一 forward / 49帧全部保留</small></div><div class="metrics-grid" id="frame-metrics"></div>
<details class="fold" open><summary>全部49帧分数与 signed delta<span class="summary-detail">点击行跳转 / 统计和不利帧完整保留</span></summary><div class="fold-body"><div class="table-tools"><label class="control-select">指标 <select id="metric"></select></label><span class="tiny" id="metric-semantics"></span></div><div id="metric-statistics" class="metrics-grid"></div><div class="table-wrap" style="max-height:420px;margin-top:12px"><table id="frame-table"></table></div><p class="tiny">delta=arm−C00；PSNR越高越好。低动态alpha/支撑仅描述分层，不单独判定整体质量。最不利帧是终态后选择，未删除任何49帧。</p></div></details>
<details class="fold" open><summary>完整资源与耗时<span class="summary-detail">allocated / reserved / 进程与设备采样 / 全阶段</span></summary><div class="fold-body"><div class="table-wrap"><table id="memory-table"></table></div><p class="resource-note">GiB=bytes/2³⁰。采样峰是下界；父阶段包括子阶段，不能相加峰值。CUDA iter time只覆盖forward/backward，wall time覆盖完整流程。目标3090实跑与速度仍需独立验证。</p><div id="memory-details"></div></div></details>
<details class="fold"><summary>NPS / SGCM 已存证据<span class="summary-detail">全部节点梯度 / 采样 /145个SGCM事件</span></summary><div class="fold-body"><p class="tiny">NPS保留原sum/L2/.01；ratio_valid=false不能当作有效零比率。SGCM中心投影与temporal-alpha是支撑proxy，没有遮挡或精确逐高斯贡献。这里只显示日志，不认证完整恢复或填写质量结论。</p><div id="diagnostics"></div></div></details>
<details class="fold"><summary>组件图、合同与来源<span class="summary-detail">模型分支与数据流 / CPU证据 / 验证边界</span></summary><div class="fold-body"><div class="architecture model-architecture" role="img" aria-label="RGB、LiDAR、body和sky输入静态与动态高斯，共享EnvMap，经分层renderer输出RGB与alpha，进入四臂同帧比较"><div class="arch-module">RGB · LiDAR · pose<br>body /sky mask<small>49帧共同输入</small></div><div class="arch-arrow">→</div><div class="arch-components"><div class="arch-branch">静态高斯<small>SGCM /NPS 可选</small></div><div class="arch-branch dynamic">动态高斯<small>完整body alpha</small></div><div class="arch-env">共享 EnvMap</div></div><div class="arch-arrow">→</div><div class="arch-module">分层 renderer</div><div class="arch-arrow">→</div><div class="arch-output"><span>合成 /静态 /动态</span><span>RGB /raw alpha</span></div><div class="arch-arrow">→</div><div class="arch-module">四臂对照<small>C00 /G10 /N01 /GN11<br>同49帧 · 30k</small></div></div><p class="about-text" style="margin-top:12px">保留2024活跃约束、完整动态alpha、staticRGB、duration/velocity和LiDAR衰减；四臂只按两个flag变更模块。查看页只读FP32数组、JSON、日志和文件header/stat，不加载checkpoint，不执行GPU，不填写human或gate通过。</p><div id="contract-details"></div></div></details>
<footer class="footer"><span>离线 index.html + assets/ · 原始FP32与日志保留源路径</span><span>全观测开发重建 / 无独立test与editedGT</span></footer></main>
<dialog class="modal" id="lightbox"><div class="modal-top"><span id="lightbox-title"></span><span class="spacer"></span><button class="button" id="close-lightbox">关闭</button></div><div class="modal-image-wrap fit"><img id="lightbox-image" style="display:block;max-width:100%;margin:auto" alt="同帧原图"></div></dialog>
<script type="application/json" id="viewer-data">__DATA__</script><script>
'use strict';(()=>{
const data=JSON.parse(document.querySelector('#viewer-data').textContent),arms=data.arm_order,fixture=data.mode==='fixture_ui_only';
const $=s=>document.querySelector(s),state={tab:'observed',indices:{observed:0,time:0},branch:{observed:'composite',time:'static'},left:'C00',right:'GN11',roi:false,playing:false,timer:null};
const labels={gt:'GT',composite:'合成 RGB',static:'静态 RGB',dynamic:'动态 RGB',alpha_s:'静态 α',alpha_d:'动态 α',body_mask:'Body mask'},metricLabels={full_composite_psnr_db:'全图 · 合成 PSNR',actor_composite_psnr_db:'车体 · 合成 PSNR',static_bg_psnr_db:'非天空车外 · 静态 PSNR',outside_alpha_d_mean:'非天空车外 · 动态 α',outside_dynamic_rgb_support:'非天空车外 · 动态 RGB 支撑',outside_static_composite_difference:'非天空车外 · |合成−静态|'},timeKeys={static_rgb_MAE_from_anchor:'静态 RGB MAE',static_rgb_max_abs_from_anchor:'静态 RGB max |Δ|',static_alpha_MAE_from_anchor:'静态 α MAE'};
const numeric=v=>typeof v==='number'&&Number.isFinite(v),fmt=(v,n=4)=>numeric(v)?v.toFixed(n):'—',sign=(v,n=4)=>numeric(v)?(v>0?'+':'')+v.toFixed(n):'—';
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function options(select,items,selected){select.replaceChildren();Object.entries(items).forEach(([value,label])=>{const o=el('option','',label);o.value=value;select.append(o);});select.value=selected;}
function frame(arm){return data.arms[arm][state.tab==='time'?'static_time_frames':'frames'][state.indices[state.tab]];}
function value(arm,key,index){if(state.tab==='time')return data.arms[arm].static_time_frames[index].metrics[key];return data.comparison.per_frame[index].values[arm][key];}
function mean(arm,key){return Array.from({length:49},(_,i)=>value(arm,key,i)).reduce((a,b)=>a+b,0)/49;}
$('#title').textContent=data.title;$('#fixture-warning').hidden=!fixture;if(fixture)$('#status-label').textContent='FIXTURE · 非四臂终态';
options($('#left-arm'),Object.fromEntries(arms.map(a=>[a,a])),'C00');options($('#right-arm'),Object.fromEntries(arms.map(a=>[a,a])),'GN11');
arms.forEach(arm=>{const stat=el('div','stat'),record=data.comparison.arms[arm],score=record.equal_frame_means,ds=record.paired_delta_statistics_vs_C00;stat.append(el('div','label',arm+(fixture?' · C00 fixture槽':'')),el('div','value',fmt(score.full_composite_psnr_db,3)+' dB'),el('div','detail','全图49帧等权 · ∆C00 '+sign(ds.full_composite_psnr_db.mean,3)));$('#overview').append(stat);});
function setupTab(){options($('#branch'),Object.fromEntries((state.tab==='time'?['static','alpha_s']:Object.keys(labels)).map(k=>[k,labels[k]])),state.branch[state.tab]);options($('#metric'),state.tab==='time'?timeKeys:metricLabels,Object.keys(state.tab==='time'?timeKeys:metricLabels)[0]);$('#gt-label').hidden=state.tab==='time';$('#time-notice').hidden=state.tab!=='time';document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.tab===state.tab));render();renderTable();}
function crop(img){const base=frame('C00'),box=base.bbox;if(!state.roi){img.style.width='100%';img.style.left='0';img.style.top='0';return;}let[x0,y0,x1,y1]=box,w=x1-x0,h=y1-y0,cx=(x0+x1)/2,cy=(y0+y1)/2,ratio=480/320;if(w/h<ratio)w=h*ratio;else h=w/ratio;if(w>480){w=480;h=w/ratio;}if(h>320){h=320;w=h*ratio;}const x=Math.max(0,Math.min(480-w,cx-w/2)),y=Math.max(0,Math.min(320-h,cy-h/2));img.style.width=(480/w*100)+'%';img.style.left=(-x/w*100)+'%';img.style.top=(-y/h*100)+'%';}
function setImage(id,arm,branch){const img=$(id),f=frame(arm),path=f.images[branch];crop(img);img.alt=arm+' · '+labels[branch]+' · '+f.id;img.dataset.arm=arm;img.dataset.branch=branch;if(path)img.src=path;else img.removeAttribute('src');}
function metricCard(key,label){const card=el('div','metric-card');card.append(el('div','region',label));const pairs=el('div','metric-pairs');arms.forEach(arm=>{const r=el('div');r.append(el('span','',arm),el('strong','',fmt(value(arm,key,state.indices[state.tab]),key.includes('psnr')?3:6)));pairs.append(r);});const delta=value('GN11',key,state.indices[state.tab])-value('C00',key,state.indices[state.tab]);card.append(el('div','metric-value','∆ GN11 '+sign(delta,key.includes('psnr')?3:6)),pairs);return card;}
function render(){const i=state.indices[state.tab],f=frame('C00'),branch=state.branch[state.tab];$('#frame-label').textContent=(state.tab==='time'?'模型时间 ':'帧 ')+(i+1)+' /49 · '+f.id;$('#frame-meta').textContent='model time '+fmt(f.timestamp,6)+(state.tab==='time'?' · 固定pose0017010':' · raw '+i);$('#slider').value=i;$('#prev').disabled=i===0;$('#next').disabled=i===48;$('#display-rule').textContent=state.roi?'全部面板同一 C00 canonical ROI':'完整480×320';
['left','right'].forEach(side=>{const arm=state[side];setImage('#'+side+'-image',arm,branch);$('#'+side+'-branch').textContent=labels[branch];$('#'+side+'-module').textContent=fixture?'C00素材 · fixture':({C00:'SGCM0 /NPS0',G10:'SGCM1 /NPS0',N01:'SGCM0 /NPS1',GN11:'SGCM1 /NPS1'})[arm];$('#'+side+'-footer').textContent=arm+' · '+f.id+' · '+(branch.startsWith('alpha')?'raw α · 固定0–1灰度':'savedFP32 clip[0,1] · 未预mask');});
const show=$('#show-gt').checked&&state.tab==='observed';$('#gt-card').hidden=!show;$('#panes').classList.toggle('withgt',show);if(show)setImage('#gt-image','C00','gt');const keys=state.tab==='time'?Object.keys(timeKeys):['full_composite_psnr_db','actor_composite_psnr_db','static_bg_psnr_db'];$('#frame-metrics').replaceChildren(...keys.map(k=>metricCard(k,(state.tab==='time'?timeKeys:metricLabels)[k])));document.querySelectorAll('.frame-table-row').forEach(r=>r.classList.toggle('current-row',Number(r.dataset.index)===i));}
function renderTable(){const key=$('#metric').value,psnr=key.includes('psnr'),digits=psnr?3:6;$('#metric-semantics').textContent=state.tab==='time'?'相对锚点变化 / 无GT质量分数':data.metric_paths[key]+' /49帧等权';const cards=[];['G10','N01','GN11'].forEach(arm=>{const delta=Array.from({length:49},(_,i)=>value(arm,key,i)-value('C00',key,i)).sort((a,b)=>a-b),card=el('div','metric-card'),pairs=el('div','metric-pairs'),p10=delta[4]*.2+delta[5]*.8;card.append(el('div','region',arm+'−C00'),el('div','metric-value',sign(mean(arm,key)-mean('C00',key),digits)));[['mean '+arm,fmt(mean(arm,key),digits)],['median ∆',sign(delta[24],digits)],['p10 ∆',sign(p10,digits)],['min ∆',sign(delta[0],digits)],['negative frames',delta.filter(v=>v<0).length+'/49'],...(psnr?[['∆<−1dB',delta.filter(v=>v< -1).length+'/49']]:[])].forEach(([label,v])=>{const row=el('div');row.append(el('span','',label),el('strong','',String(v)));pairs.append(row);});card.append(pairs);cards.push(card);});$('#metric-statistics').replaceChildren(...cards);const table=$('#frame-table'),head=el('thead'),tr=el('tr');['帧 /model timestamp',...arms,'∆ G10','∆ N01','∆ GN11'].forEach(s=>tr.append(el('th','',s)));head.append(tr);const body=el('tbody');for(let i=0;i<49;i++){const row=el('tr','frame-table-row');row.dataset.index=i;const base=data.arms.C00[state.tab==='time'?'static_time_frames':'frames'][i];row.append(el('td','',base.id+' /'+fmt(base.timestamp,2)));arms.forEach(arm=>row.append(el('td','numbers',fmt(value(arm,key,i),digits))));['G10','N01','GN11'].forEach(arm=>row.append(el('td','numbers',sign(value(arm,key,i)-value('C00',key,i),digits))));row.addEventListener('click',()=>{state.indices[state.tab]=i;render();});body.append(row);}table.replaceChildren(head,body);render();}
function jsonFold(parent,title,value){const d=el('details','fold'),s=el('summary','',title),pre=el('pre');d.append(s,pre);d.addEventListener('toggle',()=>{if(d.open&&!pre.textContent)pre.textContent=JSON.stringify(value,null,2);});parent.append(d);}
const mh=el('thead'),mtr=el('tr');['臂','完整wall min','allocated GiB','reserved GiB','进程采样 GiB','设备采样 GiB','FW/BW iter ms','静态/动态点数'].forEach(s=>mtr.append(el('th','',s)));mh.append(mtr);const mb=el('tbody');arms.forEach(arm=>{const a=data.arms[arm],m=a.memory||{},status=a.status||{},pt=m.pytorch||{},nv=m.nvidia_smi||{},extra=m.extra||{},row=el('tr');const vals=[arm,numeric(status.wall_seconds)?fmt(status.wall_seconds/60,2):'—',...['peak_allocated_bytes','peak_reserved_bytes'].map(k=>numeric(pt[k])?fmt(pt[k]/2**30,3):'—'),...['process_peak_used_bytes','device_peak_used_bytes'].map(k=>numeric(nv[k])?fmt(nv[k]/2**30,3):'—'),fmt(status.iter_forward_backward_ms,2),numeric(extra.render_static_points)?extra.render_static_points.toLocaleString()+' /'+extra.render_dynamic_points.toLocaleString():'—'];vals.forEach(v=>row.append(el('td','numbers',String(v))));mb.append(row);jsonFold($('#memory-details'),arm+' 完整memory_summary与status',{memory:m,status});jsonFold($('#diagnostics'),arm+' NPS梯度 /SGCM事件',a.diagnostics||{fixture:true,evidence_available:false});jsonFold($('#contract-details'),arm+' 三节点合同与静态时间原摘要',{source:a.source,node_contracts:a.node_contracts,static_time_summary:a.static_time_summary});});$('#memory-table').replaceChildren(mh,mb);jsonFold($('#contract-details'),'完整配对与显示/来源边界',{pairing:data.pairing,source_gate_plan:data.source_gate_plan,display:data.display,scope:data.scope,checkpoint_loaded:data.checkpoint_loaded,restoration_verified:data.restoration_verified,gate_evaluation_performed:data.gate_evaluation_performed,human_verdict:data.human_verdict});
function stop(){clearInterval(state.timer);state.timer=null;state.playing=false;$('#play').textContent='播放';}function toggle(){if(state.playing){stop();return;}state.playing=true;$('#play').textContent='暂停';state.timer=setInterval(()=>{state.indices[state.tab]=(state.indices[state.tab]+1)%49;render();},1000/Number($('#fps').value));}function step(d){state.indices[state.tab]=Math.max(0,Math.min(48,state.indices[state.tab]+d));render();}
$('#prev').onclick=()=>step(-1);$('#next').onclick=()=>step(1);$('#play').onclick=toggle;$('#slider').oninput=e=>{state.indices[state.tab]=Number(e.target.value);render();};$('#fps').onchange=()=>{if(state.playing){stop();toggle();}};$('#left-arm').onchange=e=>{state.left=e.target.value;render();};$('#right-arm').onchange=e=>{state.right=e.target.value;render();};$('#branch').onchange=e=>{state.branch[state.tab]=e.target.value;render();};$('#crop').onchange=e=>{state.roi=e.target.value==='roi';render();};$('#show-gt').onchange=render;$('#metric').onchange=renderTable;document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{stop();state.tab=b.dataset.tab;setupTab();});
document.querySelectorAll('.image-surface img').forEach(img=>{img.onerror=()=>{img.nextElementSibling.hidden=false;};img.onload=()=>{img.nextElementSibling.hidden=true;};img.onclick=()=>{const arm=img.dataset.arm,branch=img.dataset.branch;$('#lightbox-title').textContent=arm+' · '+labels[branch]+' · '+frame(arm).id+' · 完整源图';$('#lightbox-image').src=img.src;$('#lightbox').showModal();};});$('#close-lightbox').onclick=()=>$('#lightbox').close();document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}if(e.key==='ArrowRight'){e.preventDefault();step(1);}if(e.code==='Space'&&e.target.tagName!=='BUTTON'){e.preventDefault();toggle();}});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});setupTab();
})();</script></body></html>'''


def build(manifest_path, output, allow_fixture=False):
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    validate_manifest(data, allow_fixture)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    css = re.search(r"<style>(.*?)</style>", LEGACY_PAGE, re.S).group(1)
    page = PAGE.replace("__LEGACY_CSS__", css).replace("__TITLE__", html.escape(data["title"])).replace("__DATA__", payload)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = output / "index.html"
    with result.open("x", encoding="utf-8") as stream:
        stream.write(page)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    print(build(args.manifest, args.output, args.allow_fixture))


if __name__ == "__main__":
    main()
