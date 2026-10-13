"""One canonical viewer, using existing observed-frame PNGs and saved diagnostics."""
from pathlib import Path
import json,shutil,re
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs';B=W/'work/drivingeditor_resume/unified_viewer_20261013';B.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
four=load(O/'drivingeditor_four_arm_30k/manifest.json')
profiles=[('silver','银车路口','scene17','drivingeditor_four_arm_30k'),('black','黑车 · GMC＋Jeep','scene0008','drivingeditor_GMC_GN11_30k'),('red','红车横穿路口','scene0029','drivingeditor_red_GN11_30k')]
data={}
def frame(f,prefix,q=None):
    if q is None:
        r=f.get('metrics',{});q={'full':r.get('full',{}).get('composite_psnr_db'),'actor':r.get('actor',{}).get('composite_psnr_db'),'static':r.get('non_sky_outside_actor',{}).get('static_psnr_db')}
    return {'id':f['id'].split('/')[-1],'t':f['timestamp'],'images':{k:prefix+'/'+v for k,v in f['images'].items()},'bbox':f.get('bbox'),'q':q,'time_metrics':f.get('metrics',{}) if 'static_rgb_MAE_from_anchor' in f.get('metrics',{}) else {}}
for key,label,scene,folder in profiles:
    m=four if key=='silver' else load(O/folder/'manifest.json');arms={}
    if key=='silver':
        for arm in four['arm_order']:
            a=four['arms'][arm];obs=[]
            for i,f in enumerate(a['frames']):
                v=four['comparison']['per_frame'][i]['values'][arm]
                obs.append(frame(f,folder,{'full':v['full_composite_psnr_db'],'actor':v['actor_composite_psnr_db'],'static':v['static_bg_psnr_db']}))
            arms[arm]={'observed':obs,'fixed':[frame(f,folder) for f in a['static_time_frames']]}
    else:arms['GN11']={'observed':[frame(f,folder) for f in m['frames']],'fixed':[frame(f,folder) for f in m['static_time_frames']]}
    for arm,a in arms.items():
        assert len(a['observed'])==len(a['fixed'])==49
        for fs in a.values():
            for f in fs:
                for p in f['images'].values():assert (O/p).is_file(),p
    data[key]={'label':label,'scene':scene,'anchor':arms[next(iter(arms))]['observed'][10]['id'],'arms':arms}
payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</',r'<\/')
html=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DrivingEditor · 三场景统一查看</title>
<style>
:root{color-scheme:dark;--bg:#090f19;--panel:#101c2d;--line:#2a3a51;--text:#e3edf9;--muted:#a0b2c9;--accent:#68d8c1}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1340px;margin:auto;padding:28px 24px 50px}h1{font-size:29px;margin:5px 0 8px}h2{font-size:18px}p{color:var(--muted);margin:8px 0}.eyebrow{font-size:11px;letter-spacing:2px;color:var(--accent)}a{color:var(--accent)}button,select{font:inherit;color:inherit;background:#16263a;border:1px solid #3a4e69;border-radius:9px;padding:8px 13px;cursor:pointer}button[aria-pressed=true]{background:#1d4943;border-color:#60cbb8;color:#bbf8e9}button:disabled{opacity:.35;cursor:default}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.scenes,.modes,.buttons{display:flex;gap:9px;flex-wrap:wrap}.scenes{margin:20px 0 12px}.modes{margin:14px 0}.pill{display:inline-block;border:1px solid #405570;padding:2px 9px;border-radius:7px;color:#bed0e6;font-size:12px}.notice{border:1px solid #38526b;background:#13243a;border-radius:10px;padding:12px 15px;margin:14px 0}.notice.fixed{border-color:#786346;background:#28251e}.controls{border:1px solid var(--line);background:var(--panel);border-radius:13px;padding:14px;margin:15px 0}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.row .label{color:var(--muted);font-size:13px}.frame{flex:1;font-weight:600;min-width:190px}#frame-detail{font-weight:400;font-size:12px;color:var(--muted)}input[type=range]{width:100%;accent-color:var(--accent);margin:17px 0 6px}.hint{font-size:12px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{border:1px solid var(--line);border-radius:13px;background:var(--panel);overflow:hidden;min-width:0}.heading{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 14px}.heading strong{font-size:15px}.heading small{font-size:11px;color:var(--muted)}.surface{aspect-ratio:3/2;position:relative;overflow:hidden;background:#000}.surface img{position:absolute;left:0;top:0;width:100%;height:auto;max-width:none;display:block}.card footer{padding:7px 14px;font-size:12px;color:var(--muted)}.gt .heading{background:#172b39}.baseline .heading{background:#1b2b43}.joint .heading{background:#17382f}.matrix{display:grid;gap:10px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 12px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}th{color:#b9cde5}.conclusion{margin-top:20px;padding:13px 16px;background:#132033;border-left:3px solid #7ca8ef;border-radius:6px}.conclusion p{color:#b7c8dd;font-size:13px}details{border:1px solid var(--line);border-radius:10px;padding:12px 15px;margin-top:17px}summary{cursor:pointer;color:#b7cbe4}details img{max-width:100%;margin:15px 0}details ul{color:var(--muted);font-size:13px}.foot{color:#7d94b3;font-size:12px;margin-top:20px}[hidden]{display:none!important}@media(max-width:740px){main{padding:19px 13px 35px}h1{font-size:24px}.grid{grid-template-columns:1fr}.heading{padding:10px 12px}.heading small{font-size:10px}.scenes button{flex:1;padding:8px;font-size:13px}.modes button{font-size:13px}.controls select{max-width:100%}.frame{min-width:120px}.row{gap:8px}}
</style></head><body><main><div class="eyebrow">DRIVINGEDITOR · 30K</div><h1>银车、黑车、红车：统一查看</h1><p>默认显示真实49帧的静态分支。切换场景和分支即可；固定相机诊断单独查看。</p>
<nav class="scenes" aria-label="选择场景"><button data-scene="silver">银车 · 四臂对照</button><button data-scene="black">黑车 · SGCM＋NPS</button><button data-scene="red">红车 · SGCM＋NPS</button></nav>
<div id="scheme" class="pill"></div><nav class="modes" aria-label="相机模式"><button data-mode="observed">① 真实逐帧重建</button><button data-mode="fixed">② 固定相机 · 静态性检查</button></nav><div id="notice" class="notice"></div>
<section class="controls"><div class="row"><div class="buttons"><button id="prev" aria-label="上一帧">←</button><button id="play">播放</button><button id="next" aria-label="下一帧">→</button></div><div class="frame"><span id="frame-title"></span><div id="frame-detail"></div></div><label class="label" for="branch">查看分支</label><select id="branch"></select><label class="label" for="crop">视野</label><select id="crop"><option value="full">完整原图</option><option value="roi">同一车体区域</option></select></div><input id="timeline" aria-label="帧进度" type="range" min="0" max="48" value="10"><div class="hint" id="crop-hint"></div></section>
<div id="gallery" class="grid"></div><h2 id="metric-title">本帧质量 · 训练观测重建</h2><div class="table-wrap"><table><thead id="metric-head"></thead><tbody id="metric-body"></tbody></table></div>
<div id="scene-conclusion" class="conclusion"></div>
<details><summary>实验方案、资源与证据边界</summary><p>C00：SGCM关 / NPS关；G10：仅SGCM；N01：仅NPS；GN11：SGCM＋NPS。银车有四臂配对实验；黑车、红车只训练GN11，没有各自C00。</p><img src="drivingeditor_red_GN11_30k/architecture-components.svg" alt="输入RGB、LiDAR、mask经静态和动态高斯、SGCM与NPS，输出合成与分层图像"><p>六组各30k，共180k；10k/20k/30k保存节点通过文件/header检查，完整模型/Adam/RNG恢复未实测。网页使用既有真实PNG，未重新训练或渲染。人工GN11及两个新scene评价仍为null。</p><img src="DrivingEditor_resource_and_speed.png" alt="六组完整训练显存和耗时"><p>包括真实分割/SAM的最高显存观测10.279 GiB，通过3090 22 GiB预算容量初筛；采样峰是下界，目标3090未经实测。模型保留远端，定时任务已暂停，已发出关机请求，独立平台电源状态不可读。</p><p>银车GN11−C00全图+0.325 dB、车体−0.129 dB；已通过既定配对工程门槛，保留局部代价。黑车39/49、红车32/49个有效body帧；空body保留/null，不补0。全部是训练观测重建，没有heldout、删除后GT或固定相机反事实GT。</p><p><a href="drivingeditor_four_arm_30k/summary.html">四臂数值与坏例</a> · <a href="DrivingEditor_final_summary.json">原始汇总</a> · <a href="DrivingEditor_shutdown_receipt.json">关机请求凭据</a></p></details>
<p class="foot">RGB按保存FP32的既定clip转换展示；alpha为原raw覆盖度，黑0白1，没有逐图归一化或预先乘body mask。旧报告保留，后续统一使用本入口。</p></main>
<script id="scene-data" type="application/json">__DATA__</script><script>
'use strict';(()=>{
const D=JSON.parse(document.getElementById('scene-data').textContent),$=id=>document.getElementById(id);
const labels={static:'静态背景 RGB',dynamic:'动态车辆 RGB',composite:'完整重建 RGB',alpha_d:'动态透明度',alpha_s:'静态透明度',all:'合成＋静态＋动态总览'};
const method={C00:'C00 · 基线（SGCM关 / NPS关）',G10:'G10 · 仅SGCM',N01:'N01 · 仅NPS',GN11:'GN11 · SGCM＋NPS'};
const s={scene:'silver',mode:'observed',i:10,branch:'static',crop:false,playing:false,timer:null};
const params=new URLSearchParams(location.hash.slice(1));if(D[params.get('scene')])s.scene=params.get('scene');if(params.get('mode')==='fixed')s.mode='fixed';if(/^\d+$/.test(params.get('frame')||''))s.i=Math.max(0,Math.min(48,Number(params.get('frame'))));
function elem(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function stop(){if(s.timer)clearInterval(s.timer);s.timer=null;s.playing=false;$('play').textContent='播放';}
function f(arm){return D[s.scene].arms[arm][s.mode][s.i];}const num=(v,n=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'空body / 无分数';
function card(title,meta,src,arm,branch,bbox){const a=elem('article','card '+(arm==='GT'?'gt':arm==='C00'?'baseline':arm==='GN11'?'joint':'')),h=elem('header','heading');h.append(elem('strong','',title),elem('small','',meta));const surface=elem('div','surface'),img=elem('img');img.src=src;img.alt=title+' · '+f(Object.keys(D[s.scene].arms)[0]).id;img.dataset.arm=arm;img.dataset.branch=branch;
if(s.crop&&bbox){const [x0,y0,x1,y1]=bbox,w=x1-x0,hg=y1-y0;if(w>0&&hg>0){surface.style.aspectRatio=w+'/'+hg;img.style.width=(480/w*100)+'%';img.style.left=(-x0/w*100)+'%';img.style.top=(-y0/hg*100)+'%';}}
img.onerror=()=>{a.append(elem('p','','图片读取失败：'+src));};surface.append(img);a.append(h,surface,elem('footer','',(arm==='GT'?'GT原图（含车辆）':labels[branch])+' · '+(s.mode==='observed'?'该帧真实相机':'相机固定，模型时间变化')));return a;}
function setup(){const opts=s.mode==='fixed'?['static','alpha_s']:['static','dynamic','composite','alpha_d','alpha_s','all'];if(!opts.includes(s.branch))s.branch='static';$('branch').replaceChildren(...opts.map(v=>{const e=elem('option','',labels[v]);e.value=v;return e;}));$('branch').value=s.branch;render();}
function render(){const d=D[s.scene],arms=Object.keys(d.arms),base=f(arms[0]),bbox=s.mode==='fixed'?d.arms[arms[0]].observed[10].bbox:base.bbox;const valid=Array.isArray(bbox)&&bbox[2]>bbox[0]&&bbox[3]>bbox[1];if(!valid)s.crop=false;
document.querySelectorAll('[data-scene]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.scene===s.scene));document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode===s.mode));
$('scheme').textContent=d.label+' / '+(s.scene==='silver'?'C00、G10、N01、GN11四臂，同帧比较':'仅GN11：SGCM＋NPS')+' / 各30k';
$('notice').classList.toggle('fixed',s.mode==='fixed');$('notice').textContent=s.mode==='observed'?'真实逐帧重建：每一帧使用它自己的相机姿态和时间。下面直接显示这49个保存结果的分支；GT原图含车辆，不是删除车辆后的真值。':'固定相机诊断：相机锁在'+d.anchor+'，只改变49个模型时间。用于检查静态背景是否随时间变化；这些组合没有对应GT，不能当作逐帧重建质量。';
$('frame-title').textContent=(s.mode==='observed'?'真实相机帧 ':'固定相机时间 ')+(s.i+1)+' / 49 · '+base.id;$('frame-detail').textContent=(s.mode==='observed'?'相机随源帧变化':'相机固定在 '+d.anchor)+' · 模型时间 '+base.t.toFixed(2);$('timeline').value=s.i;$('prev').disabled=s.i===0;$('next').disabled=s.i===48;
$('crop').querySelector('[value=roi]').disabled=!valid;$('crop').value=s.crop?'roi':'full';$('crop-hint').textContent=(valid?'各方法使用同帧、同尺度、同一ROI。':'当前body为空，保留完整帧和空区域分数。')+' ← / →逐帧，空格播放。';
const cards=[];if(s.mode==='observed')cards.push(card('GT · 真实原图','包含真实车辆',base.images.gt,'GT','composite',bbox));
for(const arm of arms){const a=f(arm),branches=s.branch==='all'?['composite','static','dynamic']:[s.branch];for(const b of branches)cards.push(card(method[arm],labels[b],a.images[b],arm,b,bbox));}
$('gallery').replaceChildren(...cards);$('metric-title').textContent=s.mode==='observed'?'本帧质量 · 训练观测重建':'相对固定锚点的变化 · 不是GT质量分数';
const heads=s.mode==='observed'?['方法','全图合成 PSNR','车体合成 PSNR','非天空车体外静态 PSNR']:['方法','静态RGB MAE','静态RGB最大像素差','静态透明度 MAE'];const tr=elem('tr');heads.forEach(h=>tr.append(elem('th','',h)));$('metric-head').replaceChildren(tr);
$('metric-body').replaceChildren(...arms.map(arm=>{const a=f(arm),r=elem('tr'),q=a.q,m=a.time_metrics;[arm,...(s.mode==='observed'?[num(q.full),num(q.actor),num(q.static)]:[num(m.static_rgb_MAE_from_anchor,6),num(m.static_rgb_max_abs_from_anchor,6),num(m.static_alpha_MAE_from_anchor,6)])].forEach(v=>r.append(elem('td','',v)));return r;}));
const texts={silver:['银车没有达到干净补景和绝对静态。你看到的横道线黑块，在C00与GN11的同一固定相机时间都存在；GN11没有解决它。','49固定时间的RGB MAE：C00约0.02271，GN11约0.02154。略小只表示变化略减，不证明背景正确。NPS不是跨时间不变约束；静态位置/透明度仍可随时间变化。','body mask没有完整车影，可能使阴影被拟合到静态层；这是机制推断，具体残影的逐高斯归属尚未验证。'],black:['黑车场景使用GN11（SGCM＋NPS）。Corolla保留静态，GMC与Jeep合为一个动态分支。','保留接地阴影不足、静态道路黑残影和固定相机时间变化。末10帧body为空，分数留空；没有同场景C00，不能证明模块收益。'],red:['红车场景使用GN11（SGCM＋NPS）。真实DEVA130，末端3帧使用已审查的原SAM候选。','保留静态车底/轮胎残影；固定相机后段有移动黑块、细绿线与浅粉道路斑块。末17帧body为空，分数留空；没有同场景C00。']};$('scene-conclusion').replaceChildren(...texts[s.scene].map(t=>elem('p','',t)));
}
document.querySelectorAll('[data-scene]').forEach(b=>b.onclick=()=>{stop();s.scene=b.dataset.scene;s.i=10;s.mode='observed';s.branch='static';s.crop=false;setup();});document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{stop();s.mode=b.dataset.mode;s.branch='static';setup();});
$('branch').onchange=e=>{s.branch=e.target.value;render();};$('crop').onchange=e=>{s.crop=e.target.value==='roi';render();};$('timeline').oninput=e=>{s.i=Number(e.target.value);render();};function step(delta){s.i=Math.max(0,Math.min(48,s.i+delta));render();}$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
function play(){if(s.playing){stop();return;}if(s.i===48)s.i=0;s.playing=true;$('play').textContent='暂停';s.timer=setInterval(()=>{if(s.i===48){stop();return;}s.i++;render();},1000/6);}$('play').onclick=play;
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.ctrlKey||e.altKey||e.metaKey)return;if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}if(e.key==='ArrowRight'){e.preventDefault();step(1);}if(e.code==='Space'&&e.target.tagName!=='BUTTON'){e.preventDefault();play();}});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});setup();
})();</script></body></html>'''.replace('__DATA__',payload)
out=O/'DrivingEditor_final_report.html';backup=B/'DrivingEditor_final_report.before_unified.html'
if not backup.exists():shutil.copy2(out,backup)
out.write_text(html,encoding='utf-8')
# Also correct the old four-arm entry: default observed static, explicit mode names.
old=O/'drivingeditor_four_arm_30k/index.html';b=B/'four_arm_index.before_navigation.html'
if not b.exists():shutil.copy2(old,b)
t=old.read_text(encoding='utf-8');t=t.replace("branch:{observed:'composite',time:'static'}","branch:{observed:'static',time:'static'}")
t=t.replace('49帧重建</button>','真实逐帧重建</button>').replace('固定相机 · 49模型时间</button>','固定相机诊断 · 49模型时间</button>')
banner='<div style="padding:12px 16px;margin:12px 0;border:1px solid #4b8078;border-radius:9px;background:#15352f">银车四臂的<strong>真实逐帧静态分支</strong>为默认显示。相机固定检查在第二个标签。<a href="../DrivingEditor_final_report.html">银车／黑车／红车统一入口</a></div>'
if banner not in t:t=t.replace('<main class="shell">','<main class="shell">'+banner,1)
old.write_text(t,encoding='utf-8')
receipt={'status':'ONE_CANONICAL_THREE_SCENE_OBSERVED_STATIC_VIEWER_BUILT','canonical':'outputs/DrivingEditor_final_report.html','default_camera_mode':'observed','default_branch':'static','scene_count':3,'silver_arms':list(data['silver']['arms']),'black_arms':['GN11'],'red_arms':['GN11'],'frames_per_arm_and_mode':49,'images_reused_without_rendering':True,'old_report_files_preserved':True,'old_four_arm_default_corrected':True,'new_training_steps':0,'remote_connected':False,'human_GN11_verdict':None}
(B/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
