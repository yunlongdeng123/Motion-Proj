"""Same synchronized center viewer as the prior report, now with the real repaired4K state."""
from pathlib import Path
import json,base64,re,shutil
import numpy as np
W=Path(__file__).parent;I=W.parent/'identity_completion';O=W.parents[1]/'outputs/r15_repair_4k_20261018/tristate';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
old=read(I/'tristate/viewer_data.json');new=read(W/'tristate/viewer_panels.json');receipt=read(W/'tristate/receipt.json')
assert receipt['checkpoint_step']==4000 and receipt['parameter_updates']==0
old['panels']+=read(I/'tristate/old5_geometry/viewer_panels.json')['panels']+new['panels']
old['panels']=[p for p in old['panels'] if p['stage'] in ['init','4k','old5','repair4k']]
def array(p):return np.frombuffer(base64.b64decode(p['points']['data']),dtype='<f4').reshape(-1,12)
def put(p,a):p['points']=dict(rows=len(a),stride=12,data=base64.b64encode(a.astype('<f4').tobytes()).decode())
for p in old['panels']:
    a=array(p);limit=8000
    if len(a)>limit:a=a[np.linspace(0,len(a)-1,limit).astype(int)]
    put(p,a)
# Context views include the target asset, rather than replacing it with background.
for p in old['panels']:
    if p['scope'].endswith('context'):
        asset=next(x for x in old['panels'] if x['stage']==p['stage'] and x['scope']==p['scope'][:-7])
        put(p,np.vstack((array(asset),array(p))));p['total']+=asset['total']
        for k,n in asset['counts'].items():p['counts'][k]=p['counts'].get(k,0)+n
        p['coordinates']='target + real background in same Actor frame@raw113'
geo=read(W/'geometry_comparison.json');progress=read(W/'live_progress.json');rows=''
for a in [6,37,5,26,18]:
    orig,repair=[next(x for x in geo[k] if x['actor']==a) for k in ['original4k','repair4k']]
    last=next(x for x in progress['actor_geometry'] if x['actor']==a)
    rawcounts=next(x for x in receipt['actors'] if x['actor']==a)['counts']
    assert int(rawcounts.get('0',0))==repair['raw_status_counts']['0']
    panel=next(x for x in old['panels'] if x['stage']=='repair4k' and x['scope']=='a'+str(a))
    assert int((array(panel)[:,3]==0).sum())==repair['raw_status_counts']['0']
    rows+=f"<tr><td>a{a}</td><td>{orig['points']:,} → {repair['points']:,}</td><td>{orig['outside_exact_box']/orig['points']:.2%} → {repair['outside_exact_box']/repair['points']:.2%}</td><td>0 → {repair['raw_status_counts']['0']}</td><td>{last['outside_allowed_envelope']} / {last['drift_violations']}</td><td>{last['max_origin_displacement_m']:.6f}m</td></tr>"
source=(I/'build_tristate_report.py').read_text(encoding='utf-8')
js=source[source.index('const colors='):source.index('const actor=document.getElementById')]
start=js.index('function selectPanels(){');end=js.index('function transform(',start)
select='''function selectPanels(){let scope=scopes.value,oldPair=pair.value==='old5';context.disabled=oldPair;scopes.querySelector('option[value=all]').disabled=oldPair;if(oldPair){context.checked=false;if(scope==='all')scopes.value=scope='a6'}let stages=pair.value==='initial'?['init','repair4k']:oldPair?['old5','repair4k']:['4k','repair4k'];let key=scope+(context.checked&&scope!=='all'?'context':'');chosen=stages.map(s=>DATA.panels.find(p=>p.stage===s&&p.scope===key));chosen.forEach(p=>p&&decode(p));center=[0,0,0];radius=context.checked?9:3.3;if(scope==='all'){let vals=[[],[],[]];chosen.forEach(p=>{if(p)for(let i=0;i<p.array.length;i+=12)for(let j=0;j<3;j++)vals[j].push(p.array[i+j])});let lo=[],hi=[];for(let j=0;j<3;j++){vals[j].sort((a,b)=>a-b);lo[j]=vals[j][Math.floor(vals[j].length*.01)];hi[j]=vals[j][Math.floor(vals[j].length*.99)]}center=lo.map((v,j)=>(v+hi[j])/2);radius=Math.max(...hi.map((v,j)=>v-lo[j]))*.55}drawAll()}
'''
js=js[:start]+select+js[end:]
js=js.replace('let yaw=-.65',"const pair=document.getElementById('pair');pair.onchange=()=>{zoom=1;panx=pany=0;selectPanels()};let yaw=-.65")
js=js.replace('${p.stage} · ${p.coordinates}','${STAGES[p.stage]||p.stage} · ${p.coordinates}')
js=js.replace("ctx.fillRect(r[0]-1,r[1]-1,2,2)","let size=r[4]===0?4:2;ctx.fillRect(r[0]-size/2,r[1]-size/2,size,size)")
js=js.replace('状态原码计数 ${JSON.stringify(p.counts)}',"${p.stage==='old5'?'硬owner（无三值证据）':'raw状态计数'} ${JSON.stringify(p.counts)}")
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>修复R15 4K：三维归属与几何对照</title><style>
*{box-sizing:border-box}body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#0c1521;color:#dfeaf5;margin:auto;padding:26px;max-width:1500px}h1{font-size:28px}h2{font-size:21px;margin-top:28px}a{color:#8acaff}button,select{background:#1c3047;border:1px solid #536c85;color:#e9f4ff;border-radius:6px;padding:7px;margin:3px;font:inherit}button{cursor:pointer}.note,.card{background:#132439;border:1px solid #314963;border-radius:9px;padding:15px;margin:15px 0}.tools{display:flex;flex-wrap:wrap;align-items:center;gap:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.view{background:#09111c;border:1px solid #314963;border-radius:8px;overflow:hidden}canvas{display:block;width:100%;height:440px;touch-action:none}.caption{padding:10px;font-size:14px;min-height:75px}.details{white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace}.blue{color:#38a5f0}.green{color:#26e68c}.orange{color:#ffb627}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #314963}th{background:#18314a}.small{font-size:14px;color:#a5bdd3}img,svg{max-width:100%;height:auto}details{margin:22px 0}summary{cursor:pointer}@media(max-width:800px){body{padding:12px}.grid{grid-template-columns:1fr}canvas{height:370px}}</style>
<h1>修复 R15 4K：和原 4K 同视角旋转对比</h1>
<p><a href="../index.html">修复4K完整结果</a> · <a href="../../identity_comparison_20261018/index.html">五身份动态／静态／Delete 同步视频</a> · <a href="../../identity_comparison_20261018/tristate/index.html">原三维报告（旧5／原R15）</a></p>
<div class="note"><b>默认左侧是原R15 4K，右侧是修复R15 4K。</b>两边使用同一真实车框、米坐标和视角。修复版的外扩已明显受限，但 a6 下部白带、a37 删除后的静态车形仍在，整体资产质量未通过。</div>
<div class="tools"><label>对照 <select id="pair"><option value="original">原 R15 4K / 修复 R15 4K</option><option value="initial">相同初始化 / 修复 R15 4K</option><option value="old5">旧五身份 30K / 修复 R15 4K</option></select></label><label>身份 <select id="scope"><option value="a6">a6</option><option value="a37">a37</option><option value="a5">a5</option><option value="a26">a26</option><option value="a18">a18</option><option value="all">全场景</option></select></label><label><input id="context" type="checkbox">同时显示周围其他归属</label><input id="pool" type="checkbox" hidden></div>
<div class="tools"><label><input id="targetmode" type="checkbox" checked>按当前身份看 0/1/U</label><label class="blue"><input data-state="2" type="checkbox" checked>0：确定非目标／static</label><label class="green"><input data-state="1" type="checkbox" checked>1：目标／Actor</label><label class="orange"><input data-state="0" type="checkbox" checked>U：待定证据</label><button id="persp">斜视</button><button id="top">俯视</button><button id="side">侧视</button><button id="reset">重置</button></div>
<p class="small">拖动旋转，滚轮缩放，Shift+拖动平移，点击点查看归属、证据、来源与最大σ。两侧同步。数量来自全量状态，中心只抽样显示；修复版保留全部目标 unknown，以稍大橙点显示，不能按屏幕密度判断比例。背景框选 raw113；中心不代表完整高斯支撑或可见度。</p>
<div class="grid"><div class="view"><canvas id="left"></canvas><div class="caption" id="leftcap"></div></div><div class="view"><canvas id="right"></canvas><div class="caption" id="rightcap"></div></div></div>
<div class="card details" id="details">点击任意中心查看实际状态与硬owner。raw编码：0=unknown，1=Actor，2=static；概率为证据分数，不是经过校准的真值。</div>
<h2>实际变化（原4K → 修复4K）</h2><div style="overflow:auto"><table><tr><th>身份</th><th>点数</th><th>真实框外中心率</th><th>内部unknown</th><th>容许域外／漂移超限</th><th>最大累计位移</th></tr>@@ROWS@@</table></div>
<p>原4K点大多继承确定车辆状态；修复版通过1500/2500/3500步显式审计产生unknown，owner转移事件分别86/39/30次。状态仍是显式审计更新，不是Adam学出的归属。框外率下降不能证明初始化正确，也不能证明碎片、残影已清除。旧5没有三值证据，绿色只表示硬owner，点击的概率-1表示不存在。</p>
<h2>看 a6／a37 是否真的干净</h2><p>下图按相同raw113并排：上旧5、中原4K、下修复4K。列依次是完整场景、独立资产、静态分支、删除结果。物理阴影关闭。</p><img src="../a06_raw113.png" alt="a6真实渲染三版对照"><img src="../a37_raw113.png" alt="a37真实渲染三版对照">
<details><summary>组件与证据范围</summary><svg viewBox="0 0 1000 105"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#8daac5"/></marker></defs><g fill="#17304a" stroke="#6684a3"><rect x="5" y="15" width="205" height="75" rx="8"/><rect x="270" y="15" width="200" height="75" rx="8"/><rect x="530" y="15" width="205" height="75" rx="8"/><rect x="795" y="15" width="200" height="75" rx="8"/></g><path d="M210 52H270M470 52H530M735 52H795" stroke="#8daac5" marker-end="url(#arr)"/><g fill="#e4f1fc" font-size="17" text-anchor="middle"><text x="107" y="46">真实冻结检查点</text><text x="107" y="72">初始化／原4K／修复4K</text><text x="370" y="46">CPU只读提取</text><text x="370" y="72">位置／状态／owner</text><text x="632" y="46">同坐标／同车框</text><text x="632" y="72">两侧旋转比较</text><text x="895" y="46">实际渲染另行核对</text><text x="895" y="72">独立／静态／Delete</text></g></svg><p>本页新增的是修复检查点中心及状态的CPU提取，参数更新0，未启动新训练。修复版三值GPU支撑着色投影未另做，不能把中心视图称为完整高斯支撑；实际RGB/alpha/删除视频已覆盖49帧。原始检查点和测量池保留，human_verdict=null。</p><a href="receipt.json">修复提取记录</a></details>
<script type="application/json" id="data">@@DATA@@</script><script>const DATA=JSON.parse(document.getElementById('data').textContent),STAGES={init:'相同初始化','4k':'原R15 4K',repair4k:'修复R15 4K',old5:'旧五身份30K'};
@@JS@@
selectPanels();</script></html>'''
html=html.replace('@@ROWS@@',rows).replace('@@DATA@@',json.dumps(old,separators=(',',':'))).replace('@@JS@@',js)
(O/'index.html').write_text(html,encoding='utf-8');shutil.copy2(W/'tristate/receipt.json',O/'receipt.json')
(W/'tristate_inline_check.js').write_text('const document={},devicePixelRatio=1;\n'+js,encoding='utf-8')
print(json.dumps(dict(path=str(O/'index.html'),panels=len(old['panels']),models=['init','4k','old5','repair4k'],all_five_unknown_counts_verified=True)))
