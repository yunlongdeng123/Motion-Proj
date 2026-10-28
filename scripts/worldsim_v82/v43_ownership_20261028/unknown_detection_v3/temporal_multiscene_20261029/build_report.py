"""逐帧对照报告；只展示已完成原生输出，不生成/补景。"""
from pathlib import Path
import json,shutil,argparse,html
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2]
CONTEXTS=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 O=B/'outputs/V43_Ownership_20261027/gpu_v43/temporal_multiscene_20261029';O.mkdir(parents=True,exist_ok=True);scenes=[]
 for no,ctx in enumerate(CONTEXTS,1):
  src=a.source/ctx;summary=json.loads((src/'summary.json').read_text(encoding='utf-8'));frames=[]
  for r in summary['records']:
   if r['status']!='ok':continue
   raw=r['raw'];d=src/'source_body_contact'/f'{raw:03d}';target=O/ctx/f'{raw:03d}';target.mkdir(parents=True,exist_ok=True)
   for n in ['original','actor_omitted','white','observed','alpha','gaussian']:
    shutil.copy2(d/(n+'.png'),target/(n+'.png'))
   shutil.copy2(d/'receipt.json',target/'receipt.json');frames.append(r)
  scene=dict(context=ctx,name=f'场景{no}',frames=frames,wall_seconds=summary['wall_seconds'],cuda_peak_reserved=summary['cuda_peak_reserved'])
  scenes.append(scene)
  # 全49帧车身归一化crop，检查范围跳变；不是全景保真评价。
  sheet=Image.new('RGB',(7*210,7*155),'#111e2a');draw=ImageDraw.Draw(sheet)
  for k,r in enumerate(frames):
   im=Image.open(O/ctx/f"{r['raw']:03d}"/'white.png');x0,y0,x1,y1=r['body_bbox_xyxy'];w=x1-x0;h=y1-y0
   box=(max(0,int(x0-.65*w)),max(0,int(y0-.3*h)),min(480,int(x1+1.2*w)),min(320,int(y1+.65*h)))
   crop=im.crop(box);scale=min(208/crop.width,132/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST);x=(k%7)*210;y=(k//7)*155;sheet.paste(crop,(x,y+20));draw.text((x+3,y+3),f"raw{r['raw']} n={r['final_removed_rows']}",fill='white')
  sheet.save(O/f'scene{no}_all49_crops.png')
 audit=a.source/'audit/temporal_audit.json'
 if audit.exists():shutil.copy2(audit,O/'temporal_audit.json')
 compact=a.source/'audit/delivery_summary.json'
 if compact.exists():shutil.copy2(compact,O/'audit_summary.json')
 data=json.dumps(scenes,ensure_ascii=False)
 source='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V43 三场景逐帧 unknown 诊断</title><style>
body{background:#0b1420;color:#d8e8f6;font:16px/1.6 system-ui;margin:24px}a{color:#80c9ff}button,select{background:#1d3043;color:inherit;padding:8px;border:1px solid #43607b;border-radius:6px}input{width:55%}.note,.stats{background:#14283a;padding:16px;margin:16px 0;border-radius:8px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panels img{width:100%}figure{margin:0}svg{max-width:1180px;width:100%}table{border-collapse:collapse}td,th{padding:8px 18px;border:1px solid #344e65}details{margin:16px 0}code{color:#9ce2cc}</style>
<h1>V43 三场景：局部 unknown 真实剥离与逐帧检查</h1>
<p><a href="../../../GS_Scene_Diagnostics_20261027/index.html">返回共享工作台</a> · <a href="../bounded_white_v1/index.html">保留您认可的旧三帧 R2 对照</a> · <a href="temporal_audit.json">逐帧行 ID 与分量审计</a></p>
<div class="note">本轮复用三场景已完成的 10K 车身模型，没有重训。新增 3×49 帧冻结渲染。初始候选清空，仅用原始车身掩码＋固定归一化车底查询；蓝框范围按车身框传递，参数三场景相同。它是可迁移的局部查询对照，仍依赖您给定的蓝框形状，不是已学成的语义分类器。</div>
<svg viewBox="0 0 1180 180" role="img" aria-label="原始观测到局部候选到冻结重渲染的数据流"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#86b8d5"/></marker></defs><g fill="#193b50" stroke="#86b8d5"><rect x="5" y="20" width="250" height="120" rx="8"/><rect x="300" y="20" width="390" height="120" rx="8"/><rect x="735" y="20" width="430" height="120" rx="8"/></g><g fill="#d8e8f6" text-anchor="middle" font-size="19"><text x="130" y="52">原始观测与范围</text><text x="130" y="87">body → 车底查询＋蓝框</text><text x="130" y="118">固定规则，逐帧独立</text><text x="495" y="52">候选与剩余贡献</text><text x="495" y="87">蓝框内中心 → alpha×T → 删行</text><text x="495" y="118">重新归因 → 无新候选时停止</text><text x="950" y="52">原生渲染与诊断输出</text><text x="950" y="87">G、alpha → 局部 G＋T×白底</text><text x="950" y="118">RGB／行 ID／时序变化／3D几何</text></g><g stroke="#86b8d5" stroke-width="3" marker-end="url(#arrow)"><path d="M255 80H294"/><path d="M690 80H729"/></g></svg>
<div class="note"><b>第三列是真实 Gaussian 删行后重新渲染。</b>白色来自局部 <code>G_after+(1−alpha_after)×白底</code>，没有把不透明残留直接涂白。框外取同帧“仅去Actor”结果，以保留跨界高斯的域外影响；因此该列包含明确的局部合成，不等于全场景删除后的单一原生图，也不是背景补全。主工作台 3D 展示本帧实际静态删行集合，保留原始 checkpoint。未将这些诊断候选直接用作学习标签。</div>
<div class="note"><b>待重点验收：</b>场景1 raw104–105、108–109；场景2 raw149–153；场景3 raw21–25。这里按同一有效范围内的行ID变化选择检查段，不等于已认定RGB闪烁。AI抽查发现场景3 raw13车底仍有紫褐暗带，raw37/49/61白区贴近前景皮卡边缘。新序列尚未全部通过视觉验收。</div><div><select id="scene"></select> <button id="play">播放</button> <button id="prev">←</button> <input id="frame" type="range" min="0" max="48" value="24"> <button id="next">→</button> <span id="label"></span></div>
<div class="panels"><figure><figcaption>① V43 10K 完整重建</figcaption><img id="original"></figure><figure><figcaption>② 仅去 Actor：保留静态车影／环境</figcaption><img id="actor_omitted"></figure><figure><figcaption>③ 蓝框内候选剥离＋局部白底诊断</figcaption><img id="white"></figure></div>
<div class="stats" id="stats"></div><p id="links"></p>
<details><summary>全49帧局部检查图（每格车身框归一化，不代替全景检查）</summary><img id="sheet" style="width:100%"></details>
<details><summary>当前帧原始 RGB／alpha／Gaussian 贡献</summary><div class="panels"><img id="observed"><img id="alpha"><img id="gaussian"></div></details>
<div class="note">验收边界：每帧没有继承前帧删行，所有新增中心均在蓝框内，局部显示范围外逐值保持基线。相邻行 ID 会随可见性、投影范围和 mask 变化；集合变化不等于 RGB 闪烁。尚无独立语义真值，不能从白洞、选中点数或消失的暗纹推算召回率／精确率。三场景新完整序列等待您目视验收。</div>
<script>const DATA=__DATA__;let timer=null;const $=id=>document.getElementById(id);DATA.forEach((s,i)=>$('scene').add(new Option(s.name,i)));function show(){let si=+$('scene').value,s=DATA[si],i=Math.min(+$('frame').value,s.frames.length-1),r=s.frames[i];$('frame').max=s.frames.length-1;$('label').textContent='raw'+r.raw+' · '+(i+1)+'/'+s.frames.length;let base=s.context+'/'+String(r.raw).padStart(3,'0')+'/';for(let n of ['original','actor_omitted','white','observed','alpha','gaussian'])$(n).src=base+n+'.png';$('stats').textContent='本帧候选 '+r.final_removed_rows+' 行；框外新增中心 '+r.new_rows_outside_blue_centers+'；范围外像素差 '+r.outside_output_max_difference+'；剩余域外中心贡献 '+r.remaining_outside_center_rows+' 行（其支撑可伸进查询）；独立重置 '+r.independent_frame_reset;$('links').innerHTML='<a href="../../../GS_Scene_Diagnostics_20261027/'+s.context+'/index.html?group=v43_bounded_unknown&raw='+r.raw+'">同帧2D／3D工作台 →</a> · <a href="'+base+'receipt.json">本帧诊断凭据</a>';$('sheet').src='scene'+(si+1)+'_all49_crops.png'}$('scene').onchange=show;$('frame').oninput=show;$('prev').onclick=()=>{$('frame').value=Math.max(0,+$('frame').value-1);show()};$('next').onclick=()=>{$('frame').value=(+$('frame').value+1)%(+$('frame').max+1);show()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>$('next').click(),250);$('play').textContent='暂停'}};show();</script></html>'''
 (O/'index.html').write_text(source.replace('__DATA__',data),encoding='utf-8')
 (O/'manifest.json').write_text(json.dumps(scenes,ensure_ascii=False,indent=2),encoding='utf-8')
 print(O)
if __name__=='__main__':main()
