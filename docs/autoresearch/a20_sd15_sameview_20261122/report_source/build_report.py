"""Offline report for actual fixed-input teacher candidates and protected grey fusion."""
from pathlib import Path
import json, shutil
from html import escape

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/a20_sd15_sameview_20261122'

def main():
    r=json.loads((ROOT/'result.json').read_text(encoding='utf-8'))
    (OUT/'media').mkdir(parents=True,exist_ok=True)
    for src,dst in r['media']:
        shutil.copy2(ROOT/src,OUT/'media'/dst)
    for f in ['report.md','result.json','protocol.json']:
        shutil.copy2(ROOT/f,OUT/f)
    shutil.copy2(ROOT/'architecture_executed.svg',OUT/'architecture.svg')
    if r.get('asset'):
        shutil.copy2(ROOT/r['asset']['source'],OUT/'a20_actor_candidate.npz')
        shutil.copy2(ROOT/'a20_actor_candidate.json',OUT/'a20_actor_candidate.json')
    rows=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in r['metrics'])
    figures=''.join('<figure><h3>'+escape(x['title'])+'</h3><a href="media/'+x['file']+'"><img src="media/'+x['file']+'" alt="'+escape(x['title'])+'"></a><figcaption>'+escape(x['caption'])+'</figcaption></figure>' for x in r['figures'])
    findings=''.join('<p>'+escape(s)+'</p>' for s in r['findings'])
    data=json.dumps(r.get('viewer',{}),ensure_ascii=False).replace('</','<\\/')
    viewer=''
    if r.get('viewer'):
        viewer='''<h2>同相机原生细节对照</h2><nav class="controls"><label>视图 <select id="view"></select></label><label>左侧 <select id="left-method"></select></label><label>右侧 <select id="right-method"></select></label></nav><div class="grid" id="comparison"></div><p class="muted">合成视图源512×512；Alpha为车辆单层覆盖。联合可见贡献单独验收。</p><script>
const D='''+data+''';const select=document.getElementById('view'),L=document.getElementById('left-method'),R=document.getElementById('right-method');for(const [key,label] of Object.entries(D.views))select.add(new Option(label,key));D.methods.forEach((m,i)=>{L.add(new Option(m.label,i));R.add(new Option(m.label,i))});L.selectedIndex=D.methods.length>2?1:0;R.selectedIndex=D.methods.length-1;
function update(){const k=select.value;document.getElementById('comparison').replaceChildren();for(const a of [D.methods[+L.value],D.methods[+R.value]]){const c=document.createElement('figure');const h=document.createElement('h3');h.textContent=a.label;c.append(h);const path=a.images[k];if(path){const link=document.createElement('a');link.href=path;const im=document.createElement('img');im.src=path;im.alt=a.label+' '+D.views[k];link.append(im);c.append(link);}else{const p=document.createElement('p');p.textContent='该阶段未执行';c.append(p)}document.getElementById('comparison').append(c)}}[select,L,R].forEach(s=>s.addEventListener('change',update));update();</script>'''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>a20：同视角二维补全与三维教师对照</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0c1622;color:#e9f2fa;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:30px}h1{font-size:30px}h2{font-size:23px;margin-top:30px}h3{font-size:18px}a{color:#90ccff}nav{display:flex;gap:18px;flex-wrap:wrap}.status{padding:16px 20px;border-left:4px solid #e0b363;background:#213045}.muted,figcaption{color:#a9bdce}img{display:block;width:100%;height:auto}.table{overflow-x:auto}table{min-width:760px;width:100%;border-collapse:collapse}td,th{padding:10px 12px;border-bottom:1px solid #354d61;text-align:left}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{min-width:0;margin:20px 0;padding:16px;border:1px solid #36556c;border-radius:10px;background:#142637}select{font:inherit;padding:8px;background:#1a364b;color:#e9f2fa;border:1px solid #668ba3;border-radius:6px}footer{margin-top:30px;padding-top:18px;border-top:1px solid #354d61;color:#a9bdce}@media(max-width:760px){main{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:24px}}
</style><main><nav><a href="../omni_hq_vehicle_teacher_20261122/index.html">高质量参考灰模对照</a><a href="../vehicle_completion_prior_20261122/index.html">车辆历史实验</a><a href="../road_material_three_rounds_20261121/index.html">道路历史实验</a><a href="report.md">完整记录</a><a href="result.json">实验数值</a></nav><h1>a20：同视角二维补全与三维教师对照</h1><p class="muted">2026-11-22 · 删除a9白车，恢复a20黑色皮卡 · 构建raw86／110</p><p class="status">'''+escape(r['headline'])+'''</p><a href="architecture.svg"><img src="architecture.svg" alt="真实观测、二维补全、三维先验与缺失区融合组件"></a><h2>执行与验收</h2><div class="table"><table><thead><tr><th>实验／检查</th><th>结果</th><th>判定</th></tr></thead><tbody>'''+rows+'''</tbody></table></div><h2>输入与二维检查</h2>'''+figures+viewer+'''<h2>卡点与阶段结论</h2>'''+findings+'''<footer><b>免责声明：</b>'''+escape(r['disclaimer'])+'''</footer></main></html>'''
    def extra_table(key,title):
        data=r.get(key,[])
        if not data:return ''
        rows=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)
        return '<h2>'+escape(title)+'</h2><div class="table"><table>'+rows+'</table></div>'
    def extra_figures(key,title):
        data=r.get(key,[])
        if not data:return ''
        cards=''.join('<figure><h3>'+escape(f['title'])+'</h3><a href="media/'+f['file']+'"><img src="media/'+f['file']+'" alt="'+escape(f['title'])+'"></a><figcaption>'+escape(f['caption'])+'</figcaption></figure>' for f in data)
        return '<h2>'+escape(title)+'</h2><div class="grid">'+cards+'</div>'
    geometry=extra_table('geometry_metrics','三维教师灰模：同一固定验收门')+extra_figures('geometry_figures','灰模部件关系')
    page=page.replace('<h2>输入与二维检查</h2>'+figures+viewer,'<h2>二维输入与前置门</h2>'+figures+geometry+viewer+extra_table('appearance_metrics','真实可见区域：初始化与80步收益分开'))
    page=page.replace('<h2>卡点与阶段结论</h2>',extra_figures('scene_figures','原生场景接入：RGB与可见贡献分开')+'<h2>卡点与阶段结论</h2>')
    page=page.replace('a20：同视角二维补全与三维教师对照','a20：车辆教师与高斯资产对照')
    if r.get('asset'):
        page=page.replace('<a href="report.md">完整记录</a>','<a href="a20_actor_candidate.npz">高斯候选资产</a><a href="a20_actor_candidate.json">资产来源</a><a href="report.md">完整记录</a>')
    page=page.replace('.grid{display:grid;', '.grid img{width:auto;max-width:100%;margin:auto}.grid{display:grid;')
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(OUT/'index.html')

if __name__=='__main__': main()
