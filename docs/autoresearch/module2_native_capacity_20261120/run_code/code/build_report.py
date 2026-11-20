"""由冻结端点评价与原生预览构建深色报告，不重新渲染或选择 checkpoint。"""
from pathlib import Path
import argparse,html,json

TEMPLATE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>道路材质蒸馏与固定高斯容量</title>
<style>
:root{color-scheme:dark;--bg:#0b131f;--panel:#132131;--border:#294158;--muted:#a7bbcf;--text:#e3edf6;--accent:#6ed1ba}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1680px;padding:28px;margin:auto}h1{font-size:29px;margin:0 0 8px}h2{font-size:21px;margin:32px 0 12px}p{margin:8px 0}a{color:#92c8f4}small,.muted{color:var(--muted)}.badges{display:flex;flex-wrap:wrap;gap:9px;margin:12px 0}.badge{border:1px solid var(--border);border-radius:20px;padding:2px 12px;background:var(--panel);color:var(--accent)}.findings{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.finding,section{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}.finding strong{font-size:16px;display:block}.arch{display:block;width:100%;margin:20px 0}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:14px 0;position:sticky;top:0;background:#0b131ff2;z-index:2}label{display:flex;align-items:center;gap:8px}select{background:#172a40;color:var(--text);border:1px solid #3c5d7c;border-radius:5px;padding:9px;font-size:15px}.overviews{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.imagebox{background:#000;border:1px solid var(--border);border-radius:6px;overflow:hidden;margin:0}.imagebox figcaption{padding:8px 10px;min-height:40px;background:#162638;font-size:13px}.imagebox img{display:block;width:100%;height:auto}.strip{display:flex;align-items:flex-start;gap:12px;overflow-x:auto;padding-bottom:14px;max-width:100%}.strip .imagebox{flex:none}.strip img{width:auto;max-width:none;height:auto;image-rendering:auto}.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:right;padding:9px 11px;border-bottom:1px solid var(--border)}th{color:var(--muted);font-weight:500}th:first-child,td:first-child{text-align:left}tr.focus td{background:#1d3b43}summary{cursor:pointer;color:#92c8f4}.note{border-left:3px solid #bba079;padding:8px 14px;color:#cbd4df;background:#2a292b}.num{font-variant-numeric:tabular-nums}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.ideal-images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.links{display:flex;gap:18px;flex-wrap:wrap;padding-top:24px}@media(max-width:1100px){.findings{grid-template-columns:1fr}.overviews{grid-template-columns:repeat(2,minmax(0,1fr))}.two{grid-template-columns:1fr}}@media(max-width:500px){main{padding:14px}h1{font-size:23px}.overviews,.ideal-images{grid-template-columns:1fr}.controls{position:static;gap:8px}select{max-width:160px}section{padding:11px}}
</style><main>
<h1>道路材质蒸馏与固定高斯容量</h1><p class="muted">Scene065 · 原生 1920×1280 · 预先固定终点</p>
<div class="badges"><span class="badge">未搜索超参数</span><span class="badge">几何与覆盖冻结</span><span class="badge">GPU 保持开机</span></div>
<div id="findings" class="findings"></div><img class="arch" src="architecture.svg" alt="二维教师、世界区域划分、三种颜色优化、同一原生高斯渲染器与五视角持久重载组件图">
<p class="note" id="scope"></p>
<h2>同相机最终渲染</h2><div class="controls"><label>阶段<select id="phase"><option value="BUILD">全标签资产构建</option><option value="QUERY">留出对照实验</option></select></label><label>帧<select id="raw"><option value="110">raw110</option><option value="86">raw86</option><option value="134">raw134</option><option value="179">raw179</option><option value="198">raw198</option></select></label><label>视图<select id="view"><option value="road">道路单独渲染</option><option value="joint">场景联合渲染</option></select></label><label>区域<select id="region"><option value="C">孔洞教师</option><option value="S">去影教师</option></select></label><label>评价<select id="split"></select></label><label>高频尺度<select id="sigma"><option value="1">σ1 像素</option><option value="2">σ2 像素</option><option value="4">σ4 像素</option></select></label></div>
<div id="overview" class="overviews"></div><p class="muted">概览仅用于定位；下方为不缩放的原生 1:1 裁片，可横向滚动。教师图保持原始颜色，未乘道路透明度。</p>
<div id="crops" class="strip"></div><p id="domain" class="muted"></p><details><summary>监督范围与只读贡献定位</summary><div class="strip" id="attribution"></div></details>
<section><p class="muted">下表评价道路单独渲染的稳定材质颜色；联合画面的遮挡与保护另列。</p><div class="scroll"><table id="metrics"></table></div><p class="muted">高频对应度越接近 1 越好；能量比接近 1 仍须同时检查对应度。平坦区伪高频与边缘精度用于检查额外碎纹；空核心显示 N/A。MAE 只说明颜色拟合。</p></section>
<h2>训练目标与数值优化状态</h2><section><div class="scroll"><table id="optimization"></table></div><p id="optimization-note" class="muted"></p></section>
<h2>独立世界一致理想标线控制</h2><section><p id="ideal-note"></p><div class="ideal-images" id="ideal-images"></div><div class="scroll"><table id="ideal-table"></table></div><p class="muted">解析标线参数已知，只作条件表达能力控制；不与真实孔洞教师的方法排名混合，也不作为真实隐藏道路答案。</p></section>
<h2>覆盖、遮挡与车辆保护</h2><div class="two"><section><div class="scroll"><table id="coverage"></table></div><p class="muted">完整合法孔洞域保留无表面、薄覆盖及静态遮挡分母。颜色优化不改变这些冻结项。</p></section><section id="persistence"></section></div>
<h2>执行与可复现记录</h2><section id="execution"></section>
<div class="links"><a href="report.md">文字报告</a><a href="protocol.json">固定协议</a><a href="evaluation/results.json">完整评价</a><a href="formal_results.json">正式端点</a><a href="public_reload_receipt.json">五视角公共重载</a><a href="resource_ledger.json">计算开销</a></div>
</main><script>const D=__DATA__;
const E=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=(x,n=4)=>x===null||x===undefined||!Number.isFinite(Number(x))?'N/A':Number(x).toFixed(n),NA='N/A';
const names={F0:'共同初值',G_RGB:'共享材质 · RGB',G_DETAIL:'共享材质 · 细节目标',G_ORACLE:'独立颜色 · 投影梯度',G_NUMERIC:'独立颜色 · 数值补证',IDEAL_ORACLE:'理想标线 · 独立颜色控制',G_BUILD_FIELD:'共享材质 · 全标签拟合',G_BUILD_NUMERIC:'独立颜色 · 全标签控制'};
const methodGroups={QUERY:['F0','G_RGB','G_DETAIL','G_NUMERIC'],BUILD:['F0','G_BUILD_FIELD','G_BUILD_NUMERIC']};
function imageBox(a,title,kind){if(!a)return `<figure class="imagebox"><figcaption>${esc(title)}</figcaption><p class="muted" style="padding:20px">该视角无合格孔洞教师</p></figure>`;return `<figure class="imagebox"><figcaption>${esc(title)}</figcaption><img src="${esc(a[kind])}" alt="${esc(title)}"></figure>`;}
function table(id,heads,rows){E(id).innerHTML='<tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+esc(c)+'</td>').join('')+'</tr>').join('');}
E('findings').innerHTML=D.summary.findings.map(f=>`<article class="finding"><strong>${esc(f.title)}</strong><p>${esc(f.text)}</p></article>`).join('');E('scope').textContent=D.summary.scope;
table('optimization',D.summary.optimization.heads,D.summary.optimization.rows);E('optimization-note').textContent=D.summary.optimization.note;
table('coverage',D.summary.coverage.heads,D.summary.coverage.rows);E('persistence').innerHTML=D.summary.persistence.map(x=>'<p>'+esc(x)+'</p>').join('');E('execution').innerHTML=D.summary.execution.map(x=>'<p>'+esc(x)+'</p>').join('');E('ideal-note').textContent=D.summary.ideal_note;
function setPhase(){
 const build=E('phase').value==='BUILD';
 E('split').innerHTML=build?'<option value="BUILD_fit_all_qualified">全部合格教师 · 拟合域</option>':'<option value="seen">训练标签区域</option><option value="held">世界留出区域</option>';
 update();
}
function update(){
 const phase=E('phase').value,methods=methodGroups[phase];
 const raw=+E('raw').value,view=E('view').value,source=E('region').value,split=E('split').value,sigma=+E('sigma').value;
 const frame=D.assets.frames[String(raw)],teacher=frame.teachers?.[source];
 E('overview').style.setProperty('--cols',methods.length+1);
 E('overview').innerHTML=imageBox(teacher,source==='C'?'冻结二维孔洞教师':'冻结二维去影教师','overview')+methods.map(m=>imageBox(frame.methods[m]?.[view],names[m],'overview')).join('');
 E('crops').innerHTML=imageBox(teacher,source+' 教师 · 原生 1:1',source+'_crop')+methods.map(m=>imageBox(frame.methods[m]?.[view],names[m]+' · 原生 1:1',source+'_crop')).join('');
 const rows=D.evaluation.rows.filter(r=>r.raw===raw&&r.source===source&&r.split===split&&r.scale===sigma&&methods.includes(r.arm)&&(r.phase||'QUERY')===phase);
 table('metrics',['端点','核心像素','颜色 MAE','HF 对应度','HF 能量比','HF 相对 RMSE','平坦区伪 HF RMS','边缘精度 / 召回','Sobel RMSE'],rows.map(r=>[names[r.arm],r.core_n,fmt(r.RGB_MAE,5),fmt(r.HF_cosine),fmt(r.HF_energy_ratio,2),fmt(r.HF_relative_RMSE,3),fmt(r.flat_pred_RMS,5),fmt(r.edge_precision,3)+' / '+fmt(r.edge_recall,3),fmt(r.Sobel_RMSE,5)]));
 if(raw===86||raw===110){
  E('domain').textContent=phase==='BUILD'?'固定方法完成对照后，合并全部合格教师标签构建资产；这里是拟合结果，原留出区域已参加训练。':`固定世界区域划分；${split==='held'?'初始材质有历史教师预暴露，属于空间新标签留出诊断，不是盲测。':'只使用预先划定训练区域。'} ${source==='S'&&split==='held'?'阴影留出仅一个世界块，不作区域泛化结论。':''}`;
 }else E('domain').textContent='该视角用于检查跨视角伪影；没有真实隐藏道路真值，评价表为空。';
 const attribution=`readonly_floor_audit_r1/${String(raw).padStart(3,'0')}_${source}_readonly_context_contact.png`;
 E('attribution').innerHTML=(raw===86||raw===110)?`<figure class="imagebox"><figcaption>留出对照的范围定位；全标签构建已合并原绿色与紫色区域。橙色未进入当前监督域。</figcaption><img src="${attribution}" alt="只读贡献、可写权重及原对照监督范围"></figure>`:'';
 const ideals=frame.ideal;
 E('ideal-images').innerHTML=ideals?imageBox(ideals.target,'独立解析世界目标','overview')+imageBox(ideals.initial,'共同初值','overview')+imageBox(ideals.endpoint,'理想教师优化端点','overview'):'';
 const ir=D.evaluation.ideal_rows.filter(r=>r.raw===raw);
 table('ideal-table',D.summary.ideal_table?.heads||['评价记录'],D.summary.ideal_table?.rows_by_raw?.[String(raw)]||ir.map(r=>[JSON.stringify(r)]));
}
E('phase').addEventListener('change',setPhase);
for(const id of ['raw','view','region','split','sigma'])E(id).addEventListener('change',update);setPhase();
</script></html>'''
TEMPLATE=TEMPLATE.replace('grid-template-columns:repeat(5,minmax(0,1fr))','grid-template-columns:repeat(var(--cols,5),minmax(0,1fr))')
TEMPLATE=TEMPLATE.replace('</style>','section,.two>*,details{min-width:0;max-width:100%}</style>')

def main(root):
    root=Path(root)
    def load(p):return json.loads((root/p).read_text(encoding='utf-8'))
    d=dict(summary=load('report_summary.json'),evaluation=load('evaluation/results.json'),assets=load('previews/assets.json'))
    # Large per-pixel packets and per-tile arrays stay in the evidence directory.
    d['evaluation']={k:d['evaluation'].get(k,[])for k in('rows','ideal_rows')}
    body=TEMPLATE.replace('__DATA__',json.dumps(d,ensure_ascii=False,allow_nan=False).replace('</',r'<\/'))
    (root/'index.html').write_text(body,encoding='utf-8')
    (root/'report_build.json').write_text(json.dumps(dict(completed=True,source='frozen evaluation and fixed endpoint previews',metric_recomputed=False,GPU_operations=0,method_groups=dict(QUERY=['F0','G_RGB','G_DETAIL','G_NUMERIC'],BUILD=['F0','G_BUILD_FIELD','G_BUILD_NUMERIC']),controls=180),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(index=str(root/'index.html'),bytes=len(body.encode()),metric_rows=len(d['evaluation']['rows'])),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
