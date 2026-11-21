"""从本轮实际收据生成轻量离线报告，不覆盖旧七阶段实验。"""
from pathlib import Path
from html import escape
import json, shutil
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/vehicle_completion_prior_20261122'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    record=json.loads((ROOT/'result.json').read_text(encoding='utf-8'))
    media=OUT/'media';media.mkdir(exist_ok=True)
    for source,name in record.get('media',[]):shutil.copy2(ROOT/source,media/name)
    shutil.copy2(ROOT/'protocol.json',OUT/'protocol.json')
    shutil.copy2(ROOT/'protocol_semantic_condition.json',OUT/'protocol_semantic_condition.json')
    shutil.copy2(ROOT/'prior_review.md',OUT/'prior_review.md')
    for name in ['report.md','teacher_student_protocol_v3.md','teacher_model_audit_20261122.md','mm_trellis_audit.md','grey_gate_v2.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,OUT/name)
    shutil.copy2(ROOT/'result.json',OUT/'result.json')
    def cards():
        return ''.join(f'<article class="{"full" if m.get("wide") else ""}"><h3>{escape(m["title"])}</h3><a href="media/{escape(m["file"])}"><img src="media/{escape(m["file"])}" alt="{escape(m["title"])}"></a><p>{escape(m["caption"])}</p></article>' for m in record['figures'])
    rows=''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in record['metrics'])
    stages=''.join(f'<li><b>{escape(x[0])}</b><span>{escape(x[1])}</span></li>' for x in record['stages'])
    findings=''.join(f'<p>{escape(s)}</p>' for s in record['findings'])
    viewer=''
    if record.get('student_viewer'):
        data=json.dumps(record['student_viewer'],ensure_ascii=False).replace('</','<\\/')
        viewer='''<h2>已完成的诊断渲染 · 灰模未通过</h2><nav class="controls"><label>帧 <select id="frame"><option>86</option><option selected>110</option><option>134</option><option>179</option><option>198</option></select></label><label>视图 <select id="view"><option value="vehicle_only">车辆单层 RGB</option><option value="vehicle_alpha_isolated">车辆单层 Alpha</option><option value="vehicle_alpha_delete">联合可见贡献 Alpha</option><option value="joint_delete">删除 a9 后联合 RGB</option><option value="joint_keep">保留 a9 的联合 RGB</option><option value="road_only">道路单层 RGB</option></select></label><label>阶段 <select id="stage"><option value="canonical_prior_init">生成表面高斯初值</option><option value="vehicle_fit">车辆 200 步</option><option value="joint_fit" selected>联合 200 步</option></select></label><label>范围 <select id="crop"><option value="vehicle">车辆局部 · 原像素裁片</option><option value="full">完整画面</option></select></label></nav><div class="render-grid"><article><h3>原始真实观测</h3><canvas id="reference"></canvas></article><article><h3>上一轮 · 75 行交接终态</h3><canvas id="previous"></canvas></article><article><h3 id="new-title">本轮 · 联合终态</h3><canvas id="current"></canvas></article></div><p class="muted" id="viewer-status"></p><script>const VDATA='''+data+''';let revision=0;function draw(id,url,crop,rev){return new Promise(resolve=>{let im=new Image();im.onload=()=>{if(rev!==revision){resolve(true);return;}const c=document.getElementById(id),x=c.getContext('2d');const r=crop||[0,0,im.width,im.height];c.width=r[2];c.height=r[3];x.drawImage(im,r[0],r[1],r[2],r[3],0,0,r[2],r[3]);resolve(true);};im.onerror=()=>resolve(false);im.src=url;});}async function update(){const rev=++revision,f=document.getElementById('frame').value.padStart(3,'0'),v=document.getElementById('view').value,s=document.getElementById('stage').value,crop=document.getElementById('crop').value==='vehicle'?VDATA.crops[f]:null;document.getElementById('viewer-status').textContent='读取中';document.getElementById('new-title').textContent=document.getElementById('stage').selectedOptions[0].text;const ok=await Promise.all([draw('reference',VDATA.reference+'/'+f+'/rgb.webp',crop,rev),draw('previous',VDATA.previous+'/'+f+'/'+v+'.webp',crop,rev),draw('current',VDATA.current+'/'+s+'/'+f+'/'+v+'.webp',crop,rev)]);if(rev===revision)document.getElementById('viewer-status').textContent=ok.every(Boolean)?'源渲染 1152×768；局部为同坐标原像素裁片。Alpha 与 RGB 分开显示。':'图像读取失败，请核对本地资产是否完整。';}document.querySelectorAll('.controls select').forEach(x=>x.addEventListener('change',update));update();</script>'''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>a20 三维补全先验</title><style>
    :root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1420;color:#e4edf6;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:32px}}h1{{font-size:30px;line-height:1.3}}h2{{font-size:22px;margin-top:30px}}h3{{font-size:18px;margin:0 0 12px}}p{{margin:10px 0}}a{{color:#83c6ff}}nav{{display:flex;gap:18px;flex-wrap:wrap}}.status{{border-left:4px solid #e4b95b;padding:15px 20px;background:#1d2939}}.muted,figcaption{{color:#a0b6ca}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}article{{background:#142336;border:1px solid #304b62;border-radius:12px;padding:18px;min-width:0}}img{{display:block;width:100%;height:auto;background:#0b1420}}.wide{{max-height:800px;object-fit:contain}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid #314358;text-align:left}}.table{{overflow-x:auto}}ul{{padding:0;list-style:none;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}li{{padding:14px;background:#142336;border-radius:9px}}li span{{display:block;color:#a0b6ca}}footer{{margin-top:32px;border-top:1px solid #314358;padding-top:18px;color:#a0b6ca}}@media(max-width:800px){{main{{padding:18px}}.grid,ul{{grid-template-columns:1fr}}h1{{font-size:25px}}}}
    .full{{grid-column:1/-1}}select{{font:inherit;background:#182f43;color:#e4edf6;border:1px solid #56738a;border-radius:7px;padding:6px}}.controls{{margin:16px 0}}.render-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}canvas{{width:100%;height:auto;background:#000}}@media(max-width:900px){{.render-grid{{grid-template-columns:1fr}}}}
    </style><main><nav><a href="../vehicle_road_joint_20261121/index.html">上一轮七阶段对照</a><a href="../road_material_three_rounds_20261121/index.html">道路全部历史组</a><a href="protocol.json">点条件诊断协议</a><a href="protocol_semantic_condition.json">完整语义条件协议</a><a href="prior_review.md">接口核查</a></nav>
    <h1>a20：生成式车辆表面与材质先验</h1><p class="muted">2026-11-22 · 删除 a9 白色 Mazda，恢复后方黑色皮卡 a20 · 构建 raw86／110</p>
    <p class="status">{escape(record['headline'])}</p><ul>{stages}</ul>
    <h2>拟用教师：先验证灰模</h2><div class="table"><table><thead><tr><th>模型</th><th>已核实能力</th><th>当前工程顺序</th></tr></thead><tbody><tr><td><a href="https://github.com/Sm0kyWu/Amodal3R">Amodal3R</a></td><td>显式可见／遮挡输入；通用对象域，代码与权重公开</td><td>首个遮挡灰模候选</td></tr><tr><td><a href="https://github.com/HongliXiao/MM-TRELLIS">MM-TRELLIS</a></td><td>图像＋测距测试时引导，使用原TRELLIS权重</td><td>测距约束对照</td></tr><tr><td><a href="https://zijunkr.github.io/RECAST/">RECAST</a></td><td>车辆域适配及遮挡扰动训练；尚未找到可执行代码与适配权重入口</td><td>资源门通过后纳入</td></tr></tbody></table></div><p class="muted">三者尚未在a20上运行。顺序依据当前输入与开放资源；近同向后视帧不构成侧面／正面观测。</p>
    <p><a href="../../work/vehicle_completion_prior_20261122/teacher_inputs_amodal3r/raw110/preview.png">Amodal3R raw110 CPU输入预览</a></p><h2>新版协议：灰模验收前置</h2><a href="architecture.svg"><img class="wide" src="architecture.svg" alt="观测保护、灰模验收与教师蒸馏组件"></a><p><a href="teacher_student_protocol_v3.md">教师蒸馏协议</a> · <a href="grey_gate_v2.json">当前灰模门状态</a> · <a href="architecture_executed_r1.svg">本轮实际执行链路</a></p>
    <h2>输入与生成结果</h2><div class="grid">{cards()}</div>
    {viewer}<h2>固定候选的实测检查</h2><div class="table"><table><thead><tr><th>检查项</th><th>结果</th><th>含义</th></tr></thead><tbody>{rows}</tbody></table></div>
    <h2>本轮判断</h2>{findings}<p>车辆单层 RGB／Alpha 与共同遮挡后的可见贡献分别验收。形状、材质和真实观测指标不互相替代；人工质量评价留空。</p><p><a href="report.md">完整诊断说明</a> · <a href="teacher_model_audit_20261122.md">Amodal3R／RECAST核查</a> · <a href="mm_trellis_audit.md">MM-TRELLIS核查</a></p>
    <p>方法接口：<a href="https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni">Hunyuan3D-Omni</a> · <a href="https://github.com/microsoft/TRELLIS.2/blob/main/example_texturing.py">TRELLIS.2 固定网格材质</a> · <a href="../../work/vehicle_road_joint_20261121/hugsim_review.md">HUGSIM 接入依据</a></p>
    <footer><b>免责声明：</b>{escape(record['disclaimer'])}</footer></main></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(OUT/'index.html')
if __name__=='__main__':main()
