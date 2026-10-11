"""用已完成的真实配对产物构建深色审核页；不生成或重编码图片。"""
from pathlib import Path
import argparse,json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();root=args.output_dir
    data=json.loads((root/'run.json').read_text(encoding='utf-8'))
    from PIL import Image
    paths=[]
    for i in range(4):
        paths.extend([f'assets/input/{i:03}.png',f'assets/sky/{i:03}.png',f'assets/delta/delete/{i:03}.png'])
        for kind in ['official','premult']:
            paths.extend(f'assets/{kind}/{branch}/{i:03}.png' for branch in ['noop','delete'])
    for name in paths:
        with Image.open(root/name) as im: im.load()
    review_path=root/'assistant_review.json'
    review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else None
    payload=json.dumps({'run':data,'review':review},ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DGGT · 透明度合成配对</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c111b;color:#e8edf5}body{max-width:1600px;margin:auto;padding:24px}h1{font-size:27px;margin:8px 0}p{line-height:1.65;color:#aebbd0}.box,article{background:#151e2b;border:1px solid #2c3a4c;border-radius:12px;padding:14px;margin:14px 0}.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}figcaption{padding:10px 0;color:#c5d4e9;font-size:14px}img{width:100%;background:#070b11;display:block;border-radius:8px}button{background:#25476b;border:1px solid #5684b7;color:white;border-radius:7px;padding:8px 16px;cursor:pointer}button.active{background:#4c7eb5}nav{display:flex;gap:10px;position:sticky;top:0;background:#0c111bef;padding:12px 0;z-index:2}.badge{color:#89c6ef}.arch{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.arch span{padding:7px;border:1px solid #30455d;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#b9c8db}@media(max-width:850px){.row{grid-template-columns:1fr}}
</style><body><div class="badge">DGGT / 同一份高斯 / 零训练配对</div><h1>重复透明度衰减能解释多少黑洞？</h1>
<p>四个固定输入帧，保持高斯、相机、实例选择与权重相同。先核对官方合成重放，再比较去掉外层透明度乘法的诊断变体。变亮本身不代表道路恢复。</p>
<div class="box arch"><span>既有高斯与相机</span>→<span>同一次 gsplat 渲染</span>→<span>颜色 G / 透明度 A / 背景 S</span>→<span>官方 A×G+(1−A)×S</span><span>诊断 G+(1−A)×S</span></div>
<div class="box"><strong>审核结论</strong><p id="verdict">助手审核待完成；人工 verdict 未填写。</p></div><nav id="frames"></nav>
<article><h2>重建对照</h2><div class="row" id="noop"></div></article>
<article><h2>删除对照</h2><div class="row" id="del"></div><p>背景 S 来自原模型，不能当作车后道路真值。删除图中的结构是否可信，需要看道路纹理、邻车和接缝。</p></article>
<article><h2>删除图差值</h2><div class="row" id="delta"></div><p>差值按原幅度显示，没有逐图拉伸；黑色表示变化小。</p></article>
<details class="box"><summary>真实执行与审核记录</summary><pre id="evidence"></pre><a href="run.json">执行 JSON</a></details>
<script>const evidence=PAYLOAD;const labels=[['input','输入 RGB'],['official/noop','官方重建'],['premult/noop','去掉二次衰减 · 重建'],['sky','模型背景 S'],['official/delete','官方删除'],['premult/delete','去掉二次衰减 · 删除']];
function add(parent,path,label,f){let fig=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src='assets/'+path+'/'+String(f).padStart(3,'0')+'.png';im.alt=label+' 第'+f+'帧';cap.textContent=label;fig.append(im,cap);parent.append(fig)}
function select(f){['noop','del','delta'].forEach(id=>document.getElementById(id).replaceChildren());labels.forEach((v,i)=>add(document.getElementById(i<3?'noop':'del'),v[0],v[1],f));add(document.getElementById('delta'),'delta/delete','诊断与官方删除的绝对 RGB 差',f);document.querySelectorAll('nav button').forEach((b,i)=>b.classList.toggle('active',i===f));}
for(let f=0;f<4;f++){let b=document.createElement('button');b.textContent='帧 '+String(f).padStart(3,'0');b.onclick=()=>select(f);document.getElementById('frames').append(b)}
document.getElementById('evidence').textContent=JSON.stringify(evidence,null,2);if(evidence.review){document.getElementById('verdict').textContent=evidence.review.summary_zh||evidence.review.summary||'独立审核记录已附；人工 verdict 未填写。'}select(0);</script></body></html>'''.replace('PAYLOAD',payload)
    (root/'index.html').write_text(html,encoding='utf-8')
    validation={'decoded_png':len(paths),'missing_links':0,'browser_visual_qa':False,'review_present':review is not None}
    (root/'delivery_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation))

if __name__=='__main__': main()
