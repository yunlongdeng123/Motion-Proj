"""深色报告shell；直接引用实际takeback图片，不复制或改写图像。"""
from pathlib import Path
import json,shutil
from html import escape
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/a20_joint_reconstruction_20261122'

def main():
    r=json.loads((ROOT/'result.json').read_text(encoding='utf-8'));OUT.mkdir(parents=True,exist_ok=True)
    for name in ['result.json','report_groups.json','report_style.css','report_ui.js','architecture.svg']:shutil.copy2(ROOT/name,OUT/name)
    extension=ROOT.parent/'a20_scene_residual_cleanup_20261122/report_integration/scene_report_ui.js'
    if extension.is_file():shutil.copy2(extension,OUT/'scene_report_ui.js')
    data=json.dumps(r,ensure_ascii=False).replace('</','<\\/')
    nav=''.join('<a href="'+escape(x['href'],quote=True)+'">'+escape(x['label'])+'</a>' for x in r['history'])
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>a20 · 统一高斯场重建</title><link rel="stylesheet" href="report_style.css"></head><body><main><nav>'''+nav+'''<a href="result.json">实际结果JSON</a></nav>
<h1>a20：统一高斯场联合重建</h1><div class="muted">共同初始化419005行 · 各40步组独立重放 · GPU保持开机</div><div class="panel-note muted" id="scope-note" hidden></div><div class="status" id="status"></div><div class="links" id="final-links"></div><div class="panel-note muted" id="final-summary"></div><div class="stats" id="stats"></div><div class="architecture"><a id="architecture-link" href="architecture.svg"><img id="architecture-image" src="architecture.svg" alt="统一a20高斯场重建组件图"></a></div>
<h2>实际渲染</h2><div class="controls"><label>实验阶段<select id="group"></select></label><label>相机<select id="camera"></select></label><label>通道<select id="channel"><option value="rgb">RGB</option><option value="alpha">目标Alpha</option><option value="depth">期望深度灰模</option><option value="normal">期望深度法向</option><option value="grey">灰影（辅助）</option></select></label><button id="native-view">真实相机</button><button id="geometry-view">六视几何</button></div>
<div class="gallery"><figure class="image-card"><div class="image-title"><strong id="left-title"></strong><select id="left-mode"><option value="shared">共享场景</option><option value="isolated">单对象</option><option value="teacher">纯T</option></select></div><div class="image-body" id="left-image"></div></figure><figure class="image-card"><div class="image-title"><strong id="right-title"></strong><select id="right-mode"><option value="isolated">单对象</option><option value="shared">共享场景</option><option value="teacher">纯T</option></select></div><div class="image-body" id="right-image"></div></figure></div><div class="links" id="receipts"></div>
<h2>固定域指标</h2><div class="table-scroll"><table><thead><tr><th>相机</th><th>单对象可见区覆盖</th><th>场景内可见贡献</th><th>其他车辆变化像素</th><th>真实RGB L1</th></tr></thead><tbody id="metrics"></tbody></table></div>
<h2>阶段登记</h2><div class="table-scroll"><table><thead><tr><th>组</th><th>计划步数</th><th>实际步数</th><th>状态</th><th>记录</th></tr></thead><tbody id="ledger"></tbody></table></div><section id="extra-figures"></section><details id="grey-panel"><summary>辅助灰影六视总览</summary><div id="grey-overview"></div></details>
<footer><b>免责声明：</b>'''+escape(r['disclaimer'])+'''</footer></main><script>const D='''+data+''';</script><script src="scene_report_ui.js"></script><script src="report_ui.js"></script></body></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8');print(OUT/'index.html')

if __name__=='__main__':main()
