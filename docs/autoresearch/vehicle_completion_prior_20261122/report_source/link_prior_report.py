"""只追加新实验入口，不修改上一轮七阶段数据或选择器。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    path=ROOT/'outputs/vehicle_road_joint_20261121/index.html'
    text=path.read_text(encoding='utf-8')
    marker='id="vehicle-completion-prior-entry"'
    if marker not in text:
        entry='<nav id="vehicle-completion-prior-entry" style="padding:14px 24px;background:#122a3d;border-bottom:1px solid #38526a"><a style="color:#9fd5ff" href="../vehicle_completion_prior_20261122/index.html">车辆生成先验：新实验与架构</a></nav>'
        pos=text.find('>',text.find('<body'))+1
        if pos<=0:raise ValueError('body missing')
        text=text[:pos]+entry+text[pos:]
        path.write_text(text,encoding='utf-8')
    print(path)
    source=ROOT/'work/vehicle_road_joint_20261121/viewer.js'
    if source.exists():
        js=source.read_text(encoding='utf-8')
        marker_js='vehicle-completion-prior-entry'
        if marker_js not in js:
            js+='''\n// Regenerated reports retain the later vehicle-prior entry without changing historical data.
(()=>{const add=()=>{if(document.getElementById('vehicle-completion-prior-entry'))return;const n=document.createElement('nav');n.id='vehicle-completion-prior-entry';n.style='padding:14px 24px;background:#122a3d;border-bottom:1px solid #38526a';const a=document.createElement('a');a.href='../vehicle_completion_prior_20261122/index.html';a.style='color:#9fd5ff';a.textContent='车辆生成先验：新实验与架构';n.append(a);document.body.prepend(n);};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',add);else add();})();\n'''
            source.write_text(js,encoding='utf-8')
if __name__=='__main__':main()
