"""Expose local damage and link the next fixed protocol in the report."""
import json
from pathlib import Path
import shutil
import sys

root=Path(sys.argv[1])
source=root/'delivery'
real=json.loads((source/'rescue_rgb_evaluation.json').read_text())
positive=json.loads((source/'positive_rgb_evaluation.json').read_text())
rows=[]
for label,data in (('实际下部',real),('实测面正对照',positive)):
    for arm,name in (('cut','裁剪'),('proxy_only','仅加代理面'),('rescue','裁剪与代理面')):
        b=data['stats']['held'][arm]['black']
        k=data['stats']['held'][arm]['keep']
        delta=b['damage']['outside']
        rows.append(f'<tr><th>{label} · {name}</th><td>{delta["above_002"]}</td><td>{delta["max"]:.5f}</td><td>{k["damage"]["a37"]["max"]:.6f}</td></tr>')
section='<section><h2>局部损伤检查</h2><p>同一批留出像素，相对原资产计量。非目标域为前序固定黑底保护域。局部最大值与超阈像素单列。</p><table><tr><th>对照</th><th>非目标 RGB 变化大于 0.02 的像素</th><th>非目标最大变化</th><th>无编辑 a37 最大变化</th></tr>'+''.join(rows)+'</table></section>'
html=(source/'index.html').read_text()
html=html.replace('<section><h2>下一步固定验证</h2>',section+'<section><h2>下一步固定验证</h2>')
html=html.replace('当这组耦合启发式通过无编辑与删车的留出检查后，再比较学习型动作预测。',
                  '下一组的表面分组、配对退出、固定转移比例和验收已写入 <a href="next_coupled_contract.json">耦合接管实验合同</a>。这些是待验证设计。当耦合启发式通过无编辑与删车的留出检查后，再比较学习型动作预测。')
shutil.copyfile(root/'next_coupled_contract.json',source/'next_coupled_contract.json')
(source/'index.html').write_text(html,encoding='utf-8')
