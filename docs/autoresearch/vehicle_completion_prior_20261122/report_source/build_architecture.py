"""车辆生成先验与增密道路的组件图；虚线表示待执行环节。"""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parents[2]/'outputs/vehicle_completion_prior_20261122'
OUT.mkdir(parents=True,exist_ok=True)
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="940" viewBox="0 0 1680 940"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6" fill="#8db7d9"/></marker></defs><rect width="1680" height="940" rx="20" fill="#0b1420"/><style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#e7f0f9}.title{font-size:34px;font-weight:700}.h{font-size:24px;font-weight:700}.s{font-size:20px}.muted{fill:#a1b6cb}.line{stroke:#8db7d9;stroke-width:3;fill:none;marker-end:url(#arrow)}.pending{stroke-dasharray:9 6}</style><text x="60" y="63" class="title">删除前景 a9 · 恢复后方 a20 与道路</text><text x="60" y="101" class="s muted">观测车身保护 → 缺失表面教师 → 灰模验收 → Gaussian 蒸馏与联合渲染</text>''']
def box(x,y,w,h,title,lines,color='#1c3e56',pending=False):
    dash=' stroke-dasharray="9 6"' if pending else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{color}" stroke="#4b7d9d" stroke-width="2"{dash}/>')
    parts.append(f'<text x="{x+22}" y="{y+40}" class="h">{escape(title)}</text>')
    for i,line in enumerate(lines):parts.append(f'<text x="{x+22}" y="{y+79+i*31}" class="s">{escape(line)}</text>')
def arrow(x1,y1,x2,y2,pending=False):parts.append(f'<path class="line{" pending" if pending else ""}" d="M{x1} {y1} L{x2} {y2}"/>')
box(50,145,400,190,'1 观测与固定几何',['RGB／实例掩码／相机与轨迹','a20 对象坐标实测点','仅 raw86／110 用于构建'])
box(50,385,400,230,'2 当前道路资产',['固定路面／连续世界材质','跨帧 Alpha 找缺口，射线求交','先启用闲置锚点，再局部补点','先覆盖，后以二维教师拟合外观','车辆实验固定道路几何与覆盖'],'#3b3023')
box(510,145,510,190,'3a 观测车身＋教师补片',['锁定可信观测的几何与颜色','遮挡感知教师，提出缺失补片','对象坐标对齐，保持部件连接'],'#183d3d',True)
box(510,385,510,230,'3b 灰模验收 · 材质之前',['部件关系／轮位与接地','真实可见轮廓／实测距离贴合','可见车身保真／生成仅限缺失区','当前候选未通过，阻断材质与训练'],'#4a3326')
box(1080,145,550,190,'通过后：材质＋Gaussian 蒸馏',['缺失区几何／轮廓／新视角监督','可见部分保留真实多帧外观','表面约束与受控足迹，持久保存'],'#24394a',True)
box(1080,385,550,230,'4 统一渲染与联合优化',['车辆刚体变换，与道路共同排序','真实车辆／道路教师分支梯度隔离','分别检查单层 RGB、Alpha 与遮挡','持久保存表面、材质、高斯与来源'],'#24394a',True)
arrow(450,240,510,240,True);arrow(765,335,765,385);arrow(1020,500,1050,500,True)
parts.append('<path class="line pending" d="M1050 500 L1050 240 L1080 240"/>')
parts.append('<path class="line pending" d="M250 615 L250 650 L1355 650 L1355 615"/>')
arrow(1355,335,1355,385,True)
box(50,710,760,160,'当前结果的定位',['既有材质与 400 步结果保留为未通过灰模门的诊断','网格完整、贴图齐全与 Alpha 高均不替代车体验收'],'#182739')
box(870,710,760,160,'新版协议',['实线：已有组件或已启用的验收门；虚线：待重新推进','灰模通过后，才进入材质、表面高斯和联合优化'],'#182739')
parts.append('</svg>')
(OUT/'architecture.svg').write_text('\n'.join(parts),encoding='utf-8')
print(OUT/'architecture.svg')
