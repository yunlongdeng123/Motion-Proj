"""车辆生成先验与增密道路的组件图；虚线表示待执行环节。"""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parents[2]/'outputs/vehicle_completion_prior_20261122'
OUT.mkdir(parents=True,exist_ok=True)
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="940" viewBox="0 0 1680 940"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6" fill="#8db7d9"/></marker></defs><rect width="1680" height="940" rx="20" fill="#0b1420"/><style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#e7f0f9}.title{font-size:34px;font-weight:700}.h{font-size:24px;font-weight:700}.s{font-size:20px}.muted{fill:#a1b6cb}.line{stroke:#8db7d9;stroke-width:3;fill:none;marker-end:url(#arrow)}.pending{stroke-dasharray:9 6}</style><text x="60" y="63" class="title">删除前景 a9 · 恢复后方 a20 与道路</text><text x="60" y="101" class="s muted">完整条件图与粗框 → 生成表面与材质 → 表面高斯 → 联合渲染</text>''']
def box(x,y,w,h,title,lines,color='#1c3e56',pending=False):
    dash=' stroke-dasharray="9 6"' if pending else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{color}" stroke="#4b7d9d" stroke-width="2"{dash}/>')
    parts.append(f'<text x="{x+22}" y="{y+40}" class="h">{escape(title)}</text>')
    for i,line in enumerate(lines):parts.append(f'<text x="{x+22}" y="{y+79+i*31}" class="s">{escape(line)}</text>')
def arrow(x1,y1,x2,y2,pending=False):parts.append(f'<path class="line{" pending" if pending else ""}" d="M{x1} {y1} L{x2} {y2}"/>')
box(50,145,400,190,'1 观测与固定几何',['RGB／实例掩码／相机与轨迹','a20 对象坐标实测点','仅 raw86／110 用于构建'])
box(50,385,400,230,'2 当前道路资产',['固定路面／连续世界材质','跨帧 Alpha 找缺口，射线求交','先启用闲置锚点，再局部补点','先覆盖，后以二维教师拟合外观','车辆实验固定道路几何与覆盖'],'#3b3023')
box(510,145,510,190,'3a 缺失车体生成',['Omni · 整车条件图＋已知粗框','可见像素保护，未知部分先补全','冻结生成器 · 连续网格与米制映射'],'#183d3d')
box(510,385,510,230,'3b 车辆材质与表面高斯',['TRELLIS.2 · 固定网格材质先验','真实 RGB 拟合，测距独立核查','沿表面采样，限制法向／切向足迹','未知表面保留生成材质与来源'],'#183d3d')
box(1080,145,550,190,'4 统一渲染与联合优化',['车辆对象坐标 → 每帧刚体变换','道路＋车辆＋静态背景共同排序','KEEP／DELETE · RGB／深度／Alpha'],'#24394a')
box(1080,385,550,230,'监督与贡献隔离',['真实可见车体 → 车辆参数','冻结道路教师 → 道路参数','道路损失显式截断车辆梯度','分别看车辆单层与共同遮挡贡献'],'#24394a')
arrow(450,240,510,240);arrow(765,335,765,385);arrow(1020,500,1050,500)
parts.append('<path class="line" d="M1050 500 L1050 240 L1080 240"/>')
parts.append('<path class="line" d="M250 615 L250 650 L1355 650 L1355 615"/>')
arrow(1355,335,1355,385)
box(50,710,760,160,'分阶段验收',['完整网格 → 材质 → 高斯单层 → 最终联合','真实可见细节、表面距离、外溢与其他车辆变化'],'#182739')
box(870,710,760,160,'持久资产与实验范围',['网格／UV材质／高斯／身份／轨迹／来源','本轮 25k 表面高斯；几何与轨迹固定，外观短程优化'],'#182739')
parts.append('</svg>')
(OUT/'architecture.svg').write_text('\n'.join(parts),encoding='utf-8')
print(OUT/'architecture.svg')
