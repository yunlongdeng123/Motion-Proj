"""Archive the verified ordering controls and support ledger."""
from pathlib import Path
import json
import shutil
import sys

run, repo = (Path(x) for x in sys.argv[1:])
d = repo/'docs/autoresearch'/run.name
d.mkdir()
(d/'source').mkdir()
for file in (run/'source').iterdir():
    if file.suffix in ('.py','.cu') and file.name != 'replay_median.cu':
        code = file.read_text(encoding='utf-8')
        (d/'source'/file.name).write_text(code.rstrip('\n')+'\n', encoding='utf-8')
pub = run/'delivery'
for name in ('summary.json','black_order.json','order_controls.json','support_search.json','support_verified.json',
             'support_accepted.npz','replay_receipt.json','fixed_metrics.csv'):
    shutil.copyfile(pub/name,d/name)
shutil.copyfile(run/'delivery.sha256',d/'delivery.sha256')
if (run/'support_search.json.first').exists():
    shutil.copyfile(run/'support_search.json.first',d/'unrestricted_support_control.json')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


s = read(pub/'summary.json')
receipt = read(pub/'replay_receipt.json')
assert s['status'] == receipt['status'] == 'PASS' and s['fixed_hidden_pixels'] == 26229
assert receipt['exact_record_replay'] and s['native_center_parity'] == {'rgb':0.,'alpha':0.,'depth_sum':0.,'extra':0.}
a,b = s['black']['before'],s['black']['after']
k,x = s['street']['keep'],s['street']['delete']
receipt_orders = read(pub/'order_controls.json')
template = """# Scene008 a6 射线排序与局部接管

承接[局部接管](../scene008_a6_continuous_takeover_20261105/README.md)。PVD、道路颜色场、原生几何和已采用的裁剪保持固定。黑底为主实验，另用相同资产复核无编辑和删 a37 的街景。新增重建训练 0 步。

~~~mermaid
flowchart LR
  A["固定原生场景与 PVD"] --> C["实际光栅候选与每条射线的 alpha"]
  B["真实 RGB、LiDAR 与保护域"] --> D["逐行遮挡影响和约束"]
  C --> O["射线排序对照"]
  C --> D
  D --> E["持久不透明度系数"]
  E --> F["49 帧黑底、无编辑和删车"]
~~~

复用同一个私有渲染器的光栅输出缓冲区，逐射线重放。中心顺序在 49 帧黑底和两种街景状态的 RGB、alpha、深度辅助和标签贡献逐像素一致。诊断比较裁剪后条件深度中位数和三等份光学厚度采样。采用相同候选、材料和总不透明度。单点采用体深度中位数，三份采样等分光学厚度。原生中心深度辅助保留。两种按 ±0.25 或 ±0.75 m 限制相对中心的移动。零位移分片控制区分终止阈值影响。仅修正裁剪行且保护通过的最好黑底增益为 {order_gain:.4f} 个百分点，街景增益 {street_order_gain:.4f} 个百分点。全量改序破坏可见车身。正式方案保持中心顺序。

在实际参与光栅的行上，固定顺序和贡献集合，求每行不透明度变化对下游 PVD 和保护域的局部导数。线性规划跨 49 帧分配衰减，显露率仅按预先固定的下部像素计算。初版局部模型在 raw137 实际重放失败。采用版冻结该帧实际命中保护像素的身份，且为 raw97、112、113 的非目标 RGB 和 alpha 变化使用保守的一阶估计。所有候选再由原光栅路径复核，实际门槛沿用上一轮。每个行身份的衰减上限 {cap:.0%}，包含同源子行，跨帧和编辑状态固定。

| 固定域指标 | 上一轮 | 本轮 |
| --- | ---: | ---: |
| 黑底下部显露 | {before:.3%} | {after:.3%} |
| 删 a37 街景下部显露 | {street_before:.3%} | {street_after:.3%} |
| 黑底已观测 a6 RGB MAE | {known_before:.6f} | {known_after:.6f} |
| 黑底实测深度 MAE (m) | {depth_before:.6f} | {depth_after:.6f} |
| 无编辑道路 MAE | {keep_before:.6f} | {keep_after:.6f} |
| 删 a37 道路 MAE | {road_before:.6f} | {road_after:.6f} |

分母仍为同一批下部 PVD 的独立透明度，共 26,229 个固定像素。重开账本后独立重放 49 帧黑底和两种编辑状态，指标记录精确相同。该验证使用同场景观测，隐藏区域缺少独立几何真值。

~~~bash
/usr/local/cuda-11.8/bin/nvcc -O3 -arch=sm_89 --shared -Xcompiler -fPIC -o libreplay_quantile.so source/replay.cu
python source/depth_order.py --family median
python source/depth_order.py --family quantile
python source/order_controls.py
python source/support.py
python source/replay_support.py
~~~

命令在远端本次 campaign 目录运行，依赖同级上一轮资产、DrivingEditor 环境和先前的私有光栅器。

[排序对照](order_controls.json) · [逐帧指标](fixed_metrics.csv) · [持久系数](support_accepted.npz) · [独立重放](replay_receipt.json)。
"""
best_order = max(v['ratio'] for v in s['ordering'].values() if not v['failures'])
street_order_gain = max(x['pool']['delete']['after_fraction']-x['pool']['delete']['before_fraction']
                        for x in receipt_orders['street'].values() if not x['issues'])
(d/'README.md').write_text(template.format(
    order_gain=100*(best_order-a['ratio']),street_order_gain=100*street_order_gain,cap=s['max_attenuation'],
    before=a['ratio'],after=b['ratio'],street_before=x['before_fraction'],street_after=x['after_fraction'],
    known_before=a['known_rgb'],known_after=b['known_rgb'],depth_before=a['measured_depth'],depth_after=b['measured_depth'],
    keep_before=k['road_before'],keep_after=k['road_after'],road_before=x['road_before'],road_after=x['road_after']),
    encoding='utf-8')
link = f'autoresearch/{run.name}/README.md'
status=f'\n2026-11-05，Scene008 a6 在实际光栅射线上按逐行遮挡影响分配持久衰减。黑底显露 {a["ratio"]:.2%} → {b["ratio"]:.2%}，删 a37 街景 {x["before_fraction"]:.2%} → {x["after_fraction"]:.2%}。49 帧双状态保护与独立重放通过。[实验、排序对照与账本]({link})。\n'
index=f'\n## Scene008 a6 射线排序与局部接管（2026-11-05）\n\n- [深度顺序对照、逐行接管和 49 帧复核]({link})。\n'
for name,heading,block in (('RESEARCH_STATUS.md','# 当前研究状态\n',status),('EXPERIMENTS.md','# 实验索引\n',index)):
    p=repo/'docs'/name
    old=p.read_text(encoding='utf-8')
    assert old.startswith(heading)
    p.write_text(heading+block+old[len(heading):],encoding='utf-8')
