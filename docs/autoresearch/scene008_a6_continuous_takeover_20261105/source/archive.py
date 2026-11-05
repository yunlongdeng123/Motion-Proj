"""Archive the measured Scene008 support experiment into the research checkout."""
import json
import shutil
import sys
from pathlib import Path

run = Path(sys.argv[1])
repo = Path(sys.argv[2])
dest = repo / 'docs/autoresearch' / run.name
dest.mkdir()
(dest / 'source').mkdir()
for file in (run / 'source').glob('*.py'):
    shutil.copyfile(file, dest / 'source' / file.name)
delivery = run / 'delivery'
names = ('summary.json', 'verify.json', 'replay_receipt.json', 'fixed_metrics.csv', 'accepted.npz', 'black_search.json',
         'shader_validation.json', 'tradeoff.json', 'geometry_black_search.json',
         'geometry_verified.json', 'combined_verified.json', 'joint_optimized_verified.json', 'joint_optimized_trials.json')
for name in names:
    if (delivery / name).exists():
        shutil.copyfile(delivery / name, dest / name)
shutil.copyfile(run / 'delivery.sha256', dest / 'delivery.sha256')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


s = read(delivery / 'summary.json')
assert s['status'] == 'PASS' and s['frames'] == 49 and s['fixed_hidden_pixels'] == 26229
b, a = (s['black'][k] for k in ('before', 'after'))
keep, delete = (s['street'][k] for k in ('keep', 'delete'))
name = {'tradeoff': '保护余量重分配', 'geometry': 'LiDAR 下界局部裁剪', 'combined': '余量重分配与 LiDAR 局部裁剪', 'joint_optimized': 'LiDAR 裁剪后重分配'}[s['selected']]
geometry = read(delivery / 'geometry_black_search.json')
raw = read(delivery / 'verify.json')
assert all(not r['violations'] for r in raw['black']) and not raw['chosen']['street_failures']
template = """# Scene008 a6 局部接管

承接[车体半空间裁剪](../scene008_a6_ray_switch_20261105/README.md)与[道路颜色校正](../scene008_ray_road_quality_20261105/README.md)。黑底为主实验，另外在无编辑和仅删 a37 的街景分别复核。新增重建训练 0 步。PVD 的几何、颜色与透明度固定，原生 10K 和道路 1K 检查点保留。

~~~mermaid
flowchart LR
    A["原生 a6 与固定 PVD"] --> C["车体局部接管账本"]
    B["RGB、LiDAR 与逐帧保护"] --> C
    C --> D["同一次深度排序和合成"]
    E["冻结街景与共享道路颜色场"] --> D
    D --> F["49 帧黑底与双状态验收"]
~~~

第一组沿用相同行与平面，在原矩匹配行和精确射线裁剪行之间按持久系数分配基元不透明度。统一接管和单向分组接管未通过已接近阈值的 raw137 保护。第二组根据真实渲染的有限差分估计变化量，在 66 个组之间同时分配新增接管、撤回旧接管及小幅残余不透明度调整，随后用实际 49 帧重放筛选可行方案。可见贡献继续通过原生中心排序合成。

第三组以真实 LiDAR 的车体坐标下界 {lidar:.5f} m 提议垂直切面和局部倾斜切面。候选排除了旧平面同位替换行和实测 LiDAR 贡献较大的行。选出 {geom_rows} 行，单独施用时黑底显露率为 {geom_ratio:.3%}。所有裁剪和接管系数跨相机、时间和编辑状态共用。直接叠加分别拟合的接管强度与裁剪会触发保护，另以裁剪为起点重做强度搜索。

| 固定域指标 | 上一轮 | {name} |
| --- | ---: | ---: |
| 黑底下部显露 | {black_old:.3%} | {black_new:.3%} |
| 删 a37 街景下部显露 | {street_old:.3%} | {street_new:.3%} |
| 黑底已观测 a6 RGB MAE | {rgb_old:.6f} | {rgb_new:.6f} |
| 黑底实测深度 MAE (m) | {depth_old:.6f} | {depth_new:.6f} |
| 无编辑道路 RGB MAE | {road_keep_old:.6f} | {road_keep_new:.6f} |
| 删 a37 道路 RGB MAE | {road_delete_old:.6f} | {road_delete_new:.6f} |

49 帧沿用黑底保护阈值，并独立复核两种编辑状态的可见车身、非目标域、道路和接缝。固定下部范围为 26,229 个候选像素，分母为同一批 431 个下部 PVD 行的独立透明度。各阶段均为同场景机制验证，隐藏部分缺独立几何真值。

工程重放单列。Python 3.9 的 zip 参数兼容修复后重放黑底对照。首次街景校验把未打标签的静态贡献算进了相等式，修正为对余量的非负检查后重放搜索，并在街景入口前保存候选账本。原日志和检查点保留。

[逐帧指标](fixed_metrics.csv) · [验收](verify.json) · [选中账本](accepted.npz) · [代码](source)。
"""
report = template.format(lidar=geometry['design']['lidar_lower_m'], geom_rows=sum(geometry['selected'].values()),
    geom_ratio=geometry['ratio'], name=name, black_old=b['ratio'], black_new=a['ratio'],
    street_old=delete['before_fraction'], street_new=delete['after_fraction'],
    rgb_old=b['known_rgb'], rgb_new=a['known_rgb'], depth_old=b['measured_depth'], depth_new=a['measured_depth'],
    road_keep_old=keep['road_before'], road_keep_new=keep['road_after'],
    road_delete_old=delete['road_before'], road_delete_new=delete['road_after'])
(dest / 'README.md').write_text(report, encoding='utf-8')
link = f'autoresearch/{run.name}/README.md'
status = f'\n2026-11-05，Scene008 a6 采用{name}。黑底下部显露 {b["ratio"]:.2%} → {a["ratio"]:.2%}，删 a37 街景显露 {delete["before_fraction"]:.2%} → {delete["after_fraction"]:.2%}。49 帧保护与道路复核通过。[实验、代码与逐帧指标]({link})。\n'
index = f'\n## Scene008 a6 局部接管（2026-11-05）\n\n- [同域黑底显露和两种编辑状态复核]({link})。\n'
for file, heading, new in (('EXPERIMENTS.md', '# 实验索引\n', index),
                           ('RESEARCH_STATUS.md', '# 当前研究状态\n', status)):
    path = repo / 'docs' / file
    old = path.read_text(encoding='utf-8')
    assert old.startswith(heading)
    path.write_text(heading + new + old[len(heading):], encoding='utf-8')
