"""Version the heuristic validation protocol and its measured failures."""
from pathlib import Path
import json
import shutil
import sys

run, repo=(Path(p) for p in sys.argv[1:])
root=repo/'docs/autoresearch'/run.name
root.mkdir()
(root/'source').mkdir()
for p in (run/'source').glob('*.py'):
    if p.name=='evaluate.py':
        continue
    text=p.read_text(encoding='utf-8')
    (root/'source'/p.name).write_text(text.rstrip('\n')+'\n',encoding='utf-8')
delivery=run/'delivery'
for p in delivery.iterdir():
    if p.suffix in ('.json','.csv','.npz'):
        shutil.copyfile(p,root/p.name)
s=json.loads((delivery/'summary.json').read_text(encoding='utf-8'))
assert s['status']=='EXECUTION_PASS' and not s['method_accepted']
proof=json.loads((delivery/'verification.json').read_text(encoding='utf-8'))
assert proof['exact_replay'] and len(proof['checked'])==72
p=s['positive_held']
def v(arm,k):
    return p[arm][k]['mean']
(root/'README.md').write_text(f"""# Scene008 启发式接管验证

本轮验证三态证据和固定接管规则。父资产采用[实际射线接管版](../scene008_a6_depth_order_20261105/README.md)，PVD 和道路固定。实验执行通过，方法未通过保护验收。新增训练 0 步。

~~~mermaid
flowchart LR
  A["原始 RGB、LiDAR 第一回波与轨迹"] --> B["实测、自由线段与未知"]
  P["固定 PVD 与父场景"] --> C["同一裁剪和实测代理面"]
  B --> C
  C --> R["同一次深度排序合成"]
  R --> E["构建及留出评价"]
~~~

## 固定对照

- E0 分离原 free 的框外射线与实测穿越项，审计下部候选像素的证据状态。
- E1 对同一后下部体积统一裁剪 25%，比较原资产、裁剪、仅加代理面、裁剪加代理面。
- E2 以构建帧测量选定车尾面，统一裁剪 60%，使用相同四组检验实测面重表达。
- E3 解码原始 raw0–198 的第一有效回波。从真实侧向或 TOP 逐像素传感器原点出发，自由段终止于回波点前 15 cm。

九个留出帧为 89、95、101、105、113、121、125、130、137。新增代理面的锚点、RGB 和强度仅用构建帧。父模型、PVD、下部评价域和体积含历史全场景上下文，该划分只检验新增代理面构建。原始场景内机制验证缺少独立隐藏真值。

## 证据结论

原有 {s['legacy_free']:,} 个 free 像素全部为膨胀框外项，实测穿越项为 {s['legacy_free_direct_through']}。下部 {s['lower_pixels']:,} 个固定像素中，{s['lower_measured_occlusion']:,} 个有前方 a37 的实测遮挡，{s['lower_unknown']:,} 个对 a6 为未知。原始第一回波新增 {s['first_return_witnesses']:,} 条候选体积内自由线段，其中 {s['build_beam_witnesses']} 条属于构建集。线段命中不等价于整块体积为空，扫描内物体运动仍带来不确定性。

实际下部组有 {s['build_measured_matches']:,} 次构建投影通过测量核对，未触发实测损伤代理面规则。覆盖率触发也没有检测到正对照的损伤，进一步加入真实 a6 RGB 损伤触发后生成 {s['positive_proxy_rows']} 个代理面。每个面至少有两个构建视角支持。代理面不透明度用单步插入颜色估计，最大 0.35。

## 留出实测面正对照

| 对照 | 面板 RGB MAE | 面板深度 MAE (m) | 黑底保护失败 |
| --- | ---: | ---: | ---: |
| 原资产 | {v('parent','panel_rgb'):.5f} | {v('parent','panel_depth'):.5f} | {p['parent']['frame_failures']} / 9 |
| 同一裁剪 | {v('cut','panel_rgb'):.5f} | {v('cut','panel_depth'):.5f} | {p['cut']['frame_failures']} / 9 |
| 仅加代理面 | {v('proxy_only','panel_rgb'):.5f} | {v('proxy_only','panel_depth'):.5f} | {p['proxy_only']['frame_failures']} / 9 |
| 裁剪与代理面 | {v('rescue','panel_rgb'):.5f} | {v('rescue','panel_depth'):.5f} | {p['rescue']['frame_failures']} / 9 |

独立叠加不能形成安全接管。当前单面颜色插入估计未协调其他代理面的重叠和旧支撑的剩余贡献，留出 RGB、深度均恶化。下一组实验应按实测表面分组，把新代理面的覆盖预算与对应旧支撑的退出绑定，保留仅加代理面控制。未知区与实测区继续分开评价。

全 49 帧完成黑底、无编辑和删 a37 三状态四组评价。独立打开保存账本后，三个预设探针 raw105、113、130 的 72 项重放与记录精确一致。独立重放范围单列，未将方法失败写成质量通过。首次重放修正了 Python 模块搜索路径同名冲突，原日志保留。

~~~bash
python source/evidence_audit.py --output evidence_audit.json
python source/rescue.py
python source/positive.py
python source/heuristic_evaluate.py rescue --suffix _rgb
python source/heuristic_evaluate.py positive --suffix _rgb
mkdir first_return
python source/first_return_audit.py --output first_return
python source/beam_anchor_proximity.py
python source/verify_saved.py
~~~

命令在本轮远端 campaign 目录及 DrivingEditor 环境执行，依赖同级历史资产和原始 TFRecord。

[实验合同](contract.json) · [逐帧指标](fixed_metrics.csv) · [正对照](positive_rgb_evaluation.json) · [第一回波核查](first_return_summary.json) · [重放核对](verification.json)。
""",encoding='utf-8')
shutil.copyfile(run/'delivery.sha256',root/'delivery.sha256')
link=f'autoresearch/{run.name}/README.md'
updates=[
('EXPERIMENTS.md','# 实验索引\n',f'\n## Scene008 启发式接管验证（2026-11-05）\n\n- [固定裁剪、实测面正对照及原始第一回波核查]({link})。方法未通过保护验收。\n'),
('RESEARCH_STATUS.md','# 当前研究状态\n',f'\n2026-11-05，Scene008 完成启发式规则首轮验证。旧 free 全为膨胀框外项，原始 199 帧新增 1,003 条候选体积内第一回波自由线段。独立叠加式实测代理面在留出正对照中恶化 RGB 与深度，未采用。[证据、对照和下一步耦合接管设计]({link})。\n')]
for name,heading,insert in updates:
    path=repo/'docs'/name
    old=path.read_text(encoding='utf-8')
    assert old.startswith(heading)
    path.write_text(heading+insert+old[len(heading):],encoding='utf-8')
