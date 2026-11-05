"""Publish the paired R3 follow-up after review has been built."""
import json
from pathlib import Path
import shutil
import sys

from review import RUN, LABELS


def publish(repo):
    d = RUN/'delivery'
    summary = json.loads((d/'summary.json').read_text())
    p = summary['pooled']
    slug = RUN.name
    report = repo/'docs/autoresearch'/slug
    report.mkdir()
    fields = []
    for key, label in LABELS.items():
        cells = []
        for arm in ('parent','new','old_r3'):
            value = p[key].get(arm)
            cells.append(f"{value['mean']:.6f}" if value else '—')
        fields.append(f"| {label} | {p[key]['new']['count']:,} | {' | '.join(cells)} |")
    table = '\n'.join(fields)
    mark = f"""# Scene008 原生 10K 接 R3 道路恢复 1K

父模型使用[实例监督原生 10K](../scene008_native_ownership_20261105/README.md)检查点。对照为[旧强编辑 R3](../scene008_a37_only_B10k_road_20261103/README.md)。保留新父模型的 a6、a37、静态层、环境与原生时间状态。原始 49 帧相机和两个渲染状态共同使用同一份道路资产和静态门控。

```mermaid
flowchart LR
  I["199 帧 RGB、LiDAR 和相机"] --> R["世界坐标道路<br/>重算证据、几何和颜色"]
  N["冻结的实例监督 10K"] --> G["候选静态行<br/>1K 优化透明度门控"]
  N --> D["a6、a37、环境"]
  R --> J["联合深度排序"]
  G --> J
  D --> J
  J --> O["无编辑、删除 a37<br/>固定 49 帧配对"]
```

## 适配和验收

- 沿用 R3 的道路几何、颜色校准、1K 优化流程、学习率 0.06、门控初值 2、损失权重、seed 0 和采样规则。颜色校准、道路覆盖和候选行依赖父模型，按原流程重算。道路资产确定后保持固定。
- 删除操作按持久归属隐藏 a37 动态行。a6 在选候选、训练和评价中参与联合深度排序。可见 a6 轮廓及 1 像素边缘加入旧车辆保护掩码。
- 静态候选按删 a37 后的 49 帧联合贡献重新登记，保留 {summary['selection']['candidate_rows']:,} 行供训练，保护优势行 {summary['selection']['protected_dominant_rows']:,}。训练时只更新这些行的共享透明度系数。
- 32 步真实 GPU smoke 后两次完整恢复并分别训练一步，三帧渲染字节一致。正式从相同初始门控跑 1,000 步，逐步相机序列与旧 R3 完全一致，耗时 {summary['formal']['seconds']:.1f} 秒。最终恢复与 49 帧评价完成。父模型四个检查点 SHA-256 与上轮审计一致，关闭道路时的 49 帧删车 RGB 与直接原生渲染一致。

## 固定域配对

| 指标 | 像素数 | 本轮 10K | 本轮 10K+1K | 旧强编辑 R3 |
| --- | ---: | ---: | ---: | ---: |
{table}

raw113 与 raw130 的删车路面大暗块大幅减轻，修复区边缘仍有接缝和色差。a6 的可见上部保持，未观测下部缺可靠外观。无编辑中 a37 轮廓周围的原生溢出依旧可见，路面误差小幅上升。RGB 均从一致量化的 PNG 重算，a6 贡献来自联合排序的半精度读出，深度使用实测相机 Z。显露道路 RGB 来自同场景的跨时刻观测。道路准备保留原 R3 的少量有标记插值样本，固定显露路面评价使用重观测像素。a6 未观测下部缺独立真值，本轮只训练道路门控，未追加车辆补全或车辆外观拟合。

[49 帧指标与回执](summary.json) · [1K 回放校验](checkpoint_receipt.json) · [R3 训练入口](source/repair37.py) · [完整队列](source/run_road_queue.py)。

工程审计通过，人工验收字段 `human_verdict=null`。AutoDL 电源由用户控制。没有新增跨版本失败条目。
"""
    (report/'README.md').write_text(mark,encoding='utf-8')
    for name in ('summary.json','checkpoint_receipt.json','fixed_ray_metrics.csv'):
        shutil.copyfile(d/name,report/name)
    shutil.copyfile(RUN/'settings.json',report/'settings.json')
    (report/'source').mkdir()
    for source in (RUN/'source').iterdir():
        if source.suffix in ('.py','.sh') and '.pre_' not in source.name:
            (report/'source'/source.name).write_bytes(source.read_bytes().rstrip(b'\n')+b'\n')
    index = repo/'docs/EXPERIMENTS.md'
    original = index.read_text(encoding='utf-8')
    pos = original.index('\n\n')+2
    index.write_text(original[:pos]+f"""## Scene008 原生 10K + 道路恢复 1K（2026-11-05）

- `{slug}`。[R3 接线与 49 帧固定域对照](autoresearch/{slug}/README.md)。

"""+original[pos:],encoding='utf-8')
    b,a = (p['road_reobserved_after_delete'][key]['mean'] for key in ('parent','new'))
    xb,xa = (p['a6_visible_rgb'][key]['mean'] for key in ('parent','new'))
    old = p['road_reobserved_after_delete']['old_r3']['mean']
    (repo/'docs/RESEARCH_STATUS.md').write_text(f"""# 当前研究状态

2026-11-05，Scene008 已完成实例监督原生 10K 后接 R3 道路恢复 1K，以及完整状态恢复和 49 帧评价。[接线、配对指标与回执](autoresearch/{slug}/README.md)

删 a37 后，固定显露道路 RGB MAE {b:.4f} → {a:.4f}，旧强编辑 R3 为 {old:.4f}。已观测 a6 的 RGB MAE {xb:.4f} → {xa:.4f}。无编辑质量使用同一份道路和门控重算。人工验收尚未登记，AutoDL 电源由用户控制。

[原生 10K 的早期验收与对照](autoresearch/scene008_native_ownership_20261105/README.md)
""",encoding='utf-8')


if __name__ == '__main__':
    publish(Path(sys.argv[1]))
