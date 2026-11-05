"""Publish the frozen a6 local-fusion study with its scoped readout and provenance."""
from pathlib import Path
import json
import shutil
import sys

from report_fusion import LABELS

ROOT=Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(repo):
    result=load(ROOT/'delivery/summary.json')
    chosen=result['chosen']
    metrics=result['metrics']
    old=metrics['native']
    best=metrics[chosen]
    rel=Path('docs/autoresearch/scene008_a6_local_fusion_20261105')
    doc=repo/rel
    doc.mkdir()
    formatted=[]
    for name,item in metrics.items():
        failures=result['guard_records'][name]
        status='对照' if name in ('native','pvd','direct') else '通过' if not any(failures.values()) else f"失败 {len(failures['frames'])} 帧"
        formatted.append(f"| {LABELS[name]} | {item['reveal_fraction']:.2%} | {item['known_rgb']:.6f} | {item['measured_depth']:.6f} | {item['pvd_on_known']:.6f} | {status} |")
    table='\n'.join(formatted)
    pilot=[]
    for raw,items in result['pilot_metrics'].items():
        a,b=items['direct'],items[chosen]
        pilot.append(f"| {raw} | {a['reveal_fraction']:.2%} | {b['reveal_fraction']:.2%} |")
    pilots='\n'.join(pilot)
    text=f"""# Scene008 a6 车尾下部局部融合

`scene008_a6_local_fusion_20261105`。沿用[原生 10K + 道路 1K](../scene008_native_road1k_20261105_r1/README.md)的 a6，以及[固定 PVD 和黑底遮挡诊断](../scene008_native_road1k_20261105_r1/pvd_compare.md)的同一批生成行。主实验仅渲染黑底 a6。另在三帧联合街景诊断中保留道路与环境，检验前车和原生 a6 的遮挡。没有训练更新或资产回写。

```mermaid
flowchart LR
  O["跨帧 RGB、LiDAR 和 a6 轨迹"] --> E["固定可见保护域与下部缺失射线"]
  P["固定 PVD 提案"] --> L["车体坐标选出下部提案"]
  L --> E
  N["原生 a6 动态行"] --> A["αT 前缀归因与局部足迹"]
  E --> A
  A --> F["整行退役或保留侧拆分"]
  L --> J["同次黑底深度排序"]
  F --> J
  J --> Q["四列、缺失贡献图、RGB / 深度"]
  N --> S["单独的街景遮挡诊断"]
```

## 冻结域与融合

- 4,667 个实测车体锚点的最低点下方 3 cm 为 `z={result['evidence']['z_cut_m']:.4f} m`。取 a6 后半部且离实测锚点足够远的 {result['evidence']['lower_rows']} 个 PVD 提案行。融合只启用这些行，保留各行几何、颜色和透明度。独立 PVD 和直接叠加对照使用全量 1,477 行。
- 在上一轮固定隐藏足迹中，剔除原始可见 a6 轮廓的一像素边缘与实测深度，保留下部提案具有独立贡献的 {result['evidence']['target_pixels']:,} 个像素。分母为下部行独立渲染的透明度总和，分子为联合排序中的下部行贡献总和。全量独立 PVD 存在少量自遮挡，因而表内略低于 100%。缺失面没有独立真值。
- 候选归因使用 PVD 相机 Z 前缀内原生高斯的精确 αT 累积。深度边界来自固定提案的加权相机 Z，并采用保守分箱。跨帧投影贡献映射到车体坐标，对每行登记挡住下部的足迹和保留侧质心。低实测支撑且下部占主导的行使用共享退役系数。混合行按车体坐标中的分隔平面，只重表达保留一侧，使用截断高斯的矩匹配近似。每帧复用同一组行和规则。
- 主图采用 {LABELS[chosen]}。该版本用更宽的下部提案注册域，固定评价域仍保持原样；另外保护对 LiDAR 射线有可见支撑的原生行。RGB MAE 增量和相对原生 RGB 的差异分别限制为 0.003，实测深度 MAE 增量限制为 0.03 m，透明度至少保留 99%。总体和每帧均须通过，取通过版本中下部显露率最高者。
- 首次足迹版保存了账本和三帧后，遇到 Python 3.9 的链接 API 兼容错误。使用同一份账本、复用三帧资产后补齐其余帧。扩展注册版修正中间变量重名后重放。全程无优化器更新。

## 49 帧固定像素配对

| 方案 | 下部显露率 | 已观测 a6 RGB MAE | 实测深度 MAE (m) | 已观测 a6 的 PVD 贡献 | 保护门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
{table}

| 试看片 raw | 直接叠加 | 通过保护的局部融合 |
| --- | ---: | ---: |
{pilots}

通过保护的局部融合改善了下部显露，仍有大半下部独立贡献未进入联合图。全量直接叠加在已观测 a6 上贡献 0.2175，局部方案将该值降到约 0.000004。未做 LiDAR 保护的足迹拆分在 8 帧超过保护门槛。主结果是同场景可见性诊断，补全精度与独立场景泛化仍待验证。

[固定评价域](evidence_contract.json) · [逐帧指标及保护门槛](summary.json) · [街景诊断](street_diagnostic.json) · [融合渲染](source/fuse_footprint.py)。
"""
    (doc/'README.md').write_text(text,encoding='utf-8')
    for src,dst in [('delivery/summary.json','summary.json'),('delivery/fixed_metrics.csv','fixed_metrics.csv'),
                    ('evidence/contract.json','evidence_contract.json'),('street_diagnostic.json','street_diagnostic.json')]:
        shutil.copyfile(ROOT/src,doc/dst)
    (doc/'source').mkdir()
    for name in ('fuse_a6.py','fuse_footprint.py','report_fusion.py','publish.py'):
        (doc/'source'/name).write_bytes((ROOT/'source'/name).read_bytes().rstrip(b'\n')+b'\n')
    index=repo/'docs/EXPERIMENTS.md'
    content=index.read_text(encoding='utf-8')
    pos=content.index('\n\n')+2
    index.write_text(content[:pos]+"""## Scene008 a6 下部局部融合（2026-11-05）

- `scene008_a6_local_fusion_20261105`。[零训练黑底对照及逐帧保护检查](autoresearch/scene008_a6_local_fusion_20261105/README.md)。

"""+content[pos:],encoding='utf-8')
    (repo/'docs/RESEARCH_STATUS.md').write_text(f"""# 当前研究状态

2026-11-05，Scene008 已完成黑底 a6 局部融合和 49 帧配对，固定 PVD 参数，新增训练 0 步。[黑底对照、冻结域及逐帧保护检查](autoresearch/scene008_a6_local_fusion_20261105/README.md)

在固定的 26,229 个下部候选像素上，显露率从直接叠加 {metrics['direct']['reveal_fraction']:.2%} 升到 {best['reveal_fraction']:.2%}。与原生 a6 相比，已观测 RGB MAE {old['known_rgb']:.5f} → {best['known_rgb']:.5f}，实测深度 MAE {old['measured_depth']:.4f} → {best['measured_depth']:.4f} m。通过了逐帧保护门槛，下部仍未完整显露。

[原生 10K + 道路 1K 与 PVD 遮挡诊断](autoresearch/scene008_native_road1k_20261105_r1/README.md)。本轮的街景诊断与主实验单独归档。AutoDL 电源由用户控制。
""",encoding='utf-8')


if __name__=='__main__':
    main(Path(sys.argv[1]))
