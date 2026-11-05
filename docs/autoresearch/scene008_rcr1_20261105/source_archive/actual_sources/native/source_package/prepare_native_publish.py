"""Prepare the research index and report from the completed paired run."""
import json
from pathlib import Path
import shutil
import sys

from build_native_report import LABELS, json_read


def generate(run, review, repo):
    doc_name = 'scene008_native_ownership_20261105'
    doc = repo / 'docs/autoresearch' / doc_name
    doc.mkdir(exist_ok=False)
    old = json_read(run/'control/probe_10000/summary.json')['pooled']
    new = json_read(run/'formal/probe_10000/summary.json')['pooled']
    early = json_read(run/'formal/early_accept_01000.json')
    early_new = early['candidate']
    early_old = early['control']
    receipt = json_read(review/'data/audit.json')
    edit_old = json_read(run/'edit_union/delete_domains.json')
    edit_new = json_read(run/'edit_native/delete_domains.json')
    road_old = edit_old['road_reobserved_rgb']
    road_new = edit_new['road_reobserved_rgb']
    peak = {}
    for arm in ('smoke', 'formal'):
        lines = (run/arm/'runtime.jsonl').read_text().splitlines()
        peak[arm] = max(json.loads(line)['allocated_bytes'] for line in lines) / 2**30
    old_r1 = json_read(run.parent/'run_r1/smoke/returned.json')
    old_r2 = json_read(run.parent/'run_r2/smoke/returned.json')
    table = []
    for key, label in LABELS.items():
        table.append(f"| {label} | {new[key]['count']:,} | {old[key]['mean']:.6f} | {new[key]['mean']:.6f} |")
    entries = '\n'.join(table)
    readme = f"""# Scene008 原生 10K 实例归属与静态遮挡

`scene008_native_ownership_20261105 / run_r3`。对照为旧双动态并集监督的 1K 和 10K 检查点。沿原生 30K 学习率时钟训练 10,000 步，seed 0，使用 raw089–137。输入是同场景真实 RGB、LiDAR、相机及轨迹，以及 199 帧观测库的审核掩码和实测深度。未运行 10K 后的 1K 道路修复、旧表示替换或 PVD 接入。

```mermaid
flowchart LR
    I["RGB、LiDAR、轨迹与实例证据"] --> A6["a6 层：可见 RGB、覆盖、深度"]
    I --> A37["a37 层：可见 RGB、覆盖、深度"]
    I --> S["静态层与环境：背景监督"]
    I --> P["实测表面前的静态透射率"]
    P --> S
    S --> D["背景梯度驱动静态普通增密"]
    A6 --> J["联合深度排序：实例贡献"]
    A37 --> J
    S --> J
    J --> O["同一场景：原画面、删除 a37"]
    J --> Q["早期检查点：RGB、深度、遮挡"]
```

## 训练约束

- 沿用旧双动态初始化与持久行归属。a6 的 1,023 个初始源从静态层移出。clone、split、prune 继续传递源归属。原生动态仍采用原时域参数化。
- 两车各用可见内部的预乘 RGB 与覆盖拟合自己。覆盖的正域是各自可见掩码，负域是各自确定空射线，加上膨胀车体框和可见轮廓外的观测道路及保护区。两域合计按全图像素数归一，权重 100。两车 RGB 各保留旧动态系数 5。单车可见掩码之外的遮挡未知域保留。
- 同一次原生排序输出实例权重和实例条件深度。可见实例贡献负对数与确定空射线贡献平方按各域归一，权重 1。孤立及联合实例逆深度权重各 0.10，分别在实测域和已有联合支撑的实测域平均。
- 静态普通增密使用独立背景渲染的二维位置和时间梯度。背景排除两车的并集掩码及轮廓边缘。静态 RGB、背景 LiDAR 衰减、寿命、速度和天空项保留；背景逆深度按原生全图面积归一。原前景 NPS 与全图联合 RGB 由背景约束、各自的 RGB 和实测静态前缀项替代。静态前缀权重 0.5，只用 a6/a37 实测及 a6 跨时刻实测回投；训练深度余量 0.30 m，历史配对评价用 0.20 m。没有把固定 PVD 生成点当作训练深度。
- SGCM 及原生优化器保持启用。正式验收的场景和删除状态共用同一静态及环境层；删除只隐藏 a37 动态行。

## 验收

真实 GPU smoke 为 {receipt['smoke']['seconds']:.1f} s、{receipt['smoke']['iteration']} 步，执行保存、真实评价和重新载入后的两步训练。组件、优化器、采样器、SGCM、随机状态恢复审计通过。每百步记录的 CUDA 峰值分配量为 smoke {peak['smoke']:.2f} GiB、正式 {peak['formal']:.2f} GiB。正式训练 {receipt['formal']['seconds']:.1f} s，全部 10,000 步的视角、时间偏移和学习率与并集对照相同。首轮失败工程预检 100 步。第一版 smoke 跑到 {old_r1['iteration']} 步后停止并保存，因为 1K 道路域出现 a37 越界。第二版增加背景负约束后，试跑到 {old_r2['iteration']} 步停止。1K 静态道路单层 MAE 0.0834，旧双动态 0.0393。第三版恢复原生背景 LiDAR 损失归一化。正式步数单独记账。

正式验收前移至 100、500、1,000、3,000、6,000 步。3,000 与 6,000 当步紧接原生透明度重置，另在其后 300 步复查。1K 在同三帧与旧并集监督做配对，全部继续条件通过。可见 a6 静态前缀 {early_old['a6_static_front']['mean']:.6f} → {early_new['a6_static_front']['mean']:.6f}，可见 a6 删后 RGB MAE {early_old['a6_delete_rgb']['mean']:.6f} → {early_new['a6_delete_rgb']['mean']:.6f}，保护区 RGB MAE {early_old['protected_keep_rgb']['mean']:.6f} → {early_new['protected_keep_rgb']['mean']:.6f}。阈值为预先登记的三帧工程门槛，1K 验收同时检查两车串车贡献、严格前后遮挡、RGB、深度和道路质量。

### 10K 同一批 49 帧

| 指标 | 有效像素 | 旧双动态并集 | 新训练 |
| --- | ---: | ---: | ---: |
{entries}

删除后的道路评价使用 {road_new['count']:,} 个同时具备跨时刻 RGB 与路面深度代理的像素。旧双动态 RGB MAE {road_old['mean']:.6f}，本轮 {road_new['mean']:.6f}。49 帧联合 RGB 与原生无标签渲染的逐像素差异均在数值容差内。

本轮在已观测 a6 上同时改善了静态遮挡、实例串扰、删车 RGB 与实测深度。全场景及道路画质退化。raw113 和 raw130 的无编辑图仍有 a37 轮廓溢出，删除图中仍有后车未观测下部的灰块和深色道路缺口。道路代理域中，环境分量约占 {edit_new['pooled']['road_depth_proxy']['mean'][2]:.1%}。这部分是缺失背景读出，不能归入已测 a6 前方的静态遮挡。工程验收通过，整图画质尚未达到旧并集对照。

固定 a6 静态前缀使用与早先 0.519 → 0.165 相同的独立标记测量。重投显露参考来自同一辆车在其他时刻实测的表面，与训练有重叠，属于同场景机制验证。隐藏区没有独立准确率真值。串车权重在联合可见域统计，孤立投影重叠本身不作负样本。独立观测与原生端的解码、有效域和相机一致性检查见审计回执。

[早期继续条件](early_accept_01000.json) · [49 帧对照](union_10000.json) · [49 帧本轮](native_10000.json) · [恢复、采样和检查点回执](audit.json) · [实验入口](source/run_native.sh)。

人工验收字段 `human_verdict=null`。本轮使用原生高斯合成。AutoDL 电源由用户控制。跨版本失败条目沿用既有静态遮挡、串车诊断，`failure_ledger_delta=none`。
"""
    (doc/'README.md').write_text(readme, encoding='utf-8')
    for a, b in ((review/'data/audit.json','audit.json'),
                 (run/'formal/early_accept_01000.json','early_accept_01000.json'),
                 (run/'control/probe_10000/summary.json','union_10000.json'),
                 (run/'formal/probe_10000/summary.json','native_10000.json'),
                 (run/'edit_union/delete_domains.json','edit_union.json'),
                 (run/'edit_native/delete_domains.json','edit_native.json')):
        shutil.copyfile(a, doc/b)
    scripts = ('train_dual.py','native_ownership.py','integrate_native.py','readout_native.py',
               'dual_actor_runtime.py','diag_hooks.py','evaluate_a6_probe.py','evaluate_dual.py',
               'compare_control.py','early_accept.py','build_native_report.py','audit_native.py',
               'road_panel_audit.py','inspect_road.py','inspect_edit.py','prepare_native_publish.py','run_native.sh','finish_native.sh','closeout_native.sh')
    (doc/'source').mkdir()
    for name in scripts:
        shutil.copyfile(run.parent/'source'/name, doc/'source'/name)
    for name in ('effective_config.json', 'diagnostic_contract.json'):
        shutil.copyfile(run/'formal'/name, doc/name)
    index = repo / 'docs/EXPERIMENTS.md'
    original = index.read_text(encoding='utf-8')
    pos = original.index('\n\n') + 2
    entry = f"""## Scene008 原生 10K 实例监督（2026-11-05）

- `scene008_native_ownership_20261105 / run_r3`。[原生训练、早期验收和 49 帧同域对照](autoresearch/{doc_name}/README.md)。

"""
    index.write_text(original[:pos] + entry + original[pos:], encoding='utf-8')
    status = f"""# 当前研究状态

2026-11-05，Scene008 已从原生训练第 1 步切换为双实例监督，完成真实 GPU smoke、1K 继续条件、原生 10K 与 49 帧配对。[训练组件、配对评价与检查点](autoresearch/{doc_name}/README.md)

在固定实测 a6 射线上，静态前方遮挡从旧双动态 {old['a6_static_front']['mean']:.4f} 变为 {new['a6_static_front']['mean']:.4f}。删除后可见 a6 的 RGB MAE 从 {old['a6_delete_rgb']['mean']:.4f} 变为 {new['a6_delete_rgb']['mean']:.4f}。两车串车贡献、实测深度和道路质量见同域对照。

道路保留质量与删除显露区仍有退化，人工验收尚未登记。[上一轮 10K 后接管与 PVD 对照](autoresearch/scene008_instances_20261105/README.md) 单独保存。AutoDL 电源由用户控制。
"""
    (repo/'docs/RESEARCH_STATUS.md').write_text(status, encoding='utf-8')


if __name__ == '__main__':
    generate(*(Path(value) for value in sys.argv[1:]))
