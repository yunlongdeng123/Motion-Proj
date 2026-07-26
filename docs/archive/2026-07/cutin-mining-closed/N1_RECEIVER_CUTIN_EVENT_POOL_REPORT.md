# N1-EVENT-CUTIN-01 正式结果与第四次人工审核交付

> **运行日期**：2026-07-25 UTC / 2026-07-26 Asia/Singapore
> **Parent 终态**：`AWAITING_HUMAN_REVIEW`
> **当前研究裁决**：第四次人工 verdict 尚未填写，不得提前判定通过或拒绝
> **硬边界**：`n2_authorized=false`，N2 未启动、未解锁

## 1. 结论先行

第四版在 clean commit `f13eb0f` 上完成 official train 的 685-scene scene-disjoint 正式筛选：

```text
8,416 stable transition candidates
  → 1,824 topology candidates
    → 328 subject original-2-Hz body-entry passes
      → 18 complete receiver-centric event passes
        → 6 same-actor receiver-matched negatives / 6 pairs
```

18 个机器候选覆盖 16 scenes，其中 5 个 `parallel_lane_change`、13 个
`receiver_branch_merge`。冻结机器门槛全部通过：

- positive `18 ≥ 8`；
- negative `6 ≥ 4`；
- same-actor pair `6 ≥ 4`；
- candidate scenes `16 ≥ 5`。

因此 parent 只进入 `AWAITING_HUMAN_REVIEW`，并生成 18/18 的完整第四次盲审包。此状态不是 N1
最终通过：人工字段仍全部为空，必须由用户或指定评审者按完整提示词填写；本 run 无论结果如何都固定
`n2_authorized=false`。

## 2. 第四版解决的具体问题

第三次 12 个 bad case 的共同事实是 `SUBJECT_NO_LATERAL_MANEUVER=12`：旧算法证明了地图分支汇合，
却没有证明 subject 车身相对独立目标车流发生横移；所谓 rear 还可能来自 subject 原队列。

第四版没有继续调整第三版 branch angle，而是改成 receiver-centric 事件：

1. map token 只提供候选位置，物理判定只读原始 2 Hz annotation box；
2. subject 在 pre window 的中心位于目标车道带外，post window 的 oriented box 稳定进入带内；
3. pre/post 分别使用 3 个关键帧，稳定时长名义 1 秒；10 Hz 插值不作物理观测；
4. parallel lane change 的 target corridor 排除 source；merge 只枚举 target 的独立 direct incoming；
5. RECEIVER 必须在 pre/post 保持同一最近后车 identity、同向与 `[0.5,40] m` bumper gap；
6. 30-frame same-actor negative 不缩短、不与物理事件窗重叠，并同样要求持续 receiver。

本地参考包 `/root/autodl-tmp/third_party/data_mining/cutin_rules_package` 提供了
“相邻车流、outside→inside、持续至少 1 秒、receiver 视角、同一 identity、排除正常转弯”的设计启发。
没有照搬其中用绝对差近似单调性或 permissive fallback 的部分。外部资料只用于交叉检查场景定义与数据接口：

- [HighD cut-in scenario analysis](https://arxiv.org/abs/2402.08289)；
- [Waymo Open Dataset 官方仓库](https://github.com/waymo-research/waymo-open-dataset)；
- [nuPlan 官方 devkit](https://github.com/motional/nuplan-devkit)。

没有下载或混入外部数据，也没有启动 N2。

## 3. 正式 provenance

| 项目 | 值 |
|---|---|
| Task | `N1-EVENT-CUTIN-01` |
| Run ID | `v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3` |
| Run 根 | `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/` |
| 代码 commit | `f13eb0f1e39b608de1c5e698cd678c2dfd8365a4` |
| `code_dirty` | `false` |
| seed / split | `0` / official `train` |
| calibration / evaluation scenes | 26 / 685；交集 0 |
| 配置 SHA256 | `5b1634e3347c81ca8d6c7a1b6b3d5a737b092732a3ff9b5c79b093fccfd846c5` |
| 数据 fingerprint | `79651f9111e72a5510f5f5444a202cc0d20215ac3319a7f224fd0073202ad7e9` |
| event-pool canonical SHA256 | `850434a349c65e2f8fc9ece98357e3a0a2f94afcd55d544e7648b47e44affe7f` |
| event-pool file SHA256 | `c80f6a8732e9024cccdbbc3d63724855ca0b6b93b21d0c56c5e8b8939f4ba010` |
| artifact-set SHA256 | `02397acf300763ee8940ba1b408d3bd2f3721df50c835ccbfbf2e995a70fdce5` |
| audit immutable-set SHA256 | `5379059a2554b808000eee1b88f416a0e2dfee87d531d26e2b8f645bf9c3da30` |
| blank review SHA256 | `15c4fc52489783383e843788c15e834125b957b232f1536f6269c3e5ba7198ae` |
| prompt SHA256 | `b03267e8b63fa05a7c61c79ba55e0dcdcc518a97efebdba30e85ae92b54391bb` |
| 正式日志 SHA256 | `bdc320a48f61bafb8792c8461e2bf5db160d9573bc24334552c821c74488aeaf` |
| 开始/结束 UTC | `2026-07-25T17:28:51.496960Z` / `2026-07-25T18:03:19.129912Z` |
| 唯一 terminal | `AWAITING_HUMAN_REVIEW` |
| N2 | `n2_authorized=false` |

manifest 的四个顶层 artifact file hash 与 canonical artifact-set 已独立复算；event pool 的 8,416 个
transition records、6 个 negative records 和 canonical pool hash 均重新计算一致。

## 4. Calibration 与 split 隔离

第二次 37 条和第三次 12 条人审合计 49 条、26 scenes，只用于阈值设计与冻结 replay：

| 冻结 calibration gate | 阈值 | 结果 |
|---|---:|---:|
| 第三次 FP rejection | ≥12 | 12/12 |
| 第二次 FP rejection | ≥34 | 35/35 |
| 第二次 TP retention | ≥1 | 1/2 |

全部 26 个已审 scene 从 formal train evaluation 中排除；正式 685 scenes 与 calibration 交集为 0。
这些 replay 数字不是第四次 precision，不得与 18 个未审候选混合汇总。

## 5. 全量筛选结果

### 5.1 Transition 与互斥终因

| 项目 | 数量 |
|---|---:|
| 全部 stable transitions | 8,416 |
| route continuation | 5,627 |
| merge topology | 1,610 |
| lane-change topology | 214 |
| unresolved | 965 |
| topology candidate | 1,824 |
| annotation keyframe 不足 / `UNKNOWN` | 45 |
| subject 车身未完成目标带 outside→inside | 1,451 |
| subject entry 成立但无稳定独立 RECEIVER | 310 |
| 完整 event PASS | 18 |

`subject_entry_pass_count=328`，正好由 310 个 receiver failure 与 18 个完整 PASS 组成。
summary 中的 `receiver_interaction_pass_count=145` 是独立 receiver component 诊断计数，不是顺序漏斗；
最终 positive 必须同时满足 topology、subject 与 receiver 全部条件。

### 5.2 最终候选覆盖

| 维度 | 分布 |
|---|---|
| maneuver mode | 5 parallel lane changes；13 receiver-branch merges |
| candidate scenes | 16 |
| unique positive actors | 18 |
| map location | Boston 11；One North 4；Holland Village 3；Queenstown 0 |
| optional FRONT present | 5/18 |
| `uses_interpolated_physics=true` | 0/18 |
| matched negatives / pairs | 6 / 6 |
| paired mode | 2 lane changes；4 receiver-branch merges |

两场各有 2 个候选：`scene-0208`、`scene-1048`；其余 14 scenes 各 1 个。该分布只描述 coverage，
没有按 scene 或面板内容后验筛选。

### 5.3 机器物理量诊断

| 量 | min | median | max |
|---|---:|---:|---:|
| pre median target lateral distance (m) | 1.976 | 2.377 | 3.457 |
| post median target lateral distance (m) | 0.052 | 0.438 | 0.781 |
| lateral convergence (m) | 1.541 | 2.021 | 2.851 |
| convergence consistency | 0.608 | 0.987 | 1.000 |
| entry transition duration (s) | 0.499 | 1.525 | 2.999 |
| settled duration (s) | 0.998 | 1.000 | 1.050 |
| receiver bumper gap (m) | 5.159 | 26.713 | 34.838 |
| subject longitudinal speed (m/s) | 3.625 | 6.932 | 15.448 |
| receiver projected longitudinal speed (m/s) | -1.122 | 0.262 | 13.425 |

只有 4/18 的相对 closing speed 为正并具有有限 TTC，范围 `1.493–19.314 s`。receiver speed 与 TTC
是审计诊断，不是自动 TP 依据；静止队列、正常转弯、错误 branch 或横穿交通仍必须由人审逐项排除。

## 6. Machine gate

| 冻结检查 | 结果 | 阈值 | 通过 |
|---|---:|---:|---|
| positive candidates | 18 | ≥8 | 是 |
| candidate scenes | 16 | ≥5 | 是 |
| same-actor negative windows | 6 | ≥4 | 是 |
| same-actor pairs | 6 | ≥4 | 是 |

`machine_gate_passed=true` 只证明样本量与机器规则支持充足，不证明人类语义真实性。

## 7. 两个保留的正式工程失败

两个失败目录均保持独立 `FAILED`，不与成功 run 合并：

1. `...T170948229629Z__46186120`，commit `f5c9bbe`：负对照第一次被正式触发时发现
   `kinematics_control.min_median_speed_mps` 缺项；失败发生在研究输出前。修复提交 `8581d4d`
   加入完整启动前配置契约和异常自动落盘；
2. `...T171746938858Z__5b1634e3`，commit `8581d4d`：处理 96/685 scenes 后收到 SIGKILL。
   583 MB `sample_annotation.json` 的 cgroup file cache 是主要可控压力源；修复提交 `f13eb0f`
   加入 `POSIX_FADV_DONTNEED`、直接 file-handle JSON parsing、批后 heap trim 与内存日志。

成功 run 从新 ID 完整重跑 685 scenes；没有续跑、覆盖或把工程失败记为 research reject。完整证据与防重复
规则见 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 的 `N1-F16`、`N1-F17`。

## 8. 第四次人工审核包

审核根目录：

`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/audit/`

| 文件 | 用途 |
|---|---|
| `index.html` | 18 项连续盲序浏览入口 |
| `REVIEW_CHECKLIST.md` | Markdown 清单与逐项 panel |
| `HUMAN_REVIEW_PROMPT.md` | 完整目的、盲法、rubric、阈值和命令 |
| `review_template.jsonl` | 不可修改的空白模板基准 |
| `review_working.jsonl` | 用户唯一应编辑的文件 |
| `evidence/K4-001.json` … `K4-018.json` | 逐事件只读证据 |
| `panels/K4-001.png` … `K4-018.png` | 5 帧 CAM_FRONT + topdown + metrics |
| `topdown/K4-001.png` … `K4-018.png` | 原始 2 Hz 角色轨迹与 vector map |
| `audit_manifest.json` | 58 个 immutable files 的逐文件 hash |

交付 QA：

- population=18、audit items=18，全量审核，没有 top-k 丢弃；
- 58 个 immutable file hash 与 immutable-set 全部复算一致；
- `review_working.jsonl` 与模板逐字节相同，18 行 verdict/reviewer 为空；
- 空白 review 运行 validator 在完成全部不可变性检查后按预期 fail closed：
  `ValueError: component verdict 非法或缺失: K4-001`；
- 18 个 panel 和 18 个 topdown PNG 均通过解码；panel 尺寸统一为 `1600×780`；
- 首、中、尾 `K4-001/K4-009/K4-018` 已目检布局、原始相机帧、角色颜色、topdown 与 metrics；
  该检查只验证材料可读性，没有填写或推断人工 verdict。

仓库中的逐字节同版提示词：
[`N1_RECEIVER_CUTIN_HUMAN_REVIEW_PROMPT.md`](N1_RECEIVER_CUTIN_HUMAN_REVIEW_PROMPT.md)。

## 9. 人工门槛与完成命令

只有以下条件全部满足，第四版 N1 才可建议通过：

- 完整审核数 ≥8；
- TP≥6，且 TP 覆盖 ≥4 scenes；
- determinate precision ≥0.80；
- Wilson 95% precision lower bound ≥0.50；
- UNCERTAIN fraction ≤0.10；
- machine gate 保持全部通过。

任一失败即第四版 N1 `REJECTED`。即使全部通过，也只能请求用户对下一阶段另行授权，不能自动进入 N2。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/audit/review_working.jsonl
```

validator 只校验和汇总，不改写 parent，也不启动 N2。完成后应把 review 路径和汇总输出交回 Codex，
再由用户确认并创建独立 adjudication run。
