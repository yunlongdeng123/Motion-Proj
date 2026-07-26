# N1-EVENT-CUTIN-01 第四次人工盲审完整提示词

## 1. 评测目的与非目标

你要独立判断第四版候选是否为真实的 receiver-centric vehicle cut-in：洋红 SUBJECT
原先稳定处在蓝色 RECEIVER 所在目标车流之外，随后车身横向进入该车流并在 RECEIVER
前方稳定；RECEIVER 必须在进入前后都是目标 corridor 上最近、同向、身份连续的后车。
绿色 FRONT 只是可选的上下文，不是 TRUE_POSITIVE 的必要条件。

本评测不评价渲染质量、碰撞安全、反事实编辑、N2 raw evidence、训练收益或数据下载。
机器 PASS 只表示候选满足冻结的 2 Hz 几何规则，不等于人工 TRUE_POSITIVE。

## 2. 盲法与禁止读取的信息

- 只按 `audit/index.html` 或 `audit/REVIEW_CHECKLIST.md` 的盲序 `K4-xxx` 审核；
- 可以读取同一 item 的 panel 与 `evidence/K4-xxx.json`；
- 禁止读取第二/第三次人审文件、`calibration_audit.json`、旧审核 notes、源码中的
  calibration 结果，或按 scene/event 名猜答案；
- 禁止更改 `review_template.jsonl`、panel、evidence 或 hash 字段；只编辑
  `audit/review_working.jsonl` 的 verdict、failure_codes、reviewer 和 notes；
- 不因候选数量、机器 gate、gap/TTC 数字或“第四版应该更准”而给 TRUE。

## 3. 素材范围与颜色

- panel 顶部最多 5 张原始 CAM_FRONT 2 Hz keyframe，3D box 颜色为：
  SUBJECT=洋红 `S`，RECEIVER=蓝色 `R`，可选 FRONT=绿色 `F`；
- 左下是官方 vector-map centerline 与各角色的原始 2 Hz annotation 轨迹；
- 右下报告车身 outside/inside、横向收敛、1 秒稳定、receiver 身份支持、bumper gap 与 TTC；
- 10 Hz 插值只用于候选 token 对齐和显示，不是独立物理观测；
- 某角色不在 CAM_FRONT 中时，使用俯视轨迹、box 和其他关键帧；仍不足则判
  `UNCERTAIN/INSUFFICIENT_VISUAL_EVIDENCE`，不得猜测。

## 4. 逐项判定顺序与优先级

按以下顺序审核；任一项明确失败，overall 必须为 `FALSE_POSITIVE`：

1. **subject maneuver**：洋红身份连续，车身确实从目标车道带外横向进入其内并稳定。
   主路正常续接、道路自身弯曲、仅 token 切换、地图匹配抖动均不是 cut-in；
2. **receiver corridor**：蓝色 RECEIVER 的道路分支/车道才是被切入的目标车流，
   且与 SUBJECT 的 source stream 在进入前相互独立。不能把 SUBJECT 后方同一队列车辆
   重新命名为 RECEIVER，也不能跨到平行岔路、对向或横穿车流；
3. **receiver relation**：SUBJECT 进入后位于 RECEIVER 前方，蓝车是目标 corridor
   上最近的同向后车，bumper gap 与俯视几何相容并在冻结 `[0.5, 40] m`；
4. **temporal persistence / path clear**：同一 RECEIVER 在进入前至少 2 个、进入后至少
   2 个原始 2 Hz keyframe 保持身份与次序；二者之间没有被忽略的同车道车辆；
5. **可选 FRONT**：绿色 F 只辅助理解。FRONT 错误应写入 notes，但若 1–4 均成立，
   不单独把真实 receiver-centric cut-in 判为 FP。

## 5. Overall verdict 与 failure code

- `TRUE_POSITIVE`：第 4 节 1–4 全部 `VALID`，failure_codes 必须为空；
- `FALSE_POSITIVE`：任一必需项明确 `INVALID`；至少填一个 failure code；
- `UNCERTAIN`：没有必需项可明确判 INVALID，但证据不足；至少一个 component 为
  `UNCERTAIN`，notes 写明缺失证据。

允许的 failure codes：
`SUBJECT_IDENTITY_MISMATCH`、`SUBJECT_NO_LATERAL_MANEUVER`、`ROUTE_CONTINUATION`、
`NORMAL_TURN`、`MAP_MATCH_JITTER`、`INTERPOLATION_ONLY`、`WRONG_BRANCH`、
`OPPOSITE_OR_CROSS_TRAFFIC`、`RECEIVER_INVALID`、`RECEIVER_ON_SOURCE_STREAM`、
`GAP_INVALID`、`PATH_NOT_CLEAR`、`IDENTITY_NOT_PERSISTENT`、
`INSUFFICIENT_VISUAL_EVIDENCE`、`OTHER`。

边界例：

- SUBJECT 与蓝车始终在同一条弯道排队，仅经过 lane/connector 边界：
  `FALSE_POSITIVE/ROUTE_CONTINUATION/RECEIVER_ON_SOURCE_STREAM`；
- SUBJECT 在相邻平行车道，车身跨入蓝车车道并稳定，蓝车持续跟在其后：
  可判 TRUE，即使没有绿色 FRONT；
- SUBJECT 正常左/右转进入新道路：`FALSE_POSITIVE/NORMAL_TURN`；
- 蓝车位于另一平行岔路或横向路口：`FALSE_POSITIVE/WRONG_BRANCH/RECEIVER_INVALID`；
- 蓝车身份前后切换，或中间有另一辆目标车道车辆：分别为
  `IDENTITY_NOT_PERSISTENT` 或 `PATH_NOT_CLEAR`；
- 相机看不清但俯视 annotation 明确可判；相机与 annotation 冲突则 `UNCERTAIN`。

## 6. JSONL 填写格式

逐行保留 `audit_id`、`evidence_sha256`、`panel_sha256`，只填写：

```json
{"audit_id":"K4-001","evidence_sha256":"...","panel_sha256":"...",
"subject_maneuver_verdict":"VALID|INVALID|UNCERTAIN",
"receiver_corridor_verdict":"VALID|INVALID|UNCERTAIN",
"receiver_relation_verdict":"VALID|INVALID|UNCERTAIN",
"temporal_persistence_verdict":"VALID|INVALID|UNCERTAIN",
"overall_verdict":"TRUE_POSITIVE|FALSE_POSITIVE|UNCERTAIN",
"failure_codes":[],"reviewer":"你的名字","notes":"基于哪些关键帧、轨迹和角色关系作出判断"}
```

不得删除、增加、重排或重复 item。`FALSE_POSITIVE` 必须至少一个 failure code；
所有 item 的 reviewer/notes 必须非空。

## 7. 聚合阈值（查看第四次结果前冻结）

population=18，本次按 SHA256 确定性盲序审核 count=18；
parent event_pool SHA256=`850434a349c65e2f8fc9ece98357e3a0a2f94afcd55d544e7648b47e44affe7f`。

只有以下条件全部满足，才可建议第四版 N1 通过：

- machine gate 全部通过。machine summary：
  `{"candidate_scene_count": 16, "mode_counts": {"parallel_lane_change": 5, "receiver_branch_merge": 13}, "negative_window_count": 6, "positive_candidate_count": 18, "receiver_interaction_pass_count": 145, "same_actor_pair_count": 6, "subject_entry_pass_count": 328, "topology_pass_count": 1824, "transition_candidate_count": 8416}`；
  checks：`{"candidate_scenes": true, "negative_windows": true, "positive_candidates": true, "same_actor_pairs": true}`；
- 完整审核数 ≥ `8`；
- TRUE_POSITIVE ≥ `6` 且覆盖 ≥
  `4` scenes；
- determinate precision `TP/(TP+FP) ≥ 0.80`；
- Wilson 95% precision lower bound ≥ `0.50`；
- UNCERTAIN fraction ≤ `0.10`。

任一条件失败，第四版 N1 为 `REJECTED`。即使全部通过，也只获得请求下一阶段授权的
资格；本 run 固定 `n2_authorized=false`，不得自动启动 N2。

## 8. 完成后的精确命令与影响

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/audit/review_working.jsonl
```

该命令只校验和聚合人工填写，不启动 N2。完成后把 `review_working.jsonl` 路径和
汇总输出交给 Codex；最终 verdict 仍由用户确认并写入独立 adjudication run。
不得改写本候选 run 或用人工 verdict 覆盖失败的 machine gate。
