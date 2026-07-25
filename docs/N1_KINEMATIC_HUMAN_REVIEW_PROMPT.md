# N1-KINEMATIC-01 第三次人工盲审完整提示词

## 1. 评测目的与非目标

你要独立判断第三版机器候选是否为真实的 vehicle lane-change / cut-in / converging-branch merge，
并确认 target corridor 上标成 FRONT 与 REAR 的车辆身份、方向、纵向次序和物理 gap 都成立。

本评测**不**评价渲染质量、碰撞安全、反事实可行性、N2 raw evidence、模型训练收益，也不决定任何
传感器下载。机器 kinematic PASS 只表示候选进入人审，绝不等于 TRUE_POSITIVE。

## 2. 盲法与禁止读取的信息

- 只按 `audit/index.html` 或 `audit/REVIEW_CHECKLIST.md` 的盲序 `K3-xxx` 审核；
- 可以读取同一 item 的 panel 与 `evidence/K3-xxx.json`；
- 禁止读取第二次人审文件、`calibration_audit.json`、旧 `N1-EVENT-FULL-01/audit/`、源码中的校准结果，
  或根据旧 reviewer/notes 猜答案；
- 禁止更改 `review_template.jsonl`、panel、evidence 或任何 hash 字段；只编辑
  `audit/review_working.jsonl` 的 verdict/reviewer/notes/failure_codes；
- 不因候选数量、scene 名、机器类型或阈值“应该通过”而给 TRUE。

## 3. 素材范围与颜色

- 每项 panel 顶部最多 5 张原始 CAM_FRONT 2 Hz keyframe，已投影 3D box：
  subject=洋红 `S`，front=绿色 `F`，rear=蓝色 `R`；
- 左下是官方 vector-map centerline 与三辆车的 2 Hz annotation 轨迹；
- 右下是只读运动学与 gap 数值；
- 世界轨迹和 box 来自 nuScenes annotation；10 Hz 插值不参与速度/加速度判定；
- 若角色不在 CAM_FRONT 视野，必须用俯视轨迹与其他时刻核对；证据仍不足则判 `UNCERTAIN`。

## 4. 逐项判定顺序与优先级

按以下顺序，前项失败即 overall=`FALSE_POSITIVE`：

1. **subject identity**：洋红框/轨迹确实对应同一辆目标车，没有把自车、邻车或另一辆并道车认成 subject；
2. **subject maneuver**：subject 在给定时窗内发生相邻同向 lane crossing 或从较不连续的支路收敛到主 corridor。
   主路 lane→connector→lane 正常续接、仅道路转弯、仅 token 边界切换均不是事件；
3. **target corridor**：青色 target 及灰色上下游链连续同向，没有跳到对向、横穿或错误岔路；
4. **front/rear**：绿色 F 位于 target corridor 前方、蓝色 R 位于后方，均为正确身份与同向车辆，
   且至少 2/3 个关键帧保持同一身份和次序；
5. **gap**：panel 中 bumper gap 与俯视几何相容，并落在冻结 `[0.5, 60] m`。

## 5. Overall verdict 定义

- `TRUE_POSITIVE`：上述 1–5 全部成立；
- `FALSE_POSITIVE`：任一项明确失败。必须填至少一个 failure code，并在 notes 指出可见证据；
- `UNCERTAIN`：现有相机+地图+annotation 仍无法确定。不得把“看不清”猜成 TRUE/FALSE，notes 必须写缺什么。

优先 failure codes：
`SUBJECT_IDENTITY_MISMATCH`、`SUBJECT_NO_LATERAL_MANEUVER`、`ROUTE_CONTINUATION`、
`NORMAL_TURN`、`MAP_MATCH_JITTER`、`INTERPOLATION_ONLY`、`WRONG_BRANCH`、
`OPPOSITE_OR_CROSS_TRAFFIC`、`FRONT_INVALID`、`REAR_INVALID`、`GAP_INVALID`、
`IDENTITY_NOT_PERSISTENT`、`INSUFFICIENT_VISUAL_EVIDENCE`、`OTHER`。

边界例：

- target 有两个 incoming，但 subject 沿最顺直的主路 incoming 进入 target：`FALSE_POSITIVE/ROUTE_CONTINUATION`；
- subject 世界轨迹近直，但相邻平行 lane 的距离偏好在多个 2 Hz keyframe 前后稳定翻转：可是真实 lane change，
  需结合框与中心线判断；
- 正常左/右转后进入新 token：`FALSE_POSITIVE/NORMAL_TURN`；
- front/rear 只在单帧出现或身份切换：`FALSE_POSITIVE/IDENTITY_NOT_PERSISTENT`；
- subject 不在前视画面但俯视 annotation 明确：可据俯视判；二者冲突则 `UNCERTAIN` 并说明。

## 6. JSONL 填写格式

逐行保留 `audit_id`、`evidence_sha256`、`panel_sha256`，填写：

```json
{"audit_id":"K3-001","evidence_sha256":"...","panel_sha256":"...",
"subject_maneuver_verdict":"VALID|INVALID|UNCERTAIN",
"target_corridor_verdict":"VALID|INVALID|UNCERTAIN",
"front_relation_verdict":"VALID|INVALID|UNCERTAIN",
"rear_relation_verdict":"VALID|INVALID|UNCERTAIN",
"overall_verdict":"TRUE_POSITIVE|FALSE_POSITIVE|UNCERTAIN",
"failure_codes":["ROUTE_CONTINUATION"],"reviewer":"你的名字","notes":"基于哪些帧和轨迹作出判断"}
```

不得删除、增加或重复 item；`FALSE_POSITIVE` 必须至少一个 failure code；所有项 reviewer/notes 非空。

## 7. 聚合阈值（结果查看前已冻结）

population=12，本次以 SHA256 均匀盲抽/全审 count=12；
parent event_pool SHA256=`4778abadfc44c830f815efd4c52e544bf23e67f975d0b7df44e52e742289d6bf`。

只有同时满足才可建议第三版 N1 人审通过：

- 预注册研究支持 machine gate 全部通过。当前 machine summary：
  `{"candidate_scene_count": 9, "interaction_pass_count": 12, "negative_window_count": 2, "physical_motion_pass_count": 244, "positive_candidate_count": 12, "same_actor_pair_count": 2, "topology_pass_count": 1879, "transition_candidate_count": 8631}`；
  checks：`{"candidate_scenes": true, "negative_windows": false, "positive_candidates": true, "same_actor_pairs": false}`；
- 完整审核数 ≥ `12`；
- TRUE_POSITIVE ≥ `8`，且覆盖 ≥ `4` scenes；
- determinate precision `TP/(TP+FP) ≥ 0.80`；
- Wilson 95% precision lower bound ≥ `0.60`；
- UNCERTAIN fraction ≤ `0.10`。

任何一项失败 → 第三版 N1 `REJECTED`。全部通过也只得到“可请求下一步授权”的资格；本 run 明确
`n2_authorized=false`，不得启动 N2。

## 8. 完成后的精确命令与下一阶段影响

在仓库根目录执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. python scripts/validate_n1_kinematic_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl
```

命令只校验/汇总人工填写，不会启动 N2。然后把 `review_working.jsonl` 路径和汇总输出交给 Codex；
最终 verdict 由用户确认并写入独立 audit adjudication run。无论输出为何，都不得改写本候选 run。
