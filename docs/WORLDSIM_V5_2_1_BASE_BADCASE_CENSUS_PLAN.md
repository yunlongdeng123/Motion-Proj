# WorldSim V5.2.1 — Base Badcase Census / Failure Localization / M1-M2-M3 Re-audit Plan

> **用途**：直接交给 Codex Agent 在 `/root/autodl-tmp/motion_proj` 连续执行。
> **环境**：默认单卡 NVIDIA RTX 3090 24GB。
> **性质**：V5.2.1 是 **审计、基座 badcase 建库、故障定位、研究问题复核阶段**，不是算法改进阶段。
> **硬边界**：不继续 Stage H/BKI；不设计或实现 TrackBayes/TubeBayes；不修改 M1/M2/M3 算法；不读取 V5.2 fresh validation/test/KITTI quality。
> **计划状态**：`ready_for_execution`；2026-08-20 审阅版。执行授权以 `docs/RESEARCH_STATUS.md` 顶部 V5.2.1 条目为准。
> **执行提示**：`fc07b99...` 只是 V5.1 closeout 的 authoring base，不是要求 reset 的执行 HEAD；P0 必须冻结实际 HEAD，禁止回退用户历史。

---

# 0. 总目标

V5.2.1 不回答“下一版算法具体怎么改”，只回答下面五个问题：

1. **AD-GS / StreetGS 基座到底在哪里失败？** 失败必须落到 scene / frame / camera / actor / temporal window，而不能停留在 scene-level PSNR。
2. **失败属于什么物理类型？** 区分全局外观、动态目标、边界混合、深度/几何、遮挡/去遮挡、时序不一致、极稀疏观测等。
3. **M1 的 ownership 问题是不是基座真实瓶颈？** 区分“基座本身就重建错”与“基座 RGB 尚可但 ownership/identity 错”。
4. **M2/M3 在真实 badcase 上分别解决什么问题？** 重新判断它们与基座失败的因果关系、覆盖范围和论文价值，而不是沿用历史模块命名继续开发。
5. 形成一套以后所有 V5.2 方法都必须对齐的 **冻结 badcase 基准集与 failure registry**。

V5.2.1 结束时应能够用证据明确回答：

```text
Base reconstruction failure
        ↓
具体 failure taxonomy + failure IDs
        ↓
哪些是 M1 ownership/association 可解决的
哪些是 M2 repair 才能解决的
哪些是 M3 temporal consistency 才能解决的
哪些根本不是当前 WorldSim paper 应该解决的
        ↓
V5.2 下一阶段只针对被证据证明的重要 failure class 设计方法
```

**在这个闭环完成之前，不允许进入 V5.2 算法开发。**

---

# 1. 必须继承的当前事实

Codex 开始执行时先核对远端真实 HEAD；下面是 V5.2.1 的研究前提，不允许被旧文档或局部 run 倒写。

## 1.1 V5.1 M1 终态

仓库快照中的 V5.1 closeout 已冻结：

- authoring base commit：`fc07b9912db4df4a73b44d7a2363d9b3c6c88bc2`；
- V5.1 terminal：`closed_without_promoted_candidate`；
- `U2/B3`：保留为 V5.2 matched comparator；
- Stage H：`pending`、`executed=false`、`superseded_by_v5.2_scope`；
- fresh V5.1/V5.2 validation/test/KITTI quality：未授权作为方法选择数据；
- M2/M3：V5.2 当前均保持 `pending`。

V5.1 已确认主要瓶颈是 **有效 observation 的结构性缺失**，不是空间 propagation kernel 不够复杂。

## 1.2 V5.1 已关闭路线

以下路线不得在 V5.2.1 复活：

- raw Gaussian KNN；
- graph Laplacian / graph diffusion；
- DINO semantic graph；
- SAI3D progressive growing；
- super-primitive spatial propagation；
- anchor graph；
- BKI spatial completion；
- Gaussian Grouping 当前 faithful adapter；
- Trace3D 当前 faithful operator。

V5.2.1 只允许 **读取和重新汇总历史证据**，不允许改这些方法重新跑 parameter search。

## 1.3 已知关键历史 badcase

M1 侧至少保留以下 historical failure evidence：

- `scene-1087`：有效 observation 极端稀疏；V5.1 中 accepted view 极少，空间传播无法凭空补足真实证据；
- `scene-0379`：局部指标可能改善，但 FN semantic mass 明显增加；
- V4 historical validation `scene-0071 / 0317 / 0450`：Boundary F1 均恶化；
- `scene-0317`：historical boundary-support target recall 约 45%；
- V4 r200 存在逐 frame / camera 的 `IoU=0 / BF1=0 / FN=1` 极端视角。

这些是 **historical evidence**，不是 V5.2.1 新调参集。

## 1.4 AD-GS 基座历史事实

已有历史冻结事实：

- 六场 60k reproduction：`6/6`；
- 每场已有 60k checkpoint、42 test renders、138 train renders 的历史记录；
- official test arithmetic mean：
  - PSNR `31.174515`
  - SSIM `0.927661`
  - LPIPS(VGG) `0.163489`
- worst scene：
  - PSNR：`scene-0295 = 29.355150`
  - SSIM：`scene-0230 = 0.905364`
  - LPIPS：`scene-0230 = 0.212178`

历史六场：

```text
scene-0230
scene-0242
scene-0255
scene-0295
scene-0518
scene-0749
```

注意：上述 scene-level 数值只能用于 **候选定位**，不能替代 V5.2.1 的 frame/camera/actor badcase census。

上述“完整”是历史冻结结论，不等于 2026-08-20 当前磁盘仍 resident。最近只读审计在既有 AD-GS 六场 run/third-party
路径中没有找到 `iteration_60000/point_cloud.ply` 或 official PNG/JPG render；`model_60000` 目录虽存在，已抽查目录只见
轻量配置/事件/指标。因此 P1 必须把当前状态从历史完成状态中重新审计出来，默认起点是
`MISSING_BUT_MANIFESTED` 候选，而不是 `PRESENT_EXACT`。不得据此把 AD-GS 方法写成失败，也不得跳过受保护路径/归档来源核对。

## 1.5 StreetGS / V4 基座基础设施

当前仓库已经有可复用基础设施：

- `configs/worldsim_v4/baseline_matrix_v1.yaml`
- `configs/worldsim_v4/metrics_v1.yaml`
- `motion_proj/worldsim_v4/baseline_scene_evaluator.py`
- `motion_proj/worldsim_v4/evaluator.py`
- `motion_proj/worldsim_v4/region_masks.py`
- `scripts/run_worldsim_v4_baselines.py`
- `scripts/run_worldsim_v4_streetgs_scene.py`
- `scripts/run_worldsim_v4_adgs_scene.py`

StreetGS strict mod-5 development six场 checkpoint 在 freeze 中已有内容寻址记录：

```text
scene-0230
scene-0242
scene-0255
scene-0048
scene-0994
scene-0139
```

旧 stride=10 六场 run 仅保留 provenance，**不得重新当 matched baseline**。

已有 V4 baseline evaluator 的冻结图像指标：

- PSNR
- SSIM
- LPIPS-Alex
- global
- static
- actor
- boundary
- edit_roi

V5.2.1 应优先扩展这些 evaluator，而不是另起一套互不兼容的评测代码。

## 1.6 Failure ledger 绑定与本阶段非重复性

V5.2.1 的所有正式 config/run metadata 至少登记：

```text
failure_ledger_refs:
  - V1-F01
  - V1-F03
  - V4-F34
  - V4-F39
  - V4-F42
  - V4-F45
  - V4-F47
  - V4-F49
  - V5-F07
  - V5-F31
  - V5-F32
  - V5-F33
  - V5-F47
  - V5-F48
  - V5-F51
  - V5-F52
  - V5-F57
  - V5-F59
  - V51-F31
  - V51-F37
  - V51-F42
  - V51-F63
  - V51-F65
  - V51-F66
```

其中，`V1-F01/F03` 限制“复现成功”和 persistent identity 的 claim；`V4-F34/F39/F42/F45/F47/F49` 约束 split、
M1 validation、M2 geometry 与 M3 denominator；`V5-F07/F31-F33/F47/F48/F51/F52/F57/F59` 约束 V5 M1/M2/M3 的已推翻假设；
`V51-F31/F37/F42/F63/F65/F66` 约束 observation-source、identity persistence 与 determinism。

本阶段不是重复调参或重跑已拒绝路线：它不改变任何方法，只建立之前缺失的 frame/camera/actor/window 级基座失败分母，
并检验 ownership failure 是否独立于 base RGB/geometry failure。`BC-*`/`BCE-*` 是 census 个案/事件 ID，不是
`RESEARCH_FAILURES.md` 的 ledger ID；只有出现可复用的工程坑、协议失效、被推翻假设或正式 reject 时才新增统一 failure 条目，
不得把每个坏帧逐条灌入 failure ledger。

---

# 2. V5.2.1 明确不做什么

以下全部禁止：

```text
× 不训练新的 ownership 网络
× 不改变 U2/B3 posterior
× 不新增 Bayesian evidence term
× 不做 actor tube / TrackBayes / TubeBayes
× 不接 DeSiRe / DIAL / IDSplat / UnIRe / scene flow teacher
× 不修改 M2 router / repair 算法
× 不修改 M3 temporal delta / B-spline / constraint projection
× 不继续 Stage H / BKI
× 不调 SAM threshold / graph threshold / posterior threshold
× 不重开 V5.1 rejected arms
× 不拿 fresh validation/test 找 badcase
× 不为了“有漂亮图”事后挑 frame
× 不把不同 LPIPS backbone 的历史数值直接做横向排名
```

允许做的只有：

```text
√ 资产恢复与哈希审计
√ frozen checkpoint 的只读重渲染
√ evaluation-only GT / annotation / LiDAR 投影
√ 新增诊断指标
√ 自动 badcase 排名
√ 可视化面板
√ 历史 M1/M2/M3 证据重新聚合
√ 相关性、分桶、failure localization
√ 新增纯 diagnostics / audit / registry 代码与测试
```

---

# 3. 数据纪律：V5.2.1 必须重新冻结三个数据层级

不要把“历史已经看过的数据”和“V5.2 fresh data”混为一谈。

## Tier H — Historical Evidence

只读使用已经产生并在历史阶段看过的结果，例如：

- AD-GS 六场历史 official metrics/render；
- V4 M1 r200/r201；
- V4 M2 r212/r222；
- V4 M3 r238/r335；
- V5/V5.1 development / diagnostic 结果；
- `0471/1087/0379` 等历史 M1 failure scenes。

用途：形成初始 taxonomy、恢复资产、验证新 evaluator 与旧统计方向一致。

**不得把 Tier H 冒充 V5.2 新确认结果。**

## Tier D — V5.2.1 Active Development Census

V5.2.1 真正允许 inspect / diagnose / 形成研究假设的集合。

优先使用已经冻结为 development 的 frame/camera，不改变训练 split：

- StreetGS strict mod-5 development：沿用 `sample_index mod 5 == 2`；
- AD-GS 若能够从 exact frozen checkpoint 对同一 development frame/camera 重渲染，则加入；
- 若某 AD-GS 历史 checkpoint 的训练 split 与该 frame 不满足 no-leakage，则该 frame 只能留在 historical native census，不能进入 matched Tier D。

## Tier C — V5.2.1 Internal Confirmation

为了避免“看完 badcase 后再定义 failure”，从 Tier D **内部再次做 deterministic hash split**：

```text
key = scene | frame_or_sample | camera
sha256(key) % 5 == 0  -> confirmation
otherwise             -> discovery
```

约 20% 为 confirmation。

规则：

1. P0 冻结分区算法、候选全集来源与 quality-read lock；P1 在 exact asset/split 审计后、任何 Tier D quality read 前冻结具体 membership；
2. 在 P3 taxonomy freeze 以前禁止生成/查看 Tier C 图像 panel；
3. 可以提前做完整性与 hash 检查，但禁止依据 Tier C quality 改 failure taxonomy；
4. P9 才一次性读取 Tier C 指标并确认 taxonomy 是否成立。

分区单位不是单个 baseline 或单个 camera，而是共享的 canonical sample：

```text
unit_key = dataset | scene | sample_token_or_canonical_sample_index
digest = sha256(UTF-8(unit_key)).hexdigest()
bucket = int(digest, 16) % 5
bucket == 0  -> confirmation
otherwise    -> discovery
```

- `base`、`camera`、`actor` 不进入 hash；同一 sample 的两个基座、全部相机和 actor 必须落在同一 partition，避免 matched comparison 跨分区；
- 没有可证明 canonical sample mapping 的 row 不得进入 matched Tier D，只能进入带明确协议标签的 native historical census；
- temporal window 的全部成员必须属于同一 partition；跨分区 window 返回 `undefined_cross_partition`，不得拆帧或重分配；
- Tier C 只是 development 内部、同 scene 邻近样本仍可能相关的 sanity confirmation，不得写成 scene-disjoint 或正式 heldout 证据。

**Tier C 仍然来自 development，不是全局 heldout remainder=4，更不是 V5.2 fresh validation/test。**

## 永久锁定

V5.2.1 全程不得读取：

```text
V5.2 fresh validation quality
V5.2 fresh test quality
KITTI method-tuning quality
任何尚未授权的新 cross-domain confirmation data
```

---

# 4. 输出目录与代码边界

新代码统一放在：

```text
motion_proj/worldsim_v521/
```

建议只新增诊断模块：

```text
motion_proj/worldsim_v521/
├── asset_audit.py
├── census_schema.py
├── census_evaluator.py
├── temporal_diagnostics.py
├── lidar_diagnostics.py
├── badcase_ranker.py
├── badcase_registry.py
├── panel_builder.py
└── m123_reaudit.py
```

CLI 放在：

```text
scripts/audit_worldsim_v521_assets.py
scripts/run_worldsim_v521_census.py
scripts/build_worldsim_v521_badcase_registry.py
scripts/render_worldsim_v521_badcase_panels.py
scripts/run_worldsim_v521_m123_reaudit.py
scripts/finalize_worldsim_v521.py
```

配置：

```text
configs/worldsim_v521/p0_scope_freeze_v1.yaml
configs/worldsim_v521/base_asset_registry_v1.yaml
configs/worldsim_v521/census_protocol_v1.yaml
configs/worldsim_v521/badcase_taxonomy_v1.yaml
configs/worldsim_v521/m123_reaudit_v1.yaml
configs/worldsim_v521/closeout_v1.yaml
```

正式 run 根目录：

```text
/root/autodl-tmp/runs/worldsim_v521/
```

文档最终至少生成：

```text
docs/WORLDSIM_V5_2_1_BADCASE_REPORT.md
docs/WORLDSIM_V5_2_1_M123_REVIEW.md
docs/WORLDSIM_V5_2_1_CLOSEOUT.md
```

失败仍统一追加：

```text
docs/RESEARCH_FAILURES.md
```

不得新建平行 failure ledger。

## 4.1 正式 run 与不可变计划合同

- P0 开始时记录本计划的 bytes/SHA-256；此后本文件视为 immutable。需要纠错时新增带 hash-chain 的 authorization overlay，
  不直接改已绑定计划，也不只改 expected hash 掩盖漂移；
- 每个 P0–P10 task 使用不可复用的 run ID，建议格式
  `YYYYMMDDThhmmssZ__<stage>-<slug>-s<seed>-r<seq>`，目标目录由 runner 以 `.partial → atomic rename` 发布；
- 每个正式 run 至少保存 `resolved_config.yaml`、`status.json`、`events.jsonl`、`run_manifest.json`、
  `input_fingerprint.json`、`summary.json`、资源遥测和阶段要求的 JSON/JSONL；
- run metadata 必须有 `failure_ledger_refs`，收口必须有 `failure_ledger_delta`；没有新增 failure 时明确写 `none`；
- canonical artifact 只由 closeout config 的 exact path/bytes/SHA 绑定，不使用可漂移的 `latest` 目录或仅凭文件名发现结果；
- 大型指标、panel 和 registry 留在 canonical run；`docs/` 只保存报告、决策和可追溯入口，不复制大载荷。

---

# 5. P0 — Repo / GPU / Disk / Protocol Freeze

**Task ID**：`WS-V521-P0-BASE-CENSUS-FREEZE-01`

这是第一步，必须先完成再读取新的 Tier D quality。

## P0.1 真实仓库状态

在远端执行并保存：

```bash
cd /root/autodl-tmp/motion_proj
git status --short
git rev-parse HEAD
git branch --show-current
git log -1 --decorate --oneline
```

要求：

- 记录真实 HEAD，不假定 zip 中有 `.git`；
- 若 worktree dirty，保存 diff/fingerprint，判断是否为用户已有改动；
- 不覆盖、不 reset 用户已有改动；
- V5.2.1 新工作从可审计的当前 HEAD 继续；不得 reset 到 `fc07b99...`；
- 计划定稿提交后，从该提交创建或切换到 `research/worldsim-v5.2.1-base-badcase-census`，若同名分支已存在则先审计其
  upstream/HEAD/dirty 状态，不覆盖、不强推。

## P0.2 再确认 V5.1 terminal lock

程序化检查：

- `docs/RESEARCH_STATUS.md`
- `docs/RESEARCH_FAILURES.md`
- `docs/EXPERIMENTS.md`
- `configs/worldsim_v51/m1_closeout_v1.yaml`
- `docs/archive/2026-08/worldsim-v51-m1-closeout/README.md`
- Stage G r047 audit/freeze

必须在 P0 summary 中明确：

```text
Stage H executed = false
Stage H disposition = superseded_by_v5.2_scope
U2/B3 = retained comparator
M2 = pending
M3 = pending
fresh validation read = false
fresh test read = false
```

任何一项不一致都先 fail-closed 审计原因，不能直接开实验。

## P0.3 3090 与系统资源

采集：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,pstate --format=csv
df -h /root/autodl-tmp
free -h
```

资源规则：

- 单 GPU 串行；
- 不允许 AD-GS / StreetGS / LPIPS 多个大进程并发驻卡；
- census 第一遍采用 streaming metrics，**不保存全量 render PNG**；
- 只对最终 frozen badcase 保存 panel，避免再次制造几十 GB 可再生资产；
- 不删除 V4/V5/V5.1 canonical evidence、checkpoint 或 third-party source 来腾空间。

如果 GPU 暂时被别的任务占用：

- 先执行 CPU-only asset/hash/protocol census；
- 不修改算法以适配低显存；
- GPU 可用后继续 render 阶段。

## P0.4 冻结 internal confirmation 协议

P0 只冻结分区公式、canonical sample key schema、候选 cohort 来源、quality-read lock 与预期输出 schema，写入：

```text
configs/worldsim_v521/p0_scope_freeze_v1.yaml
```

具体 membership 依赖 P1 的 exact asset/split 审计，在 P1.4 才生成。P0 不得为了提前产出 membership 假定缺失资产存在。

P1.4 最终文件每行字段至少包含：

```json
{
  "dataset": "nuscenes",
  "scene": "...",
  "sample_token": "...",
  "canonical_sample_index": 0,
  "unit_key": "nuscenes|scene-xxxx|...",
  "partition": "discovery|confirmation",
  "split_hash": "...",
  "eligible_bases": ["streetgs", "adgs"]
}
```

### P0 Gate

只有以下全部满足才进入 P1：

```text
[PASS] repo provenance frozen
[PASS] Stage H remains untouched
[PASS] fresh validation/test locks explicit
[PASS] GPU/disk/cgroup recorded
[PASS] partition algorithm/key schema frozen before quality read
```

---

# 6. P1 — Base Asset Census：先找资产，不先重训

**Task ID**：`WS-V521-P1-BASE-ASSET-CENSUS-01`

目标：回答“到底有哪些 exact checkpoint/render/GT/processed inputs 可以直接用于 badcase census”。

## P1.1 AD-GS 资产查找优先级

必须按历史 manifest / freeze / run 路径做 deterministic search，而不是用 `find /` 盲扫后凭文件名认 checkpoint。

优先核对历史：

```text
/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/
/root/autodl-tmp/runs/dynamic_editing_v2/
/root/autodl-tmp/third_party/AD-GS
/root/autodl-tmp/data/
```

并从历史文档恢复每场：

- expected checkpoint path / bytes / SHA256；
- expected render count；
- expected metrics；
- processed input root；
- AD-GS source commit / compatibility patch hash；
- DPT / CoTracker frozen weight hash。

每个 scene 只能进入以下状态之一：

```text
PRESENT_EXACT
PRESENT_HASH_MISMATCH
MISSING_BUT_MANIFESTED
MISSING_UNRECOVERABLE
PROTOCOL_MISMATCH
```

**V5.2.1 默认禁止重新训练 AD-GS 60k。**

如果 checkpoint 缺失：

1. 先检查历史受保护路径、hardlink、archive、run manifest；
2. 记录缺失资产；
3. 继续其他 scene / StreetGS census；
4. 不为了凑 coverage 临时再训一个“近似 checkpoint”并冒充原基座。

## P1.2 StreetGS exact checkpoint audit

核对 `configs/worldsim_v4/baseline_matrix_v1.yaml` 中 strict mod-5 checkpoints：

```text
scene-0230
scene-0242
scene-0255
scene-0048
scene-0994
scene-0139
```

要求：

- path exists；
- bytes exact；
- SHA256 exact；
- DriveStudio source commit exact；
- renderer/model native resolution exact `800×450`；
- strict train remainders `[0,1,3]`；
- development remainder `2` 未进入训练；
- old stride=10 runs 永久标注 `PROTOCOL_MISMATCH`。

## P1.3 建立三类 census cohort

### A. Native Historical Census

用于发现“这个基座自己的失败类型”。允许不同基座各用自己的历史冻结协议，但 **禁止跨基座直接宣称谁更强**。

AD-GS：优先六场历史 exact reproduction。
StreetGS：优先 strict mod-5 六场。

### B. Matched Development Census

只保留：

```text
same scene
same frame/sample
same camera
same GT
same 800×450 metric resolution
both checkpoints no leakage
```

当前至少检查共同 scene：

```text
0230 / 0242 / 0255
```

但是否真正进入 matched cohort 必须由 checkpoint/split 审计决定，不能仅因为 scene name 相同就认定 matched。

### C. Historical Semantic Diagnostic Cohort

只读复用 M1 historical evidence，例如：

```text
0471 / 1087 / 0379
0071 / 0317 / 0450 (historical V4 evidence only)
```

它用于 M1 re-audit，不与 AD-GS/StreetGS RGB baseline ranking 强行拼成一个统计分母。

## P1.4 在 quality-blind 状态冻结具体 Discovery/Confirmation membership

完成 checkpoint、split、frame mapping、target path/hash 的 metadata-only 审计后，枚举 Tier D canonical sample 全集并按
§3 的共享 sample-level hash 规则生成：

```text
DISCOVERY_CONFIRMATION_FREEZE.json
```

生成过程只允许读取路径、schema、时间戳、token、bytes/hash、训练 split 和相机标识，不 decode target/prediction、
不计算 quality。输出必须同时记录：候选总数、每 partition 的 scene/sample/camera 预期分母、被 asset/split/mapping 阻塞的
样本及原因。冻结后任何 base 恢复都只能填补既有 canonical sample 的可用性，不得根据 quality 新增/删除 membership。

## P1 输出

生成：

```text
BASE_ASSET_REGISTRY.json
BASE_ASSET_REGISTRY.md
MATCHED_FRAME_REGISTRY.jsonl
ASSET_BLOCKERS.jsonl
DISCOVERY_CONFIRMATION_FREEZE.json
```

### P1 Gate

不要求 AD-GS/StreetGS 都 6/6 才继续。

只要有至少一个 exact baseline 可以产生 Tier D census，就继续；缺失 baseline 以 blocked denominator 保留。

**不要因为单个资产缺失暂停整个 V5.2.1。**

进入 P2 前还必须满足：具体 Discovery/Confirmation membership 已在任何 Tier D 图像 decode/quality read 前冻结，且同一
canonical sample 的两个 base、全部 cameras/actors partition 完全一致；否则 P1 fail-closed。

---

# 7. P2 — 统一 Base Census Evaluator

**Task ID**：`WS-V521-P2-BASE-CENSUS-EVAL-01`

目标：把“scene-level quality”下钻到 **frame / camera / actor / boundary / temporal window**。

## P2.1 复用而不是重写现有 evaluator

优先复用：

```text
motion_proj/worldsim_v4/baseline_scene_evaluator.py
motion_proj/worldsim_v4/evaluator.py
motion_proj/worldsim_v4/region_masks.py
configs/worldsim_v4/metrics_v1.yaml
```

V5.2.1 只增加 diagnostics wrapper，不改变 V4 canonical evaluator 的行为。

## P2.2 每个 frame 必须计算的基础轴

### RGB 全图

- PSNR
- SSIM
- LPIPS-Alex

### 静态区域

- static PSNR / SSIM / LPIPS-Alex

### 动态目标区域

- actor PSNR / SSIM / LPIPS-Alex
- actor valid pixel count
- actor image area ratio

必须区分 `dynamic_union` 与 `actor_instance`。DriveStudio dynamic union、projected 3D box 或 base membership 都不是天然的
2D instance GT；每行必须记录 `region_source`、`is_ground_truth`、annotation/hash 和重叠 actor 处理规则。没有可信 actor
identity region 时只报告 union 指标，actor-level metric 返回 `undefined_no_instance_region`，不得把 box/union 冒充 instance mask。

### 动态边界

沿用 frozen boundary band：

```text
dynamic mask morphological band
L1 radius = 3 px
```

计算：

- boundary PSNR
- boundary SSIM
- boundary LPIPS-Alex
- boundary pixel count

## P2.3 新增 geometry diagnostics

只作为 evaluation evidence，不输入模型。

优先用当前 processed LiDAR / calibration：

- static LiDAR depth MAE；
- actor LiDAR depth MAE（有合法 actor support 时）；
- valid projected LiDAR count；
- actor LiDAR support count；
- near/far 分桶；
- depth discontinuity 附近的 RGB residual。

没有可靠 depth GT 的 region 返回 `undefined`，禁止补伪 GT。

geometry metric 还必须先冻结 baseline depth 的语义（如 expected/median/first-hit、是否含 background、alpha threshold）、
坐标链、单位、z-buffer collision、有效深度范围与 calibration hash。某基座不能输出可比 depth 时，其 geometry 轴为
`undefined_no_comparable_base_depth`；禁止只拿稀疏 LiDAR target 而没有同像素 prediction 计算 MAE，也禁止把语义不同的
AD-GS/StreetGS depth 横向排名。

## P2.4 新增 visibility / observability diagnostics

对于 evaluation-only nuScenes actor metadata，记录：

- visibility level；
- actor category；
- bbox / projected mask area；
- camera count；
- LiDAR point count；
- actor distance；
- actor world-frame speed；
- 是否处在进入/离开视野；
- 是否处在遮挡变化窗口。

这些值只用于 **解释 badcase**，不得在 V5.2.1 进入任何 baseline inference。

## P2.5 新增 temporal diagnostics

对同一 scene / camera 的相邻 development frames 构造只读 temporal window。

优先复用已有 temporal metrics 或冻结 flow 资产；若可靠 warp 不可用，则至少计算：

- frame-to-frame residual change；
- actor crop LPIPS temporal delta；
- boundary residual temporal delta；
- same-track actor quality variance；
- visible→occluded / occluded→visible transition 前后 quality jump。

禁止为了 temporal metric 重新训练 optical flow 模型。

若已有 exact flow 可用则记录来源和 hash；没有则 metric=`undefined`，不能用临时 teacher 代替。

没有 frozen correspondence/flow/track 对齐时，raw frame-to-frame residual change、actor-crop LPIPS change 只能标记为
`unwarped_temporal_proxy`，用于候选定位，不能单独触发 `B-TEMPORAL` 或支持“时序不一致”结论。`B-OCC` 同样必须绑定
可审计的 visibility/occlusion transition annotation；仅凭 quality jump 不能命名为去遮挡失败。

## P2.6 流式执行

第一遍 census：

```text
checkpoint
  → render one frame/camera
  → compute metrics
  → append JSONL
  → discard full-resolution prediction unless already canonical
```

默认不保存所有 render。

为避免最终 panel 依赖第二次非确定性渲染，runner 可以维护 **有界在线候选缓存**：每个 leaderboard 仅保留当前 Top-K
所需的 lossless prediction/target/mask，记录进入/淘汰事件并设总 bytes 上限；排名冻结后只原子晋级最终 union，其余缓存按
manifest 精确清理。若选择 P4 重渲染，则 prediction 必须与 P2 首遍 SHA-256 exact；不一致时 panel 标记 blocked，禁止静默
用第二次图像替换首遍 metric 证据。

只保存：

- manifest；
- per-frame metrics；
- hashes；
- resource telemetry；
- 后续被选为 badcase 的图像。

这一步应显著降低磁盘压力。

## P2 输出 schema

P2 第一遍只执行 Discovery；Confirmation quality 在 P9 前不得 decode。输出分成三个粒度，避免把同一 view 的多个 actor 或
temporal window 塞进一个不可审计主键：

```text
BASE_CENSUS_METRICS.jsonl       # base × sample × camera
ACTOR_CENSUS_METRICS.jsonl      # base × sample × camera × actor_token
TEMPORAL_CENSUS_METRICS.jsonl   # base × scene × camera × window_id
```

`BASE_CENSUS_METRICS.jsonl` 每行至少：

```json
{
  "base": "streetgs",
  "scene": "scene-0230",
  "frame": 0,
  "sample_token": "...",
  "camera": 0,
  "partition": "discovery",
  "prediction_sha256": "...",
  "target_sha256": "...",
  "metrics": {
    "global": {},
    "static": {},
    "actor": {},
    "boundary": {},
    "geometry": {},
    "temporal": {}
  },
  "actor_context": {},
  "resource": {}
}
```

actor/window 行必须额外包含 `entity_kind`、`actor_token` 或 `window_id/member_sample_tokens`、region/correspondence provenance、
valid support 与 undefined reason。主键分别是：

```text
(base, scene, sample_token, camera)
(base, scene, sample_token, camera, actor_token)
(base, scene, camera, window_id)
```

### P2 Gate

必须证明：

- same input 重跑 metrics deterministic；
- existing V4 frozen metric 在 tolerance 内可复现；
- LPIPS backbone 明确记录；
- no GT / empty region 正确返回 undefined；
- minimum support、metric direction、quantile operator 与 ranking tie-break 已在读取 Discovery quality 前写入 config；
- Confirmation prediction/target 像素和 quality 均未 decode，绝不只是“未生成 panel”；
- checkpoint before/after SHA exact，renderer/source/config/target/color-space hash 完整。

---

# 8. P3 — Badcase Ranking：禁止单一手工总分

**Task ID**：`WS-V521-P3-BADCASE-RANKING-01`

这是 V5.2.1 的核心。

不要把所有 error 再压成一个人工 scalar score。不同 failure axis 独立排名。

## P3.1 先区分 census、failure label 与 panel selection

三者不能混为一谈：

1. **census denominator**：所有 eligible、undefined、blocked rows；
2. **failure label**：对所有 valid Discovery rows 应用冻结 predicate 后得到的 case/event；prevalence 必须从这个全分母计算；
3. **panel selection**：只从已标注事件中按 axis 选固定 Top-K 代表图，不参与 prevalence 或阈值估计。

在 P2 读取 Discovery quality 前，`census_protocol_v1.yaml` 必须冻结每个 metric 的方向、minimum support、quantile operator
和固定分位点；默认 bad-tail 为 scene-balanced weighted `q=0.10/0.90`，实际数值只能由 Discovery valid rows 一次计算并写入
`badcase_taxonomy_v1.yaml`。若 metadata-only 分母证明该默认值不可计算，必须在 quality read 前用 authorization overlay 改协议；
看过 Discovery/Tier C quality 后不得改 q、support、方向或布尔 predicate。

`OBSERVABILITY` 是解释轴，不因处于低 support tail 就自动成为 RGB badcase；`B-IDENTITY` 必须有独立 ownership/identity
evidence；`B-MIXED` 由多个已成立一级 label 派生，不作为一个会吞掉原标签的新排名轴。

## P3.2 固定 leaderboard

每个 baseline 至少产生以下独立榜单：

1. **GLOBAL_RGB**：最低 global PSNR，LPIPS 作为 tie-break；
2. **ACTOR_RGB**：最低 actor PSNR / 最高 actor LPIPS；
3. **BOUNDARY**：最低 boundary PSNR / 最高 boundary LPIPS；
4. **GEOMETRY**：最高 valid depth MAE；
5. **TEMPORAL**：最高 temporal inconsistency；
6. **OCCLUSION_TRANSITION**：遮挡变化窗口中最大的 quality jump；
7. **OBSERVABILITY**：最少有效 view / LiDAR / visible support，但只作为解释轴，不等价于 RGB failure。

每个 leaderboard 在 Discovery 中同时产出两张表：

- `severity_topk`：不限制 scene 的真实 tail；
- `scene_coverage_topk`：每个 scene 最多 2 个 entity，避免 panel 被单一长 scene 占满。

两者均使用固定 K：

```text
K = min(12, valid_rows)
```

并采用 deterministic tie-break：

```text
scene → frame/sample → camera
```

## P3.3 建立 badcase union

最终 panel 候选集是两类 leaderboard Top-K 的 **并集**，不是重新计算一遍总分；正式 `BADCASE_REGISTRY.jsonl`
则覆盖所有通过 failure predicate 的事件，不只覆盖 Top-K。

同一个 frame 可以命中多个 failure axis：

```json
"failure_axes": ["ACTOR_RGB", "BOUNDARY", "TEMPORAL"]
```

这样保留 failure 的多因素结构。

## P3.4 Case ID 与 Failure Event ID

一个 case 可以命中多个 axis，因此使用两级 deterministic ID：

```text
BC-<BASE>-<12hex>       # entity case，axis-independent
BCE-<BASE>-<12hex>      # failure event，axis-specific
```

hash 输入分别至少为：

```text
case:  base | dataset | scene | sample_or_window | camera | entity_kind | actor_token_if_any | metric_protocol_version
event: case_id | failure_axis | taxonomy_version
```

以后 V5.2 所有算法实验必须引用 `case_id`，针对具体轴时再引用 `event_id`，禁止只写“scene-1087 有提升”。这些 ID
是 badcase registry 标识，不替代统一 failure ledger 的 `V*-FNN`。

## P3.5 初始 failure taxonomy

冻结以下第一版 taxonomy：

| Code | failure class | 典型定义 |
|---|---|---|
| `B-RGB-GLOBAL` | 全局外观/重建 | global RGB 显著差，但不一定集中动态目标 |
| `B-ACTOR` | 动态 actor 重建 | actor region 明显差于同帧 static region |
| `B-BOUNDARY` | 动态边界混合 | actor interior 尚可，但 boundary 显著恶化 |
| `B-GEOMETRY` | 几何/深度 | LiDAR-supported depth residual 异常 |
| `B-OCC` | 遮挡/去遮挡 | visibility transition 后出现 hole/ghost/residual jump |
| `B-TEMPORAL` | 时序闪烁/漂移 | 单帧尚可但连续帧不一致 |
| `B-SPARSE-OBS` | 观测不足 | actor 有效 observation/visibility/LiDAR support 极低 |
| `B-IDENTITY` | 身份/归属错配 | RGB 可接受，但 instance/ownership evidence 与 actor 不一致 |
| `B-MIXED` | 多因素耦合 | 同时满足多个一级 failure class |
| `B-UNRESOLVED` | 当前证据不足 | 不强行解释 |

P3 只允许基于 Discovery 冻结 taxonomy。每个 class 必须在 freeze 中写出可执行 predicate、metric/support denominator、
undefined policy、数值阈值与来源；“显著差”“尚可”等自然语言不能单独充当分类规则。

---

# 9. P4 — 自动生成 Badcase Visual Panels

**Task ID**：`WS-V521-P4-BADCASE-PANEL-01`

P4 只对 P3 冻结后的 **Discovery** badcase panel union 生成 panel。Tier C/Confirmation panel 只能在 P9 一次性 quality read
后按同一冻结 builder 生成。

## P4.1 Panel 内容

基础 panel 至少包含：

```text
GT RGB
Base Render RGB
|GT - Render| residual heatmap
Dynamic/Actor Mask overlay
Boundary band overlay
LiDAR projected depth / depth residual（若存在）
```

如果 matched base 可用：

```text
GT
AD-GS
StreetGS
AD-GS residual
StreetGS residual
mask / boundary / depth
```

但只有满足严格 matched contract 的 row 才允许做这种并排比较。

## P4.2 M1 diagnostic panel 额外内容

仅针对 historical / Tier D 中有 U2/B3 evidence 的 case：

```text
SAM observation mask
accepted/rejected view status
U2/B3 posterior projection
ownership target / prediction
visibility / observation count
```

不要运行新的 graph 或 propagation。

## P4.3 Panel 命名

```text
panels/<case_id>/panel.png
panels/<case_id>/metadata.json
```

`metadata.json` 必须包含所有输入资产 path/hash 和 metric row hash。

panel 只作可追溯的诊断材料，不能由 Codex 主观看图代替 frozen predicate 或人工 verdict。V5.2.1 默认不设人工晋级门；
若后续确需人评，必须另行预注册完整、可独立执行的评测提示词，并由用户或指定评审者填写，Codex 不得代填。

---

# 10. P5 — Base Failure Localization

**Task ID**：`WS-V521-P5-FAILURE-LOCALIZATION-01`

目标不是“看图讲故事”，而是找出 badcase 与 driving-native condition 的系统关联。

## P5.1 对每个 failure class 做条件分析

至少统计：

- actor pixel area；
- actor distance；
- actor speed；
- LiDAR support；
- camera/view count；
- visibility；
- entering/leaving FOV；
- occlusion transition；
- static vs dynamic residual ratio；
- boundary vs actor-interior residual ratio；
- scene / camera distribution。

## P5.2 使用 scene-balanced 统计

不能让一个长 scene 的大量 frame 淹没其他场景。

输出：

- frame-level distribution；
- scene-balanced mean/median；
- per-scene badcase rate；
- bootstrap CI；
- Spearman correlation（仅解释相关性）；
- 按 distance / visibility / motion / support 的预注册分桶。

不要把 correlation 写成 causality。

所有 prevalence、correlation、分桶和 bootstrap 都必须使用完整 census denominator（含 undefined/blocked coverage 报告），
不得只对 Top-K/panel union 做统计。有效 scene 少于 2 或某轴有效支持不足时，CI/correlation 返回 `undefined_insufficient_denominator`。

## P5.3 要回答的六个核心问题

1. AD-GS/StreetGS 的 badcase 是否主要集中在 **dynamic actor** 而非 static background？
2. dynamic actor 失败是否主要来自 **boundary**，还是 actor interior 也明显崩？
3. 失败是否与 **distance / small projected area / low LiDAR support** 强关联？
4. **occlusion/disocclusion** 是否是独立的高频 failure class？
5. 单帧较好的 case 是否仍有明显 **temporal inconsistency**？
6. `B-SPARSE-OBS` 与 `B-IDENTITY` 是否在 RGB 较好的 case 上仍独立存在？

只有第 6 类成立，才真正直接支撑 V5.2 M1 ownership/association 的论文主问题。

---

# 11. P6 — M1 Re-audit：判断 ownership 是根因、放大器还是旁支

**Task ID**：`WS-V521-P6-M1-REAUDIT-01`

**禁止修改 U2/B3。**

M1 re-audit 只把 frozen U2/B3 与 badcase registry 对齐。

## P6.1 建立四象限

对于能同时获得 base RGB + M1 evidence 的 development/historical case，划分：

| bucket | Base RGB | M1 ownership | 含义 |
|---|---|---|---|
| Q1 | good | bad | **最纯的 M1 目标**：基座能渲染，但 ownership/identity 错 |
| Q2 | bad | bad | base 与 M1 耦合，不能把全部收益归因于 ownership |
| Q3 | bad | good | M1 解决不了基座主要 RGB/geometry 问题 |
| Q4 | good | good | control |

good/bad 的阈值必须由 Discovery distribution 的固定 quantile / historical frozen thresholds 定义，并写进 config；不能看 panel 后改。
四象限只在 base/M1 的 exact same sample/camera/target mapping 上建立；historical aggregate 不可伪配成逐 view overlap。
`m123_reaudit_v1.yaml` 必须在读取相交 quality 前冻结 minimum denominator；`M1_CORE_CONFIRMED` 至少要求 Q1 覆盖
2 个独立 scene，单一 scene 或无法证明 overlap 时不得晋级。

## P6.2 M1 需要对齐的指标

至少关联：

- U2/B3 IoU；
- Boundary F1；
- Brier / ECE / NLL；
- FP semantic mass；
- FN semantic mass；
- accepted observation views；
- zero-observation Gaussian count；
- actor foreground coverage；
- identity recall；
- persistent-track fraction（若历史已有）；
- base actor/boundary RGB error；
- visibility / LiDAR support。

## P6.3 必须专门复核 scene-1087

明确回答：

```text
1087 是因为 base reconstruction 本身就坏？
还是 RGB 尚可，但 M1 缺 observation？
还是两者同时存在？
```

如果是第三种，后续 V5.2 必须把 evaluation 设计成能分离这两个问题，不能仅看 semantic IoU。

## P6.4 M1 V5.2 readiness gate

V5.2.1 结束时对 M1 只能给以下三种结论之一：

### `M1_CORE_CONFIRMED`

满足：

- Q1 不是孤立个例；
- ownership/identity error 在多个 scene 上独立于 base RGB failure；
- sparse observation / temporal identity persistence 与 failure 有稳定关系；
- U2/B3 仍有真实正信号但 observation source 不足。

此时 V5.2 才值得进入 driving-native actor track / temporal Bayes。

### `M1_ENTANGLED`

M1 error 大部分与 base geometry/RGB failure 共现，必须先重新定义任务或联合 evaluation，不能单独把 ownership 当 paper core。

### `M1_NOT_PRIMARY`

绝大多数重要 badcase 在 M1 已经 good 的情况下仍存在，说明论文主矛盾不应继续押 ownership。

### `M1_EVIDENCE_INSUFFICIENT_KEEP_PENDING`

base RGB 与 frozen M1 evidence 的 exact overlap、有效 scene 或 actor/identity denominator 不足，无法在不补造映射的情况下建立
四象限。该结论不是 M1 reject，也不能据此自动进入 TrackBayes/TubeBayes。

---

# 12. P7 — M2 Re-audit：Repair 到底在修什么

**Task ID**：`WS-V521-P7-M2-REAUDIT-01`

不修改 router，不跑新 threshold search。

只读复用 V4/V5 已有 M2 evidence，并把 request 映射到新的 base failure taxonomy。

## P7.1 特别检查 historical geometry caveat

历史 M2 已出现：

- selective risk 有价值；
- RGB / hole quality 可以改善；
- 但 geometry MAE 存在退化 caveat。

V5.2.1 必须回答：

1. M2 触发的 request 主要来自 `B-OCC`、`B-GEOMETRY` 还是普通 `B-ACTOR`？
2. RGB improvement 是否常伴随 geometry degradation？
3. router 的 abstain 是否正确避开本来就没有可靠 repair target 的 case？
4. M2 是解决 **base representation 缺失**，还是在补 M1 错误留下的 hole？
5. 如果 M1 future ownership 完善，M2 的必要性会增加、减少还是不变？

## P7.2 M2 只给研究定位结论

最终只能输出：

```text
M2_ORTHOGONAL_KEEP_PENDING
M2_DEPENDS_ON_M1_KEEP_PENDING
M2_BASE_RECONSTRUCTION_PATCH_NOT_CORE
M2_EVIDENCE_INSUFFICIENT
```

V5.2.1 不实现任何 M2 变化。

---

# 13. P8 — M3 Re-audit：区分“历史确认的 temporal delta”与“V5 constraint route”

**Task ID**：`WS-V521-P8-M3-REAUDIT-01`

必须明确区分两个事实：

## V4 M3 historical confirmed

V4 frozen temporal SE(3) delta 已有正式 confirmation：

- validation：warp L1 relative improvement 约 `30.41%`；
- validation temporal LPIPS relative improvement 约 `2.65%`；
- exact-once test：18 scenes attempted，12 evaluable + 6 abstain；
- test warp L1 relative improvement 约 `34.39%`；
- test temporal LPIPS relative improvement 约 `16.37%`；
- rollback / identity / operation gates 通过。

这些历史正式结果只允许读取 frozen summary，不重新搜索 test case。

## V5 M3 constraint-projection route rejected

V5 constraint-projected temporal development 已因真实 T2 violation signal 不足而 rejected；不能把这个 reject 倒写成“V4 temporal 全部失败”。

## P8 要回答

把新的 base badcase taxonomy 与 development temporal evidence 对齐：

1. `B-TEMPORAL` 在基座中到底多常见？
2. 它是否主要发生在 `B-ACTOR/B-BOUNDARY/B-OCC` 上？
3. V4 temporal delta 改善的是 **真实基座 temporal badcase**，还是只对编辑轨迹本身做平滑？
4. M3 与未来 driving-native M1 是互补关系还是重复解决 actor trajectory？
5. 如果 V5.2 M1 引入 actor canonical coordinates/track，M3 是否应后移为 downstream editor，而不是 paper core？

最终只输出研究定位，不修改 M3。

可能结论：

```text
M3_ORTHOGONAL_VALIDATED_KEEP_PENDING
M3_DOWNSTREAM_OF_M1_KEEP_PENDING
M3_NOT_CURRENT_BOTTLENECK
M3_EVIDENCE_INSUFFICIENT_KEEP_PENDING
```

---

# 14. P9 — Confirmation Read：taxonomy freeze 后一次性确认

**Task ID**：`WS-V521-P9-BADCASE-CONFIRMATION-01`

只有 P3 taxonomy 与 P5–P8 的分析模板全部冻结后才能读取 Tier C quality/panels。

禁止：

- 改 ranking K；
- 改 failure class 定义；
- 改 good/bad quantile；
- 改相关性分桶；
- 删除“不符合故事”的 confirmation case。

确认内容：

- 每个 failure class 的 prevalence；
- Discovery→Confirmation 方向一致性；
- Q1/Q2/Q3/Q4 比例；
- base-specific vs shared failure；
- taxonomy 中 `UNRESOLVED` 比例。

P9 使用 P2 同一 evaluator/config 先生成 Confirmation metrics，再应用完全相同的 frozen predicate/ranking schema；只有在
verdict 和所有 denominator 落盘后，才能生成 Confirmation panel。不得把 Discovery 的数值阈值重新按 Tier C 分布拟合。

如果某个 failure class 只在 Discovery 成立、Confirmation 消失：

```text
标记 hypothesis_not_confirmed
```

不要再回去调整 taxonomy 使它通过。

---

# 15. P10 — V5.2.1 Closeout / 下一阶段 Go-NoGo

**Task ID**：`WS-V521-P10-CLOSEOUT-01`

## 15.1 必须生成的最终 artifact

```text
BADCASE_REGISTRY.jsonl
BADCASE_SUMMARY.json
BADCASE_TAXONOMY_FREEZE.yaml
DISCOVERY_CONFIRMATION_FREEZE.json
BASE_ASSET_REGISTRY.json
MATCHED_FRAME_REGISTRY.jsonl
BASE_CENSUS_METRICS.jsonl
ACTOR_CENSUS_METRICS.jsonl
TEMPORAL_CENSUS_METRICS.jsonl
BADCASE_LEADERBOARDS.json
PANEL_REGISTRY.jsonl
M1_REAUDIT.json
M2_REAUDIT.json
M3_REAUDIT.json
```

以及：

```text
docs/WORLDSIM_V5_2_1_BADCASE_REPORT.md
docs/WORLDSIM_V5_2_1_M123_REVIEW.md
docs/WORLDSIM_V5_2_1_CLOSEOUT.md
```

## 15.2 BADCASE_REGISTRY 最小字段

```json
{
  "case_id": "BC-STREETGS-xxxxxxxxxxxx",
  "event_ids": {
    "ACTOR_RGB": "BCE-STREETGS-xxxxxxxxxxxx",
    "BOUNDARY": "BCE-STREETGS-yyyyyyyyyyyy"
  },
  "base": "streetgs",
  "dataset": "nuscenes",
  "scene": "scene-0230",
  "frame": 0,
  "sample_token": "...",
  "camera": 0,
  "entity_kind": "view|actor|temporal_window",
  "actor_token": null,
  "temporal_window_id": null,
  "evidence_tier": "H|D|C",
  "split_role": "discovery",
  "failure_axes": ["ACTOR_RGB", "BOUNDARY"],
  "failure_class": ["B-ACTOR", "B-BOUNDARY"],
  "metrics": {},
  "actor_context": {},
  "m1_context": {},
  "asset_provenance": {},
  "panel_path": "...",
  "classification_status": "labeled|unresolved|blocked",
  "confirmation_verdict": "not_applicable|direction_confirmed|hypothesis_not_confirmed|insufficient",
  "selected_for_panel": true,
  "blocker_reason": null
}
```

Discovery 个案不能仅因进入 Top-K 就写 `confirmed`；Confirmation 是 failure class/hypothesis 的独立方向复核，不是把某个
Discovery frame 改名为 confirmed。historical/Tier D/Tier C 必须可分组，禁止混在同一 prevalence 分母。

## 15.3 最终必须给出一个研究决策矩阵

| 问题 | 证据 | prevalence | 跨 scene 稳定性 | 与 base RGB 独立？ | 当前模块 | V5.2 优先级 |
|---|---|---:|---|---|---|---|
| sparse observation | ... | ... | ... | ... | M1 | ... |
| identity persistence | ... | ... | ... | ... | M1 | ... |
| dynamic boundary | ... | ... | ... | ... | M1/M2 | ... |
| disocclusion hole | ... | ... | ... | ... | M2 | ... |
| geometry error | ... | ... | ... | ... | Base/M2 | ... |
| temporal inconsistency | ... | ... | ... | ... | M3 | ... |

最终不是按主观喜好选方向，而是按以下顺序：

```text
frequency
→ severity（按 axis 单独报告 tail distance/quantile，不合成总分）
→ cross-scene reproducibility
→ independence from base reconstruction artifacts
→ scientific addressability
```

这是按顺序审阅的决策清单，不是乘法公式。可以做定性/分轴判断，**不要再造一个手工总分把五项压成单 scalar。**

---

# 16. V5.2.1 的成功标准

只有以下全部成立，V5.2.1 才算完成：

### 资产

- 至少一个主基座 exact checkpoint/render 链可执行；
- AD-GS / StreetGS 每个 scene 的 presence/hash/protocol 状态都有 registry；
- 缺失资产有明确 blocker，不靠近似 checkpoint 填洞。

closeout 必须给出 coverage terminal：

```text
complete_full                    # 两个基座均有合法 census，matched cohort 按证据可用
complete_partial_base_blocked    # 至少一个基座完成，另一基座的缺失分母完整保留
blocked_no_executable_base       # 两个主基座均无 exact executable chain
```

`complete_partial_base_blocked` 不得回答跨基座优劣或 shared-failure prevalence，但可以完成单基座 census 与 M1 historical re-audit。

### Badcase

- 不再只有 scene-level worst metric；
- 至少建立 frame/camera 级 registry；
- 每个 failure 有 deterministic ID；
- 至少覆盖 global / actor / boundary / geometry / temporal / observability 六类轴中可合法计算的部分；
- panel 由 frozen ranking 自动产生，不手工挑图。

### 数据纪律

- Discovery/Confirmation 在 quality read 前冻结；
- Confirmation 只在 taxonomy freeze 后一次读取；
- fresh validation/test/KITTI 未读取；
- V5.1 Stage H 未执行。

### M1/M2/M3

- M1 得到 `CORE_CONFIRMED / ENTANGLED / NOT_PRIMARY / EVIDENCE_INSUFFICIENT_KEEP_PENDING` 之一；
- M2 得到明确研究定位；
- M3 区分 V4 confirmed temporal 与 V5 rejected constraint route 后得到明确研究定位或
  `EVIDENCE_INSUFFICIENT_KEEP_PENDING`；
- 每个判断都引用具体 failure IDs / frozen historical evidence，而不是 paper intuition。

### 最终边界

V5.2.1 **不得提交任何算法改进 candidate**。

Closeout 后停在：

```text
badcase basis frozen
problem definition frozen
M1/M2/M3 roles re-audited
ready for V5.2.2 method design
```

而不是自动进入 V5.2.2。

---

# 17. 3090 执行纪律

默认单卡 3090，优先稳定而不是并发。

## GPU

- 一次只允许一个 heavy renderer/model process；
- LPIPS 可与当前 frame eval 同进程，但不得同时起第二个 base renderer；
- 每个 scene 前记录 free VRAM；
- CUDA OOM 后允许：清理进程 / empty cache / fresh process / 降 evaluation batch；
- **不允许**：降 render 分辨率、改 scene、改 split、改 metric protocol 来“救”OOM。

## CPU / Disk

- metric rows 流式写 JSONL；
- panel 只对 frozen badcases 保存；
- 中间 render 默认 ephemeral；
- 所有 canonical output 必须先 manifest 再清理可再生缓存；
- 不删除任何受保护历史 evidence。

## 可恢复失败

以下情况不要停整个任务：

```text
单 scene checkpoint missing
单 baseline blocked
单 temporal metric undefined
单 camera 无 actor mask
单 scene 无 LiDAR support
第三方 renderer 单 scene 工程失败
```

记录 blocker，继续其他合法工作。

只有以下情况 fail-closed：

```text
即将读取 fresh validation/test
发现 split leakage 且无法隔离
canonical checkpoint hash mismatch
GT 与 render frame mapping 不可证明
repo provenance 无法审计
所有主基座均无任何 exact executable asset
```

---

# 18. 测试要求

至少新增：

```text
tests/test_worldsim_v521_scope_lock.py
tests/test_worldsim_v521_asset_registry.py
tests/test_worldsim_v521_split_freeze.py
tests/test_worldsim_v521_census_schema.py
tests/test_worldsim_v521_badcase_ranking.py
tests/test_worldsim_v521_confirmation_lock.py
tests/test_worldsim_v521_panel_manifest.py
tests/test_worldsim_v521_m123_reaudit.py
```

关键单测：

1. validation/test path 被输入时必须 fail；
2. old stride=10 StreetGS 不得进入 matched cohort；
3. asset hash mismatch 不得 silent fallback；
4. ranking deterministic；
5. 不同 failure axis 不得被自动合成单一 total score；
6. confirmation panel 在 taxonomy freeze 前必须拒绝；
7. undefined metric 不得被填 0；
8. duplicate canonical primary key 必须拒绝；不同 base 或不同 actor 的合法 row 不得误判重复；
9. matched comparison 必须 same target hash；
10. checkpoint before/after SHA exact；
11. 同一 canonical sample 的 base/camera/actor partition 必须一致，partition hash 不得消费 base/camera/actor；
12. temporal window 跨 partition 必须 undefined；
13. Top-K/panel union 不得作为 prevalence denominator；
14. `case_id` 轴无关、`event_id` 轴相关且 deterministic；
15. 没有 comparable rendered depth/flow/identity region 时对应 metric/class 必须 undefined，禁止 proxy 偷换；
16. P4 重渲染 prediction hash 与 P2 不一致时必须 fail-closed。

每阶段只跑相关定向测试；P10 再跑完整相关测试集。

---

# 19. Codex 的执行方式

你是执行 Agent，不是提案 Agent。按 P0→P10 连续推进。

新上下文启动时按 `AGENTS.md` 渐进式读取：先读 `RESEARCH_STATUS.md` 顶部当前节、`RESEARCH_FAILURES.md` 使用合同/
版本总览、`EXPERIMENTS.md` 顶部 V5.2.1 注册项和本计划；再按 §1.6 的 failure ID 展开完整条目与 evidence。不得从 archive
中的历史“下一步”取得执行授权，也不得一次性加载全 failure ledger。

## 不要因为局部异常就停

遇到非关键问题：

```text
分析原因
→ 记录 blocked/failed run
→ 只做最小工程修复
→ 新 run 重试
→ 保留旧失败
→ 继续下一合法阶段
```

不要每遇到：

- 缺一个包；
- 一个 scene 缺资产；
- 一个 metric undefined；
- 一个 CUDA process 失败；

就暂停等待用户。

## 但不要越过科研门禁

不能为了“继续跑”而：

- 读取 fresh validation/test；
- 修改 split；
- 替换 checkpoint；
- 重新训练近似 baseline；
- 改 metric definition；
- 偷做 V5.2 algorithm；
- 执行 Stage H/BKI。

## 每个阶段完成后维护事实源

```text
docs/RESEARCH_STATUS.md
docs/EXPERIMENTS.md
docs/RESEARCH_FAILURES.md
```

- `RESEARCH_STATUS.md` 与 `EXPERIMENTS.md` 在正式 milestone/terminal 后窄改；
- `RESEARCH_FAILURES.md` 仅在出现 blocked/rejected、假设推翻、协议/分母失效、工程恢复或旧风险解除时更新；
- 没有新 failure 时，不改 failure ledger，只在 `EXPERIMENTS.md`/run metadata 写 `failure_ledger_delta=none`。

要求事实与结论分离：

```text
工程 blocked != 方法 rejected
相关性 != 因果
historical evidence != V5.2 confirmation
asset missing != baseline algorithm failure
```

研究 commit 正文至少写：

```text
task/run ID
split role
seed（若适用）
source HEAD
input/output fingerprint
checkpoint hash
quality-read locks
failure_ledger_refs
failure_ledger_delta
```

---

# 20. V5.2.1 最终希望得到的科研答案

完成后，Closeout 必须用非常直接的方式回答下面问题：

### A. 基座究竟坏在哪里？

例如不能只写：

> AD-GS scene-0295 PSNR 最差。

而要能写成：

> AD-GS 的主要高严重度错误集中在远距/低 LiDAR support 的 moving actor boundary 与 visibility transition；static background 同帧正常，说明它不是单纯全局 appearance 问题。

或相反，如果证据不支持，就明确写不支持。

### B. M1 值不值得继续当核心？

关键不是“U2/B3 有正信号”，而是：

> 是否存在一批 **Base RGB good + Ownership bad** 的稳定跨场景 case，并且它们显著与 temporal observation / actor identity persistence 相关。

如果有，V5.2 的 `actor track + canonical coordinates + temporal Bayesian ownership` 才有真正靶点。

### C. M2/M3 是不是当前 paper core？

- M2 如果主要修 base disocclusion/geometry hole，应作为 downstream repair 或独立问题；
- M3 如果主要解决编辑后的 temporal smoothness，而且 V4 已验证，则它可能是稳定 downstream contribution，但不一定是 V5.2 当前首要矛盾；
- 不允许因为历史上 M2/M3 曾经“work”就自动把它们塞回新主线。

---

# 21. V5.2.1 终止条件

当且仅当以下 artifact 全部冻结后停止：

```text
[done] exact base asset registry
[done] development census metrics
[done] deterministic badcase leaderboards
[done] BADCASE_REGISTRY.jsonl
[done] frozen visual panels
[done] failure localization report
[done] M1 re-audit
[done] M2 re-audit
[done] M3 re-audit
[done] internal confirmation
[done] V5.2.1 closeout
```

然后明确写：

```text
V5.2.1 complete.
No algorithm modification performed.
Stage H not executed.
Fresh validation/test remain unread.
Await V5.2.2 method design based on frozen badcase evidence.
```

**到这里停止。不要自动开始 V5.2.2。**
