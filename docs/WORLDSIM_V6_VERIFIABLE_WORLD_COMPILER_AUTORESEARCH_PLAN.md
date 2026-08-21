# WorldSim V6 — Verifiable World Compiler Autoresearch Plan

> 中文名：**可验证生成式世界编译器**
>
> 路线代号：`WorldSim V6`
>
> 推荐论文工作名：**Beyond Logged Trajectories: Verifiable World Compilation for Closed-Loop Autonomous Driving**
>
> 当前资源默认：**单卡 RTX 3090 24GB**
>
> 执行模式：**无人值守 Autoresearch；普通科研判断、工程恢复、分支切换、候选淘汰无需人工审批**
>
> Git 前置事实：当前最新研究分支由用户指定为
> `research/worldsim-v5.1-m1`
> 远端仓库：`https://github.com/yunlongdeng123/Motion-Proj`
>
> 核心纪律：**不删除历史失败、不覆盖 immutable run、不事后改旧 gate、不重复已经被 failure ledger 否决的路线。**
>
> 本计划本身不假设任何机器上的固定绝对路径。所有 repo / data / runs / cache / models / third-party 路径必须在运行时解析并写入 local capability manifest。

---

# 0. V6 北极星

V6 不再以：

```text
某个 3DGS 基座
→ 找 badcase
→ 修 ownership / boundary / trajectory
→ 再找 badcase
```

作为论文主线。

V6 正式问题定义：

> **给定有限真实驾驶日志，以及新的路线、actor 行为、天气或场景条件，将它们编译成一个空间可扩展、显式三维、时序一致、可编辑、可存储、可精确重放，并能够声明“哪些区域对哪些任务可信”的四维仿真世界。**

核心范式：

```text
真实日志
Camera / LiDAR / Pose / Map / Tracks
                    ↓
       可替换的重建前端 Adapter
  optimization / feed-forward / specialist
                    ↓
        统一世界中间表示 SceneIR
                    ↓
          Support / Provenance
                    ↓
目标路线 / actor 行为 / 天气 / 文本 condition
                    ↓
        Missing-world Proposal
                    ↓
 Photo / Geometry / Semantic / Dynamics
          独立验证与不确定性
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
      VERIFIED              UNKNOWN
         ↓                     ↓
显式 3D/4D Bake          abstain / fallback
         ↓
  Content-addressed World Package
         ↓
   Deterministic Simulation Runtime
         ↓
 ┌────────────┬──────────────┬───────────────┐
 ↓            ↓              ↓
LogSim     WorldSim       IL / RL / Regression
```

最重要的系统边界：

```text
Compile time:
允许生成、采样、搜索、筛选、验证、重建、固化

Runtime:
禁止在线随机改世界
只运行冻结后的显式世界状态
保证同一 episode 可确定性重放
```

---

# 1. V5.2 处置：保留证据，停止作为主论文路线

V5.2/V5.2.1 已产生高价值资产：

- Base Badcase Census；
- `9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 unresolved` 人工归因；
- Base Validity Gate；
- M1/M3 causal bridge 问题定义；
- UNKNOWN / abstention 原则；
- immutable run / exact-once / failure ledger 纪律。

但 V6 不继续把以下路径当主论文：

```text
StreetGS
→ TrackBayes-GS
→ M3 SE(3)
→ M2 Router
```

原因：

1. 研究变量越来越绑定 StreetGS 内部 actor representation；
2. 主要贡献容易退化为现有基座的 repair stack；
3. 无法直接解决“大场景、偏离日志轨迹、生成扩展、闭环可用性”；
4. M1/M3 的大量能力已与新一代前馈 driving reconstruction 的 instance/canonical/motion 表示发生重叠；
5. 工业上真正的瓶颈是**日志支持域之外的世界是否能够可信扩展并固化**。

冻结处置：

```text
V5.2 = forensic evidence branch
research direction = superseded_by_v6_world_compiler

V5.2 artifacts = retained
V5.2 failure facts = retained
V5.2 confirmation/test discipline = retained
TrackBayes as V6 north star = stopped
Stage H / BKI = do not execute
```

M1/M2/M3 在 V6 中的迁移关系：

| 旧模块 | V6 中的位置 | 是否主贡献 |
|---|---|---:|
| M1 Gaussian Ownership | SceneIR 的 actor/static/unknown provenance 与 local validity | 否 |
| M2 Risk Router | Task-conditioned validity + accept/abstain | **升级为核心思想之一** |
| M3 Temporal SE(3) | Actor canonical state / dynamic consistency verifier | 否 |
| Base Validity Gate | Compiler input/support gate | 是实验协议，不是独立网络模块 |

---

# 2. 执行总原则

## 2.1 允许 Agent 自主做什么

Codex 可以自主：

- 更新研究状态文档；
- 创建提交、push 普通研究分支；
- 创建 integration branch；
- 合并项目分支；
- 解决 Git 冲突；
- 创建 V6 新分支；
- 拉取公开代码；
- 安装隔离环境；
- 编译当前 GPU 所需 CUDA extension；
- 下载公开、无需人工授权的权重；
- 设计 development-only 诊断；
- 创建新科研假设；
- 创建新 task/config/run ID；
- 淘汰不成立路线；
- 根据反思切换前端；
- 修改 V6 方法实现；
- 增加消融；
- 更新 `RESEARCH_STATUS.md` / `EXPERIMENTS.md` / `RESEARCH_FAILURES.md`；
- 在发展集上进行有限、自洽、可归因的自动研究循环；
- 在候选冻结后执行预先冻结的一次性 confirmation。

无需因为以下情况停下询问用户：

- 普通 Python/package 缺失；
- CUDA extension 需要重编；
- 单个 upstream adapter 不兼容；
- 某个候选算法 rejected；
- 某个前端无法在 3090 运行；
- 某个公开权重不可用但存在合法替代前端；
- 单个 scene/case unavailable；
- 普通 Git conflict；
- 缓存、stdout、launcher、路径变化；
- 可由新 run、新 task、新分支合法恢复的工程错误。

## 2.2 Agent 不能为了“继续跑”而做什么

禁止：

- `git push --force` 覆盖公共历史；
- 删除旧失败 run；
- 原地改写历史 frozen config；
- 看到结果后放宽旧 primary gate；
- 删除困难 case 改善 aggregate；
- 读取 confirmation/test 后继续调当前 candidate；
- 用一个新 scalar 混掉 photo/geometry/semantic/dynamics validity；
- 用 learned generator 自己产生的数据同时充当唯一 evaluator；
- 把生成内容称为真实世界 GT；
- 把 `UNKNOWN` 强行改成 PASS/FAIL；
- 把工程 blocked 写成算法 rejected；
- 把算法 rejected 写成 upstream 不可运行；
- 为适配 3090 偷偷降正式分辨率后继续称 exact upstream reproduction；
- 重启 Stage H/BKI/KNN/Graph 路线；
- 用 V6 名义重新做 cut-in mining 主线；
- 杀死非本任务用户进程抢资源；
- 删除或覆盖用户未提交工作。

---

# 3. 路径与资产合同：禁止写死机器路径

## 3.1 Repo 根目录

所有脚本必须从 Git 推导 repo root：

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
```

禁止在 committed code/config 中出现某台机器特有的 repo absolute path。

## 3.2 Local capability manifest

V6 第一次启动自动创建：

```text
.local/worldsim_v6/capabilities.local.yaml
```

该文件加入 `.gitignore`，**不得提交**。

它只保存本机事实，例如：

```yaml
repo_root: <resolved-at-runtime>
run_root: <resolved-at-runtime>
cache_root: <resolved-at-runtime>
asset_root: <resolved-at-runtime>

datasets:
  nuscenes: <resolved-if-available>
  kitti: <resolved-if-available>
  waymo: <resolved-if-available>

envs:
  primary_python: <resolved>
  drivestudio_python: <resolved-if-available>

gpu:
  name: <detected>
  total_vram_mib: <detected>
  compute_capability: <detected>
```

## 3.3 Committed config 只使用 logical URI

正式 config 使用：

```text
repo://
runs://
cache://
asset://
dataset://nuscenes
dataset://kitti
third_party://recondrive
third_party://tokengs
```

运行时由统一 resolver 转换。

禁止在新 V6 committed config 中写：

```text
<machine-specific-repo-absolute-path>
<machine-specific-home-absolute-path>
<platform-specific-local-path>
某个旧 run 的宿主绝对根路径
```

历史 frozen 文档中的旧路径保持不改写。

## 3.4 原子输出

所有新资产：

```text
<target>.partial.<uuid>
        ↓
完整校验
        ↓
atomic rename
        ↓
正式 target
```

manifest 内部只能保存：

- logical URI；
- 相对路径；
- content hash；

不能保存 `.partial` staging absolute path。

---

# 4. Phase G0 — Git / 文档 / 分支收敛前置

**这是 V6 的强制前置。没有完成 G0，不得启动任何 V6 GPU 实验。**

Task：

```text
WS-V6-G0-REPO-CONVERGENCE-01
```

## 4.1 获取真实仓库状态

自动执行：

```bash
git remote -v
git fetch --all --prune --tags
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git for-each-ref refs/remotes/origin/
```

记录：

```text
REPO_PREFLIGHT.json
```

至少包括：

- origin URL；
- current branch；
- local HEAD；
- origin/main HEAD；
- `origin/research/worldsim-v5.1-m1` HEAD；
- dirty status；
- 所有 remote branch head；
- ahead/behind；
- tags；
- submodules；
- Git LFS 状态（若存在）。

## 4.2 工作树 dirty 时的无人值守保护

不得执行：

```text
git reset --hard
git clean -fd
```

若工作树 dirty：

1. 保存 tracked diff；
2. 保存 staged diff；
3. 保存 untracked inventory + bytes/hash；
4. 创建唯一 recovery reference；
5. 对能够安全版本化的 tracked 修改创建 rescue branch；
6. untracked 大文件不得自动塞进 Git；
7. 确认 checkout/merge 不会覆盖 untracked path；
8. 完成保护后继续。

输出：

```text
PRE_V6_WORKTREE_RECOVERY.json
```

目标是**不丢用户工作**，不是把所有随机大文件都提交。

---

# 5. G1 — 在最新研究分支更新 V5.2/V6 文档并 push

用户指定最新研究分支：

```text
research/worldsim-v5.1-m1
```

首先 checkout 对应 remote tracking branch。

## 5.1 更新唯一权威文档

至少更新：

### `docs/RESEARCH_STATUS.md`

必须写清：

```text
V5.1 = closed
V5.2.1 = badcase census + human attribution completed
V5.2 TrackBayes continuation = superseded_by_v6_direction_reset
Stage H = never executed
M1/M2/M3 = retained as evidence / subsystem ideas, not current north star
V6 = active planned route
```

### `docs/RESEARCH_FAILURES.md`

这是**唯一 failure ledger**。

不得创建：

```text
WORLDSIM_V6_FAILURES.md
RESEARCH_FAILURES_V6.md
```

新增 V6 route-transition 条目，例如：

```text
V6-F01 governance/research-direction
```

记录：

- V5.2 为什么不继续作为主论文；
- 哪些 V5.2 事实继续有效；
- 哪些 V5.2 阶段没有执行；
- V6 复用哪些基础设施；
- 禁止把 V6 重新退化成 StreetGS repair project。

### `docs/EXPERIMENTS.md`

登记：

- V5.2.1 terminal；
- V5.2 autoresearch plan 未作为主路线完整执行；
- V6 G0/G1/G2/G3 任务；
- V6 尚未产生方法质量结论。

### V5.2 plan 文档

不得回写历史 frozen byte 合同。

若已有冻结计划受 hash 约束：

- 原文件保持 immutable；
- 用新的 supersession note / status doc 表明路线被 V6 取代；
- 不直接修改旧 plan 破坏历史 hash。

## 5.2 文档自审

执行：

```text
git diff --check
markdown link/path sanity
failure ledger unique-ID audit
status ↔ experiments ↔ failure ledger consistency audit
frozen-plan hash regression
```

## 5.3 提交与 push

创建清晰提交：

```text
docs: close V5.2 direction and register WorldSim V6
```

push 到：

```text
origin/research/worldsim-v5.1-m1
```

禁止 force push。

输出：

```text
G1_DOCS_CLOSEOUT.json
```

记录 commit/tree/hash。

---

# 6. G2 — 自动把项目分支收敛进 main

用户要求在 V6 前将各种项目分支 merge 到 main。

**禁止直接在 main 上边合边修。**

## 6.1 创建 integration branch

从最新 `origin/main` 创建唯一：

```text
integration/pre-v6-<utc-stamp>
```

并建立 rollback tag/reference：

```text
pre-v6-main-<utc-stamp>
```

## 6.2 自动枚举 merge candidates

从 remote refs 动态枚举。

默认纳入：

```text
origin/research/*
origin/feature/*
origin/fix/*
```

同时自动检查其他 origin branches 是否包含 main 尚未拥有的独立提交。

排除：

```text
origin/main
origin/HEAD
明确 archive-only / backup-only / generated-only ref
```

**任何排除都必须写入 `BRANCH_MERGE_MATRIX.json` 并给 reason，不能静默跳过。**

若一个 branch 已是 main ancestor：

```text
already_merged = true
```

无需制造空 merge。

## 6.3 Merge 顺序

原则：

1. 祖先/旧路线先；
2. 独立 feature 次之；
3. 更新的研究路线后；
4. `research/worldsim-v5.1-m1` 最后合入，使最新研究状态文档成为默认权威；
5. 但不能用 “latest wins” 删除其他分支独有实现或历史事实。

## 6.4 Conflict 自动解决规则

禁止：

```text
git checkout --ours .
git checkout --theirs .
git merge -X theirs
git merge -X ours
```

作为全局解决方案。

按文件类别解决：

### A. `docs/RESEARCH_FAILURES.md`

- 采用**集合并集 + 时间序 append**；
- failure ID 不得消失；
- 同 ID 冲突时检查 canonical evidence；
- 最新状态可以写 `resolved/superseded`，但不能删除旧观察事实。

### B. `docs/RESEARCH_STATUS.md`

- 最新 V6 状态作为 active；
- 旧路线保留 terminal history；
- 不允许旧 branch 把 active route 倒写回 V3/V4/V5。

### C. `docs/EXPERIMENTS.md`

- 合并 task/run 事实；
- 不因 branch merge 重命名 canonical run；
- duplicate task 用 evidence/hash 判重。

### D. frozen configs / manifests

- 不把两个不同 bytes 的 frozen artifact 强制合成一个；
- 两个都保留；
- 必要时放回其版本命名空间；
- 任何 active pointer 明确指向最新有效版本。

### E. source code

判定优先级：

```text
canonical experiment dependency
> current tests/contracts
> active route implementation
> rejected-route convenience code
```

旧 rejected 实现可以保留为：

```text
legacy / archive / optional adapter
```

但不能因为 merge 自动重新接入默认执行链。

### F. tests

测试取并集。

若两个测试相互矛盾：

1. 查 failure ledger；
2. 查 frozen protocol；
3. 判断哪个测试属于过期合同；
4. 过期测试不能简单删除，需更新为验证 supersession 或 archive invariant。

## 6.5 Integration gate

所有 branch 合并完成后，必须执行：

```text
git diff --check
unit tests
protocol tests
config schema tests
direct CLI --help/import smoke
failure-ledger consistency
frozen asset/config identity regression
```

GPU-heavy test 可以按 capability 分类，但所有 CPU/contract test 必须完成。

生成：

```text
BRANCH_MERGE_MATRIX.json
MERGE_CONFLICT_RESOLUTION.jsonl
PRE_V6_MAIN_AUDIT.json
```

## 6.6 更新 main

仅在 integration gate 通过后：

```text
main ← integration/pre-v6-...
```

允许 merge commit / fast-forward integration branch。

禁止 force push。

push `main`。

---

# 7. G3 — 从最新 main 创建全新 V6 分支

G2 完成后：

```bash
git fetch origin
git checkout main
git pull --ff-only
```

确认：

```text
local main HEAD == origin/main HEAD
```

推荐新分支名：

```text
research/worldsim-v6-world-compiler
```

若已存在，不覆盖，自动递增：

```text
research/worldsim-v6-world-compiler-rNN
```

创建后立即 push upstream。

输出：

```text
V6_BRANCH_BOOTSTRAP.json
```

记录：

- parent main commit；
- branch name；
- branch initial commit；
- merged branch matrix hash；
- docs closeout commit；
- failure ledger hash。

此后 V6 的所有正式代码和实验只在 V6 branch 上进行。

---

# 8. V6 研究架构

## 8.1 C0 — SceneIR：统一显式世界中间表示

SceneIR 不是“另一个 checkpoint 格式”，而是将不同 reconstruction/generation frontend 解耦的研究基础设施。

最小 schema：

```text
SceneIR
├── CoordinateSystem
├── StaticWorld
│   ├── chunks
│   ├── primitives
│   ├── surfaces / collision proxy
│   └── map binding
├── Actors
│   ├── actor_id
│   ├── class
│   ├── canonical representation
│   ├── state trajectory
│   └── visibility
├── Sensors
│   ├── camera
│   ├── LiDAR
│   └── calibration
├── Provenance
├── Validity
├── Support
└── Episode
```

硬要求：

- 坐标系 typed；
- `T_dst_src` 命名；
- actor/static 明确；
- observed/reconstructed/generated/unknown provenance；
- source hash；
- generator/reconstructor version；
- episode seed；
- chunk hash；
- deterministic serializer。

## 8.2 C1 — Frontend Adapter

前端只是可替换的 compiler frontend：

```text
Optimization frontend:
StreetGS / CityGS / LiHi-GS / existing assets

Feed-forward frontend:
ReconDrive / TokenGS / other executable modern reconstructor

Conditional audit-only frontend:
Instant NuRec / currently resource-or-license constrained methods
```

V6 不要求所有 frontend 同时可运行。

至少需要：

```text
1 个现有 optimization frontend
+
1 个前馈 frontend
```

能够转成同一 SceneIR。

## 8.3 C2 — Support / Provenance Field

每个 world entity / primitive / chunk 必须知道：

```text
observed
reconstructed
generated
unknown
```

并记录 support：

```text
camera_view_support
temporal_support
lidar_support
track_support
map_support
generator_support
```

核心问题：

> 当 ego / actor / camera 偏离 logged support 时，系统能否在“渲染穿帮前”预测这一区域已经超出可靠支持域？

## 8.4 C3 — Factorized Task Validity

禁止一个总 confidence。

定义：

```text
q_photo
q_geometry
q_semantic
q_dynamics
```

任务使用资格：

```text
camera perception:
q_photo + q_semantic

collision / occupancy:
q_geometry + q_dynamics

planning regression:
q_geometry + q_semantic + q_dynamics

visual replay:
q_photo
```

V6 的重要研究问题：

> **同一个生成/重建世界，不同区域、不同任务是否应该有不同的可用性判定？**

## 8.5 C4 — Missing-World Proposal

生成器只负责：

```text
propose
```

不负责：

```text
declare truth
```

输入：

```text
observed world
+
target route
+
actor intervention
+
weather / text / condition
```

输出：

```text
candidate missing world / views / geometry / appearance
```

生成可以是：

- frozen diffusion/video model；
- view synthesis；
- feed-forward reconstruction extrapolation；
- retrieval/asset composition；
- hybrid proposal。

V6 不训练大型 foundation world model。

## 8.6 C5 — Independent Verifier

验证维度独立：

### Photo

- held-out reprojection；
- cross-view consistency；
- temporal perceptual consistency；
- generated-vs-observed seam。

### Geometry

- LiDAR depth；
- multi-view consistency；
- free-space / occupancy support；
- collision surface sanity。

### Semantic

- frozen perception teacher；
- map/actor consistency；
- deletion/insertion semantic preservation。

### Dynamics

- actor trajectory；
- canonical rigidity；
- flow/pose consistency；
- collision / kinematic sanity。

输出：

```text
VERIFIED
UNKNOWN
REJECTED
```

而不是把所有问题压成一个 scalar。

## 8.7 C6 — Bake

验证通过后，将 stochastic proposal 转为：

```text
immutable explicit chunk
```

固化：

- geometry；
- appearance；
- semantic state；
- actor state；
- provenance；
- factorized validity；
- generator seed；
- all hashes。

## 8.8 C7 — Deterministic Runtime

运行期必须：

```text
same world package
+
same episode
+
same action sequence
=
same world state sequence
```

GPU rasterizer 最末位 floating-point 可允许预先冻结 tolerance；但：

- world state；
- actor state；
- label；
- collision state；
- asset selection；

必须 deterministic。

---

# 9. 三种工业模式

V6 不是三个项目，而是一个 compiled world 的三个消费模式。

## 9.1 LogSim

目标：

> 原日志 case 精确复现、模型回归、问题拦截。

特点：

- observed support 优先；
- 最低 generation；
- 最高 deterministic；
- 保留原事件因果；
- 适合收费站、匝道、施工、遮挡等已知 case replay。

## 9.2 WorldSim

目标：

> 新路线、新 actor、新策略下的反事实泛化测试。

允许：

- ego route deviation；
- actor add/remove/move；
- traffic density change；
- occlusion change；
- scene extension；
- weather/time condition。

这是 V6 的主要方法价值区。

## 9.3 Policy Post-training

目标：

> 用 compiled verified world 生成可重复 episode，服务 IL/RL/回归。

只有 V6.3 通过后才启动。

不允许反过来把 RL reward 当作 V6 前面 world validity 的唯一证明。

---

# 10. Autoresearch Loop

这是 V6 的默认持续研究机制。

每个研究循环：

```text
OBSERVE
  ↓
DIAGNOSE
  ↓
FORM HYPOTHESIS
  ↓
NOVELTY + FAILURE-LEDGER GATE
  ↓
REGISTER MINIMAL EXPERIMENT
  ↓
EXECUTE
  ↓
AUDIT
  ↓
REFLECT
  ↓
PROMOTE / REJECT / MUTATE HYPOTHESIS
  ↓
NEXT LOOP
```

## 10.1 OBSERVE

只从合法 development evidence 读取：

- badcase；
- support-deviation curve；
- verifier failure；
- frontend mismatch；
- runtime failure；
- downstream failure。

不得用 confirmation/test 生成新假设。

## 10.2 DIAGNOSE

将问题分类：

```text
algorithm
data
evaluation
protocol
engineering
resource
governance
```

先决定：

> “这是方法失败，还是根本没执行到方法？”

## 10.3 FORM HYPOTHESIS

每个 hypothesis 必须回答：

```text
HYPOTHESIS_ID
problem
mechanism
why existing method does not already solve it
which failure-ledger fact it addresses
minimum experiment
expected direction
falsification condition
compute budget
```

禁止：

```text
“再加一个模块试试”
```

## 10.4 NOVELTY GATE

在编码前检查：

- 已有 repo；
- 项目 docs/paper notes；
- 官方 paper/code；
- 当前最新相关工作。

如果核心机制与已公开工作直接重合：

```text
reject before implementation
```

除非 V6 的新增点在：

```text
verification
task-conditioned validity
provenance
deterministic bake
closed-loop functional utility
```

形成独立问题。

## 10.5 MINIMAL EXPERIMENT

每轮只允许一个 primary hypothesis。

例如：

```text
Support predictor
vs
new validity model
vs
new verifier
vs
new bake strategy
```

不能同轮全部变化。

## 10.6 AUDIT

正式结果必须写：

```text
resolved config
source commit
inputs
denominator
per-case metrics
aggregate
resource
terminal
failure_ledger_refs
failure_ledger_delta
```

## 10.7 REFLECT

Agent 每轮自动写：

```text
docs/autoresearch/worldsim_v6/REFLECTIONS.jsonl
```

每条：

```json
{
  "hypothesis_id": "...",
  "what_worked": [],
  "what_failed": [],
  "strongest_evidence": [],
  "confounders": [],
  "new_information": [],
  "old_assumption_invalidated": [],
  "next_hypotheses": [],
  "forbidden_retries": []
}
```

注意：

`REFLECTIONS.jsonl` 是过程记忆，不是 failure ledger。

真正可复用失败必须同步追加进：

```text
docs/RESEARCH_FAILURES.md
```

## 10.8 自进化允许修改什么

可以自动改：

- hypothesis；
- method architecture；
- frontend；
- verifier；
- development metric diagnostics；
- implementation；
- compute schedule；
- feature set；
- ablation；
- research plan revision。

但每次修改必须产生：

```text
new hypothesis ID
new config
new run
new commit
```

不能覆盖旧 candidate。

## 10.9 自进化不能修改什么

在结果可见后不能修改当前 experiment 的：

- primary endpoint；
- confirmation cases；
- candidate identity；
- frozen threshold；
- split；
- comparator；
- success criterion。

若这些确实需要变：

```text
close current hypothesis
→ register new hypothesis
→ new development protocol
```

---

# 11. 自动研究状态机

推荐：

```text
docs/autoresearch/worldsim_v6/AUTORESEARCH_STATE.json
```

字段：

```text
route
iteration
active_hypothesis
active_task
candidate
frontend
development_set_hash
confirmation_set_hash
test_lock
failure_ledger_hash
last_reflection
next_actions
budget
```

状态只能是：

```text
idle
researching
blocked_engineering
blocked_resource
candidate_frozen
confirmation_consumed
confirmed
rejected
closed
```

任何进程重启后：

```text
read state
→ verify Git/source/run identities
→ resume from next legal action
```

禁止仅凭聊天上下文猜当前状态。

---

# 12. 3090 资源策略

## 12.1 不把 3090 不可运行误写成算法失败

前端分三类：

```text
EXECUTABLE
ADAPTABLE
AUDIT_ONLY
```

如果某 upstream 明确需要 >24GB：

- 不反复 OOM；
- 不偷偷降低正式规格继续称 faithful；
- 记录 resource blocked；
- 切换下一合法 frontend。

## 12.2 GPU 串行

默认：

```text
one heavy GPU process at a time
```

DINO / SAM / renderer / feed-forward reconstruction 不同重型 stage 不同进程串行释放。

## 12.3 Resource contract

每个 formal run 记录：

- GPU name；
- total/free；
- process peak；
- PyTorch peak；
- cgroup memory.max/current/events；
- disk free；
- duration。

资源 stop：

```text
second OOM on same unchanged method
persistent cgroup pressure
disk unsafe
GPU health anomaly
```

工程 blocked 后：

```text
preserve run
→ new recovery hypothesis/run
```

---

# 13. V6.0 — World Compiler Feasibility Gate

目标：

> 不做生成模型创新，先证明“多前端 → SceneIR → support deviation → deterministic replay”成立。

---

## R0 — Repo convergence

对应 G0–G3。

通过：

```text
latest docs pushed
main converged
new V6 branch exists
tests pass
failure ledger consistent
```

---

## R1 — Capability + Frontend Audit

Task：

```text
WS-V6-R1-FRONTEND-CAPABILITY-01
```

自动审计：

### Existing optimization assets

至少：

- StreetGS；
- 已有 AD-GS；
- 若仓库/机器已有 CityGS/LiHi-GS 资产则登记。

### Feed-forward candidates

优先级：

```text
1. ReconDrive
2. TokenGS
3. 其他公开且 3090 可执行的 driving feed-forward reconstructor
```

Instant NuRec：

```text
conditional / audit-only first
```

只有本机资源、权重、输入/license 合同真的满足才执行，禁止重复历史“前置不足却假装 baseline 跑过”的坑。

输出：

```text
FRONTEND_CAPABILITY_MATRIX.json
```

字段：

```text
paper
official_source
commit
license
weights
input_schema
output_schema
gpu_requirement
local_status
adapter_cost
selected_role
```

晋级：

```text
>=1 optimization frontend executable
AND
>=1 feed-forward frontend executable/adaptable
```

否则自动继续搜索一个新的公开 feed-forward 候选，但最多尝试 3 个完整环境。

---

## R2 — SceneIR v0

Task：

```text
WS-V6-R2-SCENEIR-V0-01
```

先实现：

```text
StreetGS/existing → SceneIR
feed-forward frontend → SceneIR
```

必须做到：

- typed frame；
- static/dynamic split；
- actor trajectories；
- camera model；
- primitive/chunk provenance；
- deterministic serialization；
- fresh-process reload；
- content hash；
- render adapter。

SceneIR round trip：

```text
frontend native
→ SceneIR
→ frontend-compatible runtime view
```

在已支持视图上必须满足冻结的 representation-equivalence tolerance。

这一步只证明：

```text
representation interface works
```

不能写：

```text
new reconstruction method
```

---

## R3 — Support Deviation Benchmark

Task：

```text
WS-V6-R3-SUPPORT-DEVIATION-01
```

同一 scene 生成：

```text
logged trajectory
lateral 0.5m
lateral 1m
lateral 2m
lateral 3m
lateral 5m

short forward extension
route turn / branch if geometry permits

actor remove
actor translate
actor trajectory perturbation
```

不要把这些固定数写进 universal method config；它们是 benchmark profile，可按传感器/道路尺度配置化。

测量：

```text
deviation
vs
global RGB error
static error
actor error
boundary
geometry
temporal
support
```

目标：

> 验证 support signal 能否比纯图像质量更早预测 out-of-support failure。

最小晋级门：

- 至少 2 scenes；
- 至少 2 frontends；
- deviation 增加时 failure 具有可测结构；
- support score 与 downstream error 存在稳定排序；
- 不能由简单“离原相机距离”完全解释全部增益。

若纯距离 baseline 已足够：

```text
reject learned support model
```

转向 verifier / provenance，而不是强行训练网络。

---

## R4 — Deterministic Runtime v0

Task：

```text
WS-V6-R4-DETERMINISTIC-RUNTIME-01
```

同一个 SceneIR package：

```text
fresh process × 3
```

相同：

- seed；
- episode；
- action；
- sensor；
- actor initial state。

要求：

```text
world state exact
labels exact
asset/chunk selection exact
actor trajectory exact
```

RGB rasterization：

- 优先 exact；
- 若底层 CUDA 非 bit-exact，必须先冻结数值 tolerance；
- 不得让视觉末位浮点漂移改变 collision/label/world-state。

V6.0 Go：

```text
R1 + R2 + R3 + R4
```

全部成立。

---

# 14. V6.1 — Provenance + Task Validity

核心论文候选开始于这里。

## R5 — Provenance Field

Task：

```text
WS-V6-R5-PROVENANCE-01
```

为每个 chunk/actor/primitive 建立：

```text
source_type
sensor_support
time_support
view_support
reconstruction_source
generation_source
```

ground truth 边界：

```text
observed != reconstructed != generated
```

---

## R6 — Factorized Validity

Task：

```text
WS-V6-R6-FACTORIZED-VALIDITY-01
```

基线：

```text
V0 distance-to-log
V1 view-count
V2 reconstruction residual
V3 single scalar confidence
```

候选：

```text
q_photo
q_geometry
q_semantic
q_dynamics
```

设计必须证明：

> 因子化 validity 比单 scalar 更适合多任务 simulator qualification。

例如：

```text
far vegetation:
photo-valid
geometry-unknown

generated road continuation:
photo-valid maybe
geometry-valid only if LiDAR/multiview verification passes

new actor:
semantic-valid
dynamic-valid only if trajectory/interaction verifier passes
```

### 指标

- calibration；
- AUROC/AUPRC 仅作辅助；
- selective risk；
- coverage；
- task-conditioned failure prediction；
- ECE/Brier；
- worst-case false-safe rate。

主端点不能只优化 coverage。

特别报告：

```text
false-safe
```

因为工业仿真最危险的是：

> 系统说“可用”，实际上该区域对任务不可靠。

---

# 15. V6.2 — Generate → Verify → Bake

这是 V6 主方法阶段。

## R7 — Oracle Missing-World Extension

先不用生成模型。

构造：

```text
真实 held-out observation / pseudo-hole
```

模拟：

```text
missing route support
missing side view
disocclusion
actor removal hole
```

验证：

> 如果缺失内容本身是 oracle，SceneIR + verifier + bake 是否能恢复并正确判断？

如果 oracle 都无法提升 closed-loop usable region：

```text
compiler representation/metric 有问题
```

不要训练 generator。

---

## R8 — Frozen Proposal Generator

只使用 frozen generator。

候选必须满足：

- 公开合法权重；
- 3090 可运行或可离线分块；
- 不训练 foundation model；
- generator 输出永远标记为 proposal。

可以自动在 2–3 个 generator 中做 capability study，但只选择一个进入 formal method。

---

## R9 — Independent Verifier Arms

每项独立：

```text
P0 no verification
P1 photo
P2 geometry
P3 semantic
P4 dynamics
```

先证明单臂作用。

禁止第一轮直接：

```text
photo + depth + semantic + flow + occupancy + LLM
```

---

## R10 — Factorized Verification

只融合已经在 matched development 上单独有效的 verifier。

输出：

```text
ACCEPT
ABSTAIN
REJECT
```

以及 factorized validity。

主指标：

```text
usable verified world area
verified trajectory length
false-safe rate
photo/geometry/semantic/dynamics calibration
non-target regression
```

不是单纯：

```text
PSNR +0.x
```

---

## R11 — Bake

将 accepted proposal 转为：

```text
explicit static chunk
explicit actor asset
explicit trajectory
explicit provenance
explicit validity
```

要求：

```text
no online generator dependency at runtime
```

---

# 16. V6.3 — LogSim + WorldSim Functional Validation

## R12 — LogSim

在 observed support 内：

- 原轨迹；
- 原 actor；
- 原事件；
- 原 sensor calibration。

目标：

```text
same compiled world
→ deterministic replay
→ regression testing
```

指标：

- sensor similarity；
- perception output consistency；
- trajectory replay consistency；
- collision/semantic label consistency；
- repeated-run exactness。

---

## R13 — WorldSim

在 logged support 外：

```text
route deviation
new route branch
actor add/remove
actor trajectory modification
new traffic density
```

比较：

```text
native reconstruction only
generator-only
reconstruction + naive generation
V6 generate-verify-bake
```

指标：

```text
usable route length
world area
false-safe
perception failure
planning completion
collision correctness
temporal consistency
```

---

# 17. V6.4 — 可选 Policy Post-training

只有 V6.3 明确通过后才解锁。

不是 V6 成立的前置。

最小实验：

```text
Real-only
vs
Real + naive synthetic
vs
Real + V6 verified compiled episodes
```

可选任务：

- imitation learning；
- small planner policy；
- RL fine-tuning；
- regression test set。

关注：

```text
collision
route completion
stuck
comfort
generalization
```

禁止：

```text
只报 simulator reward
```

至少需要独立 held-out scenario outcome。

---

# 18. Autoresearch 的 hypothesis families

Agent 初始可从以下方向探索，但不是固定执行清单。

## H-A Support Modeling

问题：

> 能否预测“什么时候已超出日志/重建支持域”？

候选：

- visibility support；
- ray coverage；
- temporal support；
- LiDAR support；
- reconstruction uncertainty；
- frontend disagreement。

## H-B Factorized Validity

问题：

> 为什么一个 scalar confidence 不足以支撑多任务仿真？

候选：

- per-task calibration；
- conformal acceptance；
- abstention；
- cross-factor disagreement。

## H-C Generate-Verify-Bake

问题：

> generator 生成的内容怎样从 stochastic media 变成 persistent 3D/4D world？

候选：

- multi-view proposal；
- world-space canonicalization；
- consistency optimization；
- confidence-aware baking；
- chunk provenance。

## H-D Cross-Frontend Transfer

问题：

> V6 机制是否只是 StreetGS 特有？

至少一项关键方法要验证：

```text
optimization frontend
→ feed-forward frontend
```

仍成立。

这是避免“基座 patch project”的重要证据。

## H-E Closed-loop Utility

问题：

> verification 真能降低闭环 false-safe，而不只是改善画面？

---

# 19. 自主 fallback 树

## Frontend A 失败

```text
engineering blocked
→ 一次最小合法 recovery
→ 仍 blocked
→ 切换 frontend B
```

不是反复修 upstream 一周。

## Feed-forward 全部 3090 blocked

V6 仍可继续：

```text
optimization frontend
+
frozen existing explicit world
+
SceneIR/support/verify/bake
```

同时将 feed-forward claim 降级。

不能为了“前馈”标签偷偷改硬件预算。

## Generator 不可用

先跑：

```text
oracle / pseudo-hole / retrieval proposal
```

验证 compiler 方法本身。

## Geometry truth 不足

写：

```text
geometry_unknown
```

不能用 model self-render depth 当 GT。

## Confirmation 失败

当前 candidate rejected。

自动回到：

```text
new hypothesis
```

但 confirmation 数据不能再用于该 family 调参。

---

# 20. Confirmation / Validation / Test 纪律

V6 开始时重新建立：

```text
Development
→ Candidate Freeze
→ Confirmation
→ Freeze
→ Exact-once Test
```

旧 V5/V5.1/V5.2 test 不能自动当 V6 test。

## Development

Agent 可反复读取。

## Confirmation

候选冻结后才能读取。

只允许：

```text
one-shot
```

不用于反思生成同 candidate 的新参数。

## Test

exact-once ledger：

```text
attempt created before quality read
```

失败也消费 attempt。

Agent 无需人工审批即可执行，但必须遵守 frozen protocol。

---

# 21. 论文级晋级门

V6 只有同时满足以下四层才值得进入完整论文：

## Gate 1 — Representation independence

至少：

```text
2 frontends
```

可以进入同一 SceneIR。

## Gate 2 — Support / validity

比简单：

```text
distance / view count / residual scalar
```

更稳定预测 out-of-support false-safe。

## Gate 3 — Generate-Verify-Bake

相对：

```text
generator-only
naive bake
```

提高 verified usable region，同时降低 false-safe。

## Gate 4 — Functional utility

至少一个 downstream：

```text
LogSim regression
WorldSim planning/perception
IL/RL post-training
```

出现任务级收益。

若只有 PSNR：

```text
not paper-ready
```

---

# 22. 关键指标

## Reconstruction / rendering

- PSNR；
- SSIM；
- LPIPS；
- actor crop；
- boundary；
- temporal perceptual。

## Geometry

- LiDAR depth；
- Chamfer / point-to-surface（有合法真值时）；
- free-space violation；
- collision surface error。

## Support

- error-vs-deviation；
- support calibration；
- out-of-support detection；
- false-safe rate。

## Validity

- Brier；
- ECE；
- coverage；
- selective risk；
- per-task acceptance precision；
- per-task false-safe。

## World expansion

- verified route length；
- verified world area；
- accepted generated chunks；
- abstention rate；
- usable episode yield。

## Determinism

- state hash；
- actor trajectory hash；
- labels hash；
- package hash；
- fresh-process replay。

## Functional

- perception degradation；
- collision correctness；
- planner success；
- route completion；
- stuck；
- comfort；
- RL/IL downstream outcome。

---

# 23. Run Contract

每个 formal run 使用唯一 ID：

```text
<UTC>__<task-slug>__s<seed>__r<nnn>
```

不要依赖固定绝对 run root。

最小文件：

```text
resolved.yaml
manifest.json
fingerprint.json
status.json
events.jsonl
metrics.jsonl
summary.json
stdout.log
stderr.log
```

质量 run：

```text
CASE_INPUTS.jsonl
CASE_OUTPUTS.jsonl
CASE_DELTAS.jsonl
PANEL_REGISTRY.jsonl
RESOURCE.json
```

每个 run：

```text
failure_ledger_refs
failure_ledger_delta
```

必须存在。

---

# 24. Git / 实验提交纪律

## 24.1 Prereg commit

正式质量实验前：

```text
code + config + tests committed
```

run 必须记录 source commit。

## 24.2 Result commit

实验完成：

```text
docs/status/experiments/failure ledger
```

与代码更改分开提交更优。

## 24.3 Reject 也提交

rejected route 仍需要：

```text
result closeout commit
```

这样 Agent 下一轮不会重复探索。

## 24.4 定期 push

每完成一个稳定 research iteration：

```text
push V6 branch
```

无需等待用户。

禁止 push：

- 大 checkpoint；
- raw dataset；
- run outputs；
- local capability manifest；
- secret/token。

---

# 25. Failure Ledger 强制继承项

每个 V6 task 根据范围引用既有失败。

至少持续继承以下类型：

## Research / attribution

- `V52-F01`
- `V52-F02`
- `PIVOT-F02`
- `PIVOT-F03`
- `PIVOT-F04`

## Truth / evaluation

- `V2-F09`
- `V3-F10`
- `V3-F11`
- `V3-F17`
- `V3-F41`
- `V4-F42`
- `V5-F35`
- `V5-F49`

## Determinism / protocol

- `V4-F11`
- `V4-F13`
- `V4-F48`
- `V51-F14`
- `V51-F65`

## Resource / runtime

- `N1-F24`
- `PIVOT-F05`
- `PIVOT-F06`
- `PIVOT-F07`
- `PIVOT-F12`
- `PIVOT-F20`
- `PIVOT-F21`
- `PIVOT-F22`
- `PIVOT-F29`

## Path / asset / atomicity

- `PIVOT-F30`
- `V51-F13`
- `V51-F50`

新 V6 failure 使用：

```text
V6-FNN
```

在**现有 `docs/RESEARCH_FAILURES.md`** 追加。

---

# 26. 明确禁止重开的旧坑

V6 Agent 不得把下列路线包装成“新研究”：

```text
raw Gaussian KNN ownership
graph Laplacian
SAI3D-style growing
simple voxel super-primitive propagation
BKI spatial completion
Trace3D faithful alpha path
StreetGS-specific TrackBayes-only main story
cut-in mining as thesis
one scalar evidence confidence
learned occupancy as its own GT
Telea/generated image as geometry truth
PSNR-only worldsim success
exact package = streaming
checkpoint smaller = runtime faster
```

---

# 27. 反思触发条件

自动 reflection 不需要等 milestone。

以下任一发生即反思：

```text
candidate rejected
same failure repeats twice
unexpected cross-scene sign flip
resource blocked
frontend blocked
false-safe increases
coverage collapses
confirmation failure
new paper invalidates novelty
```

Reflection 必须问：

1. 当前失败是 evidence 不够，还是方法不对？
2. 当前 metric 是否真对应 industrial goal？
3. 是否又开始修某个基座而不是解决 compiler 问题？
4. 新假设是否能跨 frontend？
5. 是否存在更简单 baseline 已解释结果？
6. 是否重复旧 failure ledger 禁止项？
7. 下一轮最小可证伪实验是什么？

---

# 28. 自主停止条件

## 成功停止

满足论文级 Gate 1–4：

```text
candidate = V6 paper candidate
```

然后自动：

- freeze；
- exact-one-shot test；
- generate paper tables；
- generate failure appendix；
- close research loop。

## 研究停止 / pivot

若：

```text
3 个连续独立 hypothesis
```

都指向同一未解除硬瓶颈，且没有新增 evidence source 或可行 mechanism：

```text
close current family
→ reflect
→ pivot to next V6 hypothesis family
```

不能第 4 次继续换 threshold。

## 工程停止

仅当：

- 无合法公开权重且无替代；
- license/gated access 需要真实用户操作；
- 硬件物理不可满足且无替代 frontend；
- 数据完全缺失且无法用现有公开/本地数据回答问题；
- repo remote 权限确实拒绝 push。

此时：

```text
写 blocked
保存状态
自动继续所有不依赖该前置的 V6 分支
```

只有整个 V6 所有合法分支都被同一外部条件阻塞时才整体停止。

---

# 29. 第一轮自动执行顺序

Codex 接到本计划后不再重新向用户询问“是否继续”。

严格按：

```text
G0  repo audit
↓
G1  latest research branch docs closeout + push
↓
G2  integrate project branches → main
↓
G3  create/push fresh V6 branch
↓
R1  frontend capability audit
↓
R2  SceneIR v0
↓
R3  support deviation benchmark
↓
R4  deterministic runtime
↓
V6.0 decision
```

若 V6.0 PASS：

```text
R5 provenance
↓
R6 factorized validity
↓
autoresearch loop
↓
R7 oracle missing-world extension
↓
R8 proposal generator
↓
R9 independent verifier arms
↓
R10 factorized verification
↓
R11 bake
↓
R12 LogSim
↓
R13 WorldSim
↓
optional policy post-training
```

---

# 30. V6 第一轮必须交付的文件

在 V6 branch 中：

```text
docs/WORLDSIM_V6_AUTORESEARCH_PLAN.md
docs/autoresearch/worldsim_v6/AUTORESEARCH_STATE.json
docs/autoresearch/worldsim_v6/HYPOTHESES.jsonl
docs/autoresearch/worldsim_v6/REFLECTIONS.jsonl

configs/worldsim_v6/
scripts/worldsim_v6/
motion_proj/worldsim_v6/
tests/worldsim_v6/
```

注意：

`REFLECTIONS/HYPOTHESES` 是研究过程状态，不是第二套 failure ledger。

正式失败仍只写：

```text
docs/RESEARCH_FAILURES.md
```

运行输出不提交仓库，只保存 manifest reference。

---

# 31. 计划允许自我修订

Agent 可以生成：

```text
WORLDSIM_V6_AUTORESEARCH_PLAN_REVISION_<N>.md
```

但触发条件必须是：

```text
new empirical evidence
new upstream work
new hard resource constraint
novelty conflict
old assumption falsified
```

每次 revision 必须：

- 引用 predecessor hash；
- 列出 changed assumptions；
- 列出 retained gates；
- 列出 newly superseded task；
- 不删除历史 plan；
- 不改变已经消费 confirmation/test 的解释。

如果只是一个实验失败：

```text
不需要改整个 plan
```

优先写 hypothesis/reflection/failure entry。

---

# 32. 对 Codex 的最终执行指令

你现在是 WorldSim V6 的 Autonomous Research Agent。

你的目标不是“完成所有 checklist”，而是：

> **在单卡 RTX 3090 和已有公开/本地资产约束下，持续寻找并验证一个能够把有限驾驶日志编译为可信、可扩展、可确定性回放显式世界的核心方法，并最终形成具有 CVPR/ICCV/ECCV 主会潜力和自动驾驶闭环仿真工业价值的可证伪贡献。**

执行时：

1. 先完成 Git/doc/main/V6 branch 收敛；
2. 不硬编码机器路径；
3. 读 failure ledger 再做每个新实验；
4. 先最小实验，再扩量；
5. 方法失败继续下一 hypothesis，不停下来问用户；
6. 工程失败只做最小合法恢复；
7. 自主反思并维护研究记忆；
8. 新证据允许路线自进化；
9. 不重复旧坑；
10. 不通过调 gate、删 case、泄漏 confirmation 来制造成功；
11. 任何 V6 贡献必须尽量跨至少两种 frontend；
12. 每一步都问：**这是不是又变成了“修某个 Gaussian 基座”？**
13. 如果是，立即把问题重新提升到 SceneIR / support / provenance / validity / verify / bake / closed-loop 层。
14. 达到论文级 gate 后自动 freeze、确认、exact-once test 和 closeout。
15. 在达到真实外部权限/许可/硬件不可满足之前，不需要人工审批。

---

# 33. 一句话研究主张

V6 目标不是：

> “我们让 StreetGS/CityGS/某个 Gaussian baseline 更好。”

而是：

> **We compile finite real driving logs and stochastic generative proposals into persistent, explicit, task-verifiable 4D worlds that can be safely reused for deterministic closed-loop simulation and policy improvement.**

中文：

> **把有限真实驾驶日志和随机生成提案，编译成持久、显式、按任务可验证、能够确定性闭环复现的四维驾驶世界。**

这才是 V6 的北极星。
