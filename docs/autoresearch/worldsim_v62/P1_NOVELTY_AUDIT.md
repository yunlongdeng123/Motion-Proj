# WorldSim V6.2 P1 新颖性与迁移审计

- Task：`WS-V62-P1-NOVELTY-AUDIT-01`
- Hypothesis：`WS-V62-H-P1-001`
- 状态：`done_no_direct_overlap`
- 日期：2026-08-24
- 范围：一手论文、会议页面、作者项目页或官方仓库；未运行模型、未读取 confirmation/test。

## 结论

截至本次检索，没有发现单一工作同时覆盖：

```text
real observed FREE/OCC as immutable method constraints
+ frozen learned Occupancy as a defeasible prior
+ selective FREE/OCC/UNKNOWN output
+ proposal bake or collision-state compilation
+ independent world-simulation false-safe evaluation
```

因此 CPSC 可以进入实现，但贡献必须收窄为：

> 在 driving world compiler 中，把硬观测证据、可推翻 learned prior、选择性 UNKNOWN、集合式校准和
> proposal/collision-state bake 组织成一个端到端可验证的 Physical State Completion 接口。

本结论是基于已检索一手来源的未发现直接重合，不是对所有未公开工作的证明。

## 组件级审计

| 工作 | 已有能力 | CPSC 不可单独主张 | 与 CPSC 的剩余差异 |
|---|---|---|---|
| [ReliOcc, IJCAI 2025](https://www.ijcai.org/proceedings/2025/220) | hybrid uncertainty learning、离线校准、corruption/OOD reliability | uncertainty head、可靠性校准 | 不把真实 FREE/OCC 设为不可违反的 compiler 约束，不做 proposal bake/false-safe world gate |
| [OCCUQ, ICRA 2025](https://arxiv.org/abs/2503.10605) | 单前向 aleatoric/epistemic uncertainty 与 feature-level GMM | max-confidence/entropy/UQ filtering | 输出仍是 perception Occupancy；没有硬证据优先级或 UNKNOWN compiler authority |
| [alpha-OCC, TMLR 2026](https://openreview.net/pdf/e3d798e188fd940a22ea346ada35677284bf0a0d.pdf) | hierarchical conformal prediction、类别不平衡下的集合覆盖 | conformal prediction set | 目标是 voxel semantic coverage，不是 proposal-level false-safe-controlled compilation |
| [EvOcc, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf) | FREE/occupied/uncertain、冲突测量、evidential mapping 与训练 loss | 三态证据、UNKNOWN、冲突建模 | 没有 frozen learned prior 的可推翻权威、proposal bake 或 world-simulation false-safe gate |
| [QueryOcc, CVPR 2026](https://research.zenseact.com/publications/queryocc/) | 相邻帧射线上的连续 4D positive/negative query supervision | 4D query、ray-before-hit FREE query | 学习通用 occupancy；不做 hard projection、selective compiler 或独立 bake 评测 |
| [SUG-Occ](https://arxiv.org/abs/2601.11396) | semantic/uncertainty guided sparse representation、coarse-to-fine completion | sparse active set、uncertainty-guided sparsification | 优化 perception accuracy/efficiency，不提供观测硬约束与 world interface |
| [OccAny, CVPR 2026](https://github.com/valeoai/OccAny) | 非固定 camera rig、跨域 generalized occupancy | generalized/unconstrained-camera occupancy | 不解决 hard-evidence authority；V6.2 也不以换 backend 追求 gate |
| [GaussianFlowOcc, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Boeder_GaussianFlowOcc_Sparse_and_Weakly_Supervised_Occupancy_Estimation_using_Gaussian_Splatting_ICCV_2025_paper.pdf) | Gaussian sparse occupancy 与 temporal flow | Gaussian/temporal occupancy | 没有 selective physical-state compilation 或 false-safe bake gate |
| [DIO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Diehl_DIO_Decomposable_Implicit_4D_Occupancy-Flow_World_Model_CVPR_2025_paper.pdf) | sparse LiDAR conditioned 4D completion/forecasting；留出部分 observation 评测补全 | withheld-evidence completion、4D instance completion | 不把输入 FREE/OCC 编译成 immutable constraints，也不输出 calibrated UNKNOWN/bake authority |
| [Differentiable Projection](https://arxiv.org/abs/2111.10785) | 轻量可微 projection layer | 可微投影本身 | CPSC 只迁移通用方法学到 separable tri-state evidence constraints |
| [HardNet](https://arxiv.org/abs/2410.10807) | input-dependent affine/convex hard constraints by construction | hard-constrained neural layer | 不涉及 occupancy evidence、world compilation 或 false-safe semantics |
| [PCFM, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/eacce1f7d11e2c3a7568468e0eff5d33-Abstract-Conference.html) | 对 pretrained generative flow 的推理时非线性硬约束 | physics constraint guidance | PDE/generative sampling setting，不是 sparse occupancy evidence compiler |
| [MultiSafe](https://cmu-intentlab.github.io/multisafe/) | partially observable safety constraints 与 conformal false-safe control | conformal false-safe 控制本身 | controller-level latent safety filter；没有 driving 3D state bake 与 observed FREE/OCC projection |
| [World-model simulator admissibility](https://arxiv.org/abs/2607.07196) | world-model verdict 前的 admissibility/VV&A 分层 | “世界模型需先验证”这一 framing | 评测框架，不是 CPSC 的物理状态补全方法 |

## 贡献边界

### 可以继续验证的主张

1. Hard evidence 和 learned Occupancy prior 的权威不对称：前者不可违反，后者可被推翻。
2. 约束内生的 selective physical state，而不是 dense argmax 后的安全 filter。
3. 只把 calibrated singleton FREE/OCC 编译为 SceneIR/collision/proposal authority，其余保留 UNKNOWN。
4. 方法侧证据与独立 false-safe 评测分离，并在 world-simulation bake 产出率下进行 anti-trivial evaluation。

### 禁止作为单独贡献

- uncertainty/evidential head；
- FREE/OCC/UNKNOWN 三态；
- 4D query 或 ray supervision；
- counterfactual evidence dropout；
- differentiable projection；
- conformal prediction；
- Gaussian/sparse/temporal occupancy；
- world-model safety/admissibility framing。

## 对当前项目的最小迁移

### 复用

- `motion_proj.worldsim_v61.occupancy.VoxelGridSpec`、坐标变换、ray carving 与 actor identity/lifecycle 语义；
- V6.1 frozen `O_method`、oracle/R10 comparator 与 IR-WM capability/sidecar 资产；
- legacy28 case 类型和 false-safe evaluator，但仅在 method output 固定后读取 O_eval。

### 不复用

- V6.1 runner 中面向哈希/manifest 的重审计路径；
- predicted argmax 到 OCCUPIED 的直接映射；
- observed-FREE post-hoc 全 veto；
- 通用大规模 convex solver、dense 3D UNet、SparseConv/Transformer 第一版。

### P3 operator-first 设计

输入为 `[N,3]` logits/probability 与 query-wise evidence masks。约束优先级：

```text
contradiction
  > observed FREE / observed OCC
  > outside actor lifecycle/envelope
  > soft prior
```

第一版使用 closed-form `torch.where`/simplex renormalization：hard FREE/OCC/UNKNOWN 直接 one-hot；lifecycle 外 OCC
概率置零并把不确定质量保留给 UNKNOWN；无约束 query 保留 prior/residual 的可微概率。因为这些约束逐 query 可分，通用
凸优化器不会增加科学能力。只验证一组 synthetic tensor 和一个真实 `O_method` fixture。

P3 通过后，P2 dataset 才围绕相同 contract 生成 query/evidence/dropout/target split，避免先造数据再反向迁就算子。

## 裁决

`novelty_gate=PASS_WITH_NARROWED_CONTRIBUTION`。下一任务：`WS-V62-P3-FEASIBILITY-PROJECTION-01`。
