# 历史原始记录 030

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V7-F01 — AV2 sample split 路径错误且 AutoDL 小文件吞吐不足

- symptom：参考命令把 UUID `00a6ffc1-6ce9-3bc3-a060-6006e9893a1a` 放在 `sensor/val`，但 S3 listing 证实该
  UUID 属于 `sensor/train`；修正 split 后该 log 为 `1,263,003,345` bytes / `3035` objects。AutoDL 在
  `--numworkers 16` 下约 68 秒仅取得 `41,437,778` bytes，约 `0.58 MiB/s`。
- root cause：第一层是静态 split/UUID 不匹配；第二层是当前 AutoDL 到公开 S3 的多小对象链路吞吐，不是磁盘、GPU、
  模型或科学方法失败。测速未读取方法输出，也未选择数据质量。
- response：先列出官方 val 全 150 UUID，只按排序位置 every-fifth 冻结 30 logs；再用同版 `s5cmd v2.3.0` 在本地
  D 盘测速约 `2.86 MiB/s`，按用户预案切换为本地单进程串行下载。远端 40 MiB 残片列入清理 manifest 并删除。
- impact：status=`resolved_by_local_staging_route`；不形成 scientific read，不改变 20 quantitative + 10 qualitative
frozen cohort，不允许 AV2 fine-tune/calibration/threshold/failed-scene selection。

后续已使用 `V7-F02`--`V7-F08`；当前下一可用编号：`V7-F09`。

## V6.7 P269/P270 latest failures（2026-08-31）

### V67-F194 — P269 训练完成后的 final-reliability shortfall 轴切片错误

- run：`run://worldsim_v67/WS-V67-P269-RELIABILITY-FLOOR-GROUP-DUAL-01/20260831T145000Z__reliability-floor-group-dual-s0-r1`；
- symptom：12,000-step GPU training 完成后，`_utility` 将形状 `(group,price,member,horizon)` 的 final reliability 误切为
  `(group,price,horizon)`，与 alpha utility 的 `(group,price,member)` 无法广播，正式 source/P183/P201 summary 未生成；
- root cause：NumPy 从 trailing axes 对齐广播；原 `probability[:,ai,li,:,-1]` 的 `-1` 落在 member 轴而非 horizon 轴；
- response：参考 NumPy 官方 broadcasting 规则与 ICLR 2022 einops 的显式轴思想，唯一修复为
  `probability[:,ai,li,:,:, -1]`。不引入依赖、不增加 gate/test，不改训练合同；commit=`020726a`；
- impact：r1=`implementation_failed_after_training/no_verdict`，不解读训练 loss 为科学支持；同合同 r2 的 P201
  budget-fraction MAE=`.012541`、regret=`4.19e-6`，2/2 通过，状态=`resolved_by_minimal_axis_fix`。

### V67-F195 — P270 两次 pre-science 启动环境错误

- r1 在 import 阶段因直接脚本入口未含 repo root，报 `ModuleNotFoundError: scripts`；r2 加 `PYTHONPATH=.` 后因误传
  repo-local `runs`，在首个冻结 artifact load 前报 `FileNotFoundError`；
- exposure：两次都未完成 frozen artifact load、teacher target、optimizer step 或 metric read，不构成科学 trial；
- resolution：r3 只修启动命令为 `PYTHONPATH=.` 与 absolute runs-root `/root/autodl-tmp/runs`；实验配置、seed、模型、
  teacher、门和 claim 不变；
- impact：canonical scientific run 迁移到 `20260831T151000Z__tail-cvar-equivariant-allocator-s0-r3`；该 run P201 budget
  MAE=`.011861`、regret=`9.24e-5`，2/2 通过，状态=`resolved_before_scientific_trial`。

下一可用编号：`V67-F196`。

### P269 r2 / P270 r3 / P271 r1 outcome note — 无新增 failure

- P269 最小轴修复后 2/2；P270 fixed64 tail-CVaR allocator 2/2；P271 held-out sizes 48/96 亦 2/2；
- 三者均保持预冻结模型、条件轴、训练步数与 decision gates，没有以重复 smoke/regression 或阈值扫描补救；
- P272 现只把冻结 P271 primal 连接到 attainable-budget dual，当前运行中；下一 failure id 仍为 `V67-F196`。

### P272/P273 outcome note — 无新增 failure

- P272 在 P201 held-out cardinality 上 2/2，随后冻结 primal/dual；P273 在 allocation-family untouched 的 P243 九场景
  reuse 上按原两门一次 confirmation，attained fraction MAE=`.016783`、regret=`8.44e-6`，再次 2/2；
- 没有更换 cohort、重训、refit、gate/threshold/cardinality sweep；P243 曾用于 surface confirmation 的消费边界已显式保留；
- tail-CVaR compiler family 至此关闭为 supported；下一可用 failure id 仍为 `V67-F196`。

### P274 bootstrap epistemic surface note — active / 无新增 failure

- 新对象不改 P270--P273 verdict；5-member scene-bootstrap monotone ensemble现正训练；
- P201只用两个预冻结决策，P243明确 descriptive-only；无 hash/checksum/fingerprint 或 smoke/regression matrix；
- 下一可用 failure id 仍为 `V67-F196`。

### P274 outcome / P275 start note — 无新增 failure

- P274 P201 final MAE=`.009363`、epistemic-error Spearman=`.693292`，2/2；P243仅描述且方向一致；
- P275 只在 P274 成立后启动，固定蒸馏 `mean-beta*std` 并用解析正 rates 编码 beta/budget 单调，不更改 P274 门；
- 下一可用 failure id 仍为 `V67-F196`。

### P275 outcome / P276 start note — 无新增 failure

- P275 P201 surface/final MAE=`.002235/.002861`、三轴 violations=0，2/2；P243继续只描述；
- P276 以冻结 P275 为唯一 teacher，soft-floor allocator当前运行；不将 LCB score包装为 credible interval/hard constraint；
- 下一可用 failure id 仍为 `V67-F196`。

### P276 outcome / P277 IO / P278 GPU overlap note — 无新增 failure

- P276 P201 budget MAE=`.011545`、regret=`.0001335`，2/2；
- P277六个新场景在 sensor/target read 前冻结，archive/preprocess 与 P278 fixed-group dual GPU training 并行；
- 当前没有资源不足或多卡需求；下一可用 failure id 仍为 `V67-F196`。

### P278 outcome / P279 composite-risk start note — 无新增 failure

- P278 P201 attained fraction MAE=`.016307`、regret=`8.11e-6`，2/2；
- P279 只在 P274--P278 均支持后组合 epistemic LCB 与 Actor-tail CVaR；预先保留“经验 composite score、非 coherence
  proof/credible interval/posterior-predictive risk”的 claim boundary；
- P277 archive IO 仍与 P279 GPU 并行；下一可用 failure id 仍为 `V67-F196`。

### P279 outcome / P280 composite-risk dual start note — 无新增 failure

- P279 P201 budget MAE=`.0117673`、composite-risk regret=`.000105044`，两门通过，解析 price/floor violations=`0/0`；
- P280 只把冻结 P279 primal 接到 attainable-fraction dual，不改变 composite-risk 定义、P279 verdict 或 claim boundary；
- 一次本地 PowerShell 启动命令漏写 `ssh`，在本机 `Set-Location` 阶段退出，未接触远端、未创建 run、未加载模型，
  属于控制端命令失误而非科学/实现 trial；随后正确 SSH 命令已启动 canonical r1；
- P277 IO 继续并行；无资源不足或多卡需求，下一可用 failure id 仍为 `V67-F196`。

### P280 outcome / P281 variable-cardinality start note — 无新增 failure

- P280 P201 attained fraction MAE=`.0162331`、composite regret=`2.67e-6`，2/2，fraction-price violations=0；
- P281 仅将冻结 P279 warm-start到 sizes32/64/128，并在未参与训练的48/96上一次评估，不改变风险对象或门；
- P277 IO 与 P281 GPU 继续并行；下一可用 failure id 仍为 `V67-F196`。

### V67-F196 — P282 frozen P199 run ID复制时漏词

- run：`run://worldsim_v67/WS-V67-P282-VARIABLE-SET-EPISTEMIC-TAIL-CVAR-DUAL-01/
  20260831T193000Z__variable-set-epistemic-tail-cvar-dual-s0-r1`；
- symptom：配置将 canonical `...__joint-horizon-reliability-copula-s0-r2` 误写成少一个 `reliability` 的路径，
  在 `torch.load` 首个 P199 artifact 时 `FileNotFoundError`；
- exposure：P126/P182虽加载，但 P199/copula、P203、P275/P281、teacher targets、optimizer step和任何 metric 均未发生；
- research response：Hydra defaults composition 与 OmegaConf interpolation均建议复用共享配置值，避免变体配置重复；考虑当前
  项目已冻结 PyYAML 合同且禁止扩大工程范围，本轮不迁移框架，只从已通过 P281/P279 配置恢复同一 canonical ref；
- resolution：commit=`e806de4`；r2=`20260831T193500Z__variable-set-epistemic-tail-cvar-dual-s0-r2` 已按原 seed、
  data、model、steps、gates启动；状态=`resolved_before_scientific_trial`。

下一可用编号：`V67-F197`。

### P282 r2 outcome / P283 conformalized surface start note — 无新增 failure

- P282 r2 P201 attained fraction MAE=`.0137322`、composite regret=`4.17e-6`，2/2；F196 已按相同合同关闭；
- P283 参考 conformal risk control / risk-averse calibration，把 P274 ensemble disagreement校准为 tolerance-conditioned
  multiplier；由于 unit scene-correlated，只预注册 empirical coverage，明确禁止 distribution-free/conditional guarantee；
- P283 GPU 与 P277 archive IO 并行；下一可用 failure id 保持 `V67-F197`。

### V67-F197 — P283 ensemble-std multiplier在 P201 严重欠覆盖

- run：`run://worldsim_v67/WS-V67-P283-CONFORMALIZED-EPISTEMIC-LCB-SURFACE-01/
  20260831T194500Z__conformalized-epistemic-lcb-s0-r1`；
- symptom：student对 frozen conformal teacher 的 P201 surface MAE=`.004687`、三轴 violations=0，但四个 heldout delta 的
  simultaneous coverage仅 `.666/.647/.592/.471`，maximum undercoverage=`.2590`，超过冻结 `.12`；
- diagnosis：校准 multipliers 高达 `4.35--9.77` 仍欠覆盖；P243 max undercoverage=`.2923` 同向，说明 P274 std 在
  low-disagreement biased rows 上是错误尺度，不能靠 multiplier 蒸馏容量或更多 steps 修复；
- literature response：dependent-data conformal工作强调 exchangeability破坏会损失 coverage；ICML/UAI 的 group/cluster
  conformal与 sequential residual方法改用/分组 residual score。当前先做最小对象迁移：P284 使用未标准化 additive
  one-sided residual quantile；不调 P283 gate、不扫 multiplier、不把 P283 1/2 包装为成功；
- claim impact：关闭 standardized ensemble multiplier 的 empirical coverage claim；P274 disagreement-error association 与
  P275 beta-conditioned score fidelity仍保留，但都不升级为 coverage guarantee。

下一可用编号：`V67-F198`。

### P284 additive residual recovery start note

- P284 同一 calibration split/delta/student/steps/gates，仅替换 nonconformity score，canonical r1 已启动；
- 即使 P284 通过，也只支持 observed-cohort empirical coverage，不支持 iid/exchangeable、conditional或 safety guarantee；
- P277 IO 继续并行；下一可用 failure id 保持 `V67-F198`。

### P277 fresh confirmation terminal note — 无新增 failure

- targeted shard扫描、2,341-member extraction和6/6 preprocess全部完成，无 recovery cohort、scene替换或质量预读；
- formal one-shot 在1,080 trajectories上 budget MAE=`.0153153`、regret=`.0008650`，2/2通过；
- 该 positive result只确认 frozen P275/P276 surrogate transfer，不能修复或覆盖 P283 的 standardized coverage failure；
- 下一可用 failure id 保持 `V67-F198`。

### P284 outcome / P285 allocator start note — 无新增 failure

- P284 P201 fidelity MAE=`.005598`、max undercoverage=`.06208`，2/2；P243描述性结果同向；
- additive residual structural recovery支持 empirical tolerance-conditioned LCB surface，但不反向改写 P283 standardized
  multiplier failure，也不升级为 distribution-free/conditional guarantee；
- P285 只在 P284 成立后冻结 teacher 并训练 soft-floor allocator；下一可用 failure id 保持 `V67-F198`。

### P285 outcome / P286 group dual start note — 无新增 failure

- P285 P201 budget MAE=`.0113088`、regret=`5.01e-5`，2/2；price/floor violations=0；
- P286 仅将冻结 P285 primal 接到 attainable-fraction dual，不更改 P284 empirical coverage或 P285 allocation verdict；
- 下一可用 failure id 保持 `V67-F198`。

### P286 outcome / P287 calibrated composite-risk start note — 无新增 failure

- P286 P201 attained fraction MAE=`.0202496`、group regret=`9.55e-6`，2/2；
- P287 仅在 P284--P286 成立后组合 additive empirical LCB 与 Actor-tail CVaR，预先保留 no coherent-risk/formal-
  coverage/hard-constraint claim boundary；
- 下一可用 failure id 保持 `V67-F198`。

### P287 outcome / P288 variable-cardinality start note — 无新增 failure

- P287 P201 budget MAE=`.0130764`、composite regret=`8.71e-5`，2/2，price/floor violations=0；
- P288 仅 warm-start同一架构到 train sizes32/64/128并在48/96评估，不改变 P284 coverage或 P287 risk定义；
- 下一可用 failure id 保持 `V67-F198`。

### P288 outcome / P289 variable-set dual start note — 无新增 failure

- P288 P201 budget MAE=`.0108336`、composite regret=`3.39e-5`，heldout sizes48/96 2/2；
- P289 仅编译冻结 P288 的 attainable-budget dual，不改变 additive coverage、tail-risk对象或 cardinality protocol；
- 下一可用 failure id 保持 `V67-F198`。

