# 历史原始记录 072

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### PIVOT-F17：DGGT 扩展构建必须同时固定 compiler、headers 和 Python 依赖上界

V2 M1 表明，“已安装 torch cu121”并不足以证明 CUDA extension 可构建。宿主只有 CUDA 11.8
toolkit，会在 pointops2 编译时与 torch 2.4.1+cu121 硬失配；只补 `nvcc` 又会缺
cusparse 等 headers。正确合同是在前缀环境固定 NVIDIA CUDA 12.1 compiler/runtime/headers，
传播 `CUDA_HOME/CPATH/LD_LIBRARY_PATH`，再按 upstream `python setup.py install`。

同一里程碑还暴露了浮动 Python 树的独立风险：transformers 5.x 使用 torch 2.4.1 未提供的
DTensor API，diffusers 0.39 触发 torch schema 不兼容。最终固定
`transformers 4.48.3 / tokenizers 0.21.0 / diffusers 0.32.2 / numpy 1.26.4 /
opencv-python 4.11.0.86 / rerun-sdk 0.23.1 / flow-vis 0.1`。

对应 blocked runs 为
`20260802T120027Z__native-nusc-s0`、`120943Z__...-r2`、`122213Z__...-r3`、
`122904Z__...-r4`、`124347Z__...-r5`。这些失败是构建/依赖证据，不能推断 DGGT
方法质量。

### PIVOT-F18：原生阶段完成不应被后续评估依赖失败覆盖

M1 r6 已完成 18/18 1-view 和 18/18 3-view，但 common evaluator 导入 AD-GS 冻结
`loss_utils` 时因 `flow_vis` 未安装而 blocked。原生输出本身未损坏，但主 terminal 已转为
blocked，禁止为了“好看的 done”改写。

恢复方式是新建 r8，对 r6 `native_summary.json/metrics.json`、每个 stage 和输出哈希做
fail-closed 引用后只执行 common diagnostic。r7 中重试封装自身的 `KeyError` 也以新的
blocked run 保留，再由 r8 完成。后续所有 multi-stage run 必须把“可复用的完成阶段”与
“整个 instance 的 terminal 终态”分开；重试不得修改旧 terminal。

### PIVOT-F19：nuScenes devkit 反向索引与磁盘 metadata 不是同一 schema

M2 r1 直接读取官方磁盘 `sample.json` 时发现其中没有 `anns`；该字段是 nuScenes devkit
初始化后才注入的反向索引，不是原始 JSON 合同。正式适配器改为流式扫描
`sample_annotation.sample_token`；由于这个外键非唯一，不得用单值 dict 覆盖同一 sample
的多个 annotation。`ijson` 还必须以 `use_float=True` 读取，否则 Decimal 会污染严格 JSON
运行合同。

同一里程碑还表明，“时间最近”不足以建立 raw annotation 到 camera sweep 的真值映射。r4
中 scene-0242 boundary actor 命中更近的 sweep，但 sweep 所属 `sample_token` 与 raw 2 Hz annotation
不同，因此在 QA 前即 blocked。正确规则是先限定 exact sample token，再在候选内最小化
timestamp delta；正式 r5 达到 `4356/4356` exact mappings。后续不得仅按文件名或时间
猜测 raw/processed/render 映射。

### PIVOT-F20：CUDA 扩展 import 成功不等于包含当前 GPU 架构

M3 的 DriveStudio 环境能正常 import `gsplat` 和 `nvdiffrast`，但旧二进制没有 RTX 3090 的
SM 8.6 kernel：前者在 SH rasterization 报 `no kernel image`，后者在 EnvLight 路径报 CUDA 209。
只做 import smoke 无法发现此类错误。恢复时分别固定官方源码 commit，以
`TORCH_CUDA_ARCH_LIST=8.6+PTX` 重建，并执行真实 CUDA forward/backward；旧 `.so` 先备份，
没有修改算子语义或模型配置。

对应 blocked runs 为 r4、r6、r7；正式 binary SHA-256 为 gsplat
`6d7c8e5a...dd6131`、nvdiffrast `0d18f767...96499`。以后 CUDA 扩展 readiness 必须包含
目标 GPU 上的实际 kernel forward/backward，不能只看包版本和 import。

### PIVOT-F21：训练完成 checkpoint 与累积式 post-render 必须分开裁决

M3 r8 的 30k 原生训练已保存 `step=30000` checkpoint，但上游随后将 588 个 full-render 结果累积在
内存中；在 `577/588` 时 cgroup memory 连续两次超过 90%，资源守卫发送 SIGTERM。`oom=0 /
oom_kill=0`，checkpoint 字节数与 step 完整。r8 仍保持 `blocked`，不得改写为 done；r12 通过新的
不可变 run 对 checkpoint step/bytes/hash 和原失败 terminal 做窄范围复核，再执行流式 27-image
edit smoke 完成 M3。

同一恢复链还发现，正式训练会把某个非目标 rigid model 的全部 Gaussian 裁剪掉。token、dataset column
和 model index 仍是一一映射，但 checkpoint slice 为空。registry v2 因此将其明确标成
`unavailable_empty_checkpoint_slice`，同时对正式选中 actor 继续要求非空。禁止为了全 registry 看起来
完整而伪造 slice，也禁止因一个非目标空 slice 丢弃 23 个真实非空映射。

### PIVOT-F22：外层 timeout 不会自动回收独立 session 的 GPU 子进程

M4 controller 用 `subprocess.Popen(..., start_new_session=True)` 隔离正式渲染，使 SSH/tmux 断开不应
误杀长任务；相应地，用外层 `timeout` 调试 controller 时，SIGINT 只终止父进程，子进程会以 PPID 1
继续占用 GPU。`debug_controller_s0_r5` 复现了该行为；残留子进程通过已核实的精确 PGID 发送 SIGTERM
回收，GPU 从约 `8.1 GiB` 回到 `0 MiB`，没有终止用户服务。

以后不得用外层 timeout 探测会派生独立 session 的 controller。正式运行应直接由 nohup/tmux 托管，
同时监控 controller PID、child PID、terminal 和 resource.jsonl；确需中止时必须核实 process tree 后
显式回收 child process group。`r5/r6` 的 running terminal 保留为中断证据，不改写成 done。

### PIVOT-F23：SE(3) 一致性容差必须覆盖 float32 往返误差

M4 单帧 r1 的 actor transform 先由 checkpoint float32 tensor 变换，再写入 JSON 并读回，最大平移误差
略高于 `1e-6 m`；其余 15 项检查均通过。把该值当几何失败会制造假阴性。协议在查看正式全量结果前
固定为 `1e-4 m`，r2/r3 冒烟通过，正式 196 帧实测最大误差为
`3.814697265625e-06 m`，rotation/size/canonical drift 均为零。容差变更只反映数值精度，不降低
1 m 编辑幅度，也不得据此为真正的轨迹偏差放宽门禁。

### PIVOT-F24：冻结 heldout 资源门失败不能靠事后更换 renderer 或提高阈值挽回

A3 R1 在结果前冻结 `12,288 MiB` PyTorch allocated GPU ceiling。三条完成全部 R0/R1 指标计算的只读路径
`r2/r4/r5` 分别达到 `14,241.777 / 14,244.924 / 14,241.399 MiB`；wall、cgroup、run bytes 与 OOM delta
均通过。资源审计、CPU checkpoint staging、Rigid quota device 兼容和逐 view `trainer.info` 释放都没有改变这一
单 view 峰值。继续把 renderer 改为 `packed=true`、分块/降分辨率，或把 ceiling 提高到观测值以上，会在看到结果后
改变 source-render 路径或预算，不再是预注册评测。

r5 的资源无效 diagnostic 也不能救回方法：S-B depth-order violation 从 `0.915792` 降至 `0.908173`，但
non-target 与 original-global RGB MSE 都按 exact comparator 严格变差，故为 `tradeoff_non_dominated`。正确分层是：
r5 run 保留 `blocked`，R1 方法臂登记 `rejected_resource_gate_and_diagnostic_tradeoff`，A3 任务以负结果 `done`；
生产路由回退到 R0/D2 immutable exact alias。以后若研究 packed/分块渲染，只能在 A4 作为新的部署因子另行冻结，
不得倒写 A3 heldout 结论或解锁 R2–R4。

### PIVOT-F25：部署 profile 必须区分传感器原始尺寸与 checkpoint 原生加载尺寸

A4-P0 v1 在新测量前把 scene-0230 的 nuScenes 传感器尺寸 `1600×900` 冻结为“原生分辨率”。formal r1 实际
完成 2 次 warm-up 与 9 次 measured render 后，11 行全部为 `800×450`，因此只在 finalize 的
`native_resolution_exact` audit 失败；资源、输入 hash、无训练/无 checkpoint、同步矩阵与无 torch resume audit
均通过。source config 事后审计确认三路相机已冻结 `data.pixel_source.downscale_when_loading=[2,2,2]`，故当前
checkpoint 的模型原生加载/渲染尺寸本来就是 `800×450`。

不得修改 v1 或把 r1 改写为 done，也不得用 r1 性能数字关闭 P0。正确处理是保留 r1=`blocked`，冻结其 protocol、
manifest、runtime stage/rows、resource audit 与 terminal hash；创建 v2 只纠正分辨率语义，再从新目录完整重跑。
后续协议必须同时记录 sensor resolution、source-config downscale 与 model-native render resolution；“native”一词
不能在这三层之间无来源转换。该纠错不授权降低分辨率、切换 renderer、改变资源 ceiling 或开启 P1/P2/P3/P5。

### PIVOT-F26：checkpoint state key 不能冒充加载后的模型运行时属性

A4-P5 formal r1 已通过 9 项输入审计并生成 `14,729-byte` reference-only deployment registry，但 fresh DriveStudio
worker 在 checkpoint 成功加载后读取 `RigidNodes.points_ids` 时报 `AttributeError`。源码事实是：checkpoint
`state_dict` 以 `points_ids` 为序列化键，`load_state_dict` 将它弹出并写入运行时属性 `self.point_ids`。两者语义
相关但接口层不同；直接把 checkpoint 键拼成对象属性会使恢复链在资产已经物化后失败。

r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 保留 `blocked`，terminal SHA=`61d30a11...773e`；其已生成
registry SHA=`e48bccdf...9039d` 不覆盖。修复 `0e899b2` 只通过 fail-closed helper 读取 runtime `point_ids`，并用
两条测试分别锁定有效属性和拒绝 `points_ids` 别名；没有修改 P5 protocol、输入、资源 ceiling 或审计口径。
新目录 r2 以相同 registry SHA 完成 14/14 audits，证明问题属于 runner/runtime contract，不是资产或方法失败。

以后凡从 checkpoint 结构推断 live module API，必须同时核对 `state_dict` 保存端、加载端赋值和加载后的真实对象，
并用回归测试锁定层级；旧失败 run 维持 `blocked`，不得因修复后的新 run 成功而倒写。

### PIVOT-F27：最小预注册剪枝臂失败后不能事后补更小 fraction 或放宽质量门

A4-P1 在结果前固定 source/b05/b10/b20 四臂，并要求 global、actor、boundary 与 non-target 的全部 safeguard 同时
通过。canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1` 完成 36-view contribution、三臂物化、
四臂 57-view 质量和 9-view runtime，21/21 audits 全 true，资源门通过；因此它不是工程或资源 `blocked`。

最小候选 b05 已将 checkpoint 减少 `23,881,368 bytes`，全部 row/invariant/reload/count 审计 exact，但 global
occupied PSNR、global PSNR 与 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结的
`0.10 dB` 上限；b10/b20 分别失败 12/15 个端点。局部 actor/boundary 指标的保持或改善不能覆盖全局与非目标区
失败。运行时 P50/FPS 也非随 fraction 单调，且 filesystem cache 未控制，只能报告，不能充当事后选择理由。

正确裁决是 P1 experiment=`done`、method=`rejected_quality_or_integrity_gate`、生产资产 exact fallback 到 source。
不能在看到 b05 失败后新增 b01/b02、改排名视图、放宽 `0.10 dB` 或只保留通过的局部端点；这些都属于新的预注册
实验，而不是当前 P1 的修复。该负结果只约束 scene-0230/seed-0/冻结视图矩阵，不外推为所有贡献度剪枝均失败。

### PIVOT-F28：顶层 `named_parameters()` 不保证覆盖普通映射中的子模型参数

A4-P2 formal r1=`20260809T174337Z__a4-p2-mixed-precision-s0-r1` 已成功完成 10-field checkpoint conversion、
source/candidate 57-view quality、两臂 runtime、aggregate 与 no-torch resume；aggregate 也按冻结门选择 mixed arm。
但 finalizer 的 `checkpoint_reduction_and_runtime_matrix_exact` 唯一失败：账本只调用
`trainer.named_parameters()`，而 DriveStudio 把 Gaussian 子模型保存在普通 `trainer.models` 映射中，并未注册成
顶层 `ModuleDict`。因此账本只看见 LPIPS 等 `9,883,392 bytes`，把 source/candidate 错记为相同总量且没有 FP16
bucket，尽管候选 checkpoint 已实际从 `578,819,674` 降到 `432,111,754 bytes`。

这属于 evidence collection defect，不是 conversion、质量、资源或方法失败。r1 保留 `blocked`，terminal SHA=
`5ef3dab60ff934af19ff547c0f7e7cd0fe74b83000476888a630341ee39474c0`；不得手改 r1 runtime stage 或把它倒写为
done。修复 `dcf2822` 显式遍历 `trainer.models` 中每个 module 的 parameters，并按 Parameter identity 去重；回归 fixture
故意使用未注册的普通映射，锁定 `models.Background._scales` 等字段必须进入账本。协议、字段、阈值、renderer 与
selection 均未改变。

新目录 canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2` 完成 19/19 audits，正确记录 source/candidate
persistent bytes=`394,641,424 / 247,936,208` 与 candidate FP16 bucket=`146,705,216 bytes`，并选择 mixed arm。
以后审计复合模型时，必须同时核对容器是否已注册为 `nn.Module`、顶层 traversal 覆盖范围与逐字段预期集合；
证据账本缺失不得用 checkpoint 文件变小或 runtime 成功来推断补齐。

### PIVOT-F29：无卡实例必须以 cgroup 内存为资源合同，不能读取宿主机 `free`

V3.2 S0 在 AutoDL 无卡开机模式中观察到 `free -b` 暴露宿主机约 810 GB 内存，但
`/sys/fs/cgroup/memory.max=2,147,483,648`，实际只有 2 GiB。CoIn 的完整 partial-clone checkout 在大量 blob
物化时把 `memory.current` 推近上限且长时间无进展；继续并发创建环境或校验大权重会把可回收 page cache 与真实
匿名内存混在一起，增加无意义的 OOM 风险。

正确处理是停止该 checkout，把残留移到仓库外备份，并改用 `--no-checkout --filter=blob:none` 固定 commit/tree；
所有下载使用流式落盘，依赖安装和 hash 校验串行执行。GPU、VRAM 与 driver 则必须在有卡重启后重新审计，不能把
无卡模式的 `nvidia-smi: permission denied` 写成硬件不存在。后续任何资源 preflight 都必须同时记录
`memory.max`、`memory.current`、`memory.events` 和数据盘余量；宿主机 `free` 只作诊断，不作授权。

