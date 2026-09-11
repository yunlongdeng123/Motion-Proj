# 历史原始记录 074

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V6-F18：推理 adapter 不得反序列化 checkpoint 的训练器对象

R8 第二轮正式目录 `20260821T104107Z__frozen-generator-s20260821-r1` 中，SD-v1.5 已完整通过
4 cases × 2 repeats 的 capability/resource gate，但 Big-LaMa 在 `torch.load` 时尝试恢复 checkpoint
中未参与推理的 Lightning callback，因轻量推理环境没有完整训练框架而失败。冻结规则要求两个 ungated
候选都被实际执行，所以该轮仍如实 `rejected`；SD 的通过不能事后放松这条规则。

修复改用 PyTorch `weights_only=True`，只读取 tensor/state_dict，再按冻结 generator 结构严格加载；
不安装或执行训练器，不改变 checkpoint 字节、候选、案例、输入、seed、阈值、资源合同或选择规则。
以后第三方训练 checkpoint 的推理入口必须默认最小反序列化面，训练回调、优化器和日志对象不得成为
部署环境的隐式依赖。

### V6-F19：weights-only 仍需显式白名单化归档中的非张量全局类型

R8 第三轮正式目录 `20260821T104241Z__frozen-generator-s20260821-r1` 中，Big-LaMa 的
`weights_only=True` 正确拒绝了归档内未白名单化的
`pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`；SD-v1.5 再次完整通过，但
双候选执行门仍未满足，所以该轮保持 `rejected`。这不是显存或模型能力失败。

修复为该精确全局名注册无方法、无训练行为的占位类型，并只把该类型加入 PyTorch safe globals；
checkpoint 继续以 `weights_only=True` 读取，后续仍只消费 `state_dict`。不引入 Lightning 训练栈，
不执行 callback，也不改变任何正式实验变量。若归档再暴露未预注册类型则继续 fail-closed，禁止切回
不受限反序列化来绕过门控。

### V6-F20：safe-global 修复必须基于静态完整清单，不能逐异常猜测

R8 第四轮正式目录 `20260821T104432Z__frozen-generator-s20260821-r1` 在白名单化
`ModelCheckpoint` 后继续 fail-closed，下一项未注册类型为 `omegaconf.dictconfig.DictConfig`；
SD-v1.5 仍完整通过而双候选执行门仍失败。逐次靠异常暴露类型会无意义地重复正式运行。

修复先用 `zipfile + pickletools` 静态读取官方 checkpoint 的 `data.pkl` GLOBAL 指令，不执行
反序列化；完整非内建清单只有 OmegaConf 的 `ContainerMetadata/Metadata/DictConfig/ListConfig/AnyNode`、
`typing.Any` 和已知 `ModelCheckpoint`，其余为 PyTorch tensor/标准容器重建函数。adapter 一次性白名单化
这些精确类型后仍使用 `weights_only=True`，实验变量与选择规则不变。以后同类归档应先做静态类型清单，
再建立最小安全白名单，避免把正式 run 当依赖探针。

### V6-F21：Python 2 pickle 内建名迁移后也必须进入 safe-global 清单

R8 第五轮正式目录 `20260821T104627Z__frozen-generator-s20260821-r1` 中，OmegaConf 类型完成
白名单后，weights-only loader 继续拒绝由旧归档 `__builtin__.dict` 迁移得到的 `builtins.dict`；
SD-v1.5 再次通过，双候选执行门仍保持 `rejected`。静态 GLOBAL 清单此前列出了旧模块名，但未把
Python 3 运行时映射后的内建类型显式加入 safe globals。

修复补入清单中出现的标准容器及迁移类型：`dict/list/int/OrderedDict/defaultdict`；仍不对白名单外
类型开放，不切换为 unrestricted pickle。模型、checkpoint、输入、seed、阈值与选择规则不变。
以后审计跨 Python 版本的 checkpoint 时，GLOBAL 清单必须同时记录归档名与当前运行时解析后的类型名。

### V6-F22：跨阶段 source manifest 必须复制机器实测完整 SHA

R9 首次正式入口在创建 run 目录前 fail-closed：冻结配置把 R7 manifest 的
`73dbb2ba11bc12bc...` 手工误抄为 `73dbb2ba11bc4e22...`。现场 `sha256sum` 与 R7 closeout 均确认
源文件仍为 `73dbb2ba11bc12bc4e22ca13765af10d14ac5d183e1d529add5e6d619f2a4d0c`；因此没有 proposal、
verifier、hidden target、训练、confirmation 或方法结果产生。

修复只替换为机器实测的完整 source manifest SHA，不改变 R9 hypothesis、模型、cohort、arm、阈值、
gate 或资源合同。这是与 V6-F12 同类的 provenance 抄录错误；后续跨 run 配置应由 manifest artifact
自动生成，禁止再次从缩写或人工记忆恢复完整哈希。

### V6-F23：冻结 semantic checkpoint 必须 strict 重建 auxiliary head

R9 首个有 run 目录的正式实例 `20260821T110446Z__independent-arms-s20260821-r1` 已先生成全部
28 个 Big-LaMa proposal，随后 semantic worker 在 strict load 时拒绝 checkpoint 中的
`aux_classifier.*`：初版 adapter 以 `aux_loss=False` 构造 DeepLabV3，遗漏了训练 checkpoint 保留的官方
auxiliary head。run 保持 `failed`，未产生 arm verdict、融合或 bake，也不是模型质量/资源负结论。

修复只以 `aux_loss=True` 重建同一 19-class DeepLabV3-ResNet50，并继续 strict load 全部参数；正式推理
仍只消费主输出 `out`，aux head 不参与 arm score。权重字节、cohort、proposal、threshold、gate 与资源合同
均不改变。以后冻结视觉 checkpoint 的结构审计必须覆盖所有 state-dict head，不能以“推理不消费”为由
在 strict identity 前删除参数。

### V6-F24：第三方 checkpoint 的主 head 与 auxiliary head 类数可能不一致

R9 第二个正式 run `20260821T110616Z__independent-arms-s20260821-r1` 在启用 aux 结构后继续由
strict load 拒绝：归档主 classifier 为 Cityscapes 19 类，但 `aux_classifier.4` 仍是 torchvision 默认
21 类（权重 `21×256×1×1`），而统一 `num_classes=19` 构造出的 aux head 为 19 类。run 保持
`failed`；28 proposals 已生成，但无 verifier verdict、融合或 bake。

修复精确重建归档结构：主 head 保持 19 类，aux 最后一层单独恢复为 21 类，然后 strict load 全 state dict；
aux 输出仍不被正式分数消费。模型权重、P3 动态类定义、cohort、threshold、gate 和资源合同均不改变。
以后第三方 segmentation checkpoint 必须逐 head 审计 shape，不能假设所有 classifier 共用同一 label count。

### V6-F25：capability 最优的 generator 不等于 verifier-arm 质量最优

R9 canonical rejected run `20260821T110743Z__independent-arms-s20260821-r1` 在 28 个 matched
development pseudo-holes 上证明 Big-LaMa 虽是 R8 的资源最优候选，但没有 verifier arm 可进入 R10：P1/P2
均 `0/28` ACCEPT；P3 在 12 个 actor-evidence cases 中接受 6 个，却有 1 个 false-safe，率 `1/6=0.1667`
高于冻结 `0.10`；P4 正确 `28/28` ABSTAIN。P0 photo/geometry false-safe 均为 `1.0`，P3 为
`0.5833`。outside-mask exact、无融合/无 bake/无 confirmation 均通过，峰值仅 `428 MiB`，不是资源失败。

H-R9-001 正确裁决为 `rejected`，不得放宽 photo/depth/semantic 阈值或把全拒绝写成有效 verifier。
新 H-R9-002 只切换到 R8 已完成 capability gate 的第二候选 SD-v1.5；沿用完全相同的 28-case cohort、
hidden observations、P1–P4、truth 定义、threshold、gate、模型和资源上限。Big-LaMa 与 SD 结果必须分属
独立不可变 run，禁止在看到 SD 结果后混合选择 per-case generator。

### V6-F26：单图生成候选无法把不可观测 missing-world 内容变成可验证事实

H-R9-002 canonical rejected run `20260821T111228Z__independent-arms-s20260821-r1` 用冻结
SD-v1.5 替换 Big-LaMa，并保持所有 verifier 与 gate 不变。P1 仍 `0/28` ACCEPT；P2 仅 `2/28=0.0714`
且低于冻结 `0.10` coverage；P3 与 Big-LaMa 同为 `6/12` ACCEPT、`1/6=0.1667` false-safe；P4
`28/28` ABSTAIN。outside-mask exact，峰值 `2696 MiB`，无融合、bake、训练或 confirmation。因此
H-R9-002 同样是方法质量 `rejected`，不是工程/资源 blocked。

两种单图 inpainting 都失败后，不得调松阈值、按案例混选生成器或转向 gated 23.8GB FLUX 权重来规避
负结果。新 H-R9-003 将唯一变量改为冻结 cross-frontend reconstructed proposal：同 scene/frame/edit variant
使用另一 frontend 的对齐 RGB 填入 mask；P1–P4、truth、threshold、gate 和 denominator 原样保留。该 proposal
仍标记 reconstructed，两个 frontend 来自同一传感器支持，不能解释为新增观测或独立 ground truth。

### V6-F27：正式入口必须显式建立仓库模块搜索路径

R12 第一次启动命令在创建 run 目录、加载模型或执行 GPU 推理前失败：直接运行
`python scripts/worldsim_v6/run_logsim.py` 时，Python 只把脚本目录加入模块搜索路径，因而无法导入仓库根目录下的
`motion_proj` 包并抛出 `ModuleNotFoundError`。该失败没有产生样本、指标或方法结论，也不是资源失败。

修复只在入口脚本中根据 `__file__` 把仓库根目录加入 `sys.path`，不改 R12 hypothesis、cohort、输入哈希、模型、
阈值、gate 或资源合同。后续以完全相同命令重跑；任何模型或指标失败仍独立登记，不能用本次入口错误掩盖。

### V6-F28：启用 CUDA 确定性算法前必须冻结 cuBLAS workspace 配置

R12 首个有 run 目录的正式实例 `20260821T114117Z__logsim-s20260821-r1` 已完成两项静态 chunk 的 CPU
重放构造并成功严格加载冻结 DeepLab checkpoint，但第一次 GPU forward 被 PyTorch fail-closed：代码启用了
`torch.use_deterministic_algorithms(True)`，CUDA 10.2+ 的 cuBLAS 路径还要求进程启动前设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。该 run 保持 `blocked`，没有感知输出、完整 gate 或方法结论；也没有发生 OOM。

修复只在启动感知子进程前加入这一确定性环境变量，继续保留 deterministic algorithms、同一 checkpoint、4 个输入、
同一 cohort、阈值与资源上限。不得关闭确定性检查来换取通过；修复后新建独立 run 重试。

### V6-F29：世界空间 z-buffer 必须把空视锥投影作为有效零覆盖结果

R13 首个正式 run `20260821T120059Z__worldspace-route-s20260821-r1` 已完成两个 verified chunk 的世界坐标
提升，但部分大幅路线偏离没有任何点落入目标视锥。初版 z-buffer 仍为零长度索引构造了长度 1 的首元素布尔 mask，
触发 `IndexError`。该 run 保持 `blocked`，没有完整 baseline matrix、gate 或方法结论，也不是资源失败。

修复只在 z-buffer 中对零个可见点直接返回空的 x/y/z/source-index；下游按预注册协议记录
`projected_pixel_count=0`、指标不可用且 route-support fail。不得删掉这些偏离、降低分母或将空投影改写成 ABSTAIN。
其余输入、深度、标定、四方法、阈值、gate 和资源合同均不变，并以独立 run 重试。

### V6-F30：WorldSim evaluator 必须统一合法的 singleton-channel depth plane

R13 第二个正式 run `20260821T120208Z__worldspace-route-s20260821-r1` 在加载偏离路线的 StreetGS depth 时
fail-closed。该 renderer 保存合法的 `H×W×1` float depth，而 evaluator 的 PIL resize 入口只接收 `H×W`，
因此抛出 `TypeError`。run 保持 `blocked`，尚无完整 48-row baseline matrix 或 gate；没有 GPU/内存问题。

修复在 resize 前只接受 `H×W` 或 `H×W×1`，后者显式去掉最后 singleton channel；其他形状继续拒绝。
这与 V6-F13 的 plane normalization 原则一致，但本次记录覆盖独立的 R13 evaluator。不得改 depth 数值、插值模式、
样本、四方法、阈值或 gate，修复后以新 run 重试。

### V6-F31：verifier 相对深度不得直接解释为 WorldSim 米制相机 z

H-R13-001 canonical rejected run `20260821T120310Z__worldspace-route-s20260821-r1` 成功把两个 R11 chunk
封装为 58,273 个所谓世界点，但估计总面积仅 `0.01993 m²`，且 12/12 个非零路线偏离均为零投影覆盖，
usable lateral route 为 `0.0 m`。V6 的 matched false-safe 仍为 `0/3`，相对 naive 的降低为 `0.8214`，
所以拒绝原因不是安全 gate，而是 R9 depth 只为仿射对齐后的 verifier 几何比较服务，不能直接当作米制 z 做相机平移。

H-R13-001 按预注册门槛正式 `rejected`，不得通过放宽 256-pixel、0.12 photo 或 0.30 geometry 门槛恢复。
H-R13-002 只替换深度来源：使用同帧冻结 logged LiDAR metric depth，并限定最近填充距离不超过 8 个
512×288 像素；世界提升、四方法、12 个偏离、评估阈值、false-safe gate 和资源合同全部保持不变。

### V6-F32：无可见性约束的关键帧点云 union 会放大遮挡错误而非扩展路线

H-R13-003 canonical rejected run `20260821T121112Z__worldspace-fusion-s20260821-r1` 将两个 metric world chunk
按冻结 5cm voxel 做无目标视角过滤的 union。57,997 个输入点因近表面重复被折叠为 5,868 点，没有形成预期 densification；
共同 lateral route 从 `3.0m` 退化到 `2.0m`，5m mean geometry MRE 从 `9.0572` 升至 `10.6579`，
相对变化为 `-17.67%`，两帧 5m 继续失败。因此假设正式 `rejected`，不是工程或资源 blocked。

不得靠扫描更小 voxel 或放宽 photo/geometry 门槛复活该 union。H-R13-004 转向不同机制：保持 H-R13-002 的
逐帧 world points 和 RGB，不做 union/densification；只用同一冻结 support 中三相机 logged LiDAR 投影到目标视角，
在 4 像素邻域和 0.30 相对深度差内保留可见点。StreetGS truth proxy 只用于最终评估，不进入过滤。

### V6-F33：跨阶段安全摘要必须读取冻结 schema 的完整方法键

H-R13-004 首个正式 run `20260821T121825Z__worldspace-visibility-s20260821-r1` 已完成两次目标视角
LiDAR 投影和全部 12 个偏移的指标计算，但在汇总继承 H-R13-002 的 V6 false-safe 时 fail-closed。冻结摘要使用
`baseline_safety.v6_generate_verify_bake.false_safe_rate`，初版 runner 却读取了不存在的缩写键
`baseline_safety.v6.joint_false_safe_rate`，因此抛出 `KeyError`，run 保持 `blocked`，没有形成 gate 或方法结论。

修复只按冻结摘要的实际 schema 读取完整方法键和 `false_safe_rate` 字段；不改 world points、目标视角 LiDAR、4 像素/
0.30 visibility 合同、12 个偏移、质量阈值、false-safe 数值或资源合同。以后跨阶段消费结构化摘要时，必须把完整方法标识和
字段名纳入配置/manifest 合同，禁止用人工缩写推断 schema。

