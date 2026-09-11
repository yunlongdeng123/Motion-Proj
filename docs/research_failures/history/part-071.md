# 历史原始记录 071

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### PIVOT-F04：不能把“可见性建模”泛化成未观测背景已经解决

**观察**

AD-GS 的双向时间可见性用于动态对象生命周期和已观察运动建模；VAD-GS（CVPR 2026）的 visibility-aware
densification 已覆盖稀疏观测下的几何补密；DrivingEditor 支持对象删除/添加；Real2Sim 进一步展示对象级编辑与
物理交互。这意味着“增加一个 visibility 模块”或“支持平移对象”本身已经不足以构成新意。

**仍未闭合的问题**

反事实轨迹编辑会同时制造原位置去遮挡、新位置遮挡、跨相机深度排序和证据外外观。当前项目只有在下列内容形成
联合、可验证方案时才可提出方法 claim：

- 编辑诱发的显式可见性重计算；
- 未观测区域的真实性/置信度与拒绝机制；
- 跨视角、跨时间一致的背景恢复；
- 目标区预期变化与非目标区感知保持。

**防重复**

创新选择前必须把 AD-GS、VAD-GS、DrivingEditor、DGGT/ReconDrive 和当时最新工作重新做一次代码可用性与
claim 边界审计；不得把已有 visibility-aware densification 或基本对象变换重新命名为贡献。

### PIVOT-F05：资源不足时研究停机规则跨路线继续生效

`N1-F24` 是项目级规则，不属于 cut-in 专属逻辑。本轮 cgroup 为 `memory.max=2,147,483,648` bytes，轻量元数据
审计后 `memory.current` 一度达到 `2,129,526,784` bytes，因此立即停止 Python 扫描、conda 求解、下载、
预处理和训练，只继续轻量文本/文件操作。后续任何新路线任务遇到相同条件时，必须保存失败/现场证据并等待用户开放
资源；不得反复重跑、杀用户服务或偷偷缩减正式协议。

### PIVOT-F06：旧机器 smoke 证据不能替代新实例复验

迁移到 RTX 4080 SUPER 新容器后，已有环境目录和旧 RTX 4090 日志仍然存在，但它们不能证明当前 driver、CUDA、
扩展 ABI、显存与 cgroup 合同可用。M2 因此在新机器上重新执行 AD-GS forward/backward、DPT、SAM2、
Grounding DINO HF 和 CoTracker3 smoke，并为每项保存独立退出码。

后续换机或容器重建时，即使复用同一 env/checkpoint，也必须生成新的 instance 级环境证据；旧日志只能作为历史，
不能复制为当前 PASS。

### PIVOT-F07：非 login shell 的 PATH 不能作为 CUDA provenance

M2 首次当前机器采集因非 login shell 找不到 `nvcc` 提前失败，而 `/usr/local/cuda/bin/nvcc` 实际存在。环境报告
已改为显式设置 `CUDA_HOME` 并调用绝对路径，同时传播 smoke 的真实退出码。

后续自动任务必须显式记录并使用 toolkit 路径；“命令不在 PATH”与“机器没有 CUDA toolkit”必须分开裁决。

### PIVOT-F08：在线浮动模型不能进入 exact reproduction

upstream 的 CoTracker `torch.hub` 在线 `main` 与未固定 revision 的 Hugging Face 模型都会随时间变化。M2 将
CoTracker repo、离线 checkpoint、Grounding DINO HF revision 与 snapshot fingerprint 全部固定并哈希，
运行时使用 offline mode。

后续 baseline 不得在正式 run 中联网追随 `main`、latest 或未固定 snapshot；若必须升级，使用新 config
fingerprint 和新 run instance。

### PIVOT-F09：tar 页缓存与 nuScenes auxiliary 都属于资源/资产合同

并行流式扫描约 294 GB tar 时，文件页缓存计入本容器 cgroup，首个扫描实例峰值达到 `57,001,484,288` bytes。
本任务只对自己已读过的 tar 文件区间调用 `POSIX_FADV_DONTNEED`，没有全局 `drop_caches`、杀用户服务或清理
其他任务缓存。

第一次结构审计还发现 1,440 个 RGB/LiDAR payload 齐全并不足以初始化 nuScenes devkit；`map.json` 引用的
4 个静态 map masks 同样是必需资产。失败实例
`20260727T165549__e49a4e-4080s-r2` 保留为 `blocked`，补齐并哈希登记 maps 后由新实例
`20260727T180733__e49a4e-4080s-r3` 通过。以后 selective extraction 必须同时审计运行库隐式依赖的 auxiliary
文件，不能只按训练脚本直接打开的传感器路径计数。

### PIVOT-F10：AD-GS 的 PNG 输出与 SAM2 的 JPEG-only 枚举不兼容

M3 首个 scene-0230 实例完成 `prepare_raw` 和 180 张 depth 后，在 sky mask 初始化时报
`no images found`。AD-GS `scripts/nuscene/nuscene.py` 固定写 `000000.png`，其 `semantic.py` 自己也会枚举
PNG；但 Grounded-SAM-2 `load_video_frames_from_jpg_images` 只按 `.jpg/.jpeg` 扩展名建立 video frame 列表。
这不是空数据、模型失败或资源 OOM。

失败实例 `20260727T181617__scene0230__s0` 保留为 `blocked`。最小兼容性修复只在 instance work dir 中为每个
PNG 建立相同字节内容的 `.jpg` 硬链接，跨文件系统时复制原始字节；PIL 已验证按内容可无损读取，不做 JPEG 转码，
不改 Grounding/SAM 模型、box/text 阈值、mask、帧序或评测。修复后的 AD-GS patch SHA-256 为
`114c3976af2c80d1da5581b401b3a099f22a7483347fc401113c8439bc991eb9`，必须由新 M3 instance 复验。

### PIVOT-F11：COLMAP 默认全核并发会越过本机 cgroup 内存门禁

M3 第二个实例 `20260727T182247__scene0230__s0-r2` 已完成 180 张 depth/object/sky/semantic 与
138/138 flow，在 COLMAP feature extraction 阶段发现 upstream 未指定线程数，COLMAP 自动使用容器可见的
128 个 CPU threads。cgroup memory 峰值达到 `62,265,835,520` bytes，并连续两个采样超过
`memory.max=66,571,993,088` 的 90% 停止线；runner 只终止本 run 的进程组，`oom=0 / oom_kill=0`，
失败实例与部分 COLMAP 目录均保留。

这不是图像、SIFT、匹配或几何协议失败。最小资源兼容修复显式传入
`SiftExtraction.num_threads=16` 与 `SiftMatching.num_threads=16`；不改分辨率、相机、帧、SIFT 参数、
exhaustive matching 或评测。r3 的 COLMAP 已完成 138/138 图像注册与 70,933 points，阶段峰值降至
`35,117,174,784` bytes。当前完整 compatibility patch SHA-256 为
`49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`。

以后在 CPU 核数远大于内存预算的容器内运行 COLMAP，必须显式登记并发数并纳入资源合同；不得把减少图像、
降低分辨率或删相机伪装成等价的资源修复。

### PIVOT-F12：跨场景连续执行时，official render 可先于 OOM 触发 cgroup 90% 合同

M3 scene-0230 的 60k render 峰值已达到 `59,530,678,272` bytes，距离注册的 90% 停止线仅
384,115,507 bytes。M4 scene-0242 严格串行完成全部 preprocess 后，100-step train 峰值为
`59,359,428,608` bytes；随后的 official render 在第 2/138 帧连续两个采样达到 90%，峰值
`59,996,393,472` bytes，比停止线高约 81.6 MB。runner 只向本 stage 进程组发送 `SIGTERM`，
stage `rc=-15`、runner `rc=1`，`oom=0 / oom_kill=0`，没有影响其他服务。

该结果说明当前 `memory.max=66,571,993,088` bytes 对六场景连续 exact reproduction 没有足够安全余量；
“尚未 OOM”不能用来绕过预注册停止线。blocked 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
必须保留，不在同一资源合同下立即重跑，也不得降分辨率、删相机、调模型或全局 `drop_caches`。

恢复时应先提高 cgroup 内存额度，建议至少 80 GiB、推荐 96 GiB，再创建新 instance；允许复用逐文件冻结的
processed scene，但必须记录来源、哈希和新资源合同。若无法增加资源，则 M4 保持 `blocked`，不得将只有
scene-0230 的结果写成六场景论文复现。

### PIVOT-F13：processed scene 复用校验必须区分关键产物与合法空占位

RTX 3090 换机后的首个 scene-0242 复用实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131533__scene0242__s0-r2-wm3090/`
在训练前 fail closed：新增递归哈希校验把 COLMAP 的合法 0-byte `created/sparse/model/points3D.txt`
占位文件误判为损坏。旧实例以 `blocked` 保留，没有修改 processed scene 或启动训练。

合法修复允许 COLMAP 非关键占位文件为 0 bytes，同时继续强制 `database.db`、`cameras.txt`、
`images.txt`、`colmap.ply` 和所有训练直接消费的 image/depth/mask/flow/meta/point cloud 非空；
复用后必须重新运行独立 processed audit。修复后的新实例 output fingerprint 为
`32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`，138/138 registered images
审计通过。不得删除空占位文件、伪造非空内容或因此重跑昂贵预处理。

### PIVOT-F14：容器实例重建必须与 OOM/方法失败分开

M5 首个正式 DGGT 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T094923__native-nusc-s0-wm3090/`
在 `env_torch` 下载期间停止。日志没有 stage 终态、OOM 或 `oom_kill` 增量；当前容器 PID 1 的启动时间为
2026-07-29 13:13:43 +08:00，晚于日志停止时间，故裁决为外部容器实例重建，而不是 DGGT 精度、显存或方法失败。

旧 run 与旧 controller 已原子标为 `blocked`，部分环境移动到
`/root/autodl-tmp/envs/dggt.interrupted-20260729T094923/`，没有覆盖或删除。恢复使用新 run
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；以后同类中断必须先核对 PID 1 启动时间、stage marker、launcher 与 OOM 证据，再决定是否创建
新实例，禁止把 stale `running` 当成任务仍存活。

### PIVOT-F14B：pointops2 的 PEP 517 build isolation 没有继承已安装 torch

恢复实例完成 Python 3.10、torch 2.4.1 和全部 requirements；resolver 最终选择
`rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4`。随后 upstream pointops2 执行普通
`pip install .` 时，PEP 517 临时 build env 在读取 setup requirements 阶段报
`ModuleNotFoundError: No module named 'torch'`。正式 stage `rc=1`，峰值 cgroup memory
`16,839,843,840` bytes、GPU 0 MiB，`oom=0 / oom_kill=0`。

正式 blocked 证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`。
checkpoint、native inference 和 common-observation metrics 均未启动；没有在同一实例事后加入
`--no-build-isolation` 覆盖失败。该结果属于明确 upstream packaging blocked，不是 DGGT 质量、显存或方法裁决，
并按权威计划第 15.1 节满足继续 M6 的替代前置证据。

### PIVOT-F15：AD-GS 冻结 pseudo ID 与 checkpoint 都不能支持单对象编辑

M6 直接审计训练前冻结的 `semantic/mask_*.npy`。按 camera-local ID 统计，六个官方场景最长支持帧为
`1 / 6 / 1 / 1 / 2 / 1`，全部低于预注册 `≥20/60`；processed scene 也没有冻结的 vehicle track artifact。
与此同时，六个 60k checkpoint 的 `point_cloud.ply` 均只有二值 `obj∈{0,1}`，没有持久 instance ID。

正式证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`。稳定失败
`persistent_object_identity_unavailable` 在 6/6 scenes 重复。对象编辑、pseudo-hole 和噪声行全部保留为
`ABSTAIN`，0/12 object slots 没有从 coverage 分母删除。禁止在看到 checkpoint/场景结果后用几何 Hungarian
轨迹回填 M6 baseline；这类重关联只能作为新方法候选，并必须先过 novelty。

### PIVOT-F16：instance-aware 与 driving edit 已被 2025–2026 工作直接覆盖

M7 只沿决策表考察 A“可编辑运动表示与轨迹不确定性”。重新核对官方来源后：InstDrive 已用 SAM pseudo masks
学习动态驾驶场景 2D/3D instance identity；Director 已做 4D Gaussian identity consistency；OmniRe 已用
actor scene graph/canonical vehicle nodes 做仿真；HorizonForge 与 G²Editor 已覆盖车辆轨迹操作、删除和遮挡区
恢复。

正式 evidence：
`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`。候选的持久身份、actor-centric Gaussian binding、时序一致性和轨迹/对象编辑
核心机制均为 direct overlap；confidence/ABSTAIN 是评测与安全护栏，不构成独立技术 delta，剩余差异只是
AD-GS 适配工程。因此 M7=`rejected`，不注册事后 primary endpoint；M8/M9 均
`rejected / not authorized`，不得通过改名、挑场景或把 0 coverage 写成改进继续。

