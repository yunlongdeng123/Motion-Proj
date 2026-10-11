# DGGT Waymo 官方重建与高斯编辑

task/run：`WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1`。用户确认是小米 DGGT（基于 VGGT），并授权按论文实现车辆增删和移动。Seen-to-Scene 正式10000步及六窗完成后运行了本任务；本任务没有启动12500步或修改Seen-to-Scene训练。远端run位于`/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1`。

[论文 §4.3 Scene Editing](https://arxiv.org/html/2512.03004v1#S4.SS3)明确描述移除或平移动态高斯、插入实例，并用扩散修复移除后暴露区域。公开`inference.py`提供mode 2重建与mode 3插帧，编辑CLI未公开。这里复用官方预测和渲染公式，新增高斯编辑入口；不能将缺CLI当成论文不支持编辑，也不能把新增入口说成作者已发布的接口。

```mermaid
flowchart LR
  R[Waymo连续RGB] --> D[官方DGGT]
  D --> G[预测相机 深度 高斯 动态概率]
  R --> S[官方SegFormer与RGB人工选点]
  S --> M[车辆编辑区域]
  G --> E[删除 平移 复制插入高斯]
  M --> E
  G --> N[不编辑对照]
  E --> V[官方gsplat与天空渲染]
  N --> V
  V --> O[原始RGB与alpha洞]
  O --> F[官方Difix]
  F --> Q[独立修复结果]
```

GT位姿、深度和动态标签不进入DGGT预测，也不用GT框选择编辑实例。原始渲染、暴露洞与Difix外观修复分开保存，最终质量由真实输出判断。

## 已执行的准备

- 官方代码`/root/autodl-tmp/dggt`，固定`a3276d2bbe4cbb03bcc117830b1836110a27adeb`，官方模型结构和权重不改。
- 已下载Waymo权重`model_latest_waymo.pth`为5,411,266,466字节；模型仓库固定revision`735ac9a6486057b1eb886c33a8c6dc79e0b43214`。Difix、sd-turbo、SegFormer与LPIPS缓存均复用现有资产，不重新下载。源验证见外部`.preparation/all_asset_checks.json`，不是本轮重新验证全量权重的声明。
- 官方CPU预处理已完成scene 009与128，各199帧、五相机；images、images_4、dynamic_masks各995张，位姿199份，内外参各5份。CPU TensorFlow检测GPU为0。
- 对现有8个validation TFRecord做一次首帧RGB扫描。009夜间眩光使车辆语义连成大块；主展示改为白天scene128，`segment-4612525129938501780_340_000_360_000_with_camera_labels`。仅依据输入可见性选演示片段，未看生成结果挑样本，009证据保留。
- 官方SegFormer B5在CPU处理128前8张前视RGB，输出sky_masks、custom_masks、semantic_raw19，raw19转换与前两组逐像素一致。语义标签不能直接当实例，人工ROI与anchor只作为编辑控制。
- 隔离推理环境`/root/autodl-tmp/envs/dggt_inference`复用现有Torch/gsplat依赖，不修改Seen-to-Scene环境。CUDA扩展在GPU隐藏、MAX_JOBS=1的CPU编译阶段461.85秒成功，随后官方mode 2基线及高斯编辑真实运行完成。
- 四个新增脚本编译、官方兼容副本`--help`与队列CPU预检均通过；随后四支高斯渲染与16帧Difix真实运行完成。Difix第一次运行发现共享环境的diffusers 0.31.0使`AutoencoderKL.add_adapter`缺失；仅在隔离`dggt_inference`环境安装diffusers 0.32.2后恢复，未改官方模型、权重、推理公式或S2S环境。

普通Waymo数据没有scene-flow GT深度；原始动态掩码已转换并保留，但不假造官方fine_dynamic_masks、不下载未获访问权限的数据，也不将GT仅供展示的字段填成生成条件。

## 编辑与对照合同

主运行固定scene128、首4张前视RGB、官方mode 2、FP32。官方基线保存实际模型输出、相机、point_map、gs_map、生命周期、动态概率、天空和RGB为`baseline_official/official_trace.pt`。编辑复用此trace，不再单独预测一次模型。

| 分支 | 实际操作 | 证据边界 |
| --- | --- | --- |
| noop | 同静态与动态高斯、时间衰减、相机和天空重新渲染 | 与官方trace逐值max_abs必须≤1e-4 |
| delete | 从所有来源静态高斯及每帧动态高斯移除选中实例 | 记录被移除数量和alpha损失；未观测背景可能形成洞 |
| move | 选中高斯沿预测相机右方向平移一个预测车辆宽度 | 保持颜色、尺度、旋转和生命周期；模型尺度不是米 |
| insert_copy | 保留原场景并追加平移后的同实例高斯副本 | 同场景复制插入；尚未验证跨场景车辆迁移 |

选择方式为RGB SegFormer car类13，与归一化ROI`[.52604,.53906,.79688,.82813]`相交，再用RGB anchor`[.67,.70]`选择前景MPV。首轮CPU检查发现ROI右上混入邻工程车、左上有语义误分，已在任何生成前根据RGB添加两块排除多边形，保存首四帧明确的RGB编辑mask，设置`neighbor_radius_widths=0`。原始选择与修改证据均保留。这里是人工RGB轮廓控制，预测3D位置仅记录诊断，不声称自动3D实例关联已通过；控制不来自GT框。最终选择overlay已与四帧RGB、alpha洞和编辑结果逐项核对。

Difix使用已下载作者权重，timestep199、mv_unet=false、prompt `remove degradation`，沿官方resize、sample、bilinear流程。仅将5处硬编码sd-turbo仓库位置替换为本地完整目录；一次加载模型复用16张图片，同帧不同分支配对seed1234。PNG量化在修复前发生，该适配与官方每帧重新加载的差别明示，不覆盖原始输出。

## 实际运行、工程适配和结果

`scripts/worldsim_v81/queue_dggt_waymo_inference.py --worker --scene128`完成了官方基线→高斯编辑→Difix。第一次Difix尝试因环境中的diffusers 0.31.0缺少VAE的`add_adapter`而返回1；隔离环境升级到0.32.2后只续Difix，第二次返回0。两次命令及返回码均保留；首次失败栈已移入`resolved_engineering_failures`并另存原快照，活动状态为`complete_assistant_reviewed`，不再将历史error当成当前故障。全程只做推理，没有训练DGGT或更改已下载权重。

`prepare_dggt_mode2_runtime.py`生成隔离副本：延迟mode 2未使用的TAPIP3D/插帧依赖；修复公开args.difix拼写；保存trace；跳过缺失GT动态/深度展示并保留预测深度。此副本在mode 2不调用Difix；网络、相机解码、Gaussian组装、时间衰减、渲染和天空公式保持官方实现。上游原件及各次副本保留在run/source，外部源码不入Git。

官方基线同输入重建的PSNR为28.0511、SSIM为0.862652、LPIPS为0.0823603，日志平均推理时间1.631秒；这些数值不是删除、移动或插入后的反事实质量指标。

官方scene128首4张前视RGB基线及`noop/delete/move/insert_copy`四支均完成。所有分支复用同一次官方模型预测、预测相机和天空。原生无编辑高斯重渲染与官方trace的浮点RGB逐值`max_abs=mean_abs=0`。原始manifest自报`same_8bit_quantization=true`，但独立媒体审计发现已保存的基线PNG与noop PNG最多相差1灰阶、平均约0.5/255；因此不能把实际PNG文件说成逐像素完全相同。

RGB-derived、人工排除邻车的四帧mask在模型输入之外控制编辑：静态高斯共603,830，选中218；逐帧动态高斯约179,917–180,115，选中11,700–12,032。删除过滤该实例的原始高斯；平移沿预测第0相机右方向一车宽，预测宽度约0.01888模型单位；复制插入每帧追加11,918–12,250个同场景平移副本。此数值不是米。直接的RGB mask跨四帧来自人工控制，记录的3D距离仅用于诊断，不是已通过的自动3D实例关联。

从四帧原生PNG看，delete确实去掉前方深色目标车主体，却留下明显黑灰空洞与原车下方阴影/残留，不能称作背景自然补全。move和insert_copy显示目标车位置变化，但复制或移动后的轮廓、遮挡、阴影与邻近白色作业车互相影响，并在画面右缘部分裁切；copy是同场景实例复制，未验证跨场景迁移。alpha与alpha-loss展示该洞的覆盖损失，原始编辑没有像素级修补。独立评审同时指出：官方基线本身相较输入RGB较软，不能把此退化归咎于编辑。

随后用作者Difix权重分别处理四支各4帧，共16张：固定timestep199、`mv_unet=false`、`remove degradation`，一份模型复用全部帧，结果与原生高斯输出分开保存。Difix改变了外观并平滑部分区域，但delete中的黑灰洞仍明显，move/copy的形态、接缝、邻车遮挡问题未消失；这是逐帧外观修复，不是inpainting或车辆删除算法。其16张PNG及4支MP4与原生四支均已完成独立CPU媒体审计。实际Difix模型加载8.613秒、CUDA分配峰值6,321,043,456B；高斯编辑峰值380,165,632B，不将局部异步渲染计时当端到端FPS。

深色本地查看页`outputs/v81-dggt-waymo/index.html`交付64张PNG、16个MP4，全部80条链接存在；MP4各解码4帧、8fps，JS语法检查通过，尚未做浏览器视觉QA。独立助手已将`assistant_review.json`写入run根目录，结论为**工程执行完成，但只是存在明显伪影的局部编辑演示，不能视作视觉质量通过**：四帧选中的是同一前方深色车，四分支效果持续出现，未见明显身份切换或分支掉帧；删除后未显露可信道路纹理，Difix也未修好洞、邻车重叠或右缘裁切。路面被目标车遮挡而未观测是与洞相容的解释，现有四帧不足以证明其唯一根因。该窗口无法判断长时序跟踪、稳定性和场景泛化；`human_verdict=null`。不得据此宣称论文编辑质量通过、跨场景迁移、米尺度控制或自动3D实例关联。

证据：run内`baseline_official/official_trace.pt`、`gaussian_edits/manifest.json`、`difix_refined/manifest.json`、`assistant_review.json`、`queue_state.json`、`source/edit_config.json`及准备阶段证据。已同步的本地审计为`work/dggt-official-inspect/audit.json`、`assistant_review.json`、`completion_snapshot.json`、`local_delivery_verified.json`。无新增科学根因，`failure_ledger_refs=[V77-F02]`、`failure_ledger_delta=none`；diffusers异常是隔离依赖兼容问题，已修复。

## 2026-10-11 论文协议排查与有界迭代

用户新增要求对照官方 HTML 论文查编辑质量并自主迭代。沿用同一 task/run、scene128、四个目标 RGB、原高斯渲染和作者权重，不覆盖已完成基线。Seen-to-Scene 10000→12500 在途训练保持运行；DGGT 新 GPU 诊断只安排在正式12500完整保存、固定六窗完成及控制器退出后的短间隙，随后继续下一段 Seen-to-Scene。

```mermaid
flowchart LR
  R[原四张合法RGB] --> A[来源覆盖CPU核查]
  G[原高斯trace与RGB轮廓] --> P[静态高斯投影支持核查]
  D[既有noop与delete渲染] --> S[公开单图Difix重放]
  D --> M[官方双图注意力Difix]
  D --> I[同渲染自参考]
  R --> L[固定原000参考图]
  I --> M
  L --> M
  S --> Q[四帧补洞与重新引车独立审核]
  M --> Q
```

### 论文、公开路径和本次路径

| 核对项 | 一手材料 | 当前判断与边界 |
| --- | --- | --- |
| 修复条件 | [论文§3.3 Eq.11](https://arxiv.org/html/2512.03004v1#S3.SS3)接收渲染图与输入序列参考图；[公开 infer.py](https://github.com/xiaomi-research/dggt/blob/a3276d2bbe4cbb03bcc117830b1836110a27adeb/third_party/difix/infer.py)用`mv_unet=False`且未传参考图 | 旧结果沿用公开单图路径，确有论文条件差距；尚未证明是黑洞的唯一原因 |
| 参考图接口 | [公开 inference_difix.py](https://github.com/xiaomi-research/dggt/blob/a3276d2bbe4cbb03bcc117830b1836110a27adeb/third_party/difix/src/inference_difix.py)提供`ref_image`并启用`mv_unet=True`；[mv_unet.py](https://github.com/xiaomi-research/dggt/blob/a3276d2bbe4cbb03bcc117830b1836110a27adeb/third_party/difix/src/mv_unet.py)在自注意力中合并两图token | 普通UNet把参考图当独立batch不能构成参考内容对照；下一项走作者已有双图接口，不增算法 |
| 背景覆盖 | 原四帧与11个时间点×五相机RGB接触图，另核对八个固定候选 | 后段目标车仍挡道路，侧视未证实删除洞已显露；延长时间范围不等于获得背景，当前不盲加帧重建 |
| 高斯实例筛选 | 原官方trace、603830静态点、四张RGB轮廓及预测E/K/point_map | 218个已选静态点与来源轮廓完全对应；跨源投影有149个符合轮廓与固定深度容差的未选候选，不能凭候选数确认实例身份或残影来源 |
| 论文展示可比性 | [论文§4.3/Fig.5](https://arxiv.org/html/2512.03004v1#S4.SS3)展示高斯编辑与扩散修复；附录A.2的0/5/10/15与三相机合同用于NVS评测 | 不把NVS三相机合同直接套成Fig.5编辑必需设置；当前单景四近帧、同景复制也不能充当跨景或长时序结论 |

**已完成CPU证据。** 来源候选`015_0/025_0/040_0/060_0/085_0/120_0/198_0/198_2`均为合法原RGB；没有明确看到原目标车后的待填道路。核查限于上述抽样，不声称证明全部995图完全无微小覆盖。证据在run的`coverage_cpu_obs_20261011/`，本地`work/dggt-coverage-20261011/`。

静态高斯按来源与目标建立4×4投影统计，以0.1预测车宽、即0.0018880549模型单位为单一深度容差。跨源未选候选149个，按官方生命周期加权的opacity和26.3919；其中源3→目标0为115个、24.4408。它们也可能是路面、邻车或预测几何误差；统计未计算高斯覆盖范围、遮挡及实际渲染贡献，因此不据此扩大删除mask。该审计在单CPU线程、隐藏GPU、nice10下约3秒完成，证据在`diagnostics/edit_support_cpu_20261011/support_stats.json`及同目录脚本。

已直接检查本地原生/Difix四帧局部图：删除位仍为灰黑团块、车下黑影覆盖部分斑马线。官方图像接口只返回链接，当前Cua报告浏览器不可用；**官方Fig.5像素视觉核对未完成**，不声称已逐像素比较作者图。论文文字与源码合同已核对；独立核查记录为本地`work/dggt-edit-protocol-20261011/paper_figure5_visual_review.md`。

**部署时的固定参考修复对照计划（真实执行结果见本节末）。** `probe_dggt_reference_refinement.py`在既有noop/delete各四帧上最多生成24张：公开单图重放8张，官方双图自参考8张、双图合法原000参考8张。双图两支同尺寸、batch、seed、timestep199与作者权重，落盘实际UNet入口的首图VAE latent、文本和timestep并逐值核对；参考图只从原官方四输入取，不用GT。原参考仍含被删车辆，需要同时审核背景连续性、接缝、邻车保持和重新引车，不能预设加参考一定更好。单图与双图之间的架构及随机消费差别另记，收益主要由相同双图路径的配对对照评估。发布权重是否正是论文参考条件训练后的版本尚未确立。

`queue_dggt_edit_diagnostics.py`绑定正式12500保存校验与六窗合同，使用与S2S相同的controller/launch锁，单次子诊断上限30分钟；仅自身子作业可因超时停止，不能关机或打断S2S。队列完成后需gpt-6-sol/xhigh/no-fast独立审核全部四帧，human=null并同步深色结果页。本节不把准备/排队记成真实GPU完成，也不改旧质量hold；确定新科学根因仍无，`failure_ledger_delta=none`。

**CPU预检已实际通过。** 公开双图类在diffusers0.32.2下需要旧`unet_2d_blocks`模块名alias；另一个已移除的`PositionNet`只在未启用的gated分支构造，检查本地sd-turbo为default注意力后安装本进程“构造即抛错”名称占位，未替换执行数学、未修改外部源码/权重或S2S环境。首次导入栈保留。标准与官方双图UNet在meta设备各686键，逐键形状与本地基座/作者UNet权重一致；VAE LoRA元数据90权重键、rank4/43目标，真实完整加载、首图latent配对和前向尚待GPU。预检单CPU线程、隐藏GPU，cgroup内存78.255→78.276GB。独立队列CPU隔离模拟覆盖未就绪、GPU忙、正式六窗完成且空闲、后继段拒绝、双锁冲突释放后重试五支；不把模拟当生产GPU互斥已实测。证据为run内`source/reference_refinement_r1/`及本地`work/dggt-edit-protocol-20261011/queue_cpu_validation.json`。

**初次生产部署快照（18:23；不代表当前状态）。** Git提交`61a5a11c`已推送并在清洁远端ff-only同步，现有文件先备份到run的`source/backups/reference_protocol_20261010T182242/`。canonical脚本在隔离DGGT环境隐藏GPU预检通过；实际启动控制器PID91174，UTC18:23核对`queue_dggt_edit_diagnostics.py --worker`存活及状态`waiting_s2s_12500_and_six_windows`。同次核对Seen-to-Scene父87877/子87943仍从10000正式恢复，已连续新增304更新至10304，单GPU仅87943、loss/梯度有限冻结0、5.797秒/步；未暂停或重启训练。数据盘余131,195,490,304B、系统盘6,459,707,392B。生产证据在`diagnostics/reference_queue_deployment_verified.json`、`reference_refinement_cpu_check.json`、`reference_queue_cpu_check.json`和`reference_refinement_queue.json`；本地`work/dggt-edit-protocol-20261011/deployment_verified.json`、`s2s_concurrent_snapshot.json`。新结果页构建器已独立CPU验证缺失/在途结果显示待执行、24真实输出及配对合同须齐全才显示完成；合成测试不能算生产媒体。实际新图、配对trace、GPU耗时/显存和四帧助手结论仍待12500间隙。

### 参考图对照真实完成与独立审核

UTC2026-10-10T22:03:00正式12500六窗及S2S控制器退出后，队列单独启动GPU probe；22:03:30状态`complete_pending_assistant_review`，随后独立四帧noop/delete审核和深色页完成。没有抢占S2S、重新跑高斯基线、训练权重或改变精度/分辨率。24新输出齐全，公开单图8帧逐像素重放旧Difix；双图8组首图latent、文本、timestep与batch形状实际逐值配对全部通过。真实作者UNet686键、VAE90键形状匹配并成功加载前向；单图加载8.559秒/峰6,320,594,944B，双图加载5.204秒/峰7,438,604,800B，加载时间不等于每帧吞吐。

双图自参考与公开单图在noop/delete共8张保存PNG的最大通道差均为1灰阶，平均绝对通道差0.00727–0.00864灰阶；它们近似同一可见输出，不能把双图路径本身计为收益。原000 RGB参考使删除区的车形暗块更黑、更不透明，没有补出可信道路。三支delete全部保留宽灰色车形带与下方黑斑；邻白车和工程车大体保持，无法从残影确认被删车重新引入还是原高斯洞残留。双图参考不采用，`meaningful_gain=false`、`human_verdict=null`。发布权重是否为论文最终参考训练版本仍未确立，不能唯一归因输入缺口或权重。此对照不是等预算重训消融。

初版视觉审核曾误读接触图，把单图暗团称为连续道路；直接打开12张命名delete原PNG后纠正，初版保留`review_attempt_1.json`，只把修正版`assistant_review.json`用于报告和页面。原始公共单图输出与旧结果逐像素一致，没有出现本次新补洞收益。

`diagnostics/reference_refinement_r1/`保留真实run、8组paired_inputs、三支24PNG及审核。深色`outputs/v81-dggt-waymo/reference_refinement_r1/index.html`实际解码44张输入/旧/新RGB及4张CPU分类图，107个媒体与证据链接无缺失，不重编码视频，未做浏览器视觉QA。Seen-to-Scene随后已从12500正式原件启动至15000。后续有依据的CPU来源检查及校准停止见下节；不重复参考对照或盲扫seed/阈值，不抢占S2S。仍无确定新科学根因，`failure_ledger_delta=none`。

### 官方Figure 5实际像素核对与下一CPU问题

随后当前内置浏览器可用，已直接显示并截图审阅同篇[官方HTML Figure 5](https://arxiv.org/html/2512.03004v1#S4.SS3)的全部十个原始分图，没有下载远程媒体。此事实更新了上面“浏览器当时不可用”的旧访问快照；PDF仍未打开，不声称查看过PDF页面。[输入3.1](https://arxiv.org/html/2512.03004v1/images/3.1.png)、[未修复删除3.2](https://arxiv.org/html/2512.03004v1/images/3.2.png)、[修复删除3.3](https://arxiv.org/html/2512.03004v1/images/3.3_.png)显示：作者开阔道路示例删除近白车后，未修复渲染已经显出大部分道路，远车位置只留下较小暗条；有扩散的红框区域仍有模糊和暗痕。平移示例也有车部分出左缘，第二行跨场景插车/骑行者仍可见局部边界、碎片和裁切，作者图不能作为零伪影证明。

本地四帧是城市路口近MPV，原生删除就留下整车宽的灰黑区域，alpha损失同位置明显；Difix三支都未恢复连续道路。作者和本地的对象面积、场景及未知观测覆盖不同，不能由图像差异确认单一实现错误，也不把NVS的三相机合同当Figure 5编辑必需条件；同场景复制未复现作者跨场景插入。

这一差距支持下一项有界CPU审计：仅现有trace、相机、RGB选择与alpha图，计算投影Gaussian足迹及深度层约束下删除洞内的非目标背景支持估计。旧149个中心候选没有足迹或遮挡信息，不重复候选计数，也不据此扩大mask。新统计必须标为`estimated geometric support`，不冒充真实gsplat贡献、隐藏道路GT或根因；单CPU线程、隐藏GPU、有界时间与内存，不抢占正在运行的15000训练。具体来源与独立像素记录保存在本地`work/dggt-edit-protocol-20261011/figure5_actual_primary_review.md`。

## 已完成的 CPU 足迹与深度支持审计

本轮在原四帧 trace、已激活高斯、预测相机、原 RGB 选择掩码和已保存 alpha 损失上完成一次 CPU 审计，没有 GPU、重渲染或训练。所有占比都是 **estimated geometric support**，不能视作 gsplat 的真实颜色贡献、道路身份或唯一根因。

```mermaid
flowchart LR
  A[既有已激活高斯与相机] --> B[投影协方差与足迹]
  C[RGB选择与alpha损失] --> D[洞内审计域]
  B --> E[按预测目标深度分类]
  D --> E
  E --> F[四帧分类图与统计]
```

| 帧 | 审计域像素 | 未选前/同深度 | 未选静态后层 | 未选动态后层 | 低支持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 000 | 11,198 | 6.2% | 50.6% | 14.6% | 28.5% |
| 001 | 11,285 | 4.6% | 55.4% | 4.7% | 35.3% |
| 002 | 11,362 | 4.2% | 57.5% | 4.1% | 34.2% |
| 003 | 11,535 | 4.4% | 57.3% | 5.5% | 32.8% |

低支持占域内像素的 28.5%–35.3%；按 alpha 损失强度加权为 41.6%–47.8%。50.6%–57.5% 的域内像素有未选静态后层足迹，但这不证明它们含有可用路面纹理或能通过真正的排序与透射率合成显露。图例的前/同深度也不能认定目标车漏删。掩码外另有 625–867 个 alpha 损失像素/帧，未纳入分类。

固定使用安装的 gsplat 1.5.3 投影几何：eps2d=0.3、3.33σ 半径、有效片元 alpha≥1/255；没有扫阈值。保存 state 的 static opacity 不含时间权重，本次按原 renderer 施加一次；尺度与四元数均是已激活值。单线程、空 CUDA、nice10、外层180秒/脚本150秒与4GiB限额。实际耗时1.585秒、峰RSS629,876KiB；同时 S2S 父98990/子99056持续训练。证据在同一 DGGT run 的 `diagnostics/support_footprint_r1/support_estimate.json` 和四张分类图；本地轻量详稿 `work/dggt-edit-protocol-20261011/support_footprint_r1_review.md`。`failure_ledger_delta=none`。

数据记录修正：实际每帧随机种子是 **1234+frame_index**，同帧两支重置相同 seed。此前 `seed_per_frame:1234` 是字段命名错误，已备份后改为 `seed_base:1234` 与显式规则；随机调用、权重和已完成输出未改，未重跑 GPU。JSON 的旧部署排队段显式标为18:23 UTC历史快照；参考对照、几何支持和本次校准停止分别以 `reference_refinement_result`、`support_footprint_result`、`depth_order_cpu_result` 为准。

## 深度顺序 CPU 贡献诊断：首帧校准停止

为检验上一节的几何足迹能否进一步解释实际合成贡献，沿用已保存的高斯、预测相机、RGB选择和删除 alpha，做了一次有界、单线程、隐藏 GPU 的 CPU 投影与深度排序尝试。先要求重算的 alpha 与保存的 8-bit 删除 alpha 在洞内相差不超过固定的 1 灰阶，只有校准通过才允许解释贡献。本次 UTC 2026-10-10 23:24 在首帧触发 `contract_stop`，进程按合同返回码 1；没有继续帧 1–3，没有落盘贡献 NPZ 或结果，也没有修改 RGB 公式、重渲染或启用 GPU。正式 Seen-to-Scene GPU 仍仅由 PID 99056 使用。

```mermaid
flowchart LR
  A[既有高斯与相机] --> B[CPU投影和深度排序]
  B --> C[估计alpha累积]
  D[保存的删除alpha] --> E{8-bit alpha误差≤1?}
  C --> E
  E -->|首帧未通过| F[合同停止：无贡献结论]
```

首帧洞内域 11,198 像素，处理 92,196 个片元、875,015 次包围框像素访问；重算 alpha 的最大绝对误差为 16 灰阶，2,606 像素超过 1 灰阶门限。执行用时 3.511 秒、脚本记录峰 RSS 600,564 KiB；外层 0.2 秒间隔进程树采样峰值 597,464 KiB。这个结果说明本次 CPU 近似**未通过与已保存渲染的校准**，不能把它当作真实 gsplat 贡献、车辆删除黑洞原因或新科学根因；此前四帧几何支持统计也继续只作估计，`failure_ledger_delta=none`。比较对象本身是量化后的 8-bit alpha，且 CPU 投影、排序及浮点累积可能与 CUDA 光栅化不同，单凭此次偏差无法定位哪一步造成差异。

固定版本 [gsplat v1.5.3 `rendering.py` 716–725 行](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py#L716-L725)确认 `RGB+ED` 只对最后的累加深度通道除以渲染 alpha；[`RasterizeToPixels3DGSFwd.cu` 前向实现](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu#L126-L166)按透射率与片元 alpha 累加颜色/深度通道并输出 alpha。关于外层是否可能再次乘 alpha 的 `double-alpha` 疑问仍待独立证实；此次 CPU 校准停止不能判定官方实现有 bug。远端保留诊断脚本、输入和失败栈；本地轻量证据是 `work/dggt-edit-protocol-20261011/depth_order_execution_receipt.json`、`depth_order_contract_stop.json` 与 `depth_order_depth_order_cpu_stderr.log`。本次不重跑。


## 透明度合成实测与用户转向（2026-10-11）

用户要求暂停 Seen-to-Scene、保存最新完整断点并集中 DGGT；随后明确删除定时任务。当前训练已停止，14000 完整断点验证保留，最后日志14386后的386步未保存；`v8-1-p1-2500` 已通过应用工具删除，不再按旧20000计划唤醒。用户给出的静态背景、实例漏删、透明度合成三项是待分别验证的假设，不能先判定表示能力不足。

固定官方已有四帧、同一份激活高斯/相机/时间权重/实例选择，只做8次gsplat RGB+ED渲染（noop/delete各4），没有DGGT前向、训练或Difix。四帧同一G/A/模型背景S比较两种合成：官方 `A*G+(1-A)*S`，与去掉外层A的诊断 `G+(1-A)*S`。gsplat已对G按alpha与透射率累加，因此外层A确实再次抑制部分透明颜色；本轮只量化其影响，不修改官方源码/旧编辑输出或宣称最终训练权重采用了错误合同。

```mermaid
flowchart LR
  A[既有高斯与预测相机] --> B[同次gsplat渲染]
  B --> C[累计颜色G与Alpha A]
  D[模型背景S] --> E[官方 A乘G 加背景]
  D --> F[诊断 G直接加背景]
  C --> E
  C --> F
  E --> H[四帧逐像素配对与道路审核]
  F --> H
```

新noop浮点对旧noop和官方trace均最大差0；8张RGB与8张Alpha保存PNG逐像素重放0差。第一次启动因PATH未含现有Ninja在首个CUDA扩展加载时停止，0个完成渲染、无组件trace或图片；保留原run和栈于`diagnostics/alpha_compositing_pair_r1_startup_failure`，仅恢复已用官方队列环境后完成一次8render。实测8.119秒、峰CUDA252,608,512B，模型/选择/公式预算不变。

|帧|固定洞像素|删除Alpha均值|官方洞亮度|去掉二次衰减|增加的8-bit亮度|
|---|---:|---:|---:|---:|---:|
|000|11198|0.3133|0.1404|0.1541|3.51|
|001|11285|0.2416|0.1346|0.1490|3.69|
|002|11362|0.2783|0.1361|0.1509|3.79|
|003|11535|0.2868|0.1351|0.1512|4.09|

亮度按固定洞域RGB的Rec.709权重统计、范围0–1；洞域沿用原RGB选择、已存量化Alpha损失与预测正深度。这里的Alpha是实际gsplat输出，和此前CPU几何足迹估计不同。低平均Alpha说明当前删除区很大程度依赖背景混合，但不证明所有被遮道路都未被观测，也不判断剩余高斯的物体身份。

独立助手实际查看全部28张命名原PNG（4帧各输入/sky/noop两支/delete两支/差图），四帧去掉重复衰减后仅稍亮；整车宽的灰黑横带、下方黑斑、接缝仍在，没有连续可辨道路或斑马线。**重复衰减的颜色影响已确认，但它没有解释主要结构失败；不能据此判DGGT表示不足。** 人工verdict仍null。深色页`outputs/v81-dggt-waymo/alpha_compositing_pair_r1/index.html`、执行与审核JSON、28PNG完整解码；浮点组件只留远端。页面尚未做浏览器视觉QA。

下一项按用户方向固定frame000，关闭Difix，拆静态与动态删除前后、联合Alpha/预测深度和白底。只用于区分残留内容与低覆盖，单独层颜色不能当联合渲染真实贡献，白底变白不代表恢复道路。`failure_ledger_delta=none`；尚不新建科学根因卡。


## 按用户方向拆层：固定首帧与官方编辑质量复核

用户要求关闭Difix，先排查静态背景、目标漏删和合成，不认定DGGT表示不足；同时提出官方项目页自身编辑较差，需核对公开最佳证据后再决定路线投入。本次仅首帧000，沿用同一高斯/预测相机/原选择，5次RGB+ED分别为静态删前/删后、动态删前/删后与联合删除。没有改变mask、阈值、权重、输入或时间权重，没有网络前向、训练、Difix。联合G/A对上一真实配对最大差都0，旧删除RGB/Alpha逐像素0差，完成25张原生图；5.187秒、峰CUDA206,799,360B，结束后GPU空。

```mermaid
flowchart LR
 A[已保存高斯与原选择] --> S[静态删前与删后]
 A --> D[动态删前与删后]
 S --> R[独立层与联合gsplat渲染]
 D --> R
 R --> O[累计RGB / Alpha / 预测深度]
 O --> B[黑底与白底配对]
 B --> Q[是否有可辨道路或目标残留]
```

|层|洞内平均Alpha|黑底亮度|有效预测深度像素/11,198|
|---|---:|---:|---:|
|静态删前|0.1811|0.0428|6085|
|静态删后|0.1808|0.0427|6085|
|动态删前|0.9847|0.2329|11198|
|动态删后|0.1571|0.0093|5294|
|联合删除|0.3133|0.0464|8001|

Alpha为实际渲染输出的均值，**不是有覆盖的像素百分比**。有效深度只是finite且>0、Alpha≥1/255的显示合同，不证明真实道路；五层共享2–98百分位灰度范围0.44274–2.88291，原float范围0.43291–4.03030保留，不给米制解释。单层独立渲染不等于它在联合深度排序中的真实贡献。

独立助手实际查看全部25层PNG和3张输入/官方删除/sky参照。静态删前后几乎不变，在原目标处都没有连续可辨道路，静态218个被删高斯没有造成大幅Alpha变化；动态删前可辨目标车，删后可辨车身消失。联合白底在主要车形区域形成亮缺口，下部仍留暗片；因此灰车形轮廓不能直接认作“漏删车辆”，也不能把残余动态覆盖直接叫路面或车影。原模型sky下部接近黑，低Alpha混入该背景会形成明显暗观感。**当前首帧优先支持“可用背景渲染不足”，不支持“本来有完整道路、只是二次Alpha把它压黑”这个简单解释。** 未排除局部选择/几何/遮挡错误、其他时间/视角的合法可用背景；不外推成整个前馈表示失败。

公开证据另作核对：用户提供的项目页介绍截图确有黑碎片。浏览器直接打开[官方项目页](https://xiaomi-research.github.io/dggt/)及其[删除车辆4秒视频](https://xiaomi-research.github.io/dggt/resources/video/edit2_1.mp4)，末帧（currentTime=4）同样出现明显灰色弧顶和黑块。项目页没有逐视频raw/refined标签，不能把此片段称为最终Difix效果。已复看[论文v1扩散后删除图3.3](https://arxiv.org/html/2512.03004v1/images/3.3_.png)，近处路面较连续，红框处仍有模糊和暗痕；[同例修复前3.2](https://arxiv.org/html/2512.03004v1/images/3.2.png)近处已经有道路，不能推断其成功补出了本地这样的大洞。只通过浏览器直接显示，没有下载远程媒体。原图片分辨率不同，未对跨图大小或质量做伪定量。

[论文v1 Figure5/§4.3](https://arxiv.org/html/2512.03004v1#S4.SS3)是能明确区分有无扩散的公开编辑定性证据；所核对的论文/项目页没有编辑专属背景GT、删除洞质量、成功率或时序一致性定量。重建/NVS的PSNR/SSIM/LPIPS和相应扩散消融不能当编辑分数。公开README提供重建/插值推理而非可复现Figure5编辑CLI；CVPR最终PDF网页请求403、未读取，不以arXiv v1代称完整审过最终版。

本轮建议：DGGT保留为快速重建/几何初始化及编辑速度对照，**不再作为已接近可交付的高保真编辑主线**；这是一项基于公开展示和已测输入的研究投入判断，不是证明所有前馈路线的质量上限。用户要的静态/动态/Alpha/深度/白底检查和公开案例核对已完成；当前没有新GPU队列，不追加同类Difix或透明度扫描。下一研究方向仍需有具体证据/协议，不机械扩实验。

新总览`outputs/v81-dggt-waymo/diagnosis_review.html`连接分层、四帧Alpha、旧参考修复及官方来源；分层页25新PNG+3原参照实际解码、31本地链接0缺失，Alpha页28PNG解码，汇总页11本地链接与7原PNG引用检查通过。助手原记录、run、float trace在`diagnostics/deleted_layers_frame000_r1`与`diagnostics/alpha_compositing_pair_r1`；PT只留远端，不入Git/人工页。页面未做浏览器视觉QA。`human_verdict=null`、`failure_ledger_delta=none`，不新增未经确证的科学根因卡。Seen-to-Scene暂停与14000恢复入口保持，定时任务已删除、服务器开机。
