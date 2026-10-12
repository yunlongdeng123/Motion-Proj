# v8.2 DrivingEditor 论文版调试交接

整理 task/run：`WS-V82-DRIVINGEDITOR-DEBUG-HANDOFF-20261012/r1`。研究分支：`research/worldsim-v8.2-drivingeditor-debug`；继承 v8.1 commit `c21547193d765e8f8123b27b73e31e033295a316`。本记录汇总已经产生的证据，整理期间没有新增训练、推理、实验或关机操作。

用户已要求停止研究并交付代码；`drivingeditor-50k` 自动化已暂停，其他场景队列保持暂停。**整个调试任务未完成，论文复现和编辑效果均未通过验收。** 四个 32k 恢复诊断已经完成，不能把这些子任务的 exit0 当作整个研究完成。历史 JSON 中的 `diagnosing`、ACTIVE 和待做计划是当时快照，不授权恢复任务。此前“完整排错完成后才关机”的条件没有满足；本次不执行关机。

[轻量结果](DRIVINGEDITOR_DEBUG_HANDOFF.json) · [Git 内证据目录](../autoresearch/worldsim_v82/drivingeditor_debug_20261012/r1/) · [兼容与诊断patch](../../compatibility/DrivingEditor-v82/) · [排错工具](../../scripts/worldsim_v82/drivingeditor_debug/)。原深色报告保留在用户本地 `outputs/drivingeditor_vgpu_debug/index.html`，其交付 SHA256 为 `68a2bba9082da4f0e92abef0354808094eb2d2ae599e2011766e2d24fff54419`；媒体、完整 checkpoint 和原始数据的位置以证据目录中的资产目录为准。轻量 Git 目录不等于大型资产的完整备份。

## 输入、组件和实际输出

```mermaid
flowchart LR
  I[49 帧 FRONT RGB LiDAR 位姿] --> P[车体与天空 mask 坐标变换]
  P --> S[静态高斯 SGCM NPS]
  P --> D[动态高斯 RGB opacity]
  S --> R[光栅化与共享 EnvMap]
  D --> R
  R --> C[完整合成与原损失]
  C --> O[固定49帧 GT composite static dynamic]
  O --> A[道路 分支范围 目标代价审计]
```

主研究只覆盖 Waymo `scene0017_000_048` 的银色轿车，FRONT raw0–48 共49帧，seed0，原定训练 horizon50000。正式参考 run 为 `/root/autodl-tmp/drivingeditor_repro/runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline`；冻结源为 `/root/autodl-tmp/drivingeditor_repro/campaign/resume_vgpu_baseline_20261011/frozen_repository`。调试源、计划、日志和中间产物在 `/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012`，SSH 为 `wm-vgpu-1011`。

论文版适配和作者公开实现需分别标记。已有只读核查于2026-10-12确认公开仓库 `WangXu-xxx/DrivingEditor` 只有 main `560d57c3825ed2bdbc6a6602e0011ad51e8da60e`、无公开 tag；这不能证明私有或未发布的论文代码不存在。当前 body mask 来自 DEVA/SAM 人工核对及点投影，不是已复现的 M-detector 与 shadow 流程。

## 旧3090与vGPU reset边界

旧3090参考 `/root/autodl-tmp/drivingeditor_repro/runs/0017085_demo_000_020_dynamic_30000` 使用原2024代码、21张FRONT、30k horizon。历史 **PSNR32.7914 dB来自 reset 前的评价输出**，不是重新载入 saved30k 的成绩。该 saved30k post-reset 约9.98dB只作诊断；不能与新49帧曲线合并，也不能把两者称作同一评价合同。

当前代码在15k结束增密后仍每3000步重置静态 opacity；30k保存的2,234,298个静态点 base opacity 均≤.01。训练scalar在reset步记录的是reset前forward，保存与paper eval发生在reset/optimizer之后。现有loss审计中30k scalar34.513dB、下一步8.591dB、保存后评价10.4835dB展示了这个顺序差异。

这是已确认的调度与评价状态问题，但不能解释20k已存在的静态道路彩噪。B1和B2从同一saved30k继续2000步，完整composite恢复约36.43dB，静态道路仍差、动态层仍包含背景，因此不能将整个分支问题仅归因于reset。旧30300短诊断及其全部正反结果保留，最终比较使用32000。

## 四个32k终点与收益代价

四支均从同一30k参数、Adam、RNG和采样状态恢复，在30001–32000完成2000步；保留原50000 horizon的LR安排。B1为原目标，B2为原目标的一次重复；V只加系数.05的车体外动态opacity项，S只加系数1的已观测静态RGB项 `.8 L1 + .2 DSSIM`，没有V项。S项不在当前展示的paper Eq10内，属于诊断干预。

已有receipt记录首步原目标对齐、实际backward、2000步采样/time_shift/LR配对、30001/30300/31000/32000四节点RNG安全评价、完整终点保存以及各支exit0。每支终点保存49帧、6通道共294张PNG。固定预选raw0/10/17/20/24/30/40/48保留；V的raw45和S的raw44/48是终点后按不利误差补选，明确披露，全部49帧没有删除。

下表为已锁定49帧等权均值，RGB误差使用clamped RGB；“车体外”均指非天空、body mask外的既有观测域。pixel-weighted/pooled值单列在原receipt，不能与等权均值混用。

| 指标 | B1原目标 | B2原目标重复 | V车体外opacity | S观测静态RGB |
|---|---:|---:|---:|---:|
| 全图composite PSNR/dB | 36.43076 | 36.43047 | 35.90406 | 35.44799 |
| 全图composite MSE | .000236244 | .000236160 | .000266072 | .000300040 |
| 车体外static PSNR/dB | 11.96038 | 11.96283 | 27.33124 | 31.76342 |
| 车体外static MSE | .06398397 | .06394976 | .00192824 | .000687911 |
| 车体外dynamic α均值 | .675479 | .675413 | .113536 | .516351 |
| 车体外dynamic RGB支撑比例 | 99.86136% | 99.85967% | 79.36881% | 79.01255% |

V和S对B1/B2的static MSE均49/49改善，全图composite MSE均49/49升高。V的actor composite MSE对两原版均49/49升高；S对B1有46帧、对B2有47帧升高，44帧同时升高，pooled actor误差也升高。S在既有5像素core和inner-boundary两域的pooled composite误差均更高，因此代价不只在边缘。单次B2重复只描述本次原目标波动，不是通用CUDA噪声上界、置信区间或显著性检验。

实际图像确认V/S道路彩噪明显减轻。S的composite与static差异在已观测车体外的支撑从原版约99.34%降至31.19510%，绝对差均值.00680946；这表示影响范围缩小，不能等同于移除正确。S的动态层仍含道路、横道线和树木，raw48 static左缘保留车形彩色残片。V目标动态alpha约.961、RGB支撑仍1，没有全局动态塌缩，但车身颜色、边缘和细节有代价。颜色支撑、alpha与移除影响范围是不同测量。

S比V有更好的静态RGB，V有更小的全图composite误差；两者附加目标不同，只能描述折中，不能声称S→V是单变量因果比较。已有观测内道路可恢复支持继续研究监督与分配问题，却没有建立从零训练、完整50k、多场景或MOVE成功。没有dense edited GT，body mask外可能含shadow。

## 已确认问题与尚未确定原因

| 项目 | 实际证据 | 可以得出的结论与边界 |
|---|---|---|
| 静态reset与保存顺序 | 15k以后每3000步继续reset；30k透明度、前后scalar及保存评价 | 调度不稳定和状态差异已确认；不是20k彩噪的完整原因 |
| 稀疏深度重复像素 | 同一点集倒序改变last-write深度及Eq16梯度；实际3帧8.09%–11.73%有效像素被更远点覆盖 | 继承投影的顺序依赖bug；nearest有效正z是候选几何策略，不是已证实的分支修复 |
| SGCM不可见重复clone | 20k相同xyz额外副本999,266，最大组158,570；773,564在整个[-.48,.48]内低于实际alpha cutoff1/255 | clone准入忽略有效贡献造成资源退化；这些始终不可见副本不能产生RGB彩噪。visibility guard默认关闭、没有从零GPU收益验证 |
| 原目标的分配歧义 | 实际loss/SSIM/compose的逐像素构造中，合理分离与static空/dynamic全背景均可达约1.25e−10 | 实现目标无法唯一识别分层；不证明有限共享3D高斯可实现该构造，也不确定实际优化轨迹或作者意图 |
| 训练与导出dynamic不同 | 训练dynamic渲染带共享环境残差，导出actor通道为raw actor | 这是诊断语义差异；共享环境训练与paper文字一致，不是全局composite评价bug |
| EnvMap彩噪 | 20k分量图：static foreground暗，static env contribution含道路彩噪；分支/环境混合诊断保留 | 是实际噪声所在分量；不能唯一归因EnvMap权重、初始化或不可见clone |
| NPS、坐标、索引、mask | 已有CPU公式/梯度/旋转/optimizer索引及49帧对齐核查通过；米→模型scale=.1 | 在所查合同内未发现确定错误；不认证未公开作者拓扑或损失超参 |

paper Eq16跨页明确使用稀疏LiDAR与渲染深度的**逆深度**。因此逆深度本身不能列为bug。harmonic环境深度融合与sky_depth900是继承实现选择，论文未给出对应细节；alpha归一化和米制scale已核对，不擅自改成“作者正确公式”。

## 静态时间语义与初始化/预处理审计

32k固定raw10 pose/rays，分别读取49个原始模型timestamp并在raw0/48做只读位置或temporal opacity/support冻结。B1/V的static RGB在49时刻相对anchor最大区域FP32 MAE为.100004/.056647，确认当前static输出随时间变化。两个端点中冻结opacity/support的输出变化强于冻结SHM位置；该控制不是可加和的因果比例。

raw0时间−.48与raw10 anchor−.28相差一个.2周期，位置近相位等价，不能据此说所有时刻运动不重要。固定pose与其他timestamp是反事实组合，没有非anchor GT质量评价；`freeze_both`与anchor完全一致验证控制，不代表修复或作者要求hard-freeze。Eq4称static为time-invariant，Fig3又讨论抑制static速度；准确作者约束仍不明确。保存depth统计仍为model单位、可视化米深度除以.1；timestamp不冒称真实秒。

初始化回放精确重建seed0对8,667,791个原始return的600,000次采样和全部36,406 actor PLY行；原.1m剔除后static595,894行，抽中的2,568 actor return全部剔除，两个初始化分支没有这些选中return的精确重复。

但是native TOP点标签仅覆盖raw25/30/35/40/45。通过同时刻目标框和已有actor return记录派生目标实例链接后，static仍有169行/160个原始return被native点标签确认属于目标车，构成**0.02836%的见证下界**。541,052行（90.7967%）缺直接语义身份；884个目标OBB关联行不能全当污染，其中717无点标签、一行ROAD，还有3个目标实例见证位于框外。来源帧中163/169在FRONT外、6行在FRONT内但body mask外；独立投影审计有143行/137原return在至少一张FRONT具有中心投影资格。没有协方差足迹、遮挡或实际光栅贡献，不能确认其为彩噪原因。

actor PLY可用native标签为1692 CAR、24 ROAD，另4 UNDEFINED与34,686无标签；24 ROAD均在body mask内且目标框外，离既有拟合地面.036–.140m。ROAD不是shadow或motion标记，也不能用这24点独立解释整个动态道路层。

论文引用M-detector与shadow预处理，目前尚未产生真实detector预测或shadow输入。只读M-detector源码审计确认Waymo配置、local点云与odometry输入，filter使用 `R*p_body+t` 和scan_end_time，节点FIFO配对需adapter保证cloud/pose对应。早期host probe的ROS/PCL/Eigen不可用只是历史环境快照；**后续已有隔离native core/adapter编译成功回执**：20个vendor文件重验均未改，使用私有PCL真实headers/共享库绕过PCLConfig的未用Boost二进制导入要求，configure与compile均exit0，静态库686,864B。它只证明编译成功，没有构造detector、处理扫描或生成标签，不能称M-detector复现完成。

后续时间/几何交接仍明确 `author_equivalence=false`、I readiness=false。Waymo帧timestamp是scan/frame start，Frame.pose约在扫描中点且不对应该stamp；原converter的第六列是TOP range pixel index，并非逐点时间。TOP几何已由SDK pixel-pose补偿到原Frame.pose参考body，不能二次去畸变或擅自猜.05/.1秒偏移。自定义 `WAYMO_FRAME_START_REFERENCE_CLOCK_SECONDS_V1`可作为显式参考时钟设计，但作者输入等价性、scan-start与mid-pose时间合同及实际detector标签尚未验证，既有receipt没有批准native构造/推理或训练。

shadow工具公开流程用shadow微调SAM ViT-B及LSTN，现有通用SAM body mask不等价，真实权重下载与shadow提示未验证。GT/native标签只作审计控制，不能冒称M-detector输出或拿GT框清洗来认证论文复现。

I（双分支初始化）、H（shadow）、T（静态时间）均仅为未执行的后续设计。I readiness仍false，genuine detector输出和scan-time/pose合同未建立；没有因本次整理安装检测器、下载数据、改训练输入或恢复GPU队列。

## 证据和恢复入口

遵循仓库第三方源码留在外部checkout的约定，Git内 `compatibility/DrivingEditor-v82/` 保存author2024→frozen_baseline、legacy_compatibility、paper_working_copy及frozen_baseline→V/S的patch，`scripts/worldsim_v82/drivingeditor_debug/`保存已有排错工具。冻结实现不宣称等于完整作者paper代码；候选是否执行以对应plan、status和receipt为准，不能由文件存在推断已验证。原始完整source snapshots保存在本地/远端 `drivingeditor_v82_remote_code_evidence.tar.gz`，原目录与Git入口的精确映射由root交付目录记录，Git不保存重复第三方checkout。

Git内 `docs/autoresearch/worldsim_v82/drivingeditor_debug_20261012/r1/evidence/{debug,provenance}` 保留四臂summary、锁定指标、首步/恢复比较、RNG与公共输入记录、terminal manifest、独立数值/实际图像解读，以及reset、SGCM、depth、static time、初始化、M-detector和shadow审计。完整TFRecord、PLY/lineage、FP32数组、checkpoint/optimizer/RNG、全量图像和视频保留在原本地/远端位置；资产目录记录原路径、角色、字节和既有receipt，不能写成所有资产已入Git。原深色HTML和媒体以本地输出及交付zip保存，不在本文创建不存在的Git网页入口。

本地交付的原深色报告已有138个引用通过静态核查，B1/B2/S新同步视频49帧完整解码；root查看了实际decoded raw10及native raw48。没有浏览器播放验收，不能补写。整理期间只核对已有证据和文档，不复跑历史test、CPU科学审计或GPU实验。

## 失败账与新知识

相关历史边界为 [V76-F03](../research_failures/entries/V76-F03.md)：场景重建指标不等于对象资产可编辑资格。该卡的HUGSIM/VAD-GS分母与用户路线判定保持原义。 [V77-F02](../research_failures/entries/V77-F02.md)记录的是DriveEditor扩散补景与输入合法性，名称近似不能把它当作本次DrivingEditor实现的同一失败。

原诊断receipt和首次交付r1的 `failure_ledger_delta=none` 保持历史原值，没有确立新的跨方法科学否定。后续用户于2026-10-12明确要求登记[工程卡V82-F01](../research_failures/entries/V82-F01.md)：论文适配替换opacity目标时丢失掩码外透明度监督，以及在正式长训前分层验收不足；并指定[后续默认双域监督](DRIVINGEDITOR_OPACITY_DEFAULTS.md)。这是新增工程知识，不改写既有实验receipt，也不将作者公式未交代监督域写成作者实际实现已遗漏。原r1人工 `human_verdict=null` 保持，新的用户工程指令在卡中单列，不代填质量评分。

本轮增加的知识是：**同一已有观测和模型可以恢复静态道路，但静态收益、目标保真和整体重建仍有明确折中；reset瞬态不能解释全部分层失败。** 这支持保存分支级证据，并保留监督、初始化、预处理和时间语义的开放问题；没有足够证据认证完整论文复现或高保真反事实编辑。
