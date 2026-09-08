# V7.3：实验可比条件与技术报告主张边界

状态快照：2026-09-08 10:44 UTC。R10/PID53472第21轮、R14/PID68108第3轮继续；R11完成，R12已登记尚未启动，R13一轮真实输入训练完成。本文整理允许的比较，不是新增验收流程、不触发重跑。用户最新方向以计划revision4为准：三角比较若仍显示Query coverage强但physics差，下一轮优先改Query surface parameterization，保持可训练几何基座/显式表面/物理约束。原run与三本台账保留执行证据。

## 输入、训练标签与评价是三件事

主population固定为31窗口/25日志的489个刚体Actor，414 FIT来自20日志，75 development来自5日志。模型输入为4个build时刻的多相机图像、稀疏LiDAR、标定和只读轨迹；每窗口24视图、378×672。训练使用的后续测量不因此成为模型输入。完整population仍保留51个零build LiDAR对象、无相机位姿对象及无heldout owned返回对象。

短窗all_window标签来自既定窗口，full_track标签来自同一FIT Actor的完整可用轨迹；development不扩成训练标签。414份full_track文件共有1690284个可用测量点，但这不是每个训练都实际消费的数量：旧输入路线只训练371个FIT对象，采样又受每步1024个目标/512条原束限制。每个run的呈现、抽样和真实更新数应分别报告，不能拿整个标签池数量冒充优化观测量。

表面损失是测量target→surface覆盖，不是对稀疏target做无条件双向Chamfer。自由空间来自原始首回波前的已观测束；原首回波后的区域仍未知。规范点经Actor在相机曝光时刻的已知刚体变换投影。输入构建已将属于刚体的LiDAR端点从LiDAR时刻搬到相机时刻，去除重叠归属并在相机像素做最近z选择；这不等同于密集真值可见性。

## 主路径及强控制

以下R编号均指`WS-V73-M2-GLOBAL-ACTOR-01`，不是同编号的场景或数据run。共同输入cohort/空间查询大小/原束语义保持，具体差异列于表中。

| 方法 | 模型通路 | FIT几何标签 | free/event目标 | 当前状态及更新预算 |
|---|---|---|---|---|
| R5 joint | M1r3全DPT＋空间查询 | 短窗all_window | hard range .5 / event 0 | 完成30轮，11130个有效更新；曾中断并恢复 |
| R6 LiDAR-only | 同query解码器，视觉关闭 | 短窗all_window | hard range .5 / event 0 | 完成30轮，11130更新 |
| R7 LiDAR-only | 同R6 | full_track | hard range .5 / event 0 | 完成30轮，11130更新 |
| R8 LiDAR-only | 同R7 | full_track | finite-beam range .5 / event 0 | 完成30轮，11130更新 |
| R9 LiDAR-only | 同R8 | full_track | finite-beam range .5 / event .01 | 完成30轮，11130更新 |
| R10 joint | 同R5，从M1r3重新初始化 | full_track | hard range .5 / event 0 | 正在训练，目标30轮；没有最终比较 |
| R11 native_only | M1r3全DPT，native＋LiDAR PCA表面 | full_track | hard range .5 / event 0 | 完成30轮，11130呈现/10550实际更新/580无梯度跳步 |
| R12 joint | 同R10，从相同M1r3初始化 | full_track | finite-beam range .5 / event 0 | 已登记，未启动 |
| R14 native_only | 同R11，从相同M1r3初始化 | full_track | finite-beam range .5 / event 0 | 正在训练，目标30轮；无梯度呈现不计更新 |

joint和native_only的直接native数据项权重均为1，来自当前build Actor在正确相机像素上的真实LiDAR轴向深度；不依赖预测框内候选是否存在。它不是旧位移标签，也不是冻结深度自蒸馏。DPT训练参数32654562；joint另有1670517个query参数。全共享路径使用AdamW lr1e-5与全局梯度裁剪1；LiDAR-only没有DPT梯度。不能仅因同epoch数就宣称相同训练计算量。

主解码器读取原DPT四级中间特征，早期聚合器前缀冻结并缓存；DPT输出在优化期间重算。query使用局部近邻交互、带视图身份/时刻/方向的投影邻域读取、可更新三维位置与固定尺度曲面片。当前query的读取mask表示在相机前方、图像/有效裁剪范围内，未构成密集遮挡真值。曲面片内部连通，但相邻query片未显式焊接为闭合流形；不作无裂缝或完整拓扑保证。

## 每一对比较能回答什么

| 比较 | 可回答的问题 | 不应作的归因 |
|---|---|---|
| R7−R6 | 同LiDAR解码器扩大FIT标签时间范围的效果 | 视觉或空间查询带来的收益 |
| R8−R7 | 同模型/标签下finite-beam free代理的变化 | 事件似然、更多输入或新几何架构的收益 |
| R9−R8 | 在同finite-beam模型上增加event .01的效果 | event单独解决无表面支持，或已经证明场景优势 |
| R10−R5 | joint由短窗标签改为full_track的主要设计影响 | 完全相同随机轨迹下的严格数值单变量试验；R5存在恢复RNG差异 |
| R10−R7 | 同full_track下视觉预训练联合路径相对LiDAR路径的整体增量 | 单独归因为某一个attention层或视觉特征维数 |
| R10−R11 | 同标签/目标下query生成通路相对原生头/PCA融合的增量 | 自动把生成参数化、法向/片形状等差异都归为三维消息交互 |
| R12−R10 | 在joint本身检验finite-beam free，而不是外推LiDAR实验 | 同时改变输入cohort、event权重或增加LoRA后的混合收益 |
| R12−R14 | 同finite-beam目标下Query生成相对原生PCA融合的整体增量 | 只归因于attention，忽略表面参数化/法向/连通差异 |
| R14−R11 | 在原生控制本身检验finite-beam free | Query或新输入条件的收益 |

完整489 cohort的同容量pointwise控制尚未训练；旧25 Actor控制不足以证明population上的空间交互因果收益。若主query候选成立，该控制需要在同监督与生成参数化下比较。R11使用hard range，R14才是已启动的同finite-beam原生对照。R10−R14同时改变生成与目标，不能当单因素；R10/R12/R14三角与R11锚点共同解释结果。

若三角比较后仍是Query覆盖占优但物理首表面不佳，按用户revision4决策优先改Query surface parameterization；原生基座仍可训练、物理损失和硬首交点评价继续。先查顶会与优秀官方开源，再结合实际片形状/方向/重叠/连通等失败迁移一个新候选；不继续以loss扫描为主、不因负结果默认换成native-only。该判断不是已完成比较的结论，也不是新门控。

R5原进程在第22轮退出，保存到第21轮7791次更新；107个未保存更新保留在原日志。恢复又完成3339次更新，checkpoint有效更新11130，实际执行11237。旧checkpoint未含完整RNG，恢复明确重启CUDA seed并重放Python顺序。因此保留这项执行差异，不自动发起重复训练，也不把107次未保存更新重复计入最终模型。

## 其他基线的适配方式不同

| 方法 | 使用观测及优化方式 | 表面读出与解释边界 |
|---|---|---|
| LiDAR PCA | 原build LiDAR，无网络优化 | 固定.06m PCA片；观测稀疏且无点对象保留缺失 |
| M1r3 native融合r2 | M1r3已用FIT build深度训练；之后固定权重 | native＋build LiDAR，0.06m PCA片；不是full_track原生强控制R11 |
| CAPA r2 | 每个窗口重置LoRA，FIT与development都做100步build TTA；没有跨窗口权重继承 | 官方per-image affine scale/shift与主方法固定build尺度不同；相同原信息来源不等于同一监督组织/校准自由度 |
| AdaPoinTr r2 | 官方PCN预训练全模型32494657参数；371个FIT对象、full_track、30轮，11130呈现/2790累积更新 | 原生16384点；主评价按共同预算采样并转PCA片，另报原生点集与全密度片敏感性 |
| Actor TSDF r1 | 全489对象，仅build LiDAR，无学习；.1m/.3m、观测角MC、无补洞 | 原生网格和单次build雕刻都保存；密度不同且严重缺失，只作为本配置适用性诊断 |

CAPA实际优化早期patch编码器qkv LoRA，rank4/alpha8，393216个可训练参数；每步随机3/24视图，最终全部24视图联合推理。它使用重新计算的编码通路，不能写成读取旧最终特征却微调了其来源。官方对齐的精确分块保留全部选定anchors，不把运行时分块当新方法贡献。CAPA属于本任务的稀疏观测适配及规范融合迁移，不是原论文动态Actor补全指标的精确复现；TTA延迟应单独列示。

AdaPoinTr r2采用PCN y-up坐标桥接(x,z,y)，对所有输出逆变换；标定、真实坐标、轨迹和评价目标不变。它使用AdamW lr1e-4、累积4、clip10和额外coarse/denoising项，不能被称为与joint同优化器预算的单因素架构试验。对完整16384点增密后，覆盖和命中会上升，early/free也会上升；不同密度不能混入同预算主张。

普通LoRA“给大模型加低秩参数”本身不是贡献。当前主joint适配全DPT，尚未证明联合上层聚合LoRA带来的收益；若将来改变可训练前缀边界，必须重新计算该边界之后的特征，不能沿用旧最终缓存并作微调主张。

## 零LiDAR输入条件是单独的变化

原R5/R10/R11/R12/R14默认训练371 FIT、预测67 development可用输入对象，同时把51个零LiDAR对象留在评价分母并输出缺失。native融合与CAPA则可能在零LiDAR时仍有native表面，因此它们与旧joint之间存在输入入口差异，不能当作同样缺视觉。

显式visual-only固定推理已经完成：41个FIT/5个development对象可生成表面，但旧R5从未在这一输入条件训练，FIT free明显恶化。R13随后从M1完成全部51对象子集的一轮41次真实更新，两组梯度均为正；7个无正目标对象中5次只有envelope梯度，开发唯一归属束从miss变early。它确认新训练入口，不代表充分拟合或泛化。完整新cohort将纳入412 FIT/72 development，另外2 FIT/3 development没有相机或LiDAR仍缺失；该训练仍未执行。输入变化应与目标/参数化独立解释，不与R12或下一轮表面设计同时混改。

## 单体、组合与新日志结论不能互换

单体指标先在Actor内按真实束/点数加权，再日志内Actor与独立日志等权。原生点集、显式表面、首交点各自有不同读出对象；当前one-way测量距离不是完整表面Chamfer。观测首交点点集的P/R/F/Chamfer也不代表整个不可见Actor。无返回必须进入miss，缺少owned测量时指标未定义；距离的配对比较仅用双方都定义的对象/日志。

场景评价使用全原始束，背景与所有Actor按同一字面首交点排序，统计在日志内按真实束加权。它与Actor等权单体表不是相同统计单位。PCA→TSDF背景改善了同一R5表面的组合free/early，但也增加背景缺失；这是背景变化，不能归给Actor网络。没有反事实真值的轨迹编辑只证明规范形状复用与查询能力。

外部20个AV2日志已完成输入、轨迹、背景和28-view/672×672冻结前缀准备，模型质量尚未读取。它们是项目未用于选择的新域确认，不是nuScenes IID日志，也不自动保证公开基础模型预训练语料级未见。7相机embedding桥接、有效letterbox区域与逐返回原点/轨迹均须披露；外部输入像素/视图预算与主nuScenes不同，不能宣称同推理预算。旧单个AV2开发日志已用于实现选择，不能充作这20个新日志的确认。

R11结果已完成并归档；R10/R12/R14收口、条件触发的表面参数化研究、同容量pointwise、最终物理目标/背景及新日志结果仍待推进，不预先宣布主假设成立或结束V7.3。资源表应报每项实际wall time、更新数、峰值allocated/RSS及并行背景，不能将并行作业各自wall简单相加为独占GPU小时。

## 证据入口

核心run根目录为`/root/autodl-tmp/runs/worldsim_v73`。R5最终权重见GLOBAL-ACTOR下`20260908T012500Z__population-joint-r5-epoch21-resume-r1`；R10/R11分别为`20260907T233000Z__population-joint-full-track-s7304-r10`和`20260907T233000Z__population-native-only-full-track-s7304-r11`。其余精确ID及结果见以下已有报告和三本台账：

- `WORLDSIM_V7_3_JOINT_R5_RESULTS.md`、`WORLDSIM_V7_3_POPULATION_RESULTS.md`、`WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md`。
- `WORLDSIM_V7_3_CAPA_BASELINE.md`、`WORLDSIM_V7_3_ADAPOINTR_BASELINE.md`、`WORLDSIM_V7_3_ACTOR_TSDF.md`、`WORLDSIM_V7_3_EMPTY_INPUTS.md`。
- `WORLDSIM_V7_3_OBSERVED_METRICS.md`、`WORLDSIM_V7_3_SCENE_COMPOSITION.md`、`WORLDSIM_V7_3_VDB_BACKGROUND.md`、`WORLDSIM_V7_3_CONFIRMATION_DATA.md`。
