# V7.3 独立确认数据准备

## 外部20日志冻结前缀完成（2026-09-08）

`WS-V73-M4-AV2-PREFIX-01/20260908T043000Z__external20-28view-prefix-r1` code36b22287完整完成，407.9406s/GPU7.64359GiB/RSS10.63879GiB。原20身份/560视图，每日志28×672²共同进入冻结聚合器；只分块独立图像patch编码，每块7视图。4个官方DPT输入层缓存42371687680字节（约39.46GiB）。

用原预训练头与6722839条build对应估计固定IRLS米制尺度，范围11.0605–55.1629，再安装M1r3头。没有新日志梯度、heldout模型质量或方法选择，尺度差异未触发外部调参。原PID52753退出，摘要`autoresearch/worldsim_v73/coverage/av2_external20_prefix_summary.json`保留每日志尺度/计数/资源。

输入、逐束背景与前缀已齐备，最终质量确认仍等待旧开发方法选择；这是AV2新日志的跨数据集确认，不能写成nuScenes同分布新日志。当前r5的旧开发物理退化已记录，未据此读取外部分数试选候选。

更新于2026-09-08。20条预登记AV2日志的输入已全部导出，尚未运行外部网络推理、heldout重建质量评价或共享参数更新。方法选择继续使用旧开发数据；外部确认不是已经取得的结果。

## 独立性与身份选择

nuScenes本地35个可用场景属于27日志，当前31窗口属于25日志；额外scene-0139、scene-0379分别有V6.4 fit和V5诊断/开发历史。更完整的旧角色表覆盖trainval全部850场景/68日志（train54、legacy2、exposure_unknown2、dev4、route_select3、source_test3），因此没有可保守视为全新身份的trainval日志。公开挂载十个blob约294GiB能补载荷，不能创造日志独立性，本轮未整体解压。证据`docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json`及既有V7.2角色表。

改用[官方AV2 Sensor数据](https://argoverse.org/av2.html)中未使用的公开日志。官方S3 train目录700身份，排除仓库此前出现的1条，699候选按字典序取前20条；名单在获取这些日志逐文件元数据与传感器值之前确定。固定名单和760个载荷项在`configs/worldsim_v73/av2_external_confirmation_r1.json`，来源`s3://argoverse/datasets/av2/sensor/train/`，本地`/root/autodl-tmp/data/av2/sensor/train/`。

官方split=train不等于本研究训练角色；这20条在V7.3中为external_confirmation，不加入共享fit和超参数选择。新日志与传感器域同时改变，必须单列跨数据集结果，不能称nuScenes同分布确认。缺失或低覆盖对象保留，不按预测好坏换身份。

## 固定输入和时空语义

每日志原始LiDAR序号[5,15,20,30]为build，[10,25]为heldout；七路ring相机各取四个build参考时刻最近的实际曝光，共28图。保留所有七路独立观测，不把−23.10至+18.11ms的相机/扫描时间差硬置零。760文件343260344字节（327.36MiB），977.74s完成下载；无需下载整个数据集或重复解压大归档。

按[AV2官方Sweep](https://github.com/argoverse/av2-api/blob/main/src/av2/structures/sweep.py)处理：发布端点已运动补偿到扫描参考ego系，世界端点只乘一次参考ego变换；每点时间为参考时间+offset_ns，束原点取该点时刻的ego和物理LiDAR外参。Actor归属/规范坐标使用同一逐点时刻只读轨迹，相机投影使用实际曝光时刻。未知pose不夹边冒充已知。0–31→up、32–63→down来自旧AV2开发窗口ring角度一致性的推断，未按新域质量重选。

完整画幅按最长边672缩放至672²画布，内参与有效矩形同步变换；padding不作为query/native表面支持。七相机ID以已知标定方位插值原六项训练嵌入，保留独立视图投影、方向和时间；嵌入平滑性仍是假设。旧AV2已完成完整28视图前缀、固定native fusion和LiDAR-only推理；它们不使用视觉query相机ID，最终joint七相机交互仍待旧窗口实际运行。细节和边界见`WORLDSIM_V7_3_AV2_GEOMETRY.md`。

## 20日志输入导出已完成

任务`WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1`，codef70d099c，2061.27s；CPU父RSS3.66828GiB，子最大1.26955GiB。原PID41941及子任务均退出。索引归档`docs/autoresearch/worldsim_v73/coverage/av2_external20_index.json`，完整case、原始28图输入、逐点build/heldout束和日志保存于run。

| 输入覆盖记录 | 数量 |
| --- | ---: |
| 日志 / 完整相机输入 | 20 / 560 |
| build时刻有已知姿态的刚体车辆 | 936 |
| ready / unavailable_input | 878 / 58 |
| 已知轨迹速度>2m/s / 速度不可用 | 221 / 108 |
| 无可用Actor相机时刻 | 1 |
| 无自有heldout返回 | 119 |
| 原始返回总数 | 11723402 |
| build / heldout返回 | 7815152 / 3908250 |
| 传感器姿态未知返回 | 0 |
| 框归属重叠返回 | 7077 |

此cohort根据build和轨迹元数据形成，不是“全部真实可见车辆”或按target点筛选的完整表面集。58个空输入全部保留；现有主query尚未覆盖所有视觉-only空LiDAR路径。原始传感器和标注数值已经处理，不能再说“新值未读取”；目前没有外部模型质量结论。

## 场景构建与后续执行

旧AV2开发窗口已用同四build扫描构建未雕刻/雕刻背景，并在逐束已知姿态下组合21个固定Actor表面和两次187494原始heldout束。背景PCA间距0.06m、20近邻；排除所有有效时刻已知框+.1m，只按build首返回前.2m删除冲突三角面。世界背景和每个规范Actor各建一次BVH，按逐点时间逆变换射线并统一最近求交。range<1m仅报告分层，无该条件的数据删除。

已以code8a0cf6a4完成全部20日志的外部场景数据`WS-V73-M4-AV2-SCENE-DATA-01/20260908T031000Z__external20-per-return-build-background-r1`：复用上述旧域确定参数，CPU逐日志构建，包含所有原20身份/936Actor。背景build雕刻诊断用于构建，heldout只另存原始束，不做外部模型推理或heldout质量评分。运行失败保存原因且停止，不跳过或替换日志。构建耗时1519.9831s，CPU父RSS0.12975GiB、子最大0.91083GiB；父47224、外层shell47223及子任务均退出。6839807个背景点、原54718456面，按固定build-only规则删除2159469（3.9465%），余52558987面；906573条build冲突束降至23条，剩余侵入总和183.0328m，未因此调参或循环删面。936Actor的105755个已知姿态及全部80build/40heldout扫描保留。完整轨迹索引76684364字节保留在run，Git仅归档`m4/av2_external20_scene_construction.json`的构建摘要和源路径，避免重复大数组。日志`/root/autodl-tmp/controller_logs/v73_av2_external20_scene_data_r1.log`。未进行外部网络推理或heldout模型质量评分。

方法定型后才在外部集形成固定模型对比。28×672²的完整冻结前缀在旧窗口实测allocated峰值7.64359GiB；当前r5/r11并行时不挤入该GPU任务。只缓存完全冻结前缀；可训练几何头实时解码，不把旧最终特征缓存冒充内部PEFT。

F05的身份独立性已有数据对策，域迁移/实际覆盖/空输入问题仍在。20日志下载与输入完成不能代替独立确认，当前也没有证据表明必须因资源不足关机。整个V7.3继续自动研究，最终保存并push，确认无训练、评价、数据或启动控制任务后才shutdown。
