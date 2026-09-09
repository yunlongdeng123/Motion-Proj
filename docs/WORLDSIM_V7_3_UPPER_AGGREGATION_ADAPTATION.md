# V7.3：上层跨视图适配的接口与资源边界

2026-09-09 00:40 UTC。已实现独立的上层LoRA与冻结前缀→DPT接口，并完成一次CPU接口检查；尚未接入正式训练，完整24视图传感器反传与GPU峰值待实测。三角比较已收口，Q-v2共享网格联合r1正在运行，同网格LiDAR r2已完成。两者均未使用本接口；不得把既有结果称为上层适配结果。

![上层适配组件接口已实现，CPU检查通过，传感器训练待进行；层号从0开始](autoresearch/worldsim_v73/upper_tail/V73_UPPER_TAIL_INTERFACE.png)

## 可以复用的是适配位置之前的状态

本机VGGT和[官方aggregator](https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/models/aggregator.py)均按frame/global顺序运行，DPT输入为第4/11/17/23组的frame与global结果拼接。当前接口只适配18–23这6组，以第17组拼接的后1024通道恢复global状态，复用第4/11/17组DPT输入，重新计算第18–23组及最终第23组拼接。已用小型随机模型对照官方交替处理函数，尚未做真实24视图预训练输出的数值恢复；不将旧第23组最终缓存冒充已适配输出。若以后从第12组开始，17/23都需重算，缓存边界相应前移。

本项目原`NativeGeometryPyramid`按Actor选择已完成全窗口聚合的视图，这对冻结结果是合法读取。新`FrozenUpperPrefix`按窗口原顺序保留全部视图的4/11/17层CPU值；`UpperAdaptedGeometryPyramid`先重算完整窗口的上层，再选择Actor的DPT输入。特殊token和位置编码保持官方定义，未减少24视图。图示边界与默认rank8已经实现，正式训练配置仍待登记。

前次准备阶段CPU读取原M1r3的`scene-0450_03.pt`：4层形状均为`[1,1,1301,2048]`、dtype为FP32、patch_start为5。M1缓存写入直接保存aggregator结果，没有额外投影；不能因当时启用BF16 autocast就假定缓存也是BF16。新接口保留现有缓存值精度，只在计算时启用原BF16 autocast；同文件中的旧23层不作为适配输入使用。元数据证据见`autoresearch/worldsim_v73/m2/global/upper_tail_metadata_r1.json`。

## 梯度路径与执行组织

`motion_proj/worldsim_v73/upper_aggregation.py`采用官方12个frame/global Block，加载18–23组真实权重后冻结原参数，在qkv加入LoRA。低秩A采用Kaiming初始化、B为零、缩放alpha/rank，依据[微软LoRA参考实现](https://raw.githubusercontent.com/microsoft/LoRA/main/loralib/layers.py)；不做train/eval权重合并。其间冻结的原projection、FFN/归一化仍正常传递输入梯度。适配到DPT/Query/表面的完整路径已经提供接口，传感器端到端梯度尚未实测；正式接入时DPT和Query继续训练，保留build直接深度、统一硬物理读出与只读标定/轨迹/尺度。

本机以及官方aggregator在training模式使用`use_reentrant=False`的checkpoint；[PyTorch 2.4.1源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/utils/checkpoint.py)说明这一版本不要求缓存输入本身requires_grad。沿用非重入重算即可让冻结输入上的新参数接收梯度，无需给整个早期编码器制造梯度。首个真实训练步骤就地确认新参数/几何的实际梯度与资源，不另建反复smoke或门控。

按当前每Actor一次optimizer更新的组织，上层权重更新后必须重算整个窗口的上层结果，不能像冻结前缀一样跨步共享。若为吞吐改为同窗口多Actor共用一次前向后合并loss，会改变更新频率和梯度权重，应单独披露实际预算，不能无声替换成“同样30轮”。冻结早期权重可留CPU或不驻留GPU；训练只需当前上层模块、DPT、Query及其优化器状态。

## 资源估算不是峰值实测

当前378×672、patch14、24视图：每视图1296个图像token加5个特殊token，全局序列31224，隐藏维1024、16头。以下是算术量，不是已申请显存：

| 对象 | 理论量 |
|---|---:|
| 单份全窗口1024维状态，FP32 | 121.969 MiB |
| 4份frame/global拼接DPT输入，FP32 | 975.750 MiB |
| 同两项若采用BF16，仅作对照 | 60.984 / 487.875 MiB |
| 显式16头全局attention score，BF16 | 29.055 GiB |
| 同score若FP32 | 58.111 GiB |
| 最后6组frame/global的qkv LoRA，假设rank8 | 393216参数 |
| 上项再加入projection LoRA | 589824参数 |

权重文件元数据确认第18组frame/global的qkv为3072×1024，projection为1024×1024。本轮已在CPU加载全部12个尾部Block，实计冻结参数151182336、qkv LoRA参数393216；未加载早期编码器或做GPU计算。表中再适配projection的参数数目仍为算术量；该可选配置未运行。LoRA数字不含完整DPT/Query，也不含冻结上层权重和激活。

本机[attention实现](https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/layers/attention.py)调用SDPA。PyTorch 2.4.1会依输入选择实现，不能把调用名等同于一定用了FlashAttention；其[官方接口源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/nn/functional.py)提供选择fused后端及解释不兼容原因的入口。29/58GiB是显式score的反例预算，绝非该提案已经OOM；高效attention和checkpoint能减少中间存储，但不消除全局计算和反向开销。首个实际运行若遇不兼容，先按官方后端条件修复等价执行，不裁掉视图；只有实际仍无法继续才走资源不足出口。

## 一次CPU接口检查及其边界

命令：`/root/autodl-tmp/envs/motionproj/bin/python scripts/check_worldsim_v73_upper_tail_path.py`，seed7305。真实12个尾部Block权重只在CPU加载计数后释放；执行前向/反向的是3视图、每视图11个token、宽32、rank2的随机模型。参照来自官方`Aggregator._process_frame_attention`和`_process_global_attention`，验证交替顺序、非重入checkpoint及跨视图影响。没有读取DEV/新日志质量。

| 检查结果 | 数值 |
|---|---:|
| 对官方交替函数输出的最大差值 | 0 |
| 第一frame块LoRA B梯度范数 | 0.0004783543 |
| 最后global块LoRA B梯度范数 | 0.0002375138 |
| 一次合成SGD更新后输出最大变化 | 7.152557e-7 |
| 改变第三视图后第一视图输出最大变化 | 0.001531661 |
| 缓存输入requires_grad / 冻结原权重有梯度 | false / false |
| CPU脚本总耗时 / 进程峰值RSS | 1.779956 s / 1.508625 GiB |

B为零初始化，首步A梯度为零符合该参数化，不代表梯度路径断开。脚本首次遇到位置张量expand后非连续，官方global函数的view无法重排；按照[官方PositionGetter](https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/layers/rope.py)增加clone修复，随后重跑同一检查通过。首个错误保存在`upper_tail/interface_check_r1_attempt1.txt`，通过结果在`upper_tail/interface_check_r1.jsonl`（均位于`autoresearch/worldsim_v73/`）。这是未进入训练的接口修复，不是科学负结果，也不新增失败ID。

本次没有证明实际24视图的数值复原、DPT/Query的真实传感器loss反传、GPU可承受性或重建收益。`UpperAdaptedGeometryPyramid`已实现但未执行，正式训练入口和优化器尚未接入。首次真实步骤需要把LoRA显式加入优化器，保存适配config与adapter状态，并就地记录实际全窗口梯度和资源；不另外重复小型检查。

## 当前结论

上层有限适配的独立代码与CPU接口证据已具备，不必重算完全冻结的早期部分；还没有证据证明它能改善本任务或单卡能承受完整训练。代码保留全窗口上下文，每次调用重建适配输出，返回后模块清除对上一计算图的引用。按当前每Actor更新方式，完整上层每步都要重算；这项计算代价仍待实测。

Q-v2联合r1在00:26 UTC已完成第11轮，继续原配置；LiDAR r2已显示更多交点未转化为正确首表面，详见Q-v2报告。先收口r1−r2，再选择下一项机制研究；表面参数化、对应一致性、认证free边界和上层适配分别解释，不混入新cohort/event/新基座后只归因于LoRA。关联V73-F01/F02/F03/F06/F09，failure_ledger_delta=none；没有新的正式训练/评价run或环境/权重下载。代码以3eeca1e8为基线新增，检查与本报告同提交；20新日志质量未读，30分钟跟进ACTIVE，完成不关机。
