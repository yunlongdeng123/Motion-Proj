# V7.3：同全轨迹监督的原生几何控制

截至2026-09-08 10:08 UTC，R11原生强控制已完成30轮及最终评价，PID38460退出；R10联合模型仍运行到第19轮。R11训练集硬hit提高，开发5日志硬hit均下降，尚未解决原生适配的泛化问题。R12联合模型的有限宽束free已登记、尚未启动；R14同目标原生控制已启动，PID68108，第1轮。完整V7.3未完成。

## 对照定义

共同数据是完整489 Actor的population，旧输入路径中371 FIT参与训练、67 development可预测，51零build LiDAR对象保留缺失。两项都从M1r3初始化，读取全24视图/744份冻结前缀；仅FIT使用既定full_track几何标签，模型输入仍只有build图像/LiDAR/标定/已知轨迹，development没有梯度。

共同目标为target→surface覆盖＋0.5 hard range free＋0.05 box envelope＋1.0 build native像素深度，event关闭；seed7304、AdamW lr1e-5、目标30轮。native depth使用当前相机像素的真实build LiDAR，独立于预测框内候选，保留F06修复。相机、轨迹、build尺度只读。

| 方法 | 实际run与进程 | 可训练通路 | 物理表面 |
|---|---|---|---|
| R10 joint | `20260907T233000Z__population-joint-full-track-s7304-r10`；code26a7e509，PID53472，running | DPT32654562＋query1670517参数 | 局部空间查询生成的位置/法向/片形状 |
| R11 native_only | `20260907T233000Z__population-native-only-full-track-s7304-r11`；code9541ac7d，done | DPT32654562参数，query冻结 | native＋LiDAR中心的PCA曲面片 |

以上均位于`WS-V73-M2-GLOBAL-ACTOR-01`。日志分别为`/root/autodl-tmp/controller_logs/v73_population_joint_r10.log`、`v73_population_native_r11.log`。当前observed allocated峰值分别10.213784/2.346402GiB；两者运行不同阶段且并行，不能直接用峰值相加代替总显存测量。

R11原生深度在正确相机曝光和Actor轨迹下回投规范坐标，融合原build LiDAR和512个native FPS支持，使用min(build,1024)+512中心、0.06m PCA片。PCA邻域/方向每步重算但本步停止梯度；所选native中心的真实位置梯度回传DPT。无相机或无有效几何/native数据梯度时，保留相应LiDAR读出，呈现不虚报为optimizer更新。

query模块在R11仅提供固定grid/数量定义，无逐点MLP或局部交互；DPT内部多层多尺度解码仍完整。训练每步重新解码，绝不跨优化步缓存DPT输出。只有一次固定权重的初始化/最终no-grad评价可按scene/view共享深度；冻结聚合前缀继续共享。不减少视图、分辨率或原始观测。

## R11真实训练与资源

30轮11130次Actor呈现、10550次optimizer更新、580次无梯度跳步，涉及22个Actor；其中420次来自14个无相机Actor，160次来自有相机但无框内native支持、退回固定LiDAR的情况。全部跳步记录均无native build对应，DPT/query梯度范数均为0。旧日志没有显式`skip_reason`字段，通用摘要中的该字段全0不能解释为没有跳步。

完整DPT的32654562参数参与训练，query可训练参数为0；native_project相对M1r3最大变化0.005223576。wall30855.513131s（8.571h）、GPU allocated峰值2.346402GiB、RSS峰值34.607677GiB；共享GPU下的wall不是独占吞吐，无恢复、无OOM。所有489对象最终均保留记录，51零LiDAR对象保持缺失。

| 过程量 | 第1轮 | 第30轮 |
|---|---:|---:|
| 实际更新 / 371呈现 | 351 | 352 |
| 有相机 / native有监督呈现 | 357 / 346 | 357 / 346 |
| native测量加权Huber（m） | 0.859150 | 0.361852 |
| native有监督Actor平均Huber（m） | 3.411858 | 1.068846 |
| 采样target→surface平均距离（m） | 0.301641 | 0.229765 |
| 采样hard free平均侵入（m） | 0.121494 | 0.131002 |
| 有相机但native支持为空 | 16 | 5 |
| DPT梯度范数中位数，裁剪前 | 300.158264 | 149.705505 |

每轮native测量173972点；query梯度始终为0。采样射线和目标随训练变化，表中过程均值不是同一组射线的配对泛化评价。过程图直接读取完整训练日志，按native_only模式标注，不套用R5的epoch21恢复叙事。

## R11物理评价

指标先按Actor/日志归并，日志等权。surface distance为单向target→surface距离，不能称为对称Chamfer；空表面保留miss与recall分母，条件距离缺失。下表为75个开发Actor、5个独立日志；初始和最终均有8个空表面。

| 方法 | hit | early | miss | free（m） | 单向distance（m） | recall@.2m |
|---|---:|---:|---:|---:|---:|---:|
| R11初始化 | .227771 | .091551 | .573543 | .150304 | .162474 | .781294 |
| R11最终 | .176111 | .076432 | .600949 | .164965 | .168504 | .751499 |
| LiDAR PCA | .212448 | .043527 | .734104 | .017708 | .304790 | .654182 |
| R7 LiDAR/full_track/hard free | .317549 | .118390 | .498974 | .071575 | .209524 | .745468 |
| R9 LiDAR/full_track/beam+event | .324688 | .060060 | .545686 | .038543 | .227363 | .724665 |
| R5 joint/短窗/hard free | .222773 | .209807 | .377094 | .351717 | .145289 | .797126 |

同run最终减初始化的95%区间由5日志配对bootstrap获得，10000次、seed7304；正负方向按列解释，不以大量射线冒充独立样本量。

| 指标 | 配对差值 | 95%区间 | 改善日志数 |
|---|---:|---:|---:|
| hit，百分点 | −5.166 | [−10.920, −1.125] | 0/5 |
| early，百分点 | −1.512 | [−2.872, −0.660] | 5/5 |
| miss，百分点 | +2.741 | [−2.697, +8.178] | 2/5 |
| free，m | +.014661 | [−.006742, +.031388] | 1/5 |
| 单向distance，m | +.006030 | [−.006994, +.028313] | 4/5 |
| recall，百分点 | −2.979 | [−8.339, +1.135] | 3/5 |

旧native_fusion对额外5个零LiDAR开发对象产生表面，而R11保持这些对象缺失，故其条件距离均值.175425m与R11初始化.162474m不是同分母。对共同ready67对象，初始化和旧融合的所有指标一致；距离比较必须取共同可评价对象，不能直接混减两个无条件表格均值。

R11−R7的hit−14.144pp、95%[−18.582,−8.623]pp，miss+10.197pp、[+4.691,+16.870]pp；没有超过同full_track LiDAR强控制。R11−R9的free+.126422m、[+.046265,+.199240]m，但R9含beam/event目标，不能视为纯架构差异。相对R5，R11的early/free改善而miss增加22.386pp，[+13.711,+35.555]pp；R5标签较短，此处不能替代与R10的匹配比较。

FIT完整414对象/20日志：hit.230922→.253677，配对+2.276pp、[+.925,+3.701]pp，16/20日志改善；early.121261→.084194，−3.707pp、[−6.431,−1.301]pp，16/20改善；free.141084→.146142m，distance.127989→.121995m、recall.808731→.816866。FIT读数在full_track训练标签范围内，不能当泛化证据。

moving仅9对象/2日志、1个空表面：最终hit.092571、early.032221、miss.678701、free.037033m、distance.120863m、recall.884831；初始化hit.090155、early.099170、free.208179m、distance.080574m、recall.952421。样本有限，不能据此宣称动态形状恢复成功。

证据：`docs/autoresearch/worldsim_v73/m2/global/population_native_r11_summary.json`、`population_native_r11_analysis.json`和`population_native_r11_training.json`。原始run保留checkpoint、train.jsonl和全部owner_surface.pt。

## 比较与尚未完成的结论

R10−R5主要回答FIT标签由短窗扩大为full_track的问题；R10−R7是同full_track下视觉联合路径与LiDAR路径比较；R10−R11比较query生成通路与认真训练的原生头/融合控制。最后一对同时改变位置生成及法向/片形状参数化，不能把差异自动归因为空间attention；完整population同容量pointwise控制仍未运行。

R5本身在覆盖改善时出现early/free退化，已在`WORLDSIM_V7_3_JOINT_R5_RESULTS.md`记录；这既不能替代当前R10/R11比较，也不能推导所有视觉几何适配失败。R5原第22轮意外退出只保留第21轮状态，107个未保存更新留原日志；恢复完成有效11130更新，旧checkpoint缺完整RNG的执行差异已披露，不声称逐比特连续。

R12登记`20260908T050000Z__population-joint-full-track-beam-range-s7304-r12`，从与R10相同的M1r3/seed初始化，仅改变free为`beam_tube_range`、width.03m/resolution32；full_track/native1/free.5/event0/30轮不变。R14登记`20260908T100000Z__population-native-full-track-beam-range-s7304-r14`，同样只将R11的free改为该有限宽束目标，复用同输入的R11初始化及原PCA，重新训练而非恢复。R14已于登记bf04ef32之后真实启动，与R10并行；R12仍待完整query显存可用，无自动GPU队列。

R11新增V73-F09：本配置训练拟合改善却在开发硬hit上退化，观测污染、共享参数漂移、支持/PCA读出及目标语义的归因尚未完成。卡点先核对[LoRA3D ICLR2025官方项目](https://520xyxyzq.github.io/lora3d/)与[CAPA官方项目](https://research.nvidia.com/labs/dvl/projects/capa/)：两者的场景适配/稀疏测量校准是迁移参考，不证明本项目当前失败原因。优先完成R10/R11及R12/R14同目标比较，之后才依据证据独立改变DPT适配范围或build场景校准；保留CAPA R2既有负结果，不直接重跑、不用confidence逃避free、不否定全部视觉几何适配。

visual-only的R13已完成一轮41次实际更新、DPT/query真实反向，专项结果已归档EMPTY_INPUTS报告；这仅确认新输入训练能执行，不是完整训练或质量成功。R10/R11/R12/R14继续原cohort。输入条件扩大、物理目标变化与空间机制分别解释，见`WORLDSIM_V7_3_COMPARISON_PROTOCOLS.md`。新20日志质量确认仍未读取。

原生解码路径依据：[VGGT官方训练说明](https://github.com/facebookresearch/vggt/blob/main/training/README.md)和[DPT实现](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/dpt_head.py)。微调DPT是强控制的既定工作，本身不作为新贡献。
