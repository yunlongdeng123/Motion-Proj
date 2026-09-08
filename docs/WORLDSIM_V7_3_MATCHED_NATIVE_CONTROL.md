# V7.3 同全轨迹监督的原生几何控制

## full_track主联合对照已启动（2026-09-08）

r10 `20260907T233000Z__population-joint-full-track-s7304-r10` 已以code26a7e509/PID53472启动并进入实际反向，日志`/root/autodl-tmp/controller_logs/v73_population_joint_r10.log`，当前GPU allocated峰值10.19725GiB。从原M1r3/seed7304初始化，无resume-from；仅相对r5扩fit标签为full_track，free/native/event/epoch保持。r11原PID38460继续，不重复启动。

r5恢复已完成30轮/489对象最终评价并退出。其覆盖改善伴随early/free退化，详见`WORLDSIM_V7_3_JOINT_R5_RESULTS.md`；不是停止路线B或停止整个研究的理由。后续r12有限宽束free仅登记，r10与r11仍用于同标签强控制，不用旧短窗r5冒充该比较。

## 实际调度与恢复边界（2026-09-08）

native-only r11 `20260907T233000Z__population-native-only-full-track-s7304-r11` 已启动，code9541ac7d、PID38460，日志 `/root/autodl-tmp/controller_logs/v73_population_native_r11.log`；初始评价覆盖371fit/67dev ready与51空输入，全部744冻结视图。模型训练与最终效果尚未完成。

主joint r5在epoch22意外退出，无异常栈，cgroup OOM0，原因尚不明。第21轮7791次完整更新及模型/优化器保存完好；107个未保存更新留在原日志但不进入恢复状态。恢复run `20260908T012500Z__population-joint-r5-epoch21-resume-r1` 已启动并实际进入epoch22反向，code914d582d、PID39009，峰值allocated10.19840GiB。只读冻结前缀/图像改为mmap共享文件页，不改变输入和监督。旧checkpoint缺RNG，明确记录CUDA抽样seed7304重启，并重放Python shuffle顺序，不声称逐比特连续；以后checkpoint保留RNG和fit顺序。

该恢复依然是r5的短窗口监督，不能代替r10完整轨迹joint对照。r10尚未运行，应等待恢复主模型结束和实际资源释放。F08记录意外中断与恢复，F01不能仅凭不明退出认定资源不足。整个研究未完成，仍不关机。


状态：实现与登记，尚未运行。已有M1r3 DPT深度适配与其固定native+LiDAR fusion用build标签，而r7/r8及AdaPoinTr使用更充分的fit全轨迹标签；不能把这种标签差异解释为架构收益。现在补齐原生保守控制，与后续full_track joint在同样标签上比较。

入口为 `train_worldsim_v73_global_actors.py --mode native_only`。冻结原聚合前缀，仍使用完整24视图的原生多层特征，直接微调全部DPT几何头参数；不训练query模块，后者只提供同一固定曲面片grid和数量定义。每步重新解码原生深度，在正确相机曝光/Actor轨迹下回投影，生成规范点；融合原build LiDAR和512个native FPS支持，使用同min(build,1024)+512曲面预算及0.06m PCA片。没有逐点残差MLP、空间查询交互、opacity或额外可学习曲面半径。

PCA邻域和方向每步重新估计但本步停止梯度，surface/free位置梯度通过选中的native三维中心回传DPT深度；与已登记AdaPoinTr点集转换一样，必须披露这种条件位置梯度近似。直接build像素深度数据项独立于native候选筛选，保留F06修复。与主joint比较时，差别包括空间查询及表面生成参数化，不能直接把任何收益都单独归因于attention。

监督为同一fit全轨迹target→surface覆盖、原始首回波前的hard range free（权重0.5）、box envelope0.05和直接build native depth1.0，event关闭。seed7304、AdamW1e-5、30epoch、11130个fit Actor呈现；共享fit参数，dev不反传。原输入可用的371fit/67dev参与对应路径，51个无输入对象按当前主模型边界保留空预测，不使用后续测量充当输入。无Actor相机位姿或当前所有native支持/像素数据梯度均缺失时，保留LiDAR融合读出并记录无优化的呈现；不能虚报为一次DPT更新。日志与最终summary分别报告Actor呈现数、真实optimizer更新数、无梯度呈现数和trainable参数数。

执行优化仅去除query不用的多尺度副输出拼接；DPT内部原有多层/多尺度解码仍完整。训练不缓存DPT结果；仅在每次权重固定的no-grad初始化/最终评价中按scene/view缓存CPU深度，同一窗口多个Actor复用，评价结束即丢弃。冻结聚合前缀继续按已有scene/view共享。不减少视图数、图像分辨率或原始观测。

登记 `WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-native-only-full-track-s7304-r11`，从M1r3原生头初始化，固定PCA评价复用r5，初始表面按本方法实际计算。对应主joint全轨迹控制登记 `20260907T233000Z__population-joint-full-track-s7304-r10`：从与r5相同的M1r3/seed初始化，仅将fit标签由短窗改成full_track，free仍为range、event关闭，其他超参相同。二者尚未启动；与当前r5共享大量前缀数据但进程显存需按实际容量安排，不能在已接近满载时盲目叠加。

r10对r5回答标签范围问题，r10对r7提供同全轨迹监督的视觉联合路径与LiDAR控制，r10对r11提供原生几何保守适配与查询生成比较。event r9仍为独立LiDAR机制对照，不替代路线B主模型。最终还需选定架构/目标后的新日志与场景验证，当前不宣称假设成立。

原生路径来源：[VGGT官方训练说明](https://github.com/facebookresearch/vggt/blob/main/training/README.md)和[原生DPT实现](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/dpt_head.py)。这属于认真训练强控制的既定工作，不将“微调DPT”称为新贡献。
