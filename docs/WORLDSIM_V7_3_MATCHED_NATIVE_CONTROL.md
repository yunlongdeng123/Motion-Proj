# V7.3 同全轨迹监督的原生几何控制

状态：实现与登记，尚未运行。已有M1r3 DPT深度适配与其固定native+LiDAR fusion用build标签，而r7/r8及AdaPoinTr使用更充分的fit全轨迹标签；不能把这种标签差异解释为架构收益。现在补齐原生保守控制，与后续full_track joint在同样标签上比较。

入口为 `train_worldsim_v73_global_actors.py --mode native_only`。冻结原聚合前缀，仍使用完整24视图的原生多层特征，直接微调全部DPT几何头参数；不训练query模块，后者只提供同一固定曲面片grid和数量定义。每步重新解码原生深度，在正确相机曝光/Actor轨迹下回投影，生成规范点；融合原build LiDAR和512个native FPS支持，使用同min(build,1024)+512曲面预算及0.06m PCA片。没有逐点残差MLP、空间查询交互、opacity或额外可学习曲面半径。

PCA邻域和方向每步重新估计但本步停止梯度，surface/free位置梯度通过选中的native三维中心回传DPT深度；与已登记AdaPoinTr点集转换一样，必须披露这种条件位置梯度近似。直接build像素深度数据项独立于native候选筛选，保留F06修复。与主joint比较时，差别包括空间查询及表面生成参数化，不能直接把任何收益都单独归因于attention。

监督为同一fit全轨迹target→surface覆盖、原始首回波前的hard range free（权重0.5）、box envelope0.05和直接build native depth1.0，event关闭。seed7304、AdamW1e-5、30epoch、11130个fit Actor呈现；共享fit参数，dev不反传。原输入可用的371fit/67dev参与对应路径，51个无输入对象按当前主模型边界保留空预测，不使用后续测量充当输入。无Actor相机位姿或当前所有native支持/像素数据梯度均缺失时，保留LiDAR融合读出并记录无优化的呈现；不能虚报为一次DPT更新。日志与最终summary分别报告Actor呈现数、真实optimizer更新数、无梯度呈现数和trainable参数数。

执行优化仅去除query不用的多尺度副输出拼接；DPT内部原有多层/多尺度解码仍完整。训练不缓存DPT结果；仅在每次权重固定的no-grad初始化/最终评价中按scene/view缓存CPU深度，同一窗口多个Actor复用，评价结束即丢弃。冻结聚合前缀继续按已有scene/view共享。不减少视图数、图像分辨率或原始观测。

登记 `WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-native-only-full-track-s7304-r11`，从M1r3原生头初始化，固定PCA评价复用r5，初始表面按本方法实际计算。对应主joint全轨迹控制登记 `20260907T233000Z__population-joint-full-track-s7304-r10`：从与r5相同的M1r3/seed初始化，仅将fit标签由短窗改成full_track，free仍为range、event关闭，其他超参相同。二者尚未启动；与当前r5共享大量前缀数据但进程显存需按实际容量安排，不能在已接近满载时盲目叠加。

r10对r5回答标签范围问题，r10对r7提供同全轨迹监督的视觉联合路径与LiDAR控制，r10对r11提供原生几何保守适配与查询生成比较。event r9仍为独立LiDAR机制对照，不替代路线B主模型。最终还需选定架构/目标后的新日志与场景验证，当前不宣称假设成立。

原生路径来源：[VGGT官方训练说明](https://github.com/facebookresearch/vggt/blob/main/training/README.md)和[原生DPT实现](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/dpt_head.py)。这属于认真训练强控制的既定工作，不将“微调DPT”称为新贡献。
