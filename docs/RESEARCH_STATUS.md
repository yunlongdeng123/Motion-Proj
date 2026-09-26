# 当前研究状态

更新：2026-09-26（Asia/Singapore）。当前分支 **`v77`**，远端工作区 **`/root/autodl-tmp/motion_proj_v77`**。阶段：V7.7 / VGGT-Ω Structured Edit P0准备完成，实验尚未运行。唯一协议见[P0问题、范围与architecture](v77/P0_PROTOCOL.md)，实验入口见[EXPERIMENTS](EXPERIMENTS.md)。

## 已关闭的前序路线

V7.6在commit `6350b258`关闭，failure为[V76-F03](research_failures/entries/V76-F03.md)：**当前HUGSIM和VAD-GS两批资产的对象级质量，尚不足以直接支撑我们想要的高保真反事实编辑。** 不再继续“逐场景Gaussian资产 → editor”及对象级修补。已有模型、输入、日志、正反结果保留为baseline/failure evidence，见[V7.6收口](v76/CLOSEOUT.md)。

## V7.7范围与准备状态

后续重建基座统一使用 **VGGT系列**。首阶段为 **冻结VGGT-Ω + GT实例选择 + 解析MOVE / DELETE / clone-INSERT**，零训练；暂不引入自然语言、闭环、扩散视频精修或新学习模块。先检验对象分离、米制编辑遵循、背景保持和多视角完整性，再由实际failure决定后续方法。

已准备权重 `/root/autodl-tmp/models/worldsim_v77/vggt_omega_1b_512.pt`，这是指向远端已有完整512权重的符号链接，文件大小4,576,706,117字节。以官方实现commit `b2c61f6631d9f344a2d914bfba5d9529d6fc1d35`完成CPU `weights_only`严格state_dict加载：1411项、missing/unexpected均为0。实际来源与检查见[manifest](v77/checkpoint_manifest.json)。这是参数兼容性检查，不是GPU前向或画质验证。

用户提供的Drive地址已登记并确认文件名和大小。重复下载因速度慢，在确认已有完整可用权重后停止；已下载的75,497,472字节与已有文件前缀一致，partial保留。不宣称已完整下载该Drive镜像或已证明不同来源全文件一致。

## 下一项工作与资源边界

下一步按P0协议固定开发集scene/actor/相机及命令，首先验证同步多视角重建、数据集米制坐标对齐和单对象解析编辑，再扩展规划的20–30actor。GT标定、box/ID或mask均披露为额外输入；未见表面及disocclusion缺口如实保留。此时尚无VGGT-Ω优于HUGSIM/VAD-GS的结论。

当前远端 `nvidia-smi` 返回Permission denied，未启动GPU推理或训练；P0前仍需检查可用GPU及推理依赖。现有CPU环境torch2.4.1能完成权重加载，官方requirements要求torch>=2.6，尚未将其宣称为完整合格的P0运行环境。

旧路线没有存活训练/渲染/评价或启动控制器，V7.6关闭标记已保存；本次未创建自动续跑、未执行关机。后续工作使用本v77工作区，不在另有用户改动的V7.5共享工作区切分支。

`failure_ledger_refs: [V76-F03]`；本次V7.7准备的 `failure_ledger_delta: none`。
