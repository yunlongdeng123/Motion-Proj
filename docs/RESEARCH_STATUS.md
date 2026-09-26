# 当前研究状态

更新：2026-09-26（Asia/Singapore）。分支 **`v77`**，远端工作区 **`/root/autodl-tmp/motion_proj_v77`**。阶段：**P0已完成，连续三路视频复核材料已补齐，等待用户人眼评价；V77继续，尚未取得高保真对象资产资格。** 视频合同与architecture见 [VIDEO_REVIEW](v77/VIDEO_REVIEW.md)，前轮结果见 [P0_RESULTS](v77/P0_RESULTS.md)，实验按run查 [EXPERIMENTS](EXPERIMENTS.md)。

## 已完成与当前判断

三个现有nuScenes开发导出固定24对象：六相机单时刻Ω、GT标定/背景尺度控制，以及0/20/40帧按GT轨迹的规范坐标并集。九次冻结前向、54张RGB、零训练。MOVE/DELETE/clone-INSERT解析点集操作通过，非目标点不变。观测LiDAR20cm召回宏平均分别6.93%、27.71%、55.02%，但并集指标必不下降；缺面、重影、提取不完整仍在，不能声明高保真编辑通过。

新增 [V77-F01](research_failures/entries/V77-F01.md)。已验证算子数值实现，尚未分清局部米制深度、可见实例绑定与未见表面缺失。GT框不保证无遮挡，原始camera timestamp缺失，额外GT/LiDAR与开发集曝光均披露；人工verdict保持null。

## 下一项工作与停止边界

用户要求直接看重建时序：已补齐三个场景完整连续序列的原始/factual/固定2m平移九段视频，六相机可同步切换。当前优先接收用户对具体相机、时间点的评价；页面人工字段均空，未代填。视频是逐时刻独立重建回放，并非持久4D资产或跨时刻融合结果，输入角色和播放方式详见视频报告。

继续VGGT系列。下一项先建立**可见对象的局部几何/实例绑定强控制**：明确可被相机观察到的目标表面，以投影和保留LiDAR诊断局部深度、框内污染和框外漏选；先保留原24对象完整分母，不能换成功样例后冒称确认实验。若只能用对象真值修复，需要单列上限控制。

可见表面支持与绑定可靠后，再把剩余未见面/disocclusion交给补全设计。当前不加学习head、不扫阈值、不做自然语言或扩散精修。clone仍是源点复制；持久化新实例ID与图像层存在性尚未实施，不称产品级editor完成。此后另取未曝光样例才能做确认评价。

## 运行与恢复

GPU已可用：RTX3090 24GiB，冻结六视图前向峰值约5.26GiB。推理环境`/root/autodl-tmp/envs/worldsim-v77/bin/python`（torch2.12.1+cu130、numpy1.26.4）；共享包与未用scipy依赖冲突见报告。三个几何测试通过。核心实现已以`90528aa8`推送；预测、点资产、日志、离线审核页均保留在`/root/autodl-tmp/runs/worldsim_v77/`，详见 [execution_manifest](autoresearch/worldsim_v77/p0_20260926/execution_manifest.json)。

权重`/root/autodl-tmp/models/worldsim_v77/vggt_omega_1b_512.pt`复用已有完整512文件；实际来源和用户Drive重复下载边界见 [checkpoint_manifest](v77/checkpoint_manifest.json)，未宣称完整下载Drive新副本。有限试验均已结束，无续跑控制器，未执行关机。

## 前序路线约束

V7.6按用户要求在`6350b258`关闭：[V76-F03](research_failures/entries/V76-F03.md)——当前HUGSIM和VAD-GS两批资产的对象级质量，尚不足以直接支撑想要的高保真反事实编辑。保留baseline/failure证据，不继续“逐场景Gaussian资产→editor”及对象级修补，后续基座统一VGGT系列。未修改另有用户工作的V7.5共享目录。

`failure_ledger_refs: [V76-F03, V77-F01]`；本轮视频交付`failure_ledger_delta: none`。已有V77-F01保留，未从视频选点数新增科学否定。
