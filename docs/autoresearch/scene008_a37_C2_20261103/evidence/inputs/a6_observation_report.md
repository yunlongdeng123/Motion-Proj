# a6 合法观测与覆盖审计

![Architecture components](observation_architecture.svg)

从原始 Waymo TFRecord 的 199 帧、5 LiDAR × 2 returns 和逐帧标注框获得 54,715 条真实 a6 测量行；重新解码后与原 canonical 数据的最大差为 2.98e-8 m。所有记录 CRC32C 通过，源文件大小和修改时间不变。canonical 坐标为 x 向前/长度、y 向左/宽度、z 向上/高度。

固定完整框约为 4.33404 × 2.07754 × 1.44 m。Omni 输入仅交换 y/z 并除以完整框最大边的一半 2.16702 m；没有根据部分点云的 bounds 居中或拉伸。

原生 B 的 raw89–137 共 49 帧、480×320，canonical→world→model 与 B 实际 camera 配准最大绝对误差 5.46e-13。`a6_views/*.npz` 保留原始 RGB、完整 identity labels、a6 可见轮廓、1px 不确定边界、稀疏深度可见光度证据以及 unknown 区。actor label4 是此前逐帧 reviewed identity silhouette，label5 是 a37，并非 bbox；与原 actor3 body mask 逐像素一致。它是经 review 的分割结果，不是 Waymo 官方像素 GT。

光度 mask 使用 erode1 的 a6 轮廓、dilate1 的 a37 排除区、完整真实点云 nearest-depth 和实际 a6 测量点 1px 支持，深度容差 0.20 m。49 帧合计 31,610 个可靠光度像素，逐帧 297–890。`a6_colour_evidence.npz` 含 15,021 行 native49 canonical 测量，8,930 行通过此可见性过滤；所有颜色均来自相应原图，非渲染或生成结果。

全 199 帧包另外保留 image-pose + Brown-Conrady 原始1920×1280投影的稀疏颜色。`rgb_valid` 仅包含 native49 有 reviewed 轮廓和深度支持的 9,705 点；49以外的几何颜色提示单独标记，不能当密集 GT。两者采样与相机约定不同，初轮 B 光度与颜色拟合应使用 native49 views / colour_evidence。

原始 a37 三维 cuboid 会与 53,802 条 a6 点射线相交，但这只是过宽的可能遮挡 envelope；它覆盖大量实际可见的 a6。该 flag 不能推翻 reviewed a6 正例与真实深度支持，不得把 cuboid 当真实 silhouette 或遮挡 GT。

![Measured coverage](a6_measured_coverage.png)

54,715 行只对应 5,976 个 5cm occupied voxel，其中 4,343 有至少两帧支持。测得 z 范围为 -0.21045 至 0.71983 m，而完整框底部为 -0.72 m：约 0.51 m 的下部/车轮没有实测几何。点云占据范围不是完整车表面覆盖率。缺点区域始终 unknown；后续 learned prior 补全和保守主色须标记为 inferred。

默认先验 conditioning 来自 raw106 原图的 80×63 crop，仅 890 个可见证据像素保留 alpha。unknown 和 a37 像素 alpha=0、RGB=0。透明表示未知，不表示测得背景或车身 GT。

`prepare_actor_surface.py` 仅使用实际生成的 learned Omni mesh候选，与 5,976 个固定真实 voxel anchors 合并；不使用手工替代 mesh。helper 以 seed0 均匀采样约14k prior surface 点，去掉距真实 anchor<0.04m 的 prior 点；已知颜色来自干净多帧测量，其余采用中性车体主色并标记 inferred。两个候选的面积、范围、残差和底部覆盖均已计算；相应asset的receipt和CPU合法表面三视图保留完整诊断。

全部准备和审计仅 CPU；GPU 使用及训练更新均为零。

Point-conditioned seed0 的原始主连通体宽度4.325 m，约为合法宽度的2.08倍。严格全盒内筛选后保留6,694个实际learned表面采样，与5,976真实voxel anchors组成12,670-anchor诊断资产；6,035个盒外sample被拒绝、1,271个距real<0.04m被去重，无封口或形状修补。原learned三角面到真实voxel anchors的精确残差 mean=0.04074 m、p90=0.09111 m。它尚未获质量接纳，允许root进行0/32step工程渲染检查，不能仅凭inside比例称作完整车体。

唯一bbox-controlled seed0候选缩成1.702×1.156×1.520 m，无法覆盖4.334 m长的真实观测。即使到它的整个mesh AABB，真实anchors的距离下界p90也为1.188 m；小normal offset不能补救。其19,597-anchor文件仅保留为诊断，formal_eligible=false，不安排该欠尺寸候选的优化smoke。

两份诊断asset都通过ZIP CRC、finite值、单位normal、provenance互斥、observed索引合法和全部prior samples位于合法盒内的验收。本分支没有运行GPU、训练、服务停止或关机；后续point候选smoke与最终科研结论由root汇总。
