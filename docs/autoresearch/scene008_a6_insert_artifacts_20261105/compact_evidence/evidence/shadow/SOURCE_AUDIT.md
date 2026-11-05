# 删 a37 后道路暗斑：实际渲染来源

同相机逐层干预表明，两个可见暗斑的直接载体不同：raw105 中央暗团由黑色旧静态高斯遮住恢复道路；raw113 底边暗块主要由 road1K 基底 RGB 携带。没有发现独立的物理 shadow 层，也不能把这些渲染归因等同于对真实阴影成因的证明。

![architecture](architecture.png)

保留所有组的几何、opacity 与深度排序，只置零其他组的 RGB，得到完整 joint alpha×T 下各组颜色；各组颜色、残余 T×env、常数背景相加，和原图的最大差为 8.35×10⁻⁷。相反，仅分别渲染各组到黑底会改变透射，不能当成该像素的来源分解。

| 固定像素域 | raw105 中央暗核 | raw113 底边暗核 |
|---|---:|---:|
| ROI | [239,256,294,291] | [281,301,336,320] |
| ROI 中 luma index < .25 像素 | 293 | 44 |
| static alpha×T | .371647 | .006582 |
| road alpha×T | .627972 | .993221 |
| 环境 RGB 的 luma index | .000149 | .000170 |
| static RGB=.4、opacity 不变的 luma 增量 | +.148659 | +.002633 |
| road RGB=.4、opacity 不变的 luma 增量 | +.027590 | +.150240 |
| 关闭 fine 色场的 luma 增量 | −.007271 | +.002781 |

raw105 暗核里的 static 颜色贡献为零，但它占 37.2% 的 alpha×T，因而压暗后面的道路。关闭道路让它更黑；只把 static RGB 换成灰色则暗团消退，证明它不是单纯道路色纹理。旧 opacity gates 已修复原生 10K 的大部分黑洞，但仍留有这一静态前缀。

raw113 的 road 覆盖约 99.3%；灰化 road RGB 后暗块消失，灰化 static 基本无效。关闭 fine 仍然保留暗块，说明主要暗色已经位于 road1K 基底材质。它来自真实路面深色纹理、来源照片中烘焙的暗影，还是恢复参考/标定残留，需要再追溯具体 road source samples；本轮没有把其中任何一种当作事实。

![same camera controls](shadow_source_controls.png)

raw130 使用相同图像位置作为控制，没有 luma<.25 的暗核；没有声称它对应同一世界位置。最初按整幅道路最暗像素计算的统计包含其他地点，不能直接套到这两个视觉暗斑。

三帧中 a37 都被物理移除，packed/active 行数为零，和原先 masked deletion 的 RGB 完全相同。直接残留的 a37 宽核因此被排除；a37 训练期间烘焙进其他组的信息仍可能留下。环境是 `(1-alpha)·env` 加法背景，不是乘在道路上的照明。上述两个暗核的透射都接近零，环境贡献很小。

旧 10K 报告已经记载 raw113/130 删车后的深色道路缺口。当前 Runtime 走 `repair37 → native_binding` 和整场原生深度排序；历史 `road_runtime.Experiment.render` 的屏幕 route 不是当前调用链。`shadow_proxy/new_shadow_proxy` 是证据/损失掩码，没有被渲染成独立阴影层。

本轮只做 12.37 秒诊断前向，未训练、未改 checkpoint、未改源模型或资产。灰化、关层都不是修复方案。最小后续分开处理：对 raw105 追踪该 ROI 的黑色 static 贡献行与 gate 选择；对 raw113 追溯当前 road 基底颜色对应的参考来源。先分别确定来源，再讨论局部门控或材质修正，避免使用全局“去阴影”开关。


## 同一暗核的条件深度与道路相对位置

只对raw105的同一293像素暗核，再做24-feature joint累加。附加feature后RGB、alpha、depth_sum均逐字不变。static的ΣWz/ΣW为8.9648m，道路为9.2204m；static中心沿现有道路切平面法向的加权高度约5.76cm。keep状态a37条件深度5.4296m，同一暗核50条实际a37 LiDAR均值5.4222m。

当前暗载体的几何靠近道路，并不位于当前车尾/牌照深度；图像位置重合不能证明它是漂浮车尾纹理。它仍可能携带烘焙暗观测，但本轮不能证明历史来源，也不能证明真实光照造成cast shadow。

road1K的RGB、道路几何和道路opacity全部冻结，1K正式仅训练static opacity gates。因此raw113暗色基底在道路输入时已存在，不能称为“1K训练学进了阴影”。其后fine色场是另一项固定几何的颜色校准，本次ROI中作用很小。
