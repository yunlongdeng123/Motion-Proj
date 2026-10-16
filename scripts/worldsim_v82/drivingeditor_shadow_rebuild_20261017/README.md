# 从零重建实例场景与车辆关联阴影

任务 `WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017 / v3`。此目录是新训练分支，初始化仅来自真实 LiDAR/RGB/轨迹和 reviewed body masks。不读取 GN11 checkpoint、旧道路替换层、原渲染图或五盒体代理。

运行顺序由只读数据预检、实际CUDA ABI检查、分阶段测速与训练入口约束；CPU合成夹具不能替代驾驶场景实验。当前执行状态统一见仓库 `docs/RESEARCH_STATUS.md`。

```mermaid
flowchart LR
    A[训练帧 RGB / LiDAR / mask / 轨迹] --> B[Stage 1 表面几何 + 独立 Actor]
    A --> R[跨时刻真实道路参考]
    B --> C[Stage 2 材质 + 太阳 / 环境光]
    R --> C
    B --> D[同一 Actor 高斯的可微光线积分]
    E[编辑后 Actor 位姿] --> D
    C --> F[Stage 3 受约束联合优化]
    D --> F
    F --> G[场景 RGB / 无车辆影道路 / 编辑阴影]
```

## 表示和训练

| 阶段 | 更新 | 固定/限制 | 监督 |
|---|---|---|---|
| 几何，拟定30k步 | 逐高斯位置/尺度/旋转/opacity、canonical颜色与 Actor 小幅视角项、天空 | 静态层无 time/velocity/lifetime；真实轨迹固定；几何锚在测量点附近；只在此阶段增密 | 非道路重建、车身 RGB、完整 mask 内/外、全场景可见实例、稀疏 LiDAR 逆深度、跨时刻干净道路颜色、局部法线 |
| 材质/阴影，拟定10k步 | 道路逐高斯 albedo、太阳方向/角半径/强度、环境光比例、Actor/canonical外观 | 默认几何和opacity固定；阴影只读同一 Actor 几何 | 整图均匀像素 RGB、随机patch SSIM、车身分层 RGB、canonical重建与教师、同一组深度/实例/道路约束 |
| 联合，拟定5k步 | 材质、canonical外观、光照及受约束几何/opacity | 几何相对上一阶段最多2cm，尺度log±.12、opacity logit±.3；学习率×.1；不增密 | 持续保留上述重建/车身/实例/深度/跨时刻约束 |

这些步数是初始预算，未经真实吞吐测量，不是承诺的收敛步数或总时长。

道路为局部 PCA 定向的薄各向异性3D高斯，具有独立颜色和opacity，不是固定平面。第一阶段把道路的跨时刻稳定颜色作为 canonical 外观目标；带车影的道路像素不直接推动几何/本色。第二、三阶段主RGB覆盖整幅图的均匀像素，patch用于SSIM；不使用历史人工修复ROI。视角残差仅给Actor，幅度±0.08线性RGB且无时间输入。

`tracing.py` 沿太阳射线解析积分 Actor 3D高斯密度；同一 `field.evaluate()` 同时给主相机渲染和遮挡计算提供编辑后的几何/opacity。投影接收位置来自主渲染的三维 G-buffer。太阳盘采样产生软边，ambient/sun共享场景参数。3DGS opacity 转密度使用最薄轴穿心光学厚度匹配；6σ聚合AABB粗剔除省算，积分块反传重算省显存。这些是明确的近似，GPU成本与阴影轮廓仍须实测。本实现没有迁入IRGS的OptiX、间接反射、多光源或完整车身BRDF；太阳/材质分解也不保证从单段视频唯一识别。

本轮材质分解专注道路，Actor正常暗面、自阴影和反射由**新训练**的canonical radiance及小幅视角项解释，Actor外观在后两阶段仍可优化，不强制整车变成Lambert材质。道路颜色/位置/法线的条件G-buffer全部共享全场景alpha*T；重着色只替换新场景自身的可见道路贡献，保留车身贡献。这里没有使用旧场景或观测RGB作图像补丁。

## 损失约定

完整不透明度监督使用**未预乘GT mask**的逐Actor独立渲染alpha：mask内 `(alpha-1)^2`，mask外 `alpha^2`，分别按各域实际元素数归一化，默认权重各0.05。全场景alpha*T实例贡献另外监督可见身份，默认0.1。未知mask不参与两域；默认正式训练直接拒绝未齐全mask。已知不存在的Actor有明确零目标。`--allow-partial-masks`仅用于显式登记的数据缺口运行，不能冒充完整监督实验。

RGB=1，SSIM=.2，LiDAR逆深度=.1，跨时刻道路=1，canonical重建=.3，车身RGB=1，Stage1教师=.1，几何/法线各.02，视角项=.01。权重集中于`config.py`，尚未调参。教师保留新分支Stage1的车身重建，不加载旧GN11。

## CPU准备与审计

在本目录执行；远端仍使用已保存的DrivingEditor环境，不安装或覆盖旧内核。

```bash
python run.py prepare \
  --scene /path/to/original/inputs/scene0008_089_137 \
  --annotations /path/to/complete/real/laser_labels_audit \
  --reviewed-instances /path/to/real/reviewed/rigid_prepared \
  --output /path/to/new/v3_data

python run.py preflight --data /path/to/new/v3_data \
  --config configs/scene0008.json --output /path/to/new/full_run
```

`annotations`必须来自真实完整laser_labels，不能由渲染或框推断造标签；`reviewed-instances`只读`instances.npz`中的frame/track/mask_known和`instance_masks/*.png`。导出器为所有原始车辆轨迹建立Actor，训练帧LiDAR源点按当帧真实框分离；重叠歧义点与其他类别动态点不进入静态层。Actor缺少训练帧真实LiDAR则停止，不用OBB补成车辆。所有坐标以第一帧原始ego位置重置原点，单位保持米；不使用旧模型归一化。

点的初始颜色来自同帧深度可见RGB；没有可见颜色的点仅用0.5作为初始化先验，不把该值作为训练标签。单相机未见的车辆背面/道路不能由代码验证自动认证为真实恢复。

初始 ground 是LiDAR局部低表面/法线候选，**不是人工语义道路真值**，仍需实际覆盖审计。只对真实深度一致、车体保护外、训练帧可见的同一道路点收集RGB；要求亮簇多次重复，拒绝一次亮异常；完全未见或支持不足的点没有伪参考。始终在阴影里的道路仍有不可辨识性。缺少GT时不声称恢复了真实干净颜色。原始raw101/113/125只用于新分支测试，种子/参考不读其RGB或LiDAR；这些时刻过去被旧实验用过，不能称整个研究过程未见过的最终测试集。

## GPU启动后依次执行，当前没有运行

```bash
# 先实际ABI/梯度与各阶段测速，不直接排45k长作业
python run.py benchmark --data /path/to/new/v3_data \
  --config configs/scene0008.json --output /path/to/new/benchmark --steps 20 --execute

python run.py train --data /path/to/new/v3_data \
  --config configs/scene0008.json --output /path/to/new/full_run --execute

python run.py progress /path/to/new/full_run

# 中断恢复要求相同配置、数据契约和ablation
python run.py train --data /path/to/new/v3_data \
  --config configs/scene0008.json --output /path/to/new/full_run \
  --resume /path/to/new/full_run/latest.pt --execute

python run.py evaluate --data /path/to/new/v3_data \
  --checkpoint /path/to/new/full_run/joint_final.pt \
  --track-id 3S_GJHyCGGK4ypFPVKOOqA --output /path/to/new/eval49 --execute
```

上述raw_track_id来自已保存的黑车诊断记录；也可用`--target`指定新manifest实例索引，不能沿用旧actor0假定。正式场景主渲染只接受已审计DrivingEditor四输出feature rasterizer；GPU probe验证13个含负值通道跨批共享合成、CPU参考投影与反传。不兼容即停止，无隐式内核替换、CPU大场景fallback或自动续跑。

Stage1增密及结束时，会按新点位置重新投影真实训练RGB/深度，刷新道路参考；不把父点颜色作为子点永久纹理。Stage1→Stage2→Stage3保存独立final、latest、Adam、RNG、教师、增密后点表与输入契约。每50步更新progress/metrics，原子保存。`progress.py`只读、不会开机或启动worker；进度文件不等于进程存活证明。测速只反映初始点数，还需给增密、教师、I/O和完整视频留预算。

评价导出8条逐帧流：原位、关影、删目标、删全部、连续平移、连续±30°旋转、覆盖率、固定相机删全部；同时保留实际body/road/identity分母、无有效样本为null。没有反事实GT就不伪造编辑质量指标，人工verdict始终由用户填写。

## 消融

`--ablation full|no_shadow|no_clean_road|no_body_teacher|frozen_joint_geometry|hard_sun|opacity_stage2`。每臂独立新输出、同seed/数据划分/初始点；记录resolved_config、输入角色与缺口。`no_clean_road`同时禁用参考初始化、参考损失和参考刷新，恢复保存的原始测量颜色，避免消融仍偷用亮道路先验。消融不自动排队，逐臂显式运行。旧五盒体/固定平面结果作为历史对照保留，不接回新模型。用同样49帧指标、原始及编辑视频核对①道路细节、②阴影轮廓、④车身质量，特别单列raw106–108与±30°。

## CPU验证

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v
```

包括独立SciPy数值积分、解析梯度、共享几何编辑、mask外梯度、几何冻结、非平面原始导出、测试帧RGB隔离、真实三阶段小场景更新、增密后逐位恢复、连续编辑与空分母。详细结果见随交付保存的CPU检查记录。测试不调用CUDA或远端主机。

## 外部依据

- [LumiMotion论文](https://arxiv.org/html/2604.10994v1)与[官方训练脚本](https://github.com/joaxkal/LumiMotion/blob/main/bash_scripts/enerf-actor1.sh)：借鉴几何先行、后续材质光照及持续重建约束；不复制其动态形变系统。
- [IRGS官方代码](https://github.com/fudan-zvg/IRGS)与[几何学习率默认值](https://github.com/fudan-zvg/IRGS/blob/main/arguments/__init__.py)：借鉴适于光照的表面与第二阶段冻结；没有声称复现其完整光线传输。
- [Gaussian Shadow Casting论文](https://arxiv.org/html/2401.06116v1)：借鉴各向异性密度射线解析积分；这里直接绑定同一Actor高斯，不另设随RGB任意变形的遮挡代理。
- [RNG论文](https://arxiv.org/html/2409.19702v3)：作为后续深度/边缘修正参考；本轮没有加入该网络。
