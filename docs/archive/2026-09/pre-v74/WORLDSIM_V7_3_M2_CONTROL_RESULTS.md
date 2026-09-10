> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 同信息控制与自由空间失败来源

批次20260907T154500Z__physical-controls五个任务全部done（native融合+四组训练），code512048bb；全组相同初值/120steps，单fit Actor/同一输入。原生head只在带视觉候选训练；LiDAR-only保留同query解码器。全部方法继续用字面三角首交点。

| 同初值/120steps候选 | hit | early | miss | 原始束free均值(m) |
|---|---:|---:|---:|---:|
| joint-no-free-r1 | 0.7616 | 0.0445 | 0.1710 | 0.0902 |
| joint-r2 | 0.7543 | 0.0524 | 0.1651 | 0.1333 |
| lidar-only-r1 | 0.7276 | 0.0608 | 0.1753 | 0.1058 |
| pointwise-r1 | 0.7460 | 0.0474 | 0.1774 | 0.1497 |

空间查询相对等容量逐点只增加约0.84百分点hit；带视觉相对LiDAR-only增加约2.67百分点hit，但free更差，尚不能支持视觉或局部交互的净优势。joint-no-free的hit与free反而更好；不能从120步一个Actor得出free监督无用，只能判断当前梯度/支持组织尚不奏效。

失败来源诊断：joint-r2 heldout的1330.58m累计侵入中，completion面遮挡原始非Actor返回贡献952.42m（71.58%），同类evidence贡献238.33m；completion遮挡真正Actor返回贡献92.58m。build同样由completion遮挡非Actor返回主导（1827.24/2556.72m）。这是少量过剩曲面遮挡远背景，单次侵入可达56m；不是仅target深度平均值的噪声。LiDAR-only也出现同方向问题，不能归咎视觉信息本身。

研究调整依据：原来512个completion query均匀初始化在Actor框体积内，一侧coverage只牵引最近的面，无法监督所有多余支持；hard free在支持/排序切换处缺少轮廓梯度。已经重新检索[On-Surface Prior CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Reconstructing_Surfaces_for_Sparse_Point_Clouds_With_On-Surface_Priors_CVPR_2022_paper.html)、[作者代码](https://github.com/mabaorui/OnSurfacePrior)、[AdaPoinTr官方代码](https://raw.githubusercontent.com/yuxumin/PoinTr/master/models/AdaPoinTr.py)。迁移的是表面支持初始化/投影查询思想，不引入未经训练的SDF先验，也不声称复现。

下一轮只改变completion初始化：从可训练原生DPT点图的米制Actor支持中选空间分散种子，保留所有后续不受小位移上限约束的查询更新；深度输出本身进入这条几何梯度路径。配一个LiDAR表面种子控制，区分“离开任意体积初始化”与“视觉几何提供新增支持”。保持原有几何/free权重、曲面尺度、视图、训练步数，不同时引入event或更多正则。没有可靠视觉支持时保留LiDAR种子并报告覆盖。目标点/heldout结果不参与种子生成。

F02继续active且已有强负证据；F03未通过事件实验，不误写为已解决；F04场景拼接未进入本组；F05独立日志泛化待后续，当前10k build点近距离Actor不能代替稀疏观测主队列。下一编号V73-F06。
