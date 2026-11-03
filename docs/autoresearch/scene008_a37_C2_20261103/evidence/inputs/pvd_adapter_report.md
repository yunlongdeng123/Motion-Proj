# PVD 实际 completion 到 a6 局部 patch

![Architecture components](pvd_adapter_architecture.svg)

本文件仅记录进入受控优化前的CPU检查，不代表最终图像质量通过。实际输出是唯一seed0、1000步DDPM产生的2048点，包含200个固定真实partial点和1848个生成点。官方car/train共2458个shape的统计及逐npy CRC均经prior分支验证。

坐标使用生产者明确提供的逆矩阵：native x向前/长度、y向左/宽度、z向上/高度；PVD X宽度/右、Y向上、Z长度/后。固定完整a6框先映射到训练集平均完整框span与center，再应用官方训练mean/std。方向假设native+X→PVD-Z明确记录，未进行目标GT、方向或seed搜索。adapter没有再次居中或调整尺寸。200固定点与原始canonical行最大误差1.19e-7m，模型坐标逆变换复核误差1.19e-7m。

![PVD canonical views](actor_surface_pvd_seed0_views.png)

实际点位不变，5mm去重只选择原点。strict完整框过滤拒绝95点；200个固定partial永不标成prior；8个局部PCA无法提供二维平面支持的点被拒绝；218个距固定真实anchor<0.04m的生成点去重。最终为5976固定真实anchors加1527个实际PVD生成patch，共7503 anchors。

PCA只使用实际completion邻域，最多16邻点，物理半径上限0.25m；normal符号相对固定actor原点。tangent σ=min(0.60×最近3个其他点的中位物理间距,0.060m,0.45×局部normal半径)，不设人工正最小尺度。保留prior σ中位0.05683m，最大0.06m，防止低密度点变成巨型splat。没有造mesh、封口或补车形。

全部5976真实anchor的位置、normal以及原始行映射与先前Omni adapter真实块逐值相同。3586 anchors使用可靠多帧颜色；prior与其他unknown仅采用中性实测车体主色，始终标记inferred。测量缺失的z=-0.72..-0.21045m下部有856个actual generated patch中心，不称GT车轮或完整封闭表面。

均匀5cm真实voxel-anchor到稀疏2048点的nearest point距离mean=0.08533m、p90=0.14683m；prior报告的原始测量行加权mean=0.0657m、p90=0.1167m使用不同权重。两者都是point-to-point距离，不是surface残差。0.20m半径点图的最大分量包含99.17%输出点；这只用于排查大块分离结构，不当作mesh连通性证明。三视图显示车尺度的车顶/舱、下车体及轮区支持，未见整体塌缩或Omni式大宽壳。

因此按root授权登记status=PROPOSAL_VALID_FOR_CONTROLLED_OPTIMIZATION，formal_eligible=true仅表示可控工程/优化入口；quality_accepted=false、human_verdict=null。CPU adapter未运行任何训练或GPU操作，已通知baseline_audit重新选择PVD贡献候选。后续32step与1k实际优化、评价、最终质量判断由root记录。
