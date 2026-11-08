# 新街段车辆删除实验快照

Task `WAYMO-VAL029-V44-TRANSFER-20261108`，seed 0。源码来自实际执行版本，报告见 `docs/autoresearch/waymo_val029_v44_transfer_20261108/index.html`。

实验已结束，无待启动队列。当前自动曲面／道路适配与旧街段预处理有差异；结果未通过质量验收，不能宣称严格冻结方法迁移成功。报告保留全部可见帧分母、负例和40帧留出结果。

实际运行根目录 `/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1`，CPU库 `/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs`。`source/`依赖该实验根目录中的引擎、模型、输入及新资产；这些大文件不入Git。旧街段模块仅复用算法源码和冻结先验，不加载旧街段车辆／道路资产。

完整恢复包 `recovery_core.tar.gz` 已在用户本机 `work/val029_gpu/archives/` 核验，压缩1,307,738,240字节，解压流2,978,457,600字节；包括10K完整Adam/RNG运行状态、输入观测库、掩码、新车辆／道路资产和引擎API。`inherited_source/`另保存 SHAsset、SurfaceLODAsset、full_pvd、C²拟合工具源码。原TFRecord按输入收据SHA引用，保留远端原文件，没有重复打包。运行路径重定位、符号链接和已安装CUDA扩展需保持或重新绑定，尚未做异机恢复测试。

`review_delivery.tar.gz`已取回，含198帧和79帧视频、全部逐帧指标、所有1424份远端浮点层SHA、四个关键帧的浮点原件及实际日志／收据。本地报告的`verification.json`独立重算四帧浮点MAE和分组贡献，并完整解码两段视频。浏览器file协议预览被策略拒绝，仅做静态资源和实际图片核验。

正式原生10K；工程smoke/恢复5916步；C² 5轮IRLS；材质1000次FISTA；道路门32+968=1000步；PVD1000次冻结扩散推断。各项不能相加称为神经训练步数。

失败归档沿用V82-F06。原始失败目录、旧场景代码、未编辑用户工作保留。

Git快照已规范化行尾和行尾空白；原始字节源码保存在经SHA核验的恢复包与审阅包，未改动实际执行现场。
