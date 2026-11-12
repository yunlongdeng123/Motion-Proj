# 当前研究状态

更新：2026-11-12（Asia/Singapore）。

本轮任务 `WORLD-ROAD-MATERIAL-FIELD-20261112` 已完成少量真实道路数据上的预训练模型对照、600步LoRA及世界场→高斯的真实GPU验证。[图像、架构与完整分母](autoresearch/road_material_field_gpu_20261112/index.html)。当前代理图像质量未通过，human_verdict=null。用户要求AutoDL保持开机。

统一固定LiDAR拟合道路面，图块为128×128、2.56米、2cm纹理格，特征格4cm。清晰度在划分前统一门控，最差原图投影轴≥64px，实际64.5–153.7px。12train/4eval/44异地候选全部世界格去重，eval含2标线。原生街段重建全帧合同未改变。这里是同街段空间留出，未验证跨街段泛化。源RGB属于带照明观测，首轮材质代理是参考曝光下的外观纹理，未辨识真实PBR。

实际学习为SD1.5原生补全9通道UNet的rank4 LoRA，797184可训练参数。固定600正式更新，每个训练块50次，0/200/400/600完整状态和便携权重均保留并取回。固定异地参考、相同上下文/掩码/prompt/DDIM20步/seed101、202下，4eval×2seed、21000像素对MAE 0.0291271→0.0186040，3/4世界块、7/8单seed改善。紫色色差、接缝与局部标线仍在，均值不代表画质通过。

在唯一有效连续面fineguard源上，原生冻结检索A几何颜色/频谱、B SuperPoint、C DINOv3二维、D世界特征融合分别为MAE .04687969/.05119263/.06861912/.05038235，4query、10500像素。C/D共有候选池重排相同趋势。D优于C，仍比A差约7.47%，D模型更新0。用冻结D_top1再做native/同一个600终点的零更新完整通路前向，误差.0263989→.0231615，4/4块均值改善，单seed仍回退；它未超过固定参考的LoRA结果，不择优改主结论。

特征通过真实Brown相机投影与可见权重绑定世界道路。最终输入在所有199原帧遮住连续query表面，51来源供C/D共用。fine中心/四角/内部随机点及42000细texel×51来源的原图双线性四邻独立审计均无泄漏。旧coarse中心审计漏掉细格，旧特征/旧D参考作废保留，E训练PNG/分割/几何/固定参考不变。当前入口为data/scene065_v3_clear64/dataset_fineguard.json→sources_fineguard.json，数据最终包名scene065_v3_continuous_sourceguard_data_bundle.tar.gz。

固定世界材质场实际输出两相机的原生/LoRA CUDA高斯图，几何、alpha与depth逐值一致，保存世界坐标、纹理、支持度与来源。float32大世界坐标插值最大linear RGB往返误差约.0041，未把toy误差当真实误差。该验证只覆盖路块渲染，不宣称完整街段删除残影解决。

工程账单列2101更新（600秒合成smoke2048、旧实拍接线37、最终真实接线16），与正式600分开。原生/零LoRA像素一致，非空Adam完整恢复后的下一更新精确，终点600独立进程恢复图精确。TRELLIS.2仅完成固定网格归一化/逆变换和纹理接口前置，未下载4B或运行。

远端现场：/root/autodl-tmp/drivingeditor_repro/campaign/road_material_field_20261112_r1/。代码为scripts/worldsim_v82/road_material_field/独立通用实验入口，原生30K与既有008/029/065结果保留。沿用V82-F06；后续重点由本轮候选和生成图像决定，不把少量拟合收益写成通用方法成立。AutoDL保持开机，无关机动作。
