# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

二维教师独立范围 `ROAD-2D-TEACHER-PROMPT-20261120`：[五seed提示词对照与组件](autoresearch/road_2d_teacher_prompt_20261120/publication.html)完成。用户已采纳整帧1152×768＋注意力LoRA T0为后续默认；本轮5seed×2prompt×6任务正式60次，另4次真实GPU smoke，4进程并行，训练与三维修改0。合成两底图等权ΔE76 2.239597→2.242254、共同频谱偏差1.201894→1.141373，但改善配对仅7/20与8/20，seed波动未降低；新prompt不晋升，不再追加调词。12次旧1152三图重放和实际W/K门通过，60原始PNG及全部对照资产已取回核验；真实隐藏道路无GT。保持历史活动提示词，原CPU/GPU报告及其他三维任务保留。method human_verdict=用户采纳当前审阅样例；prompt human_verdict/problem_solved=null。实例按用户明确要求保持开机。

本轮任务 `WORLD-ROAD-MATERIAL-FIELD-20261112` 已完成少量真实道路数据上的预训练模型对照、600步LoRA及世界场→高斯的真实GPU验证。[图像、架构与完整分母](autoresearch/road_material_field_gpu_20261112/index.html)。当前代理图像质量未通过，human_verdict=null。用户要求AutoDL保持开机。

统一固定LiDAR拟合道路面，图块为128×128、2.56米、2cm纹理格，特征格4cm。清晰度在划分前统一门控，最差原图投影轴≥64px，实际64.5–153.7px。12train/4eval/44异地候选全部世界格去重，eval含2标线。原生街段重建全帧合同未改变。这里是同街段空间留出，未验证跨街段泛化。源RGB属于带照明观测，首轮材质代理是参考曝光下的外观纹理，未辨识真实PBR。

实际学习为SD1.5原生补全9通道UNet的rank4 LoRA，797184可训练参数。固定600正式更新，每个训练块50次，0/200/400/600完整状态和便携权重均保留并取回。固定异地参考、相同上下文/掩码/prompt/DDIM20步/seed101、202下，4eval×2seed、21000像素对MAE 0.0291271→0.0186040，3/4世界块、7/8单seed改善。紫色色差、接缝与局部标线仍在，均值不代表画质通过。

在唯一有效连续面fineguard源上，原生冻结检索A几何颜色/频谱、B SuperPoint、C DINOv3二维、D世界特征融合分别为MAE .04687969/.05119263/.06861912/.05038235，4query、10500像素。C/D共有候选池重排相同趋势。D优于C，仍比A差约7.47%，D模型更新0。用冻结D_top1再做native/同一个600终点的零更新完整通路前向，误差.0263989→.0231615，4/4块均值改善，单seed仍回退；它未超过固定参考的LoRA结果，不择优改主结论。

特征通过真实Brown相机投影与可见权重绑定世界道路。最终输入在所有199原帧遮住连续query表面，51来源供C/D共用。fine中心/四角/内部随机点及42000细texel×51来源的原图双线性四邻独立审计均无泄漏。旧coarse中心审计漏掉细格，旧特征/旧D参考作废保留，E训练PNG/分割/几何/固定参考不变。当前入口为data/scene065_v3_clear64/dataset_fineguard.json→sources_fineguard.json，数据最终包名scene065_v3_continuous_sourceguard_data_bundle.tar.gz。

固定世界材质场实际输出两相机的原生/LoRA CUDA高斯图，几何、alpha与depth逐值一致，保存世界坐标、纹理、支持度与来源。float32大世界坐标插值最大linear RGB往返误差约.0041，未把toy误差当真实误差。该验证只覆盖路块渲染，不宣称完整街段删除残影解决。

工程账单列2101更新（600秒合成smoke2048、旧实拍接线37、最终真实接线16），与正式600分开。原生/零LoRA像素一致，非空Adam完整恢复后的下一更新精确，终点600独立进程恢复图精确。TRELLIS.2仅完成固定网格归一化/逆变换和纹理接口前置，未下载4B或运行。

远端现场：/root/autodl-tmp/drivingeditor_repro/campaign/road_material_field_20261112_r1/。代码为scripts/worldsim_v82/road_material_field/独立通用实验入口，原生30K与既有008/029/065结果保留。沿用V82-F06；后续重点由本轮候选和生成图像决定，不把少量拟合收益写成通用方法成立。AutoDL保持开机，无关机动作。

## ROAD-GAUSSIAN-TRANSFER-20261122

065相似路面高斯迁移已完成原资产/零步迁移/局部拟合400步及五视角公共重载。[算法、组件、结果与来源](autoresearch/road_gaussian_transfer_20261122/README.md)。采用局部保距三维路块与完整协方差旋转，新增扩散0次；smoke10和恢复探针2次丢弃。零步到拟合的固定C/S覆盖改善、保护区RGB/alpha差0；覆盖仍弱于原资产，细裂缝、短缺口和joint车形残留仍在，未晋升默认资产，human_verdict/problem_solved=null。原报告增加实验组，资产/完整状态取回，GPU按用户要求保持开机。failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

源质量补充 `ROAD-GAUSSIAN-SOURCE-QUALITY-20261122`：[原尺寸核查与固定源RGB控制](autoresearch/road_gaussian_source_quality_20261122/README.md)。1920原生核心确认当前源块存在弱纹理对应；仅核心RGB、真实照片98/137训练400步后，本次未拟合的86/110/173查询细高频对应均下降，86颜色误差也回退。因此暂停搬运这份源修复候选，不增加目标更新或扩散；完整源几何/尺度重建尚未测试，不据此物理否定全部搬运方法。最终资格掩码重放与旧工程臂分账、资产取回验证完成，人工verdict/problem_solved保持null，GPU按用户要求开机。

源材质优先 `ROAD-SOURCE-FIDELITY-20261122`：[00／02实拍与固定高斯颜色链路](autoresearch/road_source_fidelity_20261122/README.md)。同物理面积、原生像素人工核验选02／115实拍、117复核，00／137对照；本次保存加载和常色渲染未引入偏色，历史RGB拟合与原照片Brown／原生针孔相机不一致分别核验。固定几何实测颜色算子、有界求解及相机校正改善参考色调，但跨视角纹理与86色调仍有退化，未找到通过本轮检查的可迁移源高斯；目标迁移／融合和新扩散均0。918F／1B中反向仅为贡献测量，12通道RGB求解、梯度优化更新0；完整包、CPU恢复和原报告同页对照保留。原资产不改，GPU保持开机；human_verdict/problem_solved=null，沿用V82-F06。

本轮七臂仅覆盖单参考与链路诊断，未测试多帧参考池；对应400步随机训练对照另在road_source_multiframe_20261122_r3记录。

参考数量配对 `ROAD-SOURCE-MULTIFRAME-20261122`：[source02两帧／八帧各400步及恢复入口](autoresearch/road_source_multiframe_20261122/README.md)。共同原始900核心RGB、固定形状与halo、共享核心外真实道路曝光，seed0均衡每步两帧；主QUERY100／110／117的原始Brown域平均MAE .008283→.007765，HF相关 .1250→.1977，117略回退且纹理能量下降。说明多帧参考池有可测收益，不能用此前单／双参考诊断提前否定；本轮限固定核心颜色对照，干净核心单独原生渲染与后续搬运另记。1684F／1612B、正式800更新与smoke6分列，QUERY核心未入loss或曝光，86单列曝光较弱stress；目标迁移／融合和新扩散均0，沿用V82-F06，human_verdict/problem_solved=null。
