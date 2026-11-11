# scene-065 白色轿车删除：GPU诊断

task=WAYMO-065-GPU-DIAGNOSIS-20261111。用户授权继续GPU诊断、sol6.1 max独立复核；AutoDL保持开机。入口index.html，简单模块图architecture.svg。065原生30K用户通过、029未通过的状态保留；本轮未改正式代码或发布新的编辑质量结果。

冻结原30K模型、归属、道路与交接，四帧60/119/150/198；20次同排序RGB／四格开关，4次局部损失梯度，199次候选复算＋2次逐行最终贡献，另2次极端候选退出，共227次光栅前向，0更新、0优化器。每个成功模块时间含加载与输出，不是CUDA核时间。RTX4080 SUPER，显存峰值约3.22GB，单线程CPU整理。短程零优化诊断，不适用小时级训练smoke；未追加训练、重建或删车正式预算。

颜色分层加和最大差1.311302185e-6；已发布与原语义删除四帧RGB差0。全199候选IDs完全复现，selected score因浮点累计非逐位一致；数值差与集合一致性分开记录。225次基础诊断delivery.tar.gz已取回，另2次极端控制的图、receipt、源码补取回，完整NPZ、全量score、逐行权重留远端。所有原PNG解码与报告静态引用检查通过；未进行浏览器UI验收。

结论：静态RGB本身携带车形；fit内超过99.94%的静态贡献已入候选，keep/left局部梯度与恢复项冲突；fit外约76%静态贡献不在候选。仅因positive-owner的额外排除在所测两帧贡献0，非主因。极端退出所有原候选仍有上部车形与下部动态／环境污染，不能作正式方法。参考不足、保持冲突、候选覆盖须分开验证。alphaT和代理MAE不当隐藏真值或修复成功。

源码见source，独立复核independent_review.md。GPU调用与参数只读检查见各receipt。failure_ledger_refs=[V82-F06]；failure_ledger_delta=updated:V82-F06；human_verdict=null。未关机。
