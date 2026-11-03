# 当前研究状态

2026-11-03按用户最终要求，scene008在输入身份阶段仅选择a37，重新初始化并沿用原v41 B训练设置，完成fresh10k后停止；不跑30k，不使用subagent、不关闭GPU。原五车B动态检查点不作单车初始化或临时归属筛选；此前R4等结果保持原样。

[单a37 B10k组件、完整49帧分层与删除结果](autoresearch/scene008_a37_only_B10k_20261103/result_report.html)。唯一动态真实track为vBOasuXpjI7h7d-X_VEQdA，原actor_4重编号actor_0；49帧动态mask/RGB和98,452个真实动态初始点只来自a37。a5/a26/a18/a6保留在原B的时间静态/背景表示中，初始真实种子分别1442/1004/151/798。a6原场景采样命中855点，57点被原B跨时间0.1m排除规则去除；保持原规则，未另加种子。

正式更新严格10000，完整四件状态、配套49帧评价发布后退出，无10001步。原生循环iterations与LR horizon仍保留30000，以维持原B的LR公式；只由外部完整节点截止。实际与原B全部10000步的相机ID、time_shift、各组LR逐项相同；配置差异只有输入/输出路径和单一动态PLY列表，完整mask内外alpha监督、损失、分辨率、SGCM/NPS、增密及opacity reset均继承。

真实GPU smoke为3000步、11.07分钟（含1k/2k/3k检查点与完整49帧评价）。新进程完整参数/Adam/RNG/相机采样器/SGCM/EMA/运行状态恢复逐项相同，恢复后完成2次更新并进入下一评价入口。独立评价入口曾因PyTorch mmap不接受Path参数失败，改为str路径后0更新重放通过；旧失败现场保留，未重跑smoke训练。正式从零开始，未续用smoke。

10k原生训练域合成PSNR=30.454978dB、SSIM=0.936408、LPIPS=0.086104；a37mask内动态alpha均值0.98887808、mask外0.00400283。完整分母、其他保留车域的动态泄漏与删前后RGB变化见逐帧指标；输入只有a37不等于假定边界泄漏为0。49视图均为observedtrain BUILD，没有独立测试或无遮挡道路GT。当前为用户要求的单目标B10k身份基线，没有加载R4或a37/a6补丁，新增道路修复更新0；残留与保留车重建质量由完整图像检查，不以合成PSNR宣称黑影修复完成。

正式耗时28.84分钟，峰值CUDA分配8.23GiB、保留13.80GiB。重新载入正式模型后49帧原生合成/动态图/静态PNG最大8-bit差值0；三视频均完整解码49帧。原B四件权重大小/mtime核验不变。完整新10k权重/Adam/状态包已取回并核对SHA256；Git保存报告、配置、代码与媒体，原生大权重包保留在远端现场及本地outputs，不写入Git。

本轮正式10000+smoke3000+恢复2，共13002次更新；此前11238，累计24240。human_verdict=null、quality_accepted=false，failure_ledger_refs=[V82-F01,V82-F06]、failure_ledger_delta=none。远端现场/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_only_B10k_20261103_r1。GPU按用户指令保持开启。
