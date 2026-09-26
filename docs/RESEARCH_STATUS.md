# 当前研究状态

更新：2026-09-26（Asia/Singapore）。本分支 `research/worldsim-v7.6-ego-view-densification` 的 V7.6 已按用户最新要求关闭，终态 `CLOSED_BY_USER / ASSET_QUALITY_INSUFFICIENT`。详细证据和保留边界见 [V7.6 收口](v76/CLOSEOUT.md)，失败卡见 [V76-F03](research_failures/entries/V76-F03.md)。

## 关闭原因与执行范围

**当前 HUGSIM 和 VAD-GS 两批资产的对象级质量，尚不足以直接支撑我们想要的高保真反事实编辑。** 停止“逐场景 Gaussian 资产 → editor”主线，不再继续对象级修补、追加该路线训练或重启旧队列。现存结果作为 baseline / failure evidence 保留，不删除或清理资产。

VADGS-P0R1-000 已完成30k训练、75视图官方时间test和61视图Camera5外推。30k官方test为25.2378 / 0.8028 / 0.1717（PSNR / SSIM / LPIPS）；横移队列在actor 7世界变换断言处退出。整体图像指标不能代替可编辑对象质量，断言错误也不单独作为资产质量失败的依据。原始结果和未完成范围见收口报告。

## 后续方向

后续重建基座统一采用 **VGGT 系列**，V7.7 起点为 **冻结 VGGT-Ω + GT实例选择 + 解析 MOVE / DELETE / INSERT**。P0零训练，不加入自然语言、闭环、视频扩散精修或新学习模块；新阶段先检验对象分离、米制编辑遵循、背景保持和多视角完整性。新分支 `v77` 从本收口节点创建，后续当前状态只在该分支维护。

## 任务与电源

收口检查时未发现 V7.6 / HUGSIM / VAD-GS 训练、评价、渲染及后续启动控制器；没有需要强制终止的存活任务。原pipeline状态保留并另存关闭标记，不将历史失败改写成成功。未发现本地V7.6自动化；现有HUGSIM自动化已暂停。

本次授权为关闭V7.6、建立v77并准备权重；旧报告的完成后关机安排不作为本次新阶段的电源指令。本次不执行关机。
