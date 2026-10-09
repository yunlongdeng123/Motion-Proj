# 当前研究状态

更新：2026-10-09。当前分支 `research/worldsim-v8.1-seen-to-scene`，继承 v7.7 的 `000ad1f0`。已确认工作主机 `wm-3090-1009`；远端使用独立 `/root/autodl-tmp/motion_proj_v81` checkout，保留原 v7.7 checkout。

当前 task `WS-V81-SEEN-TO-SCENE-20261009`，准备 run `r0`。按用户最新要求放下 DriveEditor，沿 Seen-to-Scene 公开方法从原始 SVD + RAFT + ProPainter 建立自有训练、推理与评测基础设施。路线为 P0 真实训练闭环 → P1 YouTube-VOS / DAVIS 外扩复现 → P2 驾驶遮挡—显露 DELETE 基线 → 固定 P2 后开展 P3 编辑感知传输。[架构、协议与准入](v81/SEEN_TO_SCENE_P0.md)。

已建立数据清单／真实 RGB 读取、显式外扩 mask、可微 flow resize／latent warp／合法参考融合、输入依赖预检与源码 ZIP 大小检查。CPU 13 项定向测试通过；这些只是基础接口验证。真实 SVD 训练 0 步、模型推理 0 窗，P0 尚未通过。官方源码与论文在传播参数、参考选择、损失及 mask 定义上有差异，已记录；尚不能称模型复现成功。

远端盘点：RTX 3090 一张；无本任务 GPU 作业。数据盘约 109 GiB 可用。原始 RAFT 与 ProPainter flow completion 权重存在；当前搜索未找到 YouTube-VOS / DAVIS RGB 和原始 SVD XT 1.1 组件目录。已恢复远端外网代理；现有 Hugging Face 凭据下载原始 SVD XT 1.1 返回 403 GatedRepo，需要访问授权或原始权重目录。下一步先补齐真实 RGB / 原始 SVD 并接入模型入口，确认输入与协议后才执行首个真实优化步；不用 DriveEditor 微调权重替代 SVD 初始化，也不提前加入 P3 创新。

源码归档通过 `.gitattributes` 排除历史 `docs/autoresearch` 附件，完整材料仍保留在 Git/GitHub 与 v7.7 历史；新数据／权重／视频／第三方代码放仓库外。本轮不清盘、不重写历史、不关机、不新建自动化。人工 verdict 留空。

v7.7 已退役矩形道路 proxy 与硬拼接伪标签；历史结果、候选、关键 checkpoint、SAM3 与 failure 资产保留。`failure_ledger_refs=[V77-F02]`，本轮 `failure_ledger_delta=none`，未新增模型失败结论。
