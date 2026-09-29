# 交付入口

本地最终合成全检：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-synthetic/index.html`。

49 case / 11 receiver scenes；28独立通过 / 7拒绝 / 14待定。通过例来自8个receiver scene，共420帧供人工全检。30帧控制与10帧原生窗口分别标注，人工verdict初始为空。不是49个独立合格scene，密集类0、训练准入0。

四列：真实GT Y、合成输入X、A/B/C实例标注、masked-X数据条件。最后一列不是模型生成。网页支持视频同步、逐帧放大、每帧0/1及JSON导出导入。来源预审页仍在邻目录`v77-target-protected/index.html`。

本地3240张预览JPEG实解码、196视频存在/容器头检查；完整196视频实解码在远端完成。JavaScript语法检查通过，未声称浏览器点击播放已实测。

轻量实图：[通过P002](contacts/P002_f15.jpg)、[机盖覆盖D006](contacts/D006_f29.jpg)、[失焦W006](contacts/W006_f05.jpg)、[通过W019](contacts/W019_f05.jpg)、[T028实例污染](contacts/T028_f00_isolated_zoom.png)。所有完整原始与对照见同task/run目录，未删除拒绝例。
