# 当前研究状态

更新：2026-09-28，v77，当前task WS-V77-NINE-FULL-20260928/r1正在执行。用户要求九个scene全部同一配置跑完整background＋显式actor DELETE，并做原视频／原位重建／DELETE三列视频。九例均已接触，定位为开发对比，不能称独立泛化评测。

旧0230/22、0255/25、official_000/12与新增350/7、663/6、191/12、425/1、382/4、756/3，统一首30帧六相机。17个目标相机流SAM2已完成；所有非空目标mask按同一规则进入补景，空mask如实保留并标输入问题。全部继续一次有界诊断链，结果不得冒称通过；旧r18/r21/r30和九例上一轮输出不覆盖。

单卡有界队列正在依次执行9个Hunyuan shape/PBR、DriveEditor、270次冻结Ω；本地Blender CPU随资产完成生成GLB及原位RGBA/深度层。实际状态先查 /root/autodl-tmp/runs/worldsim_v77/WS-V77-NINE-FULL-20260928/r1/sequence_state.json 和对应log；不要重复启动。不是定时任务，没有自动重试、训练或电源操作。

所有例统一GT位姿/尺度/高度与固定光照。首例暴露固定Rz-90造成水平轴错放，错误GLB及局部渲染已保留；九例共同采用水平PCA长轴对齐＋真实参考0/180分数判别后烘焙方向，未从编辑结果人工逐例调参，模型未重跑。已知截边、微小/遮挡参考、暗夜mask缺失均保留。QUERY读取同一B_t与GLB，DELETE只改visible；默认三列展示纯几何，RGB无支持区回填单独开关。

完成前不得把上一轮52视频或15测试冒称本轮验收。预期审核 outputs/v77-nine-full/index.html 尚未发布；见[共同协议与组件图](v77/NINE_FULL_PROTOCOL.md)及[冻结登记](autoresearch/worldsim_v77/nine_full_20260928/registration.json)。本轮人工verdict空。完成后更新同一V77-F02、资源/测试/失败证据，并提交推送。
