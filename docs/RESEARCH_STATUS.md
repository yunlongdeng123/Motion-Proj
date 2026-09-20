# 当前研究状态

更新：2026-09-20。当前分支：`research/worldsim-v7.4-h2-generative-surface`；研究阶段已转为V7.5。本文件是唯一当前快照；[V7.4 收口快照](archive/2026-09/v74-0920/STATUS_PRE_V75.md)与[更早历史](archive/2026-09/v74-0920/STATUS_HISTORY.md)保留。

## 执行范围

用户当前授权为单卡GPU下完成正式推理前的准备。V7.5围绕“重建状态误差经由真实条件接口影响动作条件自回归生成，并可能改变闭环决策”开展problem-first研究；“误差放大”尚是假设。协议见 [PROBLEM](v75/PROBLEM.md)。没有恢复V7.4旧队列，没有训练或关机任务。

## 当前证据

- 官方OmniDreams源码与主权重、一个公开场景、文本编码器和LightVAE/LightTAE已就绪。镜像来源与固定官方配置核对范围见 [运行准备](v75/PREFLIGHT.md)。
- 独立Python3.12/CUDA13环境完成依赖检查、3090 BF16运算和视频编码检查。官方V2 CLI帮助与配置加载通过；官方条件渲染完成237帧。
- 真实初帧与场景起点相差110401µs，准备输入已按真实初帧时间对齐相机轨迹。当前只有一个真实RGB帧，没有真实未来视频；该场景仅作工程开发。
- 文本与初帧embedding已缓存；主模型权重及生成cache加载通过。峰值分配显存约15.96GiB / 8.61GiB，分别对应编码阶段 / 权重与cache阶段，不能作为正式生成峰值保证。
- 正式世界模型生成前向 **0**；尚无V7.5科学badcase、生成对比或闭环研究结果。人工verdict为null。
- 发现并复现 [V75-F01](research_failures/entries/V75-F01.md)：官方batch命令构造未绑定逐案例图像和条件；本轮改用显式输入的底层官方pipeline，没有修改第三方源码。

## 下一步与边界

准备工作停止在正式生成之前。下一步是显式执行单段clean生成，检查真实输出、输入对应关系和峰值显存；通过后才运行完整clean基线，再进入预先指定的定位干预与恢复实验。可复现命令与证据路径见 [PREFLIGHT](v75/PREFLIGHT.md)。不能从权重加载成功宣称24GB足以运行完整生成。

保留V7.4 [F20](research_failures/entries/V74-H2-F20.md)、[F21](research_failures/entries/V74-H2-F21.md)、[F22](research_failures/entries/V74-H2-F22.md) 的否定边界：局部定位影响不自动推出普遍phantom与严重驾驶危害，普通修复必须保留为强控制。

下载定时检查因资源完成已暂停。当前没有自动推理、训练、下载或电源控制队列；不把GPU预检误报为正式研究完成。
