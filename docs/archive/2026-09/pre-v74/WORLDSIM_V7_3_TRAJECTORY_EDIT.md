> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 固定规范表面的轨迹编辑演示

2026-09-08。已在一条旧AV2开发日志完成固定表面、逐束时刻和场景硬排序的轨迹编辑。此演示使用固定LiDAR-only r9表面，证明接口组合能力；没有反事实真值，不承担主视觉模型的重建增益或编辑真实性结论。

## 几何定义与信息边界

原场景为背景与各规范表面的刚体组合：

\[
\mathcal S(t)=\mathcal S_{bg}\cup\bigcup_i T_i(t)\mathcal S_i^{canonical}.
\]

对一个选定Actor执行

\[
T'_i(t)=T_i(t)\operatorname{Trans}(0,2,0),
\]

即沿该Actor即时局部+Y方向偏移2米。每个规范表面只加载一次BVH，形状/法向/连接保持同一对象；不重新生成几何或修改任何网络参数。按每束实际时间插值只读轨迹后，把原点/方向逆变换入规范坐标，和固定背景及其他Actor层求全局最近交点。

参考[Street Gaussians ECCV2024](https://github.com/zju3dv/street_gaussians)的Actor/背景组织和[NeuRAD CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.pdf)的actor shift示例，本实现保留项目既有显式三角面物理读出，没有迁移外观opacity、概率回波或其网络表示，也不声称复现这些方法的性能。

只查询原文件中有观测返回的束方向和时刻。没有可用的完整no-return束协议，故不能将其称为完整激光雷达仿真。旧实测距离不作为编辑后的监督或真值。没有存储背景/其他Actor交点的暴露方向保持未知，不通过修补背景制造完整性，也未验证编辑运动的交通或碰撞可行性。

## 实际运行

任务`WS-V73-M4-TRAJECTORY-EDIT-01/20260908T033000Z__old-av2-r9-lateral2m-r1`，代码e45118ef，CPU5.5590秒/RSS0.95334GiB，正常退出。使用旧日志02678d04-cc9f-3148-9f95-1ba66347dff9的21个Actor、同四build雕刻背景和两次原始heldout扫描，分别编辑全部4个build元数据中速度>2m/s的Actor。没有按模型输出或heldout质量选择对象。每个独立编辑情景187494条查询束。

| Actor前缀 | build点数 | 速度m/s | 原先首交点数 | 编辑后首交点数 | 新增遮挡 | 释放束 | 释放到未知 | 释放到其他Actor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6247b383 | 12 | 5.834 | 6 | 4 | 3 | 5 | 5 | 0 |
| 9054f80c | 12 | 6.026 | 2 | 2 | 2 | 2 | 2 | 0 |
| 94dede14 | 47 | 2.379 | 19 | 0 | 0 | 19 | 19 | 0 |
| f696fa0d | 98 | 6.792 | 38 | 22 | 21 | 37 | 31 | 6 |

这是四个“一次只编辑一辆”的情景，不是同时编辑四辆。63次释放中57次没有已存表面接续，6次接续其他Actor，接续背景为0。第三个对象编辑后没有与原有束子集形成首交点，也没有被排除。此类现象反映查询采样与既有几何的关系，不能解读为真实传感器的正确/错误返回。

## 图与复现

`docs/autoresearch/worldsim_v73/m4/V73_TRAJECTORY_EDIT.png/pdf`展示全部四辆车的第一个登记heldout扫描；灰色浅点是参考时刻表面中心，彩色叉是实际逐束时刻的首交点，虚线为只读尺寸框。鸟瞰图裁剪仅影响展示，所有原始查询仍保留。图不以旧目标点云作为编辑真值。

源码`scripts/demonstrate_worldsim_v73_trajectory_edit.py`、`scripts/plot_worldsim_v73_trajectory_edit.py`。逐对象/帧计数在`docs/autoresearch/worldsim_v73/m4/trajectory_edit_r1_summary.json`，逐束前后depth/owner及预览数据保存在run。没有新增哈希、回归门控或神经模型重复测试。

固定表面+轨迹的组合通路已执行；F04中的背景未知暴露和物理误差仍然存在。后续可直接载入完成的主视觉模型表面复用该入口，不能将目前r9演示挪作其方法优势。整个V7.3继续训练和独立确认准备。
