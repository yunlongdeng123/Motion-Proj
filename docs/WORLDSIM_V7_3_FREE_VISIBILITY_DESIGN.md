# V7.3 自由空间可见性代理

## V7.3 全轨迹训练结束与米制射线管free比较（2026-09-08）

LiDAR-only全轨迹标签r7 PID23188已完成30epoch/11130更新，正在完整489Actor最终曲面评价；训练结束记录elapsed3105.48s，最终wall time和开发结果待summary完成。不把最后一次loss或中间checkpoint当成最终性能。主joint r5 PID18843继续正常训练。

现在进行已实现的米制射线管free目标比较：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8`。相对r7，仅把free-mode由range改为beam_tube_range，仍为米制深度侵入、权重0.5；固定宽度0.03m、32×32栅格，通过同一显式三角面的最近深度及轮廓反传。输入、全轨迹fit标签、371fit对象、67dev可输入对象、51空输入对象、seed7304、AdamW lr1e-5、30epoch/11130更新及query架构均保持一致。复用完全相同的r6 initial与r5固定PCA评价，不重做相同模型读出。

本比较检验“原硬相交free缺少轮廓位移梯度时，保留米制严重程度的有限宽度几何目标能否改善真实侵入”；不混入event、额外opacity、自由半径或结构改动。原25Actor的beam_tube仅覆盖比例目标曾降低early数量却恶化侵入距离；此次range版本已在解析场景验证位置和轮廓梯度，但尚没有真实训练收益证据。它仍是优化代理：有限支持外可无梯度，仍需coverage吸引；宽度不是已校准的真实激光光束，不能将解析梯度结果写成全局收敛保证。官方nvdiffrast来源及已有解析证据见 `WORLDSIM_V7_3_FREE_VISIBILITY_DESIGN.md`。

r8使用现有motionproj训练环境与已编译nvdiffrast0.3.3，CUDA12.1编译器来自保留的v72-pointr环境；不升级Torch、不重建大环境。LiDAR-only作业不加载DPT，启动后以真实峰值记录资源；与r5和r7末尾评价并发的wall time不作单作业速度比较。r7评价结果一旦完成即用新增--reference配对r6，并列full cohort、原build LiDAR-ready、运动日志分层。

背景r2已完成，数据/评价进程均退出，scene结果和图已push3cb18bba。failure_ledger_delta=update V73-F02的单因素比较登记；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。CAPA、AdaPoinTr、主joint最终结果、event与新日志确认未完成，shutdown=false。


---

## 解析严重度实验结果（2026-09-08）

米制射线管解析r2完成：run `WS-V73-M3-FREE-VISIBILITY-01/20260907T191000Z__analytic-severity-r2`，codee1d328f8，1.031s，峰值0.0002GiB。5m平面片的几何coverage同为0.839767时，首回波20m得到tube intrusion12.428555m，首回波6m得到0.671814m，分别符合coverage×14.8m和coverage×0.8m；横向平移梯度分别−86.7176和−4.68744，轴向梯度均−0.839767。原硬中心束只有轴向梯度约−1。重复相同表面结果相同；首回波前无冲突及管外时均为0。这说明同一几何代理在该解析配置中保留侵入严重度和轮廓梯度，不证明真实数据提升。

证据=`docs/autoresearch/worldsim_v73/m3/free_severity_analytic_r2.json`；设计更新=`docs/WORLDSIM_V7_3_FREE_VISIBILITY_DESIGN.md`。无支持区域仍无此项梯度，必须依靠几何coverage/native数据通路恢复支持，不能把它写成已解决F03或首事件似然。3cm有限宽度、离散像素和局部AA仍有代理偏差，未来只在固定架构/标签下单独比较。

完整队列r5 PID18843正常推进：code8831def5，已完成489对象的LiDAR PCA初始基线（`lidar_baseline.json`已写出），正在原生联合模型的训练前评价；共享744冻结前缀，观测RSS33.4GiB。run `20260907T190000Z__population-shared-native-extra-time-s7304-r5`，日志 `/root/autodl-tmp/controller_logs/v73_population_joint_r5.log`。它仍是已登记的range free、371fit Actor、30epochs/11130更新；本轮新米制射线管代码不会修改它已加载的程序。等待其实际训练/summary，正常作业不重复启动、不提前终止。完整队列训练后再用同信息pointwise/LiDAR-only与原生融合比较；场景组合和新日志确认仍未完成。

failure_ledger_delta=update V73-F02（解析代理证据，不是实测缓解）；F01/F03/F04/F05仍active，F06仅当前直接数据配置缓解，下一编号V73-F07。没有资源不足或不可抗力，shutdown=false，整个研究继续。

---

# V7.3 自由空间可见性代理

## 米制严重度候选（2026-09-08）

完整队列r5已启动：code8831def5，PID18843，run `WS-V73-M2-GLOBAL-ACTOR-01/20260907T190000Z__population-shared-native-extra-time-s7304-r5`，日志 `/root/autodl-tmp/controller_logs/v73_population_joint_r5.log`。载入完成，371fit/67dev可输入Actor、51无输入对象均在队列，744共享冻结视图前缀，观测RSS约32.9GiB；正在真实训练前表面评价。保持已登记的range free/额外fit时刻标签/30epoch配置不变。

R1–R4开发权衡图已整理为 `docs/autoresearch/worldsim_v73/m2/global/mechanism_tradeoffs.png` 与 `.pdf`：六个核心读数同时列出，明确5日志范围、R1支持塌缩和R4增加训练标签预算。图仅汇总已有数值，未重跑评价；原始表格与日志配对区间继续保留。当前不能把LiDAR PCA的低侵入与高缺失分开选择性比较。

训练运行期间准备一个针对已定位F02的独立解析机制实验：`WS-V73-M3-FREE-VISIBILITY-01/20260907T191000Z__analytic-severity-r2`。r3几何覆盖代理不区分浅/深侵入，已有真实结果表现为侵入次数下降而总距离增加。先查[nvdiffrast官方插值与antialias接口](https://nvlabs.github.io/nvdiffrast/)：原始raster z/w不传播位置梯度，须显式插值米制顶点距离；antialias为轮廓提供局部位置梯度。因此候选将每个像素的二值覆盖乘以真实首回波前的侵入米数，再做同一个固定Gaussian footprint积分，即 mean_ray sum_pixel w(pixel) * 1[几何首面存在] * max(d_observed - 0.2 - d_surface, 0)。权重分母固定，不随点/面数量变化；像素距离来自同一显式首表面，无opacity/existence。

新增可选 `beam_tube_range`（单位m），原 `beam_tube` 保持覆盖比例、`range` 保持字面中心束距离，r5不切换。解析实验用同一个5m平面片比较首回波6m与20m，检验损失是否按侵入严重度缩放，以及横向/轴向位置梯度是否同时存在；同时记录无free冲突、管外与重复表面情形。它只回答具体梯度语义，不预设真实数据收益，也不解决射线管完全缺支持、裁剪/可见性切换或全局正确性。固定3cm仍为优化宽度，不是标定激光束；偏移像素不作为新增传感器真值。是否进入全队列训练，待r5结果后决定。

仅此一次解析实验与正常r5不冲突；不重复smoke/回归，不新增hash/指纹。failure_ledger_delta=update V73-F02 candidate，F03缺支持event与F04场景拼接仍未实现验证，F05新日志确认待办，F06直接深度约束配置缓解，下一编号V73-F07。资源充足，shutdown=false，持续研究。

---

# V7.3 射线管可见性梯度实验

`WS-V73-M3-FREE-VISIBILITY-01` 一次解析机制实验已完成（code c6129bcc，run `20260907T174500Z__analytic-r1`）。5m处、横向偏移2.5cm的12cm曲面片遮挡20m返回：硬range侵入14.8m、整体平移梯度[0,0,-1]；有限射线管覆盖0.83977、平移梯度[-5.8593,0,0]。负梯度更新给出退出横向射线管的方向。位于原始首回波后或完全在管外时均为0；重复同一表面覆盖和梯度均不变。证据=`docs/autoresearch/worldsim_v73/m3/free_visibility_analytic_r1.json`。这只说明代理能提供不同局部几何方向，不保证真实数据收益、任意可见性切换或缺失支持出生。

下一受控训练r3：task `WS-V73-M2-GLOBAL-ACTOR-01`，run `20260907T175000Z__surround-native-beam-tube-s7304-r3`。从同一M1 r3初始权重/seed7304、20fit/5dev Actor、30epochs/600更新，保留native-data weight1、coverage及弱envelope，仅把free训练目标从硬range侵入换为同一三角曲面的有限射线管覆盖。free weight0.5、宽度0.03m、32×32分辨率、每步原始512束不变。覆盖比例与米制range是不同单位，不声称权重相同等于梯度强度相同；另外持续记录原始硬free米值，最终统一硬表面评价。

将原始返回−0.2m设为free终点，nvdiffrast栅格化几何占据、antialias提供轮廓位置梯度；相机/轨迹/表面半径/opacity均无新可训练控制。固定3σ范围和高斯足迹是训练代理，不能把横向偏移射线称为新增观测真值。边界裁剪、正好对称或完全离开有限支持仍可能无梯度，宽度在掠射表面上的偏差需结合hard hit/early/miss判断。保留真实depth与coverage的支持恢复路径，不以event替代。

运行依赖：motionproj保留Torch2.4.1+cu121，新增nvdiffrast0.3.3、ninja1.13.2；仅复用`/root/autodl-tmp/envs/worldsim-v72-pointr`的CUDA12.1编译器，库仍在motionproj运行。源码`/root/autodl-tmp/external/worldsim_v73/nvdiffrast` tag v0.3.3，扩展缓存`/root/autodl-tmp/torch_extensions/worldsim_v73`；不升级Torch，不新建环境，不添加校验工件。来源：[官方API/SIGGRAPH Asia 2020](https://nvlabs.github.io/nvdiffrast/)、[兼容版本源码](https://github.com/NVlabs/nvdiffrast/tree/v0.3.3)。

r2已结束，r3提交后启动；当前无资源不足、shutdown=false。failure_ledger_delta=update V73-F02（已得到横向代理梯度，实际收益待测），F01:F05仍active，F06在直接测量路径范围缓解；下一编号V73-F07。完整同信息控制、更多动态独立日志、event和背景组合仍pending，不把这一次解析结果当成V7.3完成。
