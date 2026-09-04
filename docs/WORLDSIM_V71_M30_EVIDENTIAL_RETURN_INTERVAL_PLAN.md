# WorldSim V7.1 M30：Evidential First-Return Interval

## 动机

M29表明，使用同一点集同时优化GT coverage与free-before-hit会在尾部继续冲突。M30不再
强迫单一确定表面回答两个问题，而是保留两个由监督来源定义的表面集：

- `S_known`：由build LiDAR endpoint直接支持的immutable anchors；
- `S_possible`：`S_known` 加上由GT completion losses训练的全部M8 children。

UNKNOWN不在部署时被硬删成FREE，也不被单值化为OCCUPIED。

## 一手依据

- EvOcc（CVPR 2025）直接以evidential belief masses监督模型，将unobserved/contradictory uncertainty
  保留为独立假设，而非普通confidence：
  <https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf>
- Accurate Training Data for Occupancy Map Prediction（CVPR 2024）从LiDAR reflection/transmission构造
  occupied/free/uncertain evidence，并用ray-rendered depth回到原始测量评估：
  <https://openaccess.thecvf.com/content/CVPR2024/papers/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_CVPR_2024_paper.pdf>

M30不复现voxel BBA或其decision rule，只迁移“不确定性是表示本身，不是后处理删除器”。

## 集序区间

对冻结beam-tube literal first-return operator `d_S(r)`，有

\[
S_{\rm known}\subseteq S_{\rm true}\subseteq S_{\rm possible}
\quad\Longrightarrow\quad
d_{\rm possible}(r)\le d_{\rm true}(r)\le d_{\rm known}(r).
\]

推导只用到first-return对set deletion的单调性。`d_possible`是最早可能返回，`d_known`是最早
已知支持返回；二者之间是epistemic interval，不被折叠成单一depth。

必须明确：`S_true subset S_possible`只是completion-hypothesis coverage assumption，并非已证明事实。
所以M30同时报告GT target在区间内的empirical bracketing rate、上界无穷的比例、有限区间宽度，
但不把它写成coverage或safety guarantee。

## 冻结执行

- model=M8 r2；M29 rejected checkpoint不使用；
- data=与M8相同的66 pretrained-exposed source holdout Actors；
- representation=raw immutable anchors vs raw anchors+all M8 children，保持严格set inclusion；
- metric=same literal beam tube `0.20m` lateral/depth tolerance；
- report=ordering violations, target bracketing, finite/unbounded interval, width median/q90, early/hit；
- strata=all/hazard/clear/moving/quasi-static；
- GPU=只运行literal ray queries；无训练、checkpoint、阈值扫描或第二个下载器；
- M21=只读进度，不读partial quality。

M30的structural success只是0 set-order violation。interval是否足够有限/紧致为descriptive utility boundary，
不用新阈值给它包装成成功。
