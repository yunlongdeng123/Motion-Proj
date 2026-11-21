# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

`VEHICLE-COMPLETION-PRIOR-20261122`：[结果与组件](autoresearch/vehicle_completion_prior_20261122/report.md)、[固定域35条指标](autoresearch/vehicle_completion_prior_20261122/student/delivery/fixed_detail_metrics.json)、[独立保护核查](autoresearch/vehicle_completion_prior_20261122/student/independent_qa.json)。当前 a20 整车先验为 `diagnostic_rejected_geometry`，不可晋升。BUILD 仅 raw86/110，1033 实测点经2cm体素选1030真实代表；134/179/198仅实拍诊断，未读取其LiDAR。纯点Omni候选产生扫描线栅架后拒绝；单张 amodal 图像＋已知尺寸 bbox 候选完成一次固定网格材质与25,000表面高斯初始化。车辆200＋联合200步仅为先验初始化和可见RGB拟合，不构成完整教师几何／轮廓／新视角蒸馏。总物理更新409=400正式＋9 smoke，9内已含2 replay；original/corrected账本成对保存。

固定旧评分域中 raw110 的 RGB MAE／SSIM／HF1 对应度相对上一轮75行交接终态由0.02017／0.93247／0.60749变为0.03088／0.87547／0.36875；raw86 a10/native11有2948/3029像素变化>1/255，真实RGB MAE由0.10102升至0.14937。梯度隔离、参数冻结及五帧八输出 fresh 重载 maxdiff=0 为工程证据，不能替代灰模质量。新版门先验收部件关系、轮位、可见轮廓、测距贴合，优先锁定真实可见几何/RGB，仅在缺失区生成；当前公共v2入口已在Torch/CUDA前拒绝该候选。历史只读复核使用冻结actual-v1与原始entry/compact binding。Amodal3R规范输入CPU准备与Amodal3R／MM-TRELLIS／RECAST接口核查完成，三者尚未运行；不追加GPU实验，GPU保持开机。旧车辆7阶段与道路55项/8批次保留；human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。
