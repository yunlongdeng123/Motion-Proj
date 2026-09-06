# WorldSim V7.1 M34 — Producer-evidential Anchor Authority

日期：2026-09-05  
状态：frozen

## 单一问题

M34只回答：M33保留下来的 source/provenance/build-ray 证据，是否足以在冻结M8 geometry下辨识并连续
降权具体矛盾 anchor。completion、trajectory、dynamic/static、image appearance均不参与这一问题。

## 表示与监督

- 冻结M8全部 anchor/child center、scale和Actor trajectory；anchor scale固定0.08 m；
- 模型只读取M33 `input_*`：局部位置、KEEP/PROJECT、source ray/range、projection displacement、canonical
  hit/temporal/view支持、build F/O/U及ray opportunity；不读取hazard、held-out或image；
- 只为anchor预测连续FREE/OCCUPIED/UNKNOWN mass；completion child authority恒为1；
- `supervision_*` held-out native LiDAR soft mass与32-bin first-return共同训练；occupied mass连续进入Gaussian
  energy，不阈值、不删除、不移动或缩放primitive；
- 固定M32同容量/epochs/loss权重，只运行seed 71134一次，不sweep。

## 冻结开发判定

在与M32相同的66-Actor exposed holdout相对unit-authority M21 energy：

1. all early不增加；
2. hazard与clear early均不增加；
3. all hit下降不超过1 pp；
4. predicted-vs-GT anchor occupied相关系数至少0.25，以排除全局常数降权伪改善。

任一失败则M34不进入external且停止本family调参。M34不读取AV2 M21 partial，不读取selection/source-final，
不做surface filtering；下一failure ID=`V71-F36`。
