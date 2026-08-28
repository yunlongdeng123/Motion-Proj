# WorldSim V6.7 ray-terminated Actor surface protocol

- 日期：2026-08-28
- 分支：`research/worldsim-v6.7-anisotropic-surface`
- 基线：V6.6 terminal commit `c05ca27`

V6.7不复开V6.6的radius recovery。新问题是：source LiDAR已经提供ray termination语义时，能否用方向性可见性证据
区分合法Actor surface support与future observed-FREE conflict。

## Frozen method

对P7-style fixed-budget acted Actor boundary：

```text
KEEP = exact same-Actor motion-compensated hit
    OR (nearest same-Actor hit within 0.512m AND source behind_hit)
otherwise UNKNOWN
```

`0.512m`是native representation scale，不扫radius。新增变量只有已有`METHOD_EVIDENCE.behind_hit`，它来自sensor ray
终点之后的方向性区域；target只作最终评估。Actor shell/ID/track/trajectory/hazard保持。

## Data roles

P1使用V65 P3C六场景`0030/0055/0453/0501/1046/1085`。它们在V65用于calibration，但从未用于V6.6/V6.7
surface method或P3L训练/selection/confirmation，因此记为task-untouched legacy selection，不称fresh V6.7。

顺序：P1 frozen head transfer → P2 Actor package → P3 fixed-budget actions → P4 single ray-terminated physical repair。
P4沿用V6.6九门，不扫threshold/budget/rule。失败则登记`V67-F01`并先检索新方案。

## P4 result and single recovery

P4 canonical=`20260828T105253Z__ray-surface-s0-r1`。conflict reduction=`0.678963`通过，但overall/clean retention=
`0.392368/0.396519 < 0.40`，因此按原门拒绝并登记`V67-F01`。检索后唯一P4R改为在target frame从nearest
motion-compensated same-Actor hit构造inward ray half-ball；半径、actions、gates与claim boundary不变。P4R失败则关闭
当前physical repair family，不再调参。

## Claim boundary

成功也只支持task-untouched legacy ray-terminated physical surface capability；在独立新cohort confirmation之前，不称
fresh generalization、RL-ready world、planner、policy、closed-loop RL或safety。
