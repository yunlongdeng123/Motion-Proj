# WorldSim V6.2 P2 query probe

- Task：`WS-V62-P2-EVIDENCE-QUERY-DATASET-01`
- Substage：`query_probe`
- 状态：`done_resource_and_query_denominator_passed`
- Canonical probe：`run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T082318Z__query-probe-s20260824-r2`

## 实现边界

V6.2 新建轻量 evidence builder，不调用 V6.1 的哈希/manifest runner。它复用坐标和 oriented-box 基元，构建：

- ray-before-hit FREE；
- static 与 motion-compensated actor OCC hit；
- behind-hit UNKNOWN；
- 同位置 FREE/OCC contradiction；
- 独立 actor envelope/identity 层。

每 target 分别构建 visible method、counterfactual dropout、independent training target 三个 evidence field，再按固定 quota
采样 hard FREE/OCC、behind-hit UNKNOWN、boundary、actor envelope 和 contradiction queries。P4 只在相同 query 坐标附加
frozen IR-WM prior，不重新选择 P2 scene/target。

## r2 结果

scene-0071/f017：

```text
queries                       100,000
source-role overlap                 0
disk bytes                    2,036,102
wall seconds                       2.96
target-supervised queries       38,088
actor-bound queries             36,786
```

候选池与 quota：

| type | pool | quota |
|---|---:|---:|
| hard FREE | 168,487 | 25,000 |
| hard OCC | 11,936 | 15,000 |
| behind-hit UNKNOWN | 13,282 | 25,000 |
| boundary | 382,175 | 15,000 |
| actor envelope | 150,158 | 15,000 |
| contradiction | 5,849 | 5,000 |

pool 小于 quota 时只在当前 training unit 内有放回采样，并在 manifest 报告 unique pool denominator。

## V62-F02

r1 probe 使用模糊的 `*_state` 字段。V6.1 evidence 编码为 `UNKNOWN/FREE/OCCUPIED=0/1/2`，P3 model head 为
`FREE/OCCUPIED/UNKNOWN=0/1/2`，若训练 loader 直接消费会静默换标签。r1 没有进入训练或 formal dataset。

r2 明确写入两组数组：

```text
*_evidence_state   U/F/O = 0/1/2   只作 hard-evidence feature
*_class_index      F/O/U = 0/1/2   送入 three-state loss
```

`V62-F02=resolved`。下一步不再增加 smoke，从 clean commit 执行72-unit formal materialization。
