# WorldSim V6.4 Fresh Evidence Closeout

- Task: `WS-V64-P2E-FRESH-EVIDENCE-01`
- Hypothesis: `WS-V64-H-P4-001`
- Status: `done / surface unlocked`
- Canonical: `run://worldsim_v64/WS-V64-P2E-FRESH-EVIDENCE-01/20260826T084000Z__fresh-evidence-s0-r1`

固定6-scene、72-target denominator全部完成：`unit_count=72`、`scene_count=6`、`source_role_overlap_count=0`、
`passed=true`。输出`68,444,954 bytes`，wall=`118.2903 s`，最大单unit wall=`4.2750 s`；CPU双worker，GPU未使用。

每unit保存method、dropout与target evidence grid；未被surface或UQ消费的100k query sampling按freeze关闭，
`query_count=0`、candidate quota gate为空。target evidence已物化，因此fresh fit/evaluation target quality从此为已读；
尚未读取U0/U2 score或任何gate verdict。calibration、confirmation与exact-once test仍未读。

formal run failure delta=`none`。首次launcher被本地PowerShell提前展开`$()`，未创建run目录，登记`V64-F07`；
删除subexpression后同一冻结输入以唯一r1正常执行。下一步只运行固定surface r1。
