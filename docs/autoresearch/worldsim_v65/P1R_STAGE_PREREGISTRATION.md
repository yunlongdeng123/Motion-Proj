# P1R Task-Aligned Monotone Risk Preregistration

冻结语义：`score = frozen r_phys + relevance_continuous × softplus(r_task)`。

- `r_task` 只能增加近轨迹风险，不能把 q0 的物理风险降掉；
- continuous relevance 来自 `exp(-distance/3m)`，不输入 hard corridor bit；
- 训练仍使用 P1 Tier-L split/cache/seed/capacity；
- task weighting=`0.05 + 0.95×relevance`；
- primary：matched 40% pooled fixed-route opportunity risk relative reduction `>=5%`；
- support：scene lower>=higher、non-route emission risk relative increase `<=5%`、真实 query 优于 unit 内 shuffle；
- global AUROC 只作描述，不作为 r_task gate；
- 不读取 V6.5 formal selection，不做 seed/capacity/threshold sweep。
