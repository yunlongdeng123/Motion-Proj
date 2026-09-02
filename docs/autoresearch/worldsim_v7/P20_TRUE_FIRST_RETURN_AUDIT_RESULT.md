# P20 True First-Return Audit Result

Date: 2026-09-02

Canonical: `run://worldsim_v7/WS-V7-P20-TRUE-FIRST-RETURN-AUDIT-01/20260903T124500Z__true-first-return-audit-r1`

The metric correction is supported and materially changes the measured safety exposure. Under the old target-nearest proxy,
always-COMPLETE total/hazard new-early rates were `.9662%/1.4362%`; literal first return measures `5.9561%/8.7421%`, roughly
six times larger. This is not a threshold effect: query and compiled surfaces use the same frozen lateral/depth tolerances, and only
the point-selection operator changes from target-nearest to minimum positive ray depth.

All three frozen deletion policies reduce true first-return exposure but fail the unchanged Chamfer gate:

| policy | coverage | hazard early events / rate | mean Chamfer | new hits |
|---|---:|---:|---:|---:|
| always-COMPLETE | 100.00% | 14,219 / 8.7421% | .1945869 m | 32,575 |
| P17 | 71.43% | 12,649 / 7.7769% | .1994111 m | 29,191 |
| P17R | 88.96% | 13,386 / 8.2300% | .1957160 m | 31,620 |
| P19 | 98.95% | 14,088 / 8.6616% | .1946787 m | 32,504 |

Thus P17/P17R/P19 each pass the true-ray hazard direction and fail population Chamfer. P19 is the most efficient intervention: 131
hazard early events removed for 71 new hits lost and `.0000919 m` mean Chamfer penalty. The audit took `257.19 s`, peak GPU
`.1839 GiB`, peak RSS `1.177 GiB`, and read no fresh AV2 payload.

Register `V7-F29` for the all-policy Pareto failure, but retain the first-return correction as positive hard-evidence infrastructure.
P21 formalizes the set-inclusion safety theorem and the empirical safety--surface frontier without changing any policy.
