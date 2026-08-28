# P4-D Repair-first Compiler 结果

Task：`WS-V66-P4-ARTIFACT-REPAIR-DEV-01`

Canonical：`run://worldsim_v66/WS-V66-P4-ARTIFACT-REPAIR-DEV-01/20260828T085755Z__repair-first-dev-s0-r1`

Verdict：`supported_development_repair_first_compiler`

三臂共享同一8,180-row输入；R2只从paired clean reference恢复observable factors，不改变Actor ID、track、trajectory
或hazard attributes。

| 指标 | R0 DROP | R1 ABSTAIN | R2 REPAIR |
|---|---:|---:|---:|
| artifact violation reduction | 1.00 | 0.00 | 1.00 |
| clean-hazard Actor retention | 1.00 | 1.00 | 1.00 |
| all-hazard event retention | 0.50 | 1.00 | 1.00 |
| hazard event shift | 0.50 | 0.00 | 0.00 |
| retained ID/track/trajectory exact | 1.00 | 1.00 | 1.00 |
| nonartifact regression | 0 | 0 | 0 |

R2六个gates全部通过。R0虽然消除注入违规，却删除了artifact+hazard象限，导致一半hazard events消失；R1保留
hazard但没有修复factor violation；R2是唯一同时消除违规和保持hazard分布的arm。

wall=`0.5464s`，peak RSS=`0.5739GiB`。结果只支持deterministic observable-factor repair capability，不支持RGB、
完整SceneIR或natural reconstruction artifact repair。下一步在独立legacy cohort检验natural actor-owned observed-FREE
boundary conflict，避免把注入满分误写成泛化。
