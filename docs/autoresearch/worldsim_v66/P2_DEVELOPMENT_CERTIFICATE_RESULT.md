# P2-D Deterministic Actor Certificate 结果

Task：`WS-V66-P2-FACTOR-CERTIFICATE-DEV-01`

Canonical：`run://worldsim_v66/WS-V66-P2-FACTOR-CERTIFICATE-DEV-01/20260828T085346Z__factor-certificate-dev-s0-r1`

Verdict：`supported_development_factor_certificate`

P2不读取P1预写certificate decision，只从sensor/provenance、duplicate、lifecycle、kinematic/identity与shape factor
重新生成reason codes和compiler action。artifact family/label与hazard属性只在证书完成后用于评测。

| 指标 | 结果 |
|---|---:|
| rows / base actor-unit | 8,180 / 409 |
| pooled recall / AUROC / AUPRC | 1 / 1 / 1 |
| minimum family recall | 1 |
| clean-hazard / benign false artifact | 0 / 0 |
| legitimate hazard / benign retention | 1 / 1 |
| Actor existence / ID / lifecycle retention | 1 / 1 / 1 |
| hazard-pair score delta | 0 |
| hard observed evidence violations | 0 |

八个gates全部通过。actions为KEEP 4,090、ABSTAIN_LOCAL_GEOMETRY 818、DROP_ARTIFACT_PRIMITIVE 818、
REPAIR 2,454。wall=`0.2487s`，peak RSS=`0.5316GiB`。

该结果仍是deterministic injected Tier-L development capability。P3 relative gate在该分母上没有数值headroom，故不训练
learned model；下一步直接比较DROP/ABSTAIN/REPAIR是否保持hazard event。自然artifact/fresh泛化仍未验证。
