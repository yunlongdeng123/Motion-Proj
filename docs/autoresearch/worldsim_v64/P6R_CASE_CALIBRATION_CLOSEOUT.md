# P6R independent case-calibration closeout

Date: 2026-08-26

Canonical run:

`run://worldsim_v64/WS-V64-P6R-CALIBRATION-01/20260826T141500Z__case-calibration-s0-r1`

The frozen full-native selective MLP was scored once on 96 cases from eight independent scenes. The new confirmation cohort remained unread.

| nominal coverage | mean realized | failures / 96 | empirical risk | simultaneous upper bound |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.049961 | 0 | 0 | 0.048647 |
| 0.10 | 0.099963 | 0 | 0 | 0.048647 |
| 0.20 | 0.199969 | 0 | 0 | 0.048647 |
| 0.30 | 0.299958 | 0 | 0 | 0.048647 |
| 0.40 | 0.399967 | 0 | 0 | 0.048647 |
| 0.50 | 0.499979 | 3 | 0.03125 | 0.103218 |

At every passing coverage each of construction, night, rain, and vulnerable-transit had zero failures over 24 cases. All three 50% failures were rain cases. Under the frozen largest-passing rule, the selected nominal coverage is 0.40.

Verdict: `supported_selective_policy`. This is independent finite-sample calibration evidence, not confirmation or a real-world safety claim. GPU scoring wall time was 13.0083 seconds and peak RSS was 0.7943 GiB. The exact-once new confirmation cohort is now unlocked without model refit or policy change.
