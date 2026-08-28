# V6.7 P2 Actor package result

Canonical run:
`run://worldsim_v67/WS-V67-P2-ACTOR-PACKAGE-01/20260828T104733Z__actor-package-s0-r1`.

- units / unique Actors / Actor states: `70 / 107 / 517`;
- owned primitives: `1,093,082`;
- package files / bytes: `8 / 14,808,617`;
- Actor state retention / metadata completeness: `1 / 1`;
- Actor removed / hidden-target fields: `0 / 0`;
- runtime model loading / hazard-conditioned existence: `false / false`;
- wall / RSS / GPU: `13.40932s / 0.81198GiB / false`.

All six package gates pass. Two of the 72 evidence units contain no eligible Actor row, so the package has 70 units while
preserving all 517 P1 Actor states. This is package capability only, not physical repair.
