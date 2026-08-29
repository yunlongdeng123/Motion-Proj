# WorldSim V6.7 ArXiv Evidence Index

- 工作标题：**From End-to-End Reliability Shortcuts to Actor-Uncertainty Boundary Queries**
- 分支：`research/worldsim-v6.7-anisotropic-surface`
- 状态：`P183 fresh marginals + P201 fresh joint-horizon confirmation + P205 prospective beta secondary supported`
- 当前主结论：Actor uncertainty × trajectory boundary的continuous cost density在不同10-scene/10-log五时域确认中同时改善proper score与marginal reliability

本文件是V6.7技术报告的证据导航。逐实验数值以`docs/EXPERIMENTS.md`为准，失败与防重复规则以
`docs/RESEARCH_FAILURES.md`为准，实时状态以`docs/RESEARCH_STATUS.md`为准，论文叙事见
`V67_ARXIV_TECHNICAL_REPORT.md`。

## 1. Canonical run index

| 阶段 | canonical run | 核心证据 |
|---|---|---|
| P81 | `run://worldsim_v67/WS-V67-P81-FRESH-TEST-ACTOR-EVENT-01/20260829T211500Z__fresh-test-actor-event-s0-r1` | all-row query/Actor/P73=`26/57/45`；10/10 scenes |
| P95 | `run://worldsim_v67/WS-V67-P95-TRAJECTORY-OCCUPANCY-FLIP-01/20260830T002500Z__trajectory-occupancy-flip-s0-r1` | consumed development `7/28/13` |
| P96 | `run://worldsim_v67/WS-V67-P96-OCCUPANCY-FLIP-CONFIRMATION-01/20260830T004000Z__occupancy-flip-confirmation-s0-r1` | independent `8/5/12`；relative reject |
| P102 | `run://worldsim_v67/WS-V67-P102-HIERARCHICAL-TEMPORAL-INTERACTION-01/20260830T022000Z__hierarchical-temporal-interaction-s0-r1` | development best `4/27/13` |
| P103 | `run://worldsim_v67/WS-V67-P103-HIERARCHICAL-CONFIRMATION-01/20260830T024000Z__hierarchical-confirmation-s0-r1` | same independent read `9/7/12`；relative reject |
| P107 | `run://worldsim_v67/WS-V67-P107-ACTOR-UNCERTAINTY-TUBE-01/20260830T061000Z__actor-uncertainty-tube-s0-r2` | P81/P96=`2/36/13`与`2/9/12` |
| P108 | `run://worldsim_v67/WS-V67-P108-UNCERTAINTY-TUBE-CONFIRMATION-01/20260830T063500Z__uncertainty-tube-confirmation-s0-r1` | independent P107/Actor/P75=`5/35/20` |
| P109 | `run://worldsim_v67/WS-V67-P109-DIRECTIONAL-ACTOR-UNCERTAINTY-01/20260830T062500Z__directional-actor-uncertainty-s0-r1` | directional development P81/P96均0 events |
| P110 | `run://worldsim_v67/WS-V67-P110-DIRECTIONAL-CONFIRMATION-01/20260830T064000Z__directional-confirmation-s0-r1` | same-read directional=`1`，AUROC=.96027 |
| P111 | `run://worldsim_v67/WS-V67-P111-CLEARANCE-CONFIRMATION-BASELINE-01/20260830T064500Z__clearance-confirmation-baseline-s0-r1` | same-read clearance=`1`，AUROC=.91644 |
| P112 | `run://worldsim_v67/WS-V67-P112-NONLINEAR-GAUSSIAN-CROSSING-01/20260830T065000Z__nonlinear-gaussian-crossing-s0-r1` | P96 nonlinear 3/.85852 vs linear 0/.90434；reject |
| P114 | `run://worldsim_v67/WS-V67-P114-MONOTONE-TAIL-RISK-01/20260830T071000Z__monotone-tail-risk-s0-r1` | P81/P96 AUROC gain=`-.01626/-.00137`；reject |
| P115 | `run://worldsim_v67/WS-V67-P115-SPECTRAL-ACTOR-UNCERTAINTY-01/20260830T071500Z__spectral-actor-uncertainty-s0-r1` | P81/P96 AUROC gain=`+.00945/-.05722`；reject |
| P116 | `run://worldsim_v67/WS-V67-P116-DIRECTIONAL-QUANTILE-FIELD-01/20260830T072000Z__directional-quantile-field-s0-r1` | P81/P96 AUROC gain=`-.00380/-.01503`；reject |
| P117 | `run://worldsim_v67/WS-V67-P117-FULL-COVARIANCE-ACTOR-UNCERTAINTY-01/20260830T072500Z__full-covariance-actor-uncertainty-s0-r1` | P81/P96 AUROC gain=`+.00490/+.00932`；development support |
| P118 | `run://worldsim_v67/WS-V67-P118-CORRELATION-ABLATION-01/20260830T073000Z__correlation-ablation-s0-r1` | conditional-vs-zero rho gain=`+.00030/-.00012`；reject mechanism |
| P113 | `run://worldsim_v67/WS-V67-P113-DIRECTIONAL-VS-CLEARANCE-CONFIRMATION-01/20260830T070500Z__directional-vs-clearance-s0-r1` | directional/clearance events=`6/5`；AUROC gain=`+.04486`；composite reject |
| P119 | `run://worldsim_v67/WS-V67-P119-RANKED-RANGE-TAIL-01/20260830T074500Z__ranked-range-tail-s0-r1` | P81/P96/P113 events=`0/0/6`；reject |
| P120 | `run://worldsim_v67/WS-V67-P120-CONTINUOUS-BOUNDARY-STATE-COST-01/20260830T075000Z__continuous-boundary-state-cost-s0-r1` | P109 continuous-cost Spearman=`.8065/.7183/.7921`；new head reject |
| P122 | `run://worldsim_v67/WS-V67-P122-FULL-COVARIANCE-CONTINUOUS-SELECTION-01/20260830T081000Z__full-covariance-continuous-selection-s0-r1` | mean Spearman gain=`+.00941`，但P96/P113 selected cost回退；reject |
| P123 | `run://worldsim_v67/WS-V67-P123-CONTINUOUS-RANK-RESIDUAL-01/20260830T081500Z__continuous-rank-residual-s0-r1` | dense continuous pairs；P81/P96 rank退化、P96 selected cost回退 |
| P124 | `run://worldsim_v67/WS-V67-P124-CORRELATED-STUDENT-T-UNCERTAINTY-01/20260830T082000Z__correlated-student-t-uncertainty-s0-r1` | fixed df4；P96 AUROC `-.05407`、events `7>0`；reject |
| P125 | `run://worldsim_v67/WS-V67-P125-TWO-MODE-ACTOR-UNCERTAINTY-01/20260830T082500Z__two-mode-actor-uncertainty-s0-r1` | K2 mixture未collapse但三cohort AUROC均退化；reject |
| P126 | `run://worldsim_v67/WS-V67-P126-ACTOR-DEEP-ENSEMBLE-01/20260830T083000Z__actor-deep-ensemble-s0-r1` | AUROC三组同增，P113 events 6→4；P96 0→1故binary reject |
| P127 | `run://worldsim_v67/WS-V67-P127-ENSEMBLE-CONTINUOUS-SELECTION-01/20260830T083500Z__ensemble-continuous-selection-s0-r1` | selected cost全降；Spearman gain=`+.0470/+.1351/+.0755`；support |
| P128 | `run://worldsim_v67/WS-V67-P128-ENSEMBLE-CONTINUOUS-CONFIRMATION-01/20260830T084000Z__ensemble-continuous-confirmation-s0-r1` | ensemble gain=`+.04721`、cost `.27051<.27796`；same-read support with timing caveat |
| P129 | `run://worldsim_v67/WS-V67-P129-ENSEMBLE-INDEPENDENT-CONFIRMATION-01/20260830T085000Z__ensemble-independent-confirmation-s0-r1` | gain=`+.04257`、cost `.30867<.32934`；scene-level independent support |
| P130 | `run://worldsim_v67/WS-V67-P130-ENSEMBLE-DISTRIBUTION-DISTILLATION-01/20260830T091000Z__ensemble-distribution-distillation-s0-r1` | rank retained；P113 cost regression，rejected/F93 |
| P131 | `run://worldsim_v67/WS-V67-P131-TASK-CONDITIONED-SCORE-DISTILLATION-01/20260830T091500Z__task-conditioned-score-distillation-s0-r1` | pointwise loss low但trajectory rank collapse；rejected/F94 |
| P132 | `run://worldsim_v67/WS-V67-P132-TRAJECTORY-RANK-DISTILLATION-01/20260830T092000Z__trajectory-rank-distillation-s0-r1` | rank恢复但仍低P126、cost全退；rejected/F95 |
| P133 | `run://worldsim_v67/WS-V67-P133-BATCHENSEMBLE-ACTOR-UNCERTAINTY-01/20260830T092500Z__batchensemble-actor-uncertainty-s0-r1` | diversity collapse、mean rank delta `-.01454`；rejected/F96 |
| P134 | `run://worldsim_v67/WS-V67-P134-PACKED-INDEPENDENT-ACTOR-ENSEMBLE-01/20260830T093000Z__packed-independent-actor-ensemble-s0-r1` | rank retained；P96 cost miss，rejected/F97 |
| P135 | `run://worldsim_v67/WS-V67-P135-FULL-BUDGET-PACKED-ACTOR-ENSEMBLE-01/20260830T093500Z__full-budget-packed-actor-ensemble-s0-r1` | rank retained但cost三组略退；rejected/F98 |
| P136 | `run://worldsim_v67/WS-V67-P136-SNAPSHOT-ACTOR-ENSEMBLE-01/20260830T094000Z__snapshot-actor-ensemble-s0-r1` | P96 mode diversity不足；rejected/F99 |
| P137 | `run://worldsim_v67/WS-V67-P137-SWAG-ACTOR-ENSEMBLE-01/20260830T094500Z__swag-actor-ensemble-s0-r1` | rank retained但P81/P96 cost微退；rejected/F100 |
| P138 | `run://worldsim_v67/WS-V67-P138-FULL-COVARIANCE-DEEP-ENSEMBLE-01/20260830T095000Z__full-covariance-deep-ensemble-s0-r1` | P81/P113 gain、P96反转；rejected/F101 |
| P139 | `run://worldsim_v67/WS-V67-P139-SCENE-BALANCED-DEEP-ENSEMBLE-01/20260830T095500Z__scene-balanced-deep-ensemble-s0-r1` | cost全退、mean rank `-.01275`；rejected/F102 |
| P140 | `run://worldsim_v67/WS-V67-P140-SCENE-BAGGED-DEEP-ENSEMBLE-01/20260830T100000Z__scene-bagged-deep-ensemble-s0-r1` | mean rank `-.00405`、P81/P96 cost退；rejected/F103 |
| P141 | `run://worldsim_v67/WS-V67-P141-FIVE-MEMBER-DEEP-ENSEMBLE-01/20260830T100500Z__five-member-deep-ensemble-s0-r1` | mean gain=`+.00056`、P96/P113 cost退；rejected/F104 |
| P142 | `run://worldsim_v67/WS-V67-P142-TASK-CONDITIONED-PROJECTED-ENSEMBLE-01/20260830T101000Z__task-conditioned-projected-ensemble-s0-r1` | P129 `+.01319`但P96退；rejected/F105 |
| P143 | `run://worldsim_v67/WS-V67-P143-CONDITIONAL-RESIDUAL-ENSEMBLE-01/20260830T101500Z__conditional-residual-ensemble-s0-r1` | cost全退、mean rank `-.01344`；rejected/F106 |
| P144 | `run://worldsim_v67/WS-V67-P144-TRAJECTORY-SET-RANK-COMPILER-01/20260830T102000Z__trajectory-set-rank-compiler-s0-r1` | P96反转、mean rank `-.00096`；rejected/F107 |
| P145 | `run://worldsim_v67/WS-V67-P145-ABSOLUTE-TIME-ACTOR-ENSEMBLE-01/20260830T102500Z__absolute-time-actor-ensemble-s0-r1` | 3/4 rank增但P96强退；rejected/F108 |
| P146 | `run://worldsim_v67/WS-V67-P146-MONOTONE-TIME-SCALE-ADAPTER-01/20260830T103000Z__monotone-time-scale-adapter-s0-r1` | mean rank `-.001607`、仅P129 cost改善；rejected/F109 |
| P147 | `run://worldsim_v67/WS-V67-P147-MULTI-HORIZON-INDEPENDENT-CONFIRMATION-01/20260830T104000Z__multi-horizon-independent-confirmation-s0-r1` | five-H mean gain=`+.17419`、cost delta=`-.01777`；2/2 independent support |
| P148 | `run://worldsim_v67/WS-V67-P148-FULL-SEQUENCE-ACTOR-ENSEMBLE-01/20260830T104500Z__full-sequence-actor-ensemble-s0-r1` | 四cohort rank全退、mean `-.012380`；rejected/F110 |
| P149 | `run://worldsim_v67/WS-V67-P149-COHERENT-TRAJECTORY-MIXTURE-01/20260830T105000Z__coherent-trajectory-mixture-s0-r1` | modes active但mean rank `-.098286`；rejected/F111 |
| P150 | `run://worldsim_v67/WS-V67-P150-DENSE-BOUNDARY-COST-ENSEMBLE-01/20260830T105500Z__dense-boundary-cost-ensemble-s0-r1` | P81/P129 rank增但P96反转；rejected/F112 |
| P151 | `run://worldsim_v67/WS-V67-P151-GROUP-DRO-BOUNDARY-COST-01/20260830T110000Z__group-dro-boundary-cost-s0-r1` | P96 `-.1154`、mean `-.0467`；rejected/F113 |
| P152 | `run://worldsim_v67/WS-V67-P152-RANDOMIZED-PRIOR-ACTOR-ENSEMBLE-01/20260830T110500Z__randomized-prior-actor-ensemble-s0-r1` | cost全退、mean rank `-.006852`；rejected/F114 |
| P153 | `run://worldsim_v67/WS-V67-P153-BAYESIAN-LAST-LAYER-ACTOR-01/20260830T111000Z__bayesian-last-layer-actor-s0-r1` | epistemic约`1e-4`、mean rank `+.000844`；rejected/F115 |
| P154 | `run://worldsim_v67/WS-V67-P154-DENSITY-AWARE-ACTOR-ENSEMBLE-01/20260830T111500Z__density-aware-actor-ensemble-s0-r1` | shift detected但P81 cost退；rejected/F116 |
| P155 | `run://worldsim_v67/WS-V67-P155-REGMIXUP-ACTOR-ENSEMBLE-01/20260830T112000Z__regmixup-actor-ensemble-s0-r1` | mean rank `-.007045`；rejected/F117 |
| P156 | `run://worldsim_v67/WS-V67-P156-INTEGRATED-INCREMENT-ACTOR-ENSEMBLE-01/20260830T112500Z__integrated-increment-actor-ensemble-s0-r1` | rank mean `-.028923`；rejected/F119 |
| P157 | `run://worldsim_v67/WS-V67-P157-HORIZON-SPECIALIST-ACTOR-ENSEMBLE-01/20260830T113500Z__horizon-specialist-actor-ensemble-s0-r1` | H3.5→H3.0 mean rank `-.59434`；rejected/F120 |
| P158 | `run://worldsim_v67/WS-V67-P158-CRPS-ACTOR-ENSEMBLE-01/20260830T114500Z__crps-actor-ensemble-s0-r1` | old mean rank `-.02371`、cost全退；rejected/F121 |
| P159 | `run://worldsim_v67/WS-V67-P159-JOINT-ENERGY-SCORE-ACTOR-ENSEMBLE-01/20260830T115000Z__joint-energy-score-actor-ensemble-s0-r1` | old mean rank `-.04251`、cost全退；rejected/F122 |
| P160 | `run://worldsim_v67/WS-V67-P160-EXACT-ENSEMBLE-MIXTURE-BOUNDARY-01/20260830T115500Z__exact-ensemble-mixture-boundary-s0-r1` | exact linear pool mean rank `-.04330`；rejected/F123 |
| P161 | `run://worldsim_v67/WS-V67-P161-EPISTEMIC-VARIANCE-ABLATION-01/20260830T120000Z__epistemic-variance-ablation-s0-r1` | full-vs-no-ep rank `-.00009`；epistemic attribution rejected/F124 |
| P162 | `run://worldsim_v67/WS-V67-P162-ORIENTED-FOOTPRINT-ACTOR-ENSEMBLE-01/20260830T121000Z__oriented-footprint-actor-ensemble-s0-r3` | mean rank `-.000284`、2/4 cost；rejected/F126 |
| P163 | `run://worldsim_v67/WS-V67-P163-DIRECT-FOOTPRINT-SUPPORT-ENSEMBLE-01/20260830T121500Z__direct-footprint-support-ensemble-s0-r1` | mean rank `-.001156`、P96 cost回退；rejected/F127 |
| P164 | `run://worldsim_v67/WS-V67-P164-INTERACTION-CONTEXT-ACTOR-ENSEMBLE-01/20260830T123000Z__interaction-context-actor-ensemble-s0-r1` | mean rank `-.04474`、cost 4/4退；rejected/F129 |
| P165 | `run://worldsim_v67/WS-V67-P165-JOINT-MULTI-ACTOR-DIFFUSION-01/20260830T124000Z__joint-multi-actor-diffusion-s0-r1` | old mean rank `+.00811`、P147 5/5 rank+cost；old fixed50 cost reject/F130 |
| P166 | `run://worldsim_v67/WS-V67-P166-MONOTONE-EXPECTED-COST-CALIBRATION-01/20260830T125000Z__monotone-expected-cost-calibration-s0-r1` | mean MSE reduction `4.09%<20%`、bin error恶化；rejected/F131 |
| P167 | `run://worldsim_v67/WS-V67-P167-PIPELINED-MULTI-HORIZON-CONFIRMATION-01/20260830T135000Z__pipelined-multi-horizon-confirmation-s0-r2` | 五H mean rank gain `+.21412`、mean cost delta `-.01684`；2/2 second scene-level support |
| P168 | `run://worldsim_v67/WS-V67-P168-JOINT-TAIL-MEAN-COMPILER-01/20260830T131000Z__joint-tail-mean-compiler-s0-r1` | mean rank `+.00460<.005`、旧四cost全退；未读P167，rejected/F132 |
| P169 | `run://worldsim_v67/WS-V67-P169-SOFT-FIXED-COVERAGE-COMPILER-01/20260830T131500Z__soft-fixed-coverage-compiler-s0-r1` | cost 3/4微降、P96微退，mean rank `+.00212<.005`；rejected/F133 |
| P170 | `run://worldsim_v67/WS-V67-P170-CONFORMAL-COST-UPPER-BOUND-01/20260830T132500Z__conformal-cost-upper-bound-s0-r2` | old4 2/2；P167 coverage 4/5 fail despite mean sharpness `25.09%`；rejected/F138 |
| P171 | `run://worldsim_v67/WS-V67-P171-RECTIFIED-CONFORMAL-COST-BOUND-01/20260830T133000Z__rectified-conformal-cost-bound-s0-r1` | coverage pass、mean sharpness over P170 `-19.48%`；未读P167，rejected/F135 |
| P172 | `run://worldsim_v67/WS-V67-P172-CONFORMAL-COST-INTERVAL-01/20260830T133500Z__conformal-cost-interval-s0-r1` | mean width -18.55%但P81 coverage `.732<.78`；rejected/F136 |
| P173 | `run://worldsim_v67/WS-V67-P173-MONOTONE-VISIT-RELIABILITY-CDF-01/20260830T134500Z__monotone-visit-reliability-cdf-s0-r2` | old4 integrated-Brier -34.49%/-45.27%/-46.20%/-31.15%；2/2 dev，calibration boundary retained |
| P174 | `run://worldsim_v67/WS-V67-P174-GROUP-SPLIT-BETA-RELIABILITY-CALIBRATION-01/20260830T140000Z__group-split-beta-reliability-calibration-s0-r1` | Brier 4/4优于control；mean calibration improvement仅5.68%，rejected/F140 |
| P175 | `run://worldsim_v67/WS-V67-P175-VISIT-RELIABILITY-CDF-CONFIRMATION-01/20260830T141500Z__visit-reliability-cdf-confirmation-s0-r1` | mean Brier -34.14% pass；calibration `.07102>.06101` fail，rejected/F147 |
| P176 | `run://worldsim_v67/WS-V67-P176-INTEGRATED-BRIER-VISIT-RELIABILITY-CDF-01/20260830T142000Z__integrated-brier-visit-reliability-cdf-s0-r1` | mean Brier -40.95%但calibration 4/4劣于control；rejected/F141 |
| P177 | `run://worldsim_v67/WS-V67-P177-SCENE-UNIFORM-BRIER-VISIT-RELIABILITY-CDF-01/20260830T142500Z__scene-uniform-brier-visit-reliability-cdf-s0-r1` | mean Brier -40.82%但calibration 4/4劣于control；rejected/F142 |
| P178 | `run://worldsim_v67/WS-V67-P178-CLEARANCE-CONDITIONED-RELIABILITY-CDF-01/20260830T143000Z__clearance-conditioned-reliability-cdf-s0-r1` | Brier/calibration 4/4同向改善；mean calibration improvement仅5.08%，rejected/F143 |
| P179 | `run://worldsim_v67/WS-V67-P179-SET-CONTEXT-RELIABILITY-CDF-01/20260830T143500Z__set-context-reliability-cdf-s0-r1` | Brier 3/4回退、mean calibration change -8.45%；rejected/F144 |
| P180 | `run://worldsim_v67/WS-V67-P180-EFFECTIVE-ERROR-THRESHOLD-RELIABILITY-CDF-01/20260830T144500Z__effective-error-threshold-reliability-cdf-s0-r1` | Brier 4/4回退、mean calibration change -4.38%；rejected/F145 |
| P181 | `run://worldsim_v67/WS-V67-P181-SCENE-BOOTSTRAP-RELIABILITY-CDF-ENSEMBLE-01/20260830T145500Z__scene-bootstrap-reliability-cdf-ensemble-s0-r1` | mean calibration change -0.25%、member deviation约1.5%；rejected/F146 |
| P182 | `run://worldsim_v67/WS-V67-P182-LOG-COST-MIXTURE-DENSITY-01/20260830T150500Z__log-cost-mixture-density-s0-r1` | Brier vs P173 -16.46%至-31.17%；mean calibration improvement 61.17%，2/2 dev |
| P183 | `run://worldsim_v67/WS-V67-P183-LOG-COST-DENSITY-CONFIRMATION-01/20260830T152500Z__log-cost-density-confirmation-s0-r1` | 五H Brier -17.14%至-32.26%，macro calibration +69.38%；fresh supported |
| P184 | `run://worldsim_v67/WS-V67-P184-SCENE-BOOTSTRAP-LOG-COST-DENSITY-ENSEMBLE-01/20260830T153000Z__scene-bootstrap-log-cost-density-ensemble-s0-r1` | mean calibration +20.57%但P81 Brier回退2.18%；rejected/F148 |
| P185 | `run://worldsim_v67/WS-V67-P185-WORST-ENVIRONMENT-LOG-COST-DENSITY-01/20260830T154000Z__worst-environment-log-cost-density-s0-r1` | mean calibration +13.02%但P81 Brier回退2.64%；rejected/F149 |
| P186 | `run://worldsim_v67/WS-V67-P186-NOISE-REGULARIZED-LOG-COST-DENSITY-01/20260830T155000Z__noise-regularized-log-cost-density-s0-r1` | mean calibration +19.60%但Brier 4/4回退；rejected/F150 |
| P187 | `run://worldsim_v67/WS-V67-P187-STUDENT-T-LOG-COST-MIXTURE-DENSITY-01/20260830T160000Z__student-t-log-cost-mixture-density-s0-r1` | P81/P113改善但P96/P129回退，mean calibration +3.85%；rejected/F151 |
| P188 | `run://worldsim_v67/WS-V67-P188-CONDITIONAL-SPLINE-LOG-COST-DENSITY-01/20260830T161000Z__conditional-spline-log-cost-density-s0-r1` | source NLL更低但mean calibration -23.89%；rejected/F152 |
| P189 | `run://worldsim_v67/WS-V67-P189-BUDGET-BRIER-LOG-COST-CDF-01/20260830T161500Z__budget-brier-log-cost-cdf-s0-r2` | mean calibration +11.09%但P81/P129 Brier回退；rejected/F153 |
| P190 | `run://worldsim_v67/WS-V67-P190-PCGRAD-LOG-COST-CDF-01/20260830T162500Z__pcgrad-log-cost-cdf-s0-r1` | 3/4 Brier改善、mean calibration +7.19%，P96 +.62%；rejected/F154 |
| P191 | `run://worldsim_v67/WS-V67-P191-DECOMPOSED-BOUNDARY-EVIDENCE-DENSITY-01/20260830T163000Z__decomposed-boundary-evidence-density-s0-r1` | 仅P113改善，P129 Brier +16.70%；rejected/F155 |
| P192 | `run://worldsim_v67/WS-V67-P192-SCENE-BALANCED-LOG-COST-DENSITY-01/20260830T164000Z__scene-balanced-log-cost-density-s0-r1` | Brier 4/4改善、mean calibration +16.65%；development supported |
| P193 | `run://worldsim_v67/WS-V67-P193-SCENE-BALANCED-POST-CONFIRMATION-01/20260830T170000Z__scene-balanced-post-confirmation-s0-r2` | consumed P183：短H退化、macro calibration -.03%；rejected/F156 |
| P194 | `run://worldsim_v67/WS-V67-P194-MIXED-SCENE-EMPIRICAL-LOG-COST-DENSITY-01/20260830T171000Z__mixed-scene-empirical-log-cost-density-s0-r1` | mean calibration -12.98%；rejected/F157 |
| P195 | `run://worldsim_v67/WS-V67-P195-HORIZON-CONDITIONED-SCENE-SAMPLING-01/20260830T172000Z__horizon-conditioned-scene-sampling-s0-r1` | calibration +9.35%但P81/P113 Brier回退；rejected/F158 |
| P196 | `run://worldsim_v67/WS-V67-P196-MONOTONE-HORIZON-DENSITY-ROUTER-01/20260830T173000Z__monotone-horizon-density-router-s0-r1` | old dev Brier 4/4改善、calibration +18.73%；dev supported |
| P197 | `run://worldsim_v67/WS-V67-P197-ROUTED-DENSITY-POST-CONFIRMATION-01/20260830T174000Z__routed-density-post-confirmation-s0-r1` | consumed P183短H回退；rejected/F159 |
| P198 | `run://worldsim_v67/WS-V67-P198-SHORT-LONG-DENSITY-EXPERTS-01/20260830T175000Z__short-long-density-experts-s0-r1` | P81/P113 Brier回退4.20%/5.56%；rejected/F160 |
| P199 | `run://worldsim_v67/WS-V67-P199-JOINT-HORIZON-RELIABILITY-COPULA-01/20260830T181000Z__joint-horizon-reliability-copula-s0-r2` | joint Brier -16.97%、calibration error -71.85%；source dev supported |
| P200 | `run://worldsim_v67/WS-V67-P200-JOINT-HORIZON-COPULA-POST-CONFIRMATION-01/20260830T183000Z__joint-horizon-copula-post-confirmation-s0-r1` | consumed P183 joint Brier -17.40%、calibration -66.72%；secondary supported |
| P201 | `run://worldsim_v67/WS-V67-P201-JOINT-HORIZON-COPULA-CONFIRMATION-01/20260830T184500Z__joint-horizon-copula-confirmation-s0-r2` | fresh 1,846 joint trajectories；Brier -17.52%、calibration -53.37%；supported |
| P202 | `run://worldsim_v67/WS-V67-P202-DIRECT-MONOTONE-JOINT-CDF-01/20260830T185000Z__direct-monotone-joint-cdf-s0-r1` | calibration -32.51%但Brier +10.32%；rejected/F161 |
| P203 | `run://worldsim_v67/WS-V67-P203-MONOTONE-BETA-JOINT-CALIBRATION-01/20260830T190000Z__monotone-beta-joint-calibration-s0-r1` | Brier -1.32%、calibration -54.32%；source dev supported |
| P204 | `run://worldsim_v67/WS-V67-P204-BETA-JOINT-POST-CONFIRMATION-01/20260830T191000Z__beta-joint-post-confirmation-s0-r1` | consumed P183：Brier -2.57%、calibration -34.51%；secondary supported |
| P205 | `run://worldsim_v67/WS-V67-P205-BETA-JOINT-PROSPECTIVE-SECONDARY-01/20260830T211000Z__beta-joint-prospective-secondary-s0-r2` | same fresh read：Brier -3.70%、calibration -49.12%；prospective secondary supported |
| P206 | `run://worldsim_v67/WS-V67-P206-CONSTANT-JOINT-COPULA-ABLATION-01/20260830T193000Z__constant-joint-copula-ablation-s0-r3` | constant Brier +1.02%、calibration +24.94%；rejected/F162 |
| P207 | `run://worldsim_v67/WS-V67-P207-LOW-RANK-CONDITIONAL-JOINT-COPULA-01/20260830T195500Z__low-rank-conditional-joint-copula-s0-r2` | Brier -.077%但calibration +1.52%；rejected/F163；r1 zero-factor no-verdict |
| P208 | `run://worldsim_v67/WS-V67-P208-CONDITIONAL-COPULA-SHRINKAGE-01/20260830T200500Z__conditional-copula-shrinkage-s0-r1` | mean P199 weight .9828但Brier/calibration回退；rejected/F164 |
| P209 | `run://worldsim_v67/WS-V67-P209-CONDITIONAL-STUDENT-T-COPULA-01/20260830T201500Z__conditional-student-t-copula-s0-r1` | Brier +.56%、calibration +2.20%；rejected/F165 |
| P210 | `run://worldsim_v67/WS-V67-P210-JOINT-MAX-COST-DENSITY-01/20260830T203000Z__joint-max-cost-density-s0-r1` | calibration -28.42%但Brier +1.30%；rejected/F166 |
| P211 | `run://worldsim_v67/WS-V67-P211-JOINT-PROBABILITY-LINEAR-POOL-01/20260830T204000Z__joint-probability-linear-pool-s0-r1` | P210 weight .9812但dev Brier +.96%；rejected/F167 |
| P212 | `run://worldsim_v67/WS-V67-P212-DEEPSET-JOINT-MAX-COST-DENSITY-01/20260830T205000Z__deepset-joint-max-cost-density-s0-r1` | Brier -3.84%、calibration -17.80%；source dev supported |
| P213 | `run://worldsim_v67/WS-V67-P213-DEEPSET-MAX-DENSITY-POST-CONFIRMATION-01/20260830T210000Z__deepset-max-density-post-confirmation-s0-r1` | P183 Brier +2.74%但calibration -19.31%；rejected/F168 |
| P214 | `run://worldsim_v67/WS-V67-P214-PREFIX-SURVIVAL-MAX-COST-DENSITY-01/20260830T213000Z__prefix-survival-max-cost-density-s0-r1` | prefix macro Brier -1.16%但calibration +30.33%；rejected/F169 |
| P215 | `run://worldsim_v67/WS-V67-P215-CALIBRATED-PREFIX-SURVIVAL-DENSITY-01/20260830T214500Z__calibrated-prefix-survival-density-s0-r1` | Brier -.895%、calibration -38.31%；source dev supported |
| P216 | `run://worldsim_v67/WS-V67-P216-CALIBRATED-PREFIX-SURVIVAL-POST-CONFIRMATION-01/20260830T220000Z__calibrated-prefix-survival-post-confirmation-s0-r1` | P183 calibration -19.87%但Brier +1.89%；rejected/F170 |
| P217 | `run://worldsim_v67/WS-V67-P217-TARGET-WEIGHTED-PREFIX-SURVIVAL-01/20260830T221500Z__target-weighted-prefix-survival-s0-r1` | P183 calibration -19.60%但Brier +1.93%；rejected/F171 |
| P218 | `run://worldsim_v67/WS-V67-P218-CUMULATIVE-EXPOSURE-DENSITY-01/20260830T224000Z__cumulative-exposure-density-s0-r1` | Brier -.387%但calibration +26.25%；rejected/F172 |
| P219 | `run://worldsim_v67/WS-V67-P219-CALIBRATED-CUMULATIVE-EXPOSURE-TRANSFER-01/20260830T230500Z__calibrated-cumulative-exposure-transfer-s0-r2` | P183 Brier +4.49%、calibration +25.27%；rejected/F173；r1 syntax no-read |
| P220 | `run://worldsim_v67/WS-V67-P220-SELECTIVE-JOINT-RELIABILITY-AUTHORITY-01/20260830T232000Z__selective-joint-reliability-authority-s0-r1` | selected Brier -23.61%、calibration -36.43%；source dev supported |
| P221 | `run://worldsim_v67/WS-V67-P221-SELECTIVE-AUTHORITY-POST-CONFIRMATION-01/20260830T233000Z__selective-authority-post-confirmation-s0-r1` | P183 selected Brier -13.71%、calibration -27.69%；secondary supported |
| P222 | `run://worldsim_v67/WS-V67-P222-SELECTIVE-AUTHORITY-P201-TERTIARY-01/20260830T234000Z__selective-authority-p201-tertiary-s0-r1` | P201 selected Brier +1.31%、calibration +9.67%；rejected/F174 |
| P223 | `run://worldsim_v67/WS-V67-P223-PAIRWISE-SELECTIVE-AUTHORITY-RECOVERY-01/20260830T235500Z__pairwise-selective-authority-recovery-s0-r1` | same-budget pairwise loss ranking recovery；running |
| P121 | `run://worldsim_v67/WS-V67-P121-CONTINUOUS-BOUNDARY-CONFIRMATION-01/20260830T080500Z__continuous-boundary-confirmation-s0-r1` | Spearman `.76147`、cost reduction `77.36%`；2/2 independent support |

上述locator已按run tree精确对齐；任何metric disagreement仍回到对应canonical summary，不重算quality。

## 2. 论文主表证据

| 问题 | 数据角色 | 结果 | 可写结论 |
|---|---|---|---|
| 全row unreliable-event排序 | P81 independent | query/Actor/P73=`26/57/45` | 窄triage signal |
| visited Actor max-error是否task-conditioned | P84--P94 same-read | 全family未稳定超过Actor-only | negative object result |
| end-to-end occupancy flip是否迁移 | P95 dev → P96 independent | `7/28/13` → `8/5/12` | development shortcut/relative reversal |
| hierarchical query是否恢复 | P102 dev → P103 secondary | `4/27/13` → `9/7/12` | relative reversal retained |
| scalar Actor uncertainty factorization | P107 consumed ×2 → P108 independent | P108=`5/35/20` | scene-level independent support |
| directional uncertainty | P109 consumed ×2 → P110 secondary | P110 directional=`1`, AUROC=.96027 | strong secondary ranking |
| geometry-only解释 | P111 same read | clearance=`1`, AUROC=.91644 | event tie, ranking gap |
| nonlinear sampled crossing | P112 consumed ×2 | cross-cohort worse than linear | reject Monte Carlo recovery |
| downstream monotone tail pool | P114 consumed ×2 | P81/P96都低于P109 max | reject aggregation recovery |
| joint low-frequency Actor sequence | P115 consumed ×2 | P81小增益、P96强退化 | reject spectral recovery |
| directional distribution-free q90 | P116 consumed ×2 | P81/P96都低于P109 Gaussian | reject quantile recovery |
| full bivariate residual covariance | P117 consumed ×2 | P81/P96均0 events，mean AUROC gain=`+.00711` | development mechanism support |
| conditional rho direct contribution | P118 same-checkpoint ablation | mean gain=`+.000094`且P96反向 | reject direct-rho attribution |
| learned uncertainty超过clearance | P113 independent | AUROC `.92016>.87529`但events `6>5` | reject fixed50 composite claim |
| ranked-range tail recovery | P119 consumed ×3 | P113仍6 events、AUROC全退化 | reject binary tail recovery |
| continuous τ-conditioned boundary-state cost | P120 consumed ×3 | P109 strong across all；new regressor worse | freeze P109 object for P121 |
| full covariance用于continuous selection | P122 consumed ×3 | rank gain成立但fixed50 cost nonregression失败 | reject P121 secondary |
| continuous operating-range rank residual | P123 consumed ×3 | P96 cost回退，P81/P96 Spearman下降 | reject downstream head family |
| correlated Student-t Actor residual | P124 consumed ×3 | P96/P113 events和AUROC退化 | reject uniform heavy-tail family |
| two-mode correlated Gaussian residual | P125 consumed ×3 | components active；三cohort AUROC仍全退化 | reject single-model multimodal family |
| epistemic+aleatoric deep ensemble | P126 consumed ×3 | AUROC全增；P96 fixed50多1 event | reject binary composite；transfer continuous object once |
| ensemble continuous selection | P127 consumed ×3 | cost全降、mean Spearman gain=`+.08586` | freeze P128 same-read secondary |
| ensemble continuous confirmation | P128 P121 same read | gain=`+.04721`、selected cost更低 | same-read secondary support；commit timing caveat |
| ensemble increment independent transfer | P129 new scene cohort | gain=`+.04257`、cost `.30867<.32934` | scene-level independent support |
| ensemble distribution distillation | P130 consumed P81/P96/P113 | mean rank delta=`-.00202` | P113 cost regression；reject |
| task-conditioned functional distillation | P131 consumed P81/P96/P113 | mean rank delta=`-.36263` | pointwise→max collapse；reject |
| trajectory-max rank distillation | P132 consumed P81/P96/P113 | mean rank delta=`-.02018` | improved vs P131 but reject |
| BatchEnsemble Actor UQ | P133 consumed P81/P96/P113 | epistemic fraction `.13%--.34%` | shared diversity collapse；reject |
| packed independent Actor ensemble | P134 consumed P81/P96/P113 | mean rank delta=`+.00187` | P96 cost miss；reject |
| full-budget packed ensemble | P135 consumed P81/P96/P113 | mean rank delta=`-.00127` | cost all slightly regress；reject |
| snapshot Actor ensemble | P136 consumed P81/P96/P113 | mean rank delta=`-.00855` | P96/P113 cost regress；reject |
| SWAG Actor ensemble | P137 consumed P81/P96/P113 | mean rank delta=`+.00231` | P81/P96 tiny cost regress；reject |
| full-covariance deep ensemble | P138 consumed P81/P96/P113 | mean gain=`+.00359` | P96 reversal；reject |
| scene-balanced deep ensemble | P139 consumed P81/P96/P113 | mean gain=`-.01275`、cost全退 | reject simple balancing |
| scene-bagged deep ensemble | P140 consumed P81/P96/P113/P129 | mean gain=`-.00405` | P113/P129 cost改善但整体reject |
| five-member deep ensemble | P141 consumed P81/P96/P113/P129 | mean gain=`+.00056` | reject member scaling |
| task-conditioned projected ensemble | P142 consumed P81/P96/P113/P129 | mean gain≈0、cost 3/4退 | task signal但reject replacement |
| conditional residual ensemble | P143 consumed P81/P96/P113/P129 | mean gain=`-.01344`、cost全退 | close per-time correction |
| trajectory-set rank compiler | P144 consumed P81/P96/P113/P129 | mean gain=`-.00096` | reject downstream capacity route |
| absolute-time Actor ensemble | P145 consumed H3.5 ×4 | mean gain=`-.00155` | time signal but reject retraining |
| monotone time-scale adapter | P146 consumed H3.5 ×4 | mean gain=`-.001607` | reject scalar time growth |
| multi-horizon independent transfer | P147 new scene cohort × five H | running | test horizon-wise generalization |
| full-sequence residual ensemble | P148 consumed H3.5 ×4 | mean gain=`-.012380` | reject single-mode joint decoder |
| coherent trajectory mixture | P149 consumed H3.5 ×4 | mean gain=`-.098286` | reject any-time generative score |
| dense boundary-cost ensemble | P150 consumed H3.5 ×4 | mean gain=`-.005427` | object signal but ERM transfer reversal |
| group-DRO dense cost | P151 consumed H3.5 ×4 | mean gain=`-.046683` | reject worst-group direct cost |
| randomized-prior Actor ensemble | P152 consumed H3.5 ×4 | mean gain=`-.006852` | reject forced function diversity |
| Bayesian last layer | P153 consumed H3.5 ×4 | mean gain=`+.000844` | reject overconcentrated token posterior |
| density-aware P126 | P154 consumed H3.5 ×4 | mean gain=`-.001727` | rarity not reliability |
| RegMixup Actor ensemble | P155 consumed H3.5 ×4 | mean gain=`-.007045` | reject train-time interpolation |
| integrated increment ensemble | P156 consumed H3.5 ×4 | mean gain=`-.028923` | reject independent increment integration |
| horizon-specialist Actor ensemble | P157 consumed H3.5 ×4 | mean gain=`-.59434`、cost 3.9--5.8× P126 | reject nearest-lower extrapolation |
| multi-horizon ensemble increment | P147 new 10-scene × five H | per-H gain全正，macro=`+.17419`；cost delta全负 | scene-level independent multi-H support |
| continuous object independent transfer | P121 new scene cohort | Spearman `.76147`、cost reduction `77.36%` | scene-level independent support |

## 3. Failure map

| ID | 状态 | 论文处理 |
|---|---|---|
| `V67-F67` | closed negative | visited max-error对象不支持task-conditioned claim |
| `V67-F68--F73` | mixed scientific/engineering chain | subtype、auxiliary与data scaling不恢复独立迁移 |
| `V67-F74` | closed negative | P95 independent relative reversal |
| `V67-F75` | closed negative | P102/P103 hierarchical relative reversal |
| `V67-F76` | resolved pre-run | launcher工作目录丢失；0 scientific exposure |
| `V67-F77` | resolved pre-optimizer | NPZ原子交付race；0 optimizer step |
| `V67-F78` | closed negative | nonlinear Gaussian sampling跨cohort退化 |
| `V67-F79` | closed negative | P114 top-k/union tail pool稀释P109 max |
| `V67-F80` | closed negative | P115 low-frequency Actor sequence在P96过度平滑 |
| `V67-F81` | closed negative | P116 directional q90低于P109 standardized margin |
| `V67-F82` | resolved pre-target | P113 scene-0003 exact archive locator corrected `04→01` |
| `V67-F83` | closed negative | P118 conditional-rho direct mechanism not supported |
| `V67-F84` | closed negative | P113 AUROC gain did not yield fixed50 event noninferiority |
| `V67-F85` | closed negative | P119 ranked-range objective did not change P113 tail ordering |
| `V67-F86` | closed negative | P120 continuous regressor did not exceed frozen P109 |
| `V67-F87` | closed negative | P122 full-covariance rank gain did not preserve fixed50 cost on all cohorts |
| `V67-F88` | closed negative | P123 dense continuous pairs still caused cross-cohort rank drift |
| `V67-F89` | closed negative | P124 fixed heavy-tail likelihood over-broadened P96 boundary uncertainty |
| `V67-F90` | closed negative | P125 learned mixture modes were not cross-cohort boundary-relevant |
| `V67-F91` | closed binary composite | P126 AUROC consistently improved but P96 fixed50 event noninferiority failed |
| `V67-F92` | resolved pre-run | P129 asynchronous launcher相对入口工作目录错误；0 target read |
| `V67-F93--F101` | closed negative | distillation、efficient/snapshot/SWAG与full-covariance路线未稳定保持P126 |
| `V67-F102` | closed negative | uniform source-scene sampling三cohort一致退化 |
| `V67-F103` | closed negative | scene bootstrap仅在P113/P129改善cost，四cohort rank不稳定 |
| `V67-F104` | closed negative | five-member scaling未超过three-member P126 boundary |
| `V67-F105` | closed negative | direct conditional projection在P129增益但P96 transfer反转 |
| `V67-F106` | closed negative | P126-standardized conditional correction四cohort cost全退 |
| `V67-F107` | closed negative | trajectory-set compiler仍在P96 transfer反转 |
| `V67-F108` | closed negative | absolute-time retraining三cohort rank增但P96强反转 |
| `V67-F109` | closed negative | frozen mean的monotone time-scale仍未跨四cohort迁移 |
| `V67-F110` | closed negative | full-resolution sequence ensemble四cohort rank一致下降 |
| `V67-F111` | closed negative | coherent modes未collapse但any-crossing score严重错位 |
| `V67-F112` | closed negative | direct cost在P81/P129有信号但P96 ERM反转 |
| `V67-F113` | closed negative | scene×horizon worst-group NLL进一步放大P96反转 |
| `V67-F114` | closed negative | randomized function priors四cohort rank全退 |
| `V67-F115` | closed negative | exact last-layer posterior epistemic约`10^-4`而过度集中 |
| `V67-F116` | closed negative | hidden density识别shift但blind inflation使P81 cost显著回退 |
| `V67-F117` | closed negative | same-fraction RegMixup只改善P129 cost且rank mean为负 |
| `V67-F118` | resolved pre-target | P147 scene0110 shard locator `01→02`；3,909/3,909 mapped，随后primary支持 |
| `V67-F119` | closed negative | continuous-time increment integration四cohort rank一致下降 |
| `V67-F120` | closed negative | nearest-lower H3.0 specialist外推H3.5导致严重rank/cost collapse |
| `V67-F121` | closed negative | marginal CRPS跨旧cohort退化；P147 rank亮点未转成中长H fixed50 cost |
| `V67-F122` | closed negative | joint Energy Score未恢复，旧cohort rank/cost退化更大 |
| `V67-F123` | closed negative | exact member-CDF linear pool在旧四与P147五H均退化 |
| `V67-F124` | closed negative mechanism | explicit between-member variance约1%--3%，不是P147增益主因 |
| `V67-F125` | resolved pre-training | P162 three-digit + V4/V67 dual-root routing；r3 training active |
| `V67-F127` | closed negative | direct query-normal footprint support仍无稳定rank/cost增量；oriented-footprint family关闭 |
| `V67-F128` | resolved pre-training | P164 non-login `PYTHONPATH`入口；0 step后原样恢复 |
| `V67-F129` | closed negative | nearest-neighbor marginal adapter source NLL改善但旧四/P147全面退化 |
| `V67-F130` | closed composite negative | joint diffusion 9/9 rank slices正向，但旧P81/P96/P129 q75 fixed50 cost回退 |
| `V67-F131` | closed negative | monotone point-cost calibration raw MSE仅改善4.09%，expected-cost bin error恶化 |
| `V67-F132` | closed negative | P165 top-4/16 upper-tail mean仍使旧四fixed50 cost全部回退；未读P167 |
| `V67-F133` | closed negative | direct soft fixed50训练收敛到P126邻域；P96微退且mean rank gain不足 |
| `V67-F134` | resolved pre-evaluation | P109 source已排除absolute mod-5 scenes；改为artifact内ordered unique scene group split |
| `V67-F135` | closed negative | conditional conformity rectifier保持coverage但mean upper-bound sharpness恶化19.48% |
| `V67-F136` | closed negative | q10/q90 interval width缩短18.55%但P81 coverage仅.732 |
| `V67-F137` | resolved post-aggregate | P167 constant-input local Spearman用`null`表示；macro未改，r2支持 |
| `V67-F138` | closed negative | P170 P167 coverage 4/5 horizon低于.88，尽管bound更窄 |
| `V67-F139` | resolved pre-training | P173 direct-script module path；0 step后process-local recovery |
| `V67-F140` | closed negative | P174 Beta calibration mean improvement 5.68%且P81反向 |
| `V67-F141` | closed negative | P176 Brier objective保留refinement但4/4 calibration error高于control |
| `V67-F142` | closed negative | P177 scene-uniform sampling仍4/4 calibration error高于control |
| `V67-F143` | closed negative | P178 absolute clearance带来4/4一致小改善但mean calibration gain仅5.08% |
| `V67-F144` | closed negative | P179 set-context residual使Brier 3/4回退且mean calibration change -8.45% |
| `V67-F145` | closed negative | P180 minimum-clearance effective threshold使Brier 4/4回退、mean calibration change -4.38% |
| `V67-F146` | closed negative | P181 scene-bootstrap member probability deviation约1.5%，mean calibration change -0.25% |
| `V67-F147` | closed negative | P175 fresh Brier gain 34.14%但marginal calibration error高于horizon-only |
| `V67-F148` | closed negative | P184 mean calibration改善20.57%但P81 Brier回退2.18% |
| `V67-F149` | closed negative | P185 mean calibration改善13.02%但P81 Brier回退2.64% |
| `V67-F150` | closed negative | P186 mean calibration改善19.60%但Brier四cohort全部回退1.63%--27.51% |
| `V67-F151` | closed negative | P187 heavy tails仅改善P81/P113，P96/P129概率刻度回退 |
| `V67-F152` | closed negative | P188 spline source NLL改善但仅P96 Brier受益，mean calibration恶化23.89% |
| `V67-F153` | closed negative | P189 pure Brier改善P96/P113但P81/P129回退，NLL refinement丢失 |
| `V67-F154` | closed negative | P190 PCGrad仅余P96 Brier +.62%，严格noninferiority仍失败 |
| `V67-F155` | closed negative | P191 decomposed evidence仅P113改善，P129 Brier回退16.70% |
| `V67-F156` | closed negative | P192纯scene等权在consumed P183的H.8/1.5 Brier回退，macro calibration无改善 |
| `V67-F157` | closed negative | P194全局50/50采样仅P96 Brier改善，mean calibration退化12.98% |
| `V67-F158` | closed negative | P195条件sampler改善P96/P129但P81/P113 Brier回退，共享参数干扰未解 |
| `V67-F159` | closed negative | P196 source-NLL pool在consumed P183短H仍回退，router suitability不迁移 |
| `V67-F160` | closed negative | P198 horizon specialists放大P81/P113回退；P192--P198 refinement family terminal |
| `V67-F161` | closed negative | P202 direct joint CDF改善平均校准但Brier恶化10.32%，instance refinement丢失 |
| `V67-F162` | closed negative | P206 global constant copula的Brier/校准均低于P199，条件dependence不可省略 |
| `V67-F163` | closed negative | P207 rank-2 conditional结构仅微增Brier且校准回退1.52% |
| `V67-F164` | closed negative | P208 P199/independence conditional shrinkage的Brier与校准均回退 |
| `V67-F165` | closed negative | P209 fixed-nu4 Student-t copula的Brier与校准均低于Gaussian P199 |
| `V67-F166` | closed negative | P210 direct max-cost density改善校准28.42%但Brier退化1.30% |
| `V67-F167` | closed negative | P211 source-trained P199/P210 scalar pool在dev仍退化Brier |
| `V67-F168` | closed negative | P212 DeepSet max density在P183 secondary中Brier退化2.74% |
| `V67-F169` | recovered source-only | P215 disjoint beta calibration在source dev同时改善Brier/calibration |
| `V67-F170` | recovery failed | P216跨P183 Brier退化1.89%；P217 importance weighting亦未恢复 |
| `V67-F171` | closed negative | P217 target covariates近乎不可分且Brier仍退化1.93%；prefix density family关闭 |
| `V67-F172` | recovery failed | P218 source calibration issue由P219 source修复但未跨P183迁移 |
| `V67-F173` | closed negative | P219在P183 Brier/calibration均退化；direct aggregate density关闭 |
| `V67-F174` | recovery active | P220 absolute loss head未过P201；P223 pairwise ranking运行中 |
| `V67-F126` | closed negative | yaw→rectangle-support一阶传播未形成稳定cost/rank增量 |

## 4. Artifact inventory

| artifact | location |
|---|---|
| canonical run summaries/configs/status | `/root/autodl-tmp/runs/worldsim_v67/<task>/<run>/` |
| P107 scalar checkpoint | P107 canonical run |
| P109 directional checkpoint | P109 canonical run |
| P117 full-covariance checkpoint | P117 canonical run |
| P118 same-checkpoint rho ablation | P118 canonical run |
| P122 full-covariance continuous selection | P122 canonical run |
| P123 continuous rank-residual checkpoint/result | P123 canonical run |
| P124 correlated Student-t checkpoint/result | P124 canonical run |
| P125 two-mode Gaussian checkpoint/result | P125 canonical run |
| P126 three-member ensemble checkpoint/result | P126 canonical run |
| P127 ensemble continuous selection | P127 canonical run |
| P121 independent continuous rows/summary | P121 prep与primary canonical runs |
| P128 ensemble same-read summary | P128 canonical run；timing caveat见ledger/report |
| P129 independent rows/summary | P129 prep与primary canonical runs |
| P108 independent rows/summary | P108 prep与primary canonical runs |
| P111 clearance comparator | P111 canonical run |
| P113 independent rows/summary | P113 prep与primary canonical runs |
| chronological ledger | `docs/EXPERIMENTS.md` |
| failure ledger | `docs/RESEARCH_FAILURES.md` |
| live status | `docs/RESEARCH_STATUS.md` |
| technical narrative | `docs/autoresearch/worldsim_v67/V67_ARXIV_TECHNICAL_REPORT.md` |

## 5. Claim boundary

- independent表示scene-level；不是session-level，因为历史cohort可能含相邻session scene。
- occupancy-decision flip不是collision label；risk score不是calibrated probability。
- fixed-coverage排序不授权planner、policy、closed-loop、control或deployment。
- P110/P111共享P108 target read，只是prospective secondary/comparator，不是第二个独立cohort。
- P113只裁决learned directional uncertainty相对clearance-only的ranking/event增量，不重裁P108相对Actor-only/P75的主结论。
- P113 composite verdict为reject；可以报告AUROC gate通过，但不得省略events `6>5`或改称整体支持。
- P117在P113冻结后完成且只读consumed cohorts；它不能替换P113的P109 checkpoint或充当独立确认。
- 未新增hash/checksum/fingerprint；没有smoke/regression matrix或参数扫描。

## 6. Source-of-truth order

1. canonical `summary.json` / `status.json` / retained row artifacts；
2. `docs/EXPERIMENTS.md`；
3. `docs/RESEARCH_FAILURES.md`；
4. `docs/RESEARCH_STATUS.md`与`AUTORESEARCH_STATE.current.json`；
5. 本index与`V67_ARXIV_TECHNICAL_REPORT.md`。
