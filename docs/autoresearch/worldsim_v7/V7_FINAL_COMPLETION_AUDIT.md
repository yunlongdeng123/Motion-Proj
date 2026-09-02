# WorldSim V7 final completion audit

Audit date: 2026-09-02. Governing plan: `docs/WORLDSIM_V7_CVPR_RESEARCH_PLAN.md`.

## Authority and branch

- Active branch: `research/worldsim-v7-harp3d-cvpr`.
- Required base: terminal `research/worldsim-v6.7-anisotropic-surface`.
- Observed merge-base equals the required branch tip; ancestor check is true. At this audit the V7 branch is 122 commits ahead and
  zero behind.
- The corrected V6.7 instruction supersedes the stale V6.4/V6.5 objective text retained by older task metadata.

## Method-expansion stop conditions

| Plan condition | Authoritative evidence | Audit state |
| --- | --- | --- |
| C1 has hard 3D physical evidence and qualitative evidence | P3/P3-C/P20--P23 canonical summaries; main Figures 1--2; 30 P3-B panels and 30 videos | satisfied |
| C2 has paired counterfactual disentanglement evidence | P1 deterministic paired atlas; P4 factorized selector; Table 1; P7/P7-C shortcut and stable-error boundaries | satisfied |
| C3 has nuScenes-to-AV2 zero-shot evidence | P4 consumed AV2, P6-C exact-once fresh AV2, P8-A fresh nuScenes reversal; reliability Table 4 | satisfied with the documented sensor-dependence boundary |
| Final exact-once and minimal downstream are complete | P8-A, P6-C external phase, P3-C fresh, and P23 exact-once runs; P9 fixed-lattice 2x2 audit | satisfied |

The plan therefore permits paper writing, figure redraws, code convergence, supplement work, and legitimate bug fixes only.
No further model family, test-cohort read, AV2 holdout, threshold sweep, or policy selection is authorized.

## Research assets

| Required asset | Current implementation/evidence | Audit state |
| --- | --- | --- |
| HARP-3D physical SceneIR | `motion_proj/worldsim_v7/sceneir_adapter.py`, `physical_compiler.py` | complete |
| Actor canonical physical surface | `av2_canonical_surface.py`, `nuscenes_actor_surface.py`, P1--P3 | complete |
| Validity/hazard factorization | `validity_hazard.py`, `selective_validity_hazard.py`, P4/P7 | complete |
| Continuous task-cost density | `boundary_cost_density.py`, inherited frozen P182/P183 evidence | complete |
| Joint multi-horizon reliability | `runtime_surface.py`, inherited frozen P199/P201 evidence | complete |
| nuScenes-to-AV2 zero-shot benchmark | three disjoint metadata-frozen AV2 cohorts and P4/P6-C/P23 results | complete |
| Exact-once final cohort | P8-A 20 fresh nuScenes scenes; P6-C/P3-C 20 fresh AV2 logs; P23 10 fresh AV2 logs | complete |
| Minimal downstream demo | P9 retained-source fixed-lattice 2x2 composed-authority audit | complete, explicitly non-causal/open-loop |

`paper/CONTRIBUTION_MAP.md` registers 29 paper-owning V6.7/V7 canonical run URIs. Direct filesystem inspection found a nonempty
`status.json` and `summary.json` for all 29, and every status is `done`. The P1 and P2 canonical run directories and summaries are
also present. Rejected P6/P8/deletion/router paths remain recorded rather than being replaced by favorable reruns.

## Data, qualitative assets, and storage

- `av2_zero_shot_cohort_v1.json`, `av2_zero_shot_recovery_cohort_v1.json`, and
  `av2_evicomp_fresh_cohort_v1.json` contain 30/20/10 unique log IDs. Their union is 60 IDs and all 60 Sensor-val directories exist.
- `/root/autodl-tmp/data/av2/sensor/val` contains 60 log directories and `66,481,709,102` bytes at this audit.
- The P3-B canonical run contains exactly 30 PNG panels and 30 MP4 videos (`47,495,050` bytes), indexed by
  `paper/PROJECT_PAGE_ASSET_INDEX.md` without duplicating the binary bundle in Git.
- P0 deleted only four exact, recoverable targets totaling `23,027,117,355` bytes (`21.446 GiB`). The retained/deleted boundary is
  documented in `docs/archive/2026-09/worldsim-v7-p0-cleanup/CLEANUP_MANIFEST.md`; no unique run, dataset, or V7 dependency was removed.
- Current `/root/autodl-tmp` free space is approximately 98 GiB. No additional deletion is justified by the V7 dependency graph.
- No checksum, content fingerprint, or hash-based research gate was introduced.

## Paper package

| Required paper asset | Audit state |
| --- | --- |
| `paper/main.tex` / anonymous PDF | complete under the latest official public CVPR kit; 8 content pages + 1 references-only page |
| `paper/supplement.tex` / PDF | complete; 10 pages with exact-once protocol, literal audit, failure ledger, intervals, and frozen gallery |
| `paper/results/results_macros.tex` | complete and remains the sole manuscript numeric interface |
| Main visual roles | adapted: Figure 1 combines teaser and architecture; Figure 2 combines physical mechanism, cross-sensor literal evidence, monotone utility boundary, and factorized sensitivity |
| Continuous reliability / cross-dataset / downstream presentation | complete as Tables 1 and 4--6; not duplicated as a fifth main figure under the 8-page limit |
| Condensed failure appendix | complete in the supplement |
| Project-page/video asset inventory | complete in `paper/PROJECT_PAGE_ASSET_INDEX.md` |
| Non-anonymous arXiv package | externally pending author/affiliation/contact, category, license, and acknowledgement metadata |
| Final CVPR 2027 package | externally pending the official 2027 kit/policy and assigned paper ID |

The five originally planned visual roles were consolidated rather than silently dropped: the two main floats plus six main tables
cover the same evidence while preserving the official eight-page content limit. Complete risk/geometry coverage curves remain in the
supplement. This is a documented layout adaptation, not a claim or result change.

## Frozen claim boundary

- Literal first-return correction transfers as a measurement/provenance result, not as learned model or policy generalization.
- nuScenes-to-AV2 selector transfer is empirical; the cross-fresh rank reversal and sensor-opportunity shortcut remain visible.
- Set deletion proves only non-increasing literal early returns for fixed rays; it does not prove hit retention, Chamfer improvement,
  collision freedom, planning benefit, or road safety.
- Stable interval decisions can remain wrong, and physical repairability remains distinct from motion uncertainty and task authority.
- P9 is an exact-identity open-loop interface audit, not a causal downstream or closed-loop result.

## Remaining external actions

The science, data processing, code-facing assets, anonymous manuscript, and supplement are complete under the V7 stop contract.
The overall publication handoff remains active because the following cannot be inferred or fabricated:

1. final author order, affiliations, contacts, and acknowledgement text;
2. arXiv category and license choice;
3. conference paper ID;
4. the official CVPR 2027 author kit and final policy when published.

When these inputs exist, follow `paper/SUBMISSION_CHECKLIST.md`. They do not authorize another experiment or test-set read.
