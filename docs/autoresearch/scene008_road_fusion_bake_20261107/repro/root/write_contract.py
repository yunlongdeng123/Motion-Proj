"""登记本轮边界融合目标与持久三维写回；保留前轮成像支撑。"""
from pathlib import Path
import json
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=BASE/'scene008_road_fusion_bake_20261107_r1'
scope=dict(
    task_id='scene008_road_fusion_bake_20261107',run_id=RUN.name,
    parent_run='scene008_road_seam_contact_20261107_r1',
    user_objective='Blue-circle seam: shared fusion target -> persistent 3D writeback; preserve red-wheel support and other imaging.',
    base_candidate='canonical_n2_mip_candidate_v2: 40392 children / 10098 persistent world road parents; relative-Mip amplitude + parent cellfit1000 material',
    primary_hypothesis='A multi-view shared road-chart target can reconcile measured interior appearance with reliable preserved boundary context better than directly fitting unblended observations at equal optimization budget.',
    controls=['Frozen preceding AA candidate','Direct real RGB writeback','Shared fusion-target writeback'],
    frozen=['All road geometry/covariance/intrinsic opacity and child-parent mapping','AA projection/filter policy','C2/SurfaceLOD vehicle, a37 and fixed trajectories','Original native 10K / road1K / gates / environment'],
    trainable='Only persistent parent-road RGB [10098,3], children inherit the same parent colour. Valid [0,1]; rows without legal fit influence remain frozen.',
    target_protocol='Aggregate legal real-road observations into a shared world road chart; blend with reliable preserved boundary context in that chart; reproject the same shared target. A target is supervision, never a screen-space edit at inference.',
    build_raws=[r for r in range(89,138) if r not in (89,95,101,105,113,121,125,130,137)],
    query_raws=list(range(89,138)),
    query_role='Same-scene development evaluation; prior source proxies and prior reconstructions already saw this clip. Not independent generalization or a blinded road prior test.',
    optimizer_contract='Equal direct/fusion fixed-budget pair: 400 paired keep/delete steps = 10 epochs over BUILD40, Adam LR .01; no weight, learning-rate, seed or budget sweep.',
    actual_environment_smoke='Separate two-frame save/eval/full Adam/RNG restore/next-stage exercise. If measured ETA becomes hour-scale, complete approximately ten-minute real smoke before formal run.',
    primary_blue_evaluation='Reuse original raw128 observed-road ROI and all fixed five-key world-edge masks. Additional full49 masks must use the same original geometry/road/guard rules, saved before candidate reads. RGB and paired tangent-bin residual jumps; unsupported is null.',
    target_evaluation='Shared target boundary continuity and deviation from actual measured road, followed separately by 3D target-writeback error. Do not infer success from target visual smoothness alone.',
    protection_evaluation='49 frames, KEEP/DELETE separate: original visible a6/a37, road, protected region, measured depth, group contributions and fixed raw128/raw130 wheel strata. Colour-only alpha/depth/group contribution should remain exact.',
    capacity_diagnostic='Per-pixel relaxation RGB in [F,F+Wnewroad] at fixed geometry. Positive lower bound can diagnose fixed contributions; zero lower bound does not prove shared 3D reachability.',
    only_possible_fallback='If target/bake evidence establishes the need: measured road surface takeover from conflicting legacy support, persistent in world coordinates and both states; protect mixed footprints and other surfaces. Requires a written concrete contract before execution, not a blind proximity prune.',
    stopping_rule='Finish fixed direct/fusion controls and full49 evaluation. No parameter grid or geometry modification to rescue a negative. At most the one explicitly registered, evidence-motivated legacy compatibility fallback.',
    external_methods=dict(
        GS_RoadPatching='https://arxiv.org/html/2509.19937',
        GIMP_3D='https://arxiv.org/html/2607.20789',
        CoGeo_GS='https://arxiv.org/html/2608.26656',
        TexMOR_3D='https://arxiv.org/html/2610.11198'),
    scope_limits=['Not an exact reproduction of the four papers','No learned road generator or PBR/intrinsic decomposition this round','No new monocular depth needed for an observed LiDAR-supported road patch','Native Gaussian depth sorting, not MOG-2 opaque first-hit guarantee'],
    failure_ledger_refs=['V82-F13'],human_verdict=None,whole_scene_adoption_claim=False,
    subagents='gpt-6.1-sol max only',power_policy='KEEP_ON: no AutoDL shutdown')
folder=RUN/'contract';folder.mkdir(parents=True,exist_ok=True)
p=folder/'round_scope.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n')
print(p)
