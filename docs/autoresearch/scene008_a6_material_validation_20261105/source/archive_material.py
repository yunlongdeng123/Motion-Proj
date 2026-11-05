"""Archive this fixed negative material result and benchmark interfaces only."""
import shutil,json
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DEST=REPO/'docs/autoresearch/scene008_a6_material_validation_20261105'
def main():
    assert not DEST.exists();shutil.copytree(RUN/'delivery',DEST)
    shutil.copyfile(RUN/'delivery.tar.gz.sha256',DEST/'delivery.tar.gz.sha256')
    (DEST/'README.md').write_text('''# Scene008 vehicle appearance: strict spatial holdouts and surface continuation

The previous audit attributed the flat lower body to missing material supervision and a global blue-grey fallback. This experiment tests whether geometry-only colour continuation can recover real withheld vehicle appearance while preserving the existing asset and scene.

![Architecture components](architecture.svg)

## Result

No new scene asset is adopted. Fixed same-scene material holdouts on left/right/top yield case-equal HELD RGB MAE:

| Rule | HELD masked MAE |
|---|---:|
| Fresh global median |.173269|
| Child geodesic-nearest |.188268|
| Child harmonic |.174285|
| Parent-lift geodesic-nearest |.191566|
| Parent-lift harmonic |.179191|

All four candidates fail the fixed quality gates. All49-frame observed-surface/road/other-actor preservation gates pass; alpha,depth and group contributions are bitwise unchanged. More reachable or more varied material is not evidence of accurate hidden texture. The original M2 remains the reference.

## Controlled protocol

Geometry, opacity, trajectory, point count/order, road, environment, native background and renderer stay frozen. Spatial regions use canonical y+/y-/z+ weighted quartiles from BUILD normalized operator mass, never RGB. Drop every BUILD row with ANY nonzero hidden-point contribution, with no epsilon exemption. Left/right/top retain18710/17885/16822 RGB pixels; BUILD withheld target-dominant rows6279/6095/5197. Evaluation uses original observed rows with hidden contribution>=.9; mixed rows and unreachable areas remain separately reported, never removed to improve a score.

Each case uses a fresh median from retained real RGB and one shared2000-step bounded SH0 fit. No full-fit M2 RGB/prior/native predicted appearance enters pseudo training. All methods freeze every nonzero TRAIN column and modify only strict zero columns. Unknown colours are absent from the extension API. Export/reload preserves exact training predictions. The pseudo operator has33331 extra-eroded observed rows across49 frames; real-scene evaluation separately uses the original39308-pixel core and8930 depth measurements.

The original child graph is geometry-only mutual8NN with fixed normal,tangency,distance and same-sheet barriers. The ancestry control applies the same formulas on2048 exact-source mother points and lifts edges to existing child0 vertices with verified within-family stars; it creates no primitives or coarse averaged colours. Coarse spacing changes physical edge radii/tolerances and does not prove thin-surface topology. Child/parent components1659/1088; reachable original zero columns1662/3321 out of10557. Both graph families and all five arms were declared before pseudo quality outcomes. Engineering gates were frozen after32-step wiring smoke and before formal fitting or quality-score inspection, as explicitly recorded.

## Input eligibility and next model interface

All original200 PVD points exactly map to raw LiDAR UIDs, with no other annotated-vehicle physical-box overlap.199/200 pass a fixed5cm,>=3raw,>=2otherraw,>=.5sec geometry-consistency test. After excluding14 old HELD source rows,185 are admitted for a future clean-conditioning control. The new5399-candidate pool keeps real representatives and provenance, and remains semantic-unknown. No new PVD inference was run in this material test.

Four independent appearance input archives contain full/left/right/top data, each with ONLY its own40 BUILD FRONT views, geometry and scoped contract. They are the same40 cameras under four masks, not160 independent views. RGBA alpha0 meansunknown; float32 NPZ RGB and explicit known/unknown masks are authoritative. Crop intrinsics, metre object-to-camera transform, meter_to_model and prior-depth roles are explicit. No HELD/masked RGB, full-fit M2 material or sibling/full-case answers are included in a pseudo-case archive. This interface prepares a future explicit appearance prior; it does not itself complete texture or train PVD.

## Validation and accounting

6000 formal numerical colour updates;96 smoke updates and6 next-step replay calls extra, each with save/full-state restore/extension/export/scene entry. No neural model training. Independent CPU review verifies actual lifted endpoint barriers, the held-out row boundary, real exported DC/RGB preservation, normalization and isolated input archives. GPU full49 validation checks the identical material and geometry in keep/delete; deletion removes a37 only and native a6 has zero renderer rows.

[Interactive report](index.html) · [Fixed decision](decision.json) · [Numerical gate contract](evaluation_contract.json) · [Pseudo metrics](evidence/pseudo_summary.json) · [Real49 metrics](evidence/real_summary.json) · [Independent implementation review](evidence/review_final_implementation.json) · [Input QA](evidence/appearance_input_QA.json)

[Full input](appearance_inputs_full.tar.gz) · [Left input](appearance_inputs_left.tar.gz) · [Right input](appearance_inputs_right.tar.gz) · [Top input](appearance_inputs_top.tar.gz) · [Input contracts](appearance_input_contract.json)

Candidate assets and pseudo replays are retained for reproducibility, not adopted into the current scene. Executed source/review bytes and logs are archived separately; published text uses LF. Full replay uses the existing campaign siblings and deterministic CSR library from the prior controls. Standalone asset loading uses sh_asset.py/full_pvd.py and explicit NPZ paths.

No old checkpoint, vehicle asset or road representation was overwritten. AutoDL remains ON as requested.
''',encoding='utf-8')
    for name in ('EXPERIMENTS.md','RESEARCH_STATUS.md'):
        with (REPO/'docs'/name).open('a',encoding='utf-8') as f:
            f.write('\n\n2026-11-05，Scene008车辆外观严格伪遮挡完成：3case/5规则，HELD宏RGB误差默认色.17327、细图调和.17429、父面调和.17919，四候选未过质量门，不采用；49帧保护门均通过、alpha/depth/groups逐位不变。原200点均实测来源、199跨帧自洽，无大规模他车实体重叠证据。交付4个隔离的显式mask/米制camera外观模型输入包，PVD未训练，M2保留，AutoDL开机。[报告与重放](autoresearch/scene008_a6_material_validation_20261105/README.md)。\n')
    print(json.dumps(dict(archived=str(DEST))))
if __name__=='__main__':main()
