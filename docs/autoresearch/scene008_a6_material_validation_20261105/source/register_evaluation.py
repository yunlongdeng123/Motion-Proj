"""Freeze numerical quality gates before formal fit or quality evaluation."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
RUN=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (RUN/'formal').exists() and not (RUN/'pseudo_evaluation').exists() and not (RUN/'real_evaluation').exists()
with np.load(RUN/'data/sealed_targets.npz') as data:raw=data['raw'];build_ids=data['BUILD'];held_ids=data['HELD']
build=np.isin(raw,build_ids);A=load_npz(RUN/'data/all_design.npz');records=[]
for name in ('left','right','top'):
    with np.load(RUN/'cases'/name/'split.npz') as spec:H=spec['hidden']
    part=A[:,H];hit=part.getnnz(axis=1)>0;fraction=np.asarray(part.sum(1)).ravel()/np.asarray(A.sum(1)).ravel();test=hit&(fraction>=.9)
    train=build&~hit;held=~build
    sufficient=dict(retained_train_pixels=int(train.sum())>=1000,retained_BUILD_frames=len(np.unique(raw[train]))>=10,
        BUILD_test_pixels=int((test&build).sum())>=200,HELD_test_pixels=int((test&held).sum())>=100,
        HELD_frames_20pixels=sum(int((test&(raw==r)).sum())>=20 for r in held_ids)>=3)
    records.append(dict(case=name,sufficient=sufficient,status='INFORMATIVE' if all(sufficient.values()) else 'INCONCLUSIVE',
        train_pixels=int(train.sum()),BUILD_test_pixels=int((test&build).sum()),HELD_test_pixels=int((test&held).sum())))
value=dict(status='FROZEN_BEFORE_FORMAL_FIT_AND_QUALITY_EVALUATION',
    timing='Numerical gates registered after engineering-only32step smoke, before formal2000step fits or inspecting pseudo RGB comparison scores. Not claimed before every smoke update.',
    registration_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    case_contract_sha256=sha(RUN/'cases/contract.json'),smoke_receipt_sha256=sha(RUN/'smoke/receipt.json'),
    fit_source_sha256=sha(Path(__file__).with_name('fit_material.py')),module_sha256=sha(Path(__file__).with_name('surface_material.py')),
    methods=['median','child_nearest','child_graph','parent_nearest','parent_graph'],cases=records,
    gates=dict(absolute_HELD_macro_improvement_min=.003,relative_HELD_macro_improvement_min=.10,
        maximum_case_worsening=.003,case_improvements_at_least=2,case_improvement_min=.003,
        HELD_mixed_macro_worsening_max=.003,specific_graph_claim_macro_improvement_min=.002,
        real_pooled_and_each_frame_RGB_MAE_increase_max=.003,real_mean_RGB_parent_change_max=.003,
        alpha_depth_group_tolerance=5e-6,retained_BUILD_operator_prediction_must_be_exact=True),
    source='Independent pre-outcome evaluation_review.json; engineering thresholds, not universal perceptual criteria.',
    no_case_specific_winner=True,unreachable_test_pixels_stay_in_denominator=True,scene_adoption_requires_quality_and_preservation=True)
(RUN/'evaluation_contract.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
