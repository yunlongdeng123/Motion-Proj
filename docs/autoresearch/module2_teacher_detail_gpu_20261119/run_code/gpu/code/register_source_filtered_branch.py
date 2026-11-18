"""Prospective, separate matched-budget branch prompted by observed HF/noise tradeoff."""
from pathlib import Path
import datetime,json
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
R=Q/'gpu_20261119';out=R/'source_filtered_training';out.mkdir(exist_ok=True)
assert json.loads((R/'formal_r1/completed.json').read_text())['formal_physical_updates']==3400
prior=json.loads((R/'source_footprint_filter_r1/results.json').read_text())
contract=dict(schema='teacher_detail_source_filtered_branch.v1',
    task_id='MODULE2-TEACHER-SOURCE-FILTER-20261119',arm='M12',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    user_authorization='GPU已开，继续推进实验，目标是把高频的改善好',
    formal_authorized=True,formal_requires_new_real_lifecycle=True,
    rationale='Teacher-H permission improves C/S HF fit but adds fine bands; post-filtering removes ~15% diagnostic Ey but damages teacher fit. Train through the same fixed geometric filter to test whether shared field can compensate without those unresolved bands.',
    comparison='M11 vs M12, same pixel reader, teacher permission, A200 prefix, Adam/LR, sampling, masks, bounds and loss; only H parameterization adds a fixed source-footprint lowpass.',
    source_sigma_uv=prior['contract']['sigma_uv'],
    sigma_selection=prior['contract']['formula'],sigma_source='frozen C union S teacher-camera Jacobians only, no metric-based selection',
    source_filter='separable Gaussian, replicate boundary, truncate3; unmodulated H filtered before KH applies separately interpolated mH',
    unchanged=dict(L=True,A0=True,mH=True,scope=89887,low_bound=.30,high_bound=.34,
        teacher=True,geometry=True,coverage=True,static=True,vehicles=True,cameras=True,
        optimizer='P Adam low.01 high.003 betas.9/.999 eps1e-8 reg unchanged',
        losses='A1+C.5+S1 premult with fixed denominators + original effective field regularizers'),
    schedule=dict(parent_common_A200=str(R/'formal_r1/pixel_common_A200/checkpoint.pt'),
        H_initial='zero; never warm-start from M11 C800 or smoke',B=300,C=300,
        additional_physical_updates=600,logical_per_arm=800,seed=101,
        sampler='same balanced saved trace, each frame400 in logical800'),
    original_factorial_contract_and_results_immutable=True,original_formal_physical_updates=3400,
    combined_formal_physical_updates_if_complete=4000,automatic_extension=False,
    additional_fallbacks_allowed=0,no_parameter_sweep=True,no_TRELLIS_inference=True,
    lifecycle='Real CUDA forward/grad/save/restore/next B/C step + specific field serializer; separate engineering updates <=20, measured queue is minutes rather than hour-scale.',
    evaluation='Same frozen material/HF/stripe/flat-noise/visibility domains as original five; report tradeoffs, no aggregate score or automatic promotion.',
    acceptance='Claim improvement only if HF/edge fit and diagnostic artifacts jointly improve; otherwise retain negative result and identify remaining mechanism.',
    GPU_keep_on=True,power_operations_authorized=False,production_promoted=False,human_verdict=None)
path=out/'prospective_contract.json';assert not path.exists()
path.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(path)
