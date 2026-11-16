"""汇总CPU能力检查准备；只写计划，不含GPU执行路径或后台控制器。"""
from pathlib import Path
import json

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')


def read(path):return json.loads(Path(path).read_text())


def main():
    targets=ROOT/'ideal_targets_r2';data=read(targets/'receipt.json')
    audit=read(ROOT/'qa/ideal_targets_cpu_acceptance.json');toy=read(ROOT/'qa/bounded_ls_toy.json')
    protocol=read(ROOT/'protocol_cpu_prepared.json')
    out=ROOT/'check2';out.mkdir(exist_ok=True)
    plan=dict(schema='fixed_native_RGB_capacity_plan.v1',status='CPU_PREPARED_WAIT_USER_GPU',
        execution_authorized=False,GPU_executed=False,real_scene_render_calls=0,actual_model_training_updates=0,
        background_controller_created=False,automatic_resume=False,native_backend_imported=False,
        canonical_asset=data['source_asset'],fixed=['geometry','opacity','all scales','active/topology/density','cameras','480x320'],
        target_pack=str(targets),target_freeze=str(targets/'freeze.json'),all_five_views_supervised=True,
        real_data_roles='separate check3 retains two train and three diagnose; no oracle generalization claim',
        field='independent analytic world asphalt and flat-top smooth-edge curved dashed paint on frozen G_s mesh',
        main_domain=38416,keep_domain=77266,plane_fallback_pixels=218,
        negative_capacity_evidence=dict(exact_G_s_pixel_frames=38198,fallback_pixel_frames=218,
            fallback_role='full-domain reporting / geometry-inconclusive only; never negative model-capacity evidence',
            native_operator_linearity_and_adjoint_required=True,lower_bound_safety_margin=1e-5,
            decision='valid exact-G_s lower f-G minus1e-5 > registered objective tolerance; boundary cases inconclusive',
            denominator_policy='full38416 unchanged; exact-G_s/fallback are predeclared evidence strata'),
        sampling='pixel-center main targets; ≥3px width/on/off for detail stratum; no full antialias reference claimed',
        cases=[case['name'] for case in read(targets/'field_config.json')['cases']],stress_case_never_a_failure_criterion=True,
        primary_scope=data['scopes'],conditional_scope='all active road colors only, no geometry edits, no keep/production claim',
        fixed_contribution=data['fixed_contribution'],
        mathematical_operator=dict(prediction='P0 + A_E*(c_E-c0_E)',fixed_term='b=P0-A_E*c0_E',
            variables='independent native-sRGB ci in [0,1]',normalization='f=.5 weighted RGB SSE/(3*sum q)',
            gradient='native matrix-free adjoint of weighted residual with identical normalization',
            certificate='G=sum(g*c-min(g,0)); upper=f, lower=max(0,f-G-error_allowance)',
            certificate_scope='independent fixed-linear ci box only, not W4+clip or full model/prior',
            CPU_exact_frozen_b_available=False,baseline_P0_role='archived forward reference only; never teacher'),
        future_native_preflight=['explicit user GPU enable and resume; environment/source revalidation',
            'same fixed canonical geometry/alpha/sigma/active and fresh P0 agreement',
            'native linearity and forward/adjoint dot identity with recorded numerical error',
            'actual eligible/frozen contribution and fixed b verification; alpha remains detached/fixed',
            'full primary joint K unchanged; relaxed oracle explicitly road-only',
            'native save/eval/fullrestore/next-update replay and persistent loader entry'],
        convergence_rules=['feasible RGB+resolved-detail solution witnesses capacity without needing a global optimum',
            'failure needs lower bound above registered objective tolerance including numerical allowance',
            'small gap alone, Adam plateau, time cap or optimizer failure does not establish a representation limit',
            'a protected-scope floor must be separated by conditional relaxed color-only oracle',
            'fallback/sampling mismatch is geometry or sampling inconclusive, never evidence against a generation prior'],
        registered_reference=protocol['check2']['error_envelope'],work_cap=protocol['check2']['work_cap'],
        launch='No executable GPU entry is provided here. The project CPU entry hard-refuses --execute before backend import.',
        next_action='stop and wait for user GPU authorization')
    (out/'native_operator_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    summary=dict(schema='check2_CPU_preparation_acceptance.v1',status='READY_CPU_ONLY_WAIT_USER_GPU',
        teacher_data_acceptance=audit['CPU_preparation_status'],teacher_checks=len(audit['checks']),
        bounded_LS_toy_status=toy['status'],toy_checks=len(toy['checks']),
        native_capacity_verdict='not_evaluated',GPU_executed=False,actual_model_training_updates=0,
        real_scene_render_calls=0,background_controller_created=False,threads=1,
        resource=dict(memory_limit_GiB=2,CPU_quota_cores=.5,independent_audit_peak_MiB=audit['process_peak_RSS_MiB']),
        target_pack=str(targets),plan=str(out/'native_operator_plan.json'),
        fixed_evaluation_pixels=38416,keep_pixels=77266,plane_fallback_pixels=218,
        negative_capacity_exact_G_s_pixels=38198,negative_lower_bound_safety_margin=1e-5,
        negative_evidence_rule='native numerical verification then exact-G_s lower f-G-1e-5 > objective tolerance; fallback and boundary cases inconclusive',
        scope=data['scopes'],teacher_source='independent canonical G_s mesh hit plus explicit plane fallback; analytic world RGB, not GS RGB',
        case_sampling=[dict(raw=row['raw'],cases=row['cases']) for row in data['rows']],
        error_boundaries=['global plane differs materially from G_s; only 218 declared fallback points use it',
                          'main pixel-center convention and resolvable detail strata are explicit',
                          'fixed baseline RGB is a forward offset, not synthetic truth',
                          'CPU eligibility/center-triangle guard flags are not actual native alphaT attribution'],
        human_verdict=None,failure_ledger_delta='none; preparation only')
    (out/'cpu_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    review=dict(schema='independent_three_gate_CPU_review.v1',status='logic_ready_wait_GPU_no_scientific_verdict',
        check1='supervision sufficiency/ownership/coverage first; missing or conflicting teacher is not model-capacity evidence',
        check2='ideal all-five supervised oracle, bounded-ci forward witnesses and certified lower bounds; primary constraint floor separated from relaxed operator capacity',
        check3='real two-train/three-diagnostic matched I0/W4, same legal information/coverage/budget; lower stripes alone is insufficient if fit/detail is lost',
        critical_boundaries=['FW gap small alone is insufficient: adjusted lower f-G must exceed registered tolerance for a negative claim',
                            'W4 shared-node bounds and shader clipping are additional function/optimization constraints; ci box certificate does not transfer',
                            'if I0 delta-striping energy<=0, percentage reduction is not applicable; no shared-basis necessity is established',
                            '218 plane fallbacks and undersampled stress never certify an intrinsic representation failure',
                            'negative evidence uses38198 exact-G_s pixels only, valid native operator, and1e-5 lower-bound safety margin',
                            'all-five ideal fitting is not held-out or novel-view generalization',
                            'GPU resource control requires explicit new user resume; no automatic start or queue'],
        candidate_switch='no drop-in larger-generator conclusion from these CPU preparations',
        GPU_executed=False,actual_model_training_updates=0,human_verdict=None)
    (ROOT/'qa/three_gate_independent_review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],targets=str(targets),teacher_checks=summary['teacher_checks'],
                          toy_checks=summary['toy_checks'],GPU_executed=False)))


if __name__=='__main__':main()
