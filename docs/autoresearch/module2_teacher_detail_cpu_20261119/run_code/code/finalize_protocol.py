from pathlib import Path
import json,shutil
W=Path(__file__).resolve().parent
p=json.loads((W/'protocol_draft.json').read_text(encoding='utf-8'))
b=json.loads((W/'bound_recipe.json').read_text(encoding='utf-8'))
p.update(state='CPU_PREPARED_STOP_BEFORE_GPU',frozen_before_GPU=True,GPU_experiments_executed=False)
p['common_inputs']['scope']='89887 CPU candidate color-write rows; original84266 all retained +5621 independently qualified rows; actual native/K qualification is a mandatory future smoke gate'
p['common_inputs']['base']='all original valid A0 values immutable; only1267 qualified missing-base nodes receive frozen low-frequency teacher seed (Gaussian sigma12cm), separately labelled'
p['field'].pop('high_bound_candidate',None);p['field'].pop('HF_bound_status',None)
p['field'].update(high_bound=.34,bound_recipe=b,
    teacher_gate='bool independently qualified teacher-HF support; HF4code consistency and clean-realA-HF protection; RGB differences remain diagnostic',
    rejected_teacher_HF_support='147 world nodes; no deletion from original C/S losses or evaluation',
    original_provenance_updated=False)
p['reader_rules']['average']='qualified GH31x31 physical tangent covariance -> exact fixed sparse K -> derived ci -> actual native GS; no new geometry or colors per frame'
p['reader_rules']['pixel_footprint']='uncapped per-axis nx,ny adaptive pixel-box quadrature using actual canonical G_s UV derivatives; full5 compiled6,093,288 samples, density2, no full Gaussian Sigma'
p['reader_rules']['static_lowering']='given fixed geometry/valid/mH, material reads exactly factor B+KL*L+KH*H; KL/B shared across permission arms, KH includes separately interpolated mH; CPU forward and gradients checked'
p['reader_rules']['nonmaterial_fallback']='7 of16904 real-A pixels have zero common source mass, retain original eligible-ci contribution; report fullA16904 and qualifiedA16897, no loss denominator shrinking. C/S/fullS source-mass0 pixels=0'
p['arms'][-1].update(role='bounded_projected_HF_initialization_only',primary_four_arms_depend_on_it=False,
    strict_faithful_transfer_gate=False,exploratory_seed_valid=True,
    interpretation='partial realized prior only; negative contrast cannot reject TRELLIS capacity',
    seed_readiness='finite bounded nonzero encoding, pure-outside protection and unchanged base; faithful source transfer gate remains recorded FAIL')
p['proposed_future_GPU_budget'].update(primary_four_formal_physical_updates=2800,secondary_extra_physical_updates=600,
    smoke='first real GPU short lifecycle with native basis/constant field/CSR CUDA gradients/save-eval-restore-next-stage/persistence; if measured queue is hour-scale, require about10min as project rule',
    timing='equal update budget, not equal GPU time; benchmark actual sparse runtime during smoke, no assumed speed-up used as evidence')
p['evaluation']['historical_reference']='previous P independentRGB/.04 controls are historical only; new common bounds/base/scope differ, never label that historical comparison a single-factor causal gain'
p['CPU_evidence_summary']=dict(old_H=dict(C_rows=47886,C_full_kernel_zero=25103,S_rows=6707,S_full_kernel_zero=170,
    weighted_C_gate=.06675339956230941,weighted_S_gate=.2313222576706251,weight_definition='saved rowmax native contribution across86/110; not per-pixel or joint contribution'),
    teacher_material_nodes=34416,teacher_H_allowed_nodes=34269,new_H_on_old_zero_nodes=25378,
    common_write_rows=89887,common_base_seed_nodes=1267,
    average=dict(kernel_order=31,points_per_row=961,queries=86381407,source_chart_world_error_max_m=9.09e-13,CSR_dense_RGB_max=1.33e-15),
    pixel=dict(canonical_pixels=122604,quadrature_samples=6093288,density=2.0,source_quadrature_RGB_p99=.00131081,source_quadrature_RGB_max=.00134760,CSR_dense_RGB_max=1.81e-14),
    geometry=dict(train_A_C_S_fullS_all_canonical=True,original_D=38416,D_fallback=218,ideal_RGB_or_marking_labels_used=False),
    T2=dict(strict_faithful_gate=False,exploratory_only=True,energy_fraction=.5805223971791532,cosine=.8280024393499665,effective_HF_RMS=.0033478752134889706,effective_HF_abs_max=.01999998064511086,legal_prior_nodes_DC_abs_max=3.1116949269685046e-5),
    scene_optimizer_updates=0,native_forward_calls=0,native_backward_calls=0,new_model_inferences=0,power_operations=0)
p['future_GPU_smoke_gates']+=['CUDA sparse forward/adjoint matches qualified CPU reads; sparse cache is derived, field parameters remain authority',
    'joint K hard gate only; road-only K may contain road hidden behind a kept vehicle, report separately',
    '7A fallback floors and world source-mass gaps remain explicitly counted; no simulated native visibility claims']
p['reader_rules']['row_identity']='average cache row_ids are global full-topology world rows and must map to active positions; pixel cache row_ids are camera-local flattened pixels, may be grouped/permuted and must scatter, never reshape directly'
(W/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('frozen CPU protocol; all GPU authorization remains false')
