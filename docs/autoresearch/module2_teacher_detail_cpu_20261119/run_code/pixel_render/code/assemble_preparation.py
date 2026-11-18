"""只读汇总CPU准备，显式保留未通过的GPU/AA接线门；不运行任何模型。"""
from pathlib import Path
import json,hashlib

R=Path(__file__).resolve().parent


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cpu=load(R/'validation/cpu_acceptance.json');maps=load(R/'actual_geometry_evidence/maps_receipt.json')
    coverage=load(R/'actual_geometry_evidence/training_domain_coverage.json')
    budget=load(R/'actual_geometry_evidence/footprint_budget.json');ledger=load(R/'actual_geometry_evidence/final_ledger_adapter.json')
    compiled=load(R/'actual_geometry_evidence/compiled_pixel_operator.json');verified=load(R/'actual_geometry_evidence/compiled_pixel_verification.json')
    numeric=load(R/'actual_geometry_evidence/pixel_numeric_refinement.json');cache=load(R/'actual_geometry_evidence/cache_bytes.json')
    exact={}
    for domain in('A','C','S','fullS'):
        rows=[r for r in coverage['rows']if r['domain']==domain]
        exact[domain]={key:sum(r[key]for r in rows)for key in('total','exact','fallback','miss')}
    cap={key:sum(r['cap16_unqualified']for r in budget['rows']if r['domain']==key)for key in('A','C','S')}
    qualified=bool(compiled['completed']and verified['passed']and numeric['RGB_p99']<=1/255 and numeric['RGB_max']<=4/255)
    ready=cpu['passed']and ledger['passed']and qualified and all(r['miss']==r['fallback']==0 for r in exact.values())
    out=dict(schema='teacher_detail_pixel_render_CPU_preparation.v1',status='CPU_PREPARED_GPU_BLOCKED'if ready else'CPU_INCOMPLETE',
        CPU_core_ready=ready,native_feature_export_source_ready=True,actual_native_basis_prepared=False,
        actual_native_binary_or_adjoint_validated=False,actual_preclamp_binary_validated=False,
        CPU_actual_pixel_filter_qualified=qualified,uncapped_compiled_cache_ready=compiled['completed'],GPU_ready=False,
        native_smoke_authorized=False,formal_allowed=False,automatic_queue=False,
        GPU_executed=False,native_calls=0,optimizer_updates=0,inference_calls=0,power_actions=0,
        common_Hbound=.34,common_low_bound=.30,source_ledger=ledger['ledger'],bound_recipe=ledger['bound_recipe'],
        true_G_s_maps=dict(remote='Q/pixel_render/maps/{raw:03d}.npz',frames=[86,110,134,179,198],
            fixed_D=maps['fixed_D'],exact_D=maps['exact_D'],fallback_D=maps['fallback_D'],train_domain_coverage=exact,
            teacher_or_baseline_RGB_read_as_input=False,real_map_uses_fitted_plane=False,
            fallback='original eligible native averaged-color contribution, not fitted-plane geometry',
            XYZ_triangle_reconstruction_max=max(r['triangle_XYZ_reconstruction_max_m']for r in maps['frames']),
            ray_triangle_hit_max_m=max(r['ray_triangle_plane_hit_max_m']for r in maps['frames']),
            chart_UV_spotcheck_max=max(r['chart_spot_UV_max']for r in maps['frames'])),
        CPU_tests=dict(passed=cpu['unit_tests'],failures=cpu['failures'],errors=cpu['errors'],runtime_parity=cpu['runtime_parity']),
        ideal_oracle=cpu['marking'],alias=cpu['alias'],
        ideal_boundary='analytic synthetic fixed-weight fixtures only; not actual P/native renderer, not a real-road GT or representation-limit proof',
        sampler=dict(current='uncapped local pixel-UV box; independent nx/ny; sparse fixed material read',
            equation='B+KL*L+KH*H, valid/mH separately bilinear; H nonlinear grid construction retained',
            density=compiled['density'],compiled_pixels=sum(x['pixels']for x in compiled['frames']),compiled_samples=compiled['samples'],
            CPU_compile_seconds=compiled['CPU_seconds'],cache_bytes=cache['bytes'],cache_files=cache['files'],cache_path=cache['cache'],
            dense2x_selected_pixels=numeric['selected_pixels'],dense2x_RGB_p99=numeric['RGB_p99'],dense2x_RGB_max=numeric['RGB_max'],
            cached_read_vs_existing_dense_RGB_max=verified['RGB_max'],cached_read_mass_max=verified['mass_max'],
            historical_cap16_unqualified=cap,no_supervision_domain_shrunk=True,
            qualification_boundary='representative density2vs4 refinement plus exactness to fixed quadrature; not a global integration theorem. Native binary/CUDA sparse/reload still untested.'),
        material_floor=dict(A_full=16904,A_source_mass_zero=7,A_qualified_world_material=16897,
            C10830_S3131_fullS3187_source_mass_zero=0,action='original eligible ci fallback; fullA loss denominator unchanged',
            future_gate='actual We/gradient/error share of these7 in GPU smoke; no commonbase/scope modification',IDs=verified['A_source_floor']),
        contribution_contract=dict(scope='explicit active subset; final candidate qualification not assumed',
            road_weight='eligible native road-only alphaT',joint_weight='eligible full-scene same-sort alphaT',
            readonly='all other road and nonroad RGB remain original',joint_K='W_e[K] exactly0; never pixel-copy protected joint output',
            road_only_K='may contain road behind protected actors; diagnostic, not joint-protectionfailure',
            exactness='conditional on shared pixel material/color for eligible fragments, preclamp native RGB and fixedgeometry; not a physical subpixelvisibility integral theorem'),
        persistence=dict(renderer_mode='eligible_native_weights_pixel_world',authority='world texture + scope + geometry/source references',
            baked_average_RGB_is_not_pixel_asset=True,five_view_W_is_derived_cache=True,
            new_camera_or_edit='regenerate native contribution and true G_s map, requalify protection; stale basis rejected in CPUfixture'),
        next_GPU_smoke=['explicit new user/root authorization and lease','native constantfield/RGBunitbasis + actual same-sort decomposition',
            'joint K and readonly/source/geometry exact safeguards','actual CUDA sparse/native colour + current field pixel integration checks',
            'about10min save/eval/fullstate restore/nextstage/authoritative pixelworld asset reload'],
        evidence=['validation/cpu_acceptance.json','validation/pytest.xml','actual_geometry_evidence/maps_receipt.json',
            'actual_geometry_evidence/training_domain_coverage.json','actual_geometry_evidence/footprint_budget.json','actual_geometry_evidence/final_ledger_adapter.json',
            'actual_geometry_evidence/compiled_pixel_operator.json','actual_geometry_evidence/compiled_pixel_verification.json','actual_geometry_evidence/pixel_numeric_refinement.json','actual_geometry_evidence/cache_bytes.json'],
        owned_sources=['pixel_world_material.py','native_basis_adapter.py','prepare_cached_maps.py','test_pixel_render_cpu.py','run_cpu_evidence.py','assemble_preparation.py',
            'pixel_quadrature_stream.py','audit_pixel_quadrature_cpu.py','compile_pixel_reads_cpu.py','verify_compiled_pixel_reads_cpu.py'],
        human_verdict=None,production_promoted=False)
    (R/'preparation_receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    inventory=dict(schema='pixel_material_CPU_source_inventory.v1',source_files=[dict(path=str(R/name),relative_path=name,
        bytes=(R/name).stat().st_size,sha256=hashlib.sha256((R/name).read_bytes()).hexdigest())for name in out['owned_sources']],
        tests='validation/pytest.txt + pytest.xml',cache_bytes=cache['bytes'],large_cache_downloaded=False,native_default=False,formal_allowed=False)
    (R/'source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],CPU_tests=cpu['unit_tests'],training_domains=exact,GPU_ready=False,native_calls=0)))


if __name__=='__main__':main()
