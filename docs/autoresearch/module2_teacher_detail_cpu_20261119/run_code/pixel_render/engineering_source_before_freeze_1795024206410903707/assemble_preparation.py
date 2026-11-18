"""只读汇总CPU准备，显式保留未通过的GPU/AA接线门；不运行任何模型。"""
from pathlib import Path
import json

R=Path(__file__).resolve().parent


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cpu=load(R/'validation/cpu_acceptance.json');maps=load(R/'actual_geometry_evidence/maps_receipt.json')
    coverage=load(R/'actual_geometry_evidence/training_domain_coverage.json')
    budget=load(R/'actual_geometry_evidence/footprint_budget.json');ledger=load(R/'actual_geometry_evidence/final_ledger_adapter.json')
    exact={}
    for domain in('A','C','S','fullS'):
        rows=[r for r in coverage['rows']if r['domain']==domain]
        exact[domain]={key:sum(r[key]for r in rows)for key in('total','exact','fallback','miss')}
    cap={key:sum(r['cap16_unqualified']for r in budget['rows']if r['domain']==key)for key in('A','C','S')}
    ready=cpu['passed']and ledger['passed']and all(r['miss']==r['fallback']==0 for r in exact.values())
    out=dict(schema='teacher_detail_pixel_render_CPU_preparation.v1',status='CPU_PREPARED_GPU_BLOCKED'if ready else'CPU_INCOMPLETE',
        CPU_core_ready=ready,native_feature_export_source_ready=True,actual_native_basis_prepared=False,
        actual_native_binary_or_adjoint_validated=False,actual_scene_pixel_filter_qualified=False,GPU_ready=False,
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
        sampler=dict(current_CPU_candidate='local pixel-UV box, up to16 samples/axis; no full Gaussian Sigma',
            cap16_unqualified=cap,no_supervision_domain_shrunk=True,
            pending='Mip/EWA or adaptive integration engineering + actual-field convergence, required before formal; unqualified sampling is not capacityfailure'),
        contribution_contract=dict(scope='explicit active subset; final candidate qualification not assumed',
            road_weight='eligible native road-only alphaT',joint_weight='eligible full-scene same-sort alphaT',
            readonly='all other road and nonroad RGB remain original',joint_K='W_e[K] exactly0; never pixel-copy protected joint output',
            road_only_K='may contain road behind protected actors; diagnostic, not joint-protectionfailure',
            exactness='conditional on shared pixel material/color for eligible fragments, preclamp native RGB and fixedgeometry; not a physical subpixelvisibility integral theorem'),
        persistence=dict(renderer_mode='eligible_native_weights_pixel_world',authority='world texture + scope + geometry/source references',
            baked_average_RGB_is_not_pixel_asset=True,five_view_W_is_derived_cache=True,
            new_camera_or_edit='regenerate native contribution and true G_s map, requalify protection; stale basis rejected in CPUfixture'),
        next_GPU_smoke=['explicit new user/root authorization and lease','native constantfield/RGBunitbasis + actual same-sort decomposition',
            'joint K and readonly/source/geometry exact safeguards','actual field pixel filter numerical qualification',
            'about10min save/eval/fullstate restore/nextstage/authoritative pixelworld asset reload'],
        evidence=['validation/cpu_acceptance.json','validation/pytest.xml','actual_geometry_evidence/maps_receipt.json',
            'actual_geometry_evidence/training_domain_coverage.json','actual_geometry_evidence/footprint_budget.json','actual_geometry_evidence/final_ledger_adapter.json'],
        owned_sources=['pixel_world_material.py','native_basis_adapter.py','prepare_cached_maps.py','test_pixel_render_cpu.py','run_cpu_evidence.py','assemble_preparation.py'],
        human_verdict=None,production_promoted=False)
    (R/'preparation_receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],CPU_tests=cpu['unit_tests'],training_domains=exact,GPU_ready=False,native_calls=0)))


if __name__=='__main__':main()
