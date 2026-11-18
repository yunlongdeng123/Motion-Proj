"""从已冻结许可记录提取B32候选；伪教师有效颜色不覆盖观测台账。"""
from pathlib import Path
import argparse,json,os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np


def take(path,keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files if keys is None or k in keys}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();N=args.campaign
    review_path=N/'supervision_review_r1/teacher_permission_review.npz'
    parent=N/'native_r1/asset_A1_guarded'
    r=take(review_path)
    selector=r['associated_with_remaining15_thin']&r['permission_qualified']
    ids=r['world_rows'][selector]
    if len(ids)!=32 or len(np.unique(ids))!=32:raise ValueError('B32 selector changed')
    road=take(parent/'road.npz',['active','reference_srgb','minimum_model_depth','gaussian_sigma_m','road_opacity'])
    p=take(parent/'provenance.npz',['known','unknown','source','surface_supported','guarded_world_rows',
        'observed_reference_srgb','legal_write','coverage_requested'])
    if road['active'][ids].any():raise ValueError('B32 already active in parent')
    if p['known'][ids].any() or not p['unknown'][ids].all() or (p['source'][ids]!=0).any():raise ValueError('B32 must stay unknown in immutable observed ledger')
    if not p['surface_supported'][ids].all() or p['guarded_world_rows'][ids].any() or (road['minimum_model_depth'][ids]>0).any():raise ValueError('parent physical/guard evidence changed')
    rgb=r['candidate_effective_rgb_all118'][selector].copy()
    if not np.isfinite(rgb).all() or ((rgb<0)|(rgb>1)).any():raise ValueError('invalid effective pseudo RGB')
    if not (r['selected_source_raw'][selector]==110).all() or not (r['selected_source_role'][selector]==3).all():raise ValueError('frozen S pseudo source changed')
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'activation_pseudo_init.npz'
    if target.exists():raise FileExistsError('preserve old B32 input')
    proposed=dict(proposed_world_rows=ids,proposed_effective_rgb_sparse=rgb,
        parent_effective_rgb_sparse=road['reference_srgb'][ids].copy(),
        material_init_required_sparse=np.any(rgb!=road['reference_srgb'][ids],axis=1),
        pseudo_source_raw=r['selected_source_raw'][selector],pseudo_source_uv_native=r['selected_source_uv_native'][selector],
        pseudo_source_role=r['selected_source_role'][selector],pseudo_source_permission_q=np.ones(32,np.float32),
        original_known_sparse=p['known'][ids],original_unknown_sparse=p['unknown'][ids],original_source_sparse=p['source'][ids],
        original_observed_reference_srgb_sparse=p['observed_reference_srgb'][ids],
        original_legal_write_sparse=p['legal_write'][ids],original_coverage_requested_sparse=p['coverage_requested'][ids],
        original_sigma_sparse=road['gaussian_sigma_m'][ids],original_opacity_sparse=road['road_opacity'][ids],
        source_world_xyz=r['world_xyz'][selector],
        both_teachers_valid=r['both_sources_valid'][selector],
        crossview_abs_RGB_delta=r['crossview_abs_RGB_delta'][selector],
        crossview_max_channel_delta=r['crossview_max_channel_delta'][selector],
        prior_ledger_must_stay_immutable=np.asarray(True),native_newrow_K_qualified=np.asarray(False),
        material_role=np.asarray('effective target-shadow S pseudo initialization; old known=false/source=0 unchanged'),
        world_row_id_semantics=np.asarray('original complete road.npz row index; not active-local'))
    np.savez_compressed(target,**proposed)
    both=proposed['both_teachers_valid'];delta=proposed['crossview_abs_RGB_delta']
    receipt=dict(schema='road_activation_B32_pseudo_init.v1',parent_asset=str(parent),source_permission_review=str(review_path),
        exact_selector='world_rows[associated_with_remaining15_thin & permission_qualified]',proposed_world_rows=32,
        full118_other_permitted_not_activated=81,source_view_failed_not_activated=5,
        related_fixed_remaining_pixels=15,other_remaining_thin_pixels=58,
        material_source='frozen original raw110 B/raw_native.png at stored Brown-calibrated bilinear UV; target-shadow pseudo, never observed GT',
        material_source_raw=110,old_known_false=32,old_unknown_true=32,old_source_zero=32,old_legal_route_false=32,
        surface_supported=32,parent_near_conflict=0,parent_guard_conflict=0,
        crossview_both_valid=int(both.sum()),crossview_RGB_meanabs=float(delta[both].mean()),crossview_RGB_maxabs=float(delta[both].max()),
        pseudo_RGB_range=[float(rgb.min()),float(rgb.max())],
        scope='ONLY these32 inactive parent rows may be considered by registered B32 builder; no other row/color/geometry/opacity/sigma edits',
        source_conflict_resolved=False,source_conflict_policy='fixed one-source candidate, no view averaging or GT claim',
        prior_ledger_policy='keep known/unknown/source/observed_reference_srgb/source UV and all raw records exact; record effective pseudo separately',
        needed_next='actual new packet native K/near envelope, full5frame K77266 exact, native support attribution, persistence/public reload',
        native_newrow_K_qualified=False,proposal_only=True,GPU_operations=0,optimizer_updates=0,old_assets_modified=False,human_verdict=None)
    (args.output/'activation_pseudo_init.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
