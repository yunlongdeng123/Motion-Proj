"""CPU breakdown of full 2D C footprint; protected pixels are not road holes."""
from pathlib import Path
import argparse,hashlib,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from PIL import Image


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def partition(domain,reasons):
    remaining=domain.copy();masks={}
    for key,mask in reasons:
        masks[key]=remaining&mask;remaining&=~mask
    masks['unclassified_remaining']=remaining
    if sum(int(x.sum()) for x in masks.values())!=int(domain.sum()):raise ValueError('partition changed the denominator')
    return masks


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only audit requires CUDA_VISIBLE_DEVICES empty')
    ap=argparse.ArgumentParser();ap.add_argument('--campaign-root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);base=args.campaign_root;rows=[];sources=[]
    for raw in [86,110]:
        paths=dict(r3=base/'road_direct_teacher_20261116_r1/supervision_r3'/f'{raw:03d}.npz',
            projection=base/'road_asset_20261116_r1/phase0/supervision_triangle_r8'/f'{raw:03d}.npz',
            legacy_view=base/'waymo_val065_road_completion_20261111_r1/multiview008_r2/world_road/views'/f'{raw:03d}.npz',
            candidate=args.candidate/f'{raw:03d}.npz')
        d,f,v,new=[load(paths[k]) for k in ['r3','projection','legacy_view','candidate']]
        full=d['C_full_hole'];missing=full&~f['surface_intersection_valid'];depth=v['ground_depth']
        near_or_outside=~v['inside']|~np.isfinite(depth)|(depth<=.2)|(depth>60)
        retained=d['K']|d['retained_projection_guard'];raw_teacher_invalid=d['q_W_native_footprint']<1-1e-7
        reasons=dict(retained_K=d['K'],retained_projection_guard=d['retained_projection_guard'],
            no_existing_road_mesh_intersection=~f['surface_intersection_valid'],no_cached_geometry_support=~f['geometry_support'],
            legacy_ray_outside_chart=~v['inside'],legacy_ray_near_or_outside_range=near_or_outside,
            legacy_retained_reveal_guard=v['retained_reveal_guard'],legacy_retained_box_guard=v['retained_box_guard'],
            raw_teacher_incomplete_native_write_footprint=raw_teacher_invalid,
            F0_color_unavailable=~f['legal_color'],physical_native_mixed_boundary=~d['C_full_core'],
            fixed_reference_low_alpha=~d['fixed_high_alpha'],non_target_shadow_prediction=new['S_other_shadow_guard'])
        overlap={key:dict(in_full_2d_C=int((full&mask).sum()),in_missing_world=int((missing&mask).sum())) for key,mask in reasons.items()}
        missing_parts=partition(missing,[('protected_retained_projection_or_K',retained),
            ('legacy_near_or_outside',near_or_outside),('original_raw_teacher_write_unavailable',raw_teacher_invalid)])
        missing_parts['unprotected_mesh_miss_unresolved']=missing_parts.pop('unclassified_remaining')
        full_parts=partition(full,[('protected_retained_projection_or_K',retained),('v4_C_geometry_qualified',new['C_hole']),
            ('unprotected_no_existing_mesh_hit',~f['surface_intersection_valid']),('legacy_near_or_outside_unqualified_only',near_or_outside),
            ('original_raw_teacher_write_unavailable',raw_teacher_invalid),('mixed_physical_native_footprint',~d['C_full_core']),
            ('non_target_shadow_prediction_conflict',new['S_other_shadow_guard'])])
        if int(full_parts['v4_C_geometry_qualified'].sum())!=int(new['C_hole'].sum()):raise ValueError('legacy proxy overrode current valid geometry domain')
        row=dict(raw=raw,full_2d_physical_C_footprint=int(full.sum()),missing_world_triangle=int(missing.sum()),
            overlapping_reasons=overlap,overlap_counts_are_not_additive=True,
            missing_world_exclusive_priority_counts={k:int(x.sum()) for k,x in missing_parts.items()},
            full_2d_C_exclusive_priority_counts={k:int(x.sum()) for k,x in full_parts.items()},
            no_mesh_hit_does_not_prove_recoverable_road=True,
            protected_regions_are_not_requested_road_restoration=True,
            legacy_ray_diagnostics_are_not_new_triangle_geometry=True)
        np.savez_compressed(args.output/f'{raw:03d}_classification.npz',full_2d_C=full,missing_world=missing,
            **{'missing_world_'+k:mask for k,mask in missing_parts.items()},**{'full_'+k:mask for k,mask in full_parts.items()})
        rgb=np.uint8(np.floor(d['target_A_srgb'].transpose(1,2,0)*255+.5)).astype(np.float32)
        for mask,color in [(missing_parts['protected_retained_projection_or_K'],[255,70,100]),
            (missing_parts['unprotected_mesh_miss_unresolved'],[255,200,40])]:rgb[mask]=.45*rgb[mask]+.55*np.asarray(color)
        Image.fromarray(np.uint8(np.clip(np.floor(rgb+.5),0,255))).save(args.output/f'{raw:03d}_missing_world_overlay.webp',lossless=True,method=4)
        rows.append(row);sources.extend(dict(role=k,path=str(p),sha256=sha(p)) for k,p in paths.items())
        print(json.dumps(row,ensure_ascii=False),flush=True)
    parent=args.candidate.parent;old=read(parent/'supervision_cpu_acceptance_r1/acceptance.json');new=read(parent/'supervision_cpu_acceptance_r2/acceptance.json')
    correction=dict(schema='road_capacity_comparator_correction.v1',prior_status=old['status'],corrected_status=new['status'],
        prior_failure='np.array_equal treats matching NaN invalid-world entries as unequal',
        corrected_comparison='array shape, dtype and bytes including NaN payload must match exactly',
        old_and_new_candidate_freeze_same=old['candidate_freeze_sha256']==new['candidate_freeze_sha256'],
        source_and_candidate_bindings_unchanged_in_both=old['checks']['all_frozen_bindings_unchanged'] and new['checks']['all_frozen_bindings_unchanged'],
        candidate_data_modified=False,first_failure_artifacts_preserved=True)
    write(args.output/'comparator_correction_note.json',correction)
    write(args.output/'full_C_missing_world_breakdown.json',dict(schema='road_capacity_full_C_missing_world_breakdown.v1',rows=rows,source_files=sources,
        exclusive_priority='retained protection first; then legacy near/outside; raw teacher-write invalid; unprotected unresolved mesh miss',
        interpretation='full C is a 2D source footprint that can include pixels occupied by retained geometry after target removal; it is not a verified clean-road restoration domain',
        F0_unavailable_is_distinct_from_original_raw_teacher_unavailable=True,all_denominators_preserved=True,
        GPU_operations=0,training_updates=0,render_operations=0))


if __name__=='__main__':main()
