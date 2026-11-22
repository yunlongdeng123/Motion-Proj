"""Fixed-denominator review of two actual zero-step controls; no promotion."""
from pathlib import Path
import argparse,shutil,zipfile,sys
import numpy as np
from r10_common import read,load,write,sha,contracts
from local_row_control import metrics,pin


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.root;cfg,base=contracts(r/'local_measure_contract_v2.json');a.output.mkdir(parents=True,exist_ok=False)
    stages=dict(baseline=r/'measure_v1',strict_zero=r/'retire0_strict_v1',body_overlap_zero=r/'body_overlap_diagnostic_v1')
    receipts={key:read(path/'receipt.json') for key,path in stages.items()};assert all(q['complete'] for q in receipts.values())
    assert receipts['baseline']['actual_nativeF']==5 and receipts['baseline']['nativeB']==5
    assert all(receipts[k]['nativeF']==5 and receipts[k]['updates']==0 for k in ['strict_zero','body_overlap_zero'])
    preview=a.output/'original_preview_pngs';preview.mkdir();images=[];rows=[]
    for raw in cfg['frames']:
        ev=load(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz');reference=load(stages['baseline']/f'{raw:03d}/baseline_joint.npz')
        source={key:(load(directory/f'{raw:03d}/baseline_joint.npz') if key=='baseline' else load(directory/f'{raw:03d}/joint_render.npz')) for key,directory in stages.items()}
        frame=dict(raw=raw,stages={},fixed_original_band_pixels=int(ev['full_original_band'].sum()),
            certified_ground_pixels=int(ev['ground_eval'].sum()),ground_target_unknown_pixels=int((ev['ground_band_before_support']&~ev['ground_eval']).sum()))
        for name,z in source.items():
            K=ev['K_protected'];assert z['rgb'][:,K].tobytes()==reference['rgb'][:,K].tobytes() and z['alpha'][K].tobytes()==reference['alpha'][K].tobytes()
            if name=='strict_zero':
                B=ev['body_protected'];assert z['rgb'][:,B].tobytes()==reference['rgb'][:,B].tobytes() and z['alpha'][B].tobytes()==reference['alpha'][B].tobytes()
            m=metrics(z['rgb'],z['alpha'],reference,ev);m['fixed_colour_masks']={}
            for label in ['red_diagnostic','blue_cyan_diagnostic']:
                d=ev[label];rgb=z['rgb']
                condition=(rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12) if label=='red_diagnostic' else (rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08)
                m['fixed_colour_masks'][label]=dict(original_pixels=int(d.sum()),remaining_in_same_fixed_mask=int((condition&d).sum()),
                    alpha_lt95=int((d&(z['alpha']<.95)).sum()),target_unknown_pixels=int((d&~ev['ground_eval']).sum()),
                    mean_RGB=rgb[:,d].mean(1).tolist() if d.any() else None)
            if name=='body_overlap_zero':m['body_comparison']=read(stages[name]/f'{raw:03d}/metrics.json')['body_comparison']
            frame['stages'][name]=m
            for image_name in ['shared','alpha']:
                src=stages[name]/f'{raw:03d}/{image_name}.png';dst=preview/f'{raw:03d}_{name}_{image_name}.png';shutil.copy2(src,dst)
                assert sha(src)==sha(dst);images.append(dict(raw=raw,stage=name,kind=image_name,original=pin(src),copy=pin(dst)))
        for name in ['original_A0_joint','RGB_change_display']:
            src=stages['body_overlap_zero']/f'{raw:03d}/{name}.png';dst=preview/f'{raw:03d}_{name}.png';shutil.copy2(src,dst);assert sha(src)==sha(dst)
            images.append(dict(raw=raw,stage='body_overlap_zero',kind=name,original=pin(src),copy=pin(dst)))
        rows.append(frame);write(a.output/f'{raw:03d}_fixed_scope_comparison.json',frame)
    write(preview/'manifest.json',dict(schema='R10.original_zero_control_PNG_copies.v1',complete=True,files=images,no_resize_or_processing=True,nativeF=0,nativeB=0,updates=0))
    result=dict(schema='R10.zero_step_old_representation_diagnostic_review.v1',complete=True,CPU_only=True,new_GPU_F=0,new_GPU_B=0,
        actual_stage_resources=dict(measure=dict(F=5,B=5,updates=0),strict_zero=dict(F=5,B=0,updates=0),body_overlap_zero=dict(F=5,B=0,updates=0)),
        R10_total_actual=dict(nativeF=15,nativeB=5,formal_updates=0),R8_prior_history=dict(nativeF=970,nativeB=833,formal_updates=400,discarded_smoke_updates=13),
        cumulative_formal_existing_representation=400,formal_existing_representation_ceiling=2000,new_600_phase_not_started=True,
        accepted_actor_original_SHA=cfg['accepted_actor_sha256'],source_field_sha256=cfg['source']['sha256'],
        strategy_receipts={key:pin(path/'receipt.json') for key,path in stages.items()},
        strict_candidate_proof=pin(r/'measure_v1/native_gate0.proof.json'),body_overlap_scope=pin(r/'body_overlap_policy_v1/scope.json'),
        per_view_comparisons=[pin(a.output/f'{raw:03d}_fixed_scope_comparison.json') for raw in cfg['frames']],preview_manifest=pin(preview/'manifest.json'),
        candidate_world_locator=[pin(r/f'body_overlap_diagnostic_v1/{raw:03d}/retired_prepared_geometry.npz') for raw in cfg['frames']],
        all_other_K_exact=True,strict_body_exact=True,body_overlap_occlusion_change_quantified=True,
        full_band_and_unknown_target_denominators_retained=True,fixed_original_red_blue_masks_never_reselected=True,
        not_a_local_rebuild=True,no_new_geometry=True,no_formal_optimizer_updates=True,production_asset_ready=False,quality_accepted=False,human_verdict=None,
        next_phase='Root reviews actual joint/body/ground contact and complete-band alpha before selecting any gate fit or separately qualified new local geometry')
    write(a.output/'review_summary.json',result)
    # Small PNG/JSON package first; large all-context and full float rasters stay remote.
    package=a.output/'original_PNG_JSON_small.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(a.output.rglob('*')):
            if path.is_file() and path!=package:archive.write(path,path.relative_to(a.output))
    assert 'torch' not in sys.modules
    print(dict(complete=True,summary=pin(a.output/'review_summary.json'),preview_PNGs=len(images),small_package=pin(package)))


if __name__=='__main__':main()
