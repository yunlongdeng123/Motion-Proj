"""Compact17-frame physical height readout, distinct from geometry activation."""
import argparse,json
from pathlib import Path
from diagnose_nonplanar_inference_cpu import read,write,pin


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args()
    R=a.campaign_root/'road_underbody_local_20261123_r10';D=R/'audit/timewindow17_geometry_v1/loo_selection'
    sp=D/'training_only_CV_selection.json';s=read(sp)
    fp=D/'training_fold_original_height_audit_v2.json';f=read(fp);old=read(D/'training_fold_original_height_audit.json')
    assert len(s['train_frames'])==17
    rows=[]
    for model,oldmodel in zip(f['models'],old['models']):
        assert {k.replace('all8_','all17_'):v for k,v in oldmodel.items()}==model
        assert len(model['frames'])==17
        rows.append(dict(name=model['name'],context_all17=model['all17_LOO_context_height_pass'],
            actual_all17=model['all17_LOO_actual_requested_height_pass'],fitted_actual_all17=model['all17_fitted_actual_requested_height_pass'],
            worst_LOO_actual_median_m=max(x['LOO_actual_requested']['absolute_height_m']['median'] for x in model['frames']),
            worst_LOO_actual_p95_m=max(x['LOO_actual_requested']['absolute_height_m']['p95'] for x in model['frames']),
            worst_LOO_context_median_m=max(x['LOO_context']['absolute_height_m']['median'] for x in model['frames']),
            worst_LOO_context_p95_m=max(x['LOO_context']['absolute_height_m']['p95'] for x in model['frames'])))
    chosen=next(x for x in rows if x['name']==s['one_SE_preferred_model'])
    target=R/'audit/timewindow17_geometry_v1/training_physical_gate_summary.json';assert not target.exists()
    result=dict(schema='R10.timewindow17_training_physical_height_summary.v1',complete=True,
        decision='PASS_TRAINING_HEIGHT_ONLY' if all(chosen[k] for k in ['context_all17','actual_all17','fitted_actual_all17']) else 'FAIL',
        selection=pin(sp),all_original_point_height_audit=pin(fp),legacy_field_label_correction_numbers_byteequivalent=True,
        selected=chosen,models=rows,original_frame_pool_unchanged=s['train_frames'],all17_retained=True,
        heldout_opened_here=False,geometry_accepted_for_zero_step=False,GPU_forward=0,backward=0,updates=0)
    write(target,result);print(json.dumps(dict(receipt=pin(target),**result)),flush=True)


if __name__=='__main__':main()
