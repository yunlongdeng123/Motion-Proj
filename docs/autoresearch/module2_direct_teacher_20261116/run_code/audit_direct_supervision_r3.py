"""只读独立重建native足迹、A/C/q/outside与旧phase2共同评价；不生成训练输入。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1')
OLD=ROOT.parent/'road_asset_20261116_r1'


def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
def read(path):return json.loads(Path(path).read_text())
def pin_matches(row):return hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
def block(value,hw,operation='mean'):
    h,w=hw;H,W=value.shape[:2]
    if H%h or W%w:raise ValueError('Noninteger native footprint')
    a=value.reshape(h,H//h,w,W//w,*value.shape[2:])
    return getattr(a,operation)(axis=(1,3))


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=read(args.supervision/'manifest.json');freeze=read(args.supervision/'freeze.json')
    rows=[]
    for row in manifest['rows']:
        raw=row['raw'];s=load(args.supervision/f'{raw:03d}.npz');paths={key:Path(value['path']) for key,value in row['source_files'].items()}
        native_obs=np.asarray(Image.open(paths['observation']).convert('RGB'))
        native_teacher=np.asarray(Image.open(paths['teacher_C']).convert('RGB'))
        roles=load(paths['roles']);guard=load(paths['guard']);common=load(paths['common']);oldF=load(paths['F_control'])
        init=load(paths['initial']);view=load(paths['view'])
        native_road=np.asarray(Image.open(paths['road_observed']).convert('L'))>127
        W,K,H=guard['W_guard'],guard['K'],guard['H_B'];physical=common['physical']
        Anative=native_road&roles['C_mat']&roles['visible_support']&~H&~K
        Cnative=physical&W&~K;effect=W&~physical&~K
        shape=s['fixed_initial_alpha'].shape
        Afrac=block(Anative.astype(np.float64),shape);Cfrac=block(Cnative.astype(np.float64),shape)
        Ksmall=block(K,shape,'any')|oldF['K']
        geom=oldF['surface_intersection_valid']&oldF['geometry_support']
        retained=view['retained_reveal_guard']|view['retained_box_guard']|oldF['retained_world_guard']
        Avisible=(Afrac==1)&geom&~retained&~Ksmall
        Cteacher=(Cfrac==1)&geom&~retained&~Ksmall
        high=init['road_only_alpha'].squeeze()>.9
        Fvalid=oldF['legal_color']
        Aphoto=Avisible&high;Cphoto=Cteacher&high&Fvalid
        covA=Avisible;covC=Cteacher;coverage=covA|covC
        expected_A=(block(native_obs.astype(np.float64),shape)/255).transpose(2,0,1)
        expected_C=(block(native_teacher.astype(np.float64),shape)/255).transpose(2,0,1)
        errorA=float(np.max(np.abs(expected_A-s['target_A_srgb'])))
        errorC=float(np.max(np.abs(expected_C-s['target_C_srgb'])))
        source_rounding_tolerance=4*float(np.finfo(np.float32).eps)
        checks=dict(frozen_source_files_exact=all(pin_matches(value) for value in row['source_files'].values()),
            original_teacher_RGB_reconstructs=errorC<source_rounding_tolerance,original_observed_A_reconstructs=errorA<source_rounding_tolerance,
            A_visible_exact=np.array_equal(s['A_visible'],Avisible),C_teacher_geometric_exact=np.array_equal(s['C_teacher_hole'],Cteacher),
            A_photo_fixed_gate_exact=np.array_equal(s['A_mask'],Aphoto),C_common_fixed_gate_exact=np.array_equal(s['C_mask'],Cphoto),
            initial_alpha_exact_parent=np.array_equal(s['fixed_initial_alpha'],init['road_only_alpha'].squeeze()),
            native_A_fraction_exact=np.array_equal(s['q_A_native_footprint'],Afrac),native_C_fraction_exact=np.array_equal(s['q_C_native_footprint'],Cfrac),
            C_photo_has_complete_physical_W_footprint=not np.any(Cphoto&(Cfrac<1)),
            no_retained_A_C_overlap=not np.any((Aphoto|Cphoto)&Ksmall),no_A_C_overlap=not np.any(Aphoto&Cphoto),
            C_effect_diagnostic_only=not np.any(Cphoto&block(effect,shape,'any')),
            F_control_target_exact=np.array_equal(s['F_control_C_srgb'],oldF['target_rgb_srgb']),
            F_control_common_valid=not np.any(Cphoto&~Fvalid),
            coverage_A_independent_initial_alpha=np.array_equal(s['coverage_A'],covA),
            coverage_C_independent_initial_alpha=np.array_equal(s['coverage_C'],covC),
            coverage_C_independent_F_colour_validity=np.array_equal(s['coverage_C'],Cteacher),
            all_geometry_only_teacher_pixels_authorized=not np.any(Cteacher&~Fvalid&~s['coverage_C']),
            coverage_union_exact=np.array_equal(s['coverage_mask'],coverage),
            outside_is_full_authorized_complement=np.array_equal(s['outside'],~(coverage|Ksmall)),
            authorized_coverage_disjoint_outside=not np.any(coverage&s['outside']),
            alpha_q_A_independent_initial_alpha=np.array_equal(s['q_A_coverage'],Afrac*covA),
            alpha_q_C_independent_initial_alpha=np.array_equal(s['q_C_coverage'],Cfrac*covC),
            photo_q_A_exact=np.array_equal(s['q_A'],Afrac*Aphoto),photo_q_C_exact=np.array_equal(s['q_C'],Cfrac*Cphoto))
        P=init['road_only_rgb'];alpha=init['road_only_alpha'].squeeze()
        def mae(T,mask):return float(np.abs(P-alpha[None]*T)[:,mask].mean()) if mask.any() else None
        A0,C0=mae(s['target_A_srgb'],Aphoto),mae(s['target_C_srgb'],Cphoto)
        rows.append(dict(raw=raw,status='pass' if all(checks.values()) else 'failed',checks={key:bool(value) for key,value in checks.items()},
            source_A_RGB_max_error=errorA,source_C_RGB_max_error=errorC,A_photo_pixels=int(Aphoto.sum()),C_common_photo_pixels=int(Cphoto.sum()),
            source_RGB_float32_rounding_tolerance=source_rounding_tolerance,
            full_C_any_footprint_pixels=int((Cfrac>0).sum()),full_C_complete_footprint_pixels=int((Cfrac==1).sum()),
            qualified_C_geometry_pixels=int(Cteacher.sum()),teacher_only_C_geometry_pixels=int((Cteacher&~Fvalid).sum()),
            coverage_A_pixels=int(covA.sum()),coverage_C_pixels=int(covC.sum()),
            low_initial_alpha_A_coverage_pixels=int((covA&~high).sum()),low_initial_alpha_C_coverage_pixels=int((covC&~high).sum()),
            low_initial_alpha_C_has_coverage_weight=int((covC&~high&(s['q_C_coverage']>0)).sum()),
            baseline_A_real_premult_MAE=A0,baseline_C_pseudo_premult_MAE=C0,
            proposed_frozen_A_max=A0+.002 if A0 is not None else None,
            proposed_frozen_C_max=.8*C0 if C0 is not None else None,
            control_teacher_signal_MAE=float(np.abs(s['target_C_srgb']-s['F_control_C_srgb'])[:,Cphoto].mean()) if Cphoto.any() else None))
    checks=dict(two_teacher_frames_only=manifest['teacher_frames']==[86,110],diagnostic_not_train=manifest['diagnostic_only_frames']==[134,179,198],
                no_blind_B_claim=not manifest['blind_B_claimed'],no_new_SD=manifest['SD_frozen'],formal_updates_zero=manifest['formal_updates']==0,
                all_input_masks_pass=all(row['status']=='pass' for row in rows),
                immutable_target_files_exact=all(pin_matches(row) for row in freeze['target_files']))
    report=dict(schema='independent_direct_teacher_supervision.v1',status='pass' if all(checks.values()) else 'failed',checks=checks,frames=rows,
        actual_supervision=str(args.supervision),coverage_protocol='all physically qualified teacher geometry independent of F color availability',
        A_role='real visible RGB training fidelity',C_role='frozen generated teacher agreement, not hidden GT',
        paired_control='same masks/weights; only C RGB target source differs; R8 pretraining F projection explicitly identified',
        common_C_is_not_full_hole=True,no_trainer_or_supervision_mutations=True,new_GPU_forwards=0,formal_updates=0,human_verdict=None,
        remaining_gate='actual GPU routed gradients, actual row scopes/maximum-state K guard, complete restore/stage-boundary loader smoke')
    (args.out/'acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--supervision',type=Path,default=ROOT/'supervision_r3')
    p.add_argument('--out',type=Path,default=ROOT/'qa/supervision_r3_independent')
    main(p.parse_args())
