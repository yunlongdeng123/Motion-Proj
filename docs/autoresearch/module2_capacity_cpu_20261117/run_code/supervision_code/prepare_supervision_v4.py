"""Prepare a versioned CPU-only supervision candidate; never run a renderer."""
from pathlib import Path
import argparse,hashlib,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from PIL import Image
from road_asset_supervision_v4 import candidate_arrays,previous_mask_audit,scope_audit,world_domain_audit,crossview


def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,d):Path(path).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def camera_matrices(bank,observation,frame_id,wh):
    lines=(bank/'calib'/f'{frame_id}.txt').read_text().splitlines()
    calibration={line.split(':')[0]:np.fromstring(line.split(':')[1],sep=' ') for line in lines}
    intrinsic=calibration['P0'].reshape(3,4)[:,:3].copy();intrinsic[:2]*=wh[0]/observation['camera_size'][0]
    camera=np.eye(4);camera[:3]=calibration['Tr_velo_to_cam_0'].reshape(3,4)
    return intrinsic,camera@np.linalg.inv(np.asarray(observation['ego_world']).reshape(4,4))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only preparation requires CUDA_VISIBLE_DEVICES empty')
    parser=argparse.ArgumentParser();parser.add_argument('--campaign-root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--identity',type=Path,required=True)
    args=parser.parse_args();base=args.campaign_root;out=args.output;out.mkdir(parents=True,exist_ok=False)
    oldroot=base/'road_direct_teacher_20261116_r1';maskroot=base/'road_mask_decoupling_20261116_r1';shadowroot=base/'object_shadow_gpu_20261116_r1'
    sources={};previous=[];rows=[];data={};cameras={};worldrows=[]
    def pin(path):
        path=Path(path).resolve();sources[str(path)]=sha(path);return path
    identity=read(pin(args.identity/'source_identity.json'))
    if not all(row['identity_verified'] and row['original_teacher_W_output_equals_guard_W_old'] and row['original_teacher_W_old_subset_W_guard'] and row['S_outside_original_teacher_W_guard']==0 for row in identity['rows']):
        raise ValueError('source identity or original teacher write domain is not established')
    for source in identity['source_files']:
        if sha(source['path'])!=source['sha256']:raise ValueError('source changed since identity audit: '+source['path'])
        sources[source['path']]=source['sha256']
    config=read(pin(base/'scene065_material_loop_20261114_r1/config.json'));bank=Path(config['bank'])
    ids={row['raw']:row['id'] for row in read(pin(bank/'receipt.json'))['frames']}
    asset=base/'road_asset_20261116_r1/phase2/formal_r1/asset'
    prior=load(pin(asset/'provenance.npz'),['world_xyz','known','surface_supported','legal_write','guarded_world_rows'])
    road=load(pin(asset/'road.npz'),['active','minimum_model_depth'])
    formal=oldroot/'direct_C/formal_r1';control=oldroot/'matched_F_control/formal_r1'
    scopes=load(pin(formal/'row_scopes.npz'));controlscopes=load(pin(control/'row_scopes.npz'))
    scope=scope_audit(scopes,road,prior)
    scope['matched_Fcontrol_scopes_equal_exact']=all(np.array_equal(v,controlscopes[k]) for k,v in scopes.items())
    code=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/road_direct_teacher_training.py')
    contracts={role:read(pin(folder/'frozen_contract.json')) for role,folder in [('direct_C',formal),('matched_F_control',control)]}
    source_sha=sha(pin(code))
    scope['trainer_source_matches_both_formal_bindings']=all(c['input_binding'].get(str(code))==source_sha for c in contracts.values())
    scope['arms_selected_C_targets']={'direct_C':'target_C_srgb: frozen raw 2D pseudo teacher','matched_F_control':'F_control_C_srgb: old immutable atlas projection'}
    scope['displayed_raw_C_reference_is_Fcontrol_loss_target']=False
    write(out/'previous_scope_and_loss_audit.json',dict(schema='road_capacity_previous_scope_loss_audit.v1',scope=scope,
        frozen_loss_weights=contracts['direct_C']['loss'],source_sha256=source_sha,
        photo='P_sRGB vs stopgrad(alpha)*selected target; photo raster detaches opacity/scales; independent A/C q normalization',
        coverage='separate RGB-detached render, full Avisible/Cteacher geometry alpha MSE to1, independent high-alpha and F0 color-validity',
        keep='outside_RGB/outside_alpha independently preserve old S2 road-only premult RGB and alpha; retained K has weight4',
        candidate_shadow_gpu_contributions_available=False,formal_updates=0,GPU_operations=0))
    for raw in [86,110]:
        sid=f'scene065_raw{raw:03d}'
        old=load(pin(oldroot/'supervision_r3'/f'{raw:03d}.npz'))
        roles=load(pin(maskroot/'prepared'/sid/'B/roles_native.npz'))
        guard=load(pin(maskroot/'output_guard_probe'/sid/'native_guard_roles.npz'))
        field=load(pin(shadowroot/'runs/postprocess_local_u_r2'/sid/'provider/shadow_field.npz'))
        metadata=json.loads(str(field['metadata_json'].item()))
        routing=read(pin(shadowroot/'evaluation/results_postprocess_r2/routing'/sid/'routing_receipt.json'))
        evidence=load(pin(args.identity/f'{raw:03d}_native_routing_evidence.npz'),['other_shadow_prediction'])
        candidate,audit=candidate_arrays(old,roles,field['probability'],field['uncertainty'],field['valid'],metadata,
            evidence['other_shadow_prediction'],routing['public_routing_receipt']['policy'],guard['W_guard'],shadow_weight_mode='binary')
        np.savez_compressed(out/f'{raw:03d}.npz',**candidate);data[raw]=candidate
        previous.append(dict(raw=raw,masks=previous_mask_audit(old,roles)))
        observation=read(pin(bank/'observations'/f'{raw:03d}.json'));pin(bank/'calib'/f'{ids[raw]}.txt')
        cameras[raw]=camera_matrices(bank,observation,ids[raw],(480,320))
        domains={name:world_domain_audit(candidate,mask,prior['world_xyz'],cameras[raw]) for name,mask in [
            ('legacy_full_C_footprint',old['C_full_hole']),('C_hole_geometry',candidate['C_hole']),('C_hole_photo',candidate['C_hole_photo']),
            ('target_shadow_possible',candidate['S_target_shadow_possible']),('target_shadow_full_footprint',candidate['S_target_shadow_full_native_footprint']),
            ('target_shadow_qualified',candidate['S_target_shadow']),('target_shadow_photo',candidate['S_target_shadow_photo']),
            ('target_shadow_boundary_unknown',candidate['S_target_shadow_boundary_unknown']),('A_visible',candidate['A_visible'])]}
        row=dict(raw=raw,candidate=audit,world_mapping=domains);rows.append(row)
        all_requested=candidate['coverage_A']|candidate['coverage_C']|candidate['coverage_S']
        tri=candidate['triangle_world_rows'][all_requested];weights=candidate['triangle_weights'][all_requested]
        worldrows.append(np.unique(tri[(tri>=0)&(weights>1e-7)]))
        overlay=np.uint8(np.floor(candidate['target_A_srgb'].transpose(1,2,0)*255+.5)).astype(np.float32)
        for mask,color in [(candidate['A_mask'],[60,155,255]),(candidate['C_hole_photo'],[60,225,140]),
            (candidate['S_target_shadow_photo'],[255,190,35]),(candidate['S_target_shadow_boundary_unknown'],[235,100,230])]:
            overlay[mask]=.5*overlay[mask]+.5*np.asarray(color)
        Image.fromarray(np.uint8(np.clip(np.floor(overlay+.5),0,255))).save(out/f'{raw:03d}_A_C_S_overlay.webp',lossless=True,method=4)
        print(json.dumps(dict(raw=raw,**audit),ensure_ascii=False),flush=True)
        del roles,guard,field,evidence
    write(out/'previous_mask_audit.json',dict(schema='road_capacity_previous_mask_audit.v1',rows=previous,
        conclusion='old target-associated shadow and C_effect pixels received no A/C photo or coverage teacher; all were inside preservation outside',
        target_shadow_is_not_all_effect=True,GPU_operations=0))
    crossdir=out/'crossview';crossdir.mkdir();crossrows=[]
    for a,b in [(86,110),(110,86)]:
        for sa,sb in [('C','C'),('C','A'),('S','S'),('S','A'),('S','C'),('C','S'),('A','A')]:
            row,arrays=crossview(data,cameras,a,b,sa,sb)
            np.savez_compressed(crossdir/f'{a:03d}_{sa}_to_{b:03d}_{sb}.npz',**arrays);crossrows.append(row)
    write(crossdir/'conflicts_and_missing_support.json',dict(schema='road_capacity_same_world_teacher_audit.v1',rows=crossrows,
        correspondence='same original cameras; full four-neighbor role membership and interpolated mesh world error<=.05m inherited from prior diagnostic',
        conflict_values_do_not_change_masks_q_or_loss_weights=True,exposure_or_white_balance_fitted=False,
        interpretation='teacher/view/exposure/resampling disagreement, not independently identified hidden-road error',GPU_operations=0))
    np.savez_compressed(out/'cpu_world_request.npz',world_rows=np.unique(np.concatenate(worldrows)),
        actual_renderer_contribution=np.asarray(False),eligible_scope_gpu_validated=np.asarray(False))
    write(out/'manifest.json',dict(schema='road_capacity_supervision_v4_candidate.v1',teacher_frames=[86,110],diagnostic_only=[134,179,198],
        candidate_only=True,formal_ready=False,teacher_human_verdict=None,shadow_model_calibrated=False,
        q_S='binary full-native-footprint eligibility inside inherited confident target-shadow interior; not calibrated probability of teacher correctness',
        C_hole='physical teacher geometry independent of old F0 color availability; non-target-shadow prediction conflicts excluded/reported',
        S_target_shadow='inherited high/low-uncertainty target pair interior, no uncertain boundary, original raw teacher W_guard, actual road geometry and retained guards',
        A='original real visible road, no predicted target-shadow or uncertain boundary conflict; non-target shadow appearance may remain real A',
        colors='native sRGB raw 2D completion; black-background premult photo protocol; no RGB/alpha or fitted exposure',
        coverage='A/C/S full geometry domains separately recorded; proposed capacity comparison freezes geometry/opacity/scales, so no coverage updates authorized',
        outside='~(A_visible|C_hole|S_target_shadow|K); unresolved boundary/teacher-missing remains preservation-only',
        weights_recommendation_not_training_contract=dict(A=1.,C_hole=.5,S_target_shadow=.25,shadow_edge=0.,
            normalization='each domain divided by its own fixed q sum; no thresholds or weights tuned from crossview residuals'),
        GPU_operations=0,training_updates=0,new_render_operations=0,old_source_or_run_modified=False,rows=rows,
        required_before_GPU_training=['confirm candidate masks/teacher quality','freeze initialization and revalidate fixed alpha gate',
            'actual A/C/S renderer contribution scopes and protected row guard','actual joint full-K and lifecycle smoke'],
        prior_scope_note='cpu_world_request is geometric mapping only, never an actual alphaT eligibility certificate'))
    targets=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'freeze.json',dict(schema='road_capacity_supervision_candidate_freeze.v1',source_files=[dict(path=k,sha256=v) for k,v in sorted(sources.items())],
        code_files=[dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),dict(path=str(Path(__file__).with_name('road_asset_supervision_v4.py').resolve()),sha256=sha(Path(__file__).with_name('road_asset_supervision_v4.py')))],
        target_files=targets,candidate_only=True,formal_ready=False,GPU_operations=0,training_updates=0))
    print(json.dumps(dict(status='CPU_CANDIDATE_PREPARED',output=str(out),formal_ready=False,GPU_operations=0),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
