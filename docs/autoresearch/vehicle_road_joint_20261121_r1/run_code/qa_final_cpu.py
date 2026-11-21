"""Final independent CPU audit; never imports a renderer or initializes CUDA."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image

FRAMES=(86,110,134,179,198)
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
ROAD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset')
MASK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/repaired_masks_all_r4')


def read(p):
    return json.loads(Path(p).read_text())


def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def diff(a,b):
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(max=float(d.max()) if d.size else 0.,mean=float(d.mean()) if d.size else 0.,identical=bool(np.array_equal(a,b)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--visible-support',type=Path)
    args=parser.parse_args();root=args.root
    checks=[];pending=[]
    def check(name,passed,**data):
        checks.append(dict(name=name,passed=bool(passed),**data))
    def needed(p):
        if not p.exists():pending.append(str(p));return False
        return True
    contract=read(root/'contract.json')
    check('formal_budget',contract['vehicle_steps']==200 and contract['joint_steps']==200,
          vehicle=contract['vehicle_steps'],joint=contract['joint_steps'],train_frames=contract['train_frames'])
    prior=read(root/'lower_prior.json')
    check('prior_explicit_non_GT',prior.get('true_hidden_shape_and_RGB_GT') is False and prior.get('new_prior_gaussians')==624,
          prior=prior.get('prior'),new_rows=prior.get('new_prior_gaussians'),visible_mass=prior.get('prior_visible_contribution_mass_pixels'))
    mask_receipt=read(MASK/'receipt.json')
    provenance=dict(kind='estimated/repaired segmentation proxy, not manual hidden-shape or RGB ground truth',
                    receipt_status=mask_receipt.get('status'),source=mask_receipt.get('source'),
                    receipt_sha256=sha(MASK/'receipt.json'),contract_mask_description=contract.get('mask_content'))
    frozen=root/'code/frozen_formal_r2'
    code_hashes={str(p.relative_to(root)):sha(p) for p in frozen.glob('*.py')}
    asset=root/'assets/joint_fit'
    geometry={};native=[];frame_rows=[];other_actor_rows=[];detail_check={}
    if needed(asset/'road.npz'):
        before=load(ROAD/'road.npz');after=load(asset/'road.npz')
        immutable={k:bool(k in after and np.array_equal(v,after[k])) for k,v in before.items() if k!='reference_srgb'}
        check('road_geometry_coverage_and_all_noncolor_fields',all(immutable.values()),fields=immutable)
        allowed=load(ROAD.parents[1]/'train_inputs/allowed_color_world_rows.npz')['world_row_ids']
        changed=np.flatnonzero(np.any(before['reference_srgb']!=after['reference_srgb'],axis=1))
        unauthorized=np.setdiff1d(changed,allowed)
        check('road_color_update_scope',len(unauthorized)==0,allowed_rows=len(allowed),changed_rows=len(changed),unauthorized_rows=len(unauthorized))
        proof=read(asset/'persistent/proof.json');parent=read(ROAD/'persistent/proof.json')
        for source in parent['native_sources']:
            actual=sha(source['path']);native.append(dict(role=source['role'],matches=actual==source['sha256'],sha256=actual))
        check('native_checkpoint_files_unchanged',all(x['matches'] for x in native),rows=native)
        check('saved_native_source_bindings',proof['native_sources']==parent['native_sources'])
        for name,entry in [('ownership',parent['ownership']),('static_indices',parent['gates']['indices']),('static_gate_checkpoint',parent['gates']['checkpoint'])]:
            check(name+'_source_unchanged',sha(entry['path'])==entry['sha256'])
        initial=load(root/'initial_vehicle_prior.npz');final=load(asset/'vehicle.npz')
        for key in ['xyz','rotation','source_row','source_branch','source_raw']:
            geometry[key]=bool(np.array_equal(initial[key].astype(final[key].dtype),final[key]))
        check('canonical_geometry_and_provenance_unchanged',all(geometry.values()),fields=geometry,rows=len(final['xyz']),comparison='same FP32 runtime initialization; source quaternion array was FP64')
        check('prior_row_count',int((final['source_branch']==3).sum())==624,prior_rows=int((final['source_branch']==3).sum()))
        support_path=args.visible_support
        if support_path is None:
            for name in ['vehicle_fit_visible.npz','visible_support.npz','vehicle_visible_support.npz']:
                if (root/name).exists():support_path=root/name;break
        if support_path is None:
            pending.append('per-row final VJP visible-support artifact')
        else:
            values=load(support_path);support=next(v for v in values.values() if v.dtype==bool and v.shape==(len(final['xyz']),))
            hidden=~support
            opacity=diff(np.clip(initial['opacity'][hidden],1e-5,1-1e-5),final['opacity'][hidden])
            scales=diff(initial['scales'][hidden].astype(final['scales'].dtype),final['scales'][hidden])
            check('hidden_opacity_and_scales_preserved',opacity['max']<2e-6 and scales['max']==0,hidden_rows=int(hidden.sum()),opacity=opacity,scales=scales,support_sha256=sha(support_path))
        pose=load(asset/'poses.npz')
        check('pose_export_keys',np.array_equal(pose['raw'],FRAMES),poses=list(map(int,pose['raw'])))
    for raw in FRAMES:
        final_path=root/'delivery/joint_fit'/f'{raw:03d}'/'render_float.npz'
        if not needed(final_path):continue
        final=load(final_path);baseline=load(root/'delivery/canonical_prior_init'/f'{raw:03d}'/'render_float.npz')
        native_image=load(root/'delivery/native_before'/f'{raw:03d}'/'render_float.npz')
        ref=root/'delivery/references'/f'{raw:03d}'
        vehicle=np.asarray(Image.open(ref/'vehicle_mask.png'))>0
        legal=np.asarray(Image.open(ref/'road_legal.png'))>0
        box=np.asarray(Image.open(ref/'vehicle_potential_bbox.png'))>0
        negative=np.asarray(Image.open(ref/'vehicle_negative_valid.png'))>0
        ignored=np.asarray(Image.open(ref/'vehicle_occlusion_ignore.png'))>0
        labels=np.asarray(Image.open(MASK/f'{raw:03d}'/'labels.png').resize((1152,768),Image.Resampling.NEAREST))
        check(f'frame{raw:03d}_finite',all(np.isfinite(v).all() for v in final.values()))
        check(f'frame{raw:03d}_road_coverage_unchanged',np.array_equal(final['road_alpha'],baseline['road_alpha']))
        check(f'frame{raw:03d}_supervision_exclusion',not (legal&box).any() and not (negative&ignored).any() and not (legal&(labels>0)&(labels!=10)).any(),road_pixels=int(legal.sum()),vehicle_pixels=int(vehicle.sum()),negative_pixels=int(negative.sum()))
        delta=np.abs(final['joint_keep']-baseline['joint_keep'])
        row=dict(raw=raw,vehicle_fixed_mask_sha256=hashlib.sha256(vehicle.tobytes()).hexdigest(),road_fixed_mask_sha256=hashlib.sha256(legal.tobytes()).hexdigest(),
                 keep_change_visible_MAE=float(delta[:,vehicle].mean()),
                 DELETE_change=diff(final['joint_delete'],baseline['joint_delete']))
        reloaded=root/'delivery/joint_fit_reloaded'/f'{raw:03d}'/'render_float.npz'
        if needed(reloaded):
            reload=load(reloaded);errors={key:diff(value,reload[key]) for key,value in final.items()}
            check(f'frame{raw:03d}_disk_reload_render',all(x['max']<2e-5 for x in errors.values()),fields=errors)
        for actor in np.unique(labels):
            if actor in [0,10,21]:continue
            mask=labels==actor
            for name,start in [('prior_to_final',baseline),('native_to_final',native_image)]:
                d=np.abs(final['joint_delete']-start['joint_delete'])[:,mask]
                other_actor_rows.append(dict(raw=raw,native_actor_id=int(actor),comparison=name,pixels=int(mask.sum()),MAE=float(d.mean()),maximum=float(d.max()),fraction_channels_over_1_255=float((d>1/255).mean())))
        frame_rows.append(row)
    detail=root/'delivery/detail_metrics.json'
    if needed(detail):
        rows=read(detail)['rows'];groups={}
        for row in rows:
            groups.setdefault((row['raw'],row['role']),[]).append(row)
        detail_check={f'{raw}:{role}':dict(pixel_counts=sorted(set(r['pixels'] for r in records)),HF_core_pixel_counts=sorted(set(r.get('HF_core_pixels_1',0) for r in records)),stages=[r['stage'] for r in records]) for (raw,role),records in groups.items()}
        check('bench_fixed_pixel_domains',all(len(x['pixel_counts'])==1 and len(x['HF_core_pixel_counts'])==1 for x in detail_check.values()),groups=detail_check)
    for name in ['smoke/proof.json','delivery/proofs.json','disk_reload_proof.json','post_numeric_probe.json']:
        path=root/name
        if needed(path):
            proof=read(path)
            if name=='post_numeric_probe.json':
                check(name,proof.get('passed') is True,proof=proof)
            elif name!='delivery/proofs.json':check(name,proof.get('passed') is True,proof=proof)
            else:check('final_native_and_geometry_proof',proof.get('native_scene_unchanged') and proof.get('local_xyz_equal') and proof.get('trajectory_equal') and proof.get('joint_real_renderer_two_branch_nonzero_all_steps'),proof=proof)
    record=root/'training_records.json'
    if needed(record):
        records=read(record)['rows'];formal=[r for r in records if r['role']=='formal']
        counts={stage:sum(r['stage']==stage for r in formal) for stage in ['vehicle','joint']}
        check('formal_step_counts_and_finite_losses',counts==dict(vehicle=200,joint=200) and all(np.isfinite(r['loss']) for r in records),counts=counts,total_logged_steps=len(records))
    ledger=read(root/'resource_ledger.json') if (root/'resource_ledger.json').exists() else None
    output=dict(schema='vehicle_road_joint.final_independent_cpu_qa.v1',CPU_only=True,checks=checks,
                passed=not pending and all(r['passed'] for r in checks),pending=pending,
                mask_provenance=provenance,frozen_code_sha256=code_hashes,resource_ledger=ledger,frames=frame_rows,
                other_actor_image_differences_are_descriptive_not_zero_constraint=other_actor_rows)
    target=root/'qa_final_cpu.json';target.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(output=str(target),passed=output['passed'],checks=len(checks),failed=[r['name'] for r in checks if not r['passed']],pending=pending),indent=2))


if __name__=='__main__':main()
