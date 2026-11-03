"""Fresh-process C2 acceptance; no optimizer updates, no disabled C2 branch."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import random
import sys
import time
import itertools

RUN = Path(__file__).resolve().parent.parent
THRESHOLDS = dict(full_PSNR_delta_min=-.05, full_SSIM_delta_min=-.001,
    full_LPIPS_delta_max=.002, road_core_ratio_max=1.05,
    a6_observed_delete_MAE_must_improve=True,
    visual_review_required=True, tolerance_is_not_an_improvement_target=True)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode() + str(array.shape).encode() + str(array.dtype).encode() + array.tobytes())
    return h.hexdigest()


def possible_a6_projection(geometry, records, raw, dilation_px=2):
    """Independent physical possible occupancy, never a completed-body label."""
    import cv2
    import numpy as np
    from scipy.ndimage import binary_dilation
    ids=np.flatnonzero(np.asarray(geometry['raw_ids'])==raw)
    assert len(ids)==1
    index=int(ids[0])
    matrix=np.asarray(geometry['model_from_canonical'][index],dtype=float)
    size=np.asarray(geometry['box_lwh_per_raw'][index] if 'box_lwh_per_raw' in geometry else geometry['box_lwh_m'],dtype=float)
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    model=(signs*size[None]/2)@matrix[:3,:3].T+matrix[:3,3]
    record=next(row for row in records if row['img_name']==f'0008{raw:03d}')
    c2w=np.eye(4);c2w[:3,:3]=record['rotation'];c2w[:3,3]=record['position']
    world_to_camera=np.linalg.inv(c2w)
    camera=model@world_to_camera[:3,:3].T+world_to_camera[:3,3]
    near=1e-4
    points=[point for point in camera if point[2]>=near]
    for i in range(8):
        for j in range(i+1,8):
            if np.count_nonzero(signs[i]!=signs[j])!=1:
                continue
            a,b=camera[i],camera[j]
            if (a[2]<near)!=(b[2]<near):
                points.append(a+(b-a)*((near-a[2])/(b[2]-a[2])))
    mask=np.zeros((320,480),np.uint8)
    if len(points)>=3:
        xyz=np.asarray(points)
        uv=np.c_[xyz[:,0]/xyz[:,2]*(record['fx']/4)+record['cx']/4-.5,
                 xyz[:,1]/xyz[:,2]*(record['fy']/4)+record['cy']/4-.5]
        hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
        cv2.fillConvexPoly(mask,np.rint(hull).clip(-1000000,1000000).astype(np.int32),1)
        mask=binary_dilation(mask.astype(bool),iterations=dilation_px) if dilation_px else mask.astype(bool)
    return mask.astype(bool)


def prepare_road_domain():
    """CPU-only domain freeze, independent of proposals, C2 images and scores."""
    import numpy as np
    from PIL import Image
    geometry_path=RUN/'inputs/a6_geometry_minimal.npz'
    cameras_path=RUN.parent/'scene008_a37_only_B10k_20261103_r1/formal10k/cameras.json'
    protocol_path=RUN/'evidence/road_measurement_correction_preformal.json'
    geometry=dict(np.load(geometry_path,allow_pickle=False));records=json.loads(cameras_path.read_text())
    evidence=RUN/'evidence/possible_a6_obb_domain';evidence.mkdir(parents=True,exist_ok=True)
    masks=[];rows=[]
    for raw in range(89,138):
        mask=possible_a6_projection(geometry,records,raw,2);masks.append(mask)
        view=np.load(RUN/'inputs/a6_views'/f'{raw:03d}.npz',allow_pickle=False)
        visible=view['actor_labels']==4
        rows.append(dict(raw=raw,possible_pixels=int(mask.sum()),visible_a6_pixels=int(visible.sum()),
            visible_a6_inside_possible=int((visible&mask).sum())))
        if raw in (101,105,113):
            Image.fromarray(np.uint8(mask)*255).save(evidence/f'{raw:03d}_possible_a6.png')
    output=RUN/'inputs/possible_a6_obb_projection.npz'
    geometry_hash=file_hash(geometry_path);camera_hash=file_hash(cameras_path);protocol_hash=file_hash(protocol_path)
    temporary=output.with_name(output.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,masks=np.stack(masks),raw_ids=np.arange(89,138),dilation_px=np.int64(2),
            geometry_sha256=np.asarray(geometry_hash),cameras_sha256=np.asarray(camera_hash),protocol_sha256=np.asarray(protocol_hash))
    os.replace(temporary,output)
    receipt=dict(geometry_sha256=geometry_hash,cameras_sha256=camera_hash,protocol_sha256=protocol_hash,
        domain_sha256=file_hash(output),source_geometry=str(geometry_path),source_camera=str(cameras_path),
        protocol=json.loads(protocol_path.read_text()),domain_is_body_GT=False,uses_proposal_or_C2_output=False,
        optimizer_updates=0,GPU_used=False,raw_frames=49,rows=rows)
    write(evidence/'receipt.json',receipt)
    print(json.dumps(dict(stage='CPU_ROAD_DOMAIN_FROZEN',file=str(output),sha256=receipt['domain_sha256'],
        possible_pixels=sum(row['possible_pixels'] for row in rows),visible_a6_coverage=sum(row['visible_a6_inside_possible'] for row in rows)/max(sum(row['visible_a6_pixels'] for row in rows),1))))
    return output,receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=RUN / 'formal/checkpoint001000.pt')
    parser.add_argument('--output', type=Path, default=RUN / 'final_acceptance')
    parser.add_argument('--asset', type=Path)
    parser.add_argument('--no-lpips', action='store_true', help='Provisional only; missing LPIPS cannot pass the full-image gate')
    parser.add_argument('--prepare-road-domain-only',action='store_true',help='Freeze independent OBB projection masks on CPU without loading a checkpoint or CUDA')
    args = parser.parse_args()
    if args.prepare_road_domain_only:
        prepare_road_domain()
        return
    # Parse help before importing CUDA libraries; this file is safe to prepare on CPU.
    sys.dont_write_bytecode = True
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion
    import torch
    import c2_runtime as c2
    from baseline_audit import render_baseline
    from utils.loss_utils import psnr, ssim
    from compare_state import differences
    started = time.monotonic()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'frames').mkdir()
    write(dest / 'thresholds.json', THRESHOLDS)
    torch.set_num_threads(1)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    contract = checkpoint['contract']
    settings = contract['settings']
    asset = args.asset or Path(contract['actor_asset'])
    receipt_path = Path(contract.get('proposal_receipt', str(asset.with_name(asset.stem + '_receipt.json'))))
    asset_receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    experiment = c2.Experiment(settings, asset)
    assert checkpoint['parent_identity'] == experiment.n.before
    experiment.actor.load_state_dict(checkpoint['actor'], strict=True)
    optimizer = torch.optim.Adam(experiment.parameter_groups(), eps=1e-15)
    optimizer.load_state_dict(checkpoint['optimizer'])
    sampler = random.Random(0)
    sampler.setstate(checkpoint['sampler'])
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])
    restored = dict(step=checkpoint['step'], actor=experiment.actor.state_dict(),
        optimizer=optimizer.state_dict(), sampler=sampler.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), parent_identity=experiment.n.before, contract=contract)
    errors = differences(checkpoint, restored)
    restore = dict(step=checkpoint['step'], exact_full_state=not errors, differences=errors,
        includes='actor parameters/buffers, Adam, sampler, Python/NumPy/Torch/CUDA RNG, frozen-parent identity',
        optimizer_updates=0, checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint))
    write(dest / 'restore.json', restore)
    assert not errors, errors
    experiment.actor.requires_grad_(False)
    experiment.actor.eval()
    road_before = tensor_state_hash(experiment.base.repair.state_dict())
    actor_before = tensor_state_hash(experiment.actor.state_dict())
    parent_before = dict(experiment.n.before)
    contributor_binding = dict(enabled=bool(settings.get('actor_gates',False)))
    if contributor_binding['enabled']:
        selection_path = Path(settings['actor_contributors'])
        selection = dict(np.load(selection_path,allow_pickle=False))
        saved_identity = json.loads(str(selection['parent_identity_json'].item()))
        current_identity = json.loads(json.dumps(experiment.n.before))
        contributor_binding.update(file=str(selection_path),sha256=file_hash(selection_path),
            static_rows=int(selection['static_rows']),
            source_asset=str(selection['asset'].item()),
            parent_matches=saved_identity==current_identity,
            ids_match_checkpoint=np.array_equal(selection['ids'],experiment.actor.contributor_ids.detach().cpu().numpy()),
            candidate_is_not_ownership_GT=True)
        contributor_binding['source_asset_differs_from_actor_asset'] = Path(contributor_binding['source_asset']).resolve() != asset.resolve()
        assert contributor_binding['static_rows']==len(experiment.n.static.get_xyz) and contributor_binding['parent_matches'] and contributor_binding['ids_match_checkpoint'], contributor_binding
        del selection
    domain_path=RUN/'inputs/possible_a6_obb_projection.npz'
    domain_receipt_path=RUN/'evidence/possible_a6_obb_domain/receipt.json'
    assert domain_path.exists() and domain_receipt_path.exists(), 'Freeze independent road domain on CPU before final scoring'
    road_domain=dict(np.load(domain_path,allow_pickle=False));domain_receipt=json.loads(domain_receipt_path.read_text())
    assert str(road_domain['geometry_sha256'].item())==file_hash(RUN/'inputs/a6_geometry_minimal.npz')
    assert str(road_domain['cameras_sha256'].item())==file_hash(Path(experiment.n.files[0]).parent/'cameras.json')
    del checkpoint, restored, optimizer
    gc.collect()
    criterion = None
    lpips_error = None
    if not args.no_lpips:
        try:
            from lpipsPyTorch.modules.lpips import LPIPS
            criterion = LPIPS('vgg', '0.1').eval().cuda()
            criterion.requires_grad_(False)
        except Exception as exc:
            lpips_error = repr(exc)
    metric_contract = dict(optimizer_updates=0, frame_count=49, raw_range=[89, 137], camera='image_0',
        resolution=[480, 320], ground_truth='Original RGB float64/255 then OpenCV INTER_LINEAR then float32',
        aggregation='Arithmetic mean of per-frame full-image PSNR/SSIM/LPIPS-vgg; mask MAE pixel-weighted across frames',
        C2_branch_enabled_in_full_and_delete=True, full_keeps_target_a37=True, delete_removes_only_a37=True,
        deletion_full_image_metrics_are_diagnostic_only=True,
        deletion_metric_warning='Observed RGB contains deleted a37; delete PSNR/SSIM/LPIPS are not edit-quality truth or acceptance gates',
        fixed_road_reference='Unchanged r3 inputs/views/*.npz; inferred cells excluded from fit/new_dark and reported separately',
        paired_protection_reference='C2.render(enabled=False), with the same sun visibility; only a reference, C2 scored outputs always enabled=True',
        evidence_role='All199 BUILD; no independent completed-actor or clean-road test ground truth',
        thresholds=THRESHOLDS, checkpoint_contract=contract, actor_asset=str(asset), actor_asset_sha256=file_hash(asset),
        proposal_receipt=str(receipt_path), asset_receipt=asset_receipt,
        prior_label=settings.get('prior_label',contract.get('prior','unrecorded')),
        contributor_binding=contributor_binding,
        road_measurement_correction=domain_receipt,
        runtime_sha256=file_hash(Path(c2.__file__)), lpips_available=criterion is not None, lpips_error=lpips_error,
        human_verdict=None, quality_accepted=False)
    write(dest / 'contract.json', metric_contract)
    methods = ('native_full', 'r3_full', 'r3_delete37', 'C2_full', 'C2_delete37')
    rows = []
    source = Path(experiment.n.config['source_path']) / 'image/image_0'
    def rgb8(image):
        return np.uint8(np.floor(image.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255 + .5))
    def image_metrics(image, truth):
        image = image.clamp(0, 1)
        return dict(psnr=float(psnr(image, truth)), ssim=float(ssim(image[None], truth[None])),
            lpips_vgg=float(criterion(image[None], truth[None]).mean()) if criterion is not None else None)
    with torch.no_grad(), (dest / 'metrics.jsonl').open('x') as log:
        for raw in range(89, 138):
            frame = dest / 'frames' / f'{raw:03d}'
            frame.mkdir()
            original = np.asarray(Image.open(source / f'0008{raw:03d}.png').convert('RGB')) / 255.
            truth = torch.as_tensor(cv2.resize(original, (480, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32).transpose(2, 0, 1).copy(), device='cuda')
            d, view = experiment.base.view(raw), experiment.view(raw)
            labels = view['actor_labels']
            masks = dict(a6_observed=view['visible_evidence_mask'], a6_visible_label=labels == 4,
                a37=labels == 5, other_retained=(labels > 0) & (labels < 4), outside_a6=labels != 4,
                marking=d['marking_mask'], outside_repair=d['protect_mask'],
                road_boundary=binary_dilation(d['repair_mask'], iterations=2) & ~binary_erosion(d['repair_mask'], iterations=2) & ~d['other_vehicle_guard'])
            for label, alias in enumerate((5, 26, 18), 1):
                masks[f'retained_a{alias}'] = labels == label
            packages = dict(native_full=render_baseline(experiment.base, raw, True, False),
                r3_full=render_baseline(experiment.base, raw, True, True),
                r3_delete37=render_baseline(experiment.base, raw, False, True),
                C2_full=experiment.render(raw, True, enabled=True),
                C2_delete37=experiment.render(raw, False, enabled=True))
            paired = dict(full=experiment.render(raw,True,enabled=False),
                delete=experiment.render(raw,False,enabled=False))
            possible=road_domain['masks'][int(np.flatnonzero(road_domain['raw_ids']==raw)[0])]
            rec = dict(raw=raw, full_image={}, masks={}, road={}, road_matched={},road_domain_accounting={}, branch_enabled=True,
                possible_a6_pixels=int(possible.sum()),
                a6_view_truth_maxabs=float((experiment.truth(raw) - truth).abs().max()),
                replaced_native_rows=int(experiment.static_cache[raw].sum()))
            arrays = {name: rgb8(package['rgb']) for name, package in packages.items()}
            paired_arrays = {name: rgb8(package['rgb']) for name, package in paired.items()}
            for method in methods:
                rec['full_image'][method] = image_metrics(packages[method]['rgb'], truth)
                Image.fromarray(arrays[method]).save(frame / f'{method}.png')
            Image.fromarray(rgb8(truth)).save(frame / 'gt_opencv.png')
            Image.fromarray(np.uint8(possible)*255).save(frame/'possible_a6_obb.png')
            for name, mask_array in masks.items():
                mask_array = np.asarray(mask_array, dtype=bool)
                mask = torch.as_tensor(mask_array, device='cuda', dtype=torch.bool)
                count = int(mask.sum())
                item = dict(samples=count, methods={})
                for method in methods:
                    pred = packages[method]['rgb'].clamp(0, 1)
                    reference_name = 'r3_delete37' if method.endswith('delete37') else 'r3_full'
                    reference = packages[reference_name]['rgb'].clamp(0, 1)
                    paired_key = 'delete' if method.endswith('delete37') else 'full'
                    paired_reference = paired[paired_key]['rgb'].clamp(0,1)
                    diff8 = np.abs(arrays[method].astype(np.int16) - arrays[reference_name].astype(np.int16))[mask_array]
                    paired_diff8 = np.abs(arrays[method].astype(np.int16)-paired_arrays[paired_key].astype(np.int16))[mask_array]
                    item['methods'][method] = dict(gt_srgb_MAE=float((pred-truth).abs()[:, mask].mean()) if count else None,
                        baseline_srgb_change=float((pred-reference).abs()[:, mask].mean()) if count else None,
                        baseline_srgb_maxabs=float((pred-reference).abs()[:, mask].max()) if count else None,
                        baseline_change_MAE_8bit=float(diff8.mean()) if count else None,
                        baseline_change_max_8bit=int(diff8.max()) if count else None,
                        baseline_pixels_changed_over1=int((diff8.max(1) > 1).sum()) if count else 0,
                        paired_baseline_srgb_change=float((pred-paired_reference).abs()[:,mask].mean()) if count else None,
                        paired_baseline_srgb_maxabs=float((pred-paired_reference).abs()[:,mask].max()) if count else None,
                        paired_baseline_change_MAE_8bit=float(paired_diff8.mean()) if count else None,
                        paired_baseline_change_max_8bit=int(paired_diff8.max()) if count else None,
                        paired_baseline_pixels_changed_over1=int((paired_diff8.max(1)>1).sum()) if count else 0)
                rec['masks'][name] = item
            for method in ('r3_delete37', 'C2_delete37'):
                linear = c2.repair37.r4.linear(packages[method]['rgb'])
                rec['road'][method] = {}
                rec['road_matched'][method] = {}
                domains = dict(fit=(d['repair_mask'] & ~d['inferred_fine'], 'recovery_reference_linear'),
                    new_dark=(d['new_shadow_proxy'] & d['repair_mask'] & ~d['inferred_fine'], 'recovery_reference_linear'),
                    original_shadow=(d['shadow_proxy'] & d['domain_a37'], 'reference_linear'),
                    inferred=(d['inferred_fine'], 'recovery_reference_linear'))
                for name, (mask_array, reference_key) in domains.items():
                    mask = torch.as_tensor(mask_array, device='cuda')
                    reference = torch.as_tensor(d[reference_key].transpose(2, 0, 1), device='cuda')
                    rec['road'][method][name] = dict(samples=int(mask.sum()), linear_MAE=float((linear-reference).abs()[:, mask].mean()) if mask.any() else None)
                    matched_array=mask_array&~possible
                    matched=torch.as_tensor(matched_array,device='cuda')
                    rec['road_matched'][method][name]=dict(samples=int(matched.sum()),linear_MAE=float((linear-reference).abs()[:,matched].mean()) if matched.any() else None)
                    rec['road_domain_accounting'][name]=dict(original_samples=int(mask_array.sum()),matched_samples=int(matched_array.sum()),excluded_samples=int((mask_array&possible).sum()))
            visible = torch.as_tensor(view['visible_evidence_mask'], device='cuda')
            rec['actor_alpha'] = dict(full_on_visible_mean=float(packages['C2_full']['actor_alpha'][0, visible].mean()) if visible.any() else None,
                delete_on_visible_mean=float(packages['C2_delete37']['actor_alpha'][0, visible].mean()) if visible.any() else None)
            if raw in (101, 105, 113):
                Image.fromarray(np.concatenate([arrays[name] for name in ('native_full','C2_full','r3_delete37','C2_delete37')], axis=1)).save(dest / f'{raw:03d}_comparison.png')
                Image.fromarray(np.uint8(view['visible_evidence_mask']) * 255).save(frame / 'a6_observed_mask.png')
                Image.fromarray(rgb8(packages['C2_full']['actor_alpha'].expand(3,-1,-1))).save(frame / 'actor_alpha_full.png')
            rows.append(rec)
            log.write(json.dumps(rec, allow_nan=False) + '\n')
            log.flush()
            write(dest / 'progress.json', dict(frames=len(rows), raw=raw, optimizer_updates=0, seconds=time.monotonic()-started))
            if raw == 89 or (raw-88)%10 == 0 or raw == 137:
                print(json.dumps(dict(stage='final_acceptance', raw=raw, full=rec['full_image']['C2_full'])), flush=True)
            del packages, arrays, paired, paired_arrays
    summary = dict(frame_count=len(rows), optimizer_updates=0, full_image={}, masks={}, road={}, road_matched={},road_domain_accounting={},thresholds=THRESHOLDS,
        input_role='BUILD, no independent test', human_verdict=None, quality_accepted=False,
        checkpoint_step=restore['step'], diagnostic_only=bool(contract.get('diagnostic_only', False)),
        asset_formal_eligible=asset_receipt.get('formal_eligible'), proposal_receipt=str(receipt_path),
        prior_label=settings.get('prior_label',contract.get('prior','unrecorded')),
        contributor_binding=contributor_binding,
        road_measurement_correction=domain_receipt,
        deletion_full_image_metrics_are_diagnostic_only=True, actor_asset=str(asset), checkpoint=str(args.checkpoint),
        a6_view_truth_maxabs=max(row['a6_view_truth_maxabs'] for row in rows), seconds=time.monotonic()-started)
    for method in methods:
        summary['full_image'][method] = {key: float(np.mean([row['full_image'][method][key] for row in rows])) if rows[0]['full_image'][method][key] is not None else None for key in ('psnr','ssim','lpips_vgg')}
    for name in rows[0]['masks']:
        count = sum(row['masks'][name]['samples'] for row in rows)
        item = dict(samples=count, methods={})
        for method in methods:
            item['methods'][method] = {}
            for key in rows[0]['masks'][name]['methods'][method]:
                values = [(row['masks'][name]['samples'], row['masks'][name]['methods'][method][key]) for row in rows if row['masks'][name]['samples']]
                item['methods'][method][key] = (sum(v for n,v in values) if key.endswith('pixels_changed_over1') else max(v for n,v in values) if 'max' in key else sum(n*v for n,v in values)/count) if values else None
        summary['masks'][name] = item
    for method in ('r3_delete37','C2_delete37'):
        summary['road'][method] = {}
        summary['road_matched'][method] = {}
        for name in rows[0]['road'][method]:
            count = sum(row['road'][method][name]['samples'] for row in rows)
            summary['road'][method][name] = dict(samples=count, linear_MAE=sum(row['road'][method][name]['samples']*(row['road'][method][name]['linear_MAE'] or 0) for row in rows)/max(count,1))
            matched_count=sum(row['road_matched'][method][name]['samples'] for row in rows)
            summary['road_matched'][method][name]=dict(samples=matched_count,linear_MAE=sum(row['road_matched'][method][name]['samples']*(row['road_matched'][method][name]['linear_MAE'] or 0) for row in rows)/max(matched_count,1))
    for name in rows[0]['road_domain_accounting']:
        item={key:sum(row['road_domain_accounting'][name][key] for row in rows) for key in ('original_samples','matched_samples','excluded_samples')}
        item['excluded_fraction']=item['excluded_samples']/max(item['original_samples'],1)
        summary['road_domain_accounting'][name]=item
    native, final = summary['full_image']['native_full'], summary['full_image']['C2_full']
    deltas = dict(psnr=final['psnr']-native['psnr'], ssim=final['ssim']-native['ssim'],
        lpips_vgg=final['lpips_vgg']-native['lpips_vgg'] if final['lpips_vgg'] is not None else None)
    full_checks = dict(psnr=deltas['psnr'] >= THRESHOLDS['full_PSNR_delta_min'],
        ssim=deltas['ssim'] >= THRESHOLDS['full_SSIM_delta_min'],
        lpips=deltas['lpips_vgg'] is not None and deltas['lpips_vgg'] <= THRESHOLDS['full_LPIPS_delta_max'])
    observed = summary['masks']['a6_observed']
    a6_improves = observed['samples'] > 0 and observed['methods']['C2_delete37']['gt_srgb_MAE'] < observed['methods']['r3_delete37']['gt_srgb_MAE']
    road_checks = {}
    for name in ('fit','new_dark','original_shadow'):
        reference = summary['road_matched']['r3_delete37'][name]['linear_MAE']
        value = summary['road_matched']['C2_delete37'][name]['linear_MAE']
        ratio = value/max(reference,1e-12)
        road_checks[name] = dict(ratio=ratio,samples=summary['road_matched']['r3_delete37'][name]['samples'],domain='matched_conservative_road',numeric_tolerance_pass=summary['road_matched']['r3_delete37'][name]['samples']>0 and ratio <= THRESHOLDS['road_core_ratio_max'])
    freeze = dict(parent_unchanged=experiment.unchanged(), parent_identity_unchanged=parent_before == experiment.n.before,
        road_state_unchanged=road_before == tensor_state_hash(experiment.base.repair.state_dict()),
        actor_no_eval_updates=actor_before == tensor_state_hash(experiment.actor.state_dict()),
        opacity_fixed_min=float(experiment.actor.opacity.min()), opacity_fixed_max=float(experiment.actor.opacity.max()),
        observed_geometry_max_displacement_m=float((experiment.actor.xyz_canonical()-experiment.actor.canonical)[experiment.actor.observed].abs().max()) if experiment.actor.observed.any() else None,
        missing_geometry_max_displacement_m=float((experiment.actor.xyz_canonical()-experiment.actor.canonical).norm(dim=1).max()),
        full_state_restored=not errors, optimizer_updates=0)
    if contributor_binding['enabled']:
        gates=experiment.actor.gate_raw.sigmoid()
        freeze['actor_gate_candidates']=len(gates)
        freeze['actor_gate_min_mean_max']=[float(gates.min()),float(gates.mean()),float(gates.max())]
        freeze['contributor_parent_matches']=contributor_binding['parent_matches']
    write(dest / 'freeze_audit.json', freeze)
    freeze_pass = all(freeze[key] for key in ('parent_unchanged','parent_identity_unchanged','road_state_unchanged','actor_no_eval_updates','full_state_restored'))
    acceptance = dict(full_vs_original_native_delta=deltas, full_numeric_checks=full_checks,
        full_numeric_pass=all(full_checks.values()), a6_observed_delete_improves=a6_improves,
        road_core_checks=road_checks, freeze_restore_pass=freeze_pass,
        automatic_numeric_checks_pass=all(full_checks.values()) and a6_improves and all(v['numeric_tolerance_pass'] for v in road_checks.values()) and freeze_pass,
        human_visual_verdict=None, visual_review_pending=True, quality_accepted=False)
    acceptance['engineering_evaluation_complete'] = criterion is not None and freeze_pass and len(rows) == 49
    acceptance['missing_required_metrics'] = [] if criterion is not None else ['lpips_vgg']
    acceptance['scientific_failed_full_metrics'] = [name for name, passed in full_checks.items()
        if not passed and (name != 'lpips' or deltas['lpips_vgg'] is not None)]
    acceptance['formal_candidate_eligible'] = (not summary['diagnostic_only'] and
        summary['asset_formal_eligible'] is True and summary['checkpoint_step'] == settings['formal_steps'])
    acceptance['automatic_numeric_checks_pass'] &= acceptance['formal_candidate_eligible']
    summary['acceptance'] = acceptance
    summary['freeze_audit'] = freeze
    write(dest / 'summary.json', summary)
    write(dest / 'metrics.json', dict(summary=summary, rows=rows))
    print(json.dumps(dict(stage='COMPLETE', acceptance=acceptance, full_image=summary['full_image'])), flush=True)
    del criterion, experiment
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
