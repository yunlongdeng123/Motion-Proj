"""Finite read-only native alpha*T attribution, after the registered review.

The color adjoint sums actual per-Gaussian raster contributions inside a pixel
ROI. Its sum is checked against rasterized alpha and its weighted RGB against
the original un-clamped RGB. No optimizer steps or owner edits are made.
"""
from pathlib import Path
import datetime, gc, hashlib, json, os, sys, time, traceback
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OUT = ROOT / 'baseline_engineering_20261018'
TASK = ROOT / 'identity_completion_20261018'
TARGET = OUT / 'static_attribution'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    temp = p.with_suffix(p.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(p)


def status(state, **kw):
    write(OUT / 'attribution_queue_status.json', dict(state=state,
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(), **kw))


def native_gradient_check(cam, static, cfg, bg, env, mask):
    import torch
    import numpy as np
    from renderer import render
    from gaussian_renderer import render as native_render
    names = ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']
    params = [getattr(static,n) for n in names]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    roi = torch.from_numpy(mask.astype(np.float32)).cuda()[None]
    assert roi.count_nonzero() > 0
    try:
        adapter = render(cam,static,None,cfg,bg,env_map=env,render_separate=False)
        native = native_render(cam,static,None,cfg,bg,env_map=env)
        value_error = float((adapter['render']-native['render']).abs().max())
        assert value_error < 2e-4
        gradients = []
        for p in [adapter,native]:
            gradients.append(torch.autograd.grad((p['render']*roi).sum()/(3*roi.sum()),params,allow_unused=True))
        checks = []
        for name, actual, reference in zip(names,*gradients):
            if actual is None or reference is None:
                checks.append(dict(parameter=name,both_disconnected=actual is None and reference is None,
                    passed=actual is None and reference is None))
                continue
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
            maximum = float((actual-reference).abs().max())
            ref_norm = torch.linalg.vector_norm(reference)
            relative_l2 = float(torch.linalg.vector_norm(actual-reference)/ref_norm.clamp_min(1e-20))
            passed = finite and bool(torch.allclose(actual,reference,atol=3e-6,rtol=3e-3)) and (float(ref_norm)<1e-12 or relative_l2<3e-3)
            checks.append(dict(parameter=name,finite=finite,max_abs=maximum,reference_L2=float(ref_norm),
                relative_L2=relative_l2,atol=3e-6,rtol=3e-3,passed=passed))
        result = dict(status='PASSED_NATIVE_STATIC_FORWARD_AND_GRADIENT' if all(v['passed'] for v in checks) else 'NATIVE_STATIC_GRADIENT_PARITY_REQUIRES_REVIEW',
            frame_id=Path(cam.image_name).stem, actual_full_static_RGB_max_abs=value_error, gradient_checks=checks,
            objective='mean rendered RGB over union of a6/a37 visible binary masks; no clean background target',
            native_renderer_sha256=hashlib.sha256(Path(sys.modules['gaussian_renderer'].__file__).read_bytes()).hexdigest(),
            optimizer_updates=0, quality_pass=False)
        write(TARGET/'native_static_gradient.json',result)
        return result
    finally:
        for p,flag in zip(params,flags):
            p.requires_grad_(flag)


def gpu_work():
    import torch
    import numpy as np
    from PIL import Image
    from runtime import load_models
    from renderer import render, prepare_static
    torch.set_num_threads(2)
    spec = read(TASK / 'spec5.json')
    files = dict(GN11=Path(spec['background_checkpoint']),
        old3=ROOT / 'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt',
        old5=TASK / 'train5/checkpoint_030000.pt')
    before = {name:[p.stat().st_size,p.stat().st_mtime_ns] for name,p in files.items()}
    env_path = Path(spec['environment_checkpoint'])
    before_env = [env_path.stat().st_size, env_path.stat().st_mtime_ns]
    base = torch.load(str(files['GN11']), map_location='cpu', weights_only=False, mmap=True)[0]
    scale = .1
    original_sigma = base[4].detach().exp().amax(-1).cuda()
    original_dc = base[2].detach().cuda()
    original_rest = base[3].detach().cuda()
    del base
    with np.load(TASK / 'prepared5/instances.npz', allow_pickle=False) as bundle:
        actual_scale = float(np.linalg.norm(bundle['normalization'][:3,:3],axis=0).mean())
        assert abs(actual_scale - scale) < 1e-12
        frame_ids = bundle['frame_ids'].tolist()
        poses, sizes = bundle['poses'].copy(), bundle['box_size'].copy()
    bg = torch.zeros(3, device='cuda')
    records = []
    native_gradient = None
    for name in ['GN11','old3','old5']:
        status('STATIC_ATTRIBUTION_RUNNING', model=name, optimizer_updates=0)
        ck = files['old5'] if name == 'GN11' else files[name]
        saved = torch.load(ck, map_location='cuda', weights_only=False)
        if name == 'GN11':
            native, step = torch.load(files['GN11'], map_location='cuda', weights_only=False)
            assert step == 30000
            environment, env_step = torch.load(env_path, map_location='cuda', weights_only=False)
            assert env_step == 30000
            saved['background'], saved['environment'] = native, environment
            del native, environment
        scratch = TARGET / (name + '_load')
        scratch.mkdir()
        cfg, scene, static, instances, optimizer, env, exclusion = load_models(saved['spec'], scratch, restore=saved)
        assert len(static.get_xyz) == len(original_sigma)
        for parameter in static.optimizer.param_groups:
            for p in parameter['params']:
                p.requires_grad_(False)
        for p in env.parameters():
            p.requires_grad_(False)
        for raw in [107,128,137]:
            fid = '0008%03d' % raw
            cam = next(c for c in scene.getTrainCameras() if Path(c.image_name).stem == fid)
            row = frame_ids.index(fid)
            labels = np.asarray(Image.open(TASK / 'prepared5/instance_masks' / (fid + '.png')))
            if name == 'GN11' and raw == 128:
                native_gradient = native_gradient_check(cam,static,cfg,bg,env,(labels==4)|(labels==5))
            with torch.no_grad():
                actual = render(cam, static, None, cfg, bg, env_map=env, render_separate=False)
                prepared = prepare_static(cam, static, cfg)
                colors = prepared['colors_precomp']
                sigma = static.get_scaling.amax(-1)
                growth = sigma / original_sigma.clamp_min(1e-30)
                appearance_delta = (static._features_dc - original_dc).flatten(1).norm(dim=1) + (static._features_rest - original_rest).flatten(1).norm(dim=1)
                folder = TARGET / name / fid
                folder.mkdir(parents=True)
                pixels = (actual['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                Image.fromarray(pixels).save(folder / 'static.png')
            query_colors = torch.zeros((len(static.get_xyz),3), device='cuda', requires_grad=True)
            query = render(cam, static, None, cfg, bg, override_color=query_colors, render_separate=False)
            regions = []
            for slot, actor in [(3,6),(4,37)]:
                body = labels == slot + 1
                if not body.any():
                    continue
                yy, xx = np.where(body)
                lower = np.zeros_like(body)
                width, height = int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)
                x0, x1 = max(0,int(xx.min())-width//4), min(body.shape[1],int(xx.max())+width//4+1)
                y0, y1 = int(yy.max())+1, min(body.shape[0],int(yy.max())+max(4,height//3)+1)
                lower[y0:y1,x0:x1] = True
                for region, mask in [('visible_body_mask',body),('below_body_rectangle',lower)]:
                    if mask.any():
                        regions.append((actor,slot,region,mask))
            for index, (actor,slot,region,mask) in enumerate(regions):
                roi = torch.from_numpy(mask.astype(np.float32)).cuda()[None]
                # Sum of the three color derivatives yields alpha*T pixel mass.
                gradient = torch.autograd.grad((query['raw_render']*roi).sum()/3., query_colors,
                    retain_graph=index < len(regions)-1)[0]
                weight = gradient.sum(-1)
                alpha_sum = (actual['alpha']*roi).sum()
                mass = weight.sum()
                alpha_error = float((mass-alpha_sum).abs())
                weighted_rgb = (weight[:,None]*colors).sum(0)
                raster_rgb = (actual['raw_render']*roi).sum((1,2))
                rgb_error = float((weighted_rgb-raster_rgb).abs().max())
                assert float(weight.min()) >= -1e-7
                assert alpha_error <= .002 + 3e-4*float(alpha_sum), (name,fid,actor,region,'alpha',alpha_error)
                assert rgb_error <= .004 + 5e-4*float(raster_rgb.abs().max()), (name,fid,actor,region,'rgb',rgb_error)
                count = min(128,len(weight))
                top = weight.topk(count).indices
                top = top[weight[top] > 0]
                pose = torch.as_tensor(poses[row,slot], device='cuda', dtype=static.get_xyz.dtype)
                half = torch.as_tensor(sizes[row,slot], device='cuda', dtype=static.get_xyz.dtype)/2
                inside = (((prepared['means3D']-pose[:3,3]) @ pose[:3,:3]).abs() <= half).all(-1)
                denominator = mass.clamp_min(1e-20)
                points = []
                for point in top.tolist():
                    points.append(dict(row=point, alpha_T_pixel_mass=float(weight[point]),
                        current_sigma_max_m=float(sigma[point]/scale), sigma_growth_from_GN11=float(growth[point]),
                        current_SH_RGB_unclamped=colors[point].cpu().tolist(),
                        SH_coefficient_delta_L2=float(appearance_delta[point]),
                        current_center_in_target_OBB=bool(inside[point])))
                record = dict(model=name, frame_id=fid, actor=actor, region=region, ROI_pixels=int(mask.sum()),
                    alpha_T_pixel_mass=float(mass), raster_alpha_sum=float(alpha_sum), alpha_sum_abs_error=alpha_error,
                    weighted_RGB_sum=weighted_rgb.cpu().tolist(), raster_raw_RGB_sum=raster_rgb.cpu().tolist(), RGB_sum_max_abs_error=rgb_error,
                    alpha_mass_from_centers_outside_target_OBB_fraction=float(weight[~inside].sum()/denominator),
                    alpha_mass_from_sigma_growth_over2x_fraction=float(weight[growth>2].sum()/denominator),
                    alpha_mass_from_sigma_growth_over10x_fraction=float(weight[growth>10].sum()/denominator),
                    top128_alpha_mass_fraction=float(weight[top].sum()/denominator), top_points=points)
                records.append(record)
                Image.fromarray((mask.astype(np.uint8)*255)).save(folder / ('a%d_%s_ROI.png'%(actor,region)))
                write(TARGET / 'progress.json', dict(model=name, frame_id=fid, records=len(records)))
            del query, query_colors, actual, prepared, colors, sigma, growth, appearance_delta
        del saved, cfg, scene, static, instances, optimizer, env, exclusion
        gc.collect(); torch.cuda.empty_cache()
    assert before == {name:[p.stat().st_size,p.stat().st_mtime_ns] for name,p in files.items()}
    assert before_env == [env_path.stat().st_size,env_path.stat().st_mtime_ns]
    receipt = dict(status='PASSED_ACTUAL_STATIC_ALPHA_T_MASS_AND_RGB_RECONCILIATION_REVIEW_PENDING',
        frames=[107,128,137], actors=[6,37], models=['GN11','old3','old5'], records=records,
        method='native rasterizer color adjoint; per-point derivatives checked against actual alpha mass and un-clamped RGB',
        native_static_gradient_check=native_gradient,
        checkpoint_size_mtime_unchanged=True, checkpoint_bindings={k:str(p) for k,p in files.items()},
        optimizer_updates=0, ownership_changes=0, new_matched_GN11_or_R15_parameters_used=False,
        ROI_is_not_a_clean_road_target=True, OBB_membership_is_not_ownership=True,
        pruning_counterfactual_or_quality_improvement_proven=False, human_verdict=None)
    write(TARGET / 'receipt.json',receipt)
    return receipt


def main():
    marker = OUT / 'attribution_queue_started.json'
    if marker.exists():
        raise FileExistsError('no duplicate attribution queue')
    write(marker,dict(pid=os.getpid(), optimizer_updates=0, maximum_models=3, maximum_frames_each=3))
    start = time.monotonic()
    while True:
        q = read(OUT / 'review_queue_status.json')
        if q['state'] == 'ENGINEERING_REVIEW_COMPLETE_AWAITING_ACTUAL_LOSS_RESTORE_AND_VISUAL_DECISION':
            break
        if q['state'].startswith('FAILED'):
            raise RuntimeError('review failed; preserve assets and do not bypass its gate')
        if time.monotonic()-start > 6*3600:
            raise TimeoutError('finite six-hour wait exceeded')
        status('WAITING_FOR_REGISTERED_REVIEW_NO_GPU_ACQUIRED', review_state=q['state'], optimizer_updates=0)
        time.sleep(25)
    sys.path.insert(0,str(TASK/'infra'))
    from priority import require_gpu_slot
    slot,admission = require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    try:
        if TARGET.exists():
            raise FileExistsError('partial attribution requires explicit diagnosis')
        TARGET.mkdir()
        result = gpu_work()
        status('STATIC_ATTRIBUTION_COMPLETE_AWAITING_REVIEW', optimizer_updates=0,
            receipt=str(TARGET/'receipt.json'), records=len(result['records']))
    finally:
        slot.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('FAILED_STATIC_ATTRIBUTION_PRESERVED_NO_RESTART',error=traceback.format_exc(),optimizer_updates=0)
        raise
