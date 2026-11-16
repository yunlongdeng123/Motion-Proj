"""Three-frame GPU attribution only; no optimization or asset/runner changes."""
from pathlib import Path
import argparse
import itertools
import json
import time

import numpy as np
from PIL import Image

import closedloop as loop


FRAMES = [86, 110, 179]
TAG_MASKS = ['target', 'strong_supported', 'unsupported_target', 'K', 'background', 'black21', 'white8']


def covariance(values, index, inverse):
    if values['cov3D_precomp'] is not None:
        packed = values['cov3D_precomp'][index].detach().cpu().numpy().astype(float)
        if packed.shape == (3, 3):
            cov = packed
        else:
            a, b, c, d, e, f = packed
            cov = np.array([[a, b, c], [b, d, e], [c, e, f]])
    else:
        scale = values['scales'][index].detach().cpu().numpy().astype(float)
        w, x, y, z = values['rotations'][index].detach().cpu().numpy().astype(float)
        norm = np.linalg.norm([w, x, y, z]); w, x, y, z = np.array([w, x, y, z]) / norm
        r = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                      [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                      [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        cov = r @ np.diag(scale**2) @ r.T
    return inverse[:3, :3] @ cov @ inverse[:3, :3].T


def intersects_ellipsoid_obb(center, cov, box):
    inv = box['inverse']; point = center @ inv[:3, :3].T + inv[:3, 3]
    half = box['half']; box_cov = inv[:3, :3] @ (9 * cov) @ inv[:3, :3].T
    if (abs(point) <= half).all(): return True
    if (abs(point) > half + np.sqrt(np.maximum(np.diag(box_cov), 0))).any(): return False
    q = np.linalg.pinv(box_cov, rcond=1e-12)
    # Exact 3-D convex box-constrained quadratic via all free/face/edge/corner sets.
    for state in itertools.product((-1, 0, 1), repeat=3):
        fixed = np.flatnonzero(state); free = np.flatnonzero(np.asarray(state) == 0)
        candidate = point.copy()
        if len(fixed): candidate[fixed] = half[fixed] * np.asarray(state)[fixed]
        if len(free) and len(fixed):
            candidate[free] = point[free] - np.linalg.pinv(q[np.ix_(free, free)]) @ q[np.ix_(free, fixed)] @ (candidate[fixed]-point[fixed])
        if len(free) and (abs(candidate[free]) > half[free] + 1e-8).any(): continue
        delta = candidate-point
        if delta @ q @ delta <= 1 + 1e-7: return True
    return False


def geometry_record(values, index, inverse, boxes, current_key):
    model = values['means3D'][index].detach().cpu().numpy().astype(float)
    world = model @ inverse[:3, :3].T + inverse[:3, 3]
    cov = covariance(values, index, inverse)
    principal = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))
    center_inside, ellipsoid_inside = False, False
    current = None
    for box in boxes:
        local = world @ box['inverse'][:3, :3].T + box['inverse'][:3, 3]
        inside = bool((abs(local) <= box['half'] + .25).all())
        center_inside |= inside
        if not ellipsoid_inside:
            ellipsoid_inside = intersects_ellipsoid_obb(world, cov, box)
        if box['key'] == current_key:
            current = dict(target_local_xyz_m=local.tolist(), height_above_target_bottom_m=float(local[2]+box['half'][2]),
                           center_in_current_OBB_plus025m=inside,
                           ellipsoid_3sigma_intersects_current_OBB=intersects_ellipsoid_obb(world, cov, box))
    return dict(world_xyz_m=world.tolist(), principal_sigma_m=principal.tolist(),
        center_in_swept_OBB_plus025m=center_inside, ellipsoid_3sigma_intersects_swept_OBB=ellipsoid_inside,
        **(current or {}))


def main(args):
    import torch
    torch.set_num_threads(2)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('preserve old diagnostic output')
    out.mkdir(parents=True, exist_ok=True)
    path = loop.active_manifest(); scene, manifest = loop.load_scene(path)
    complete = loop.read(loop.C/'formal/completed.json')
    state = torch.load(complete['checkpoint'], map_location='cpu', weights_only=False)
    assert state['step'] == 1000 and state['binding'] == loop.binding_for(path, manifest.road)
    ids_np = np.load(loop.C/'candidates.npz')['ids']
    assert np.array_equal(state['contributors'].numpy(), ids_np)
    ids = state['contributors'].to('cuda'); gates = state['gate'].to('cuda').sigmoid()
    ns = len(scene.static.get_xyz); nd = len(scene.dynamic.get_xyz) if scene._include_dynamic else 0
    candidate = np.zeros(ns, bool); candidate[ids_np] = True
    inverse = np.linalg.inv(np.loadtxt(manifest.data['world_to_model']))
    scale = float(np.load(manifest.road)['meter_to_model'])
    track = next(a['track_id'] for a in manifest.data['actors'] if a['actor_id'] == 10)
    boxes = []
    for frame in manifest.data['frames']:
        for box in loop.read(frame['observation'])['boxes']:
            if box['track'] == track:
                boxes.append(dict(key=frame['key'], inverse=np.linalg.inv(np.asarray(box['world_from_actor']).reshape(4, 4)), half=np.asarray(box['dimensions'])/2))
    frame_data = {f['key']: f for f in manifest.data['frames']}
    summaries = []; started = time.monotonic()
    for key in FRAMES:
        with torch.no_grad():
            values, owners, camera, _ = scene._rows(key, (10,), (), False)
            values = {name: value.detach() if hasattr(value, 'detach') else value for name, value in values.items()}
            values['opacities'] = values['opacities'].index_copy(0, ids, values['opacities'][ids]*gates[:, None])
            values['mask'] = values['mask'] & ~scene._contains(owners, (10,))
        with Image.open(frame_data[key]['instance_mask']) as im:
            labels = np.asarray(im.resize((camera.image_width, camera.image_height), Image.Resampling.NEAREST)).copy()
        target = labels == 10; K = (labels > 0) & ~target
        road_provider = next(p for p in scene._providers if p.kind == 'road')
        with torch.no_grad():
            rv = road_provider.rows(key, camera, scene.pipe)
            road = scene._bindings().raster(camera, scene.pipe, 0, rv, scene._background)
            road_alpha = road['alpha'][0].cpu().numpy(); road_reference = (road['rgb']/road['alpha'].clamp_min(.01)).cpu().numpy()
        n = len(values['means3D']); features = values['features'].new_zeros(values['features'].shape)
        features[:, :2] = values['features'][:, :2]
        ranges = [slice(0, ns), slice(ns, ns+nd), slice(ns+nd, n)]
        for j, sl in enumerate(ranges):
            features[sl, 2+3*j:5+3*j] = values['colors_precomp'][sl]
            features[sl, 11+j] = 1
        features[:ns, 14] = torch.as_tensor(candidate, device='cuda')
        features[:ns, 15] = ~torch.as_tensor(candidate, device='cuda')
        features[:, 17:24] = 1
        features.requires_grad_(True)
        pack = scene._bindings().raster(camera, scene.pipe, 0, dict(values, features=features), scene._background)
        with torch.no_grad():
            env = (1-pack['alpha'])*scene.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
            rgb = pack['rgb'] + env
            parts = [pack['extra'][i*3:i*3+3] for i in range(3)]
            error = float((sum(parts)+env-rgb).abs().max()); assert error < 2e-5
            alphas = [pack['extra'][9+i].cpu().numpy() for i in range(3)]
            rgb_np = rgb.cpu().numpy(); parts_np = [p.cpu().numpy() for p in parts]
            supported = road_alpha > .9
            residual = np.max(abs(rgb_np-road_reference), axis=0)
            masks = dict(target=target, strong_supported=target&supported&(residual>8/255), unsupported_target=target&~supported,
                         K=K, background=~target&~K, black21=labels==21, white8=labels==8)
            y = np.where(target)[0]; split = (float(y.min())+float(y.max()))/2 if len(y) else 0
            yy = np.arange(target.shape[0])[:, None]
            masks.update(target_upper=target&(yy<=split), target_lower=target&(yy>split))
            domains = {}
            for name, mask in masks.items():
                denom = int(mask.sum())
                domains[name] = dict(pixels=denom, road_only_supported_fraction=float(supported[mask].mean()) if denom else None,
                    alpha={label:float(a[mask].mean()) if denom else None for label,a in zip(['static','dynamic','road'],alphas)},
                    RGB={label:parts_np[i][:,mask].mean(1).tolist() if denom else None for i,label in enumerate(['static','dynamic','road'])},
                    selected_static_alpha=float(pack['extra'][12].cpu().numpy()[mask].mean()) if denom else None,
                    unselected_static_alpha=float(pack['extra'][13].cpu().numpy()[mask].mean()) if denom else None)
        objective = sum(pack['extra'][15+i][torch.as_tensor(masks[name], device='cuda')].sum() for i,name in enumerate(TAG_MASKS))
        grad = torch.autograd.grad(objective, features)[0][:, 17:24].detach().cpu().numpy()
        fixed = owners.cpu().numpy(); native = np.concatenate([scene._source_s.cpu().numpy(), scene._source_d.cpu().numpy(), np.zeros(n-ns-nd, np.int64)])
        colors = values['colors_precomp'].cpu().numpy()
        weight = grad[:, 0] * np.mean(abs(colors), axis=1)
        dynamic_matrix = []
        for f, s in sorted(set(zip(fixed[ns:ns+nd].tolist(), native[ns:ns+nd].tolist()))):
            ix = np.flatnonzero((fixed[ns:ns+nd]==f)&(native[ns:ns+nd]==s))+ns
            if grad[ix,0].sum() <= 1e-4: continue
            dynamic_matrix.append(dict(fixed_owner=int(f), native_source_owner=int(s), rows=int(len(ix)),
                target_alpha_mass=float(grad[ix,0].sum()), target_RGB_mass=float(weight[ix].sum()),
                target_mean_alpha=float(grad[ix,0].sum()/max(int(target.sum()),1))))
        dynamic_matrix.sort(key=lambda r:-r['target_alpha_mass'])
        print(json.dumps(dict(event='FAST_ATTRIBUTION',raw=key,
            domains={k:domains[k] for k in ['target','target_upper','target_lower']},
            dynamic_owner_matrix=dynamic_matrix,
            dynamic_source10_fixed_not10_mean_alpha=float(grad[(np.arange(n)>=ns)&(np.arange(n)<ns+nd)&(native==10)&(fixed!=10),0].sum()/max(int(target.sum()),1)))),flush=True)
        major = []
        for branch, sl, limit in [('static',slice(0,ns),35),('dynamic',slice(ns,ns+nd),30)]:
            rank = np.argsort(-weight[sl])[:limit]+(sl.start or 0)
            for index in rank:
                if grad[index,0] <= 1e-4: continue
                item = dict(branch=branch, row_id=int(index if branch=='static' else index-ns),
                    fixed_owner=int(fixed[index]), native_source_owner=int(native[index]),
                    in_static_candidate=bool(candidate[index]) if branch=='static' else False,
                    current_model_depth=float(values['features'][index,0]), current_camera_depth_m=float(values['features'][index,0])/scale,
                    opacity_after_formal=float(values['opacities'][index,0]), RGB=colors[index].tolist(),
                    alpha_T_mass={name:float(grad[index,i]) for i,name in enumerate(TAG_MASKS)}, target_RGB_mass=float(weight[index]))
                item.update(geometry_record(values, int(index), inverse, boxes, key))
                major.append(item)
        summary = dict(raw=key, domains=domains, dynamic_fixed_vs_native_owner=dynamic_matrix,
            dynamic_source10_fixed_not10_target_mean_alpha=float(grad[(np.arange(n)>=ns)&(np.arange(n)<ns+nd)&(native==10)&(fixed!=10),0].sum()/max(int(target.sum()),1)),
            conservation_max_abs=error, major_rows=major)
        summaries.append(summary)
        np.savez_compressed(out/f'{key:03d}_arrays.npz',rgb=rgb_np,static_rgb=parts_np[0],dynamic_rgb=parts_np[1],road_rgb=parts_np[2],
                            static_alpha=alphas[0],dynamic_alpha=alphas[1],road_alpha=alphas[2],road_only_alpha=road_alpha,target=target,
                            strong_supported=masks['strong_supported'],unsupported_target=masks['unsupported_target'])
        loop.write(out/f'{key:03d}.json',summary)
        print(json.dumps(dict(raw=key, domains={k:domains[k] for k in ['target','target_upper','target_lower']},dynamic_owner_matrix=dynamic_matrix,
            dynamic_source10_fixed_not10_mean_alpha=summary['dynamic_source10_fixed_not10_target_mean_alpha'])),flush=True)
        del pack, features, values, grad, objective, rgb, parts
    scene.assert_persistent_unchanged()
    loop.write(out/'summary.json',dict(complete=True,frames=FRAMES,manifest=str(path),checkpoint=complete['checkpoint'],
        checkpoint_sha256=loop.sha(complete['checkpoint']),candidate_rows=len(ids_np),road_support_threshold=.9,
        strong_supported_threshold_RGB8=8,ellipsoid_test='exact box-constrained Mahalanobis minimum for 3sigma ellipsoid; swept boxes without .25 padding',
        center_test='swept measured OBB plus existing .25m',optimizer_updates=0,assets_modified=False,
        seconds=time.monotonic()-started,rows=summaries,GPU='keep_on'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=loop.C/'remaining_diag')
    main(parser.parse_args())
