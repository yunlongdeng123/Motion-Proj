"""Zero-update BUILD86/110 source attribution with exact shared-sort RGB/alpha.

First pass only: no source or context row is retired. ABI follows the existing
scene008 render_layers.py: first2 depth/alpha, extra22 attribution channels.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import numpy as np

LAYERS = ('road', 'native_static_owner21', 'native_dynamic_owner21', 'native_owner10',
          'native_static_other_actor', 'native_dynamic_other_actor',
          'native_static_background', 'native_dynamic_background', 'a20_old_source0', 'a20_T_source1')
RAW110_ROIS = dict(strip=[510, 550, 635, 585], shadow=[520, 578, 682, 620], vehicle_guard=[477, 414, 651, 577])
RAW086_ROIS = dict(circled=[514, 511, 653, 579], strip=[524, 532, 627, 556],
                   shadow=[538, 553, 658, 583], vehicle_guard=[490, 411, 642, 552])


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def native_layer_ids(gids, owners, road, ns, nd):
    """Native branch derives from global index ranges, never semantic owner."""
    result = np.full(len(gids), -1, np.int8)
    is_road = road >= 0; static = (gids >= 0) & (gids < ns) & ~is_road
    dynamic = (gids >= ns) & (gids < ns+nd) & ~is_road
    result[is_road] = 0; result[(static | dynamic) & (owners == 10)] = 3
    for branch, target, other, background in [(static, 1, 4, 6), (dynamic, 2, 5, 7)]:
        result[branch & (owners == 21)] = target
        result[branch & (owners > 0) & (owners != 21) & (owners != 10)] = other
        result[branch & (owners <= 0)] = background
    if (result < 0).any(): raise ValueError('Native global row/road ownership falls outside proved topology')
    return result


def validate_rois(rois):
    for raw, views in rois.items():
        if int(raw) not in (86, 110): raise ValueError('Only BUILD86/110 diagnostic ROIs are allowed')
        for name, item in views.items():
            box = item['xyxy'] if isinstance(item, dict) else item
            if len(box) != 4 or any(type(v) is not int for v in box): raise ValueError('ROI needs integer xyxy')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= 1152 and 0 <= y0 < y1 <= 768): raise ValueError('ROI outside native1152x768')


def cpu_self_check():
    gids = np.array([1, 11, 2, 12, 20]); owner = np.array([21, 21, 8, 0, 0]); road = np.array([-1, -1, -1, -1, 0])
    assert np.array_equal(native_layer_ids(gids, owner, road, 10, 10), [1, 2, 4, 7, 0])
    validate_rois({'086': RAW086_ROIS, '110': RAW110_ROIS})
    return dict(passed=True, static_dynamic_uses_global_id_not_owner=True, explicit_ROIs_valid=True, GPU_used=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-self-check', action='store_true')
    for key in ('runtime-dir', 'checkpoint', 'actor', 'init', 'manifest', 'source-manifest', 'prior-contract', 'context-plan', 'path-map-file', 'roi-file', 'output'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--teacher', type=Path); p.add_argument('--path-map', action='append', default=[])
    p.add_argument('--expected-actor-sha-prefix', default='ab4fa110')
    p.add_argument('--top-rows', type=int, default=64)
    return p.parse_args()


class Audit:
    def __init__(self, args):
        self.args = args; self.started = time.monotonic(); self.vjp_calls = 0
        for key in ('runtime_dir', 'checkpoint', 'actor', 'init', 'manifest', 'source_manifest', 'prior_contract', 'output'):
            if getattr(args, key) is None: raise ValueError('--'+key.replace('_', '-')+' is required')
        if args.output.exists(): raise FileExistsError('Use a fresh attribution output directory')
        args.output.mkdir(parents=True)
        sys.path.insert(0, str(args.runtime_dir.resolve()))
        self.core = import_file('joint_core', args.runtime_dir/'joint_core.py')
        self.gpu = import_file('_a20_attribution_exact_runtime', args.runtime_dir/'joint_gpu.py')
        core = self.core; gpu = self.gpu; self.t = gpu.torch
        state = self.t.load(args.checkpoint, map_location='cpu', weights_only=False)
        if state['runtime_sha256'] != core.sha(args.runtime_dir/'joint_gpu.py'): raise ValueError('Exact checkpoint runtime mismatch')
        if state['phase'] != 'geometry' or state['step'] != 40: raise ValueError('Expected fixed G40 candidate checkpoint')
        actor_sha = core.sha(args.actor)
        if args.expected_actor_sha_prefix and not actor_sha.startswith(args.expected_actor_sha_prefix):
            raise ValueError('Actor asset does not match the requested frozen candidate')
        core.write(args.output/'training_contract.restored.json', state['config'])
        runtime_values = vars(args).copy(); runtime_values['asset'] = None
        runtime_args = SimpleNamespace(**runtime_values, mode='source_attribution', config=args.output/'training_contract.restored.json', resume=None, geometry_gate=None)
        self.i = core.load_inputs(runtime_args); self.rt = gpu.Runtime(runtime_args, self.i)
        self.opt = self.rt.optimizer('geometry'); self.rt.restore(state, self.opt)
        if not gpu.tree_equal(gpu.cpu(self.rt.actor.state_dict()), state['actor']): raise ValueError('Actor restore is not exact')
        if not gpu.tree_equal(gpu.cpu(self.opt.state_dict()), state['optimizer']): raise ValueError('Adam restore is not exact')
        asset = core.bank(args.actor); exported = self.rt.actor.export()
        for key in self.core.FIELDS + ('active', 'source_id', 'source_row_id'):
            if not np.array_equal(asset[key], exported[key]): raise ValueError('Actor/checkpoint mismatch: '+key)
        self.actor_before = gpu.cpu(self.rt.actor.state_dict()); self.Adam_before = gpu.cpu(self.opt.state_dict())
        self.source_paths = [args.runtime_dir/'joint_core.py', args.runtime_dir/'joint_gpu.py', args.checkpoint, args.actor]
        self.source_hashes = {str(p.resolve()): core.sha(p) for p in self.source_paths}
        self.ns = len(self.rt.e.scene.static.get_xyz)
        self.nd = 0 if self.rt.e.scene.dynamic is None else len(self.rt.e.scene.dynamic.get_xyz)
        if (self.ns, self.nd) != (self.rt.e.loader['static_rows'], self.rt.e.loader['dynamic_rows']):
            raise ValueError('Native model topology and persistent loader disagree')
        self.rois = self.prepare_rois()
        core.write(args.output/'fixed_diagnostic_rois.json', self.rois)
        core.write(args.output/'protocol.json', dict(schema='a20.scene_residual.source_attribution.v1',
            script=core.ref(__file__), exact_runtime=core.ref(args.runtime_dir/'joint_gpu.py'), checkpoint=core.ref(args.checkpoint),
            actor=core.ref(args.actor), actor_active_rows=int(self.rt.actor.active.sum()), BUILD=[86, 110], sealed_opened=[],
            native_topology=dict(static_rows=self.ns, dynamic_rows=self.nd), layers=list(LAYERS),
            ABI='first2 depth/alpha plus extra22; up to5 groups of premult RGB3+alpha in an identical shared-sort pass',
            source_RGB='actual per-row colors times shared transmittance/opacity; never divided by layer alpha',
            ablation='standalone on black with no environment, and without-group with original environment; rows/models unchanged',
            diagnostic_ROIs_are_ground_truth=False, vehicle_guard_is_RGB_truth=False,
            V_and_K='original fixed masks/labels; independent actual contribution evidence',
            optimizer_updates=0, cleanup_decisions_applied=False))

    def arr(self, tensor): return self.gpu.cpu(tensor).numpy()

    def prepare_rois(self):
        rois = {
            '086': {k: dict(xyxy=v, origin='root confirmed screenshot-to-raw86 SIFT registration; 906/917 inliers, median0.105px',
                         confirmed=True, role='diagnostic only; not supervision GT') for k, v in RAW086_ROIS.items()},
            '110': {k: dict(xyxy=v, origin='root fixed native1152x768 second-view ROI', confirmed=True,
                         role='diagnostic only; not supervision GT') for k, v in RAW110_ROIS.items()}}
        if self.args.roi_file:
            value = self.core.read(self.args.roi_file); supplied = value.get('views', value)
            if '86' in supplied and '086' not in supplied: supplied['086'] = supplied.pop('86')
            for raw, views in supplied.items(): rois[raw] = views
        validate_rois(rois)
        return rois

    def scene_rows(self, raw):
        rt = self.rt; t = self.t
        p, owners, road_index, gids = rt.context[raw]
        with t.no_grad():
            actor = rt.actor_rows(rt.actor, rt.e.poses[raw], rt.e.cameras[raw])
            rows = rt.e.scene._bindings().merge_rows(p, actor)
        g = self.arr(gids); owner = self.arr(owners); ri = self.arr(road_index)
        layer_native = native_layer_ids(g, owner, ri, self.ns, self.nd)
        road = ri >= 0
        road_mapping_exact = np.array_equal(g[road], self.ns+self.nd+ri[road])
        if not road_mapping_exact: raise ValueError('Native global road index and active road_index disagree')
        if road.any() and (ri[road].max() >= len(rt.e.active) or ri[road].min() < 0): raise ValueError('Road active index out of bounds')
        world_road = np.full(len(g), -1, np.int64); world_road[road] = rt.e.active[ri[road]]
        a = self.i['init']; n = len(g); na = len(a['xyz'])
        layer = np.r_[layer_native, 8+a['source_id']].astype(np.int8)
        scene_gid = np.r_[g, np.full(na, -1, np.int64)]
        lineage_gid = np.r_[g, a['native_global_row_id']]
        branch = np.where(g < self.ns, 0, np.where(g < self.ns+self.nd, 1, 2)).astype(np.int8)
        source_row = np.where(branch == 0, g, np.where(branch == 1, g-self.ns, world_road))
        metadata = dict(unified_packet_row=np.arange(n+na), layer_id=layer, scene_global_row_id=scene_gid,
            source_native_global_row_id=lineage_gid, native_branch=np.r_[branch, np.full(na, 3, np.int8)],
            source_row_id=np.r_[source_row, a['source_row_id']], owner=np.r_[owner, np.full(na, 21, np.int64)],
            road_active_index=np.r_[ri, np.full(na, -1, np.int64)], road_world_row_id=np.r_[world_road, np.full(na, -1, np.int64)],
            actor_source_id=np.r_[np.full(n, -1, np.int8), a['source_id']],
            retained_other_actor=np.r_[(owner > 0)&(owner != 21)&(owner != 10), np.zeros(na, bool)],
            protected_native_K=np.r_[np.isin(g, self.arr(rt.K_preserve)), np.zeros(na, bool)],
            actual_row_mask=self.arr(rows['mask']))
        cam = rt.e.cameras[raw]
        if (cam.image_width, cam.image_height) != (1152, 768): raise ValueError('Unexpected actual native camera resolution')
        recomputed_depth = rows['means3D']@cam.world_view_transform[:3, 2]+cam.world_view_transform[3, 2]
        depth_error = (recomputed_depth-rows['features'][:, 0]).abs()
        native_packet, native_owner, native_ri = rt.e.native_packets[raw]
        qa = dict(raw=raw, actual_image_size=[cam.image_width, cam.image_height], static_rows=self.ns, dynamic_rows=self.nd,
            road_global_active_index_exact=road_mapping_exact, road_world_rows=int(road.sum()),
            maximum_feature_camera_depth_error_model_units=float(depth_error.max()),
            original_native_packet_rows=len(native_owner), original_owner10_rows=int((native_owner == 10).sum()),
            current_context_owner10_rows=int((owners == 10).sum()), current_context_owner21_rows=int((owners == 21).sum()),
            world_view_transform=self.arr(cam.world_view_transform).tolist(), full_proj_transform=self.arr(cam.full_proj_transform).tolist(),
            camera_center=self.arr(cam.camera_center).tolist(), model_from_actor=self.arr(rt.e.poses[raw]).tolist(),
            labels_sampling=rt.domains[raw]['label_sampling'])
        return rows, metadata, t.as_tensor(layer, device='cuda', dtype=t.long), qa

    def raster(self, raw, rows, environment=True):
        rt = self.rt
        bg = rt.e.scene._background if environment else rt.blank
        out = rt.e.scene._bindings().raster(rt.e.cameras[raw], rt.e.scene.pipe, 0, rows, bg); rt.calls += 1
        rgb = out['rgb']+(1-out['alpha'])*rt.e.env[raw] if environment else out['rgb']
        return dict(rgb=rgb, alpha=out['alpha'].squeeze(), rgb_premultiplied=out['rgb'], extra=out['extra'])

    def save_image_set(self, raw, name, rgb, alpha=None):
        folder = self.args.output/'images'/f'raw{raw:03d}'
        self.rt.m.png(folder/(name+'.png'), rgb)
        if alpha is not None: self.rt.m.png(folder/(name+'_alpha.png'), alpha)
        for roi_name, item in self.rois[f'{raw:03d}' if raw == 86 else str(raw)].items():
            x0, y0, x1, y1 = item['xyxy'] if isinstance(item, dict) else item
            self.rt.m.png(folder/('roi_'+roi_name)/(name+'.png'), rgb[..., y0:y1, x0:x1])
            if alpha is not None: self.rt.m.png(folder/('roi_'+roi_name)/(name+'_alpha.png'), alpha[y0:y1, x0:x1])

    def domains(self, raw):
        domains = {}
        for name, item in self.rois[f'{raw:03d}' if raw == 86 else str(raw)].items():
            x0, y0, x1, y1 = item['xyxy'] if isinstance(item, dict) else item
            domain = self.t.zeros((768, 1152), device='cuda', dtype=self.t.bool); domain[y0:y1, x0:x1] = True
            domains[name] = domain
        domains['real_V'] = self.rt.domains[raw]['V']; domains['retained_K'] = self.rt.domains[raw]['K']
        return domains

    def row_contributions(self, raw, rows, metadata, domains):
        t = self.t; weights = t.ones(len(rows['means3D']), device='cuda', requires_grad=True)
        features = t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]
        features[:, 2] = weights; features[:, 3:6] = rows['colors_precomp']*weights[:, None]
        out = self.raster(raw, dict(rows, features=features)); data = {}; summaries = []
        for name, mask in domains.items():
            channels = [('alpha', out['extra'][0])]
            if name in ('strip', 'shadow'): channels += [(c, out['extra'][j+1]) for j, c in enumerate(('R', 'G', 'B'))]
            for channel, values in channels:
                g = t.autograd.grad(values[mask].sum(), weights, retain_graph=True)[0]
                if not t.isfinite(g).all() or (channel == 'alpha' and (g < -1e-7).any()): raise RuntimeError('Invalid actual row feature VJP')
                data[name+'_'+channel+'_raw'] = self.arr(g); self.vjp_calls += 1
                closure = abs(float(g.sum())-float(values[mask].sum()))
                summaries.append(dict(domain=name, channel=channel, pixels=int(mask.sum()), sum_feature_VJP_closure=closure))
        path = self.args.output/'evidence'/f'raw{raw:03d}_exact_row_contributions.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **metadata, **data)
        top = {}
        for name in domains:
            score = data[name+'_alpha_raw']; chosen = np.flatnonzero(score > 0)
            chosen = chosen[np.argsort(-score[chosen], kind='stable')][:self.args.top_rows]
            result = []
            for j in chosen.tolist():
                row = {k: (bool(v[j]) if v.dtype == np.bool_ else int(v[j])) for k, v in metadata.items()}
                row.update(layer=LAYERS[int(metadata['layer_id'][j])], ROI_alpha_pixel_sum=float(score[j]))
                if name in ('strip', 'shadow'): row['ROI_premult_RGB_sum'] = [float(data[name+'_'+c+'_raw'][j]) for c in ('R', 'G', 'B')]
                row['real_V_alpha_pixel_sum'] = float(data['real_V_alpha_raw'][j]); row['retained_K_alpha_pixel_sum'] = float(data['retained_K_alpha_raw'][j])
                result.append(row)
            top[name] = result
        self.core.write(self.args.output/'evidence'/f'raw{raw:03d}_top_rows.json', dict(top_rows=top, all_exact_rows=self.core.ref(path), VJP=summaries))
        return dict(file=self.core.ref(path), VJP=summaries)

    def run_view(self, raw):
        t = self.t; rows, metadata, layer_ids, qa = self.scene_rows(raw); domains = self.domains(raw)
        contributions = {}; parity = []; baseline = None
        with t.no_grad():
            original = self.rt.render(raw=raw)
            for start in range(0, len(LAYERS), 5):
                batch = range(start, min(start+5, len(LAYERS)))
                features = t.zeros_like(rows['features']); features[:, :2] = rows['features'][:, :2]
                for j, layer in enumerate(batch):
                    mark = (layer_ids == layer).float()
                    features[:, 2+4*j] = mark; features[:, 3+4*j:6+4*j] = rows['colors_precomp']*mark[:, None]
                out = self.raster(raw, dict(rows, features=features))
                if baseline is None: baseline = {k: self.arr(out[k]) for k in ('rgb', 'alpha', 'rgb_premultiplied')}
                rgb_parity = float((out['rgb']-original['rgb']).abs().max()); alpha_parity = float((out['alpha']-original['alpha']).abs().max())
                parity.append(dict(layer_batch_start=start, RGB_max=rgb_parity, alpha_max=alpha_parity))
                if rgb_parity > 5e-6 or alpha_parity > 1e-7: raise RuntimeError('Tagged shared-sort pass changed actual scene RGB/alpha')
                for j, layer in enumerate(batch):
                    contributions[LAYERS[layer]] = dict(alpha=self.arr(out['extra'][4*j]), RGB=self.arr(out['extra'][4*j+1:4*j+4]))
            alpha_sum = sum(v['alpha'] for v in contributions.values()); rgb_sum = sum(v['RGB'] for v in contributions.values())
            background = self.arr(self.rt.e.scene._background)[:, None, None]*(1-baseline['alpha'])
            environment = self.arr(self.rt.e.env[raw])*(1-baseline['alpha'])
            closure = dict(alpha_max=float(np.abs(alpha_sum-baseline['alpha']).max()),
                RGB_max=float(np.abs(rgb_sum+background+environment-baseline['rgb']).max()))
            if closure['alpha_max'] > 5e-5 or closure['RGB_max'] > 5e-5: raise RuntimeError('Shared-source attribution closure failed')
            self.save_image_set(raw, 'shared', baseline['rgb'], baseline['alpha'])
            actor = self.rt.render(raw=raw, actor_only=True); self.save_image_set(raw, 'actor_only', actor['rgb'], actor['alpha'])
            self.save_image_set(raw, 'observed_RGB', self.rt.domains[raw]['rgb'])
            records = []
            for j, name in enumerate(LAYERS):
                layer = contributions[name]; keep = layer_ids == j; valid_rows = int((rows['mask'] & keep).sum())
                self.save_image_set(raw, name+'_shared_source', layer['RGB'], layer['alpha'])
                self.save_image_set(raw, name+'_source_on_grey', layer['RGB']+.5*(1-layer['alpha']))
                if valid_rows:
                    only = self.raster(raw, dict(rows, mask=rows['mask'] & keep), environment=False)
                    without = self.raster(raw, dict(rows, mask=rows['mask'] & ~keep))
                    alone_rgb = self.arr(only['rgb']); alone_alpha = self.arr(only['alpha'])
                    without_rgb = self.arr(without['rgb']); without_alpha = self.arr(without['alpha'])
                else:
                    alone_rgb = np.zeros_like(baseline['rgb']); alone_alpha = np.zeros_like(baseline['alpha'])
                    without_rgb = baseline['rgb']; without_alpha = baseline['alpha']
                self.save_image_set(raw, name+'_standalone', alone_rgb, alone_alpha)
                self.save_image_set(raw, 'without_'+name, without_rgb, without_alpha)
                delta = np.abs(without_rgb-baseline['rgb']).max(0)
                self.save_image_set(raw, 'without_'+name+'_difference', np.stack([np.clip(delta*5, 0, 1), np.clip(delta*2, 0, 1), np.zeros_like(delta)]))
                statistics = {}
                for domain_name, mask_tensor in domains.items():
                    mask = self.arr(mask_tensor); n = int(mask.sum())
                    statistics[domain_name] = dict(pixels=n, shared_source_alpha_mean=float(layer['alpha'][mask].mean()),
                        shared_source_alpha_max=float(layer['alpha'][mask].max()), standalone_alpha95_fraction=float((alone_alpha[mask]>=.95).mean()),
                        opaque_standalone_but_shared_alpha_below05_pixels=int(((alone_alpha>=.95)&(layer['alpha']<.05)&mask).sum()),
                        premult_source_RGB_mean=layer['RGB'][:, mask].mean(1).tolist(),
                        without_group_RGB_difference_max=float(np.abs(without_rgb-baseline['rgb'])[:, mask].max()),
                        without_group_RGB_difference_mean=float(np.abs(without_rgb-baseline['rgb'])[:, mask].mean()))
                records.append(dict(layer=name, actual_rows=valid_rows, applicability='ACTIVE' if valid_rows else 'EMPTY in current DELETE context', domains=statistics))
            fp = self.args.output/'evidence'/f'raw{raw:03d}_shared_layer_float.npz'; fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fp, **{name+'_'+k: value for name, v in contributions.items() for k, value in v.items()},
                                shared_RGB=baseline['rgb'], shared_alpha=baseline['alpha'], environment_RGB_T=environment)
        vjp = self.row_contributions(raw, rows, metadata, domains)
        report = dict(raw=raw, camera_and_row_QA=qa, parity=parity, additive_closure=closure,
                      layers=records, exact_contribution_evidence=vjp, layer_float=self.core.ref(fp))
        self.core.write(self.args.output/f'raw{raw:03d}_metrics.json', report)
        self.core.write(self.args.output/'progress.json', dict(PID=os.getpid(), completed_raw=raw,
            native_calls=self.rt.calls, feature_VJP_calls=self.vjp_calls, optimizer_updates=0))
        print('SOURCE_ATTRIBUTION_RAW_COMPLETE', raw, json.dumps(closure), flush=True)
        return report

    def run(self):
        reports = [self.run_view(raw) for raw in self.core.BUILD]
        actor_unchanged = self.gpu.tree_equal(self.gpu.cpu(self.rt.actor.state_dict()), self.actor_before)
        Adam_unchanged = self.gpu.tree_equal(self.gpu.cpu(self.opt.state_dict()), self.Adam_before)
        source_unchanged = {str(p.resolve()): self.core.sha(p) == self.source_hashes[str(p.resolve())] for p in self.source_paths}
        if not actor_unchanged or not Adam_unchanged or not all(source_unchanged.values()): raise RuntimeError('Frozen candidate/source changed during attribution')
        proof = dict(schema='a20.scene_residual.source_attribution.proof.v1', complete=True, actor=self.core.ref(self.args.actor),
            actor_active_rows=int(self.rt.actor.active.sum()), actor_and_active_unchanged=actor_unchanged,
            Adam_unchanged=Adam_unchanged, source_files_unchanged=source_unchanged, frozen_scene=self.rt.check_frozen(),
            actual_native_CUDA=True, actual_shared_depth_sort=True, first2_plus_extra22_ABI=True,
            no_image_paste=True, no_center_box_ownership=True, native_branch_uses_global_ID_topology=True,
            optimizer_updates=0, retired_context_rows=0, heldout_frames_opened=[],
            native_calls=self.rt.calls, feature_VJP_calls=self.vjp_calls,
            elapsed_seconds=time.monotonic()-self.started, GPU_keep_on=True, power_operations=0)
        self.core.write(self.args.output/'proof.json', proof)
        self.core.write(self.args.output/'metrics.json', dict(views=reports, proof=proof))
        self.core.write(self.args.output/'resource_ledger.json', dict(native_calls=self.rt.calls, feature_VJP_calls=self.vjp_calls,
            optimizer_updates=0, cuda_peak_allocated_bytes=self.t.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-self.started, GPU_keep_on=True, power_operations=0))


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check: print(json.dumps(cpu_self_check(), indent=2))
    else:
        try: Audit(args).run()
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise
