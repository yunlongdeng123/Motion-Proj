"""Read-only CPU model adapter. Fixed campaign; no training/model writes/CUDA."""
import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from ray_reference import covariance, project, candidates, composite, sh_color

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_serial_r1_20261021')
ARMS = ('full', 'real_only', 'prior_only', 'ownership_only')
MANIFEST_SHA = 'eb926ef96b21f5b7382bc7e9eff7f4424cb13fc914246482d90ad179f6b72465'


def clean(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    return v


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


class Adapter:
    def __init__(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES', None) != '':
            raise RuntimeError('CPU-only adapter requires CUDA_VISIBLE_DEVICES empty')
        path = CAMPAIGN/'final49/summary_r2/asset_manifest.json'
        if digest(path) != MANIFEST_SHA: raise ValueError('canonical manifest mismatch')
        self.manifest = json.loads(path.read_text())['files']
        self.config = json.loads((CAMPAIGN/'arms/full/run30k/effective_config.json').read_text())
        self.bundle = Path(self.config['v3_bundle'])
        self.meta = json.loads(self.bundle.read_text())
        self.A = np.asarray(self.meta['world_to_model'])
        self.invA = np.linalg.inv(self.A)
        self.scale = float(self.meta['meter_to_model'])
        self.models, self.prepared = OrderedDict(), OrderedDict()

    def camera(self, raw):
        if type(raw) is not int or not 89 <= raw <= 137: raise ValueError('raw must be 89..137')
        m = self.meta['frames'][f'0008{raw:03d}']
        w = np.array(m['world_to_camera']); w[:3] *= self.scale
        w = (w @ self.invA).astype(np.float32)
        return dict(W=w, K=np.asarray(m['intrinsic'], np.float32),
                    center=np.linalg.inv(w)[:3, 3], timestamp=(raw-113)*.02, meta=m)

    def model(self, arm):
        if arm not in ARMS: raise ValueError('unknown fixed arm')
        if arm in self.models:
            self.models.move_to_end(arm); return self.models[arm]
        import torch
        torch.set_num_threads(4)
        branches = []
        for name, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt')]:
            rel = f'arms/{arm}/run30k/{prefix}30000.pth'; path = CAMPAIGN/rel
            expected = self.manifest[rel]
            if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise ValueError('checkpoint SHA/bytes mismatch')
            payload, step = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
            if step != 30000: raise ValueError('not terminal checkpoint')
            arr = lambda i: payload[i].detach().numpy()
            xyz, dc, rest, logs, q, logit, t, logt, vel = [arr(i) for i in range(1, 10)]
            opt = payload[14]
            road = opt.get('v3_world_road', {}).get('ids')
            road = road.numpy() if road is not None else np.full(len(xyz), -1, np.int64)
            lineage = opt.get('sgcm_lineage_v1', {}).get('lineage_ids')
            lineage = lineage.numpy() if lineage is not None else None
            scales = np.exp(logs)
            branches.append(dict(name=name, sha=expected['sha256'], xyz=xyz, dc=dc, rest=rest,
                scale=scales, q=q,
                opacity=1/(1+np.exp(-np.clip(logit[:, 0], -80, 80))),
                t=t[:, 0], sigma_t=np.exp(logt[:, 0]), vel=vel, cycle=float(payload[16]),
                degree=int(payload[0]), road=road, lineage=lineage))
            del payload, opt; gc.collect()
        result = dict(arm=arm, branches=branches)
        self.models[arm] = result
        while len(self.models) > 1: self.models.popitem(last=False)
        return result

    def prepare(self, arm, raw):
        key = (arm, raw)
        if key in self.prepared:
            self.prepared.move_to_end(key); return self.prepared[key]
        model, camera = self.model(arm), self.camera(raw)
        out = []
        for b in model['branches']:
            parts = []
            for start in range(0, len(b['xyz']), 50000):
                end = min(start+50000, len(b['xyz'])); sl = slice(start, end)
                delta = camera['timestamp']-b['t'][sl]; a = np.float32(2*np.pi/b['cycle'])
                xyz = b['xyz'][sl]+b['vel'][sl]*(np.sin(delta*a)/a)[:, None]
                marginal = np.exp(-.5*(delta/b['sigma_t'][sl])**2)
                isroad = b['road'][sl] >= 0
                xyz[isroad] = b['xyz'][sl][isroad]; marginal[isroad] = 1
                p = project(xyz, covariance(b['scale'][sl], b['q'][sl]), b['opacity'][sl]*marginal,
                            marginal > .05, camera['W'], camera['K'], bool(self.config['neg_fov']))
                local = p['row']; rows = local+start
                sh = np.concatenate((b['dc'][rows], b['rest'][rows]), 1)
                sh[b['road'][rows] >= 0, 1:] = 0
                p['color'] = sh_color(sh, xyz[local]-camera['center'], b['degree'])
                p['xyz'] = xyz[local]; p['row'] = rows
                parts.append(p)
            p = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
            p['branch'] = b; out.append(p)
        result = dict(camera=camera, branches=out)
        self.prepared[key] = result
        while len(self.prepared) > 1: self.prepared.popitem(last=False)
        return result

    def ray(self, arm, raw, x, y, mode='deleted_five'):
        if type(x) is not int or type(y) is not int or not (0 <= x < 480 and 0 <= y < 320):
            raise ValueError('pixel outside image')
        if mode not in ('deleted_five', 'joint', 'road_only', 'ordinary', 'dynamic'):
            raise ValueError('unsupported branch')
        started = time.monotonic(); p = self.prepare(arm, raw)
        entries = []
        for branch in p['branches']:
            b = branch['branch']
            if b['name'] == 'dynamic' and mode not in ('joint','dynamic'): continue
            if mode == 'dynamic' and b['name'] != 'dynamic': continue
            idx, alpha = candidates(branch, x, y)
            for i, al in zip(idx, alpha):
                row = branch['row'][i]; road = int(b['road'][row])
                if mode == 'road_only' and road < 0: continue
                if mode == 'ordinary' and road >= 0: continue
                entries.append((float(branch['depth'][i]), b['name'] == 'dynamic', int(row), branch, int(i), float(al)))
        entries.sort(key=lambda v: (v[0], v[1], v[2]))
        colors = np.array([e[3]['color'][e[4]] for e in entries], np.float32).reshape(-1, 3)
        alpha = np.array([e[5] for e in entries], np.float32)
        depth = np.array([e[0] for e in entries], np.float32)
        c = composite(alpha, colors, depth)
        weights = np.asarray(c['weights']); picked = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:40]
        color_rank = sorted(range(len(weights)), key=lambda i: float(np.linalg.norm(colors[c['indices'][i]]*weights[i])), reverse=True)[:40]
        selected = sorted(set(picked+color_rank))
        rows = []
        for k in selected:
            order = c['indices'][k]; z, dyn, row, pr, i, al = entries[order]; b = pr['branch']
            modelxyz = pr['xyz'][i]; world = self.invA[:3, :3]@modelxyz+self.invA[:3, 3]
            role = 'dynamic' if dyn else ('road' if b['road'][row] >= 0 else 'ordinary')
            rows.append(dict(id=f'{arm}:{b["sha"][:12]}:{b["name"]}:{row}', row=row, branch=role,
                checkpoint_sha256=b['sha'], road_id=int(b['road'][row]),
                source_lineage=None if b['lineage'] is None else int(b['lineage'][row]),
                world_m=world, model_xyz=modelxyz, camera_z_m=z/self.scale,
                alpha=al, T=c['transmittance'][k], weight=weights[k],
                color=colors[order], color_contribution=colors[order]*weights[k],
                sigma_m=b['scale'][row]/self.scale, quaternion=b['q'][row],
                cov2=pr['cov2'][i], uv=pr['uv'][i], radius_px=float(pr['radius'][i]),
                opacity_base=float(b['opacity'][row]), time_center=float(b['t'][row]),
                sigma_t=float(b['sigma_t'][row]), velocity_model=b['vel'][row]))
        rawpath = CAMPAIGN/f'final49/{arm}_r2/0008{raw:03d}/render_float.npz'
        with np.load(rawpath) as old:
            prefix = {'joint':'joint','dynamic':'dynamic','road_only':'road'}.get(mode,'static')
            native_alpha = float(old[prefix+'_alpha'][0, y, x])
            native_depth = float(old[prefix+'_depth_sum'][0, y, x])
            measured = float(old['lidar_depth_m'][0, y, x])
            values = {k: old[k][:, y, x].copy() for k in ('joint', 'deleted_five', 'gt', 'dynamic_raw',
                'road_raw', 'road_in_background', 'road_in_joint','static_alpha','joint_alpha',
                'dynamic_alpha','road_alpha','static_depth_m','joint_depth_m','dynamic_depth_m')}
        compare = mode in ('deleted_five', 'joint','dynamic','road_only')
        errors = dict(alpha_abs=abs(c['alpha']-native_alpha) if compare else None,
                      depth_sum_model_abs=abs(c['depth_sum']-native_depth) if compare else None)
        if mode in ('dynamic','road_only'):
            errors['rgb_pre_environment_abs_max'] = float(np.abs(c['rgb']-values['dynamic_raw' if mode=='dynamic' else 'road_raw']).max())
        return clean(dict(status='CPU_REFERENCE_NOT_CUDA_TRACE', arm=arm, raw=raw, pixel=[x, y], mode=mode,
            no_optimizer=True, parameter_updates=0, gpu_used=False, candidate_count=len(entries),
            accepted_count=len(c['indices']), top=rows,
            profile=[dict(z_m=entries[v][0]/self.scale, weight=w, T=t,
                          branch='dynamic' if entries[v][1] else ('road' if entries[v][3]['branch']['road'][entries[v][2]] >= 0 else 'ordinary'))
                     for v, w, t in zip(c['indices'], c['weights'], c['transmittance'])],
            reference=dict(rgb_gaussians=c['rgb'], alpha=c['alpha'], T=c['T'], depth_sum_model=c['depth_sum']),
            native=dict(alpha=native_alpha, depth_sum_model=native_depth, lidar_m=measured if measured > 0 else None, **values),
            replay_errors=errors, numeric_alpha_depth_agree=compare and errors['alpha_abs'] < .002 and errors['depth_sum_model_abs'] < .002,
            environment_color_not_in_reference_rgb=True, causal_claim=False, elapsed_s=time.monotonic()-started,
            limitations=['CPU float32 is not native CUDA bitwise attribution',
                'camera matrices reconstructed from bound bundle; native alpha/depth error shown',
                'center-z order is not ellipsoid first surface', 'source lineage is not unique Gaussian lifetime ID']))


def serve():
    adapter = Adapter()
    for line in sys.stdin:
        try:
            query = json.loads(line)
            value = adapter.ray(query['arm'], int(query['raw']), int(query['x']), int(query['y']), query.get('mode', 'deleted_five'))
            print(json.dumps(value, allow_nan=False), flush=True)
        except Exception as e:
            print(json.dumps(dict(error=type(e).__name__+': '+str(e), gpu_used=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stdio', action='store_true')
    if parser.parse_args().stdio: serve()
