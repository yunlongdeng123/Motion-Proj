"""Replay native tile contents with a conditional per-ray depth key."""
from pathlib import Path
import ctypes as C
import math
import sys
import time

import numpy as np
import torch

RUN = Path(__file__).resolve().parent.parent
PARENT = RUN.parent / 'scene008_a6_continuous_takeover_20261105'
sys.path.insert(0, str(PARENT / 'source'))
import takeover

takeover.calibrate.RUN = RUN
from takeover import RAWS, PILOTS, merge_rows, metric, read, save, select_rows, tag, write
from verify_geometry import Check
from verify import GUARDS, pool_street, mean
from ray_support import intrinsics
from fuse_a6 import build_rotation
from a6_clip_rasterization import _C

MODES = {
    'clip_near': (1, .25),
    'clip_wide': (1, .75),
    'all_near': (2, .25),
    'all_wide': (2, .75),
    'clip_mass_near': (3, .25),
    'clip_mass_wide': (3, .75),
    'all_mass_near': (4, .25),
    'all_mass_wide': (4, .75),
}
TILE = 16
PN = C.c_void_p
INT = C.c_int
FLOAT = C.c_float


class Kernels:
    def __init__(self):
        self.lib = C.CDLL(str(RUN / 'libreplay_quantile.so'))
        common = [INT, INT, INT] + [PN] * 10 + [INT, FLOAT]
        self.lib.count_pass.argtypes = common + [PN, PN]
        self.lib.gather_pass.argtypes = common + [PN, PN, PN, PN, PN]
        self.lib.compose_pass.argtypes = [INT] + [PN] * 9
        for name in ('count_pass', 'gather_pass', 'compose_pass'):
            getattr(self.lib, name).restype = INT

    @staticmethod
    def call(fn, *args):
        actual = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args]
        status = fn(*actual, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f'{fn.__name__} CUDA status {status}')


def layout(buffer, fields):
    """Read the verified 128-byte aligned native chunk layout."""
    offset = 0
    address = buffer.data_ptr()
    for count, dtype, width in fields:
        offset += (-(address + offset)) % 128
        end = offset + width * count
        assert end <= len(buffer), (offset, end, len(buffer))
        yield buffer[offset:end].view(dtype)
        offset = end


def native_output(trial, raw, rows, background):
    cam = trial.tr.raw(raw)['cam']
    cfg = trial.tr.n.args
    empty = rows['means3D'].new_empty(0)
    value = lambda key: rows[key] if rows[key] is not None else empty
    return _C.rasterize_gaussians(
        background, rows['means3D'], value('colors_precomp'), rows['features'], rows['opacities'],
        value('scales'), value('rotations'), 1., value('cov3D_precomp'), rows['mask'],
        cam.world_view_transform, cam.full_proj_transform,
        math.tan(-.5 if cfg.neg_fov else cam.FoVx * .5), math.tan(-.5 if cfg.neg_fov else cam.FoVy * .5),
        cam.image_height, cam.image_width, value('shs'), trial.tr.n.static.active_sh_degree,
        cam.camera_center, False, False)


def view_terms(rows, kind, camera):
    feature = rows['features']
    out = feature.new_zeros((len(kind), 9))
    clipped = kind == 1
    out[clipped] = feature[clipped, 7:16]
    remaining = (kind == 2) | (kind == 3)
    rot = build_rotation(rows['rotations'][remaining])
    w = camera.world_view_transform[:3, :3]
    local = w.T @ rot
    var = rows['scales'][remaining].square()
    var = torch.maximum(var, var.amax(1, keepdim=True) * 1e-5).clamp_min(1e-12)
    precision = (local / var[:, None, :]) @ local.transpose(1, 2)
    mu = (rows['means3D'][remaining] - camera.camera_center) @ w
    pm = (precision @ mu[:, :, None])[:, :, 0]
    out[remaining, :6] = precision[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    out[remaining, 6:] = pm
    assert torch.isfinite(out).all()
    return out


class Trace:
    def __init__(self, experiment, raw, rows, kind, background, channels, env=None):
        self.experiment = experiment
        self.raw = raw
        self.feature = rows['features'].contiguous()
        self.background = background
        self.channels = channels
        self.env = env
        self.kind = kind
        count, _, rgb, feat, radii, geom, binning, image = native_output(experiment, raw, rows, background)
        self.native = torch.cat((rgb, feat[:6]), dim=0).contiguous()
        self.radii = radii
        self.h, self.w = rgb.shape[1:]
        self.nx = (self.w + TILE - 1) // TILE
        self.ny = (self.h + TILE - 1) // TILE
        n = len(self.feature)
        f, i, u = torch.float32, torch.int32, torch.uint8
        depth, _, _, xy, _, conic, color = layout(geom, (
            (n, f, 4), (3*n, u, 1), (n, i, 4), (2*n, f, 4), (6*n, f, 4), (4*n, f, 4), (3*n, f, 4)))
        self.depth, self.xy, self.conic = depth, xy, conic
        self.color = color if rows['colors_precomp'] is None else rows['colors_precomp'].contiguous()
        self.ids, _, keys = layout(binning, ((count, i, 4), (count, i, 4), (count, torch.int64, 8)))
        p = self.h * self.w
        _, _, self.ranges = layout(image, ((p, f, 4), (p, i, 4), (2*p, i, 4)))
        # Tile selection depends only on which assigned a6/PVD supports touch a tile.
        # Every pixel in an affected tile and all scene components are jointly replayed.
        tile_hits = (keys[kind[self.ids.long()] > 0] >> 32).long()
        hit_tile = torch.zeros(self.nx*self.ny, dtype=torch.bool, device='cuda')
        hit_tile[tile_hits] = True
        pix = torch.arange(p, dtype=i, device='cuda')
        slot = (pix // self.w // TILE) * self.nx + pix % self.w // TILE
        self.pixels = pix[hit_tile[slot.long()]]
        self.terms = view_terms(rows, kind, experiment.tr.raw(raw)['cam'])
        self.ray = intrinsics(experiment.tr.raw(raw)['cam']).contiguous()
        self.count = None
        self.offsets = None
        self.count_mode = None
        self.hits = 0

    def package(self, feature):
        out = dict(rgb=feature[:3], depth_sum=feature[3:4], alpha=feature[4:5],
                   extra=feature[5:5+self.channels], radii=self.radii)
        if self.env is not None:
            out['rgb'] = out['rgb'] + (1-out['alpha'])*self.env
        return out

    def base(self):
        return self.package(self.native)

    def args(self, mode, cap):
        return (len(self.pixels), self.w, self.nx, self.pixels, self.ranges, self.ids,
                self.xy, self.conic, self.depth, self.feature, self.kind, self.terms,
                self.ray, mode, cap * self.experiment.tr.scale)

    def gather(self, mode, cap):
        k = self.experiment.kernel
        group = mode if mode >= 3 else 0
        if self.count_mode != group:
            self.count_mode = group
            self.count = torch.empty_like(self.pixels)
            k.call(k.lib.count_pass, *self.args(group, 0.), self.count)
            self.offsets = torch.cat((self.count.new_zeros(1), self.count.cumsum(0, dtype=torch.int32)))
            self.hits = int(self.offsets[-1])
        device = self.count.device
        keys = torch.empty(self.hits, device=device, dtype=torch.int64)
        ids = torch.empty(self.hits, device=device, dtype=torch.int32)
        alpha = torch.empty(self.hits, device=device, dtype=torch.float32)
        if self.hits:
            k.call(k.lib.gather_pass, *self.args(mode, cap), self.offsets, keys, ids, alpha)
        return keys, ids, alpha

    def replay(self, mode, cap):
        keys, ids, alpha = self.gather(mode, cap)
        slots = len(self.pixels)
        if not self.hits:
            return self.base(), dict(pixels=slots, hits=0, permuted=0, moved_pixels=0)
        k = self.experiment.kernel
        identity = torch.arange(self.hits, device=keys.device, dtype=torch.int64)
        ordering = keys.argsort(stable=True) if mode else identity
        moved = ordering != identity
        count = int(moved.sum())
        changed_pixels = int((keys[moved] >> 32).unique().numel()) if count else 0
        output = self.native.new_empty((slots, 9))
        k.call(k.lib.compose_pass, slots, self.offsets, ordering, ids, alpha, self.color,
               self.feature, self.background, output)
        final = self.native.flatten(1).clone()
        final[:, self.pixels.long()] = output.T
        return self.package(final.reshape(9, self.h, self.w)), dict(
            pixels=slots, hits=self.hits, permuted=count, moved_pixels=changed_pixels)


class Experiment(Check):
    def __init__(self):
        super().__init__()
        self.kernel = Kernels()
        self.prior = read(PARENT / 'delivery' / 'verify.json')
        with np.load(PARENT / 'delivery' / 'accepted.npz', allow_pickle=False) as file:
            assert np.array_equal(file['native_row_id'], self.tr.native_ids.cpu().numpy())
            self.weight, self.extra = (self.zero.new_tensor(file[k]) for k in ('strength', 'residual'))
            self.prepare_replay(torch.as_tensor(file['variant'], device='cuda'),
                                self.zero.new_tensor(file['normal_local']), self.zero.new_tensor(file['cut_local_m']))

    def black_trace(self, raw):
        actor = self.actor(raw, self.weight, self.extra)
        patch = self.cache[raw][3]
        rows = merge_rows(actor, patch)
        kind = torch.full((len(rows['features']),), 3, dtype=torch.int32, device='cuda')
        kind[:len(actor['features'])] = torch.where(actor['features'][:, 6] > .5, 1, 2)
        return Trace(self, raw, rows, kind, self.tr.black, 2)

    def street_trace(self, raw, delete):
        base = self.assemble(raw, delete)
        self.colourize(base, self.road_colour)
        rows, _, env = base
        n0 = len(self.tr.n.static.get_xyz)
        old_count = len(self.cache[raw][0]['features'])
        actor = self.actor(raw, self.weight, self.extra)
        tagged = tag(dict(actor, features=actor['features'][:, :2]), 0)
        tagged['features'][:, 6:] = actor['features'][:, 6:]
        prefix = select_rows(rows, slice(0, n0))
        suffix = select_rows(rows, slice(n0+old_count, None))
        rows = merge_rows(merge_rows(prefix, tagged), suffix)
        features = rows['features']
        kind = torch.zeros((len(features),), dtype=torch.int32, device='cuda')
        kind[features[:, 2] == 1] = 2
        kind[(kind == 2) & (features[:, 6] > .5)] = 1
        kind[features[:, 5] == 1] = 3
        return Trace(self, raw, rows, kind, self.tr.n.bg, 4, env)

    def black_row(self, raw, out, base):
        m = self.tr.measure(raw, out)
        old = self.prior['black'][raw - 89]['metrics']
        actual = base['metrics']
        for name in old:
            assert old[name]['count'] == actual[name]['count'] and abs(old[name]['sum'] - actual[name]['sum']) < .001, (raw, name)
        den = old['isolated_lower_alpha']['sum']
        if den and (old['lower_pvd']['sum'] - m['metrics']['lower_pvd']['sum']) / den > .01:
            m['violations'].append('lower_vs_parent')
        q = self.tr.evidence[raw]['mask']
        return m

    @staticmethod
    def parity(before, after):
        return {k:float((before[k]-after[k]).abs().max()) for k in ('rgb', 'alpha', 'depth_sum', 'extra')}

    def black_experiment(self, arm_names, destination='black_order.json'):
        frame_rows = []
        for raw in RAWS:
            trace = self.black_trace(raw)
            before = trace.base()
            base = self.tr.measure(raw, before)
            assert not base['violations']
            center, info = trace.replay(0, 0.)
            parity = self.parity(before, center)
            assert parity['depth_sum'] < 5e-5 and max(parity[k] for k in ('rgb', 'alpha', 'extra')) < 5e-6, (raw, parity)
            arms = {}
            for arm in arm_names:
                after, stats = trace.replay(*MODES[arm])
                row = self.black_row(raw, after, base)
                row['raster'] = stats
                row['max_delta'] = self.parity(before, after)
                arms[arm] = row
            frame_rows.append(dict(raw=raw, parity=parity, baseline=base, modes=arms))
            if raw in (*PILOTS, 89, 137):
                print('BLACK', raw, parity, {arm: (arms[arm]['metrics']['lower_pvd']['sum'],
                     arms[arm]['violations'], arms[arm]['raster']['moved_pixels']) for arm in arms}, flush=True)
        summary = {}
        for arm in arm_names:
            rows = [x['modes'][arm] for x in frame_rows]
            summary[arm] = dict(ratio=self.tr.fraction(rows), failures=sum(bool(x['violations']) for x in rows),
                                moved_pixels=sum(x['raster']['moved_pixels'] for x in rows))
        base_score = self.tr.fraction([x['baseline'] for x in frame_rows])
        data = dict(frames=49, baseline=base_score, summary=summary, rows=frame_rows, modes=MODES)
        write(RUN / destination, data)
        print('BLACK_SUMMARY', summary, flush=True)
        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('median', 'quantile'), default='quantile')
    family = parser.parse_args().family
    torch.set_num_threads(1)
    with torch.no_grad():
        arms = [name for name in MODES if ('mass' in name) == (family == 'quantile')]
        Experiment().black_experiment(arms, f'black_{family}.json')
