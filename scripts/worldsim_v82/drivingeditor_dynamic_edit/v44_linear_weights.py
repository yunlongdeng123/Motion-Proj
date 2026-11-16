"""Inherited fixed-geometry alpha*T design with explicitly supplied native ABI.

No scene paths or historical Python renderer are imported. The caller supplies
the 24-feature raw raster extension and count/gather replay library. Runtime
alpha/RGB conservation checks guard the inherited binary-buffer ABI.
"""
from __future__ import annotations
import ctypes as C
import importlib.util
import math
from pathlib import Path
import sys
import types
import numpy as np
from scipy.sparse import coo_matrix

PN, INT, FLOAT = C.c_void_p, C.c_int, C.c_float


def load_raster_extension(path):
    path = Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    parent = '_v44_material_raster'
    name = parent + '._C'
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise ValueError('One worker cannot mix different raw raster extension binaries')
        return previous
    package = sys.modules.setdefault(parent, types.ModuleType(parent)); package.__path__ = [str(path.parent)]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ValueError('Cannot load supplied native raster extension')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); sys.modules[name] = module
    if not callable(getattr(module, 'rasterize_gaussians', None)):
        raise ValueError('Native extension lacks raw rasterize_gaussians ABI')
    return module


def layout(buffer, fields):
    offset = 0; address = buffer.data_ptr()
    for count, dtype, width in fields:
        offset += (-(address + offset)) % 128; end = offset + width*count
        if end > len(buffer): raise ValueError('Native geometry/binning buffer ABI is incompatible')
        yield buffer[offset:end].view(dtype); offset = end


class Kernel:
    def __init__(self, library):
        self.path = Path(library).resolve()
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        self.lib = C.CDLL(str(self.path)); common = [INT, INT, INT] + [PN]*10 + [INT, FLOAT]
        self.lib.count_pass.argtypes = common + [PN, PN]  # count and 64-bit stream pointer
        self.lib.gather_pass.argtypes = common + [PN]*5
        for name in ('count_pass', 'gather_pass'): getattr(self.lib, name).restype = INT

    def call(self, name, *args):
        import torch
        actual = [x.data_ptr() if torch.is_tensor(x) else x for x in args]
        device = next((x.device for x in args if torch.is_tensor(x)), None)
        status = getattr(self.lib, name)(*actual, torch.cuda.current_stream(device).cuda_stream)
        if status: raise RuntimeError(name + ': CUDA status ' + str(status))


def padded(rows):
    value = rows['features']
    if value.ndim != 2 or value.shape[1] > 24: raise ValueError('Native feature rows exceed the 24-feature ABI')
    out = value.new_zeros((len(value), 24)); out[:, :value.shape[1]] = value
    if bool(out[:, 6:].any()): raise ValueError('Fixed material design cannot contain clip/research metadata')
    return dict(rows, features=out)


def native(camera, pipe, degree, rows, background, raster_extension):
    rows = padded(rows); empty = rows['means3D'].new_empty(0)
    value = lambda name: rows[name] if rows[name] is not None else empty
    result = raster_extension.rasterize_gaussians(background, rows['means3D'], value('colors_precomp'),
        rows['features'], rows['opacities'], value('scales'), value('rotations'), 1., value('cov3D_precomp'),
        rows['mask'], camera.world_view_transform, camera.full_proj_transform,
        math.tan(-.5 if pipe.neg_fov else camera.FoVx*.5), math.tan(-.5 if pipe.neg_fov else camera.FoVy*.5),
        camera.image_height, camera.image_width, value('shs'), degree, camera.camera_center, False, False)
    if len(result) != 8: raise ValueError('Raw raster extension returned the wrong binary-buffer ABI')
    return result, rows


def reference_weights(offsets, ids, alpha, columns):
    """CPU reference of the inherited per-ray log-transmittance equation."""
    offsets = np.asarray(offsets); ids = np.asarray(ids); alpha = np.asarray(alpha, np.float64)
    if (offsets.dtype.kind not in 'iu' or offsets.ndim != 1 or not len(offsets) or offsets[0] != 0
            or ids.dtype.kind not in 'iu' or ids.ndim != 1 or alpha.shape != ids.shape
            or offsets[-1] != len(ids) or (np.diff(offsets.astype(np.int64)) < 0).any()
            or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha >= 1).any()
            or type(columns) is not int or columns < 0 or (ids >= columns).any() or (ids < 0).any()):
        raise ValueError('Invalid actual ray hits for reference design')
    ray = np.repeat(np.arange(len(offsets)-1), np.diff(offsets).astype(int))
    logs = np.r_[0., np.log1p(-alpha).cumsum()]
    before = np.exp(logs[:-1] - logs[offsets[ray]])
    weight = before*alpha*(before*(1-alpha) >= .0001)
    use = weight > 0
    return coo_matrix((weight[use], (ray[use], ids[use])), shape=(len(offsets)-1, columns)).tocsr()


def design(camera, pipe, rows, mask, *, raster_extension, replay_library):
    """CSR W, original selected pixel indices, and alpha/RGB conservation data."""
    import torch
    if not mask.is_cuda or mask.dtype != torch.bool or mask.shape != (camera.image_height, camera.image_width) or mask.device != rows['means3D'].device:
        raise ValueError('Design mask must be bool on the actual source camera grid/device')
    with torch.no_grad():
        kernel = Kernel(replay_library); background = rows['means3D'].new_zeros(3)
        result, rows = native(camera, pipe, 0, rows, background, raster_extension)
        rendered, contrib, rgb, features, radii, geom, binning, pixel_buffer = result
        n = len(rows['features']); f, i, u = torch.float32, torch.int32, torch.uint8
        depth, _, _, xy, _, conic, _ = layout(geom, ((n,f,4),(3*n,u,1),(n,i,4),(2*n,f,4),(6*n,f,4),(4*n,f,4),(3*n,f,4)))
        tile_ids, _, _ = layout(binning, ((rendered,i,4),(rendered,i,4),(rendered,torch.int64,8)))
        h, w = rgb.shape[1:]; p = h*w
        _, _, ranges = layout(pixel_buffer, ((p,f,4),(p,i,4),(2*p,i,4)))
        pixels = torch.where(mask.flatten())[0].to(torch.int32); count = torch.empty_like(pixels)
        kind = torch.full((n,), 3, device=mask.device, dtype=torch.int32)
        terms = rows['features'].new_zeros((n,9)); ray = terms.new_zeros(4)
        args = (len(pixels),w,(w+15)//16,pixels,ranges,tile_ids,xy,conic,depth,rows['features'],kind,terms,ray,0,0.)
        kernel.call('count_pass', *args, count)
        if int(count.long().sum()) >= 2**31 or bool((count < 0).any()): raise ValueError('Design hit count exceeds int32 ABI')
        offsets = torch.cat((count.new_zeros(1), count.cumsum(0, dtype=torch.int32)))
        hit_count = int(offsets[-1]); keys = torch.empty(hit_count, device=mask.device, dtype=torch.int64)
        ids = torch.empty(hit_count, device=mask.device, dtype=torch.int32); alpha = torch.empty(hit_count, device=mask.device)
        if hit_count: kernel.call('gather_pass', *args, offsets, keys, ids, alpha)
        ray_index = torch.arange(len(pixels), device=mask.device).repeat_interleave(count.long(), output_size=hit_count)
        a = alpha.double(); logs = torch.cat((a.new_zeros(1), torch.log1p(-a).cumsum(0)))
        before = (logs[:-1] - logs[offsets.long()[ray_index]]).exp()
        weight = before*a*(before*(1-a) >= .0001); use = weight > 0
        matrix = coo_matrix((weight[use].cpu().numpy(), (ray_index[use].cpu().numpy(), ids[use].cpu().numpy())), shape=(len(pixels),n)).tocsr()
        index = pixels.long(); alpha_ref = features[1].flatten()[index].cpu().numpy()
        rgb_ref = rgb.flatten(1)[:,index].T.cpu().numpy()
        alpha_error = float(np.max(np.abs(np.asarray(matrix.sum(1)).ravel()-alpha_ref))) if len(index) else 0.
        colour = rows['colors_precomp']
        if colour is None: raise ValueError('Appearance design requires explicit per-point RGB')
        colour_error = float(np.max(np.abs(matrix@colour.cpu().numpy()-rgb_ref))) if len(index) else 0.
        if not np.isfinite(alpha_error+colour_error) or max(alpha_error,colour_error) >= 5e-6:
            raise RuntimeError('Raw design fails inherited alpha/RGB conservation: ' + str((alpha_error,colour_error)))
        return dict(matrix=matrix, pixels=index.cpu().numpy(), alpha=alpha_ref, rgb=rgb_ref,
                    alpha_maxabs=alpha_error, colour_maxabs=colour_error, hits=hit_count,
                    raster_rgb=rgb, raster_alpha=features[1:2])
