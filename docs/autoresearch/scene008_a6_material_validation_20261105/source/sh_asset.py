"""Canonical-local SH3 appearance on frozen complete a6 Gaussian geometry.

The SH convention is exactly native utils.sh_utils.eval_sh: camera-to-point
unit directions, standard SH coefficients, +.5, and a lower clamp at zero.
No coefficient projection or upper color clamp is performed by this asset.
Geometry is size-parameterized, retaining all source/child provenance arrays.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

RUN = Path(__file__).resolve().parent.parent
COMPLETE = RUN.parent/'scene008_a6_complete_asset_20261105'
sys.path.append(str(COMPLETE/'source'))
from full_pvd import FLOAT_FIELDS, BOOL_FIELDS, INT_FIELDS, GEOMETRY_FIELDS, hash_geometry, save_npz, sha_file

C0 = .28209479177387814
C1 = .4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, .31539156525252005,
      -1.0925484305920792, .5462742152960396)
C3 = (-.5900435899266435, 2.890611442640554, -.4570457994644658,
      .3731763325901154, -.4570457994644658, 1.445305721320277, -.5900435899266435)


def basis3(directions):
    """Native real SH basis [...,16], on unit camera-to-point directions.

    Coefficients use coeff[...,0,:]=(DC_RGB-.5)/C0. For the fitter's alternative
    theta=[DC_RGB,higher_SH], use theta_basis3, whose first column is one.
    NumPy and Torch inputs preserve their backend, dtype and device.
    """
    if directions.shape[-1] != 3:
        raise ValueError('Directions must have final dimension three')
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    xx, yy, zz = x*x, y*y, z*z
    values = (x*0+C0, -C1*y, C1*z, -C1*x, C2[0]*x*y, C2[1]*y*z,
              C2[2]*(2*zz-xx-yy), C2[3]*x*z, C2[4]*(xx-yy),
              C3[0]*y*(3*xx-yy), C3[1]*x*y*z, C3[2]*y*(4*zz-xx-yy),
              C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy),
              C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy))
    return torch.stack(values, dim=-1) if torch.is_tensor(directions) else np.stack(values, axis=-1)


def theta_basis3(directions):
    values = basis3(directions)
    values = values.clone() if torch.is_tensor(values) else values.copy()
    values[..., 0] = 1
    return values


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def pose_rotation(matrix):
    """Extract a proper rotation from an already uniform-scale actor pose."""
    value = np.asarray(_numpy(matrix), np.float64)
    if value.shape not in ((3, 3), (4, 4)) or not np.isfinite(value).all():
        raise ValueError('Expected finite 3x3 rotation or 4x4 similarity pose')
    linear = value[:3, :3]
    scale = np.linalg.norm(linear, axis=0).mean()
    if scale <= 0:
        raise ValueError('Non-positive actor scale')
    rotation = linear/scale
    if np.linalg.det(rotation) <= 0 or not np.allclose(rotation.T@rotation, np.eye(3), atol=2e-5, rtol=0):
        raise ValueError('SH rotation requires a proper uniform-scale pose, not shear/reflection')
    # Polar cleanup removes only pose floating-point roundoff; same R is used
    # by both the precomputed-color and world-coefficient paths.
    u, _, vh = np.linalg.svd(rotation)
    return u@vh


_K = 64
_Z = 1-2*(np.arange(_K, dtype=np.float64)+.5)/_K
_ANGLE = np.arange(_K, dtype=np.float64)*(np.pi*(3-np.sqrt(5)))
_DIRECTIONS = np.column_stack((np.sqrt(1-_Z*_Z)*np.cos(_ANGLE),
                              np.sqrt(1-_Z*_Z)*np.sin(_ANGLE), _Z))
_BASIS = basis3(_DIRECTIONS)
_PINV = [np.linalg.pinv(_BASIS[:, low:high]) for low, high in ((0, 1), (1, 4), (4, 9), (9, 16))]


def sh_rotation_matrix(matrix):
    """M satisfying Y(world_direction)@M == Y(world_direction@R).

    Deterministic exact-subspace collocation in FP64, separately by SH degree;
    no cross-degree mixing or view-dependent appearance parameters are added.
    World coefficients are M @ canonical coefficients.
    """
    rotation = pose_rotation(matrix)
    rotated = basis3(_DIRECTIONS@rotation)
    result = np.zeros((16, 16), np.float64)
    for inverse, (low, high) in zip(_PINV, ((0, 1), (1, 4), (4, 9), (9, 16))):
        result[low:high, low:high] = inverse@rotated[:, low:high]
    result[0, 0] = 1.
    if not np.allclose(_BASIS@result, rotated, atol=2e-13, rtol=0):
        raise RuntimeError('SH rotation-subspace reconstruction failed')
    return result


class SHAsset(nn.Module):
    degree = 3

    def __init__(self, arrays, metadata=None):
        super().__init__()
        for names, dtype in ((FLOAT_FIELDS, torch.float32), (BOOL_FIELDS, torch.bool), (INT_FIELDS, torch.int64)):
            for key in names:
                self.register_buffer(key, torch.as_tensor(np.array(arrays[key], copy=True), dtype=dtype))
        count = len(self.xyz)
        if count <= 0 or self.xyz.shape != (count, 3) or self.normal.shape != (count, 3) or self.box.shape != (3,):
            raise ValueError('Invalid frozen asset geometry shapes')
        if not bool((self.box > 0).all()):
            raise ValueError('Frozen asset box dimensions must be positive')
        for key in GEOMETRY_FIELDS:
            value = getattr(self, key)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite geometry: '+key)
            if key not in ('xyz', 'normal', 'box') and value.shape != (count,):
                raise ValueError('Wrong row count: '+key)
        if not bool((self.sigma > 0).all() and (self.thickness > 0).all() and
                    ((self.opacity > 0) & (self.opacity < 1)).all()):
            raise ValueError('Invalid frozen scale/opacity')
        if not torch.allclose(self.normal.norm(dim=1), torch.ones(count), atol=2e-6, rtol=0):
            raise ValueError('Frozen normals must be unit length')
        self.geometry_hash = hash_geometry(self._geometry_arrays())
        expected = str(np.asarray(arrays.get('geometry_hash', self.geometry_hash)).item())
        if expected != self.geometry_hash:
            raise ValueError('Frozen geometry hash mismatch')
        reserved = set(GEOMETRY_FIELDS) | {'rgb', 'colour_inferred', 'geometry_hash', 'metadata_json',
                                         'canonical_sh', 'coefficients', 'sh', 'sh_degree', 'provenance_hash'}
        self._extra_names, self._extra_arrays = [], {}
        for key, value in arrays.items():
            if key in reserved:
                continue
            value = np.array(value, copy=True)
            if value.dtype.kind in 'biuf' and not hasattr(self, key):
                self.register_buffer(key, torch.as_tensor(value))
                self._extra_names.append(key)
            else:
                value.setflags(write=False)
                self._extra_arrays[key] = value
        self.provenance_hash = self._provenance_hash()
        if 'provenance_hash' in arrays and str(arrays['provenance_hash']) != self.provenance_hash:
            raise ValueError('Extra source/child provenance hash mismatch')
        self.metadata = deepcopy(metadata or {})
        initial = np.zeros((count, 16, 3), np.float32)
        rgb = np.asarray(arrays.get('rgb', np.full((count, 3), .5)), np.float32)
        if rgb.shape != (count, 3):
            raise ValueError('RGB seed must match geometry row count')
        initial[:, 0] = (rgb-.5)/C0
        supplied = arrays.get('canonical_sh', arrays.get('coefficients', arrays.get('sh', initial)))
        self.register_buffer('coefficients', torch.zeros((count, 16, 3), dtype=torch.float32))
        self.register_buffer('colour_inferred', torch.as_tensor(
            np.array(arrays.get('colour_inferred', np.ones(count, bool)), copy=True), dtype=torch.bool))
        if self.colour_inferred.shape != (count,):
            raise ValueError('Wrong inferred-color row count')
        self.set_coefficients(supplied)
        self._rotation_cache = {}
        self.last_diagnostics = None

    @classmethod
    def from_geometry(cls, path):
        with np.load(path, allow_pickle=False) as bank:
            arrays = {key: bank[key] for key in bank.files}
        metadata = json.loads(str(np.asarray(arrays.get('metadata_json', '{}')).item()))
        return cls(arrays, metadata)

    load = from_geometry

    @property
    def rgb(self):
        return self.coefficients[:, 0]*C0+.5

    def _geometry_arrays(self):
        return {key: _numpy(getattr(self, key)) for key in GEOMETRY_FIELDS}

    def _extras(self):
        return dict(self._extra_arrays, **{key: _numpy(getattr(self, key)) for key in self._extra_names})

    def _provenance_hash(self):
        digest = hashlib.sha256()
        for key, value in sorted(self._extras().items()):
            value = np.ascontiguousarray(value)
            digest.update(key.encode('utf-8')); digest.update(str((value.shape, value.dtype.str)).encode('ascii'))
            digest.update(value.tobytes())
        return digest.hexdigest()

    def assert_geometry_unchanged(self):
        if hash_geometry(self._geometry_arrays()) != self.geometry_hash or self._provenance_hash() != self.provenance_hash:
            raise RuntimeError('Frozen geometry or source/child provenance changed')
        return True

    @torch.no_grad()
    def set_coefficients(self, values, colour_inferred=None):
        value = torch.as_tensor(values, dtype=self.coefficients.dtype, device=self.coefficients.device)
        if value.shape != self.coefficients.shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Expected finite canonical SH coefficients [N,16,3]')
        self.coefficients.copy_(value)  # Deliberately no range constraint/projection.
        if colour_inferred is not None:
            flag = torch.as_tensor(colour_inferred, dtype=torch.bool, device=self.colour_inferred.device)
            if flag.shape != self.colour_inferred.shape:
                raise ValueError('Wrong inferred-color row count')
            self.colour_inferred.copy_(flag)
        return self

    def local_directions(self, matrix, camera):
        pose = torch.as_tensor(matrix, dtype=self.xyz.dtype, device=self.xyz.device)
        if pose.shape != (4, 4):
            raise ValueError('Expected a 4x4 actor pose')
        means = self.xyz@pose[:3, :3].T+pose[:3, 3]
        center = torch.as_tensor(camera.camera_center, dtype=means.dtype, device=means.device)
        world = torch.nn.functional.normalize(means-center, dim=1)
        rotation = world.new_tensor(pose_rotation(pose))
        return torch.nn.functional.normalize(world@rotation, dim=1)

    def world_coefficients(self, matrix):
        rotation = pose_rotation(matrix)
        key = rotation.tobytes()
        if key not in self._rotation_cache:
            self._rotation_cache[key] = sh_rotation_matrix(rotation)
        transform = self.coefficients.new_tensor(self._rotation_cache[key])
        return torch.einsum('ij,njc->nic', transform, self.coefficients)

    def unclamped_colors(self, matrix, camera):
        return torch.einsum('ni,nic->nc', basis3(self.local_directions(matrix, camera)), self.coefficients)+.5

    def output_diagnostics(self, matrix, camera):
        value = self.unclamped_colors(matrix, camera).detach()
        return dict(rows=len(value), unclamped_min=float(value.min()), unclamped_max=float(value.max()),
                    negative_channels=int((value < 0).sum()), above_one_channels=int((value > 1).sum()),
                    rows_below_zero=int((value < 0).any(1).sum()), rows_above_one=int((value > 1).any(1).sum()),
                    lower_shader_clamp=0., upper_shader_clamp=None, coefficient_projection=False)

    def rows(self, matrix, camera, pipe, channels, precomputed=False):
        pose = torch.as_tensor(matrix, dtype=self.xyz.dtype, device=self.xyz.device)
        if pose.shape != (4, 4):
            raise ValueError('Expected a 4x4 actor pose')
        rotation = pose_rotation(pose)
        scale = pose[:3, :3].norm(dim=0).mean()
        means = self.xyz@pose[:3, :3].T+pose[:3, 3]
        normal = torch.nn.functional.normalize(self.normal@pose[:3, :3].T, dim=1)
        q = torch.stack((1+normal[:, 2], -normal[:, 1], normal[:, 0], torch.zeros_like(self.sigma)), 1)
        q = torch.where((q.norm(dim=1) < 1e-5)[:, None], q.new_tensor([0., 1., 0., 0.]), q)
        q = torch.nn.functional.normalize(q, dim=1)
        scales = torch.stack((self.sigma, self.sigma, self.thickness), 1)*scale
        covariance = None
        if pipe.compute_cov3D_python:
            tangent, thin = scales[:, 0].square(), scales[:, 2].square()
            full = tangent[:, None, None]*torch.eye(3, dtype=means.dtype, device=means.device)
            full = full+(thin-tangent)[:, None, None]*normal[:, :, None]*normal[:, None, :]
            covariance = full[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].contiguous()
        view = torch.as_tensor(camera.world_view_transform, dtype=means.dtype, device=means.device)
        depth = means@view[:3, 2:3]+view[3, 2]
        shs, colors = None, None
        if precomputed:
            colors = self.unclamped_colors(pose, camera).clamp_min(0)
        else:
            if int(channels) < 16:
                raise ValueError('SH3 native path needs at least16 SH channels; use precomputed=True for fewer')
            coefficients = self.world_coefficients(rotation)
            shs = torch.cat((coefficients, coefficients.new_zeros((len(self.xyz), int(channels)-16, 3))), 1)
        self.last_diagnostics = self.output_diagnostics(pose, camera)
        return dict(means3D=means, means2D=torch.zeros_like(means), shs=shs, colors_precomp=colors,
                    features=torch.cat((depth, torch.ones_like(depth)), 1), opacities=self.opacity[:, None],
                    scales=scales if covariance is None else None, rotations=q if covariance is None else None,
                    cov3D_precomp=covariance, mask=torch.ones(len(self.xyz), dtype=torch.bool, device=means.device))

    def export(self, path):
        self.assert_geometry_unchanged()
        metadata = dict(self.metadata, appearance='canonical_local_native_real_SH3', sh_degree=3,
                        direction='unit camera-to-point world direction @ actor rotation',
                        shader='native eval_sh + .5, clamp_min(0), no upper clamp',
                        geometry_hash=self.geometry_hash, provenance_hash=self.provenance_hash,
                        last_output_diagnostics=self.last_diagnostics)
        arrays = dict(self._geometry_arrays(), **self._extras())
        arrays.update(canonical_sh=_numpy(self.coefficients), rgb=_numpy(self.rgb),
                      colour_inferred=_numpy(self.colour_inferred), sh_degree=np.int16(3),
                      geometry_hash=np.asarray(self.geometry_hash), provenance_hash=np.asarray(self.provenance_hash),
                      metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)))
        save_npz(path, arrays)
        self.assert_geometry_unchanged()
        return dict(path=str(path), file_sha256=sha_file(path), geometry_hash=self.geometry_hash,
                    provenance_hash=self.provenance_hash, rows=len(self.xyz), sh_degree=3)

    save = export
