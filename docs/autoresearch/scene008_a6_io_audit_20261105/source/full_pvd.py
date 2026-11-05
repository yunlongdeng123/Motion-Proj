"""Complete frozen 2048-point a6 PVD asset; CPU construction, RGB-only updates.

This module never loads native a6, an old 1477-point patch, or historical RGB.
The original conditioning points are retained along with all generated points,
out-of-box points and low-confidence normals.  Geometry is initialized once.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

ROOT = RUN = Path(__file__).resolve().parent.parent
SOURCE = RUN.parent/'scene008_a37_C2_20261103_r1'
RAW_PVD = SOURCE/'source/prior/proposals/pvd_seed000/candidate_points_actor_native.npz'
OBSERVATIONS = SOURCE/'inputs/a6_geometry_minimal.npz'
TRANSFORM = SOURCE/'source/prior/proposals/pvd_seed000/transform_contract.json'
RAW_SHA256 = '1d1b7e20ca389c184f54094674d40d1aa842e61da708b34f8b1ff2911d731562'
OBSERVATIONS_SHA256 = 'c112b730ce327084766978aa6aa764b1654c32fe5d2bcb5676c396ee763bc8b2'
TRANSFORM_SHA256 = '84121bdacabac4eee9652ef04c9a1bc8d9a5e84ce585cd905a7a24d2b2552b31'
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)
FLOAT_FIELDS = ('xyz', 'normal', 'sigma', 'opacity', 'thickness', 'box',
                'knn_spacing', 'normal_radius', 'normal_confidence')
BOOL_FIELDS = ('generated', 'conditioning', 'normal_valid', 'out_of_box', 'conditioning_held')
INT_FIELDS = ('source_index', 'conditioning_observation_index', 'conditioning_raw_frame', 'conditioning_cloud_row')
GEOMETRY_FIELDS = FLOAT_FIELDS+BOOL_FIELDS+INT_FIELDS


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_geometry(arrays):
    digest = hashlib.sha256()
    for key in sorted(GEOMETRY_FIELDS):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode('ascii'))
        digest.update(str((value.dtype.str, value.shape)).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def pca_point_patches(points):
    """Exact CPU PCA/spacing algorithm of prepare_pvd_surface.py, no filtering."""
    _, first = np.unique(np.floor(points/.005).astype(np.int64), axis=0, return_index=True)
    original_index = np.sort(first); p = points[original_index]
    tree = cKDTree(p); k = min(16, len(p)); distance, index = tree.query(p, k=k)
    if k < 4:
        raise ValueError('Too few distinct actual PVD points for point patches')
    spacing = np.median(distance[:, 1:min(4, k)], axis=1)
    radius = np.minimum(.25, np.maximum(4*spacing, .03))
    weight = (distance <= radius[:, None]).astype(float)
    q = p[index]-p[:, None]
    mean = (q*weight[:, :, None]).sum(1)/np.maximum(weight.sum(1), 1)[:, None]
    q = q-mean[:, None]
    covariance = np.einsum('nki,nkj,nk->nij', q, q, weight)/np.maximum(weight.sum(1), 1)[:, None, None]
    eigenvalue, eigenvector = np.linalg.eigh(covariance); normal = eigenvector[:, :, 0]
    flip = (normal*p).sum(1) < 0; normal[flip] *= -1
    valid = (weight.sum(1) >= 4) & (eigenvalue[:, 1] > 1e-7) & (spacing > 1e-6)
    confidence = np.clip((eigenvalue[:, 1]-eigenvalue[:, 0])/np.maximum(eigenvalue[:, 2], 1e-10), 0, 1)
    sigma = np.minimum(.60*spacing, .060)
    sigma = np.minimum(sigma, .45*radius)
    return dict(points=p, normal=normal, sigma=sigma, normal_valid=valid,
                normal_confidence=confidence, knn_spacing=spacing,
                normal_radius=radius, original_index=original_index)


def geometry_builder(raw_path=RAW_PVD, geometry_path=None, audit_path=None,
                     observation_path=OBSERVATIONS, transform_path=TRANSFORM,
                     expected_sha256=RAW_SHA256):
    """Build fixed geometry.npz and provenance JSON using CPU arrays only.

    expected_sha256=None is reserved for explicit alternate/synthetic inputs;
    the default CLI requires the already-frozen original completion hash.
    """
    raw_path, observation_path, transform_path = map(Path, (raw_path, observation_path, transform_path))
    geometry_path = Path(geometry_path) if geometry_path is not None else RUN/'geometry.npz'
    audit_path = Path(audit_path) if audit_path is not None else RUN/'geometry_provenance.json'
    source_hash = sha_file(raw_path)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError('Frozen raw PVD SHA256 mismatch')
    canonical_source = source_hash == RAW_SHA256
    if canonical_source and (sha_file(observation_path) != OBSERVATIONS_SHA256 or sha_file(transform_path) != TRANSFORM_SHA256):
        raise ValueError('Frozen conditioning geometry or coordinate contract SHA256 mismatch')
    with np.load(raw_path, allow_pickle=False) as bank:
        points = np.asarray(bank['points_actor_native_m'], np.float64)
        generated = np.asarray(bank['generated_mask'], bool)
        key = 'conditioning_observed_row_indices' if 'conditioning_observed_row_indices' in bank else 'observed_raw_row_indices'
        conditioning_index = np.asarray(bank[key], np.int64)
        if 'conditioning_observed_row_indices' in bank and 'observed_raw_row_indices' in bank:
            if not np.array_equal(bank['conditioning_observed_row_indices'], bank['observed_raw_row_indices']):
                raise ValueError('Conditioning source-index aliases disagree')
    if points.shape != (2048, 3) or not np.isfinite(points).all() or generated.shape != (2048,):
        raise ValueError('Expected exactly 2048 finite original PVD points')
    if generated.sum() != 1848 or conditioning_index.shape != (200,) or len(np.unique(conditioning_index)) != 200:
        raise ValueError('Expected 200 distinct conditioning rows plus 1848 generated points')
    transform = json.loads(transform_path.read_text(encoding='utf-8'))
    with np.load(observation_path, allow_pickle=False) as bank:
        actual = np.asarray(bank['canonical_xyz'])
        box = np.asarray(bank['box_lwh_m'], np.float64)
        if (conditioning_index < 0).any() or (conditioning_index >= len(actual)).any():
            raise ValueError('Conditioning index outside raw observation array')
        conditioning_error = float(np.abs(points[~generated]-actual[conditioning_index]).max())
        raw_frame = np.asarray(bank['raw_frame'][conditioning_index], np.int64)
        cloud_row = np.asarray(bank['source_cloud_row'][conditioning_index], np.int64)
    if conditioning_error > 2e-5 or box.shape != (3,) or not np.allclose(box, transform['whole_box_lwh_m'], atol=5e-6, rtol=0):
        raise ValueError('Conditioning coordinates or fixed whole-car box do not match')
    patch = pca_point_patches(points)
    if not np.array_equal(patch['original_index'], np.arange(2048)):
        raise ValueError('Raw points are not 2048 distinct 5mm samples; refusing silent deduplication/reordering')
    arrays = dict(xyz=points.astype(np.float32), normal=patch['normal'].astype(np.float32),
        sigma=patch['sigma'].astype(np.float32), opacity=np.full(2048, .995, np.float32),
        thickness=np.full(2048, .003, np.float32), box=box.astype(np.float32),
        knn_spacing=patch['knn_spacing'].astype(np.float32), normal_radius=patch['normal_radius'].astype(np.float32),
        normal_confidence=patch['normal_confidence'].astype(np.float32), normal_valid=patch['normal_valid'],
        source_index=np.arange(2048, dtype=np.int64), generated=generated, conditioning=~generated,
        out_of_box=(np.abs(points) > box/2+1e-6).any(1))
    for key, values in [('conditioning_observation_index', conditioning_index),
                        ('conditioning_raw_frame', raw_frame), ('conditioning_cloud_row', cloud_row)]:
        arrays[key] = np.full(2048, -1, np.int64); arrays[key][~generated] = values
    arrays['conditioning_held'] = (~generated) & np.isin(arrays['conditioning_raw_frame'], HELD)
    geometry_hash = hash_geometry(arrays)
    metadata = dict(schema=1, asset='complete_original_2048_PVD_a6', raw_pvd=str(raw_path), raw_pvd_sha256=source_hash,
        canonical_source_sha256_verified=canonical_source, observation_geometry=str(observation_path),
        observation_geometry_sha256=sha_file(observation_path), transform_contract=str(transform_path),
        transform_contract_sha256=sha_file(transform_path), coordinate_contract=transform,
        points=2048, generated_points=1848, conditioning_points=200,
        normal_valid=int(arrays['normal_valid'].sum()), normal_invalid=int((~arrays['normal_valid']).sum()),
        out_of_box=int(arrays['out_of_box'].sum()), conditioning_previous_held_points=int(arrays['conditioning_held'].sum()),
        expected_canonical_conditioning_previous_held_points=14, previous_held=list(HELD),
        conditioning_source_frames=np.unique(raw_frame).tolist(), conditioning_source_max_abs_error_m=conditioning_error,
        canonical_axes='x forward/length, y left/width, z up/height; metres; input is already actor_native_m',
        pose_interface='Caller supplies the fixed annotated a6 model_from_local matrix; no native-a6 model is needed to load this asset',
        conditioning_provenance='Canonical source: 200 seed0 samples from raw LiDAR-associated OBB returns, checked by existing 199-frame TFRecord re-decode/CRC audit; not native Gaussian centers or rendered points',
        geometry_context='Frozen prior used raw0..198 geometry, including 14 old HELD conditioning points; RGB BUILD-only fitting is not independent geometry generalization',
        point_policy='Keep every original row, including conditioning, out-of-box and invalid-normal flags; no generated-only/near-measured/box/PCA rejection',
        normal_policy='PCA16 inside min(.25,max(4*median nearest3 spacing,.03)); smallest eigenvector; outward sign n dot point >=0; invalid flags retained without replacement',
        sigma_policy='min(.60*median nearest3 spacing,.060,.45*normal_radius) metres',
        gaussianization='Normals/scales estimated once; .995 peak opacity and .003m thickness are fixed rendering conventions, not diffusion-model outputs',
        geometry_hash=geometry_hash, geometry_hash_fields=list(GEOMETRY_FIELDS),
        appearance='Uniform .5 RGB; every colour initially inferred; no old1477 colours or historical199 RGB read',
        frozen=['xyz', 'normal', 'sigma', 'opacity', 'thickness', 'source_and_quality_flags'], geometry_only=True)
    if canonical_source and metadata['conditioning_previous_held_points'] != 14:
        raise ValueError('Canonical conditioning HELD provenance changed')
    arrays.update(rgb=np.full((2048, 3), .5, np.float32), colour_inferred=np.ones(2048, bool),
                  geometry_hash=np.asarray(geometry_hash), metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)))
    save_npz(geometry_path, arrays)
    audit = dict(metadata, status='GEOMETRY_BUILT_CPU', geometry_file=str(geometry_path),
                 geometry_npz_sha256=sha_file(geometry_path), GPU_used=False, training_updates=0)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return audit


class FixedAsset(nn.Module):
    """One complete car asset; all state is buffers and only RGB may be updated."""
    def __init__(self, arrays, metadata=None):
        super().__init__()
        for key in FLOAT_FIELDS:
            self.register_buffer(key, torch.as_tensor(np.array(arrays[key], copy=True), dtype=torch.float32))
        for key in BOOL_FIELDS:
            self.register_buffer(key, torch.as_tensor(np.array(arrays[key], copy=True), dtype=torch.bool))
        for key in INT_FIELDS:
            self.register_buffer(key, torch.as_tensor(np.array(arrays[key], copy=True), dtype=torch.int64))
        if self.xyz.shape != (2048, 3) or self.normal.shape != (2048, 3) or self.box.shape != (3,):
            raise ValueError('FixedAsset requires the unique complete 2048-row geometry')
        if not bool((self.box > 0).all()) or not torch.equal(self.source_index, torch.arange(2048)):
            raise ValueError('Invalid box or altered complete-PVD source order')
        if int(self.generated.sum()) != 1848 or not torch.equal(self.conditioning, ~self.generated):
            raise ValueError('Complete asset must retain 200 conditioning and 1848 generated rows')
        for key in GEOMETRY_FIELDS:
            value = getattr(self, key)
            if not bool(torch.isfinite(value).all()) or (key not in ('xyz', 'normal', 'box') and value.shape != (2048,)):
                raise ValueError(f'Invalid geometry field {key}')
        if not bool((self.sigma > 0).all() and (self.thickness > 0).all() and ((self.opacity > 0) & (self.opacity < 1)).all()):
            raise ValueError('Invalid fixed Gaussian scale/opacity')
        if not torch.allclose(self.normal.norm(dim=1), torch.ones(2048), atol=2e-6, rtol=0):
            raise ValueError('Fixed normals must be unit length')
        self.geometry_hash = hash_geometry(self._geometry_arrays())
        expected = str(np.asarray(arrays.get('geometry_hash', self.geometry_hash)).item())
        if self.geometry_hash != expected:
            raise ValueError('Saved geometry content hash mismatch')
        self.metadata = deepcopy(metadata or {})
        self.register_buffer('rgb', torch.full((2048, 3), .5, dtype=torch.float32))
        self.register_buffer('colour_inferred', torch.ones(2048, dtype=torch.bool))
        self.set_colours(arrays.get('rgb', self.rgb.numpy()), arrays.get('colour_inferred', None))
        # Loading geometry/exported colours must not rewrite their provenance.
        self.metadata = deepcopy(metadata or {})

    @classmethod
    def from_geometry(cls, path):
        with np.load(path, allow_pickle=False) as bank:
            arrays = {key:bank[key] for key in bank.files}
        metadata = json.loads(str(np.asarray(arrays.get('metadata_json', '{}')).item()))
        return cls(arrays, metadata)

    def _geometry_arrays(self):
        return {key:getattr(self, key).detach().cpu().numpy() for key in GEOMETRY_FIELDS}

    def current_geometry_hash(self):
        return hash_geometry(self._geometry_arrays())

    def assert_geometry_unchanged(self):
        if self.current_geometry_hash() != self.geometry_hash:
            raise RuntimeError('Frozen full-PVD geometry or provenance flags changed')
        return True

    @torch.no_grad()
    def set_colours(self, colours, colour_inferred=None):
        self.assert_geometry_unchanged()
        colour = torch.as_tensor(colours, dtype=self.rgb.dtype, device=self.rgb.device)
        if colour.shape != (2048, 3) or not bool(torch.isfinite(colour).all()) or not bool(((colour >= 0) & (colour <= 1)).all()):
            raise ValueError('Colours must be finite 2048x3 sRGB in [0,1]; no silent clipping')
        self.rgb.copy_(colour)
        if colour_inferred is not None:
            inferred = torch.as_tensor(colour_inferred, dtype=torch.bool, device=self.rgb.device)
            if inferred.shape != (2048,):
                raise ValueError('colour_inferred must have 2048 entries')
            self.colour_inferred.copy_(inferred)
        self.metadata['appearance'] = 'Caller-supplied RGB; FixedAsset reads no images/models; caller must record BUILD-only evidence'
        self.assert_geometry_unchanged()
        return self

    def rows(self, matrix, camera, pipe, channels, precomputed=False):
        matrix = torch.as_tensor(matrix, device=self.xyz.device, dtype=self.xyz.dtype)
        if matrix.shape != (4, 4) or not bool(torch.isfinite(matrix).all()) or int(channels) < 1:
            raise ValueError('Expected 4x4 actor pose and at least one SH channel')
        transform = matrix[:3, :3]
        model_scale = transform.norm(dim=0).mean()
        means = self.xyz@transform.T+matrix[:3, 3]
        normal = torch.nn.functional.normalize(self.normal@transform.T, dim=1)
        q = torch.stack((1+normal[:, 2], -normal[:, 1], normal[:, 0], torch.zeros_like(self.sigma)), 1)
        q = torch.where((q.norm(dim=1) < 1e-5)[:, None], q.new_tensor([0., 1., 0., 0.]), q)
        q = torch.nn.functional.normalize(q, dim=1)
        scales = torch.stack((self.sigma, self.sigma, self.thickness), 1)*model_scale
        cov = None
        if pipe.compute_cov3D_python:
            # Isotropic tangent plane: equivalent to R diag(scales^2) R^T.
            tangent, thin = scales[:, 0].square(), scales[:, 2].square()
            full = tangent[:, None, None]*torch.eye(3, dtype=means.dtype, device=means.device)
            full = full+(thin-tangent)[:, None, None]*normal[:, :, None]*normal[:, None, :]
            cov = full[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].contiguous()
        view = torch.as_tensor(camera.world_view_transform, dtype=means.dtype, device=means.device)
        depth = means@view[:3, 2:3]+view[3, 2]
        dc = ((self.rgb-.5)/.28209479177387814)[:, None]
        shs = None if precomputed else torch.cat((dc, dc.new_zeros((2048, int(channels)-1, 3))), 1)
        return dict(means3D=means, means2D=torch.zeros_like(means), shs=shs,
            colors_precomp=self.rgb if precomputed else None, features=torch.cat((depth, torch.ones_like(depth)), 1),
            opacities=self.opacity[:, None], scales=scales if cov is None else None,
            rotations=q if cov is None else None, cov3D_precomp=cov,
            mask=torch.ones(2048, dtype=torch.bool, device=means.device))

    def export(self, path):
        self.assert_geometry_unchanged()
        arrays = self._geometry_arrays()
        arrays.update(rgb=self.rgb.detach().cpu().numpy(), colour_inferred=self.colour_inferred.detach().cpu().numpy(),
            geometry_hash=np.asarray(self.geometry_hash), metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)))
        save_npz(path, arrays)
        self.assert_geometry_unchanged()
        return dict(path=str(path), file_sha256=sha_file(path), geometry_hash=self.geometry_hash, rows=2048)


def main():
    parser = argparse.ArgumentParser(description='CPU-only complete PVD Gaussianization; no model/render fitting')
    parser.add_argument('--raw', type=Path, default=RAW_PVD)
    parser.add_argument('--observations', type=Path, default=OBSERVATIONS)
    parser.add_argument('--transform', type=Path, default=TRANSFORM)
    parser.add_argument('--geometry', type=Path, default=RUN/'geometry.npz')
    parser.add_argument('--audit', type=Path, default=RUN/'geometry_provenance.json')
    args = parser.parse_args()
    result = geometry_builder(args.raw, args.geometry, args.audit, args.observations, args.transform)
    print(json.dumps({key:result[key] for key in ('status', 'points', 'geometry_file', 'geometry_hash', 'geometry_npz_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
