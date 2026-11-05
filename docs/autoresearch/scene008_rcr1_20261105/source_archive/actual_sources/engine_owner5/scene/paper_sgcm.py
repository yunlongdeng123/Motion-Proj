"""Paper Eq8 SGCM with explicit, bounded engineering choices.

The published paper specifies distance-based prune/clone/split, dr=0.5m, and
dynamic-LiDAR support, but omits region representation, ground extraction,
fitting, schedule and split threshold. This module records those choices;
it does not assert their equivalence to unpublished author implementation.
Geometry/fitting are CPU NumPy. Torch is imported only by refine().
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np

from scene.hybrid_sgcm_helpers import (bind_sgcm_guard, guard_additions, select_clone_once,
                                       mark_clone_lineages_spent)


class SGCMGeometryError(ValueError):
    """A trustworthy metric-space ground model cannot be established."""


@dataclass
class SGCMConfig:
    dr_m: float = 0.5
    interval: int = 100
    offset: int = 50
    start: int = 500
    end: int = 15000
    max_new_points: int = 10000
    split_radius_px: float = 20.0
    ground_margin_m: float = 3.0
    ground_cell_m: float = 0.5
    plane_inlier_m: float = 0.15
    ground_min_cells: int = 30
    ground_min_fraction: float = 0.35
    ground_max_tilt_deg: float = 20.0
    ground_max_rmse_m: float = 0.10
    region_margin_m: float = 0.25
    max_points: int = 3000000
    guard_enabled: bool = True

    @classmethod
    def from_args(cls, args):
        values = {}
        for name in cls.__dataclass_fields__:
            key = 'sgcm_' + name
            value = getattr(args, key, None)
            if value is not None:
                values[name] = value
        config = cls(**values)
        if type(config.guard_enabled) is not bool:
            raise ValueError('SGCM guard_enabled must be a boolean')
        if config.interval <= 0 or not 0 <= config.offset < config.interval:
            raise ValueError('Invalid SGCM interval/offset')
        if config.dr_m <= 0 or config.split_radius_px <= 0:
            raise ValueError('SGCM distance and projected-radius thresholds must be positive')
        if config.max_new_points < 0 or config.max_points < 1:
            raise ValueError('SGCM point budgets must be finite and nonnegative')
        if config.end <= config.start:
            raise ValueError('SGCM end must exceed start')
        if config.ground_cell_m <= 0 or config.plane_inlier_m <= 0:
            raise ValueError('Ground grid and inlier distances must be positive')
        # Refuse any schedule that can share an iteration with generic topology
        # updates. Matching point counts alone cannot prove indices are current.
        generic_interval = int(getattr(args,'densification_interval',100))
        generic_start = int(getattr(args,'densify_from_iter',500))
        generic_end = int(getattr(args,'densify_until_iter',15000))
        common = math.gcd(config.interval,generic_interval)
        if config.offset % common == 0:
            modulus = config.interval // common
            k = (config.offset//common * pow(generic_interval//common,-1,modulus)) % modulus if modulus>1 else 0
            first = k*generic_interval
            period = config.interval*generic_interval//common
            lower = max(config.start,generic_start+1)
            first += max(0,(lower-first+period-1)//period)*period
            if first < min(config.end,generic_end):
                raise ValueError('SGCM schedule overlaps generic densification at iteration%d' % first)
        return config


def validate_similarity(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SGCMGeometryError('World-to-model transform must be finite4x4')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
        raise SGCMGeometryError('World-to-model transform is not affine')
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    scale = float(singular.mean())
    if scale <= 0 or not np.allclose(singular, scale, rtol=1e-4, atol=1e-10):
        raise SGCMGeometryError('SGCM requires an isotropic metric similarity transform')
    return scale


def transform_points(points, affine):
    points = np.asarray(points, dtype=np.float64)
    return points @ affine[:3, :3].T + affine[:3, 3]


def transform_plane(plane, source_to_target):
    result = np.linalg.solve(np.asarray(source_to_target).T,
                             np.asarray(plane, dtype=np.float64))
    norm = np.linalg.norm(result[:3])
    if norm <= 0 or not np.isfinite(result).all():
        raise SGCMGeometryError('Invalid transformed ground plane')
    return result / norm


def lower_grid_samples(points, cell_m):
    """One lower-envelope representative per occupied XY cell, min3points.

    Ten-percentile height avoids trusting the absolute minimum of noisy LiDAR.
    Equal cell weighting avoids high-density surfaces dominating plane fitting.
    These are reproducibility choices absent from the paper.
    """
    cells = np.floor(points[:, :2] / cell_m).astype(np.int64)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(group, kind='stable')
    cuts = np.flatnonzero(np.diff(group[order])) + 1
    rows = []
    for indices in np.split(order, cuts):
        if len(indices) < 3:
            continue
        cell = points[indices]
        z = np.quantile(cell[:, 2], 0.10)
        row = cell[np.argmin(np.abs(cell[:, 2] - z))]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def fit_ground_plane(points_anchor, actor_bounds, config):
    """Fit z=a*x+b*y+c in the first vehicle's up-oriented frame.

    Bounded deterministic RANSAC, then inlier least-squares. No flat-plane or
    all-points fallback: failure is explicit. Ground selection is static LiDAR
    near the actor XY footprint; this extends the paper's Hd support by a
    declared margin to obtain unobscured nearby road evidence.
    """
    points = np.asarray(points_anchor, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    neighbourhood = np.zeros(len(points), dtype=bool)
    for bounds in actor_bounds:
        lower = bounds[0, :2] - config.ground_margin_m
        upper = bounds[1, :2] + config.ground_margin_m
        neighbourhood |= np.all((points[:, :2] >= lower) & (points[:, :2] <= upper), axis=1)
    points = points[valid & neighbourhood]
    seeds = lower_grid_samples(points, config.ground_cell_m)
    if len(seeds) < config.ground_min_cells:
        raise SGCMGeometryError('Insufficient occupied ground cells: %d' % len(seeds))
    # Keep maximum computation bounded, independently of sequence length.
    if len(seeds) > 10000:
        indices = np.linspace(0, len(seeds)-1, 10000, dtype=int)
        seeds = seeds[indices]
    xy1 = np.column_stack((seeds[:, :2], np.ones(len(seeds))))
    random = np.random.RandomState(0)
    best = None
    for _ in range(256):
        indices = random.choice(len(seeds), 3, replace=False)
        design = xy1[indices]
        if abs(np.linalg.det(design)) < 0.1:
            continue
        coefficients = np.linalg.solve(design, seeds[indices, 2])
        slope = np.linalg.norm(coefficients[:2])
        if slope > np.tan(np.deg2rad(config.ground_max_tilt_deg)):
            continue
        distances = np.abs(xy1 @ coefficients - seeds[:, 2]) / np.sqrt(1+slope*slope)
        inliers = distances <= config.plane_inlier_m
        count = int(inliers.sum())
        score = (count, -float(np.median(distances[inliers])) if count else -np.inf)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        raise SGCMGeometryError('No low-tilt ground plane has sufficient noncollinear support')
    inliers = best[1]
    for _ in range(3):
        if inliers.sum() < config.ground_min_cells:
            raise SGCMGeometryError('Ground plane has insufficient inlier cells')
        coefficients, _, rank, _ = np.linalg.lstsq(xy1[inliers], seeds[inliers, 2], rcond=None)
        if rank != 3:
            raise SGCMGeometryError('Ground support is collinear')
        slope = np.linalg.norm(coefficients[:2])
        distances = np.abs(xy1 @ coefficients - seeds[:, 2]) / np.sqrt(1+slope*slope)
        inliers = distances <= config.plane_inlier_m
    count = int(inliers.sum())
    fraction = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(distances[inliers]**2))) if count else np.inf
    tilt = float(np.rad2deg(np.arctan(slope)))
    span = np.ptp(seeds[inliers, :2], axis=0) if count else np.zeros(2)
    if (count < config.ground_min_cells or fraction < config.ground_min_fraction or
            rmse > config.ground_max_rmse_m or tilt > config.ground_max_tilt_deg or
            np.any(span < 2.0)):
        raise SGCMGeometryError('Ground validation failed: cells=%d fraction=%.3f rmse=%.3fm tilt=%.2fdeg span=%s' %
                                (count, fraction, rmse, tilt, span.tolist()))
    plane = np.array([-coefficients[0], -coefficients[1], 1.0, -coefficients[2]])
    plane /= np.linalg.norm(plane[:3])
    return plane, {'candidate_points':len(points), 'lower_envelope_cells':len(seeds),
                   'inlier_cells':count, 'inlier_fraction':fraction,
                   'rmse_m':rmse, 'tilt_from_first_vehicle_up_deg':tilt,
                   'inlier_xy_span_m':span.tolist(), 'plane_anchor':plane.tolist()}


class SGCMController:
    def __init__(self, config, model_scale, plane_model, model_to_anchor, regions_anchor, metadata):
        self.config = config
        self.model_scale = model_scale
        self.plane_model = plane_model
        self.model_to_anchor = model_to_anchor
        self.regions_anchor = regions_anchor
        self.metadata = metadata
        self.training_views = ()

    def due(self, iteration):
        cfg = self.config
        return cfg.start <= iteration < cfg.end and iteration % cfg.interval == cfg.offset

    def classify_numpy(self, xyz, radii):
        """Metric-invariant Eq8 classification for diagnostics and CPU tests."""
        xyz = np.asarray(xyz, dtype=np.float64)
        radii = np.asarray(radii)
        if radii.shape != (len(xyz),) or not np.isfinite(xyz).all() or not np.isfinite(radii).all():
            raise SGCMGeometryError('Invalid radii or Gaussian coordinates')
        anchor = transform_points(xyz, self.model_to_anchor)
        inside = np.zeros(len(xyz), dtype=bool)
        for bounds in self.regions_anchor:
            inside |= np.all((anchor >= bounds[0]) & (anchor <= bounds[1]), axis=1)
        distance_m = np.abs(xyz @ self.plane_model[:3] + self.plane_model[3]) / self.model_scale
        dr = self.config.dr_m
        return {'prune':inside & (distance_m > dr),
                'clone':inside & (distance_m > 0.5*dr) & (distance_m <= dr),
                'split':inside & (distance_m <= 0.5*dr) & (radii > self.config.split_radius_px),
                'inside':inside, 'distance_m':distance_m}

    def refine(self, gaussians, radii, iteration, visibility_filter=None):
        if not self.due(iteration):
            return {}
        import torch
        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            if radii.ndim != 1 or len(radii) != len(xyz):
                raise RuntimeError('SGCM radii must match current topology; schedule must avoid generic densification')
            if visibility_filter is not None:
                if (visibility_filter.shape != radii.shape or visibility_filter.dtype != torch.bool
                        or not torch.equal(visibility_filter, radii > 0)):
                    raise RuntimeError('SGCM current visibility must be the aligned static rendered radii>0 filter')
            if not torch.isfinite(xyz).all() or not torch.isfinite(radii).all():
                raise SGCMGeometryError('Nonfinite current Gaussian geometry or projected radius')
            cfg = self.config
            affine = torch.as_tensor(self.model_to_anchor, dtype=xyz.dtype, device=xyz.device)
            anchor = xyz @ affine[:3, :3].T + affine[:3, 3]
            inside = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
            for region in self.regions_anchor:
                bounds = torch.as_tensor(region, dtype=xyz.dtype, device=xyz.device)
                inside |= ((anchor >= bounds[0]) & (anchor <= bounds[1])).all(dim=1)
            plane = torch.as_tensor(self.plane_model, dtype=xyz.dtype, device=xyz.device)
            distance_m = torch.abs(xyz @ plane[:3] + plane[3]) / self.model_scale
            prune = inside & (distance_m > cfg.dr_m)
            clone = inside & (distance_m > 0.5*cfg.dr_m) & (distance_m <= cfg.dr_m)
            split = inside & (distance_m <= 0.5*cfg.dr_m) & (radii > cfg.split_radius_px)
            guard_statistics = {}
            if cfg.guard_enabled:
                clone, split, guard_statistics = guard_additions(self, gaussians, clone, split)
            # Exact geometry rules, bounded resource policy: split largest image
            # footprints first, then clone largest visible footprints. Selection
            # cap is an explicit approximation, not a paper-reported parameter.
            prune_count = int(prune.sum().item())
            after_prune = len(xyz)-prune_count
            if after_prune == 0:
                raise SGCMGeometryError('SGCM would remove every static Gaussian')
            budget = min(cfg.max_new_points, max(0, cfg.max_points-after_prune))
            split_ids = torch.nonzero(split, as_tuple=False).flatten()
            if len(split_ids):
                split_ids = split_ids[torch.argsort(radii[split_ids], descending=True, stable=True)[:budget]]
            budget -= len(split_ids)
            clone_ids = torch.nonzero(clone, as_tuple=False).flatten()
            if len(clone_ids):
                if cfg.guard_enabled:
                    clone_ids, duplicate_count = select_clone_once(gaussians, clone_ids, radii, budget)
                    guard_statistics['excluded_duplicate_lineage_in_step'] = duplicate_count
                else:
                    clone_ids = clone_ids[torch.argsort(radii[clone_ids], descending=True, stable=True)[:budget]]
            remove = prune.clone()
            remove[split_ids] = True
            parents = torch.cat((clone_ids, split_ids, split_ids))
            if cfg.guard_enabled:
                mark_clone_lineages_spent(gaussians, clone_ids)
            # Reuse GaussianModel's optimizer-aware append/prune methods. Clone
            # copies parameters, as standard3DGS. Spatial split is a symmetric,
            # deterministic two-component moment-matching mixture along the
            # largest covariance axis; temporal fields are copied unchanged.
            if len(parents):
                fields = ['_xyz','_features_dc','_features_rest','_opacity','_scaling',
                          '_rotation','_t','_scaling_t','_velocity']
                children = [getattr(gaussians, name)[parents].detach().clone() for name in fields]
                if len(split_ids):
                    nclone, nsplit = len(clone_ids), len(split_ids)
                    scales = gaussians.get_scaling[split_ids]
                    largest = scales.argmax(dim=1)
                    quaternion = gaussians._rotation[split_ids]
                    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    w,x,y,z = quaternion.unbind(dim=1)
                    rotation = torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                                            2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                                            2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)), dim=1).reshape(-1,3,3)
                    direction = rotation[torch.arange(nsplit,device=xyz.device),:,largest]
                    width = scales.gather(1,largest[:,None]) / np.sqrt(2.0)
                    offset = direction * width
                    children[0][nclone:nclone+nsplit] -= offset
                    children[0][nclone+nsplit:] += offset
                    split_scales = scales.clone()
                    split_scales.scatter_(1,largest[:,None],width)
                    children[4][nclone:] = gaussians.scaling_inverse_activation(split_scales).repeat(2,1)
                # Preserve accumulated statistics for all existing survivors;
                # densification_postfix normally resets them globally.
                statistics = {name:getattr(gaussians,name).detach().clone() for name in
                              ['max_radii2D','xyz_gradient_accum','t_gradient_accum','denom']}
                gaussians.densification_postfix(*children, sgcm_parent_indices=parents)
                for name, old in statistics.items():
                    getattr(gaussians,name)[:len(old)] = old
                remove = torch.cat((remove,torch.zeros(len(parents),dtype=torch.bool,device=xyz.device)))
            if remove.any():
                gaussians.prune_points(remove)
            return {'iteration':int(iteration),'inside_region':int(inside.sum().item()),
                    'pruned':prune_count,'cloned':len(clone_ids),'split_parents':len(split_ids),
                    'candidate_clone':int(clone.sum().item()),'candidate_split':int(split.sum().item()),
                    'points_before':len(xyz),'points_after':len(gaussians.get_xyz),
                    'dr_m':cfg.dr_m,'dr_model':cfg.dr_m*self.model_scale,
                    **guard_statistics}


def create_sgcm_controller(source_path, args, training_views=None, gaussians=None):
    """Initialize from unnormalized world LiDAR and the actual loader transform.

    Requires Scene construction first so the current camera configuration's
    transform exists. This factory never changes priors or the source dataset.
    """
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    cfg = SGCMConfig.from_args(args)
    root = Path(source_path)
    precise = root/'sgcm_transform_matrix.txt'
    transform_path = precise if precise.exists() else root/'transform_matrix.txt'
    if not transform_path.is_file():
        raise SGCMGeometryError('Current world-to-model loader transform is missing')
    transform = np.loadtxt(transform_path)
    scale = validate_similarity(transform)
    poses = sorted((root/'pose').glob('*.txt'))
    if not poses:
        raise SGCMGeometryError('First vehicle pose is missing')
    first_pose = np.loadtxt(poses[0])
    pose_scale = validate_similarity(first_pose)
    if not np.isclose(pose_scale,1.0,rtol=1e-4):
        raise SGCMGeometryError('Vehicle pose must be rigid in meters')
    def read_xyz(path):
        vertices = PlyData.read(str(path))['vertex']
        xyz = np.column_stack([vertices[key] for key in ['x','y','z']]).astype(np.float64)
        if len(xyz)<3 or not np.isfinite(xyz).all():
            raise SGCMGeometryError('Empty or invalid prior: '+str(path))
        return xyz
    static_world = read_xyz(root/'points3d.ply')
    dynamic_paths = sorted(root.glob('points3d_dynamic_*.ply'))
    if not dynamic_paths:
        raise SGCMGeometryError('SGCM requires real dynamic LiDAR priors')
    actors_world = [read_xyz(path) for path in dynamic_paths]
    dynamic_world = np.concatenate(actors_world)
    # Match the loader's strict0.1m removal rule before selecting static ground.
    distance,_ = cKDTree(dynamic_world).query(static_world,k=1,eps=0,distance_upper_bound=0.1,workers=4)
    static_world = static_world[distance>=0.1]
    world_to_anchor = np.linalg.inv(first_pose)
    static_anchor = transform_points(static_world,world_to_anchor)
    actors_anchor = [transform_points(points,world_to_anchor) for points in actors_world]
    bounds = [np.stack((points.min(axis=0),points.max(axis=0))) for points in actors_anchor]
    plane_anchor,fit = fit_ground_plane(static_anchor,bounds,cfg)
    anchor_to_model = transform @ first_pose
    plane_model = transform_plane(plane_anchor,anchor_to_model)
    regions = []
    for box in bounds:
        region = box.copy()
        region[0,:2] -= cfg.region_margin_m
        region[1,:2] += cfg.region_margin_m
        corners = np.array([[x,y] for x in region[:,0] for y in region[:,1]])
        ground_z = -(corners @ plane_anchor[:2]+plane_anchor[3])/plane_anchor[2]
        region[0,2] = min(region[0,2],float(ground_z.min()))-cfg.region_margin_m
        region[1,2] = max(region[1,2],float(ground_z.max()))+cfg.region_margin_m
        regions.append(region)
    metadata = {'status':'validated','paper_reference':'Eq8,printed1701/PDF6',
                'config':asdict(cfg),'meter_to_model_scale':scale,'dr_model':cfg.dr_m*scale,
                'transform_path':str(transform_path),'transform_validation':'finite_isotropic_similarity',
                'world_to_model':transform.tolist(),'anchor_pose':str(poses[0]),
                'plane_model':plane_model.tolist(),'ground_fit':fit,
                'regions_anchor':[region.tolist() for region in regions],
                'assumptions':['First vehicle up defines local vertical; low-tilt ground required.',
                    'Hd is the union of actor time-aggregated LiDAR AABBs in the first vehicle frame; extends vertically to fitted ground.',
                    'Ground support includes a3m XY margin around actor support;0.5m grid10th-percentile lower envelope; deterministic256trial RANSAC and least-squares.',
                    'SGCM executes after optimizer on an offset schedule avoiding generic densification;20px radius is the undeclared projected-footprint threshold choice.',
                    'Resource cap limits additions per step; large projected footprints prioritized.',
                    'Clone copies parameters; split uses symmetric two-child largest-axis moment-matched spatial subdivision and unchanged temporal fields.'],
                'unknown_author_details':['Region shape/update','Ground extraction/fit','Schedule','Footprint threshold','Clone/split policy/budget']}
    controller = SGCMController(cfg,scale,plane_model,np.linalg.inv(anchor_to_model),regions,metadata)
    bind_sgcm_guard(controller, training_views, gaussians,
                    resuming=bool(getattr(args, 'start_checkpoint', None)))
    print('SGCM_INIT '+json.dumps(metadata,sort_keys=True),flush=True)
    return controller
