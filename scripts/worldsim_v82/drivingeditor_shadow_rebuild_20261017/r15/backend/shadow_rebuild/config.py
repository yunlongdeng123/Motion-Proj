"""配置只允许明确字段；每个消融会写入独立的 resolved_config.json。"""
from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path


@dataclass
class LossWeights:
    rgb: float = 1.0
    ssim: float = .2
    depth: float = .1
    mask_inside: float = .05
    mask_outside: float = .05
    visible_instance: float = .1
    clean_road: float = 1.0
    reconstruction: float = .3
    body_rgb: float = 1.0
    body_teacher: float = .1
    geometry: float = .02
    normal: float = .02
    view_residual: float = .01
    road_identity: float = 0.0
    ownership_occupancy: float = .1
    ownership_render: float = .05


@dataclass
class Config:
    schema: str = "instance_surface_shadow_v3"
    backend: str = "legacy_sh3_envlight"
    sh_degree: int = 3
    env_map_res: int = 1024
    fixture_small_environment: bool = False
    position_lr_final: float = .0000016
    position_lr_delay_mult: float = .01
    spatial_lr_scale: float = 1.0
    sh_increase_interval: int = 1000
    shadow_train_stride: int = 16
    feature_rest_lr_divisor: float = 20.0
    topology_grad_threshold: float = .00017
    topology_percent_dense: float = .01
    prune_opacity_threshold: float = .005
    opacity_reset_interval: int = 3000
    legacy_reference_config: str = ""
    depth_normalization_scale: float = .1
    sgcm_geometry_path: str = ""
    sgcm_enabled: bool = True
    box_half_sizes_path: str = ""
    legacy_box_boundary_weight: float = 1.0
    legacy_visible_mask_weight: float = 100.0
    seed: int = 0
    threads: int = 2
    geometry_steps: int = 30000
    material_steps: int = 10000
    joint_steps: int = 5000
    position_lr: float = .00016
    appearance_lr: float = .0025
    opacity_lr: float = .005
    shape_lr: float = .005
    light_lr: float = .002
    joint_lr_scale: float = .1
    # Opt-in repaired R15; old checkpoint/config replay keeps its exact schema.
    repair_actor_geometry: bool = False
    actor_drift_bound_m: float = .25
    actor_box_margin_m: float = .10
    actor_scale_growth_limit: float = 1.5
    actor_max_growth: float = 4.0
    repair_evidence_audit: bool = False
    ownership_audit_first: int = 1500
    ownership_audit_interval: int = 1000
    ownership_measurement_root: str = ""
    shadow_integrator: str = "legacy_covariance"
    geometry_bound_m: float = 1000.0
    joint_bound_m: float = .02
    scale_min_m: float = .00001
    scale_max_m: float = 1000.0
    initial_opacity: float = .01
    train_opacity_stage2: bool = False
    freeze_joint_geometry: bool = False
    use_clean_road: bool = True
    use_body_teacher: bool = True
    shadows: bool = True
    instance_balanced_supervision: bool = False
    outside_fullscene_visibility: bool = False
    instance_balanced_visible_identity: bool = False
    continuous_ownership: bool = False
    ownership_depth_tolerance_m: float = .15
    ownership_warmup_steps: int = 500
    material_opacity_lr_scale: float = .1
    caster: str = "actor_geometry"
    sun_samples: int = 5
    sun_angular_radius_degrees: float = .53
    sun_elevation_degrees: float = 45.0
    sun_azimuth_degrees: float = 0.0
    ambient_fraction: float = .35
    shadow_strength: float = .95
    ray_bias_m: float = .003
    ray_chunk: int = 256
    gaussian_chunk: int = 512
    shading_patch: int = 32
    uniform_rays: int = 1024
    densify_interval: int = 500
    densify_from: int = 500
    densify_until: int = 18000
    densify_fraction: float = .02
    max_gaussians: int = 3000000
    max_features: int = 10
    log_interval: int = 50
    checkpoint_interval: int = 1000
    ablation: str = "full"
    weights: LossWeights = field(default_factory=LossWeights)

    def validate(self):
        if type(self.repair_actor_geometry) is not bool or type(self.repair_evidence_audit) is not bool:
            raise ValueError('repair switches must be explicit bools')
        if self.shadow_integrator not in ('legacy_covariance','factor_fp64'):
            raise ValueError('unknown shadow integrator')
        if self.repair_actor_geometry and (self.actor_drift_bound_m<=0 or self.actor_scale_growth_limit<1 or self.actor_max_growth<1 or not 0<self.densify_fraction<=1):
            raise ValueError('invalid actor geometry budget')
        if self.repair_evidence_audit:
            if not self.repair_actor_geometry or not self.ownership_measurement_root:
                raise ValueError('evidence audit requires explicit geometry/provenance contract')
            if self.ownership_audit_first<1 or self.ownership_audit_interval<1:
                raise ValueError('invalid evidence schedule')
            if any(s%self.opacity_reset_interval==0 for s in range(self.ownership_audit_first,self.geometry_steps+1,self.ownership_audit_interval)):
                raise ValueError('ownership audit must not coincide with opacity reset')
        if self.backend!="legacy_sh3_envlight" or self.sh_degree!=3:
            raise ValueError("R15 requires restored third-order SH backend")
        if self.env_map_res!=1024 and not self.fixture_small_environment:
            raise ValueError("Production EnvLight must retain 6x1024x1024 capacity")
        if self.max_gaussians<3000000 and not self.fixture_small_environment:
            raise ValueError("Production background capacity silently downgraded")
        if self.shadow_train_stride<1 or self.sh_increase_interval<1 or self.spatial_lr_scale<=0:
            raise ValueError("invalid restored training schedule")
        if any(type(v) is not bool for v in [self.instance_balanced_supervision,self.outside_fullscene_visibility,self.instance_balanced_visible_identity,self.continuous_ownership]):
            raise ValueError('instance supervision flags必须为布尔值')
        if self.schema != "instance_surface_shadow_v3" or self.caster != "actor_geometry":
            raise ValueError("此重建分支仅接受原始测量初始化和直接 Actor 高斯遮挡体")
        positive = ("threads", "geometry_steps", "material_steps", "joint_steps", "ray_chunk",
                    "gaussian_chunk", "shading_patch", "sun_samples", "max_gaussians", "log_interval", "checkpoint_interval")
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in positive):
            raise ValueError("阶段步数、线程和分块必须为正整数")
        if type(self.ownership_warmup_steps) is not int or self.ownership_warmup_steps<1 or self.ownership_depth_tolerance_m<=0:
            raise ValueError('ownership warmup/tolerance无效')
        if self.max_features not in (9, 10) or not 0 < self.initial_opacity < 1 or not 0 < self.ambient_fraction < 1:
            raise ValueError("feature 容量/opacity/ambient 无效")
        if not 0 < self.geometry_bound_m or not 0 < self.joint_bound_m <= self.geometry_bound_m:
            raise ValueError("几何漂移边界无效")
        if not 0 < self.scale_min_m < self.scale_max_m or not 0 <= self.densify_fraction <= 1:
            raise ValueError("尺度/增密范围无效")
        import math
        numeric = [v for k, v in asdict(self).items() if type(v) is float and k != "sun_azimuth_degrees"]
        if not all(math.isfinite(v) and v >= 0 for v in numeric):
            raise ValueError("非有限或负配置")
        if not all(math.isfinite(v) and v >= 0 for v in asdict(self.weights).values()):
            raise ValueError("损失权重无效")
        if not 0 < self.shadow_strength < 1 or not 0 < self.sun_elevation_degrees < 89.9:
            raise ValueError("太阳初始值无效")
        if not math.isfinite(self.sun_azimuth_degrees) or not 0 < self.sun_angular_radius_degrees <= 5:
            raise ValueError("太阳方向/盘半径无效")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 or type(self.uniform_rays) is not int or self.uniform_rays < 1:
            raise ValueError("seed/整图采样数无效")
        return self

    @classmethod
    def load(cls, path, ablation=None):
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = set(raw)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"未知配置字段: {sorted(unknown)}")
        raw["weights"] = LossWeights(**raw.get("weights", {}))
        c = cls(**raw)
        if ablation:
            if ablation not in ABLATIONS:
                raise ValueError(f"未知消融 {ablation}")
            for key, value in ABLATIONS[ablation].items():
                setattr(c, key, value)
            c.ablation = ablation
        return c.validate()


ABLATIONS = {
    "full": {},
    "no_shadow": {"shadows": False},
    "no_clean_road": {"use_clean_road": False},
    "no_body_teacher": {"use_body_teacher": False},
    "frozen_joint_geometry": {"freeze_joint_geometry": True},
    "hard_sun": {"sun_samples": 1},
    "opacity_stage2": {"train_opacity_stage2": True},
}
