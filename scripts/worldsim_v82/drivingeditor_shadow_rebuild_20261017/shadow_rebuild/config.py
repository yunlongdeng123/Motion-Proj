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


@dataclass
class Config:
    schema: str = "instance_surface_shadow_v3"
    seed: int = 0
    threads: int = 2
    geometry_steps: int = 30000
    material_steps: int = 10000
    joint_steps: int = 5000
    position_lr: float = .0002
    appearance_lr: float = .002
    opacity_lr: float = .005
    shape_lr: float = .0005
    light_lr: float = .002
    joint_lr_scale: float = .1
    geometry_bound_m: float = .2
    joint_bound_m: float = .02
    scale_min_m: float = .002
    scale_max_m: float = .5
    initial_opacity: float = .6
    train_opacity_stage2: bool = False
    freeze_joint_geometry: bool = False
    use_clean_road: bool = True
    use_body_teacher: bool = True
    shadows: bool = True
    instance_balanced_supervision: bool = False
    outside_fullscene_visibility: bool = False
    instance_balanced_visible_identity: bool = False
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
    densify_until: int = 18000
    densify_fraction: float = .02
    max_gaussians: int = 450000
    max_features: int = 10
    log_interval: int = 50
    checkpoint_interval: int = 1000
    ablation: str = "full"
    weights: LossWeights = field(default_factory=LossWeights)

    def validate(self):
        if any(type(v) is not bool for v in [self.instance_balanced_supervision,self.outside_fullscene_visibility,self.instance_balanced_visible_identity]):
            raise ValueError('instance supervision flags必须为布尔值')
        if self.schema != "instance_surface_shadow_v3" or self.caster != "actor_geometry":
            raise ValueError("此重建分支仅接受原始测量初始化和直接 Actor 高斯遮挡体")
        positive = ("threads", "geometry_steps", "material_steps", "joint_steps", "ray_chunk",
                    "gaussian_chunk", "shading_patch", "sun_samples", "max_gaussians", "log_interval", "checkpoint_interval")
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in positive):
            raise ValueError("阶段步数、线程和分块必须为正整数")
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
