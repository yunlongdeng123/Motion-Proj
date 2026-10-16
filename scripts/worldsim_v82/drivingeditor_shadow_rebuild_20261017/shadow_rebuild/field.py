"""同一组几何同时用于主相机与太阳遮挡；不存在独立 RGB 阴影形变参数。"""
import torch
from torch import nn
from .math3d import covariance, logit, normalize, quaternion_matrix, rotation_z, srgb_to_linear


class GaussianField(nn.Module):
    point_parameters = ("xyz", "log_scales", "quaternion", "opacity_logit", "appearance_logit", "albedo_logit", "view_coeff")

    def __init__(self, seed, config):
        super().__init__()
        self.config = config
        xyz = torch.as_tensor(seed["xyz"], dtype=torch.float32)
        initial = seed["rgb"] if config.use_clean_road else seed["observed_rgb"]
        color = srgb_to_linear(torch.as_tensor(initial, dtype=torch.float32)).clamp(.01, .99)
        n = len(xyz)
        self.xyz = nn.Parameter(xyz.clone())
        self.log_scales = nn.Parameter(torch.as_tensor(seed["scales"], dtype=torch.float32).log())
        self.quaternion = nn.Parameter(torch.as_tensor(seed["quaternion"], dtype=torch.float32).clone())
        self.opacity_logit = nn.Parameter(torch.full((n, 1), torch.logit(torch.tensor(config.initial_opacity)).item()))
        self.appearance_logit = nn.Parameter(logit(color))
        self.albedo_logit = nn.Parameter(logit(color))
        self.view_coeff = nn.Parameter(torch.zeros(n, 3, 3))
        for key, dtype in (("owner", torch.long), ("road", torch.bool), ("reference_rgb", torch.float32),
                           ("reference_confidence", torch.float32), ("neighbor", torch.long)):
            self.register_buffer(key, torch.as_tensor(seed[key], dtype=dtype).clone())
        self.register_buffer("seed_xyz", xyz.clone())
        self.register_buffer("anchor_xyz", xyz.clone())
        self.register_buffer("anchor_log_scales", self.log_scales.detach().clone())
        self.register_buffer("anchor_quaternion", self.quaternion.detach().clone())
        self.register_buffer("anchor_opacity", self.opacity_logit.detach().clone())
        self.register_buffer("stage3_anchor_valid", torch.tensor(False))

    @property
    def scales(self):
        return self.log_scales.exp().clamp(self.config.scale_min_m, self.config.scale_max_m)

    def set_stage(self, stage):
        geometry = stage == "geometry" or (stage == "joint" and not self.config.freeze_joint_geometry)
        for name in self.point_parameters:
            allowed = geometry if name in ("xyz", "log_scales", "quaternion") else True
            if name == "opacity_logit":
                allowed = geometry or (stage == "material" and self.config.train_opacity_stage2)
            if name == "albedo_logit":
                allowed = stage != "geometry"
            getattr(self, name).requires_grad_(allowed)

    @torch.no_grad()
    def anchor(self):
        for key, value in (("anchor_xyz", self.xyz), ("anchor_log_scales", self.log_scales),
                           ("anchor_quaternion", self.quaternion), ("anchor_opacity", self.opacity_logit)):
            getattr(self, key).copy_(value)
        self.stage3_anchor_valid.fill_(True)

    @torch.no_grad()
    def project_constraints(self, stage):
        center = self.anchor_xyz if stage == "joint" else self.seed_xyz
        bound = self.config.joint_bound_m if stage == "joint" else self.config.geometry_bound_m
        delta = self.xyz-center
        delta *= (bound/delta.norm(dim=-1, keepdim=True).clamp_min(bound)).clamp_max(1)
        self.xyz.copy_(center+delta)
        self.log_scales.clamp_(torch.log(torch.tensor(self.config.scale_min_m)).item(),
                              torch.log(torch.tensor(self.config.scale_max_m)).item())
        if stage == "joint":
            self.log_scales.copy_(self.anchor_log_scales + (self.log_scales-self.anchor_log_scales).clamp(-.12, .12))
            self.opacity_logit.copy_(self.anchor_opacity+(self.opacity_logit-self.anchor_opacity).clamp(-.3, .3))
        self.quaternion.copy_(normalize(self.quaternion))
        self.opacity_logit.clamp_(-8, 8)

    def evaluate(self, poses, valid, camera_center, edits=None):
        """Actor 删除/旋转/平移仅此处执行；主渲染和遮光共享返回的 xyz/covariance。"""
        edits = edits or {}
        n = len(self.xyz)
        rotation = torch.eye(3, device=self.xyz.device).expand(n, 3, 3).clone()
        translation = torch.zeros_like(self.xyz)
        keep = torch.ones(n, dtype=torch.bool, device=self.xyz.device)
        for actor in range(len(poses)):
            mask = self.owner == actor
            edit = edits.get(actor, edits.get(str(actor), {}))
            if not bool(valid[actor]) or edit.get("delete", False):
                keep[mask] = False
                continue
            r, t = poses[actor, :3, :3], poses[actor, :3, 3]
            if "yaw" in edit:
                r = rotation_z(edit["yaw"], device=r.device, dtype=r.dtype) @ r
            if "translation" in edit:
                t = t+torch.as_tensor(edit["translation"], device=t.device, dtype=t.dtype)
            rotation[mask], translation[mask] = r, t
        xyz = (rotation @ self.xyz[..., None]).squeeze(-1)+translation
        local_r = quaternion_matrix(self.quaternion)
        normal = (rotation @ local_r[..., 2:3]).squeeze(-1)
        cov = rotation @ covariance(self.quaternion, self.scales) @ rotation.transpose(-1, -2)
        direction = normalize((rotation.transpose(-1, -2) @ (camera_center-xyz)[..., None]).squeeze(-1))
        # 只允许 Actor 有小幅、无时间输入的视角残差。道路材质没有 SH。
        residual = .08*torch.tanh(torch.einsum("nij,ni->nj", self.view_coeff, direction))*(self.owner >= 0)[:, None]
        return dict(xyz=xyz[keep], cov=cov[keep], normal=normal[keep], opacity=self.opacity_logit.sigmoid()[keep],
                    appearance=(self.appearance_logit.sigmoid()+residual).clamp(0, 1)[keep],
                    albedo=self.albedo_logit.sigmoid()[keep], view_residual=residual[keep], owner=self.owner[keep],
                    road=self.road[keep], rows=torch.arange(n, device=xyz.device)[keep])

    @torch.no_grad()
    def split(self, indices, optimizer):
        """仅 Stage1 使用；保留 Adam 旧行，子点矩清零，不隐式复制训练历史。"""
        if not len(indices):
            return
        count = len(self.xyz)
        row_map = torch.cat((torch.arange(count, device=self.xyz.device), indices))
        params = {}
        for name in self.point_parameters:
            old = getattr(self, name)
            new = nn.Parameter(old[row_map].clone(), requires_grad=old.requires_grad)
            setattr(self, name, new)
            params[name] = (old, new)
        for name, buffer in list(self.named_buffers()):
            if buffer.ndim and len(buffer) == count:
                setattr(self, name, buffer[row_map].clone())
        axis = quaternion_matrix(self.quaternion[indices])[..., 0]
        shift = axis*self.scales[indices, :1]*.35
        self.xyz[indices] -= shift
        self.xyz[count:] += shift
        self.seed_xyz[indices] -= shift
        self.seed_xyz[count:] += shift
        shrink = torch.log(torch.tensor(.8, device=self.xyz.device))
        self.log_scales[indices] += shrink
        self.log_scales[count:] += shrink
        for group in optimizer.param_groups:
            for i, param in enumerate(group["params"]):
                match = next(((a, b) for a, b in params.values() if a is param), None)
                if match is None:
                    continue
                old, new = match
                group["params"][i] = new
                state = optimizer.state.pop(old, {})
                optimizer.state[new] = {key: torch.cat((value, torch.zeros_like(value[indices])), 0)
                                        if torch.is_tensor(value) and value.ndim and len(value) == count else value
                                        for key, value in state.items()}

    def regularizers(self, stage):
        source = self.anchor_xyz if stage == "joint" else self.seed_xyz
        geo = (self.xyz-source).square().mean()
        if stage == "joint":
            geo = geo+(self.log_scales-self.anchor_log_scales).square().mean()
            q = normalize(self.quaternion)
            qa = normalize(self.anchor_quaternion)
            geo = geo+(1-(q*qa).sum(-1).square()).mean()
        normals = quaternion_matrix(self.quaternion)[..., 2]
        agree = (normals*normals[self.neighbor]).sum(-1).square()
        normal = (1-agree).mean()
        thin = (self.scales[:, 2]/self.scales[:, :2].mean(-1).clamp_min(1e-5)).clamp_min(.2).square().mean()
        return geo, normal+.01*thin, self.view_coeff.square().mean()
