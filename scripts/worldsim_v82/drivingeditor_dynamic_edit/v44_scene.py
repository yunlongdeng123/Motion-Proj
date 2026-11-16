"""共享只读渲染核心；只依赖传入模型/归属/provider，不绑定scene、路径或帧数。

固定语义owner权威，-1为unknown、0为非actor背景。构造时冻结归属和retire集合；
provider可用性只依赖frame/camera。None默认回退native，不能随编辑请求切handoff。
测试注入NativeOps只验证CPU协议；真实native RGB/CUDA等价须由运行入口验收。
"""
from __future__ import annotations
from dataclasses import dataclass

GROUPS = ("target", "other_static", "other_dynamic", "road", "other_surface", "unassigned")
FEATURE_CHANNELS, MAX_DIAGNOSTIC_ACTORS = 24, 16
BRANCH_ALPHA_NAMES = ("nonfocus_static", "nonfocus_dynamic")


@dataclass(frozen=True)
class NativeOps:
    prepare_rasterization: object
    eval_sh: object
    merge_rows: object
    raster: object


@dataclass(frozen=True)
class RowProvider:
    rows: object                         # rows(frame_key, camera, pipe) -> native dict | None
    actor_owner: object
    kind: str = "surface"               # surface或road；owner与kind均不可按request变化
    retire_static: object = None
    retire_dynamic: object = None
    sh_degree: int = 0
    require_complete_coverage: bool = False
    state_objects: tuple = ()            # 只给持久模型/buffers；LoD临时缓存不放入
    assert_persistent_unchanged: object = None  # 可由clip入口/结束显式调用，不逐帧hash大资产


class V44Scene:
    def __init__(self, static, dynamic, environment, pipe, camera_provider, *,
                 source_owner_ids, semantic_owner_ids=None, row_providers=(),
                 static_gates=None, retire_static=None, retire_dynamic=None,
                 background=None, native_ops=None, require_complete_coverage=False,
                 alpha_tolerance=2e-5):
        import torch
        self._torch, self.static, self.dynamic, self.environment = torch, static, dynamic, environment
        self.pipe, self.camera_provider, self._ops = pipe, camera_provider, native_ops
        self._include_dynamic = dynamic is not None and (getattr(pipe, "including_dynamic", True) or getattr(pipe, "adding", False))
        self._require_coverage, self._alpha_tolerance = require_complete_coverage, alpha_tolerance
        if alpha_tolerance < 0 or not torch.isfinite(torch.tensor(alpha_tolerance)):
            raise ValueError("alpha_tolerance must be finite and nonnegative")
        self._device = static.get_xyz.device
        ns, nd = len(static.get_xyz), 0 if dynamic is None else len(dynamic.get_xyz)
        semantic = {} if semantic_owner_ids is None else semantic_owner_ids
        self._source_s = self._ids(source_owner_ids.get("static", -1), ns)
        self._source_d = self._ids(source_owner_ids.get("dynamic", -1), nd)
        self._owners_s = self._ids(semantic.get("static", self._source_s), ns)
        self._owners_d = self._ids(semantic.get("dynamic", self._source_d), nd)
        self._retire_s, self._retire_d = self._bools(retire_static, ns), self._bools(retire_dynamic, nd)
        self._background = static.get_xyz.new_zeros(3) if background is None else torch.as_tensor(background, device=self._device, dtype=static.get_xyz.dtype).clone()
        if self._background.shape != (3,) or not torch.isfinite(self._background).all():
            raise ValueError("background must be finite and have shape(3,)")
        self._gates = None if static_gates is None else torch.as_tensor(static_gates, device=self._device, dtype=static.get_xyz.dtype).clone()
        if self._gates is not None and (self._gates.shape != (ns,) or not torch.isfinite(self._gates).all() or ((self._gates < 0) | (self._gates > 1)).any()):
            raise ValueError("static_gates must be finite(N,) in[0,1]")
        providers, watched = [], [static, dynamic, environment, self]
        for provider in row_providers:
            if provider.kind not in ("surface", "road") or not callable(provider.rows):
                raise ValueError("Provider needs rows callable and kind=surface/road")
            frozen = RowProvider(provider.rows, self._ids(provider.actor_owner), provider.kind,
                self._bools(provider.retire_static, ns), self._bools(provider.retire_dynamic, nd),
                provider.sh_degree, provider.require_complete_coverage, tuple(provider.state_objects),
                getattr(provider, "assert_persistent_unchanged", None))
            providers.append(frozen)
            watched.extend((frozen, *provider.state_objects))
        self._providers, self._coverage_seen = tuple(providers), {}
        self._retire_counts = tuple((int(p.retire_static.sum()), int(p.retire_dynamic.sum())) for p in providers)
        self._watched = tuple(x for x in watched if x is not None)
        self._versions = self._snapshot()

    def _ids(self, value, count=None):
        t = self._torch.as_tensor(value, device=self._device)
        if count == 0 and t.numel() == 0:
            return t.to(self._torch.int64).clone()
        if t.dtype not in (self._torch.uint8, self._torch.int8, self._torch.int16, self._torch.int32, self._torch.int64) or t.ndim > 1:
            raise ValueError("Owners must be integer scalar or row array")
        if (t < -1).any():
            raise ValueError("Owners must be -1(unknown),0(background),or positive IDs")
        t = t.to(self._torch.int64).clone()
        if count is not None:
            t = t.expand(count).clone() if t.ndim == 0 else t
            if t.shape != (count,):
                raise ValueError("Owner row count does not match native rows")
        return t

    def _bools(self, value, count):
        t = self._torch.zeros(count, dtype=self._torch.bool, device=self._device) if value is None else self._torch.as_tensor(value, device=self._device)
        if t.dtype != self._torch.bool or t.shape != (count,):
            raise ValueError("Retire masks must be bool(native_row_count,)")
        return t.clone()

    def _snapshot(self):
        result = {}
        for index, obj in enumerate(self._watched):
            values = {"self": obj} if self._torch.is_tensor(obj) else dict(vars(obj)) if hasattr(obj, "__dict__") else {}
            for method in ("named_parameters", "named_buffers"):
                if callable(getattr(obj, method, None)):
                    values.update(dict(getattr(obj, method)()))
            for key, tensor in values.items():
                if self._torch.is_tensor(tensor):
                    result[index, key] = (id(tensor), tensor._version, tensor.data_ptr(), tuple(tensor.shape))
        return result

    def assert_unchanged(self):
        if self._snapshot() != self._versions:
            raise RuntimeError("Native/provider parameters, row order or fixed owners were mutated")

    def assert_persistent_unchanged(self):
        """调用方可在clip前后使用；昂贵资产校验不进入逐帧render。"""
        self.assert_unchanged()
        for provider in self._providers:
            if callable(provider.assert_persistent_unchanged):
                provider.assert_persistent_unchanged()

    def _bindings(self):
        if self._ops is None:
            from gaussian_renderer import prepare_rasterization
            from utils.sh_utils import eval_sh
            from utils.v4_road_render import merge_rows, raster
            self._ops = NativeOps(prepare_rasterization, eval_sh, merge_rows, raster)
        return self._ops

    @staticmethod
    def _query_ids(values):
        result = []
        for value in values:
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError("Query IDs must be positive integers; -1 remains unknown")
            result.append(int(value))
        return tuple(result)

    def _contains(self, owners, values):
        return self._torch.isin(owners, owners.new_tensor(values)) if values else self._torch.zeros_like(owners, dtype=self._torch.bool)

    def _native_colors(self, rows, degree, camera):
        t, ops = self._torch, self._bindings()
        result = dict(rows)
        if result["colors_precomp"] is None:
            if result["shs"] is None:
                raise ValueError("Rows need native SH or precomputed RGB")
            direction = t.nn.functional.normalize(result["means3D"] - camera.camera_center, dim=1)
            result["colors_precomp"] = (ops.eval_sh(degree, result["shs"].transpose(1, 2), direction) + .5).clamp_min(0)
        result.update(shs=None, means2D=t.zeros_like(result["means3D"]))
        return result

    def _diagnostic_rows(self, rows, owners, kind, focus, diagnostic_ids, branch_alpha):
        t = self._torch
        if len(rows["means3D"]) != len(owners) or rows["mask"].shape != owners.shape or rows["mask"].dtype != t.bool:
            raise ValueError("Native row order/mask differs from fixed ownership")
        if rows["features"].shape[0] != len(owners) or rows["features"].shape[1] < 2:
            raise ValueError("Rows require native depth/alpha as first two feature columns")
        index = {"static": 1, "dynamic": 2, "road": 3, "surface": 4}[kind]
        groups = t.full_like(owners, index)
        groups[owners == -1] = 5
        groups[self._contains(owners, focus) & (owners != -1)] = 0
        weights = t.nn.functional.one_hot(groups, len(GROUPS)).to(rows["features"].dtype) * rows["features"][:, 1:2]
        features = rows["features"].new_zeros((len(owners), FEATURE_CHANNELS))
        features[:, :2], features[:, 2:8] = rows["features"][:, :2], weights
        if diagnostic_ids:
            tags = owners[:, None] == owners.new_tensor(diagnostic_ids)[None, :]
            features[:, 8:8+len(diagnostic_ids)] = tags.to(features.dtype) * rows["features"][:, 1:2]
        if branch_alpha and kind in ("static", "dynamic"):
            nonfocus = ~self._contains(owners, focus)  # unknown也保留其实际native分支归因
            features[:, 22 if kind == "static" else 23] = nonfocus.to(features.dtype) * rows["features"][:, 1]
        return dict(rows, features=features)

    def _rows(self, frame_key, focus, diagnostic_ids, branch_alpha):
        camera, ops = self.camera_provider(frame_key), self._bindings()
        rs, rd = self._retire_s.clone(), self._retire_d.clone()
        supplied, coverage = [], []
        for index, provider in enumerate(self._providers):
            rows = provider.rows(frame_key, camera, self.pipe)  # 绝不传edit/focus
            available = rows is not None
            if not available and (self._require_coverage or provider.require_complete_coverage):
                raise ValueError(f"Provider{index} coverage gap at frame {frame_key!r}")
            ns, nd = self._retire_counts[index]
            coverage.append(dict(provider=index, kind=provider.kind, available=available,
                rows=0 if rows is None else len(rows["means3D"]), native_fallback=not available and bool(ns or nd),
                retire_static_rows=ns if available else 0, retire_dynamic_rows=nd if available else 0))
            if available:
                rs, rd = rs | provider.retire_static, rd | provider.retire_dynamic
                supplied.append((rows, self._ids(provider.actor_owner, len(rows["means3D"])), provider.kind, provider.sh_degree))
        signature = tuple((x["available"], x["rows"]) for x in coverage)
        if frame_key in self._coverage_seen and self._coverage_seen[frame_key] != signature:
            raise RuntimeError("Provider availability/row count changed for the same frame")
        self._coverage_seen[frame_key] = signature
        branches = []
        for model, owners, retire, dynamic in ((self.static, self._owners_s, rs, False), (self.dynamic, self._owners_d, rd, True)):
            if model is None or (dynamic and not self._include_dynamic):
                continue
            rows = self._native_colors(ops.prepare_rasterization(camera, model, self.pipe, dynamic=dynamic), model.active_sh_degree, camera)
            rows["mask"] = rows["mask"] & ~retire
            if not dynamic and self._gates is not None:
                rows["opacities"] = rows["opacities"] * self._gates[:, None]
            branches.append((self._diagnostic_rows(rows, owners, "dynamic" if dynamic else "static", focus, diagnostic_ids, branch_alpha), owners))
        for rows, owners, kind, degree in supplied:
            branches.append((self._diagnostic_rows(self._native_colors(rows, degree, camera), owners, kind, focus, diagnostic_ids, branch_alpha), owners))
        values = branches[0][0]
        for rows, _ in branches[1:]:
            values = ops.merge_rows(values, rows)
        return values, self._torch.cat([owners for _, owners in branches]), camera, coverage

    def render(self, frame_key, delete_actor_ids=(), focus_actor_ids=(), diagnostic_actor_ids=(), diagnostic_branch_alpha=False):
        deleted, focus = self._query_ids(delete_actor_ids), self._query_ids(focus_actor_ids)
        diagnostic_ids = self._query_ids(diagnostic_actor_ids)
        if type(diagnostic_branch_alpha) is not bool:
            raise ValueError("diagnostic_branch_alpha must be bool")
        capacity = MAX_DIAGNOSTIC_ACTORS - (2 if diagnostic_branch_alpha else 0)
        if len(diagnostic_ids) > capacity:
            raise ValueError(f"At most {capacity} diagnostic_actor_ids fit the 24-feature ABI in this mode")
        if len(set(diagnostic_ids)) != len(diagnostic_ids):
            raise ValueError("diagnostic_actor_ids cannot contain duplicate IDs")
        self.assert_unchanged()
        try:
            with self._torch.no_grad():
                values, owners, camera, coverage = self._rows(frame_key, focus, diagnostic_ids, diagnostic_branch_alpha)
                values = dict(values, mask=values["mask"] & ~self._contains(owners, deleted))
                background = self._background if self.environment is not None else self._torch.zeros_like(self._background)
                out = dict(self._bindings().raster(camera, self.pipe, 0, values, background))  # 只一次raster
                out["raw_rgb"] = out["rgb"]
                out["env_map"] = self._torch.zeros_like(out["rgb"])
                if self.environment is not None:
                    env = self.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
                    out["env_map"] = (1 - out["alpha"]) * env
                    out["rgb"] = out["rgb"] + out["env_map"]
                if out["extra"].shape[0] != FEATURE_CHANNELS - 2:
                    raise ValueError("Raster did not preserve the 24-feature ABI")
                group_alpha = out["extra"][:len(GROUPS)]
                error = (group_alpha.sum(0, keepdim=True) - out["alpha"]).abs().max()
                if not self._torch.allclose(group_alpha.sum(0, keepdim=True), out["alpha"], atol=self._alpha_tolerance, rtol=self._alpha_tolerance):
                    raise RuntimeError("Diagnostic sumalpha conservation failed")
                out.update(diagnostics={name: group_alpha[i:i+1] for i, name in enumerate(GROUPS)},
                    owner_alpha_T=out["extra"][len(GROUPS):len(GROUPS)+len(diagnostic_ids)],
                    owner_alpha_actor_ids=diagnostic_ids,
                    branch_alpha_T=out["extra"][20:22] if diagnostic_branch_alpha else out["extra"][:0],
                    branch_alpha_names=BRANCH_ALPHA_NAMES if diagnostic_branch_alpha else (),
                    diagnostic_alpha_error_max=float(error), coverage=coverage, row_owners=owners,
                    delete_actor_ids=deleted, focus_actor_ids=focus)
                return out
        finally:
            self.assert_unchanged()
