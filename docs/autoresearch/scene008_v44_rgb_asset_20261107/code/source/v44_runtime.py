"""V44：冻结 RCR 原生模型/道路，显式接入 C2 或同 S/F 的 LOD 资产。

模块导入仅使用标准库。构造时先验收 CPU 输入合同，再延迟导入 CUDA 执行源码。
不加载旧场景 Runtime、final_runtime、M2 资产或颜色校准求解器。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


GROUPS = ("asset", "a37", "road", "native_static")
RAW_IDS = tuple(range(89, 138))
FEATURE_COUNT = 24
GROUP_ALPHA_SLICE = slice(2, 6)
GROUP_RGB_SLICE = slice(6, 18)
C2_SHA256 = "0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d"
POSTERIOR_SHA256 = "b98a0d932730fbf1462e555eb7d5eb931030b51fe68848990f073b9752879e06"
LOD_SOURCE_SHA256 = "fb3c746a04bf17b1f84b127e44769b4ff294567e98aec744131ddc3f3c4d76ca"
SH_ASSET_SOURCE_SHA256 = "ff9ee4b0f6a0bc90fccedfd4be17404d2df129072d6da1562b211a8aa6469bd4"
RCR_FINE_BINDING_SHA256 = "6a955f0fa78646264366d985e45b481b7bf8698998764927bf906e5730943e00"
RCR_FINE_COEFFICIENTS_SHA256 = "131bc37ed2711701ef5526caed8021a5971fa10a56ae670d3cd6a82dffc250e3"
RCR_SOURCE_M2_SHA256 = "d84133bb1a9767e00301857cc1d8f293f506faaafc58da598c8d0d7c0c8ae0e6"
REQUIRED_PATHS = ("native_parent", "road_run", "road_checkpoint", "fine", "asset",
                  "posterior", "asset_source", "lod_source", "observations",
                  "fixed_reference", "engine")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def tensor_hash(values):
    """逐名记录 dtype/shape/原始字节，不假定资产 tensor 数量。"""
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        if value is None:
            digest.update((key + ":None").encode())
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def module_tensors(module):
    return dict((*module.named_parameters(), *module.named_buffers()))


def select_rows(rows, keep):
    return {key: None if value is None else value[keep].contiguous()
            for key, value in rows.items()}


def world_basis(xy, knots):
    """在原有世界局部坐标节点上求值已保存的双线性场。"""
    import numpy as np
    xy = np.asarray(xy)
    low = np.asarray(knots["low_m"])
    high = np.asarray(knots["high_m"])
    grid = np.asarray(knots["grid"], dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 2 or tuple(grid) != (11, 9):
        raise ValueError("Expected finite world-local xy[N,2] and the saved 11x9 field")
    if low.shape != (2,) or high.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Invalid world field coordinates")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not (high > low).all():
        raise ValueError("Invalid saved world field bounds")
    scaled = np.clip((xy-low)/np.maximum(high-low, 1e-8)*(grid-1), 0, grid-1)
    index = np.minimum(scaled.astype(int), grid-2)
    fraction = scaled-index
    result = np.zeros((len(xy), int(grid.prod())), np.float32)
    for ox in (0, 1):
        for oy in (0, 1):
            weight = ((fraction[:, 0] if ox else 1-fraction[:, 0]) *
                      (fraction[:, 1] if oy else 1-fraction[:, 1]))
            result[np.arange(len(xy)), (index[:, 0]+ox)*grid[1]+index[:, 1]+oy] = weight
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    contract_path: Path
    paths: dict[str, str]
    sampler: str = "surface_lod"
    transfer_fine: bool = True

    def __post_init__(self):
        if self.sampler not in ("canonical", "surface_lod"):
            raise ValueError("sampler must be canonical or surface_lod")
        missing = [name for name in REQUIRED_PATHS if not self.paths.get(name)]
        if missing:
            raise ValueError(f"Missing explicit V44 input paths: {missing}")

    @classmethod
    def from_json(cls, contract_path, sampler="surface_lod", transfer_fine=True):
        path = Path(contract_path).resolve()
        contract = read_json(path)
        return cls(path, dict(contract["paths"]), sampler, bool(transfer_fine))

    def path(self, name):
        return Path(self.paths[name]).resolve()


def _validate_cpu_contract(config):
    """加载同目录树的 CPU 检查器，不导入执行依赖。"""
    checker = Path(__file__).resolve().parent.parent / "contract" / "v44_contract.py"
    if not checker.is_file():
        raise FileNotFoundError(f"Required V44 CPU input validator is missing: {checker}")
    spec = importlib.util.spec_from_file_location("_v44_input_contract", checker)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V44 input validator: {checker}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 在导入期间可能访问 sys.modules。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    receipt = module.validate_contract(config.contract_path)
    if receipt.get("status") != "PASS_V44_CPU_INPUT_CONTRACT":
        raise ValueError("V44 CPU input contract failed: " + json.dumps(receipt, ensure_ascii=False))
    declared = read_json(config.contract_path)["paths"]
    if dict(declared) != config.paths:
        raise ValueError("Runtime paths differ from the validated V44 input contract")
    return receipt, checker


class V44Runtime:
    """冻结背景与替换 a6 共用一次原生深度排序。

    scene() 返回 {rows, cam, env, bg, meta}。render_scene()/decompose()
    额外返回 group_rgb_T、group_alpha_T、component_alpha_T、env_rgb_T
    及颜色重建误差。
    """

    def __init__(self, config_or_contract_path):
        self.config = (config_or_contract_path if isinstance(config_or_contract_path, RuntimeConfig)
                       else RuntimeConfig.from_json(config_or_contract_path))
        self.cpu_contract, checker = _validate_cpu_contract(self.config)
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        if not torch.cuda.is_available():
            raise RuntimeError("V44 CUDA execution is pending: no CUDA device is available")
        torch.set_num_threads(1)
        sys.dont_write_bytecode = True
        self.parent = self.config.path("native_parent")
        self.road_run = self.config.path("road_run")
        self.road_checkpoint = self.config.path("road_checkpoint")
        self.asset_path = self.config.path("asset")
        self.posterior_path = self.config.path("posterior")
        self.observations = self.config.path("observations")
        self.fixed_reference = self.config.path("fixed_reference")
        asset_source = self.config.path("asset_source")
        lod_source = self.config.path("lod_source")
        if lod_source.is_file():
            lod_source = lod_source.parent
        expected_files = {self.asset_path: C2_SHA256, self.posterior_path: POSTERIOR_SHA256,
                          lod_source / "surface_lod_asset.py": LOD_SOURCE_SHA256,
                          asset_source / "sh_asset.py": SH_ASSET_SOURCE_SHA256}
        for path, expected in expected_files.items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen C2/S/F input differs: {path}")
        binding_path = self.road_run / "replay_binding.json"
        replay = read_json(binding_path)
        self.iteration = int(replay["native_iteration"])
        if Path(replay["native_parent"]).resolve() != self.parent or self.iteration != 10000:
            raise ValueError("V44 requires the explicit RCR native10K replay parent")
        metadata_paths = [self.parent/name for name in
                          ("effective_config.json", "cameras.json", "actual_scene_binding.json")]
        for path in metadata_paths:
            if replay["parent_metadata_sha256"].get(str(path)) != sha256(path):
                raise ValueError(f"Frozen replay parent metadata differs: {path}")
        forbidden = ("editor_common", "native_binding", "repair37", "r4_runtime", "road_runtime",
                     "scene_runtime", "final_runtime", "sh_asset", "surface_lod_asset")
        loaded = [name for name in forbidden if name in sys.modules]
        if loaded:
            raise RuntimeError(f"Use an isolated V44 worker; scene modules already imported: {loaded}")
        source = self.road_run / "source"
        sys.path[:0] = [str(source), str(asset_source), str(lod_source), str(self.config.path("engine"))]
        repair37 = importlib.import_module("repair37")
        native_binding = importlib.import_module("native_binding")
        common = importlib.import_module("editor_common")
        sh_asset = importlib.import_module("sh_asset")
        eval_sh = importlib.import_module("utils.sh_utils").eval_sh
        for name in ("repair37", "native_binding", "editor_common", "road_runtime", "r4_runtime"):
            if Path(sys.modules[name].__file__).resolve().parent != source:
                raise ValueError(f"Imported a different fresh-road source: {name}")
        if Path(sh_asset.__file__).resolve() != asset_source / "sh_asset.py":
            raise ValueError("Imported a different SH asset implementation")
        if Path(native_binding.PARENT).resolve() != self.parent:
            raise ValueError("Road module resolved a different native parent")
        self.prepare, self.road_kwargs = common.prepare_rasterization, common.road_kwargs
        self.merge, self.raster, self.eval_sh = common.merge_rows, common.raster, eval_sh
        with torch.no_grad():
            self.exp = repair37.Experiment()
            self.n, self.pipe = self.exp.n, self.exp.n.args
            self.scale = float(self.exp.meta["meter_to_model"])
            expected = {key: tuple(value) for key, value in replay["parent_identity"].items()}
            if self.n.before != expected or self.n.checkpoint_iteration != self.iteration:
                raise ValueError("Loaded native differs from this fresh-road parent")
            if self.road_checkpoint.parent.parent != self.road_run:
                raise ValueError("Road checkpoint is outside the explicit fresh-road run")
            state = torch.load(str(self.road_checkpoint), map_location="cpu", weights_only=False)
            if state["step"] != 1000 or state["parent_identity"] != self.n.before:
                raise ValueError("Fresh-road checkpoint step/native identity differs")
            method = state["contract"]
            if (Path(method["native_parent"]).resolve() != self.parent
                    or int(method["native_iteration"]) != self.iteration
                    or not method["only_static_opacity_gate_trainable"]
                    or not method["road_colour_geometry_opacity_fixed"]):
                raise ValueError("Unexpected fresh-road method contract")
            contributors = self.road_run / "inputs" / "contributors.npz"
            with np.load(contributors, allow_pickle=False) as data:
                ids = data["ids"].copy()
            checkpoint_ids = state["repair"]["ids"].detach().cpu().numpy()
            if not np.array_equal(ids, checkpoint_ids):
                raise ValueError("Gates address a different fresh-native contributor list")
            if (ids.ndim != 1 or len(np.unique(ids)) != len(ids)
                    or (len(ids) and (ids.min() < 0 or ids.max() >= len(self.n.static.get_xyz)))):
                raise ValueError("Invalid fresh-native gate indices")
            self.exp.repair.load_state_dict(state["repair"], strict=True)
            self.exp.repair.requires_grad_(False)
            self.n.env.requires_grad_(False)
            self.base_asset = sh_asset.SHAsset.from_geometry(self.asset_path).cuda().requires_grad_(False)
            if (len(self.base_asset.xyz) != 45314
                    or int((self.base_asset.representation_kind == 1).sum()) != 42536
                    or not bool((self.base_asset.coefficients[:, 1:] == 0).all())):
                raise ValueError("V44 expects the frozen 45314-row effective-SH0 C2 asset")
            if self.config.sampler == "surface_lod":
                lod_module = importlib.import_module("surface_lod_asset")
                if Path(lod_module.__file__).resolve() != lod_source / "surface_lod_asset.py":
                    raise ValueError("Imported a different SurfaceLOD wrapper")
                self.asset = lod_module.SurfaceLODAsset(self.base_asset, self.posterior_path)
            else:
                self.asset = self.base_asset
            self.baseline_colours = self.exp.repair.road.colors().detach().clone()
            self.road_colours = self.baseline_colours
        self.parent_identity = dict(self.n.before)
        self.repair_hash = tensor_hash(self.exp.repair.state_dict())
        self.baseline_hash = tensor_hash({"colours": self.baseline_colours})
        self.asset_tensor_hash = tensor_hash(module_tensors(self.base_asset))
        self.asset_tensor_schema = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
                                    for key, value in module_tensors(self.base_asset).items()}
        self._data, self._asset_hashes = {}, {}
        self.render_calls = 0
        self.fine_binding = None
        self.file_identities, self.content_hashes = {}, {}
        paths = [self.config.contract_path, checker, binding_path, self.road_checkpoint, contributors,
                 self.asset_path, self.posterior_path, self.fixed_reference,
                 self.road_run / "inputs" / "road_reference.npz", *metadata_paths,
                 asset_source / "sh_asset.py", lod_source / "surface_lod_asset.py", Path(__file__),
                 Path(self.raster.__code__.co_filename)]
        paths += [source/name for name in
                  ("editor_common.py", "native_binding.py", "repair37.py", "road_runtime.py", "r4_runtime.py")]
        for path in paths:
            self._track_file(path)
        with np.load(self.fixed_reference, allow_pickle=False) as fixed:
            if fixed["layer_names"].tolist()[3] != "ground_beneath_a37_proxy":
                raise ValueError("Original fixed delete-road supervision changed")
            np.testing.assert_array_equal(fixed["raw_ids"], np.arange(89, 138))
            self.fixed_ground = fixed["depth_m"][:, 3] > 0
            self.fixed_cameras = fixed["model_to_camera"].copy()
            self.fixed_times = fixed["model_time"].copy()
        matrix = np.asarray(self.exp.meta["world_to_model"]) @ np.asarray(self.exp.meta["coordinate_anchor"])
        inverse = np.linalg.inv(matrix)
        xyz = self.exp.repair.road.xyz.detach().cpu().numpy()
        self.road_xy = (xyz @ inverse[:3, :3].T+inverse[:3, 3])[:, :2]
        self.binding = dict(version="v44", native_parent=str(self.parent), native_iteration=self.iteration,
            parent_identity=self.parent_identity, road_run=str(self.road_run), road_iteration=1000,
            road_checkpoint_sha256=sha256(self.road_checkpoint), contributors_sha256=sha256(contributors),
            repair_state_sha256=self.repair_hash, road_colour_baseline_sha256=self.baseline_hash,
            asset_sha256=C2_SHA256, posterior_sha256=POSTERIOR_SHA256,
            asset_canonical_rows=45314, asset_effective_material="frozen SH0 within SHAsset SH3 storage",
            sampler=self.config.sampler, sampler_source_sha256=LOD_SOURCE_SHA256,
            persistent_SF_hash=getattr(self.asset, "persistent_SF_hash", None),
            native_metadata_sha256={str(path): sha256(path) for path in metadata_paths},
            road_source_sha256={path.name: sha256(path) for path in paths if path.parent == source},
            native_degrees=dict(static=self.n.static.active_sh_degree, a37=self.n.dynamic.active_sh_degree),
            groups=list(GROUPS), native_a6_rows_input=0,
            native_static_opacity="own_native_opacity * matching fresh-road1K gate",
            edit_operator="physical a37 removal only; shared frozen a6/road/static/env/camera",
            raster="original native GaussianRasterizer; one shared depth sort",
            transfer_policy="frozen_worldspace_residual_transfer" if self.config.transfer_fine else "fresh_road_baseline",
            reestimated_calibration=False, quality_accepted=False)
        self.base_scene_binding_sha256 = stable_hash(self.binding)
        if self.config.transfer_fine:
            self._load_fine_transfer(self.config.path("fine"))
            self.binding["fine_transfer_source"] = {key: self.fine_binding[key] for key in
                ("source_binding_sha256", "coefficients_sha256", "source_asset_sha256", "policy")}
        self.binding_sha256 = stable_hash(self.binding)
        if self.fine_binding:
            self.fine_binding["new_scene_binding_sha256"] = self.binding_sha256
        self._versions = self.tensor_versions()
        self._colour_versions = self.colour_versions()
        self.road_colour_hash = tensor_hash({"colours": self.road_colours})
        self.assert_unchanged(full=True)

    def _track_file(self, path):
        key = str(Path(path).resolve())
        value = file_identity(path)
        if key in self.file_identities and value != self.file_identities[key]:
            raise RuntimeError(f"Frozen input changed: {path}")
        if key not in self.file_identities:
            self.file_identities[key] = value
            self.content_hashes[key] = sha256(path)

    def _load_fine_transfer(self, folder):
        np, torch = self.np, self.torch
        metadata = read_json(folder / "scene_binding.json")
        source = metadata["binding"]
        if (metadata["stage"] != "fine11x9" or metadata["preflight_only"]
                or stable_hash(source) != metadata["binding_sha256"]
                or metadata["binding_sha256"] != RCR_FINE_BINDING_SHA256
                or sha256(folder / "coefficients.npz") != RCR_FINE_COEFFICIENTS_SHA256):
            raise ValueError("Fine transfer must come from the intact recorded RCR fine11x9 scene")
        # M2 仅作为来源字段保留，不加载该资产，也不把旧绑定冒充 C2 校准。
        for key in ("native_parent", "native_iteration", "parent_identity", "road_run",
                    "road_iteration", "road_checkpoint_sha256", "contributors_sha256",
                    "repair_state_sha256", "road_colour_baseline_sha256",
                    "native_metadata_sha256", "road_source_sha256", "native_degrees"):
            if stable_hash(source[key]) != stable_hash(self.binding[key]):
                raise ValueError(f"Saved fine field has a different native/road dependency: {key}")
        if source["M2_sha256"] != RCR_SOURCE_M2_SHA256:
            raise ValueError("Expected the explicit recorded source-M2 calibration binding")
        with np.load(folder / "coefficients.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["colour_baseline"], self.baseline_colours.cpu().numpy()):
                raise ValueError("Frozen fine transfer baseline does not match this exact fresh road")
            coefficients = saved["world"].copy()
        if (coefficients.shape != (99, 3) or not np.isfinite(coefficients).all()
                or np.max(np.abs(coefficients)) > .09000001):
            raise ValueError("Unexpected frozen fine11x9 coefficients")
        knots = read_json(folder / "design.json")["knots"]
        np.testing.assert_allclose(self.road_xy.min(0), knots["low_m"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(self.road_xy.max(0), knots["high_m"], atol=1e-6, rtol=0)
        basis = self.baseline_colours.new_tensor(world_basis(self.road_xy, knots))
        if not torch.allclose(basis.sum(1), torch.ones_like(basis[:, 0]), atol=2e-6, rtol=0):
            raise ValueError("World field basis does not partition unity")
        with torch.no_grad():
            self.road_colours = (self.baseline_colours+basis@basis.new_tensor(coefficients)).clamp(0, 1)
        for name in ("scene_binding.json", "coefficients.npz", "design.json"):
            self._track_file(folder/name)
        self.fine_binding = dict(path=str(folder), policy="frozen_worldspace_residual_transfer",
            source_binding_sha256=metadata["binding_sha256"], source_asset_sha256=source["M2_sha256"],
            new_asset_sha256=C2_SHA256, coefficients_sha256=sha256(folder/"coefficients.npz"),
            source_scene_binding=source, new_base_scene_binding_sha256=self.base_scene_binding_sha256,
            scene_binding_changed=True, knots=knots, coefficient_updates=0, robust_linear_solves=0,
            calibrated_on_new_C2_or_LOD=False, same_colour_table_not_same_occlusion=True,
            GPU_occlusion_and_RGB_comparison_required=True, quality_accepted=False)

    def tensor_versions(self):
        torch = self.torch
        result = {}
        for name, model in (("static", self.n.static), ("dynamic", self.n.dynamic)):
            for key, value in vars(model).items():
                if torch.is_tensor(value):
                    result[name, key] = (id(value), value._version)
        for name, module in (("road_repair", self.exp.repair), ("environment", self.n.env),
                             ("canonical_asset", self.base_asset)):
            for key, value in module_tensors(module).items():
                result[name, key] = (id(value), value._version)
        return result

    def colour_versions(self):
        return {name: (id(value), value._version) for name, value in
                (("baseline", self.baseline_colours), ("rendered", self.road_colours))}

    def assert_unchanged(self, full=False):
        if not self.n.unchanged() or self.n.before != self.parent_identity:
            raise RuntimeError("Frozen native checkpoint changed")
        if self.tensor_versions() != self._versions or self.colour_versions() != self._colour_versions:
            raise RuntimeError("Frozen tensor identity or version changed")
        if not all(file_identity(path) == value for path, value in self.file_identities.items()):
            raise RuntimeError("Frozen input file identity changed")
        if full:
            if not all(sha256(path) == value for path, value in self.content_hashes.items()):
                raise RuntimeError("Frozen input file content changed")
            if tensor_hash(self.exp.repair.state_dict()) != self.repair_hash:
                raise RuntimeError("Frozen road or gates changed")
            if tensor_hash(module_tensors(self.base_asset)) != self.asset_tensor_hash:
                raise RuntimeError("Frozen canonical asset tensors changed")
            if tensor_hash({"colours": self.baseline_colours}) != self.baseline_hash:
                raise RuntimeError("Frozen baseline road colours changed")
            if tensor_hash({"colours": self.road_colours}) != self.road_colour_hash:
                raise RuntimeError("Frozen transferred road colour table changed")
            self.base_asset.assert_geometry_unchanged()
            if self.config.sampler == "surface_lod":
                self.asset.assert_persistent_unchanged()
        return True

    def raw(self, raw):
        np, torch = self.np, self.torch
        if raw not in RAW_IDS:
            raise KeyError(raw)
        if raw not in self._data:
            obs_path = self.observations/f"{raw:03d}.npz"
            self._track_file(obs_path)
            with np.load(obs_path, allow_pickle=False) as saved:
                obs = {key: saved[key].copy() for key in saved.files}
            camera = self.exp.camera(raw)
            if (camera.image_width, camera.image_height) != (480, 320):
                raise ValueError("V44 native street replay requires the original 480x320 camera")
            np.testing.assert_allclose(obs["model_to_camera"].T,
                camera.world_view_transform.cpu().numpy(), atol=1e-6, rtol=0)
            np.testing.assert_allclose(self.fixed_cameras[raw-89], obs["model_to_camera"], atol=1e-6, rtol=0)
            if (abs(float(obs["model_time"])-camera.timestamp) >= 1e-6
                    or abs(float(self.fixed_times[raw-89])-camera.timestamp) >= 1e-6):
                raise ValueError("Observation and frozen native timestamps differ")
            record = next(row for row in self.n.records if row["img_name"] == f"0008{raw:03d}")
            np.testing.assert_allclose(obs["K"], [[record["fx"]/4, 0, record["cx"]/4],
                [0, record["fy"]/4, record["cy"]/4], [0, 0, 1]], atol=1e-6, rtol=0)
            with torch.no_grad():
                cv = lambda value: torch.as_tensor(value, device=self.n.bg.device)
                rgb = cv(obs["rgb"]).float().permute(2, 0, 1).contiguous()
                if float(rgb.max()) > 1:
                    rgb = rgb/255.
                env = self.n.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1).detach()
                self._data[raw] = dict(cam=camera, pose=cv(obs["model_from_local"][0]).float(),
                    rgb=rgb, obs=obs, env=env, a6_visible=cv(obs["visible"][0]).bool(),
                    a6_depth_valid=cv(obs["depth_valid"][0]).bool(), depth=cv(obs["depth_m"][0]))
        return self._data[raw]

    def _native_rows(self, camera, dynamic=False):
        torch = self.torch
        model = self.n.dynamic if dynamic else self.n.static
        rows = self.prepare(camera, model, self.pipe, dynamic=dynamic)
        if rows["colors_precomp"] is None:
            direction = rows["means3D"]-camera.camera_center
            direction = direction/direction.norm(dim=-1, keepdim=True)
            colour = (self.eval_sh(model.active_sh_degree, rows["shs"].transpose(1, 2), direction)+.5).clamp_min(0)
        else:
            colour = rows["colors_precomp"]
        return dict(rows, means2D=torch.zeros_like(rows["means3D"]), shs=None, colors_precomp=colour)

    @staticmethod
    def _tag(rows, group):
        index = GROUPS.index(group)
        feature = rows["features"].new_zeros((len(rows["features"]), FEATURE_COUNT))
        feature[:, :2] = rows["features"][:, :2]
        feature[:, 2+index] = 1
        feature[:, 6+3*index:9+3*index] = rows["colors_precomp"]
        return dict(rows, features=feature)

    def scene(self, raw, delete=False):
        torch = self.torch
        self.assert_unchanged()
        with torch.no_grad():
            data = self.raw(raw)
            camera = data["cam"]
            static = self._native_rows(camera)
            ids = self.exp.repair.ids
            static["opacities"] = static["opacities"].index_copy(
                0, ids, static["opacities"][ids]*self.exp.repair.gate_raw.sigmoid()[:, None])
            dynamic = self._native_rows(camera, dynamic=True)
            if len(self.n.keep_a6) != len(dynamic["means3D"]):
                raise ValueError("Native actor identity map does not match the dynamic row table")
            a37 = select_rows(dynamic, ~self.n.keep_a6)
            if delete:
                a37 = select_rows(a37, torch.zeros_like(a37["mask"]))
            asset_rows = self.asset.rows(data["pose"], camera, self.pipe, channels=16, precomputed=True)
            if len(asset_rows["means3D"]) != len(self.asset.xyz) or len(self.asset.xyz) == 0:
                raise ValueError("Replacement rows do not cover this complete asset sampler")
            asset_hash = tensor_hash(asset_rows)
            if raw in self._asset_hashes and self._asset_hashes[raw] != asset_hash:
                raise RuntimeError("Asset rows differ between calls or keep/delete edit states")
            self._asset_hashes[raw] = asset_hash
            road = self.road_kwargs(camera, self.exp.repair.road, self.pipe,
                                    sh_channels=1, precomputed_color=True)
            road["colors_precomp"] = self.road_colours
            joint = self.merge(self.merge(self._tag(static, "native_static"), self._tag(asset_rows, "asset")),
                               self._tag(a37, "a37"))
            joint = self.merge(joint, self._tag(road, "road"))
            if joint["shs"] is not None or joint["features"].shape[1] != FEATURE_COUNT:
                raise RuntimeError("Unexpected V44 precomputed-colour feature ABI")
            if bool((joint["features"][:, 18:] != 0).any()):
                raise RuntimeError("Unused native feature channels are nonzero")
            meta = dict(raw=raw, delete_a37=bool(delete), native_a6_rows_input=0,
                native_a37_packed_rows=len(a37["mask"]), native_asset_row_hash=asset_hash,
                asset_rows=len(asset_rows["means3D"]), canonical_asset_rows=len(self.base_asset.xyz),
                asset_sampler=self.config.sampler, native_static_rows=len(static["mask"]),
                road_rows=len(road["mask"]), groups=list(GROUPS), keep_delete_only_a37=True,
                same_frozen_material_surface_background=True,
                LOD=deepcopy(getattr(self.asset, "last_lod_record", None)),
                fine_transfer_recalibrated=False, scene_binding_sha256=self.binding_sha256)
            return dict(rows=joint, cam=camera, env=data["env"], bg=self.n.bg, meta=meta)

    def render(self, raw, scene):
        torch = self.torch
        with torch.no_grad():
            out = self.raster(scene["cam"], self.pipe, 0, scene["rows"], scene["bg"])
            self.render_calls += 1
            out["rgb_premultiplied"] = out["rgb"]
            out["group_alpha_T"] = out["extra"][:4]
            out["component_alpha_T"] = out["group_alpha_T"]
            out["group_rgb_T"] = out["extra"][4:16].reshape(4, 3, *out["alpha"].shape[-2:])
            out["env_rgb_T"] = (1-out["alpha"])*scene["env"]
            out["constant_background_rgb_T"] = (1-out["alpha"])*scene["bg"][:, None, None]
            out["rgb"] = out["rgb_premultiplied"]+out["env_rgb_T"]
            reconstruction = out["group_rgb_T"].sum(0)+out["constant_background_rgb_T"]+out["env_rgb_T"]
            group_error = float((out["group_alpha_T"].sum(0)-out["alpha"][0]).abs().max())
            colour_error = float((reconstruction-out["rgb"]).abs().max())
            if group_error >= 2e-5 or colour_error >= 5e-5:
                raise RuntimeError(f"Single-sort group conservation failed: alpha={group_error}, RGB={colour_error}")
            if not all(bool(torch.isfinite(out[key]).all()) for key in
                       ("rgb", "alpha", "depth_sum", "extra", "group_rgb_T")):
                raise RuntimeError("Nonfinite native V44 render")
            out["group_conservation_maxabs"] = group_error
            out["color_reconstruction_maxabs"] = colour_error
            out["colour_reconstruction_maxabs"] = colour_error
            out["meta"] = dict(scene["meta"], attribution="one-forward shared-sort RGB/alpha*T features")
            return out

    def render_scene(self, raw, delete=False):
        return self.render(raw, self.scene(raw, delete=delete))

    def decompose(self, raw, delete=False):
        return self.render_scene(raw, delete=delete)

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(status="PASS_V44_FROZEN_RUNTIME_ENGINEERING", binding=self.binding,
            binding_sha256=self.binding_sha256, fine_binding=self.fine_binding,
            cpu_input_contract=self.cpu_contract, asset_tensor_schema=self.asset_tensor_schema,
            asset_all_named_tensors_bitwise_hash_unchanged=True,
            frozen_native_road_gates_environment_unchanged=True,
            persistent_SF_unchanged=True, canonical_asset_sha256=C2_SHA256,
            posterior_sha256=POSTERIOR_SHA256, native_a6_rows_removed=int(self.n.keep_a6.sum()),
            native_a37_rows=int((~self.n.keep_a6).sum()), render_calls=self.render_calls,
            input_identities=self.file_identities, input_sha256=self.content_hashes,
            gradient_updates=0, colour_fit_steps=0, calibration_solves=0,
            frozen_transfer_is_not_C2_or_LOD_recalibration=True,
            RGB_occlusion_quality_verdict_pending=True, quality_accepted=False, human_verdict=None)
