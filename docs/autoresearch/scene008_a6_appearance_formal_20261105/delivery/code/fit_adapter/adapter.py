"""CPU colour fitting for external a6 RGB candidates on one frozen M2 asset.

This module imports neither Torch nor a renderer.  Formal ray packets must
come from native_extract.py; its tiny test packets are explicitly labelled.
Only exact zero columns of the ORIGINAL BUILD operator can change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.sparse import csr_matrix, load_npz, vstack

C0 = np.float32(0.28209479177387814)
SCHEMA = "a6.external_appearance_candidates.v1"
NATIVE_SCHEMA = "a6.native_fixed_geometry_alphaT.v1"
TEST_SCHEMA = "a6.cpu_fixture_not_native.v1"
APPEARANCE = {"canonical_sh", "rgb", "colour_inferred", "metadata_json"}
HELD_RAW_IDS = {89, 95, 101, 105, 113, 121, 125, 130, 137}


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def bank(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def resolve(root: Path, path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def load_reference(ref: Any, root: Path, default_key: str) -> tuple[np.ndarray, Path]:
    """A reference is a path, or {path, key}; never infer masks from RGB."""
    entry = {"path": ref} if isinstance(ref, str) else ref
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("Expected a file path or an explicit {path,key} reference")
    path = resolve(root, entry["path"])
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as values:
            result = values[entry.get("key", default_key)].copy()
    elif path.suffix.lower() == ".npy":
        result = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im:
            result = np.asarray(im).copy()
    return result, path


def as_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected a finite explicit mask of shape {shape}")
    if value.dtype != bool and not np.isin(value, [0, 1, 255]).all():
        raise ValueError(f"{name}: use boolean or binary 0/1/255; no implicit threshold")
    return value != 0


def project_local(xyz: np.ndarray, view: dict[str, np.ndarray], *, crop: bool) -> np.ndarray:
    """Independent pinhole projection, pixel centre at x+.5/y+.5."""
    transform = np.asarray(view["camera_from_local_m"], np.float64)
    xyz_camera = np.asarray(xyz, np.float64) @ transform[:3, :3].T + transform[:3, 3]
    K = np.asarray(view["camera_K_crop" if crop else "camera_K"], np.float64)
    homogeneous = xyz_camera @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:] - .5


def validate_view(view: dict[str, np.ndarray]) -> dict[str, Any]:
    required = ("camera_K", "camera_K_crop", "camera_from_local_m", "model_from_local",
                "model_to_camera", "meter_to_model", "full_image_hw", "crop_xyxy",
                "RGB_known_mask", "RGB_unknown_mask")
    missing = set(required) - set(view)
    if missing:
        raise ValueError("Camera package missing fields: " + ", ".join(sorted(missing)))
    for key in required:
        if not np.isfinite(view[key]).all():
            raise ValueError("Nonfinite camera/mask field: " + key)
    h, w = np.asarray(view["full_image_hw"], dtype=int)
    x0, y0, x1, y1 = np.asarray(view["crop_xyxy"], dtype=int)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Invalid full-image/crop bounds")
    if not np.array_equal(view["crop_xyxy"], [x0, y0, x1, y1]):
        raise ValueError("Crop coordinates must be integer pixel boundaries")
    shape = (int(y1-y0), int(x1-x0))
    known = as_mask(view["RGB_known_mask"], shape, "RGB_known_mask")
    unknown = as_mask(view["RGB_unknown_mask"], shape, "RGB_unknown_mask")
    if not np.array_equal(unknown, ~known):
        raise ValueError("Camera package requires explicit known/unknown complements")
    K = np.asarray(view["camera_K"], np.float64)
    Kc = np.asarray(view["camera_K_crop"], np.float64)
    if (K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0 or
            not np.array_equal(K[2], [0, 0, 1]) or abs(K[0, 1])+abs(K[1, 0]) > 1e-12):
        raise ValueError("Expected the native zero-skew camera intrinsic matrix")
    expected_Kc = K.copy(); expected_Kc[:2, 2] -= [x0, y0]
    if not np.allclose(Kc, expected_Kc, atol=1e-10, rtol=0):
        raise ValueError("Crop K does not agree with the full K and crop offsets")
    pose = np.asarray(view["model_from_local"], np.float64)
    camera = np.asarray(view["model_to_camera"], np.float64)
    metric = np.asarray(view["camera_from_local_m"], np.float64)
    for key, matrix in (("pose", pose), ("camera", camera), ("metric_camera", metric)):
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError(key + " must be an affine 4x4 transform")
    scale = float(view["meter_to_model"])
    if scale <= 0 or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3)*scale**2,
                                     atol=3e-8, rtol=0):
        raise ValueError("Actor pose is not the declared proper uniform similarity")
    if np.linalg.det(pose[:3, :3]) <= 0:
        raise ValueError("Reflections are forbidden")
    composed = camera @ pose; composed[:3, :4] /= scale
    error = float(abs(composed-metric).max())
    if error > 2e-5 or not np.allclose(metric[:3, :3].T @ metric[:3, :3], np.eye(3),
                                      atol=3e-6, rtol=0):
        raise ValueError("Metric camera disagrees with native model-space insertion")
    # Off-centre points make this a real crop check, not a principal-point-only identity.
    probes = np.array([[0., 0., 0.], [.7, -.4, .2], [-.9, .3, -.25]], np.float64)
    full = project_local(probes, view, crop=False) - [x0, y0]
    cropped = project_local(probes, view, crop=True)
    projection_error = float(abs(full-cropped).max())
    if not np.isfinite(full).all() or projection_error > 1e-9:
        raise ValueError("Independent crop projection failed")
    return dict(crop_shape=list(shape), full_image_hw=[int(h), int(w)],
                native_metric_transform_maxabs=error,
                independent_crop_projection_maxabs=projection_error,
                pixel_centre="u=K*camera_xyz/z-.5")


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != SCHEMA or not manifest.get("views"):
        raise ValueError("Expected nonempty " + SCHEMA)
    ids = [v["id"] for v in manifest["views"]]
    if len(ids) != len(set(ids)):
        raise ValueError("View IDs must be unique")
    return manifest, path.parent


def view_inputs(entry: dict[str, Any], root: Path, erosion: int) -> dict[str, Any]:
    if entry.get("rgb_mode") != "straight_srgb":
        raise ValueError("Declare rgb_mode=straight_srgb; black composited/transmissive RGB is not a target")
    camera_path = resolve(root, entry["camera_npz"])
    view = bank(camera_path); audit = validate_view(view)
    if "raw" in view and int(view["raw"]) in HELD_RAW_IDS:
        raise ValueError("HELD views are final evaluation only; they cannot supervise candidate fitting")
    shape = tuple(audit["crop_shape"])
    foreground, fg_path = load_reference(entry["foreground_mask"], root, "foreground_mask")
    unknown, unknown_path = load_reference(entry["unknown_mask"], root, "RGB_unknown_mask")
    foreground = as_mask(foreground, shape, "foreground_mask")
    unknown = as_mask(unknown, shape, "unknown_mask")
    if (unknown & np.asarray(view["RGB_known_mask"], bool)).any():
        raise ValueError("Candidate unknown mask overlaps genuine observed RGB")
    if erosion < 0:
        raise ValueError("Foreground erosion cannot be negative")
    interior = binary_erosion(foreground, iterations=erosion) if erosion else foreground
    eligible = interior & unknown & ~np.asarray(view["RGB_known_mask"], bool)
    return dict(view=view, camera_path=camera_path, foreground_path=fg_path,
                unknown_path=unknown_path, eligible=eligible, audit=audit,
                eligible_sha256=array_sha(eligible))


def load_rgb(entry: dict[str, Any], root: Path, shape: tuple[int, int]) -> tuple[np.ndarray, Path]:
    rgb, path = load_reference(entry["rgb"], root, "RGB_srgb")
    if rgb.shape != (*shape, 3):
        raise ValueError("Candidate RGB shape must exactly equal the camera crop; no silent resize/alpha")
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float64)/255.
    elif rgb.dtype.kind == "f":
        rgb = rgb.astype(np.float64)
    else:
        raise ValueError("RGB must be uint8 or explicit float sRGB")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB must be finite sRGB in [0,1]")
    return rgb, path


def original_zero_columns(design: csr_matrix, count: int) -> np.ndarray:
    A = design.tocsr(copy=True); A.eliminate_zeros()
    if A.shape[1] != count or not np.isfinite(A.data).all() or (A.data < 0).any():
        raise ValueError("Original BUILD design must be finite, nonnegative, and bind to every primitive")
    return np.asarray(A.getnnz(axis=0) == 0).ravel()


def validate_asset(arrays: dict[str, np.ndarray], expected_count: int | None = 18432) -> None:
    n = len(arrays["xyz"])
    if expected_count is not None and n != expected_count:
        raise ValueError(f"Expected the same {expected_count} M2 primitives")
    if arrays["canonical_sh"].shape != (n, 16, 3) or arrays["rgb"].shape != (n, 3):
        raise ValueError("Expected the old-runtime canonical SH asset schema")
    if not np.isfinite(arrays["canonical_sh"]).all() or not np.isfinite(arrays["rgb"]).all():
        raise ValueError("Nonfinite source appearance")
    if np.any(arrays["canonical_sh"][:, 1:] != 0):
        raise ValueError("This RGB operator adapter requires the M2 SH0 baseline; SH3 needs per-view fixed terms")


def collect_problem(manifest_path: str | Path, base_path: str | Path, build_path: str | Path,
                    *, min_alpha: float = .9, min_unknown_fraction: float = .25,
                    allow_test_operator: bool = False,
                    additional_zero_mask_path: str | Path | None = None,
                    additional_protected_design_path: str | Path | None = None) -> dict[str, Any]:
    if not (0 < min_alpha <= 1 and 0 < min_unknown_fraction <= 1):
        raise ValueError("Alpha and unknown-fraction thresholds must be in (0,1]")
    manifest, root = load_manifest(manifest_path)
    base = bank(base_path); validate_asset(base)
    build = load_npz(build_path).tocsr(); unknown = original_zero_columns(build, len(base["xyz"]))
    allowed = unknown.copy(); additional_zero = None; additional_design = None
    if additional_zero_mask_path is not None:
        additional_zero = bank(additional_zero_mask_path)["unknown"]
        if additional_zero.dtype != bool or additional_zero.shape != unknown.shape:
            raise ValueError("Additional observation zero mask must be explicit bool[N]")
        allowed &= additional_zero
    if additional_protected_design_path is not None:
        additional_design = load_npz(additional_protected_design_path).tocsr()
        exact_additional_zero = original_zero_columns(additional_design, len(base["xyz"]))
        if additional_zero is None or not np.array_equal(exact_additional_zero, additional_zero):
            raise ValueError("Additional observed design and declared zero-column mask differ")
    matrices, targets, row_weights, records, hashes = [], [], [], [], {}
    for entry in manifest["views"]:
        opdir = resolve(root, entry["operator_dir"])
        meta = json.loads((opdir/"operator.json").read_text(encoding="utf-8"))
        accepted = {NATIVE_SCHEMA, TEST_SCHEMA} if allow_test_operator else {NATIVE_SCHEMA}
        if meta.get("schema") not in accepted:
            raise ValueError("Formal fits require an exact native alpha*T operator packet")
        inputs = view_inputs(entry, root, int(meta["foreground_erosion_pixels"]))
        view, eligible = inputs["view"], inputs["eligible"]
        if meta["camera_package_sha256"] != sha(inputs["camera_path"]) or meta["eligible_mask_sha256"] != array_sha(eligible):
            raise ValueError("Ray packet no longer matches the explicit masks/camera")
        for key in ("geometry_hash", "provenance_hash"):
            if meta[key] != scalar(base[key]):
                raise ValueError("Ray packet frozen-asset identity mismatch: " + key)
        if meta["primitive_count"] != len(base["xyz"]):
            raise ValueError("Primitive identity/row count mismatch")
        Wpath, rpath = opdir/"weights.npz", opdir/"rays.npz"
        if meta["weights_sha256"] != sha(Wpath) or meta["rays_sha256"] != sha(rpath):
            raise ValueError("Ray packet bytes changed")
        W = load_npz(Wpath).astype(np.float64).tocsr(); W.eliminate_zeros(); rays = bank(rpath)
        alpha, crop_pixels, full_pixels = rays["alpha"], rays["crop_pixels"], rays["full_pixels"]
        if W.shape != (len(alpha), len(base["xyz"])) or (W.data < 0).any() or not np.isfinite(W.data).all():
            raise ValueError("Malformed nonnegative native W")
        x0, y0, x1, y1 = view["crop_xyxy"].astype(int); h, w = view["full_image_hw"].astype(int)
        expected = np.flatnonzero(eligible.ravel())
        expected_full = (expected//(x1-x0)+y0)*w + expected%(x1-x0)+x0
        if not np.array_equal(crop_pixels, expected) or not np.array_equal(full_pixels, expected_full):
            raise ValueError("Rows must exactly follow eligible crop/full-image pixel identities")
        alpha_error = float(abs(np.asarray(W.sum(axis=1)).ravel()-alpha).max()) if len(alpha) else 0.
        if alpha_error > 5e-6 or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1+1e-6).any():
            raise ValueError("Native alpha conservation failed")
        rgb, rgbpath = load_rgb(entry, root, eligible.shape)
        A = W.multiply(1/np.maximum(alpha, 1e-12)[:, None]).tocsr()
        fraction = np.asarray(A[:, allowed].sum(axis=1)).ravel()
        use = (alpha >= min_alpha) & (fraction >= min_unknown_fraction)
        count = int(use.sum()); weight = float(entry.get("weight", 1.))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("View weight must be finite and positive")
        if not count:
            raise ValueError(f"{entry['id']}: no opaque explicit unknown foreground rows survive admission")
        matrices.append(A[use]); targets.append(rgb.reshape(-1, 3)[crop_pixels[use]])
        # Equal default total mass per view, rather than letting large crops dominate.
        row_weights.append(np.full(count, weight/count, np.float64))
        records.append(dict(id=entry["id"], candidate_pixels=int(eligible.sum()), fitted_pixels=count,
                            rejected_low_prior_alpha=int((alpha < min_alpha).sum()),
                            rejected_low_unknown_fraction=int((fraction < min_unknown_fraction).sum()),
                            native_alpha_conservation_maxabs=alpha_error,
                            min_admitted_prior_alpha=float(alpha[use].min()),
                            min_admitted_unknown_fraction=float(fraction[use].min()),
                            operator_schema=meta["schema"], camera_audit=inputs["audit"]))
        for path in (inputs["camera_path"], inputs["foreground_path"], inputs["unknown_path"], rgbpath,
                     opdir/"operator.json", Wpath, rpath):
            hashes[str(path)] = sha(path)
    A = vstack(matrices, format="csr"); target = np.concatenate(targets)
    weights = np.concatenate(row_weights); weights /= weights.sum()
    active = allowed & (np.asarray(A.getnnz(axis=0)).ravel() > 0)
    if not active.any():
        raise ValueError("No original zero columns are reached")
    B = A[:, active].multiply(np.sqrt(weights)[:, None]).tocsr()
    prior = base["rgb"][active].astype(np.float64)
    fixed = A @ base["rgb"].astype(np.float64) - A[:, active] @ prior
    residual_target = (target-fixed)*np.sqrt(weights)[:, None]
    identity = dict(schema=SCHEMA, manifest_sha256=sha(manifest_path), base_sha256=sha(base_path),
                    original_BUILD_design_sha256=sha(build_path), min_prior_alpha=min_alpha,
                    min_unknown_fraction=min_unknown_fraction, source_SHA256=hashes)
    if additional_zero_mask_path is not None:
        identity["additional_zero_mask_sha256"] = sha(additional_zero_mask_path)
    if additional_protected_design_path is not None:
        identity["additional_protected_design_sha256"] = sha(additional_protected_design_path)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return dict(base=base, build=build, unknown=unknown, allowed=allowed,
                additional_zero=additional_zero, additional_design=additional_design,
                active=active, A=A, B=B,
                target=target, residual_target=residual_target, prior=prior, weights=weights,
                views=records, identity=identity, fingerprint=fingerprint)


class Solver:
    """Deterministic bounded FP64 CPU FISTA; complete state is restartable."""
    def __init__(self, problem: dict[str, Any], ridge: float = .001):
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("Ridge must be finite and nonnegative")
        self.problem, self.ridge = problem, float(ridge)
        self.B, self.target, self.prior = problem["B"], problem["residual_target"], problem["prior"]
        upper = np.asarray(abs(self.B).T @ np.asarray(abs(self.B).sum(axis=1)).ravel()).ravel()
        self.D = 2*(upper+self.ridge/len(self.prior))[:, None]
        if not np.isfinite(self.D).all() or (self.D <= 0).any():
            raise ValueError("The admitted fit has an unconstrained active column")
        self.fingerprint = hashlib.sha256((problem["fingerprint"]+repr(self.ridge)).encode()).hexdigest()

    def objective(self, x: np.ndarray) -> float:
        residual = self.B@x-self.target
        return float(np.square(residual).sum()+self.ridge*np.square(x-self.prior).sum()/len(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2*(self.B.T@(self.B@x-self.target)+self.ridge*(x-self.prior)/len(x))

    def begin(self) -> dict[str, Any]:
        x = self.prior.copy()
        return dict(x=x, y=x.copy(), t=1., loss=self.objective(x), step=0, restarts=0)

    def step(self, state: dict[str, Any]) -> dict[str, Any]:
        x, y, t = state["x"], state["y"], state["t"]
        z = np.clip(y-self.gradient(y)/self.D, 0, 1); loss = self.objective(z)
        restarts = state["restarts"]
        if loss > state["loss"]+1e-12:
            t = 1.; z = np.clip(x-self.gradient(x)/self.D, 0, 1)
            loss = self.objective(z); restarts += 1
        if not np.isfinite(z).all() or not math.isfinite(loss):
            raise FloatingPointError("Nonfinite fit")
        tn = (1+math.sqrt(1+4*t*t))/2
        return dict(x=z, y=z+(t-1)/tn*(z-x), t=tn, loss=loss,
                    step=state["step"]+1, restarts=restarts)

    def save_state(self, path: str | Path, state: dict[str, Any]) -> None:
        np.savez_compressed(path, **state, fingerprint=np.asarray(self.fingerprint),
                            active_columns=np.flatnonzero(self.problem["active"]))

    def load_state(self, path: str | Path) -> dict[str, Any]:
        values = bank(path)
        if scalar(values["fingerprint"]) != self.fingerprint or not np.array_equal(
                values["active_columns"], np.flatnonzero(self.problem["active"])):
            raise ValueError("State does not bind to identical inputs/operators/settings")
        state = {k: values[k] for k in ("x", "y")}
        state.update({k: values[k].item() for k in ("t", "loss", "step", "restarts")})
        if (state["x"].shape != self.prior.shape or state["y"].shape != self.prior.shape or
                not np.isfinite(state["x"]).all() or not np.isfinite(state["y"]).all()):
            raise ValueError("Malformed full optimiser state")
        return state


def export_asset(path: str | Path, problem: dict[str, Any], state: dict[str, Any],
                 metadata: dict[str, Any]) -> dict[str, Any]:
    base, active, unknown = problem["base"], problem["active"], problem["unknown"]
    result = {key: value.copy() for key, value in base.items()}
    rgb = np.asarray(state["x"], np.float32)
    result["canonical_sh"][active, 0] = (rgb-np.float32(.5))/C0
    result["rgb"][active] = result["canonical_sh"][active, 0]*C0+np.float32(.5)
    result["colour_inferred"][active] = True
    old_meta = json.loads(scalar(base.get("metadata_json", np.asarray("{}"))))
    old_meta["external_generated_appearance"] = dict(metadata,
        scope="Only exact zero columns of original BUILD; generated colours remain inferred",
        no_geometry_or_opacity_updates=True, no_background_RGB_targets=True,
        protected_nonzero_BUILD_columns_bitwise_preserved=True)
    result["metadata_json"] = np.asarray(json.dumps(old_meta, ensure_ascii=False, sort_keys=True, allow_nan=False))
    for key in set(base)-APPEARANCE:
        if not same_bytes(base[key], result[key]):
            raise AssertionError("Frozen geometry/provenance changed: " + key)
    for key in ("canonical_sh", "rgb", "colour_inferred"):
        if not same_bytes(base[key][~active], result[key][~active]):
            raise AssertionError("Unselected/protected source rows changed: " + key)
    if not np.array_equal(problem["build"]@base["rgb"], problem["build"]@result["rgb"]):
        raise AssertionError("Original BUILD predictions changed")
    extra = problem.get("additional_design")
    if extra is not None and not np.array_equal(extra@base["rgb"], extra@result["rgb"]):
        raise AssertionError("Additional high-resolution BUILD predictions changed")
    if (active & ~unknown).any():
        raise AssertionError("Nonzero original BUILD column selected")
    np.savez_compressed(path, **result)
    reloaded = bank(path)
    if not all(same_bytes(value, reloaded[key]) for key, value in result.items()):
        raise AssertionError("Export/reload lost array bytes")
    return dict(path=str(Path(path).resolve()), sha256=sha(path),
                original_zero_columns=int(unknown.sum()), updated_columns=int(active.sum()),
                permitted_zero_columns=int(problem["allowed"].sum()),
                additional_observation_zero_columns=(int(problem["additional_zero"].sum())
                    if problem["additional_zero"] is not None else None),
                additional_BUILD_prediction_bitwise_unchanged=(True if extra is not None else None),
                unhit_unknown_columns_unchanged=int((unknown & ~active).sum()),
                protected_columns=int((~unknown).sum()), geometry_and_provenance_exact=True,
                original_BUILD_RGB_prediction_maxabs=0., all_protected_SH_coefficients_bitwise_unchanged=True)


def fit_cli(args: argparse.Namespace) -> None:
    dest = Path(args.output).resolve(); dest.mkdir(parents=True, exist_ok=False)
    problem = collect_problem(args.manifest, args.asset, args.build_design,
                              min_alpha=args.min_prior_alpha,
                              min_unknown_fraction=args.min_unknown_fraction,
                              additional_zero_mask_path=getattr(args, "additional_zero_mask", None),
                              additional_protected_design_path=getattr(args, "additional_protected_design", None))
    solver = Solver(problem, args.ridge)
    state = solver.load_state(args.resume) if args.resume else solver.begin()
    first_step, initial = state["step"], state["loss"]
    trace = []
    if args.steps < first_step:
        raise ValueError("Final total step count cannot precede a resumed state")
    while state["step"] < args.steps:
        state = solver.step(state)
        if state["step"] % 100 == 0 or state["step"] == args.steps:
            trace.append(dict(step=state["step"], objective=state["loss"], restarts=state["restarts"]))
    solver.save_state(dest/"full_state.npz", state)
    restored = solver.load_state(dest/"full_state.npz")
    next1, next2 = solver.step(state), solver.step(restored)
    if not all(same_bytes(next1[k], next2[k]) for k in ("x", "y")):
        raise AssertionError("Full-state next-step replay failed")
    exported = export_asset(dest/"complete_a6.npz", problem, state,
                            dict(identity=problem["identity"], fingerprint=solver.fingerprint,
                                 steps=state["step"], ridge=args.ridge))
    np.savez_compressed(dest/"replay.npz", fitted_unknown_RGB_srgb=state["x"],
                        updated_primitive_rows=np.flatnonzero(problem["active"]),
                        original_BUILD_zero_columns=problem["unknown"],
                        permitted_zero_columns=problem["allowed"],
                        protected_primitive_rows=np.flatnonzero(~problem["unknown"]),
                        full_state_fingerprint=np.asarray(solver.fingerprint),
                        metadata_json=np.asarray(json.dumps(problem["identity"], sort_keys=True)))
    write_json(dest/"receipt.json", dict(status="PASS_UNKNOWN_COLOUR_ADAPTER",
        formal_generated_colour_steps=state["step"]-first_step, neural_training_steps=0,
        replay_nextstep_checks_extra=2, initial_objective=initial, final_objective=state["loss"],
        exact_full_state_nextstep_replay=True, native_render_validation_required=True,
        CPU_solver=True, geometry_updates=0, views=problem["views"], export=exported,
        trace=trace, identity=problem["identity"], fingerprint=solver.fingerprint))
    print(json.dumps(dict(status="PASS", output=str(dest), **exported), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset", required=True, help="Original M2 or same-geometry SH0 HR-refit asset")
    parser.add_argument("--build-design", required=True, help="ORIGINAL M2 BUILD design.npz, never a refit design")
    parser.add_argument("--output", required=True, help="New directory; old results are never overwritten")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--ridge", type=float, default=.001)
    parser.add_argument("--min-prior-alpha", type=float, default=.9)
    parser.add_argument("--min-unknown-fraction", type=float, default=.25)
    parser.add_argument("--resume")
    parser.add_argument("--additional-zero-mask", help="NPZ unknown bool[N]; intersect with original zero columns")
    parser.add_argument("--additional-protected-design", help="Extra real observed BUILD design for exact invariance verification")
    fit_cli(parser.parse_args())


if __name__ == "__main__":
    main()
