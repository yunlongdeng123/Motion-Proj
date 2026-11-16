"""Manifest-driven inspect/render CLI; help/inspect never call the CUDA factory.

Examples: python -m PACKAGE.v44_cli inspect --manifest scene.json
          python -m PACKAGE.v44_cli render --manifest scene.json --query query.json
                 --frames-json '["frame-A",17]' --output new_render_directory
Render uses the real factory by default. Injected factories in tests verify only CPU I/O protocols.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np


def _module(name):
    if __package__:
        return importlib.import_module("." + name, __package__)
    directory = Path(__file__).resolve().parent
    if str(directory.parent) not in sys.path:
        sys.path.insert(0, str(directory.parent))
    return importlib.import_module(directory.name + "." + name)


def _factory():
    return _module("v44_factory").from_manifest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    def encode(value):
        if isinstance(value, Path): return str(value)
        if isinstance(value, np.generic): return value.item()
        raise TypeError("Non-JSON provenance value: " + type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=encode) + "\n", encoding="utf-8")


def _frames(manifest, keys):
    registered = {frame.key: frame for frame in manifest.frames}
    if keys is None:
        return manifest.frames
    if not isinstance(keys, (list, tuple)) or any(type(key) not in (int, str) for key in keys):
        raise ValueError("Frame subset must be a JSON array of explicit string/integer keys")
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(registered):
        raise ValueError("Frame subset must be nonempty, distinct and registered in the manifest")
    return tuple(registered[key] for key in keys)


def _frame_info(frame):
    return dict(key=frame.key, image_name=frame.image_name, model_time=frame.model_time,
                source_time_s=frame.source_time_s,
                observation=None if frame.observation is None else str(frame.observation))


def resolve_query(manifest, query=None):
    query = {} if query is None else query
    allowed = {kind + suffix for kind in ("delete", "focus", "diagnostic")
               for suffix in ("_track_ids", "_actor_ids")} | {"schema", "diagnostic_branch_alpha"}
    if not isinstance(query, dict) or not set(query) <= allowed:
        raise ValueError("Unknown query fields or non-object query")
    if query.get("schema", "v44.query.v1") != "v44.query.v1":
        raise ValueError("Expected v44.query.v1 query schema")
    branch_alpha = query.get("diagnostic_branch_alpha", False)
    if type(branch_alpha) is not bool:
        raise ValueError("diagnostic_branch_alpha must be a JSON bool")
    result = {"diagnostic_branch_alpha": branch_alpha}
    for kind in ("delete", "focus", "diagnostic"):
        tracks, actors = query.get(kind + "_track_ids", []), query.get(kind + "_actor_ids", [])
        if not isinstance(tracks, (list, tuple)) or not isinstance(actors, (list, tuple)) or (tracks and actors):
            raise ValueError(kind + " must use either a track list or an actor list")
        if any(not isinstance(value, str) or not value for value in tracks):
            raise ValueError("Track IDs must be nonempty strings")
        if tracks:
            ids = manifest.actor_ids(tracks)
        else:
            if any(type(value) is not int or value <= 0 for value in actors):
                raise ValueError("Actor IDs must be positive integers")
            ids = tuple(actors)
            if len(set(ids)) != len(ids) or not set(ids) <= set(manifest.actor_to_track):
                raise ValueError("Actor IDs must be distinct registered identities")
        capacity = 14 if branch_alpha else 16
        if kind == "diagnostic" and len(ids) > capacity:
            raise ValueError(f"At most {capacity} diagnostic actors fit the native feature ABI in this mode")
        result[kind + "_actor_ids"] = ids
    return result


def inspect_manifest(path, frame_keys=None):
    path = Path(path).resolve(strict=True)
    manifest = _module("v44_manifest").SceneManifest.load(path)
    return dict(schema="v44.inspect.v1", manifest=str(path), manifest_sha256=_sha(path),
        model=str(manifest.model), engine=str(manifest.engine), iteration=manifest.iteration,
        full_frame_count=len(manifest.frames), frames=[_frame_info(f) for f in _frames(manifest, frame_keys)],
        actors=[dict(actor_id=actor, track_id=track) for actor, track in manifest.actor_to_track.items()],
        zero_semantic_policy=manifest.zero_semantic_policy, surfaces=len(manifest.surfaces),
        road=None if manifest.road is None else str(manifest.road),
        validation="manifest schema/registry only; no factory/model loading or GPU execution")


def _float_array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Renderer returned nonfinite float arrays")
    return arr


def _export_frame(output, index, frame, result, resolved, manifest_sha):
    from PIL import Image
    groups = _module("v44_scene").GROUPS
    rgb, alpha = _float_array(result["rgb"]), _float_array(result["alpha"])
    if rgb.ndim != 3 or rgb.shape[0] != 3 or alpha.shape != (1,) + rgb.shape[1:]:
        raise ValueError("Expected RGB(3,H,W) and alpha(1,H,W)")
    group_alpha = np.concatenate([_float_array(result["diagnostics"][name]) for name in groups], axis=0)
    owner_ids = tuple(result["owner_alpha_actor_ids"])
    owner_alpha = _float_array(result["owner_alpha_T"])
    if (group_alpha.shape != (len(groups),) + rgb.shape[1:]
            or owner_ids != resolved["diagnostic_actor_ids"]
            or owner_alpha.shape != (len(owner_ids),) + rgb.shape[1:]):
        raise ValueError("Diagnostic actor ordering or alpha grid differs from the resolved query")
    # Six mutually exclusive groups conserve alpha; actor labels are separate overlapping diagnostics.
    if not np.allclose(group_alpha.sum(0, keepdims=True), alpha, atol=2e-5, rtol=2e-5):
        raise ValueError("Six-group alpha conservation failed during export")
    arrays = dict(rgb=rgb, alpha=alpha, group_alpha_T=group_alpha,
        group_names=np.asarray(groups), owner_alpha_T=owner_alpha,
        owner_alpha_actor_ids=np.asarray(owner_ids, dtype=np.int64))
    branch_names = tuple(result.get("branch_alpha_names", ()))
    expected_branch_names = _module("v44_scene").BRANCH_ALPHA_NAMES if resolved["diagnostic_branch_alpha"] else ()
    branch_alpha = _float_array(result.get("branch_alpha_T", np.empty((0,) + rgb.shape[1:], np.float32)))
    if branch_names != expected_branch_names or branch_alpha.shape != (len(branch_names),) + rgb.shape[1:]:
        raise ValueError("Native nonfocus branch-alpha labels do not match the query/grid")
    arrays.update(branch_alpha_T=branch_alpha, branch_alpha_names=np.asarray(branch_names, dtype="U32"))
    for name in ("raw_rgb", "env_map", "depth_sum"):
        if name in result:
            arrays[name] = _float_array(result[name])
            if arrays[name].shape != (alpha.shape if name == "depth_sum" else rgb.shape):
                raise ValueError(name + " does not match the rendered image grid")
    folder = output / "frames" / f"{index:06d}"
    folder.mkdir(parents=True, exist_ok=False)   # key is provenance, never an output pathname
    Image.fromarray(np.rint(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)).save(folder / "rgb.png")
    np.savez_compressed(folder / "arrays.npz", **arrays)
    metadata = dict(schema="v44.render.frame.v1", frame=_frame_info(frame), manifest_sha256=manifest_sha,
        resolved_query=resolved, owner_alpha_actor_ids=owner_ids, diagnostic_group_names=groups,
        branch_alpha_names=branch_names,
        coverage=result.get("coverage", []), diagnostic_alpha_error_max=result.get("diagnostic_alpha_error_max"),
        arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in arrays.items()},
        png_policy="native RGB clamped/rounded to uint8 only for display; NPZ retains unclamped float RGB",
        files={"rgb_png": str((folder / "rgb.png").relative_to(output)),
               "float_npz": str((folder / "arrays.npz").relative_to(output))}, training_approved=False)
    _write(folder / "provenance.json", metadata)
    return dict(frame_key=frame.key, provenance=str((folder / "provenance.json").relative_to(output)),
                files=metadata["files"], coverage=metadata["coverage"])


def render_manifest(path, output, *, query=None, frame_keys=None, factory=None, query_source=None):
    path, output = Path(path).resolve(strict=True), Path(output).resolve()
    manifest = _module("v44_manifest").SceneManifest.load(path)
    frames, resolved = _frames(manifest, frame_keys), resolve_query(manifest, query)
    digest = _sha(path)
    if output == path.parent or output in path.parents:
        raise ValueError("Output must be a new independent artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="v44.render.run.v1", status="RUNNING", manifest=str(path), manifest_sha256=digest,
        manifest_data=manifest.data, model=str(manifest.model), iteration=manifest.iteration,
        engine=str(manifest.engine), requested_query={} if query is None else query,
        resolved_query=resolved, query_source=query_source, selected_frame_keys=[f.key for f in frames],
        full_frame_count=len(manifest.frames), frames=[], persistent_check_before=False,
        persistent_check_after=False, training_approved=False,
        validation="render output; native GPU equivalence requires the calling workflow's actual comparison")
    _write(output / "provenance.json", receipt)
    try:
        scene, loaded = (factory or _factory())(path)
        if _sha(path) != digest or loaded.data != manifest.data:
            raise RuntimeError("Manifest changed while loading the native scene")
        scene.assert_persistent_unchanged()
        receipt["persistent_check_before"] = True
        try:
            for index, frame in enumerate(frames):
                result = scene.render(frame.key, **resolved)
                receipt["frames"].append(_export_frame(output, index, frame, result, resolved, digest))
        finally:
            scene.assert_persistent_unchanged()  # includes optional provider clip-boundary checks
            receipt["persistent_check_after"] = True
        receipt["status"] = "COMPLETE"
    except Exception as error:
        receipt.update(status="FAILED", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        _write(output / "provenance.json", receipt)
    return receipt


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "render"):
        entry = commands.add_parser(command)
        entry.add_argument("--manifest", type=Path, required=True)
        subset = entry.add_mutually_exclusive_group()
        subset.add_argument("--frames-json", help="JSON array preserving string/integer frame-key types")
        subset.add_argument("--frame-key", action="append", help="Repeat a JSON string/integer scalar, or a plain nonnumeric string")
        if command == "render":
            entry.add_argument("--output", type=Path, required=True)
            entry.add_argument("--query", type=Path, help="JSON object of delete/focus/diagnostic track_ids or actor_ids")
            entry.add_argument("--diagnostic-branch-alpha", action="store_true", default=None,
                               help="Export native nonfocus static/dynamic alpha, including unknown owners; owner capacity becomes 14")
            for kind in ("delete", "focus", "diagnostic"):
                group = entry.add_mutually_exclusive_group()
                group.add_argument("--" + kind + "-track-id", action="append")
                group.add_argument("--" + kind + "-actor-id", type=int, action="append")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        keys = None
        if args.frames_json is not None:
            keys = json.loads(args.frames_json)
        elif args.frame_key is not None:
            keys = []
            for key in args.frame_key:
                try: keys.append(json.loads(key))
                except json.JSONDecodeError: keys.append(key)
        if args.command == "inspect":
            report = inspect_manifest(args.manifest, keys)
        else:
            query, source = {}, None
            for kind in ("delete", "focus", "diagnostic"):
                for identity in ("track", "actor"):
                    values = getattr(args, kind + "_" + identity + "_id")
                    if values is not None: query[kind + "_" + identity + "_ids"] = values
            if args.query is not None:
                if query: raise ValueError("Use either --query or direct identity flags")
                query = json.loads(args.query.read_text(encoding="utf-8"))
                source = dict(path=str(args.query.resolve()), sha256=_sha(args.query))
            if args.diagnostic_branch_alpha is not None:
                if "diagnostic_branch_alpha" in query and query["diagnostic_branch_alpha"] != args.diagnostic_branch_alpha:
                    raise ValueError("Branch-alpha flag conflicts with the JSON query")
                query["diagnostic_branch_alpha"] = args.diagnostic_branch_alpha
            report = render_manifest(args.manifest, args.output, query=query, frame_keys=keys, query_source=source)
            report = {key: report[key] for key in ("status", "manifest", "selected_frame_keys", "resolved_query", "persistent_check_before", "persistent_check_after")}
            report["output"] = str(args.output.resolve())
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps(dict(error=type(error).__name__, message=str(error)), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
