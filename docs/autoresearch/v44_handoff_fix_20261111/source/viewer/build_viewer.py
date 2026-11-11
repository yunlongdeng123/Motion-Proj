"""Build an isolated five-column comparison stage. Never install into the platform."""
from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent
ASSET = "native_handoff_fix_20261111"
BEFORE_ASSET = "native_delete30k_20261109"
BEFORE_PAGE = "native_delete30k.html"
SCENE = "065"
CONTEXT = "15021599536622641101_556_150_576_150"
TARGET = "NTjq-OhZ_97OMJ4xwJAxXg"
DEFAULT_EXPERIMENT = "WAYMO-065-HANDOFF-FIX-20261111-R1"
BASELINE_EXPERIMENT = "WAYMO-065-DELETE-WHITE30K-20261109"
EXPECTED_IMAGES = {f"{mode}/{raw}.png" for mode in ("delete_white", "keep") for raw in range(199)}
EXPECTED_VIDEOS = {"delete_white_30k.mp4", "keep_30k.mp4"}
GENERATED_FILES = {"data.js", "viewer.js", "manifest.json"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_js_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    prefix = "window.NATIVE_QUALITY_DATA="
    require(text.startswith(prefix) and text.endswith(";"), f"Unexpected data assignment: {path}")
    return json.loads(text[len(prefix):-1])


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.refs: list[str] = []

    def handle_starttag(self, _tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attributes = dict(attributes)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        self.refs.extend(attributes[key] for key in ("src", "href") if attributes.get(key))


def safe_manifest_path(value: object) -> str:
    require(isinstance(value, str), "Manifest file path must be a string")
    parts = PurePosixPath(value)
    require(value and not parts.is_absolute() and ".." not in parts.parts and "\\" not in value
            and ":" not in value and str(parts) == value and bool(parts.parts)
            and all(part == part.rstrip(". ") for part in parts.parts), f"Unsafe manifest path: {value!r}")
    require(value.casefold() not in GENERATED_FILES, f"Manifest attempts to overwrite generated file: {value}")
    return value


def read_manifest(path: Path | None) -> dict | None:
    if path is None:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("scene") == SCENE, "Manifest scene is not 065")
    require(manifest.get("context") == CONTEXT, "Manifest context does not match scene 065")
    require(manifest.get("frames", manifest.get("frame_count")) == 199, "Manifest must describe 199 frames")
    require(manifest.get("iteration") == 30000, "Manifest must describe the 30K source")
    require(manifest.get("target_track") == TARGET, "Manifest target track mismatch")
    experiment = manifest.get("experiment_id")
    require(isinstance(experiment, str) and experiment.strip() and experiment != BASELINE_EXPERIMENT,
            "Manifest requires a distinct, nonempty experiment_id")
    require(isinstance(manifest.get("files"), list), "Manifest requires files:[{path,bytes,sha256}]")
    paths: set[str] = set()
    for entry in manifest["files"]:
        require(isinstance(entry, dict), "Invalid manifest file entry")
        relative = safe_manifest_path(entry.get("path"))
        require(relative.casefold() not in {path.casefold() for path in paths}, f"Duplicate manifest path: {relative}")
        require(type(entry.get("bytes")) is int and entry["bytes"] > 0, f"Invalid file size: {relative}")
        require(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]),
                f"Invalid SHA-256: {relative}")
        paths.add(relative)
    require(EXPECTED_IMAGES | EXPECTED_VIDEOS <= paths, "Manifest omits required frames or videos")
    return manifest


def validate_asset_files(root: Path, manifest: dict) -> list[dict]:
    """Check every delivered hash, all PNGs, and both complete videos before staging."""
    from PIL import Image

    root = root.resolve(strict=True)
    for entry in manifest["files"]:
        path = root / entry["path"]
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                f"Missing or unsafe asset: {path}")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"].lower(),
                f"Asset integrity mismatch: {path}")
        if entry["path"] in EXPECTED_IMAGES:
            with Image.open(path) as image:
                require(image.format == "PNG" and image.size == (480, 320), f"Wrong image grid: {path}")
                image.verify()
    executable = shutil.which("ffprobe")
    require(executable is not None, "ffprobe is required to validate the delivered videos")
    movies = []
    for name in sorted(EXPECTED_VIDEOS):
        result = subprocess.check_output([
            executable, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,nb_read_frames,avg_frame_rate,duration",
            "-of", "json", str(root / name)
        ], text=True)
        streams = json.loads(result).get("streams", [])
        require(len(streams) == 1, f"Missing video stream: {name}")
        info = streams[0]
        require(info.get("codec_name") == "h264" and info.get("width") == 480 and info.get("height") == 320,
                f"Unexpected video encoding or grid: {name}")
        require(info.get("avg_frame_rate") == "10/1" and int(info.get("nb_read_frames", -1)) == 199,
                f"Unexpected video frame count or FPS: {name}")
        movies.append({"name": name, **info})
    return movies


def make_data(before: dict, manifest: dict | None) -> dict:
    data = {
        "schema": "v44.handoff_fix.viewer.v1",
        "experiment_id": manifest["experiment_id"] if manifest else DEFAULT_EXPERIMENT,
        "baseline_experiment_id": before["experiment_id"],
        "scene": SCENE, "context": CONTEXT, "frame_count": 199,
        "training_frame_count": before["training_frame_count"], "heldout": before["heldout"],
        "iteration": 30000, "default_raw": 119, "default_start": 0,
        "width": 480, "height": 320, "target_track": TARGET,
        "roles": before["roles"], "jump_raws": [0, 30, 60, 99, 119, 133, 150, 173, 198],
        "videos": {"keep": "keep_30k.mp4", "delete_white": "delete_white_30k.mp4"},
        "versions": [
            {"id": "fixed", "label": "修复前后比较"},
            {"id": "before30", "label": "修复前 · 30K 白车删除", "url": BEFORE_PAGE,
             "params": {"step": "30000", "edit": "delete"}},
            {"id": "native10", "label": "原生重建 · 10K", "url": before["native10_url"]},
            {"id": "native20", "label": "原生重建 · 20K", "url": before["continuation_url"], "params": {"step": "20000"}},
            {"id": "native30", "label": "原生重建 · 30K", "url": before["continuation_url"], "params": {"step": "30000"}},
            {"id": "previous", "label": "历史重建与编辑", "url": before["previous_url"]}
        ],
        "frames": []
    }
    if before.get("complete_url"):
        data["versions"].append({"id": "complete", "label": "此前完整重建与编辑", "url": before["complete_url"]})
    for row in before["frames"]:
        raw = row["raw"]
        data["frames"].append({
            "raw": raw, "time_s": row["time_s"], "observed": row["observed"],
            "native10": row["native"], "native20": row["native20"], "native30": row["native30"],
            "delete_before": f"../{BEFORE_ASSET}/delete_white/{raw}.png",
            "before_keep": f"../{BEFORE_ASSET}/keep/{raw}.png",
            "fixed_keep": f"keep/{raw}.png", "fixed_delete": f"delete_white/{raw}.png",
            "boxes": row["boxes"]
        })
    return data


def check_stage(output: Path, scene: Path, data: dict, assets_present: bool) -> dict:
    page = Page()
    page.feed((output / "index.html").read_text(encoding="utf-8"))
    require(len(page.ids) == len(set(page.ids)), "Duplicate HTML IDs")
    code = (output / ASSET / "viewer.js").read_text(encoding="utf-8")
    require(not re.search(r"\bfetch\s*\(|XMLHttpRequest", code), "Viewer must work without fetch")
    required_ids = set(re.findall(r"\$\('([^']+)'\)", code))
    required_ids.update(("observed", "native30", "deleteBefore", "fixedKeep", "fixedDelete", "view", "overlay", "sampling"))
    require(required_ids <= set(page.ids), f"Missing controls: {sorted(required_ids - set(page.ids))}")
    pending: list[str] = []
    for ref in page.refs + [v["url"] for v in data["versions"] if "url" in v]:
        parts = urlsplit(ref)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        relative = unquote(parts.path)
        if (output / relative).is_file() or (scene / relative).is_file():
            continue
        if not assets_present and relative in {f"{ASSET}/{p}" for p in EXPECTED_VIDEOS}:
            pending.append(relative)
        else:
            raise ValueError(f"Unresolved local reference: {ref}")
    for row in data["frames"]:
        for key in ("observed", "native10", "native20", "native30", "delete_before", "before_keep"):
            require((scene / ASSET / row[key]).is_file(), f"Missing existing frame: {key} raw={row['raw']}")
        for key in ("fixed_keep", "fixed_delete"):
            path = output / ASSET / row[key]
            if assets_present:
                require(path.is_file(), f"Missing staged frame: {path}")
    node = shutil.which("node")
    require(node is not None, "Node.js is required for viewer syntax and behavior checks")
    subprocess.run([node, "--check", str(output / ASSET / "viewer.js")], check=True)
    subprocess.run([node, str(HERE / "test_viewer.cjs"), str(output)], check=True)
    return {"html_unique_ids": len(page.ids), "local_references_valid_against_existing_platform": True,
            "reference_resolution_mode": "stage overlay on existing scene; installation still required",
            "existing_frame_references": 199 * 6, "pending_new_asset_references": pending,
            "script_syntax": "PASS", "browser_qa": False}


def build(scene: Path, output: Path, manifest_path: Path | None = None, assets_dir: Path | None = None) -> dict:
    scene = scene.resolve(strict=True)
    output = output.resolve()
    require(not output.is_relative_to(scene.parent), "Stage output must be outside the existing platform")
    require(not output.exists(), f"Output already exists: {output}")
    require(assets_dir is None or manifest_path is not None, "--assets-dir requires --manifest")
    before = read_js_data(scene / BEFORE_ASSET / "data.js")
    require(before.get("schema") == "v44.delete30k.viewer.v1" and before.get("experiment_id") == BASELINE_EXPERIMENT,
            "Expected the existing 30K deletion data")
    require(before.get("scene") == SCENE and before.get("context") == CONTEXT, "Wrong source scene")
    require(before.get("frame_count") == 199 and [r["raw"] for r in before["frames"]] == list(range(199)),
            "Existing viewer does not have every frame 0..198")
    require(before.get("width") == 480 and before.get("height") == 320, "Unexpected existing image grid")
    source_paths = [scene / "index.html", scene / BEFORE_ASSET / "data.js", scene / BEFORE_ASSET / "viewer.js"]
    source_paths += [scene / name for name in ("native_10k.html", "native_continuation.html", "history_previous.html")]
    source_hashes = {str(path): sha(path) for path in source_paths}
    old_page = (scene / "index.html").read_bytes()
    require(f'{BEFORE_ASSET}/viewer.js'.encode() in old_page,
            "Current scene entry is no longer the expected four-column viewer")
    manifest = read_manifest(manifest_path)
    movies = validate_asset_files(assets_dir, manifest) if assets_dir else []
    data = make_data(before, manifest)
    output.mkdir(parents=True)
    asset = output / ASSET
    asset.mkdir()
    if assets_dir:
        for entry in manifest["files"]:
            target = asset / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assets_dir / entry["path"], target)
            require(sha(target) == entry["sha256"].lower(), f"Staged copy mismatch: {target}")
    if manifest_path:
        shutil.copyfile(manifest_path, asset / "manifest.json")
    template = (HERE / "index.template.html").read_text(encoding="utf-8")
    require(template.count("@@ASSET@@") == 5, "Unexpected template asset placeholders")
    (output / "index.html").write_text(template.replace("@@ASSET@@", ASSET), encoding="utf-8")
    (output / BEFORE_PAGE).write_bytes(old_page)
    shutil.copyfile(HERE / "viewer.js", asset / "viewer.js")
    (asset / "data.js").write_text("window.HANDOFF_FIX_DATA=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    checks = check_stage(output, scene, data, assets_dir is not None)
    require(all(sha(Path(path)) == digest for path, digest in source_hashes.items()), "An existing source file changed during build")
    require(sha(output / BEFORE_PAGE) == source_hashes[str(scene / "index.html")], "Four-column history was not preserved exactly")
    receipt = {
        "status": "STAGED_WITH_VERIFIED_ASSETS" if assets_dir else "STAGED_AWAITING_ASSETS",
        "scene": SCENE, "context": CONTEXT, "experiment_id": data["experiment_id"],
        "output": str(output), "installed": False, "columns": 5, "frames": 199, "default_raw": 119,
        "default_overlay": "none", "new_asset_directory": ASSET, "preserved_four_column_page": BEFORE_PAGE,
        "source_files_sha256": source_hashes, "source_files_unchanged": True,
        "manifest_sha256": sha(manifest_path) if manifest_path else None,
        "verified_pngs": 398 if assets_dir else 0, "movies": movies, **checks
    }
    (output / "build_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    args = parser.parse_args()
    build(args.scene_root, args.output, args.manifest, args.assets_dir)
