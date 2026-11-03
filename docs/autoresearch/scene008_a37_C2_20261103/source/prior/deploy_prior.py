"""Download/verify the explicitly authorized, pinned C2 frozen shape prior."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OMNI_REV = "70e803bfb4e127d534049d8ab8c8cb511780d485"
DINO_REV = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
OMNI_ROOT = Path("/root/autodl-tmp/models/c2_hunyuan_omni_70e803")
DINO_ROOT = Path("/root/autodl-tmp/models/c2_dinov2_large_47b73")
RUN_ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1")
SOURCE = RUN_ROOT / "source/prior"
OMNI_FILES = {
    "config.json": (321, None),
    "cond_encoder/config.json": (641, None),
    "cond_encoder/pytorch_model.bin": (464828, "f3f308e73153000ed566e985e2b1e38e9b40c12e23b7e6ecf89ce3ea1699c432"),
    "image_processor/config.json": (78, None),
    "model/config.json": (478, None),
    "model/pytorch_model.bin": (12203119470, "01c95e5f2588e324ebc05268231c92b3b1d9e83e5163723100ac958226c12a24"),
    "scheduler/config.json": (549, None),
    "vae/config.json": (1217, None),
    "vae/pytorch_model.bin": (1311145138, "379995ca170d8a899019125f389ba8692b2e35625ff64ddc3fdaa8c9302ac340"),
    "License.txt": (16966, None),
    "Notice.txt": (16695, None),
}
DINO_FILES = {
    "config.json": (549, None),
    "preprocessor_config.json": (436, None),
    "model.safetensors": (1217522888, "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"),
}


def manifest():
    result = []
    for repo, rev, root, files in [
        ("tencent/Hunyuan3D-Omni", OMNI_REV, OMNI_ROOT, OMNI_FILES),
        ("facebook/dinov2-large", DINO_REV, DINO_ROOT, DINO_FILES),
    ]:
        for name, (size, sha) in files.items():
            result.append(dict(repo=repo, revision=rev, relative_path=name, path=str(root / name),
                               bytes=size, sha256=sha,
                               url=f"https://huggingface.co/{repo}/resolve/{rev}/{name}"))
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def start(proxy, endpoint, connections, signed_urls):
    SOURCE.mkdir(parents=True, exist_ok=True)
    rows = manifest()
    signed = json.loads(Path(signed_urls).read_text()) if signed_urls else {}
    input_lines = []
    for row in rows:
        target = Path(row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == row["bytes"] and not Path(str(target) + ".aria2").exists():
            continue
        url = row["url"].replace("https://huggingface.co", endpoint.rstrip("/"))
        url = signed.get(row["path"], url)
        input_lines.extend([url, f" dir={target.parent}", f" out={target.name}"])
    input_path = SOURCE / "aria2_minimal_inputs.txt"
    input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    write_json(SOURCE / "download_manifest.json", {"files": rows, "total_bytes": sum(r["bytes"] for r in rows)})
    args = ["aria2c", f"--input-file={input_path}", "--continue=true", "--auto-file-renaming=false",
            "--allow-overwrite=false", "--file-allocation=none", "--max-concurrent-downloads=3",
            f"--split={connections}", f"--max-connection-per-server={connections}", "--min-split-size=10M",
            "--summary-interval=10", "--console-log-level=notice", "--download-result=full",
            "--retry-wait=5", "--max-tries=15", "--timeout=30", "--connect-timeout=10",
            f"--all-proxy={proxy}", "--auto-save-interval=10", f"--log={SOURCE / 'aria2_download.log'}", "--log-level=notice"]
    with (SOURCE / "aria2_console.log").open("ab", buffering=0) as stream:
        proc = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    status = dict(pid=proc.pid, started_epoch=time.time(), args=args, endpoint=endpoint,
                  total_bytes=sum(r["bytes"] for r in rows))
    write_json(SOURCE / "download_started.json", status)
    with (SOURCE / "download_attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(status) + "\n")
    print(json.dumps(status))


def verify():
    started = time.monotonic()
    rows = manifest()
    all_ok = True
    for row in rows:
        target = Path(row["path"])
        actual_size = target.stat().st_size if target.is_file() else -1
        row["actual_bytes"] = actual_size
        row["size_ok"] = actual_size == row["bytes"]
        actual_sha = None
        if row["size_ok"]:
            h = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    h.update(block)
            actual_sha = h.hexdigest()
        row["actual_sha256"] = actual_sha
        row["sha_ok"] = actual_sha == row["sha256"] if row["sha256"] else actual_sha is not None
        all_ok = all_ok and row["size_ok"] and row["sha_ok"]
    result = dict(all_ok=all_ok, files=rows, total_expected_bytes=sum(r["bytes"] for r in rows),
                  verification_wall_seconds=time.monotonic() - started, checked_epoch=time.time())
    write_json(SOURCE / "download_verified.json", result)
    print(json.dumps(result))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "verify"])
    ap.add_argument("--proxy", default="http://127.0.0.1:45957")
    ap.add_argument("--endpoint", default="https://huggingface.co")
    ap.add_argument("--connections", type=int, default=4)
    ap.add_argument("--signed-urls")
    ns = ap.parse_args()
    start(ns.proxy, ns.endpoint, ns.connections, ns.signed_urls) if ns.action == "start" else verify()
