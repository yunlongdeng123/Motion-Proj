#!/usr/bin/env python3
"""Plan-only by default; download only the reviewed public training index 29.

Use the frozen search module's lock, persistent body ledger, limits and HTTP
Range implementation. Raw input lives outside its 512 MiB thumbnail directory.
No GPU, training, cleanup, search-state reset or frame-window identity verdict.
"""

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

TRAINING_INDEX = 29
EXPECTED_SEGMENT = "10526338824408452410_5714_660_5734_660"
EXPECTED_SIZE = 1160423074
EXPECTED_REVISION = "8a6fccd2b022077308c167a4d3c6f646d6e5e65b"
VERSION = "red_raw_index029_cpu_20261013_v1"
DEFAULT_PROJECT = Path("/root/autodl-tmp/drivingeditor_repro")
DEFAULT_FOLLOWUP = DEFAULT_PROJECT / "campaign/followup_scene_prep_20261013"


def load_frozen_module(path):
    if not path.is_file():
        raise RuntimeError("frozen search module is missing")
    spec = importlib.util.spec_from_file_location("red_raw_frozen_search", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.REV != EXPECTED_REVISION or module.VERSION != "red_scene_cpu_20261013_v1":
        raise RuntimeError("frozen search module revision or owner mismatch")
    return module


def read_contract(module, search_root):
    owner_path = search_root / module.OWNER_FILE
    state_path = search_root / "status.json"
    if not owner_path.is_file() or not state_path.is_file():
        raise module.IntegrityError("existing owned search ledger is required; never create or reset it")
    if json.loads(owner_path.read_text(encoding="utf-8")).get("owner") != module.VERSION:
        raise module.IntegrityError("search owner marker mismatch")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    contract = state.get("contract", {})
    if state.get("owner") != module.VERSION:
        raise module.IntegrityError("search state owner mismatch")
    required = {"network_cap_bytes": 5 * module.GIB,
                "disk_free_floor_bytes": 40 * module.GIB,
                "max_output_bytes": 512 * module.MIB,
                "max_single_record_bytes": 16 * module.MIB, "workers": 1}
    if any(contract.get(key) != value for key, value in required.items()):
        raise module.IntegrityError("shared 5 GiB / 40 GiB floor / thumbnail contract changed")
    for key in ("network_reserved_bytes", "network_received_bytes"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise module.IntegrityError("shared ledger counters are invalid; never reset them")
    if state["network_received_bytes"] > state["network_reserved_bytes"]:
        raise module.IntegrityError("shared received bytes exceed reservations")
    for key in ("request_timeout_seconds", "attempts_per_range", "wall_time_limit_seconds_per_invocation"):
        if not isinstance(contract.get(key), (int, float)) or contract[key] <= 0:
            raise module.IntegrityError("shared request contract is missing or invalid")
    if type(contract["attempts_per_range"]) is not int:
        raise module.IntegrityError("shared attempts_per_range must be an integer")
    return state, contract


def expected_identity(module, row, full_path):
    return {"owner": VERSION, "global_training_index": TRAINING_INDEX,
            "segment": row["segment"], "public_repository": module.REPO,
            "pinned_revision": module.REV, "public_source_path": row["path"],
            "local_raw_path": str(full_path), "raw_expected_size_bytes": row["size"],
            "raw_expected_full_sha256": row["lfs"]["oid"],
            "raw_frame_window_confirmed": False, "gpu_used": False}


def inspect_plan(args, module):
    row = module.load_index(args.source_index)[TRAINING_INDEX]
    if row["segment"] != EXPECTED_SEGMENT or row["size"] != EXPECTED_SIZE:
        raise module.IntegrityError("pinned index29 segment or expected full size mismatch")
    expected_source = "individual_files/training/segment-" + EXPECTED_SEGMENT + "_with_camera_labels.tfrecord"
    if row["path"] != expected_source:
        raise module.IntegrityError("pinned index29 public source path mismatch")
    project = args.project_root.resolve()
    followup = args.followup_root.resolve()
    search_root = args.search_root.resolve()
    campaign = project / "campaign"
    if followup == campaign or not followup.is_relative_to(campaign):
        raise module.IntegrityError("followup-root must be an independent project campaign descendant")
    if search_root == campaign or not search_root.is_relative_to(campaign):
        raise module.IntegrityError("search-root must be inside project campaign")
    raw_dir = followup / "inputs/raw_scene0029"
    if raw_dir.resolve() != raw_dir:
        raise module.IntegrityError("raw input directory must not be redirected by symlinks")
    if raw_dir.is_relative_to(search_root) or search_root.is_relative_to(raw_dir):
        raise module.IntegrityError("raw input must remain outside the capped thumbnail search-root")
    protected = [project / name for name in ("data", "runs", "DrivingEditor")]
    protected += [campaign / name for name in ("scene_search", "scene_search_mirror", "hybrid30k_20261012", "next_scene30k_20261012")]
    if any(raw_dir == path or raw_dir.is_relative_to(path) or path.is_relative_to(raw_dir) for path in protected):
        raise module.IntegrityError("raw input output overlaps preserved inputs, caches or matrix assets")
    full_path = raw_dir / Path(row["path"]).name
    part_path = full_path.with_suffix(full_path.suffix + ".part")
    receipt_path = raw_dir / "download_receipt.json"
    if any(path.is_symlink() for path in (full_path, part_path, receipt_path)):
        raise module.IntegrityError("raw input files or receipt must not be symlinks")
    identity = expected_identity(module, row, full_path)
    if part_path.exists() and not part_path.is_file():
        raise module.IntegrityError("partial raw input is not a regular file")
    part_bytes = part_path.stat().st_size if part_path.exists() else 0
    if not 0 <= part_bytes <= row["size"]:
        raise module.IntegrityError("partial raw input size exceeds pinned full size")
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in identity.items()):
            raise module.IntegrityError("partial raw receipt provenance mismatch; never overwrite it")
    elif part_path.exists():
        raise module.IntegrityError("partial raw input lacks its matching provenance receipt")
    state, contract = read_contract(module, search_root)
    remaining = row["size"] - part_bytes
    disk_path = raw_dir
    while not disk_path.exists():
        disk_path = disk_path.parent
    if os.stat(disk_path).st_dev != os.stat(search_root).st_dev:
        raise module.IntegrityError("raw input and shared matrix/search reservation must be on the same filesystem")
    free_bytes = shutil.disk_usage(disk_path).free
    blocks = []
    if full_path.exists():
        blocks.append("full raw already exists; refuse copy, overwrite or re-download")
    if state["network_reserved_bytes"] + remaining > contract["network_cap_bytes"]:
        blocks.append("remaining one-pass raw body does not fit the cumulative 5 GiB ledger")
    if free_bytes < contract["disk_free_floor_bytes"] + remaining:
        blocks.append("remaining raw input would consume the retained 40 GiB matrix/save floor")
    thumbnail_bytes = sum(p.stat().st_size for p in search_root.rglob("*") if p.is_file())
    if thumbnail_bytes + contract["max_single_record_bytes"] > contract["max_output_bytes"]:
        blocks.append("shared thumbnail reserve reached")
    plan = {**identity, "status": "blocked" if blocks else "plan_ready",
            "mode": "download" if args.download else "plan_only", "blocks": blocks,
            "part_path": str(part_path), "receipt_path": str(receipt_path),
            "progress_path": str(raw_dir / "download_status.json"),
            "search_lock_path": str(search_root / "search.lock"),
            "shared_ledger_path": str(search_root / "status.json"),
            "shared_search_status_preserved": state.get("status"),
            "network_reserved_bytes_before": state["network_reserved_bytes"],
            "network_received_bytes_before": state["network_received_bytes"],
            "network_cap_bytes": contract["network_cap_bytes"],
            "partial_raw_bytes": part_bytes, "remaining_raw_bytes": remaining,
            "chunk_bytes": args.chunk_mib * module.MIB,
            "remaining_one_pass_chunks": (remaining + args.chunk_mib * module.MIB - 1) // (args.chunk_mib * module.MIB),
            "disk_free_bytes": free_bytes,
            "retained_matrix_save_floor_bytes": contract["disk_free_floor_bytes"],
            "required_free_bytes_before_download": contract["disk_free_floor_bytes"] + remaining,
            "thumbnail_output_bytes_excludes_raw": thumbnail_bytes,
            "candidate_reviewed": args.candidate_reviewed,
            "download_requires": ["--download", "--candidate-reviewed", "shared search.lock idle"],
            "lock_checked": False, "network_requests_this_plan": 0}
    return plan, row, state, contract, raw_dir, full_path, part_path, receipt_path


def atomic_rename_no_overwrite(module, part_path, full_path):
    """Linux renameat2 keeps promotion atomic and refuses an existing full asset."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise module.IntegrityError("atomic no-overwrite renameat2 is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(part_path), -100, os.fsencode(full_path), 1) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise module.IntegrityError("full raw already exists; atomic rename refuses overwrite")
        raise OSError(number, "atomic raw promotion failed")


def download(args, module):
    if not args.candidate_reviewed:
        raise module.IntegrityError("--download requires explicit --candidate-reviewed; frame window remains unconfirmed")
    import fcntl
    lock_path = args.search_root.resolve() / "search.lock"
    if not lock_path.is_file():
        raise module.IntegrityError("existing shared search.lock is required")
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise module.StopSearch("shared search.lock busy; no raw download started")
        plan, row, state, contract, raw_dir, full_path, part_path, receipt_path = inspect_plan(args, module)
        if plan["blocks"]:
            raise module.StopSearch("; ".join(plan["blocks"]))

        class RawSearch(module.Search):
            # Do not call Search.__init__: it rewrites search status/inventory.
            def check_limits(self):
                super().check_limits()
                durable = part_path.stat().st_size if part_path.exists() else 0
                remaining = row["size"] - durable
                if shutil.disk_usage(raw_dir).free < self.contract["disk_free_floor_bytes"] + remaining:
                    raise module.StopSearch("raw input would consume retained 40 GiB matrix/save floor")

        worker = RawSearch.__new__(RawSearch)
        worker.root = args.search_root.resolve()
        worker.started = time.monotonic()
        worker.args = SimpleNamespace(max_seconds=contract["wall_time_limit_seconds_per_invocation"],
                                      request_timeout_seconds=contract["request_timeout_seconds"],
                                      attempts_per_range=contract["attempts_per_range"])
        worker.state_path = worker.root / "status.json"
        worker.state, worker.contract, worker.lock = state, contract, lock
        raw_dir.mkdir(parents=True, exist_ok=True)
        receipt = {**expected_identity(module, row, full_path), "status": "downloading",
                   "candidate_reviewed": True, "started_utc": module.utc(),
                   "part_path": str(part_path), "shared_ledger_path": str(worker.state_path),
                   "search_lock_path": str(lock_path),
                   "network_reserved_bytes_before": state["network_reserved_bytes"],
                   "network_received_bytes_before": state["network_received_bytes"],
                   "resumed_from_bytes": plan["partial_raw_bytes"],
                   "raw_size_bytes": plan["partial_raw_bytes"],
                   "size_verified": False, "full_sha256_verified": False}
        if receipt_path.exists():
            prior_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["first_started_utc"] = prior_receipt.get("first_started_utc", prior_receipt.get("started_utc"))
        else:
            receipt["first_started_utc"] = receipt["started_utc"]
        progress_path = raw_dir / "download_status.json"
        if progress_path.is_symlink():
            raise module.IntegrityError("download progress must not be a symlink")

        def save_progress(run_status, received_offset, error=None):
            progress = {"owner": VERSION, "pid": os.getpid(), "run_status": run_status,
                        "updated_utc": module.utc(), "started_utc": receipt["started_utc"],
                        "global_training_index": TRAINING_INDEX,
                        "expected_bytes": row["size"], "received_offset": received_offset,
                        "local_raw_path": str(full_path), "part_path": str(part_path),
                        "receipt_path": str(receipt_path), "shared_ledger_path": str(worker.state_path),
                        "shared_lock_path": str(lock_path), "gpu_used": False,
                        "candidate_reviewed": True, "raw_frame_window_confirmed": False,
                        "network_reserved_bytes": state["network_reserved_bytes"],
                        "network_received_bytes": state["network_received_bytes"]}
            if error is not None:
                progress["error"] = module.safe_error(error)
            module.atomic_json(progress_path, progress)

        module.atomic_json(receipt_path, receipt)
        state["raw_download"] = {"status": "downloading", "pid": os.getpid(),
                                 "global_training_index": TRAINING_INDEX,
                                 "receipt_path": str(receipt_path), "progress_path": str(progress_path),
                                 "candidate_reviewed": True, "raw_frame_window_confirmed": False}
        worker.save()
        save_progress("downloading", plan["partial_raw_bytes"])
        try:
            mode = "ab" if part_path.exists() else "xb"
            with part_path.open(mode) as stream:
                offset = stream.tell()
                while offset < row["size"]:
                    end = min(offset + args.chunk_mib * module.MIB, row["size"]) - 1
                    payload, metadata = worker.range_request(row, offset, end)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                    offset = stream.tell()
                    receipt.update(raw_size_bytes=offset, last_verified_range=metadata["content_range"],
                                   updated_utc=module.utc())
                    module.atomic_json(receipt_path, receipt)
                    save_progress("downloading", offset)
            if part_path.stat().st_size != row["size"]:
                raise module.IntegrityError("full raw byte count differs from pinned expected size")
            save_progress("verifying_full_sha256", row["size"])
            digest = hashlib.sha256()
            with part_path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * module.MIB), b""):
                    digest.update(block)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != row["lfs"]["oid"]:
                raise module.IntegrityError("full raw SHA256 differs from pinned LFS OID; keep partial asset for inspection")
            atomic_rename_no_overwrite(module, part_path, full_path)
            receipt.update(status="complete", completed_utc=module.utc(), raw_size_bytes=full_path.stat().st_size,
                           raw_full_sha256=actual_sha256, size_verified=True, full_sha256_verified=True,
                           network_reserved_bytes_after=state["network_reserved_bytes"],
                           network_received_bytes_after=state["network_received_bytes"])
            state["raw_download"].update(status="complete", completed_utc=receipt["completed_utc"])
            module.atomic_json(receipt_path, receipt)
            worker.save()
            save_progress("complete", row["size"])
            return receipt
        except BaseException as error:
            receipt.update(status="failed_resumable", updated_utc=module.utc(), error=module.safe_error(error),
                           raw_size_bytes=part_path.stat().st_size if part_path.exists() else 0,
                           network_reserved_bytes_after=state["network_reserved_bytes"],
                           network_received_bytes_after=state["network_received_bytes"])
            state["raw_download"].update(status="failed_resumable", error=module.safe_error(error))
            module.atomic_json(receipt_path, receipt)
            worker.save()
            save_progress("failed_resumable", receipt["raw_size_bytes"], error)
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-index", type=int, choices=[TRAINING_INDEX], required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--followup-root", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--search-module", type=Path, default=Path(__file__).with_name("search_red_scene_cpu.py"))
    parser.add_argument("--source-index", type=Path, default=Path("/root/autodl-tmp/dggt/.preparation/hf_waymo_training_tree.json"))
    parser.add_argument("--chunk-mib", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--candidate-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.search_root is None:
        args.search_root = args.followup_root / "red_search_cpu"
    return args


def main(argv=None):
    args = parse_args(argv)
    module = None
    try:
        module = load_frozen_module(args.search_module)
        result = download(args, module) if args.download else inspect_plan(args, module)[0]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2 if result.get("blocks") else 0
    except Exception as error:
        # Never emit traceback, signed redirects, URL query strings or headers.
        detail = module.safe_error(error) if module is not None else {"type": type(error).__name__}
        print(json.dumps({"status": "blocked_or_failed", "gpu_used": False, "error": detail}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
