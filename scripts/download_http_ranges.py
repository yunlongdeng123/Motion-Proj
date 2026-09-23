#!/usr/bin/env python3
"""用有界且逐块校验的 HTTP Range 请求下载大文件。

适用于无界下载被上游拒绝、或长连接经过代理时不稳定的主机。每个区间先写入
临时文件，字节数与请求严格一致后才写入最终 offset。下载期间保留邻接
``.aria2`` 标记，防止下游把尚未完成的稀疏文件误判为可用资产。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="支持 byte range 的 HTTP(S) URL")
    parser.add_argument("output", type=Path, help="目标路径")
    parser.add_argument("--total-size", type=int, required=True)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument(
        "--request-chunks",
        type=int,
        default=1,
        help="每次 HTTP 请求合并的连续校验块数",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument(
        "--proxy",
        action="append",
        default=[],
        help="传给 curl 的可选 HTTP proxy；可重复指定以分摊区间",
    )
    parser.add_argument(
        "--cache-bust-query-key",
        help="为每个区间/重试附加唯一 query，避开上游错误页缓存",
    )
    return parser.parse_args()


def iter_ranges(start: int, stop: int, chunk_size: int):
    offset = start
    while offset < stop:
        end = min(offset + chunk_size, stop) - 1
        yield offset, end
        offset = end + 1


def group_contiguous_ranges(
    ranges: list[tuple[int, int]], max_chunks: int
) -> list[tuple[tuple[int, int], ...]]:
    groups: list[tuple[tuple[int, int], ...]] = []
    current: list[tuple[int, int]] = []
    for item in ranges:
        if current and (
            len(current) >= max_chunks or item[0] != current[-1][1] + 1
        ):
            groups.append(tuple(current))
            current = []
        current.append(item)
    if current:
        groups.append(tuple(current))
    return groups


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def with_query_value(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append((key, value))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def main() -> int:
    args = parse_args()
    if args.total_size <= 0:
        raise SystemExit("--total-size must be positive")
    if (
        args.chunk_mib <= 0
        or args.request_chunks <= 0
        or args.workers <= 0
        or args.max_attempts <= 0
    ):
        raise SystemExit("chunk size, workers, and attempts must be positive")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(f"{output}.aria2")
    state_dir = Path(f"{output}.ranges")
    meta_path = state_dir / "meta.json"
    chunk_size = args.chunk_mib * 1024 * 1024

    if output.exists() and output.stat().st_size == args.total_size and not marker.exists():
        print(f"already complete: {output} ({args.total_size} bytes)")
        return 0

    state_dir.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {
            "url": args.url,
            "output": str(output),
            "total_size": args.total_size,
            "chunk_size": chunk_size,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise SystemExit(
                    f"resume metadata mismatch for {key}: {meta.get(key)!r} != {value!r}"
                )
        initial_size = int(meta["initial_size"])
    else:
        initial_size = output.stat().st_size if output.exists() else 0
        if initial_size > args.total_size:
            raise SystemExit(
                f"existing output is larger than --total-size: {initial_size}"
            )
        meta = {
            "url": args.url,
            "output": str(output),
            "total_size": args.total_size,
            "chunk_size": chunk_size,
            "initial_size": initial_size,
        }
        write_json_atomic(meta_path, meta)

    marker.write_text(
        json.dumps(
            {
                "state": "range_download_in_progress",
                "state_dir": str(state_dir),
                "total_size": args.total_size,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ranges = list(iter_ranges(initial_size, args.total_size, chunk_size))
    pending = [
        item
        for item in ranges
        if not (state_dir / f"{item[0]:020d}-{item[1]:020d}.done").exists()
    ]
    requests = group_contiguous_ranges(pending, args.request_chunks)
    completed_before = len(ranges) - len(pending)
    print(
        f"resume={initial_size} total={args.total_size} chunk={chunk_size} "
        f"workers={args.workers} proxies={len(args.proxy)} "
        f"pending={len(pending)} requests={len(requests)} done={completed_before}"
    )

    file_descriptor = os.open(output, os.O_CREAT | os.O_RDWR, 0o644)
    progress_lock = threading.Lock()
    completed_now = 0
    downloaded_now = 0
    started_at = time.monotonic()

    def download_range(
        request: tuple[tuple[int, int], ...]
    ) -> tuple[int, int]:
        nonlocal completed_now, downloaded_now
        start, end = request[0][0], request[-1][1]
        expected_size = end - start + 1
        done_paths = [
            state_dir / f"{part_start:020d}-{part_end:020d}.done"
            for part_start, part_end in request
        ]
        last_error = ""
        for attempt in range(1, args.max_attempts + 1):
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f"range-{start}-", dir=state_dir, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                command = [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "120",
                    "--range",
                    f"{start}-{end}",
                    "--output",
                    str(temporary_path),
                ]
                if args.proxy:
                    proxy_index = ((start - initial_size) // chunk_size) % len(args.proxy)
                    command.extend(["--proxy", args.proxy[proxy_index]])
                request_url = args.url
                if args.cache_bust_query_key:
                    request_url = with_query_value(
                        request_url,
                        args.cache_bust_query_key,
                        f"{start}-{attempt}",
                    )
                command.append(request_url)
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                actual_size = temporary_path.stat().st_size
                if result.returncode != 0 or actual_size != expected_size:
                    last_error = (
                        f"curl={result.returncode} size={actual_size}/{expected_size} "
                        f"stderr={result.stderr.strip()!r}"
                    )
                    time.sleep(min(30, attempt * 2))
                    continue

                payload = temporary_path.read_bytes()
                written = 0
                while written < expected_size:
                    count = os.pwrite(
                        file_descriptor, payload[written:], start + written
                    )
                    if count <= 0:
                        raise OSError("os.pwrite returned no progress")
                    written += count
                os.fsync(file_descriptor)
                for done_path in done_paths:
                    done_path.touch()
                temporary_path.unlink(missing_ok=True)
                with progress_lock:
                    completed_now += len(request)
                    downloaded_now += expected_size
                    total_done = completed_before + completed_now
                    if completed_now == 1 or total_done % 16 == 0 or total_done == len(ranges):
                        elapsed = max(time.monotonic() - started_at, 0.001)
                        rate_mib = downloaded_now / elapsed / (1024 * 1024)
                        print(
                            f"progress={total_done}/{len(ranges)} "
                            f"session_rate={rate_mib:.2f} MiB/s",
                            flush=True,
                        )
                return start, end
            except Exception as error:  # noqa: BLE001 - I/O 异常必须保留重试
                last_error = repr(error)
                time.sleep(min(30, attempt * 2))
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"range {start}-{end} failed after {args.max_attempts} attempts: {last_error}"
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_range, request) for request in requests]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        os.ftruncate(file_descriptor, args.total_size)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)

    missing = [
        item
        for item in ranges
        if not (state_dir / f"{item[0]:020d}-{item[1]:020d}.done").exists()
    ]
    if missing:
        raise SystemExit(f"download stopped with {len(missing)} ranges incomplete")
    if output.stat().st_size != args.total_size:
        raise SystemExit(
            f"final size mismatch: {output.stat().st_size} != {args.total_size}"
        )

    marker.unlink()
    shutil.rmtree(state_dir)
    print(f"download complete: {output} ({args.total_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
