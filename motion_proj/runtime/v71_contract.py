"""V7.1 证据索引与 fail-closed run contract。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic import atomic_write_json
from .fingerprint import file_fingerprint, sha256_json

TERMINAL_MARKERS = ("COMPLETE", "FAILED", "REJECTED", "BLOCKED")
TERMINAL_STATUS = {
    "COMPLETE": "completed",
    "FAILED": "failed",
    "REJECTED": "rejected",
    "BLOCKED": "blocked",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_REQUIRED_FILES = (
    "manifest.json",
    "resolved.yaml",
    "fingerprints/code.json",
    "fingerprints/environment.json",
    "fingerprints/data.json",
    "fingerprints/third_party.json",
    "fingerprints/checkpoints.json",
    "proposal_bank.json",
    "actor_registry.json",
    "world_state_manifest.json",
    "render_request_manifest.json",
    "artifact_manifest.json",
    "metrics.jsonl",
    "summary.json",
)
DEFAULT_REQUIRED_DIRECTORIES = ("artifacts", "logs")

MANIFEST_REQUIRED_FIELDS = (
    "task_id",
    "run_id",
    "parent_run_id",
    "command",
    "plan_version",
    "code_commit",
    "code_dirty",
    "dirty_diff_hash",
    "config_fingerprint",
    "data_fingerprint",
    "split_fingerprint",
    "proposal_fingerprint",
    "third_party_commit",
    "checkpoint_hashes",
    "scene_ids",
    "actor_ids",
    "camera_ids",
    "time_range",
    "seed",
    "environment",
    "cuda",
    "gpu",
    "world_state_schema_version",
    "canonicalization_version",
    "world_state_hash",
    "render_request_hash",
    "artifact_set_hash",
    "safety_geometry_version",
    "observation_evidence_version",
    "render_support_version",
    "certificate_version",
    "scenario_effect_version",
    "provenance_version",
    "recovery_policy",
    "started_at",
    "ended_at",
    "exit_reason",
    "terminal_status",
)
HASH_FIELDS = ("world_state_hash", "render_request_hash", "artifact_set_hash")

RETROSPECTIVE_MISSING_FIELDS = (
    {"field": "manifest.json", "status": "missing"},
    {"field": "resolved.yaml", "status": "missing"},
    {"field": "terminal_marker", "status": "missing"},
    {"field": "run_start_commit", "status": "unknown_not_inferred"},
    {"field": "run_start_dirty_fingerprint", "status": "unknown_not_inferred"},
    {"field": "seed", "status": "unknown_not_inferred"},
    {"field": "config_fingerprint", "status": "missing"},
    {"field": "data_fingerprint", "status": "missing"},
    {"field": "world_state_hash", "status": "missing"},
    {"field": "render_request_hash", "status": "missing"},
    {"field": "artifact_set_hash", "status": "missing"},
)


class RunContractError(RuntimeError):
    """run 目录违反 V7.1 合同时抛出。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise RunContractError(f"无法读取合法 JSON: {path}") from exc


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise RunContractError(f"{name} 必须是 64 位小写 SHA256")


def generate_run_id(
    task_id: str,
    split: str,
    seed: int,
    config_fingerprint: str,
    *,
    now: datetime | None = None,
) -> str:
    """生成不可复用的 V7.1 run ID。"""
    _require_sha256("config_fingerprint", config_fingerprint)
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def clean(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        if not normalized:
            raise ValueError("run ID 字段规范化后为空")
        return normalized

    return (
        f"v71_{clean(task_id)}__{clean(split)}__s{int(seed)}__"
        f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}__{config_fingerprint[:8]}"
    )


def compute_artifact_set_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """按 relative_path 排序后计算 artifact set hash。"""
    normalized = [dict(row) for row in rows]
    normalized.sort(key=lambda row: str(row.get("relative_path", "")))
    return _canonical_json_hash(normalized)


class V71RunContract:
    """创建、终结并验证一个 V7.1 run。"""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        required_files: Sequence[str] = DEFAULT_REQUIRED_FILES,
        required_directories: Sequence[str] = DEFAULT_REQUIRED_DIRECTORIES,
    ):
        self.run_dir = Path(run_dir)
        self.required_files = tuple(required_files)
        self.required_directories = tuple(required_directories)

    @classmethod
    def initialize(
        cls,
        run_root: str | os.PathLike[str],
        run_id: str,
        manifest: Mapping[str, Any],
        **kwargs: Any,
    ) -> "V71RunContract":
        root = Path(run_root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise RunContractError(f"run ID 已存在，禁止复用: {run_id}") from exc
        (run_dir / "fingerprints").mkdir()
        (run_dir / "artifacts").mkdir()
        (run_dir / "logs").mkdir()
        data = dict(manifest)
        if data.get("run_id") != run_id:
            raise RunContractError("manifest.run_id 与目录 run ID 不一致")
        atomic_write_json(str(run_dir / "manifest.json"), data)
        return cls(run_dir, **kwargs)

    def _marker_paths(self) -> list[Path]:
        return [self.run_dir / marker for marker in TERMINAL_MARKERS if (self.run_dir / marker).is_file()]

    def _validate_required_paths(self) -> None:
        missing_files = [path for path in self.required_files if not (self.run_dir / path).is_file()]
        missing_dirs = [
            path for path in self.required_directories if not (self.run_dir / path).is_dir()
        ]
        if missing_files or missing_dirs:
            raise RunContractError(
                f"缺少必需 artifact: files={missing_files}, directories={missing_dirs}"
            )

    def _validate_manifest(self) -> dict[str, Any]:
        manifest = _load_json(self.run_dir / "manifest.json")
        if not isinstance(manifest, dict):
            raise RunContractError("manifest.json 必须是 object")
        missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in manifest]
        if missing:
            raise RunContractError(f"manifest 缺少字段: {missing}")
        if manifest["run_id"] != self.run_dir.name:
            raise RunContractError("manifest.run_id 与目录名不一致")
        for field in HASH_FIELDS:
            _require_sha256(field, manifest[field])
        return manifest

    def _validate_jsonl(self) -> None:
        path = self.run_dir / "metrics.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows = [line for line in handle if line.strip()]
        if not rows:
            raise RunContractError("metrics.jsonl 不得为空")
        for line_number, line in enumerate(rows, start=1):
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise RunContractError(f"metrics.jsonl 第 {line_number} 行非法") from exc
            if not isinstance(value, dict):
                raise RunContractError(f"metrics.jsonl 第 {line_number} 行必须是 object")

    def _artifact_path(self, relative_path: str) -> Path:
        candidate = (self.run_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise RunContractError(f"artifact 路径越界: {relative_path}") from exc
        return candidate

    def _validate_artifacts(self, manifest: Mapping[str, Any]) -> None:
        artifact_manifest = _load_json(self.run_dir / "artifact_manifest.json")
        if not isinstance(artifact_manifest, dict):
            raise RunContractError("artifact_manifest.json 必须是 object")
        rows = artifact_manifest.get("artifacts")
        if not isinstance(rows, list) or not rows:
            raise RunContractError("artifact_manifest.artifacts 必须是非空数组")
        for field in HASH_FIELDS:
            _require_sha256(field, artifact_manifest.get(field))
            if artifact_manifest[field] != manifest[field]:
                raise RunContractError(f"artifact manifest 的 {field} 与 run manifest 不一致")

        required_row_fields = {
            "relative_path",
            "world_state_hash",
            "render_request_hash",
            "artifact_hash",
            "size_bytes",
            "media_type",
        }
        for row in rows:
            if not isinstance(row, dict):
                raise RunContractError("artifact row 必须是 object")
            missing = sorted(required_row_fields - set(row))
            if missing:
                raise RunContractError(f"artifact row 缺少字段: {missing}")
            _require_sha256("artifact_hash", row["artifact_hash"])
            if row["world_state_hash"] != manifest["world_state_hash"]:
                raise RunContractError("artifact world_state_hash 不一致")
            if row["render_request_hash"] != manifest["render_request_hash"]:
                raise RunContractError("artifact render_request_hash 不一致")
            path = self._artifact_path(str(row["relative_path"]))
            if not path.is_file():
                raise RunContractError(f"artifact 文件缺失: {row['relative_path']}")
            if path.stat().st_size != row["size_bytes"]:
                raise RunContractError(f"artifact size 不一致: {row['relative_path']}")
            if file_fingerprint(str(path)) != row["artifact_hash"]:
                raise RunContractError(f"artifact bytes hash 不一致: {row['relative_path']}")

        computed = compute_artifact_set_hash(rows)
        if computed != artifact_manifest["artifact_set_hash"]:
            raise RunContractError("artifact_set_hash 与 artifact rows 不一致")

    def _validate_hash_manifests(self, manifest: Mapping[str, Any]) -> None:
        world = _load_json(self.run_dir / "world_state_manifest.json")
        render = _load_json(self.run_dir / "render_request_manifest.json")
        if not isinstance(world, dict) or world.get("world_state_hash") != manifest["world_state_hash"]:
            raise RunContractError("world_state_manifest hash 不一致")
        if not isinstance(render, dict):
            raise RunContractError("render_request_manifest.json 必须是 object")
        if render.get("world_state_hash") != manifest["world_state_hash"]:
            raise RunContractError("render request 引用的 world_state_hash 不一致")
        if render.get("render_request_hash") != manifest["render_request_hash"]:
            raise RunContractError("render_request_manifest hash 不一致")

    def validate_complete_payload(self) -> dict[str, Any]:
        self._validate_required_paths()
        manifest = self._validate_manifest()
        summary = _load_json(self.run_dir / "summary.json")
        if not isinstance(summary, dict):
            raise RunContractError("summary.json 必须是 object")
        if not (self.run_dir / "resolved.yaml").read_text(encoding="utf-8").strip():
            raise RunContractError("resolved.yaml 不得为空")
        self._validate_jsonl()
        self._validate_hash_manifests(manifest)
        self._validate_artifacts(manifest)
        return manifest

    def finalize(self, marker: str, *, exit_reason: str) -> None:
        if marker not in TERMINAL_MARKERS:
            raise ValueError(f"未知 terminal marker: {marker}")
        existing = self._marker_paths()
        if existing:
            raise RunContractError(f"run 已存在 terminal marker: {[path.name for path in existing]}")
        if marker == "COMPLETE":
            manifest = self.validate_complete_payload()
        else:
            manifest = self._validate_manifest()
        manifest["ended_at"] = utc_now()
        manifest["exit_reason"] = exit_reason
        manifest["terminal_status"] = marker
        manifest["status"] = TERMINAL_STATUS[marker]
        atomic_write_json(str(self.run_dir / "manifest.json"), manifest)
        marker_path = self.run_dir / marker
        with marker_path.open("x", encoding="utf-8") as handle:
            handle.write(f"{exit_reason}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.validate()

    def validate(self) -> str:
        markers = self._marker_paths()
        if len(markers) != 1:
            raise RunContractError(
                f"terminal marker 必须恰好一个，实际为 {[path.name for path in markers]}"
            )
        marker = markers[0].name
        manifest = self._validate_manifest()
        if manifest.get("terminal_status") != marker:
            raise RunContractError("manifest terminal_status 与 marker 不一致")
        if manifest.get("status") != TERMINAL_STATUS[marker]:
            raise RunContractError("manifest status 与 marker 不一致")
        if not manifest.get("ended_at") or not manifest.get("exit_reason"):
            raise RunContractError("terminal run 缺少 ended_at 或 exit_reason")
        if marker == "COMPLETE":
            self.validate_complete_payload()
        return marker


def validate_optional_branch_not_triggered(
    parent_summary_path: str | os.PathLike[str],
    branch_run_root: str | os.PathLike[str],
) -> None:
    """验证条件分支未触发时没有实例化 run。"""
    summary = _load_json(Path(parent_summary_path))
    if not isinstance(summary, dict):
        raise RunContractError("父 summary 必须是 object")
    if summary.get("h2_generation_branch") != "not_triggered":
        raise RunContractError("父 summary 未记录 h2_generation_branch=not_triggered")
    reason = summary.get("trigger_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise RunContractError("not_triggered 必须给出 trigger_reason")
    branch_root = Path(branch_run_root)
    if branch_root.exists():
        raise RunContractError("optional branch 未触发时禁止创建 run 目录")


def _inventory_root(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_fingerprint(str(path)),
            }
        )
    return {
        "root": str(root),
        "exists": True,
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "file_manifest_hash": _canonical_json_hash(files),
        "files": files,
        "formal_contract_missing": [dict(row) for row in RETROSPECTIVE_MISSING_FIELDS],
    }


def build_retrospective_evidence_index(
    output_path: str | os.PathLike[str],
    evidence_roots: Mapping[str, str | os.PathLike[str]],
    *,
    index_code_state: Mapping[str, Any],
) -> dict[str, Any]:
    """逐文件索引 V7 retrospective evidence，不回填未知 provenance。"""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"证据索引已存在，拒绝覆盖: {output}")
    stages = {
        task_id: {
            "evidence_mode": "retrospective",
            **_inventory_root(Path(root)),
        }
        for task_id, root in evidence_roots.items()
    }
    index = {
        "schema_version": 1,
        "task_id": "V7-EV-10",
        "plan_version": "V7.1",
        "generated_at": utc_now(),
        "index_code_state": dict(index_code_state),
        "policy": {
            "old_evidence_mutated": False,
            "unknown_provenance_inferred": False,
            "terminal_markers_backfilled": False,
        },
        "stages": stages,
        "totals": {
            "stage_count": len(stages),
            "file_count": sum(stage["file_count"] for stage in stages.values()),
            "total_size_bytes": sum(stage["total_size_bytes"] for stage in stages.values()),
        },
    }
    atomic_write_json(str(output), index)
    return index


def smoke_hash(seed: str, payload: Any) -> str:
    """为 EV-10 contract smoke 生成明确的测试 hash。"""
    return sha256_json({"ev10_contract_smoke": seed, "payload": payload})
