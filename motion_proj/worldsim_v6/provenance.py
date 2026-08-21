"""SceneIR v0 的 chunk/actor/primitive provenance overlay。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from motion_proj.worldsim_v6.sceneir import load_sceneir


SCHEMA_VERSION = "worldsim_v6.provenance_field.v0"
MANIFEST_VERSION = "worldsim_v6.provenance_package_manifest.v0"
REQUIRED_FIELDS = (
    "source_type",
    "sensor_support",
    "time_support",
    "view_support",
    "reconstruction_source",
    "generation_source",
)


class ProvenanceError(RuntimeError):
    """provenance overlay 不满足 typed coverage 合同。"""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload["content_sha256"] = "0" * 64
    return hashlib.sha256(_canonical(payload)).hexdigest()


def build_provenance_document(sceneir_package: Path) -> dict[str, Any]:
    sceneir, blobs = load_sceneir(sceneir_package)
    provenance_by_id = {row["id"]: row for row in sceneir["provenance"]}
    support_by_id = {row["id"]: row for row in sceneir["support"]}
    source_types = {row["kind"] for row in provenance_by_id.values()}
    if not source_types <= {"observed", "reconstructed", "generated"}:
        raise ProvenanceError(f"未知 source_type：{sorted(source_types)}")
    all_source_indices = []
    chunk_rows = []
    for chunk in sorted(sceneir["chunks"], key=lambda row: row["id"]):
        identity_ref = chunk["arrays"].get("source_indices")
        if identity_ref is None:
            raise ProvenanceError(f"chunk 缺少 source_indices：{chunk['id']}")
        source_indices = np.asarray(blobs[identity_ref["sha256"]])
        if source_indices.shape != (int(chunk["primitive_count"]),):
            raise ProvenanceError(f"chunk primitive identity shape 漂移：{chunk['id']}")
        if np.unique(source_indices).size != source_indices.size:
            raise ProvenanceError(f"chunk primitive identity 非唯一：{chunk['id']}")
        all_source_indices.append(source_indices.astype(np.int64, copy=False))
        source = provenance_by_id[chunk["provenance_id"]]
        support = support_by_id[chunk["support_id"]]
        observed_times = support["observed_timestamp_us"]
        sensor_unknown = len(sceneir["sensors"]) == 0
        view_ids = support["observed_view_ids"]
        fields = {
            "source_type": {"encoding": "constant", "value": source["kind"]},
            "sensor_support": {
                "encoding": "constant",
                "status": "unknown" if sensor_unknown else "supported",
                "sensor_ids": [] if sensor_unknown else sorted(row["id"] for row in sceneir["sensors"]),
                "reason": "source_sceneir_has_no_sensor" if sensor_unknown else None,
            },
            "time_support": {
                "encoding": "constant_interval",
                "status": "observed_capture_support",
                "start_timestamp_us": int(min(observed_times)),
                "end_timestamp_us": int(max(observed_times)),
                "observed_timestamp_count": len(observed_times),
            },
            "view_support": {
                "encoding": "constant",
                "status": "unknown" if not view_ids else "supported",
                "view_ids": sorted(view_ids),
                "reason": "source_sceneir_has_no_observed_views" if not view_ids else None,
            },
            "reconstruction_source": {
                "encoding": "constant",
                "value": source["id"] if source["kind"] == "reconstructed" else None,
            },
            "generation_source": {
                "encoding": "constant",
                "value": source["id"] if source["kind"] == "generated" else None,
            },
        }
        chunk_rows.append(
            {
                "chunk_id": chunk["id"],
                "actor_id": chunk.get("actor_id"),
                "role": chunk["role"],
                "primitive_count": int(chunk["primitive_count"]),
                "primitive_identity": {
                    "encoding": "sceneir_blob_reference",
                    "field": "source_indices",
                    "sha256": identity_ref["sha256"],
                    "dtype": identity_ref["dtype"],
                    "shape": identity_ref["shape"],
                    "coverage": "all_primitives_in_chunk",
                },
                "fields": fields,
            }
        )
    merged = np.concatenate(all_source_indices)
    unique_count = int(np.unique(merged).size)
    primitive_count = int(merged.size)
    actor_rows = []
    chunk_by_id = {row["chunk_id"]: row for row in chunk_rows}
    for actor in sorted(sceneir["actors"], key=lambda row: row["id"]):
        chunks = [chunk_by_id[chunk_id] for chunk_id in actor["chunk_ids"]]
        field_values = {
            name: [chunk["fields"][name] for chunk in chunks] for name in REQUIRED_FIELDS
        }
        actor_rows.append(
            {
                "actor_id": actor["id"],
                "chunk_ids": sorted(actor["chunk_ids"]),
                "primitive_count": sum(chunk["primitive_count"] for chunk in chunks),
                "fields": {
                    name: values[0]
                    if all(value == values[0] for value in values)
                    else {"encoding": "by_chunk", "values": values}
                    for name, values in field_values.items()
                },
            }
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "content_sha256": "0" * 64,
        "source_sceneir_content_sha256": sceneir["content_sha256"],
        "type_system": {
            "source_type_enum": ["observed", "reconstructed", "generated"],
            "invariant": "observed != reconstructed != generated",
            "unknown_support_is_not_false_support": True,
        },
        "reconstruction_sources": [
            row for row in sceneir["provenance"] if row["kind"] == "reconstructed"
        ],
        "generation_sources": [
            row for row in sceneir["provenance"] if row["kind"] == "generated"
        ],
        "chunks": chunk_rows,
        "actors": actor_rows,
        "coverage": {
            "chunk_total": len(sceneir["chunks"]),
            "chunk_covered": len(chunk_rows),
            "actor_total": len(sceneir["actors"]),
            "actor_covered": len(actor_rows),
            "primitive_total": primitive_count,
            "primitive_covered": sum(row["primitive_count"] for row in chunk_rows),
            "global_primitive_identity_unique": unique_count == primitive_count,
            "global_unique_primitive_identity_count": unique_count,
        },
        "claim_boundary": [
            "reconstructed_primitives_are_not_observed_ground_truth",
            "missing_sensor_or_view_evidence_remains_unknown",
            "broadcast_constant_fields_apply_to_every_referenced_primitive_identity",
        ],
    }
    document["content_sha256"] = _content_hash(document)
    return document


def write_provenance_package(output: Path, document: Mapping[str, Any]) -> None:
    output = output.resolve()
    if output.exists():
        raise ProvenanceError(f"provenance package 已存在：{output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ProvenanceError(f"provenance partial 已存在：{partial}")
    partial.mkdir(parents=True)
    provenance_path = partial / "PROVENANCE.json"
    provenance_path.write_bytes(_canonical(document))
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "provenance_content_sha256": document["content_sha256"],
        "files": {
            "PROVENANCE.json": {
                "bytes": provenance_path.stat().st_size,
                "sha256": _sha256(provenance_path),
            }
        },
    }
    (partial / "MANIFEST.json").write_bytes(_canonical(manifest))
    partial.rename(output)


def verify_provenance_package(package: Path, sceneir_package: Path) -> dict[str, Any]:
    package = package.resolve()
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    document_path = package / "PROVENANCE.json"
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ProvenanceError("provenance manifest schema 漂移")
    record = manifest["files"]["PROVENANCE.json"]
    if document_path.stat().st_size != record["bytes"] or _sha256(document_path) != record["sha256"]:
        raise ProvenanceError("provenance package 文件完整性失败")
    document = json.loads(document_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION or document["content_sha256"] != _content_hash(document):
        raise ProvenanceError("provenance content hash 失败")
    sceneir, blobs = load_sceneir(sceneir_package)
    if sceneir["content_sha256"] != document["source_sceneir_content_sha256"]:
        raise ProvenanceError("source SceneIR 漂移")
    scene_chunk = {row["id"]: row for row in sceneir["chunks"]}
    primitive_covered = 0
    source_type_counts = {name: 0 for name in ("observed", "reconstructed", "generated")}
    for row in document["chunks"]:
        if set(row["fields"]) != set(REQUIRED_FIELDS):
            raise ProvenanceError(f"provenance 字段不完整：{row['chunk_id']}")
        source = scene_chunk[row["chunk_id"]]
        identity = row["primitive_identity"]
        if source["arrays"]["source_indices"]["sha256"] != identity["sha256"]:
            raise ProvenanceError(f"primitive identity 漂移：{row['chunk_id']}")
        if blobs[identity["sha256"]].shape[0] != row["primitive_count"]:
            raise ProvenanceError(f"primitive coverage 漂移：{row['chunk_id']}")
        source_type = row["fields"]["source_type"]["value"]
        if source_type not in source_type_counts:
            raise ProvenanceError("source_type 越出 typed enum")
        source_type_counts[source_type] += row["primitive_count"]
        if source_type != "generated" and row["fields"]["generation_source"]["value"] is not None:
            raise ProvenanceError("非 generated primitive 含 generation source")
        if source_type != "reconstructed" and row["fields"]["reconstruction_source"]["value"] is not None:
            raise ProvenanceError("非 reconstructed primitive 含 reconstruction source")
        primitive_covered += row["primitive_count"]
    coverage = document["coverage"]
    passed = (
        coverage["chunk_total"] == coverage["chunk_covered"] == len(document["chunks"])
        and coverage["actor_total"] == coverage["actor_covered"] == len(document["actors"])
        and coverage["primitive_total"] == coverage["primitive_covered"] == primitive_covered
        and coverage["global_primitive_identity_unique"]
    )
    if not passed:
        raise ProvenanceError("provenance coverage gate 失败")
    return {
        "schema_version": "worldsim_v6.provenance_verification.v1",
        "content_sha256": document["content_sha256"],
        "chunk_count": len(document["chunks"]),
        "actor_count": len(document["actors"]),
        "primitive_count": primitive_covered,
        "source_type_primitive_counts": source_type_counts,
        "required_fields": list(REQUIRED_FIELDS),
        "coverage_passed": True,
        "type_separation_passed": True,
    }
