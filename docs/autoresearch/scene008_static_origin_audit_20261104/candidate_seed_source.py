"""CPU-only candidate input comparison; does not establish byte identity to B10k input."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
CANDIDATE = Path(r"C:/Users/dengyunlong/Documents/Codex/2026-10-17/c-users-dengyunlong-documents-codex-2026/work/identity_completion/renders/renders_old3/input.ply")
TARGET = ROOT / "b10k_snapshot/points3d_dynamic_0.ply"
EXPECTED = 591672


def read_ply(path):
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
             "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete PLY header")
            header.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        offset = stream.tell()
    assert "format binary_little_endian 1.0" in header
    count = int(next(line for line in header if line.startswith("element vertex ")).split()[-1])
    properties = [line.split()[1:] for line in header if line.startswith("property ")]
    dtype = np.dtype([(name, types[kind]) for kind, name in properties])
    data = np.memmap(path, dtype=dtype, offset=offset, shape=(count,), mode="r")
    xyz = np.stack([data[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    return data, xyz, header


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    candidate, xyz, candidate_header = read_ply(CANDIDATE)
    target, target_xyz, target_header = read_ply(TARGET)
    distances, _ = cKDTree(target_xyz).query(xyz, k=1, eps=0, distance_upper_bound=0.1, workers=2)
    keep = distances >= 0.1
    count = int(keep.sum())
    time = np.asarray(candidate["nx"], dtype=np.float64)
    record = {
        "candidate": str(CANDIDATE),
        "candidate_sha256": digest(CANDIDATE),
        "target_exact_run_ply": str(TARGET),
        "target_sha256": digest(TARGET),
        "candidate_rows": len(xyz),
        "target_rows": len(target_xyz),
        "rule": "cKDTree(target_xyz).query(candidate_xyz,k=1,eps=0,distance_upper_bound=0.1); retain distance>=0.1; no time condition",
        "removed_rows": int((~keep).sum()),
        "retained_rows": count,
        "expected_exact_run_initial_static_rows": EXPECTED,
        "count_matches": count == EXPECTED,
        "source_order_preserved": True,
        "candidate_header": candidate_header,
        "target_header": target_header,
        "candidate_time_source": "nx (matching the original loader convention)",
        "candidate_nx_equals_time": bool(np.array_equal(candidate["nx"], candidate["time"])) if "time" in candidate.dtype.names else None,
        "candidate_timestamp_count": int(len(np.unique(time))),
        "retained_by_timestamp": [{"time": float(t), "rows": int(np.count_nonzero(keep & (time == t)))} for t in np.unique(time)],
        "claim_limit": "Cardinality match is a candidate consistency check only. Original B10k points3d.ply bytes/hash/order were not available, so this does not establish byte-identical input or exact lineage-root coordinates.",
    }
    if count == EXPECTED:
        destination = ROOT / "candidate_filtered_seed.npz"
        np.savez_compressed(destination, xyz_world=xyz[keep], time=time[keep],
                            original_row_indices=np.flatnonzero(keep),
                            candidate_sha256=np.array(record["candidate_sha256"]),
                            target_sha256=np.array(record["target_sha256"]),
                            status=np.array("CANDIDATE_CARDINALITY_MATCH_NOT_INPUT_IDENTITY"))
        record["derived_candidate_npz"] = str(destination)
        record["derived_candidate_npz_sha256"] = digest(destination)
        record["derived_arrays"] = ["xyz_world", "time", "original_row_indices"]
    destination = ROOT / "candidate_audit.json"
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("candidate_rows", "target_rows", "removed_rows", "retained_rows", "count_matches", "candidate_sha256")}))


if __name__ == "__main__":
    main()
