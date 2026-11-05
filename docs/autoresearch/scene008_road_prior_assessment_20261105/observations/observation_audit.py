"""Read cached road evidence on CPU; never instantiate a renderer or decoder.

Only the output directory is written. In particular no torch import, checkpoint
load, original-source update, dataset extraction, training, GPU probe or shutdown.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np


DEFAULT_BASE = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
FULL_NAME = "r15_gn11_fullclip_cpu_20261026"
EXPANDED_NAME = "shadow_a5_a37_199_20261103"
CURRENT_NAME = "scene008_native_road1k_20261105_r1"


def metadata(path):
    """Only open NPY headers inside a zip, including for large source RGB."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if not item.filename.endswith(".npy"):
                continue
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            result[item.filename[:-4]] = dict(shape=list(shape), dtype=str(dtype),
                bytes=item.file_size, compressed_bytes=item.compress_size)
    return result


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def count(mask):
    return int(np.count_nonzero(mask))


def histogram(values):
    values = np.asarray(values)
    cuts = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 199)]
    return {f"{lo}" if lo == hi else f"{lo}-{hi}": count((values >= lo) & (values <= hi))
            for lo, hi in cuts}


def examples(arrays, selected, valid, source_raw, maximum=8):
    # Deterministic spread, not selected by visual outcome.
    ids = np.flatnonzero(selected)
    if len(ids) > maximum:
        ids = ids[np.linspace(0, len(ids) - 1, maximum).astype(int)]
    result = []
    for unit in ids:
        rec = dict(unit_id=int(unit), local_xyz=arrays["local_xyz"][unit].tolist(),
                   key=arrays["keys"][unit].tolist(),
                   qualified_source_raw=source_raw[valid[:, unit]].tolist())
        for name in ("world_xyz", "model_xyz"):
            if name in arrays.files:
                rec[name] = arrays[name][unit].tolist()
        result.append(rec)
    return result


def coordinate_audit(arrays, meta):
    local = arrays["local_xyz"]
    model = transform(transform(local, np.asarray(meta["coordinate_anchor"])),
                      np.asarray(meta["world_to_model"]))
    rec = dict(local_bounds=[local.min(0).tolist(), local.max(0).tolist()],
               coordinate_contract="local anchor metres -> coordinate_anchor -> world metres -> world_to_model -> model units",
               meter_to_model=float(meta["meter_to_model"]),
               model_from_local_maxabs=float(np.abs(model-arrays["model_xyz"]).max()))
    if "world_xyz" in arrays.files:
        rec["world_from_local_maxabs_m"] = float(np.abs(
            transform(local, np.asarray(meta["coordinate_anchor"]))-arrays["world_xyz"]).max())
    return rec


def audit_full(path, meta):
    with np.load(path, allow_pickle=False) as a:
        valid = a["source_valid"].astype(bool)
        raw = a["source_raw"]
        supported = a["surface_supported"].astype(bool)
        eligible = a["eligible"].astype(bool)
        sources = valid.sum(0)
        output = supported & eligible
        any_sample = sources > 0
        rec = dict(path=str(path), denominator_cells=len(sources),
                   cell_m=0.2, source_frames=len(raw), source_raw=raw.tolist(),
                   surface_supported_cells=count(supported),
                   qualified_geometry_rgb_sample_cells=count(any_sample),
                   qualified_sample_records=count(valid),
                   source_count_histogram=histogram(sources),
                   eligible_cells=count(eligible), renderer_selected_cells=count(output),
                   supported_no_qualified_sample_cells=count(supported & ~any_sample),
                   unsupported_cells=count(~supported),
                   sample_exists_but_not_renderer_selected=count(any_sample & ~output),
                   sample_ge2_but_not_renderer_selected=count((sources >= 2) & ~output),
                   sample_ge3_but_not_renderer_selected=count((sources >= 3) & ~output),
                   samples_but_rejected_examples=examples(a, any_sample & ~output, valid, raw),
                   coordinates=coordinate_audit(a, meta),
                   warning="A source_valid=False combines many rejection causes; zero accepted samples is not proof of never visible.")
        return rec


def audit_expanded(path, r4path, currentpath, meta):
    with np.load(path, allow_pickle=False) as a, np.load(r4path, allow_pickle=False) as r:
        valid = a["source_valid"].astype(bool)
        source_raw = a["source_raw"]
        supported = a["surface_supported"].astype(bool)
        domain = r["domain"].astype(bool)
        any_sample = valid.any(0)
        active = r["active"].astype(bool)
        measured = r["measured"].astype(bool)
        inferred = r["inferred"].astype(bool)
        agree = r["source_agree"].astype(bool)
        out = dict(path=str(path), r4_reference_path=str(r4path),
                   denominator_cells=len(supported), cell_m=float(a["cell_m"]),
                   source_frames=len(source_raw), exact_199=np.array_equal(source_raw, np.arange(199)),
                   coordinates=coordinate_audit(a, meta),
                   all_cells=dict(supported=count(supported), sample_exists=count(any_sample),
                                  sample_records=count(valid), source_count_histogram=histogram(valid.sum(0))),
                   domain=dict(cells=count(domain), supported=count(domain & supported),
                               sample_exists=count(domain & any_sample), no_accepted_sample=count(domain & ~any_sample),
                               measured=count(domain & measured), inferred=count(domain & inferred),
                               active=count(domain & active), inactive=count(domain & ~active),
                               sample_exists_but_inactive=count(domain & any_sample & ~active),
                               agreeing_sample_exists_but_inactive=count(domain & agree.any(0) & ~active),
                               sample_ge3_but_inactive=count(domain & (valid.sum(0) >= 3) & ~active),
                               sample_records=count(valid[:, domain]), agreeing_sample_records=count(agree[:, domain])),
                   samples_but_inactive_examples=examples(a, domain & any_sample & ~active, valid, source_raw),
                   unused_measured_outside_domain=count(measured & ~domain),
                   limitation="Measured marks >=3 RGB agreeing geometric sources; it is a clean-road proxy, not independent GT. Inferred is explicitly neighbour interpolation.")
        with np.load(currentpath, allow_pickle=False) as cur:
            parent = cur["parent_coarse_id"].astype(int)
            used = np.zeros(len(supported), bool)
            used[parent] = True
            a37 = r["domain_a37"].astype(bool)
            out["a37_current_envelope_relation"] = dict(
                coarse_a37_cells=count(a37), current_used_coarse_parents=count(used),
                current_used_in_a37=count(used & a37),
                a37_supported_with_sample_not_in_current_envelope=count(a37 & supported & any_sample & ~used),
                a37_measured_not_in_current_envelope=count(a37 & measured & ~used),
                a37_supported_with_sample_but_r4_inactive=count(a37 & supported & any_sample & ~active),
                warning="Omission from the local shadow/repair envelope is intentional scope or marking protection in many cells; not automatically a visual gap.")
            out["a37_measured_not_in_current_envelope_examples"] = examples(
                a, a37 & measured & ~used, valid, source_raw)
        return out


def audit_current(path, meta):
    with np.load(path, allow_pickle=False) as a:
        active = a["active"].astype(bool)
        valid = a["source_valid"].astype(bool)
        agree = a["source_agree"].astype(bool)
        real = a["real_active"].astype(bool)
        inferred = a["inferred"].astype(bool)
        raw = a["source_raw"]
        return dict(path=str(path), denominator_cells=len(active),
                    cell_m=float(a["cell_m"]), source_frames=len(raw),
                    source_raw=raw.tolist(), exact_199=np.array_equal(raw, np.arange(199)),
                    coordinates=coordinate_audit(a, meta), active=count(active), real_active=count(real),
                    inferred=count(inferred), inactive=count(~active),
                    qualified_sample_records=count(valid), agreeing_sample_records=count(agree),
                    no_qualified_sample_cells=count(~valid.any(0)),
                    no_agreeing_sample_cells=count(~agree.any(0)),
                    valid_count_equals_cached=bool(np.array_equal(valid.sum(0), a["source_valid_count"])),
                    agree_count_equals_cached=bool(np.array_equal(agree.sum(0), a["agree_count"])),
                    real_active_equals_ge3_agree=bool(np.array_equal(real, agree.sum(0) >= 3)),
                    inferred_intersects_real=count(inferred & real),
                    inferred_with_geometry_qualified_sample=count(inferred & valid.any(0)),
                    inferred_with_no_qualified_sample=count(inferred & ~valid.any(0)),
                    inferred_with_one_or_two_agree=count(inferred & (agree.sum(0) > 0) & (agree.sum(0) < 3)),
                    inferred_best_source_minus1=count(inferred & (a["best_source_raw"] < 0)),
                    real_best_source_qualified=bool(valid[np.maximum(a["best_source_raw"][real], 0), np.flatnonzero(real)].all()),
                    source_count_histogram=histogram(valid.sum(0)),
                    caution="The denominator is selected 2cm repair-envelope cells. inactive=0 is not evidence that every road location is observed.")


def sensor_inventory(base):
    """Parse four indexed protobuf records; never decode JPEG/range-image data."""
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(base/"nearby_actors_20261015_v1"))
    from waymo_numpy_cpu import protobuf, crc_function
    pb = protobuf()
    crc = crc_function()
    extract = json.loads((base/FULL_NAME/"extraction.json").read_text())
    source = Path(extract["raw_path"])
    rows = []
    with source.open("rb") as stream:
        for raw in (0, 102, 137, 198):
            ledger = extract["frames"][raw]["record"]
            stream.seek(ledger["header_offset"])
            header = stream.read(12)
            length, checksum = struct.unpack("<QI", header)
            assert length == ledger["payload_length"] and crc(header[:8]) == checksum
            payload, trailer = stream.read(length), stream.read(4)
            assert crc(payload) == struct.unpack("<I", trailer)[0]
            frame = pb.Frame.FromString(payload)
            rows.append(dict(raw=raw, timestamp_micros=int(frame.timestamp_micros),
                image_names=[int(x.name) for x in frame.images],
                jpeg_bytes={str(x.name):len(x.image) for x in frame.images},
                camera_calibrations=[dict(name=int(x.name), width=int(x.width), height=int(x.height),
                    intrinsic_length=len(x.intrinsic)) for x in frame.context.camera_calibrations],
                laser_names=[int(x.name) for x in frame.lasers],
                laser_calibration_names=[int(x.name) for x in frame.context.laser_calibrations],
                compressed_returns={str(x.name):[bool(x.ri_return1.range_image_compressed),
                    bool(x.ri_return2.range_image_compressed)] for x in frame.lasers},
                crc_checked=True, JPEG_decoded=False, range_images_decoded=False))
    return dict(raw_path=str(source), sampled_records=rows, original_records_parsed=4,
        all199_sensor_completeness_verified=False,
        current_RGB_cache_camera="FRONT only (camera enum 1, calibration P0)",
        current_LiDAR_reader="Five LiDAR x two returns; TOP pixel-pose compensation, NLZ filtering. Cached per-camera cloud is thinned to nearest returns in FRONT FOV.",
        unknown_visibility_scope="This audit has not searched RGB in other supplied cameras. Never-visible claims across all supplied cameras require checking their 199x5 projections/occlusion evidence.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    meta_path = base / "r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json"
    meta = json.loads(meta_path.read_text())
    paths = dict(
        full199=base/FULL_NAME/"full199_initialization.npz",
        window49=base/FULL_NAME/"window49_initialization.npz",
        expanded=base/EXPANDED_NAME/"inputs/expanded/evidence.npz",
        shadow_r4=base/"shadow_a5_a37_199_20261103_r4/inputs/road_reference.npz",
        current=base/CURRENT_NAME/"inputs/road_reference.npz")
    output = dict(schema="ROAD_OBSERVATION_CACHE_AUDIT_V1", created_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=dict(CPU_only=True, torch_imported=False, original_JPEG_or_range_image_decodes=0,
                               original_protobuf_records_parsed=4,
                               RGB_arrays_decoded=0, training_updates=0, checkpoint_loads=0),
                  files={name:dict(path=str(path), bytes=path.stat().st_size, schema=metadata(path))
                         for name, path in paths.items()},
                  meta_path=str(meta_path), meter_to_model=meta["meter_to_model"],
                  full199=audit_full(paths["full199"], meta),
                  window49=audit_full(paths["window49"], meta),
                  expanded=audit_expanded(paths["expanded"], paths["shadow_r4"], paths["current"], meta),
                  current=audit_current(paths["current"], meta),
                  original_sensor_inventory=sensor_inventory(base))
    frame_dir = base/FULL_NAME/"frames"
    output["cached_frame_inventory"] = dict(
        npz_count=len(list(frame_dir.glob("*.npz"))),
        metadata_json_count=len(list(frame_dir.glob("*.json"))),
        RGB_png_count=len(list((base/FULL_NAME/"rgb").glob("*.png"))),
        sample_frame_schemas={str(raw):metadata(frame_dir/f"{raw:03d}.npz") for raw in (0, 102, 137, 198)},
        projection_available="Per-frame K, w2l, nearest-in-front LiDAR cloud + ground flag, all 3D boxes and conservative 2D guard cached for all199.",
        visibility_missing="No per-query reject reason, triangle/conflict confidence, or dense occlusion proof for every candidate is saved in source_valid. Small CPU reprojection can recompute finite evidence states; sparse empty pixels remain unknown.")
    outpath = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists():
        raise FileExistsError("Use a new output; preserve earlier audit")
    outpath.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output":str(outpath), "full199":{k:output["full199"][k] for k in (
        "denominator_cells", "surface_supported_cells", "renderer_selected_cells", "sample_exists_but_not_renderer_selected")},
        "expanded_domain":output["expanded"]["domain"],
        "current":{k:output["current"][k] for k in (
        "denominator_cells", "active", "real_active", "inferred", "no_qualified_sample_cells")}}, indent=2))


if __name__ == "__main__":
    main()
