"""Read-only CPU metadata probe; never imports a renderer or checkpoint."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import struct
import sys
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
FULL = ROOT / "r15_gn11_fullclip_cpu_20261026"
sys.path.insert(0, str(ROOT / "nearby_actors_20261015_v1"))
from waymo_numpy_cpu import protobuf, crc_function


def main():
    extraction = json.loads((FULL / "extraction.json").read_text())
    bundle = json.loads((ROOT / "r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json").read_text())
    print(json.dumps({"extraction_keys": list(extraction), "first_record": extraction["frames"][0],
                      "last_record": extraction["frames"][-1], "raw_path": extraction["raw_path"],
                      "anchor": bundle["coordinate_anchor"]}))
    for path in (FULL / "full199_ground_pool.npz", FULL / "frames/137.npz"):
        with np.load(path, allow_pickle=False) as a:
            rec = {"path": str(path), "arrays": {k: [list(a[k].shape), str(a[k].dtype)] for k in a.files}}
            if "local_xyz" in a.files:
                rec["bounds"] = [a["local_xyz"].min(0).tolist(), a["local_xyz"].max(0).tolist()]
            print(json.dumps(rec))
    crc = crc_function()
    pb = protobuf()
    with Path(extraction["raw_path"]).open("rb") as source:
        row = extraction["frames"][137]
        source.seek(row["record"]["header_offset"])
        header = source.read(12)
        length, checksum = struct.unpack("<QI", header)
        assert crc(header[:8]) == checksum
        payload, trailer = source.read(length), source.read(4)
        assert crc(payload) == struct.unpack("<I", trailer)[0]
        frame = pb.Frame.FromString(payload)
        image = next(x for x in frame.images if x.name == 1)
        cal = next(x for x in frame.context.camera_calibrations if x.name == 1)
        print(json.dumps({"raw": 137, "image_fields": [d.name for d, v in image.ListFields()],
                          "calibration_fields": [d.name for d, v in cal.ListFields()],
                          "intrinsic": list(cal.intrinsic), "native_hw": [cal.height, cal.width],
                          "front_image_pose": list(image.pose.transform),
                          "frame_pose": list(frame.pose.transform),
                          "image_velocity_fields": [d.name for d, v in image.velocity.ListFields()]}))


if __name__ == "__main__":
    main()
