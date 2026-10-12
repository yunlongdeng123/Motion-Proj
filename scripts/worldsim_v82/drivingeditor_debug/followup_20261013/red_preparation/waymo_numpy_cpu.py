"""纯protobuf/NumPy读取；数学规则对应原2024 Waymo converter与已有SDK。

无torch/tensorflow；五LiDAR、两return、TOPpixelpose、原NLZ过滤。
浮点实现差异由真实控制帧显式审计，不能冒称bitwise相同。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import ctypes
import io
from pathlib import Path
import struct
import sys
import zlib
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
PB_ROOT = Path("/root/autodl-tmp/envs/dggt_data_cpu/lib/python3.10/site-packages")
CRC_LIBRARY = Path("/root/autodl-tmp/drivingeditor_repro/campaign/scene_search_mirror/libpreview_crc32c.so")


def protobuf(pb_root=PB_ROOT):
    sys.path.append(str(pb_root))
    from waymo_open_dataset import dataset_pb2
    if any(name == "torch" or name.startswith("tensorflow") for name in sys.modules):
        raise RuntimeError("CPU raw extraction unexpectedly imported a model framework")
    return dataset_pb2


def crc_function(path=CRC_LIBRARY):
    library = ctypes.CDLL(str(path))
    library.preview_crc32c.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.preview_crc32c.restype = ctypes.c_uint32
    if library.preview_crc32c(b"123456789", 9) != 0xE3069283:
        raise ValueError("Existing CRC32C library known-vector failed")
    def masked(data):
        value = int(library.preview_crc32c(data, len(data)))
        return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xffffffff
    return masked


def records(raw, last_index, crc_path=CRC_LIBRARY):
    crc = crc_function(crc_path)
    with Path(raw).open("rb") as stream:
        for index in range(last_index + 1):
            offset = stream.tell()
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError("Raw TFRecord ended before the declared frame window")
            length, checksum = struct.unpack("<QI", header)
            if length <= 0 or length > 96 * 1024 ** 2 or crc(header[:8]) != checksum:
                raise ValueError("TFRecord length/CRC contract failed")
            payload, trailer = stream.read(length), stream.read(4)
            if len(payload) != length or len(trailer) != 4 or crc(payload) != struct.unpack("<I", trailer)[0]:
                raise ValueError("TFRecord payload CRC failed")
            yield index, payload, {"header_offset": offset, "payload_length": length,
                                   "length_crc32c": "passed", "payload_crc32c": "passed"}


def matrix(compressed, dataset_pb2):
    if not compressed:
        raise ValueError("Required real compressed range image or TOP pose missing")
    value = dataset_pb2.MatrixFloat()
    value.ParseFromString(zlib.decompress(compressed))
    array = np.asarray(value.data, dtype=np.float32).reshape(tuple(value.shape.dims))
    if not np.isfinite(array).all():
        raise ValueError("Range image contains nonfinite values")
    return array


def rotation(roll, pitch, yaw):
    """与SDK transform_utils相同的Rz @ (Ry @ Rx)，保持float32。"""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    one, zero = np.ones_like(roll), np.zeros_like(roll)
    rx = np.stack((np.stack((one, zero, zero), -1), np.stack((zero, cr, -sr), -1), np.stack((zero, sr, cr), -1)), -2)
    ry = np.stack((np.stack((cp, zero, sp), -1), np.stack((zero, one, zero), -1), np.stack((-sp, zero, cp), -1)), -2)
    rz = np.stack((np.stack((cy, -sy, zero), -1), np.stack((sy, cy, zero), -1), np.stack((zero, zero, one), -1)), -2)
    return np.matmul(rz, np.matmul(ry, rx))


def lidar(frame, dataset_pb2):
    calibrations = sorted(frame.context.laser_calibrations, key=lambda item: item.name)
    if [item.name for item in calibrations] != [1, 2, 3, 4, 5]:
        raise ValueError("Original five-LiDAR calibration contract missing")
    lasers = {item.name: item for item in frame.lasers}
    top_pose = matrix(lasers[1].ri_return1.range_image_pose_compressed, dataset_pb2)
    if top_pose.shape[-1] != 6:
        raise ValueError("TOP pixel pose must store roll/pitch/yaw/xyz")
    top_rotation = rotation(top_pose[..., 0], top_pose[..., 1], top_pose[..., 2])
    world_to_vehicle = np.linalg.inv(np.asarray(frame.pose.transform, dtype=np.float32).reshape(4, 4))
    clouds, details = [], []
    for return_index in (0, 1):
        for calibration in calibrations:
            source = lasers[calibration.name].ri_return1 if return_index == 0 else lasers[calibration.name].ri_return2
            image = matrix(source.range_image_compressed, dataset_pb2)
            height, width, channels = image.shape
            if channels < 4:
                raise ValueError("Original range/intensity/elongation/NLZ fields missing")
            if calibration.beam_inclinations:
                inclinations = np.asarray(calibration.beam_inclinations, dtype=np.float32)
            else:
                lo, hi = np.float32(calibration.beam_inclination_min), np.float32(calibration.beam_inclination_max)
                inclinations = ((np.float32(.5) + np.arange(height, dtype=np.float32)) / np.float32(height) * (hi - lo) + lo)
            inclinations = inclinations[::-1]
            if len(inclinations) != height:
                raise ValueError("Beam inclination/height mismatch")
            extrinsic = np.asarray(calibration.extrinsic.transform, dtype=np.float32).reshape(4, 4)
            correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
            ratios = (np.arange(width, 0, -1, dtype=np.float32) - np.float32(.5)) / np.float32(width)
            azimuth = (ratios * np.float32(2.) - np.float32(1.)) * np.float32(np.pi) - correction
            distance = image[..., 0]
            ca, sa, ci, si = np.cos(azimuth)[None], np.sin(azimuth)[None], np.cos(inclinations)[:, None], np.sin(inclinations)[:, None]
            xyz = np.stack((ca * ci * distance, sa * ci * distance, si * distance), -1)
            xyz = np.einsum("kr,ijr->ijk", extrinsic[:3, :3], xyz) + extrinsic[:3, 3]
            if calibration.name == 1:
                if top_pose.shape[:2] != image.shape[:2]:
                    raise ValueError("TOP pixel pose/range dimensions differ")
                xyz = np.einsum("hwij,hwj->hwi", top_rotation, xyz) + top_pose[..., 3:]
                xyz = np.einsum("ij,hwj->hwi", world_to_vehicle[:3, :3], xyz) + world_to_vehicle[:3, 3]
            positive = distance > 0
            valid = positive & (image[..., 3] != np.float32(1.))
            yy, xx = np.where(valid)
            indices = ((return_index * height + yy) * width + xx).astype(np.float32) if calibration.name == 1 else np.full(len(yy), -1, dtype=np.float32)
            cloud = np.column_stack((xyz[valid], image[..., 1][valid], image[..., 2][valid], indices)).astype(np.float32)
            if not np.isfinite(cloud).all():
                raise ValueError("Extracted real LiDAR is not finite")
            clouds.append(cloud)
            details.append({"laser_name": calibration.name, "return_index": return_index, "positive_range_points": int(positive.sum()),
                            "NLZ_excluded": int((positive & ~valid).sum()), "retained_points": len(cloud), "TOP_pixel_motion_compensated": calibration.name == 1})
    return np.concatenate(clouds), details


def calibration_text(frame):
    camera_to_kitti = np.eye(4)
    camera_to_kitti[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    cameras = sorted(frame.context.camera_calibrations, key=lambda item: item.name)
    if [item.name for item in cameras] != [1, 2, 3, 4, 5]:
        raise ValueError("Expected original five-camera calibration")
    projections, transforms = [], []
    for camera in cameras:
        k = camera.intrinsic
        projections.append(np.array([[k[0], 0, k[2], 0], [0, k[1], k[3], 0], [0, 0, 1, 0]]))
        transforms.append((camera_to_kitti @ np.linalg.inv(np.asarray(camera.extrinsic.transform).reshape(4, 4)))[:3])
    fields = [("P%d" % i, value) for i, value in enumerate(projections)] + [("R0_rect", np.eye(3))] + [("Tr_velo_to_cam_%d" % i, value) for i, value in enumerate(transforms)]
    return "\n".join(name + ": " + " ".join("%.12e" % x for x in value.ravel()) for name, value in fields) + "\n"


def front_rgb(frame):
    front = next(item for item in frame.images if item.name == 1)
    with Image.open(io.BytesIO(front.image)) as image:
        return np.array(image.convert("RGB")), front


def save_ply(path, xyz, model_times):
    dtype = np.dtype([(key, "<f4") for key in ("x", "y", "z", "nx", "ny", "nz")] + [(key, "u1") for key in ("red", "green", "blue")] + [("time", "<f4")])
    vertices = np.zeros(len(xyz), dtype=dtype)
    for axis, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, axis]
    for key in ("nx", "ny", "nz", "time"):
        vertices[key] = model_times
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz)
    header += "".join("property %s %s\n" % ("uchar" if key in ("red", "green", "blue") else "float", key) for key in dtype.names) + "end_header\n"
    with Path(path).open("xb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
