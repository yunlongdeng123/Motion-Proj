"""Named Waymo calibration and camera selection without CUDA dependencies."""
from pathlib import Path
import numpy as np


def selected_camera_indices(args):
    if args.is_fisheye:
        return [4,5][:args.fisheye_cam_num]
    explicit = getattr(args, "camera_indices", None)
    if explicit is not None:
        indices = [int(i) for i in explicit]
    elif args.long_focus:
        indices = [0]
    else:
        indices = list(range(args.cam_num))
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i > 4 for i in indices):
        raise ValueError("Waymo camera_indices must be unique indices in 0..4")
    return indices


def read_named_camera_calibration(path, camera_indices):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, separator, payload = line.partition(":")
        if not separator or name in values:
            raise ValueError(f"Malformed/duplicate calibration record: {line}")
        values[name] = np.fromstring(payload, sep=" ")
    intrinsics, lidar_to_camera = [], []
    for index in camera_indices:
        keyp, keyt = f"P{index}", f"Tr_velo_to_cam_{index}"
        if keyp not in values or keyt not in values or len(values[keyp]) != 12 or len(values[keyt]) != 12:
            raise ValueError(f"Missing/malformed named camera calibration {keyp}/{keyt}: {path}")
        projection = values[keyp].reshape(3,4)
        transform = np.eye(4)
        transform[:3] = values[keyt].reshape(3,4)
        if not np.isfinite(projection).all() or not np.isfinite(transform).all():
            raise ValueError(f"Non-finite calibration: {path}")
        if projection[0,0] <= 0 or projection[1,1] <= 0:
            raise ValueError(f"Invalid focal length: {path}")
        intrinsics.append(projection[:,:3])
        lidar_to_camera.append(transform)
    return np.stack(intrinsics), np.stack(lidar_to_camera)


def camera_observation_identity(frame_index, frame_name, camera_indices, selected_index):
    count=len(camera_indices)
    uid=frame_index*count+selected_index
    name=frame_name if count==1 else f"{frame_name}_cam{camera_indices[selected_index]}"
    return uid,name
