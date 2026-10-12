"""CPU-only integration contract. This module does not implement M-detector.

All choices here concern transport, lineage, coordinate checks and ID mapping.
Algorithm labels must come from the pinned author backend, never GT boxes/masks.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Dict, Iterable, Optional, Sequence, Tuple


class ContractError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class FrameKey:
    segment_sha256: str
    raw_record_index: int
    frame_timestamp_micros: int

    def validate(self):
        if len(self.segment_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.segment_sha256):
            raise ContractError("segment identity must be a lowercase content SHA256")
        if type(self.raw_record_index) is not int or type(self.frame_timestamp_micros) is not int:
            raise ContractError("raw record index and microsecond stamp must be integers")
        if self.raw_record_index < 0 or self.frame_timestamp_micros < 0:
            raise ContractError("raw record index and integer microsecond stamp must be nonnegative")


@dataclass(frozen=True)
class PointId:
    frame: FrameKey
    return_index: int  # 0=first return, 1=second return
    laser_name: int  # Waymo LaserName enum 1..5, TOP=1
    range_row: int
    range_col: int

    def validate(self):
        self.frame.validate()
        if any(type(v) is not int for v in (self.return_index, self.laser_name, self.range_row, self.range_col)):
            raise ContractError("return, laser and range pixels must be integers")
        if self.return_index not in (0, 1) or self.laser_name not in (1, 2, 3, 4, 5):
            raise ContractError("unexpected Waymo return or laser")
        if min(self.range_row, self.range_col) < 0:
            raise ContractError("range image pixel must be nonnegative")


@dataclass(frozen=True)
class BodyPoint:
    point_id: PointId
    bin_row: int
    xyz_m: Tuple[float, float, float]  # undistorted vehicle body at reference pose
    intensity: float
    elongation: float
    top_range_flat_index: int  # .bin column 6; NOT a stamp; -1 for non-TOP


@dataclass(frozen=True)
class Cloud:
    key: FrameKey
    reference_time_micros: int
    points: Tuple[BodyPoint, ...]
    coordinate_receipt: str
    lineage_receipt: str
    time_binding_receipt: str
    top_range_image_hw: Tuple[int, int]
    coordinate_frame: str = "waymo_vehicle_body_reference_metres"


@dataclass(frozen=True)
class Pose:
    key: FrameKey
    reference_time_micros: int
    body_to_world_row_major: Tuple[float, ...]  # native Frame.pose, 4x4
    source_receipt: str


@dataclass(frozen=True)
class ReadyFrame:
    cloud: Cloud
    pose: Pose
    detector_time_s: float


def _check_pose(pose: Pose):
    pose.key.validate()
    a = pose.body_to_world_row_major
    if len(a) != 16 or not all(isfinite(v) for v in a):
        raise ContractError("pose must be a finite row-major 4x4 transform")
    if not pose.source_receipt:
        raise ContractError("pose source is unproven")
    # These tolerances validate an SE(3) input; they are not detector thresholds.
    if max(abs(a[12 + i] - (1.0 if i == 3 else 0.0)) for i in range(4)) > 1e-8:
        raise ContractError("invalid homogeneous transform row")
    r = (a[0:3], a[4:7], a[8:11])
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i] * r[k][j] for k in range(3)) - float(i == j)) > 1e-6:
                raise ContractError("pose rotation is not orthonormal")
    det = (r[0][0] * (r[1][1]*r[2][2] - r[1][2]*r[2][1])
           - r[0][1] * (r[1][0]*r[2][2] - r[1][2]*r[2][0])
           + r[0][2] * (r[1][0]*r[2][1] - r[1][1]*r[2][0]))
    if abs(det - 1.0) > 1e-6:
        raise ContractError("pose rotation changes handedness")


def _check_cloud(cloud: Cloud):
    cloud.key.validate()
    if cloud.coordinate_frame != "waymo_vehicle_body_reference_metres":
        raise ContractError("detector requires native metres in the body reference frame")
    if not all((cloud.coordinate_receipt, cloud.lineage_receipt, cloud.time_binding_receipt)):
        raise ContractError("coordinate, lineage and author-time binding must all be evidenced")
    if not (0 < len(cloud.points) <= 200000):
        raise ContractError("empty scan or author points_num_perframe capacity exceeded; do not truncate")
    if len(cloud.top_range_image_hw) != 2 or any(type(v) is not int or v <= 0 for v in cloud.top_range_image_hw):
        raise ContractError("native TOP range-image H/W are required")
    h, w = cloud.top_range_image_hw
    ids = set()
    previous = None
    for i, p in enumerate(cloud.points):
        p.point_id.validate()
        if p.point_id.frame != cloud.key or p.bin_row != i:
            raise ContractError("raw point frame or .bin index correspondence broken")
        if p.point_id in ids:
            raise ContractError("duplicate raw point ID")
        ids.add(p.point_id)
        # The existing converter's exact canonical order, no resorting here.
        order = (p.point_id.return_index, p.point_id.laser_name,
                 p.point_id.range_row, p.point_id.range_col)
        if previous is not None and order <= previous:
            raise ContractError("return/laser/row-major native point order changed")
        previous = order
        if len(p.xyz_m) != 3 or not all(isfinite(v) for v in (*p.xyz_m, p.intensity, p.elongation)):
            raise ContractError("non-finite native point")
        if p.point_id.laser_name != 1 and p.top_range_flat_index != -1:
            raise ContractError("non-TOP point has a fake TOP lineage index")
        if p.point_id.laser_name == 1:
            pid = p.point_id
            if pid.range_row >= h or pid.range_col >= w or p.top_range_flat_index != (pid.return_index*h+pid.range_row)*w+pid.range_col:
                raise ContractError("TOP return offset/pixel-to-flat-index correspondence broken")


def bind(cloud: Cloud, pose: Pose, origin_micros: int) -> ReadyFrame:
    _check_cloud(cloud)
    _check_pose(pose)
    if any(type(v) is not int for v in (cloud.reference_time_micros, pose.reference_time_micros, origin_micros)):
        raise ContractError("reference times and origin must be integer microseconds")
    if cloud.key != pose.key or cloud.reference_time_micros != pose.reference_time_micros:
        raise ContractError("cloud and pose do not share the exact raw frame and reference stamp")
    if cloud.reference_time_micros < origin_micros:
        raise ContractError("reference time precedes the segment-local origin")
    # Subtract as integers before conversion, retaining microsecond intervals.
    return ReadyFrame(cloud, pose, (cloud.reference_time_micros - origin_micros) / 1e6)


class ExactSynchronizer:
    """Join by identity/stamp, then drain in an explicit raw-record manifest order.

    Arrival ordering does not determine pairing. No nearest-time matching,
    independent FIFO queues, interpolation, dropped scans or implicit retries.
    """
    def __init__(self, expected: Sequence[FrameKey], origin_micros: int):
        self.expected = tuple(expected)
        if not self.expected or len(set(self.expected)) != len(self.expected):
            raise ContractError("a unique ordered raw-frame manifest is required")
        for key in self.expected:
            key.validate()
        if any(a.segment_sha256 != b.segment_sha256 or
               b.raw_record_index != a.raw_record_index + 1 or
               b.frame_timestamp_micros <= a.frame_timestamp_micros
               for a, b in zip(self.expected, self.expected[1:])):
            raise ContractError("manifest must be contiguous chronological records of one segment")
        self.origin = origin_micros
        self.clouds: Dict[FrameKey, Cloud] = {}
        self.poses: Dict[FrameKey, Pose] = {}
        self.position = 0
        self.previous_reference = None
        self._expected_set = set(self.expected)

    def add_cloud(self, cloud: Cloud):
        self._insert(cloud.key, cloud, self.clouds)

    def add_pose(self, pose: Pose):
        self._insert(pose.key, pose, self.poses)

    def _insert(self, key, value, queue):
        if key not in self._expected_set:
            raise ContractError("message identity/stamp absent from the frozen raw-frame manifest")
        if key in queue or key in self.expected[:self.position]:
            raise ContractError("duplicate cloud/pose or already processed raw frame")
        queue[key] = value

    def pop_ready(self) -> Optional[ReadyFrame]:
        if self.position == len(self.expected):
            return None
        key = self.expected[self.position]
        if key not in self.clouds or key not in self.poses:
            return None
        frame = bind(self.clouds[key], self.poses[key], self.origin)
        ref = frame.cloud.reference_time_micros
        if self.previous_reference is not None and ref <= self.previous_reference:
            raise ContractError("detector reference times must be strictly increasing")
        del self.clouds[key]
        del self.poses[key]
        self.previous_reference = ref
        self.position += 1
        return frame

    def finish(self):
        if self.position != len(self.expected) or self.clouds or self.poses:
            raise ContractError("unprocessed, missing or unmatched raw frames")


@dataclass(frozen=True)
class Assignment:
    point_id: PointId
    bin_row: int
    origin_event_tag: int
    cluster_tag: int
    final_author_state: int
    author_binary_label: int
    partition: str


def map_author_output(frame: ReadyFrame, origin: Sequence[int], cluster: Sequence[int],
                      final_states: Sequence[int]) -> Tuple[Assignment, ...]:
    """Waymo1 cluster_coupled=true: expose all tags and every point's ID.

    Author .label serializes cluster tag==1 as 251 and everything else as 9.
    Our separate invalid/other partitions prevent label 9 becoming fake static.
    States: STATIC=0, CASE1=1, CASE2=2, CASE3=3, SELF=4, UNCERTAIN=5, INVALID=6.
    """
    n = len(frame.cloud.points)
    if any(len(x) != n for x in (origin, cluster, final_states)):
        raise ContractError("author output vector length differs from unchanged input order")
    output = []
    for p, event, tag, state in zip(frame.cloud.points, origin, cluster, final_states):
        if event not in (0, 1, 2) or tag not in (-1, 0, 1) or state not in range(7):
            raise ContractError("unknown author tag/state; do not guess")
        if state == 6 or tag == -1:
            partition = "invalid"
        elif tag == 1:
            partition = "dynamic"
        elif state == 0:
            partition = "static"
        else:
            partition = "other"
        output.append(Assignment(p.point_id, p.bin_row, event, tag, state,
                                 251 if tag == 1 else 9, partition))
    return tuple(output)


def body_to_world(xyz_m: Sequence[float], pose: Pose) -> Tuple[float, float, float]:
    """Contract transform only; M-detector's native backend still computes its own."""
    _check_pose(pose)
    a = pose.body_to_world_row_major
    return tuple(sum(a[4*i+j] * xyz_m[j] for j in range(3)) + a[4*i+3]
                 for i in range(3))
