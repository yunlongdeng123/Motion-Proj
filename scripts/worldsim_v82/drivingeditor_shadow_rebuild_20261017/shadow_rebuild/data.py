"""只读真实测量；clean-road 是训练帧重投影的保守伪参考，不冒充无影 GT。"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from .raster import Camera
from .math3d import srgb_to_linear


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_seed(seed, actors):
    n = len(seed["xyz"])
    for name, shape in {"xyz": (n, 3), "rgb": (n, 3), "observed_rgb": (n, 3), "scales": (n, 3), "quaternion": (n, 4),
                        "owner": (n,), "road": (n,), "reference_rgb": (n, 3),
                        "reference_confidence": (n,), "neighbor": (n,)}.items():
        if name not in seed or seed[name].shape != shape or not np.isfinite(seed[name]).all():
            raise ValueError(f"无效 seed.{name}")
    if n < 1 or (seed["scales"] <= 0).any() or (np.linalg.norm(seed["quaternion"], axis=1) < .5).any():
        raise ValueError("真实种子为空或几何无效")
    if seed["owner"].dtype.kind not in "iu" or seed["neighbor"].dtype.kind not in "iu":
        raise ValueError("owner/neighbor 必须为整数")
    if seed["road"].dtype != bool:
        raise ValueError("road 必须明确 bool")
    if (seed["owner"] < -1).any() or (seed["owner"] >= actors).any() or (seed["road"] & (seed["owner"] >= 0)).any():
        raise ValueError("道路与 Actor 身份重叠或越界")
    if (seed["neighbor"] < 0).any() or (seed["neighbor"] >= n).any():
        raise ValueError("neighbor 越界")
    if (seed["owner"][seed["neighbor"]] != seed["owner"]).any():
        raise ValueError("法线邻居不得跨实例")
    for name in ("rgb", "observed_rgb", "reference_rgb", "reference_confidence"):
        if (seed[name] < 0).any() or (seed[name] > 1).any():
            raise ValueError(f"{name} 必须为 0..1")
    if (seed["reference_confidence"][~seed["road"]] != 0).any():
        raise ValueError("干净道路参考不能监督车身/非道路")


@dataclass
class Frame:
    frame_id: str
    index: int
    split: str
    camera: Camera
    rgb: torch.Tensor
    masks: torch.Tensor
    mask_known: torch.Tensor
    road: torch.Tensor
    depth: torch.Tensor
    poses: torch.Tensor
    valid: torch.Tensor


class Dataset:
    def __init__(self, root, device="cpu", allow_partial_masks=False, synthetic=False):
        self.root = Path(root)
        self.manifest = read_json(self.root/"manifest.json")
        m = self.manifest
        if m.get("schema") != "instance_surface_measurements_v3" or m.get("units") != "meters":
            raise ValueError("要求物理米坐标的测量数据集")
        if m.get("learned_checkpoint_used") is not False:
            raise ValueError("不允许以 GN11/训练 checkpoint 初始化本分支")
        if m.get("synthetic_fixture", False) and not synthetic:
            raise ValueError("合成 fixture 只能显式用于 CPU 测试")
        if m.get("reference_train_only") is not True:
            raise ValueError("干净道路参考必须只来自 train 帧")
        self.actor_ids = tuple(m["actor_ids"])
        if not self.actor_ids or len(set(self.actor_ids)) != len(self.actor_ids):
            raise ValueError("实例身份缺失/重复")
        self.seed = dict(np.load(self.root/"seed.npz", allow_pickle=False))
        validate_seed(self.seed, len(self.actor_ids))
        self.device, self.allow_partial_masks = device, allow_partial_masks
        self.rows = m["frames"]
        self.train_indices = [i for i, r in enumerate(self.rows) if r["split"] == "train"]
        self.test_indices = [i for i, r in enumerate(self.rows) if r["split"] == "test"]
        if len(self.train_indices) < 2 or len({r["frame_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("至少需要两个独立训练时刻与唯一帧 ID")
        with np.load(self.root/"trajectory.npz", allow_pickle=False) as trajectory:
            self.poses = trajectory["poses"]
            self.valid = trajectory["valid"]
        expected = (len(self.rows), len(self.actor_ids))
        if self.poses.shape != (*expected, 4, 4) or self.valid.shape != expected or self.valid.dtype != bool:
            raise ValueError("轨迹形状/有效性类型无效")
        p = self.poses[self.valid]
        if not np.isfinite(p).all() or not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("轨迹不是有效齐次变换")
        r = p[:, :3, :3]
        if not np.allclose(r.swapaxes(-1, -2)@r, np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(r), 1, atol=2e-4):
            raise ValueError("轨迹旋转不刚性")

    def frame(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(self.root/row["rgb"]).convert("RGB"), dtype=np.float32)/255
        h, w = image.shape[:2]
        with np.load(self.root/row["supervision"], allow_pickle=False) as sup:
            a = {key: sup[key] for key in ("masks", "mask_known", "road", "depth")}
        k = len(self.actor_ids)
        if a["masks"].shape != (h, w, k) or a["mask_known"].shape != (k,) or a["road"].shape != (h, w) or a["depth"].shape != (h, w):
            raise ValueError(f"监督尺寸不符: {row['frame_id']}")
        if a["mask_known"].dtype != bool or a["masks"].dtype != bool or a["road"].dtype != bool:
            raise ValueError("mask/known/road 必须明确 bool")
        if np.any(a["masks"].sum(-1) > 1) or np.any(a["road"] & a["masks"].any(-1)):
            raise ValueError("真实可见实例标签/道路重叠")
        if not np.isfinite(a["depth"]).all() or (a["depth"] < 0).any():
            raise ValueError("稀疏深度必须为非负有限相机 Z")
        if not self.allow_partial_masks and not a["mask_known"].all():
            raise ValueError(f"{row['frame_id']}: 真实逐实例 body mask 未齐全；不能将未知零标签当背景")
        t = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype, device=self.device)
        camera = Camera(row["intrinsic"], row["world_from_camera"], h, w, self.device)
        return Frame(row["frame_id"], index, row["split"], camera, srgb_to_linear(t(image)),
                     t(a["masks"]), t(a["mask_known"]), t(a["road"]), t(a["depth"], torch.float32),
                     t(self.poses[index], torch.float32), t(self.valid[index]))


def clean_reference(samples, valid, min_views=3, min_bright_views=2):
    """跨时刻同一测量点：要求亮簇在多个训练帧重复，而非取一帧最大值。

    暗簇可能是投影阴影，亮簇也可能反光；这是可消融先验。无足够观测时
    confidence=0，不使用均值/灰色补洞，不生成不存在的干净真值。
    """
    n, f, _ = samples.shape
    ref = np.zeros((n, 3), dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    for i in range(n):
        values = samples[i, valid[i]]
        if len(values) < min_views:
            continue
        luminance = values@np.array([.2126, .7152, .0722])
        threshold = np.quantile(luminance, .90)
        bright = values[luminance >= threshold-.015]
        if len(bright) < min_bright_views:
            continue
        estimate = np.median(bright, 0)
        spread = np.median(np.abs(bright-estimate))
        # 亮簇色度不一致说明反射/错误重投影，不将其当干净路面。
        chroma = bright/(bright.sum(-1, keepdims=True)+1e-5)
        if np.max(np.std(chroma, 0)) > .05 or spread > .04:
            continue
        ref[i], confidence[i] = estimate, min(1, len(bright)/5)*np.exp(-spread/.025)
    return ref, confidence
