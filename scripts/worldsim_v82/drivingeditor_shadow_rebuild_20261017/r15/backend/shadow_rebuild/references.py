"""几何改变后重新投影真实训练观测；不把父点颜色无限复制给细分后的纹理。"""
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from .data import clean_reference


@torch.no_grad()
def refresh(field, dataset):
    rows = field.road.nonzero().flatten()
    xyz = field.xyz[rows].detach().cpu()
    samples = np.zeros((len(rows), len(dataset.train_indices), 3), np.float32)
    observed = np.zeros((len(rows), len(dataset.train_indices)), bool)
    previous_device = dataset.device
    dataset.device = "cpu"
    try:
        from .math3d import linear_to_srgb
        for j, frame_index in enumerate(dataset.train_indices):
            frame = dataset.frame(frame_index)
            uv, depth = frame.camera.project(xyz)
            pixels = torch.floor(uv+.5).long().numpy()
            z = depth.numpy()
            h, w = frame.depth.shape
            good = (z > .1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            ids = np.flatnonzero(good)
            yy, xx = pixels[ids, 1], pixels[ids, 0]
            measured = frame.depth.numpy()[yy, xx]
            guard = binary_dilation(frame.masks.any(-1).numpy(), iterations=2)
            if frame.mask_valid is not None:guard|=~frame.mask_valid.any(-1).numpy()
            okay = (measured > 0) & (np.abs(measured-z[ids]) < .2) & ~guard[yy, xx] & frame.road.numpy()[yy, xx]
            ids, yy, xx = ids[okay], yy[okay], xx[okay]
            rgb = linear_to_srgb(frame.rgb).numpy()
            samples[ids, j], observed[ids, j] = rgb[yy, xx], True
    finally:
        dataset.device = previous_device
    color, confidence = clean_reference(samples, observed)
    field.reference_rgb[rows] = torch.as_tensor(color, device=field.xyz.device)
    field.reference_confidence[rows] = torch.as_tensor(confidence, device=field.xyz.device)
    return dict(points=len(rows), valid_reference_points=int((confidence > 0).sum()),
                source="reprojected train RGB + measured LiDAR depth; no test colors; no parent-color copying")
