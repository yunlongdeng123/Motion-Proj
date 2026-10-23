"""冻结的五身份及其他车辆像素域；不改变 V1 动态掩码或高斯归属。"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

ACTORS = (5, 26, 18, 6, 37)
VEHICLE_POLICY = dict(actors=list(ACTORS), edge_radius_px=3,
    morphology='7x7 square, erosion border_value=0',
    loss='mean nonempty identities separately for interior and edge; each category weight 1',
    other_vehicles='union all vehicle/native proposal masks minus selected five labels',
    target='current frozen V1 composite RGB; never hidden road RGB truth')


def read_mask(path):
    if str(path).endswith('.npz'):
        with np.load(path, allow_pickle=False) as z:
            image = Image.fromarray(z['masks'].astype(bool).any(0))
    else:
        image = Image.open(path)
    value = np.asarray(image.resize((480, 320), Image.Resampling.NEAREST))
    return value.any(2) if value.ndim == 3 else value


def vehicle_domains(labels, all_vehicles):
    if labels.shape != all_vehicles.shape or labels.ndim != 2 or not np.isin(labels, range(6)).all():
        raise ValueError('车身 label 必须为已绑定的 0..5 身份图')
    masks = [labels == i+1 for i in range(5)] + [all_vehicles & (labels == 0)]
    kernel = np.ones((7, 7), bool)
    bodies = np.stack([binary_erosion(m, structure=kernel, border_value=0) for m in masks])
    edges = np.stack([binary_dilation(m, structure=kernel) & ~b for m, b in zip(masks, bodies)])
    return bodies, edges


def load_vehicle_domains(frame):
    labels = read_mask(frame['label_path'])
    union = np.zeros_like(labels, bool)
    for path in frame['occluder_masks']['paths']:
        union |= read_mask(path) > 0
    bodies, edges = vehicle_domains(labels, union)
    return labels, bodies, edges
