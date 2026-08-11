"""AD-GS 的 V4 matched nuScenes adapter 参数；不修改论文模型结构。"""

num_cam = 3
use_colmap = False
resolution = 1
order_args = dict(
    xyz=[None, 5, 0, 6, 0, 0],
    rotation=[0, 0, 0, 0, None, 5],
    shs=[0, 0, 0, 6, 0, 0],
    background=[None, 5, 0, 6, 0, 0],
)
