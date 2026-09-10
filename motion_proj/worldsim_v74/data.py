"""V74 数据入口：求解器仅使用 load_build，评价器单独读取 QUERY。"""
import json
from pathlib import Path
import numpy as np


def load_build(folder):
    folder=Path(folder)
    with np.load(folder/'build.npz',allow_pickle=False) as arrays:
        build={key:arrays[key] for key in arrays.files}
    build['metadata']=json.loads((folder/'build_metadata.json').read_text())
    return build


def load_query_rays(folder):
    with np.load(Path(folder)/'query_rays.npz',allow_pickle=False) as arrays:
        return {key:arrays[key] for key in arrays.files}


def load_query_for_evaluation(folder):
    import torch
    with np.load(Path(folder)/'query_truth.npz',allow_pickle=False) as data:
        frames=json.loads(str(data['frame_metadata_json']))
        offsets=data['frame_offsets']
        keys=['origins_actor_m','directions_actor','observed_first_range_m',
              'positive_actor','ambiguous_owner','points_actor_m','point_timestamps_ns']
        return [{**meta,**{key:torch.from_numpy(data[key][offsets[i]:offsets[i+1]].copy()) for key in keys}}
                for i,meta in enumerate(frames)]
