"""Pure NumPy mask formulas from sealed4560 contract. No environment mutations."""
import numpy as np


def original_mask(value):
    result=np.asarray(value).squeeze().astype(bool)
    if result.shape!=(320,480):
        raise ValueError(('Expected original native320x480 mask',result.shape))
    return result


def fixed_domains(obs):
    a5=original_mask(obs['actor_labels']==1)
    a6=original_mask(obs['visible'][0])
    a37=original_mask(obs['visible'][1])
    road=original_mask(obs['road'])
    edit=original_mask(obs['road_edit'])
    protect=original_mask(obs['protect'])
    return dict(a5=a5,a6=a6,a37=a37,road=road,road_edit=edit,protect=protect,
        protected_nonroad=protect & ~a6 & ~a37 & ~road & ~edit,
        fixed_nonroad_geometry_guard=~road & ~edit & (protect | a5 | a6 | a37))
