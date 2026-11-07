"""Pure NumPy paired-bin domains. No CUDA/env changes and no candidate reads."""
import numpy as np

TANGENT_AXES={'near_x':1,'far_x':1,'lateral_y':0}


def registered_pair_union(world_masks,registered_edges,edge):
    minus=np.asarray(world_masks[edge+'_minus'],bool)
    plus=np.asarray(world_masks[edge+'_plus'],bool)
    selected=minus|plus
    xyz=np.asarray(world_masks['local_xyz'])
    bins=np.zeros(minus.shape,np.int32)
    bins[selected]=np.floor(xyz[...,TANGENT_AXES[edge]][selected]/.2).astype(np.int32)
    result=np.zeros_like(minus)
    pairs=registered_edges[edge]['bins']
    for pair in pairs:
        left=minus&(bins==pair['tangent_bin'])
        right=plus&(bins==pair['tangent_bin'])
        if int(left.sum())!=pair['minus_count'] or int(right.sum())!=pair['plus_count']:
            raise ValueError(('Registered bin count mismatch',edge,pair['tangent_bin']))
        result|=left|right
    return result


def primary128_pair_union(primary_masks):
    """The original primary128 six bins, distinct from worldpatch128 bands."""
    minus=np.asarray(primary_masks['band_near_x_minus'],bool)
    plus=np.asarray(primary_masks['band_near_x_plus'],bool)
    selected=minus|plus
    xyz=np.asarray(primary_masks['local_xyz'])
    bins=np.zeros(minus.shape,np.int32)
    bins[selected]=np.floor(xyz[...,1][selected]/.2).astype(np.int32)
    result=np.zeros_like(minus);counts=[]
    for number in np.unique(bins[selected]):
        left=minus&(bins==number);right=plus&(bins==number)
        if left.sum()>=3 and right.sum()>=3:
            result|=left|right;counts.append((int(left.sum()),int(right.sum())))
    if len(counts)!=6 or sum(x for x,y in counts)!=35 or sum(y for x,y in counts)!=33:
        raise ValueError(('Original primary128 6bin35/33 changed',counts))
    return result
