"""CPU-only same-global-UV canvas embedding; no colour or geometry invention."""
import numpy as np


def embed(bank,shape,original_shape=None):
    shape=tuple(map(int,shape))
    if original_shape is None:
        for key in ['permission_mask','known','valid_mask','geometry_valid','world_xyz','canonical_linear_RGB']:
            if key in bank:original_shape=bank[key].shape[:2];break
    original_shape=tuple(original_shape);assert len(shape)==2 and all(a>=b for a,b in zip(shape,original_shape))
    out={}
    for key,value in bank.items():
        value=np.asarray(value)
        if value.ndim>=2 and value.shape[:2]==original_shape:
            new=np.zeros((*shape,*value.shape[2:]),dtype=value.dtype)
            if key in ['world_xyz','surface_world_xyz'] and np.issubdtype(value.dtype,np.floating):new.fill(np.nan)
            elif key=='grid_world_row' and np.issubdtype(value.dtype,np.signedinteger):new.fill(-1)
            new[:original_shape[0],:original_shape[1]]=value
            assert new[:original_shape[0],:original_shape[1]].tobytes()==value.tobytes()
            out[key]=new
        else:out[key]=value.copy()
    return out


def minimal_vertex_scope(cell_mask):
    h,w=cell_mask.shape;out=np.zeros((h+1,w+1),bool)
    for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:out[dy:dy+h,dx:dx+w]|=cell_mask
    return out


def needed_shape(uv,scope,parent_shape):
    points=np.asarray(uv)[scope];assert np.isfinite(points).all() and len(points)
    assert (points>=0).all(),'Negative global UV needs an explicit origin translation contract'
    return (max(int(parent_shape[0]),int(np.ceil(points[:,1].max()))+1),max(int(parent_shape[1]),int(np.ceil(points[:,0].max()))+1))
