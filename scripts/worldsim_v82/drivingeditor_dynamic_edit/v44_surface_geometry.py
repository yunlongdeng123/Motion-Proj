"""Shared immutable Gaussian-surface fields; no scene or asset paths."""
from pathlib import Path
import hashlib
import numpy as np

FLOAT_FIELDS = ('xyz', 'normal', 'sigma', 'opacity', 'thickness', 'box',
                'knn_spacing', 'normal_radius', 'normal_confidence')

BOOL_FIELDS = ('generated', 'conditioning', 'normal_valid', 'out_of_box', 'conditioning_held')

INT_FIELDS = ('source_index', 'conditioning_observation_index', 'conditioning_raw_frame', 'conditioning_cloud_row')

GEOMETRY_FIELDS = FLOAT_FIELDS+BOOL_FIELDS+INT_FIELDS

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def hash_geometry(arrays):
    digest = hashlib.sha256()
    for key in sorted(GEOMETRY_FIELDS):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode('ascii'))
        digest.update(str((value.dtype.str, value.shape)).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()

def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
