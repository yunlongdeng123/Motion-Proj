"""Shared full-row semantic-removal controls; NumPy only, no GPU imports.

Source semantics are frozen observations, not edited-view labels. A missing row
is unknown. Removing unknown may remove useful background: this is an ablation,
not a safe instance deletion algorithm. All vehicle classes means 14..19.
"""
import numpy as np

RADII=(3,5,8)
MODES=('all','hide_unknown','hide_vehicle','hide_both')

def classify_rows(observed,total_rows,static_rows,static_unknown_ids=(),radius=5,fraction=.2):
    ids=np.asarray(observed['ids'],dtype=np.int64)
    if not (0<=static_rows<=total_rows) or np.any(ids<0) or np.any(ids>=total_rows):
        raise ValueError('Mismatched static/dynamic row namespace')
    if radius not in (0,)+RADII or not 0<fraction<=1:
        raise ValueError('Unsupported contour rule')
    labels=np.zeros(total_rows,dtype=np.uint8);labels[ids]=observed['class_id']
    if radius:
        band=np.asarray(observed['boundary_fraction'])[:,RADII.index(radius)]>=fraction
        labels[ids[band]]=0
    quarantine=np.asarray(static_unknown_ids,dtype=np.int64)
    if np.any(quarantine<0) or np.any(quarantine>=static_rows):
        raise ValueError('Quarantine can only address original static rows')
    labels[quarantine]=0
    return labels

def removal_mask(labels,mode):
    if mode not in MODES:raise ValueError(mode)
    unknown=labels==0;vehicle=(labels>=14)&(labels<=19)
    return (unknown if mode=='hide_unknown' else vehicle if mode=='hide_vehicle'
            else unknown|vehicle if mode=='hide_both' else np.zeros_like(unknown))
