"""Unmodified inherited C2 spline/spacing mathematical helpers, CPU only.

Extracted from fit_surface.py, full-source SHA256:
df96e49c6274f95f0c2e68cdedefe0d24d0b3e1e5435638acb66975f6cbcec73
Original description: Observed-anchored C2 cubic B-spline camera-depth
posterior. One data-derived physical control pitch; fixed 5 Huber IRLS solves.
Function bodies are retained verbatim; scene paths, assets and CLI are omitted.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def nearest3_pitch(x):
    d=cKDTree(x).query(x,k=4)[0][:,1:4]
    return float(np.median(d[d>1e-8]))


def knots_for(lo,hi,step):
    left=np.floor(lo/step)*step-step
    intervals=int(np.ceil((hi-left)/step))+1
    return left+np.arange(-3,intervals+4)*step


def tensor_basis(uv,tu,tv):
    u=BSpline.design_matrix(uv[:,0],tu,3,extrapolate=False).tocsr()
    v=BSpline.design_matrix(uv[:,1],tv,3,extrapolate=False).tocsr()
    assert np.all(np.diff(u.indptr)==4) and np.all(np.diff(v.indptr)==4)
    ui,vi=u.indices.reshape(-1,4),v.indices.reshape(-1,4)
    ud,vd=u.data.reshape(-1,4),v.data.reshape(-1,4)
    nv=len(tv)-4
    col=(ui[:,:,None]*nv+vi[:,None,:]).reshape(-1)
    values=(ud[:,:,None]*vd[:,None,:]).reshape(-1)
    out=coo_matrix((values,(np.repeat(np.arange(len(uv)),16),col)),shape=(len(uv),(len(tu)-4)*nv)).tocsr()
    out.eliminate_zeros()
    if len(uv): assert np.max(abs(np.asarray(out.sum(1)).ravel()-1))<2e-12
    return out


def curvature(nu,nv):
    rows=[];cols=[];vals=[];n=0
    for i in range(1,nu-1):
        for j in range(nv):
            rows.extend([n]*3);cols.extend([(i-1)*nv+j,i*nv+j,(i+1)*nv+j]);vals.extend([1.,-2.,1.]);n+=1
    for i in range(nu):
        for j in range(1,nv-1):
            rows.extend([n]*3);cols.extend([i*nv+j-1,i*nv+j,i*nv+j+1]);vals.extend([1.,-2.,1.]);n+=1
    return coo_matrix((vals,(rows,cols)),shape=(n,nu*nv)).tocsr()
