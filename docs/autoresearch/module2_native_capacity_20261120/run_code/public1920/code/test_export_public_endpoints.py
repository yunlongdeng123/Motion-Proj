"""CPU gate fixtures; no native/context/lease is created."""
from pathlib import Path
import numpy as np
import pytest
from export_public_endpoints import validate_asset,inspect

def pair(tmp_path):
    p=tmp_path/'parent';a=tmp_path/'asset';p.mkdir();a.mkdir()
    road=dict(active=np.array([True,False]),model_xyz=np.arange(6).reshape(2,3).astype(float),
        road_opacity=np.array([.99,.99],np.float32),reference_srgb=np.full((2,3),.2,np.float32))
    provenance=dict(known=np.array([True,False]),source=np.array([1,3]),observed_uv=np.array([[.3,.4],[np.nan,np.nan]]),effective_row_rgb=road['reference_srgb'].copy())
    for d in(p,a):np.savez_compressed(d/'road.npz',**road);np.savez_compressed(d/'provenance.npz',**provenance)
    return p,a,road,provenance

def test_rgb_only_and_effective_layer_leave_prior_bytes_exact(tmp_path):
    p,a,road,provenance=pair(tmp_path);road['reference_srgb'][0]=[.3,.5,.8];provenance['effective_row_rgb']=road['reference_srgb']
    np.savez_compressed(a/'road.npz',**road);np.savez_compressed(a/'provenance.npz',**provenance)
    assert validate_asset(p,a)['all_original_prior_ledger_bytes_exact']
    road['road_opacity'][0]=.98;np.savez_compressed(a/'road.npz',**road)
    with pytest.raises(ValueError,match='non-RGB'):validate_asset(p,a)

def test_pseudo_source_cannot_be_promoted_to_observed(tmp_path):
    p,a,road,provenance=pair(tmp_path);provenance['known'][1]=True;np.savez_compressed(a/'provenance.npz',**provenance)
    with pytest.raises(ValueError,match='prior ledger'):validate_asset(p,a)

def test_incomplete_formal_cannot_create_gpu_work(tmp_path):
    with pytest.raises(FileNotFoundError,match='not yet ready'):inspect(tmp_path)
