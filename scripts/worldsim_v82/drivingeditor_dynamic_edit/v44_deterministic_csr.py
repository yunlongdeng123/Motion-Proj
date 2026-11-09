"""Fixed 128-lane double-accumulation CUDA CSR x RGB; explicit portable library.

The CPU backend is a SciPy reference for contracts/math tests, not a claim of
bitwise equality to CUDA. Build native/v44_deterministic_csr.cu separately.
"""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import numpy as np
from scipy.sparse import isspmatrix


def library_path(value=None):
    value = value or os.environ.get('V44_CSR_LIBRARY') or Path(__file__).parent/'native/libdeterministic_csr.so'
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError('Set material_backend.csr_library or V44_CSR_LIBRARY to a compiled csr_mm3 library: ' + str(path))
    return path


def validated_csr(source):
    if not isspmatrix(source):
        raise ValueError('Expected an explicit SciPy sparse matrix')
    for name in ('indices','indptr'):
        if hasattr(source,name) and np.asarray(getattr(source,name)).dtype.kind not in 'iu':
            raise ValueError('CSR index arrays must be integer before conversion')
    source = source.tocsr(copy=True)
    source.check_format(full_check=True)
    if max(source.shape) >= 2**31 or source.nnz >= 2**31:
        raise ValueError('CSR shape/nnz exceeds the inherited int32 kernel ABI')
    if source.data.dtype.kind not in 'iuf' or not np.isfinite(source.data).all():
        raise ValueError('CSR values must be finite real numbers')
    converted = source.astype(np.float32)
    if not np.isfinite(converted.data).all():
        raise ValueError('CSR values overflow float32')
    return converted


class Matrix:
    def __init__(self, source, library=None, device='cuda'):
        source = validated_csr(source); self.rows, self.cols = source.shape
        self.device = str(device); self.source = source
        if self.device == 'cpu':
            self.backend = 'scipy_cpu_reference'; return
        self.path = library_path(library)
        import torch
        target = torch.device(device)
        if target.type != 'cuda':
            raise ValueError('Matrix device must be cpu reference or an explicit CUDA device')
        self.offsets = torch.tensor(source.indptr.astype(np.int32), device=target)
        self.ids = torch.tensor(source.indices.astype(np.int32), device=target)
        self.values = torch.tensor(source.data, device=target)
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.csr_mm3.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
        self.lib.csr_mm3.restype = ctypes.c_int
        self.backend = 'fixed_128_lane_float64_reduction_cuda'

    def __call__(self, x):
        if self.device == 'cpu':
            x = np.asarray(x)
            if x.dtype != np.float32 or x.shape != (self.cols, 3) or not np.isfinite(x).all():
                raise ValueError('CPU reference needs finite float32 (columns,3)')
            return np.asarray(self.source @ x, dtype=np.float32)
        import torch
        if not torch.is_tensor(x) or not x.is_cuda or x.device != self.values.device or x.dtype != torch.float32 or x.shape != (self.cols, 3):
            raise ValueError('CUDA CSR needs same-device float32 (columns,3)')
        x = x.contiguous(); out = torch.empty((self.rows, 3), device=x.device, dtype=x.dtype)
        if not self.rows:
            return out
        status = self.lib.csr_mm3(self.rows, self.offsets.data_ptr(), self.ids.data_ptr(), self.values.data_ptr(),
                                 x.data_ptr(), out.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError('Deterministic CSR CUDA error ' + str(status))
        return out
