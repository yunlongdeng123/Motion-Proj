import ctypes
from pathlib import Path
import numpy as np
import torch

class Matrix:
    def __init__(self,source):
        source=source.tocsr();self.rows,self.cols=source.shape
        assert source.nnz<2**31
        self.offsets=torch.tensor(source.indptr.astype(np.int32),device='cuda')
        self.ids=torch.tensor(source.indices.astype(np.int32),device='cuda')
        self.values=torch.tensor(source.data.astype(np.float32),device='cuda')
        self.lib=ctypes.CDLL(str(Path(__file__).resolve().parent.parent/'libdeterministic_csr.so'))
        self.lib.csr_mm3.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*6
        self.lib.csr_mm3.restype=ctypes.c_int
    def __call__(self,x):
        assert x.is_cuda and x.dtype==torch.float32 and x.shape==(self.cols,3)
        x=x.contiguous();out=torch.empty((self.rows,3),device=x.device,dtype=x.dtype)
        status=self.lib.csr_mm3(self.rows,self.offsets.data_ptr(),self.ids.data_ptr(),self.values.data_ptr(),
             x.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError('Deterministic CSR CUDA error '+str(status))
        return out
