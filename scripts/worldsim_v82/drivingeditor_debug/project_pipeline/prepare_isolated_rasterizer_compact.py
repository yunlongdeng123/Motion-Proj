#!/usr/bin/env python3
"""Create a separate prototype without installing or altering the live package."""
import difflib
import hashlib
import json
import shutil
from pathlib import Path
base=Path('/root/autodl-tmp/drivingeditor_repro')
source=base/'deps/diff-gaussian-rasterization'
dest=base/'infra/rasterizer_compact'
assert source.is_dir() and not dest.exists()
shutil.copytree(source,dest,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__','*.so'))
(dest/'diff_gaussian_rasterization').rename(dest/'drivingeditor_rasterization_compact')
changes={}
def modify(rel,fn):
    path=dest/rel;before=path.read_text();after=fn(before)
    assert before!=after, rel
    path.write_text(after)
    changes[rel]={'original_sha256':hashlib.sha256(before.encode()).hexdigest(),
                  'prototype_sha256':hashlib.sha256(after.encode()).hexdigest()}
    return ''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
patch=modify('setup.py',lambda s:s.replace('diff_gaussian_rasterization','drivingeditor_rasterization_compact'))
compact='''
    // Only the sorted Gaussian IDs are needed by backward. This prototype's
    // opaque saved-buffer ABI intentionally differs from the original package.
    struct CompactBinningState
    {
        uint32_t* point_list;
        static CompactBinningState fromChunk(char*& chunk, size_t R)
        {
            CompactBinningState compact;
            obtain(chunk, compact.point_list, R, 128);
            return compact;
        }
    };

'''
patch+=modify('cuda_rasterizer/rasterizer_impl.h',lambda s:s.replace('\ttemplate<typename T> \n',compact+'\ttemplate<typename T> \n'))
patch+=modify('cuda_rasterizer/rasterizer_impl.cu',lambda s:s.replace(
    '\tBinningState binningState = BinningState::fromChunk(binning_buffer, R);',
    '\tCompactBinningState binningState = CompactBinningState::fromChunk(binning_buffer, R);'))
def alter_wrapper(s):
    s=s.replace('#include <torch/extension.h>','#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n#include <c10/cuda/CUDAStream.h>')
    s=s.replace('#include "cuda_rasterizer/rasterizer.h"','#include "cuda_rasterizer/rasterizer.h"\n#include "cuda_rasterizer/rasterizer_impl.h"')
    # Original kernels and CUB calls use stream0, not PyTorch current stream.
    # Fail closed on unsupported streams rather than free a temporary unsafely.
    guard='''  TORCH_CHECK(means3D.is_cuda(), "Compact rasterizer expects CUDA means3D");
  TORCH_CHECK(c10::cuda::getCurrentCUDAStream(means3D.get_device()) ==
              c10::cuda::getDefaultCUDAStream(means3D.get_device()),
              "Compact prototype requires the default CUDA stream");
'''
    s=s.replace('  const int P = means3D.size(0);',guard+'  const int P = means3D.size(0);')
    s=s.replace('  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));',
        '  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));\n'
        '  torch::Tensor temporaryBinningBuffer = torch::empty({0}, options.device(device));')
    s=s.replace('resizeFunctional(binningBuffer);','resizeFunctional(temporaryBinningBuffer);')
    needle='''\t\tradii.contiguous().data_ptr<int>(),
\t\tdebug);
  }
  return std::make_tuple(rendered,'''
    replacement='''\t\tradii.contiguous().data_ptr<int>(),
\t\tdebug);

      // Forward has consumed full sorting state on the original default stream.
      // Retain a separate allocation so the large temporary storage is not kept
      // alive through an alias/view in autograd. Keep only sorted point IDs.
      binningBuffer.resize_({static_cast<int64_t>(
          CudaRasterizer::required<CudaRasterizer::CompactBinningState>(rendered))});
      char* destination = reinterpret_cast<char*>(binningBuffer.data_ptr());
      auto compact = CudaRasterizer::CompactBinningState::fromChunk(destination, rendered);
      char* source = reinterpret_cast<char*>(temporaryBinningBuffer.data_ptr());
      uint32_t* sorted_points;
      CudaRasterizer::obtain(source, sorted_points, static_cast<size_t>(rendered), 128);
      auto stream = c10::cuda::getDefaultCUDAStream(means3D.get_device()).stream();
      if (rendered > 0)
          C10_CUDA_CHECK(cudaMemcpyAsync(compact.point_list, sorted_points,
              static_cast<size_t>(rendered) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
      // The original implementation launches on stream0. Explicit completion
      // prevents use-after-free even when the allocator reuses the temporary
      // immediately. Prototype tradeoff: this adds a forward synchronization.
      C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return std::make_tuple(rendered,'''
    assert needle in s
    return s.replace(needle,replacement)
patch+=modify('rasterize_points.cu',alter_wrapper)
(dest/'compact_binning.patch').write_text(patch)
(dest/'isolation_manifest.json').write_text(json.dumps({'source':str(source),'prototype':str(dest),
    'package':'drivingeditor_rasterization_compact','extension':'drivingeditor_rasterization_compact._C',
    'installation':'None; build_ext --inplace within prototype only',
    'changes':changes,'semantics':'Identical kernels/CUB sort; only retained opaque binning state differs.',
    'stream_policy':'Default stream only; explicit forward-end synchronization before temporary destruction.',
    'peak_memory_caveat':'Full workspace plus compact allocation coexist briefly. Persistent autograd saved bytes shrink; single-call peak need not shrink.'},indent=2)+'\n')
print(dest)
