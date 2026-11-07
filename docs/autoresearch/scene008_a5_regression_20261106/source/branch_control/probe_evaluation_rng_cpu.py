"""Actual vendored LPIPS CPU RNG behavior; never initializes CUDA or optimizers."""
from pathlib import Path
import hashlib
import json
import pickle
import random
import sys

import numpy as np
import torch

ENGINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
ROOT = ENGINE.parents[2] / 'scene008_a5_branch_control_20261107_r1'
sys.path.insert(0, str(ENGINE))
from lpipsPyTorch import lpips
from utils.hybrid30k_runtime import preserve_rng


def states():
    return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda_initialized=torch.cuda.is_initialized())


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    x = torch.zeros((1, 3, 64, 64))
    before = states()
    with torch.no_grad():
        value = float(lpips(x, x).item())
    after = states()
    with preserve_rng():
        with torch.no_grad():
            value2 = float(lpips(x, x).item())
    after_guard = states()
    result = dict(status='PASS_ACTUAL_NATIVE_LPIPS_CPU_RNG_PROBE', cuda_initialized=torch.cuda.is_initialized(),
                  optimization_updates=0, before=before, after=after, after_guard=after_guard,
                  unguarded_changed=[k for k in before if before[k] != after[k]],
                  guard_restored=after == after_guard, lpips_zero_pair=value, guarded_zero_pair=value2,
                  native_complete_eval_explicit_rng_guard=False,
                  source=str(ENGINE / 'lpipsPyTorch/__init__.py'),
                  probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation='No CUDA evaluation run; does not prove equality to a counterfactual long-run trajectory')
    assert result['guard_restored'] and not result['cuda_initialized']
    (ROOT / 'evaluation_rng_cpu_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
