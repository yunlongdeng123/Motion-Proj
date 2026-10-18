"""Compile both actual update slices and check global bindings, without CUDA."""
from pathlib import Path
import builtins, dis, json, os, random, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / 'implementation'))
import pilot_gpu_review as review


def main():
    from omegaconf import OmegaConf
    import numpy as np
    results = []
    for name, effective_arm in [('five_replay', 'five_replay'), ('five_raw_body_isolated', 'five_raw_body')]:
        e = json.loads((OUT / 'pilots' / effective_arm / 'effective_config.json').read_text())
        spec = e['spec']
        sys.path.insert(0, spec['source_repository'])
        cfg = OmegaConf.create(json.loads(Path(spec['reference_config']).read_text()))
        with np.load(Path(spec['prepared']) / 'instances.npz', allow_pickle=False) as bundle:
            ids = bundle['instance_ids'].tolist()
        model_tuple = (cfg, SimpleNamespace(getTrainCameras=lambda: []), None,
                       SimpleNamespace(instance_ids=ids), None, None, None)
        function, source = review.make_update(review.implementation(name), model_tuple, e, random.Random(0), 301, lambda x: None)
        required = sorted(set(i.argval for i in dis.get_instructions(function) if i.opname == 'LOAD_GLOBAL'))
        missing = [n for n in required if n not in function.__globals__ and not hasattr(builtins, n)]
        assert not missing, missing
        results.append(dict(arm=name, global_names=required, missing=[], **source))
    import torch
    assert not torch.cuda.is_initialized()
    result = dict(status='PASSED_CPU_AST_UPDATE_BINDINGS_ONLY', CUDA_initialized=False,
        optimizer_updates=0, actual_GPU_execution_verified=False, variants=results)
    (OUT / 'review_CPU_binding_receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
