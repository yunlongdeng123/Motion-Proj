"""Separate read-only worker: export the accepted current M2 scene as reference.

The candidate never imports this worker or its old road/gate/fine coefficients.
This file intentionally uses the current accepted runtime ONLY for comparison.
"""
import argparse
import json
import os
from pathlib import Path
import sys

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')


def execute(args):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    from final_runtime import ASSET_SOURCE, ASSET, FIXED, sha, write, read, stable_hash
    from evaluate_outputs import evaluate
    if args.output.exists():
        raise FileExistsError('Preserve the existing reference export; use a fresh output')
    sys.path.insert(0, str(ASSET_SOURCE))
    from scene_runtime import Runtime
    from sh_asset import SHAsset

    class Reference:
        def __init__(self):
            self.source = Runtime(fine_road=True)
            self.asset = SHAsset.from_geometry(ASSET).cuda().requires_grad_(False)
            self.before = {key: value.detach().cpu().clone() for key, value in
                           (*self.asset.named_parameters(), *self.asset.named_buffers())}
            assert len(self.before) == 27
            self.road_run = self.source.road_checkpoint.parent.parent
            with np.load(FIXED, allow_pickle=False) as fixed:
                assert fixed['layer_names'].tolist()[3] == 'ground_beneath_a37_proxy'
                self.fixed_ground = fixed['depth_m'][:, 3] > 0
            self.binding = dict(accepted_current_M2_reference_only=True, M2_sha256=sha(ASSET),
                                runtime_source=str(ASSET_SOURCE/'scene_runtime.py'),
                                runtime_source_sha256=sha(ASSET_SOURCE/'scene_runtime.py'),
                                current_scene=self.source.meta)
            self.binding_sha256 = stable_hash(self.binding)

        def raw(self, raw):
            return self.source.raw(raw)

        def render_scene(self, raw, delete=False):
            return self.source.render_scene(raw, self.asset, delete=delete)

        def verify(self):
            assert self.source.modelunchanged()
            assert sha(ASSET) == self.binding['M2_sha256']
            assert self.asset.assert_geometry_unchanged()
            after = dict((*self.asset.named_parameters(), *self.asset.named_buffers()))
            assert set(after) == set(self.before)
            for key, before in self.before.items():
                current = after[key].detach().cpu().contiguous()
                assert before.dtype == current.dtype and before.shape == current.shape
                assert torch.equal(before.contiguous().reshape(-1).view(torch.uint8), current.reshape(-1).view(torch.uint8)), key

    with torch.no_grad():
        reference = Reference()
        evaluate(reference, args.output, 'CURRENT accepted native10K + fixed M2 + original road1K/fine / REFERENCE ONLY')
        reference.verify()
        manifest = read(args.output/'render_manifest.json')
        manifest.update(accepted_current_M2_reference_only=True, M2_sha256=sha(ASSET),
                        candidate_run=False, reference_binding=reference.binding,
                        M2_all_27_tensors_bitwise_unchanged_CPU=True)
        write(args.output/'render_manifest.json', manifest)
        write(args.output/'reference_binding.json', reference.binding)
        print(json.dumps(dict(status='PASS_REFERENCE_EXPORT_ONLY', output=str(args.output), training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute(args)
    else:
        print(json.dumps(dict(accepted_reference_only=True, candidate_run=False, training_updates=0,
                              output=str(args.output), execute=False), indent=2))


if __name__ == '__main__':
    main()
