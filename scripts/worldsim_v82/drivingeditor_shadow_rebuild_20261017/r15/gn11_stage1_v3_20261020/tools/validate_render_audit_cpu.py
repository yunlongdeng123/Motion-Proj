"""渲染/增密/道路本轮CPU准入；不实例化CUDA场景，不授权GPU。"""

import argparse
import datetime
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-approved-campaign', action='store_true', help='CPU-only recheck after separate human approval; never creates approval or invokes GPU')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.campaign), str(args.campaign/'engine')]
    import numpy as np
    import torch
    from run_v3_arm import validate
    from utils.v3_render_audit import RENDER_AUDIT_VERSION, RENDER_AUDIT_POLICY
    from utils.v3_runtime_context import V3_POLICY, ARM_ROAD_WEIGHTS
    plan = json.loads((args.campaign/'plan.json').read_text())
    assert plan['render_audit_version'] == RENDER_AUDIT_VERSION
    assert plan['render_audit_policy'] == RENDER_AUDIT_POLICY
    assert plan['common_v3_policy'] == V3_POLICY
    assert {key: value['road_weights'] for key, value in plan['arms'].items()} == ARM_ROAD_WEIGHTS
    if (args.campaign/'GPU_APPROVAL.json').exists():
        assert args.allow_approved_campaign
        from v3_gpu_authorization import require_gpu_authorization
        require_gpu_authorization(args.campaign/'plan.json', plan, True)
    bundle = Path(plan['input_bundle'])
    meta = json.loads(bundle.read_text())
    parent = Path(meta['parent_bundle'])
    old_meta = json.loads(parent.read_text())
    with np.load(parent.parent/old_meta['units'], allow_pickle=False) as old, np.load(bundle.parent/meta['units'], allow_pickle=False) as new:
        for key in ['unit_ids', 'keys', 'world', 'model_xyz', 'normal_world', 'normal_model',
                    'surface_supported', 'query_unit_id', 'query_frame', 'query_actor', 'source_rgb', 'source_uv', 'source_depth']:
            assert np.array_equal(old[key], new[key]), key
        assert not ((new['base_weight'] > 0) & ~(old['base_weight'] > 0)).any()
        assert new['coverage_model_xyz'].shape == (meta['unit_count'], 9, 3)
        assert all(frame['occluder_masks']['complete'] for frame in meta['frames'].values())
    v1 = Path(plan['basis']['five_identity_v1'])
    for name in ['hybrid30k_losses.py', 'paper_losses.py']:
        assert (v1/'utils'/name).read_bytes() == (args.campaign/'engine/utils'/name).read_bytes()
    meta = json.loads(bundle.read_text())
    with np.load(bundle.parent/meta['units'], allow_pickle=False) as arrays:
        supported = int(arrays['surface_supported'].sum())
        assert supported == meta['supported_units']
        accepted = arrays['base_weight'] > 0
        for key in arrays.files:
            if key.startswith('gate_'):
                assert arrays[key][accepted].all()
        assert np.isfinite(arrays['base_weight']).all() and (arrays['base_weight'] >= 0).all()
        assert arrays['base_weight'].shape[0] == 49
        source_observations = int(accepted.sum())
    from xml.etree import ElementTree
    ElementTree.parse(args.campaign/'render_audit_architecture.svg')
    with tempfile.TemporaryDirectory() as tmp:
        common = dict(plan=args.campaign/'plan.json', baseline_config=Path(plan['baseline_config']), bundle=bundle,
                      engine=args.campaign/'engine', mode='gate', gpu_gate=None)
        for arm in plan['arm_order']:
            validate(SimpleNamespace(**common, arm=arm, output=Path(tmp)/arm))
        gate = dict(status='PASSED_ACTUAL_CUDA_V3_FORWARD_BACKWARD_ZERO_UPDATES', optimizer_updates=0,
            method_version=plan['method_version'], arm='full', road_version=plan['road_version'],
            supervision_version=plan['common_v3_policy']['supervision_version'],
            background_temporal_version=plan['background_temporal_version'])
        receipt = Path(tmp)/'synthetic_stale_gate.json'
        for include_version in [False, True]:
            if include_version:
                gate['render_audit_version'] = RENDER_AUDIT_VERSION
            receipt.write_text(json.dumps(gate))
            try:
                validate(SimpleNamespace(**dict(common, mode='train', gpu_gate=receipt), arm='full', output=Path(tmp)/'unrun'))
            except RuntimeError as error:
                assert ('CUDA gate lacks' if include_version else 'CUDA gate belongs') in str(error)
            else:
                raise AssertionError('stale or evidence-free gate accepted')
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover(str(args.campaign/'tests')))
    args.output.mkdir(parents=True)
    (args.output/'unittest.log').write_text(log.getvalue(), encoding='utf-8')
    receipt = dict(task='WS-V82-GN11-V3-ROAD-REVIEW-R2-20261021', road_version=plan['road_version'],
        render_audit_version=RENDER_AUDIT_VERSION,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='CPU_PASSED_ACTUAL_CUDA_PENDING' if result.wasSuccessful() else 'CPU_FAILED_PRESERVED',
        torch_version=torch.__version__, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), full_log='unittest.log',
        road_supported_world_units=supported, hard_eligible_source_observations=source_observations,
        main49_geometry_queries_raw_RGB_unchanged=True, mask_union_repaired=meta['mask_union_repair'],
        main_reference_frames=49, V1_native_regularizer_source_bytes_unchanged=True,
        four_arm_CPU_preflight=plan['arm_order'], stale_and_evidence_free_CUDA_gates_rejected=True,
        actual_CUDA_screen_numeric_check='implemented_not_executed', actual_CUDA_cost_memory='not_measured',
        actual_scene_road_contribution='not_measured',
        GPU_used=False, training_updates=0, extra_optimizer_updates=0,
        GPU_device_nodes_present=bool(list(Path('/dev').glob('nvidia*'))),
        extended49_199_and_full199_199='deferred_not_authorized', human_verdict=None, quality_accepted=False,
        pending=['actual zero-update CUDA gates for each arm',
                 'numerical screen/count comparisons and feature alpha parity', 'actual overhead and capacity logs',
                 'four fresh30K and full49 reconstruction/edit quality review'],
        failure_ledger_refs=['V82-F01', 'V82-F06'], failure_ledger_delta='none')
    (args.output/'validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
