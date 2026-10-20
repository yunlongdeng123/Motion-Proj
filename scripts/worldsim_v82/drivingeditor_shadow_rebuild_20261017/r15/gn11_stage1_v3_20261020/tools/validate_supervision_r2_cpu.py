"""在完整campaign上执行监督R2的CPU验证并保存可复查凭据；不启动GPU。"""

import argparse
import datetime
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.campaign), str(args.campaign/'engine')]
    import torch
    from run_v3_arm import validate
    from utils.v3_runtime_context import V3_POLICY, SUPERVISION_VERSION
    plan = json.loads((args.campaign/'plan.json').read_text())
    assert plan['common_v3_policy'] == V3_POLICY
    assert not (args.campaign/'GPU_APPROVAL.json').exists()
    bundle = Path(plan['input_bundle'])
    previous_bundle = args.campaign/'source_before_exposure199_20261020/bundle_before_turn_snapshot.json'
    assert bundle.read_bytes() == previous_bundle.read_bytes()
    v1 = Path(plan['basis']['five_identity_v1'])
    for name in ['hybrid30k_losses.py', 'paper_losses.py']:
        assert (v1/'utils'/name).read_bytes() == (args.campaign/'engine/utils'/name).read_bytes()
    # 本轮不修改普通背景表示、V1正则/独立监督/原生完整alpha路径。
    from xml.etree import ElementTree
    for name in ['architecture.svg', 'supervision_architecture.svg']:
        ElementTree.parse(args.campaign/name)
    fresh_outputs = []
    with tempfile.TemporaryDirectory() as temporary:
        common = dict(plan=args.campaign/'plan.json', baseline_config=Path(plan['baseline_config']),
                      bundle=bundle, engine=args.campaign/'engine', mode='gate', gpu_gate=None)
        for arm in plan['arm_order']:
            validate(SimpleNamespace(**common, arm=arm, output=Path(temporary)/arm))
            fresh_outputs.append(arm)
        gate = dict(status='PASSED_ACTUAL_CUDA_V3_FORWARD_BACKWARD_ZERO_UPDATES', optimizer_updates=0,
                    method_version=plan['method_version'], arm='full',
                    background_temporal_version=plan['background_temporal_version'], road_version=plan['road_version'])
        old_gate = Path(temporary)/'old_gate.json'
        old_gate.write_text(json.dumps(gate))
        stale = dict(common, arm='full', output=Path(temporary)/'stale', mode='train', gpu_gate=old_gate)
        try:
            validate(SimpleNamespace(**stale))
        except RuntimeError as error:
            assert 'CUDA gate belongs' in str(error)
        else:
            raise AssertionError('old supervision gate was accepted')
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover(str(args.campaign/'tests')))
    args.output.mkdir(parents=True)
    (args.output/'unittest.log').write_text(log.getvalue(), encoding='utf-8')
    receipt = dict(task='WS-V82-R15-GN11-STAGE1-V3-20261020', supervision_version=SUPERVISION_VERSION,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), campaign=str(args.campaign),
        status='CPU_PASSED_REAL_CUDA_PENDING' if result.wasSuccessful() else 'CPU_FAILED_PRESERVED',
        torch_version=torch.__version__, tests_run=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped), full_log='unittest.log',
        real49_bundle_unchanged=True, V1_native_regularizer_source_bytes_unchanged=True,
        four_arm_CPU_preflight=fresh_outputs, old_supervision_GPU_gate_rejected=True,
        plan_matches_runtime=True, SVG_XML_valid=True,
        nonleaf_old_repro='returned7 + retained7 + manual addition ->14; numeric regression asserts this',
        nonleaf_fix='preserve all three proxy .grad buffers around each routed autograd.grad; base and routed contributions exactly once, including existing buffers and road-row mask',
        front_diagnostic_CPU='synthetic and native-render-function CPU stubs only; not actual scene CUDA front mass',
        GPU_device_nodes_present=bool(list(Path('/dev').glob('nvidia*'))),
        GPU_used=False, training_updates=0, control_updates=0, quality_accepted=False, human_verdict=None,
        pending=['user review', 'explicit GPU approval', 'actual native CUDA feature/transmittance parity and overhead',
                 'real scene front-alpha diagnostic', 'separately reviewed short control', 'four fresh30K and full49 review'],
        failure_ledger_refs=['V82-F01', 'V82-F06'], failure_ledger_delta='none')
    (args.output/'validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
