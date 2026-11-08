"""Independent real-four-input API/receipt binding, with no GPU or model import."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np


HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
API = HERE.parent / 'material_protection/road_material_supervision.py'
OP = HERE.parent / 'material_protection/operator_manifest.json'
OUT = WS / 'outputs/scene008_v44_a5_material_protection_20261108/evidence/independent_supervision_interface_cpu_v1.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    assert not OUT.exists()
    manifest = json.loads(OP.read_text(encoding='utf-8-sig'))
    source_sha = sha(API)
    assert manifest['supervision_api_actual_called'] is True
    assert source_sha == manifest['supervision_api_source_sha256']
    assert manifest['status'] == 'PASS_M4_NATIVE_PIXEL_PARENT_OPERATOR_NO_FIT'
    spec = importlib.util.spec_from_file_location('actual_road_material_supervision', API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for raw in [94, 105, 128, 137]:
        observation = load(HERE / f'inputs/observations/{raw:03d}.npz')
        target = load(HERE / f'inputs/targets/raw{raw:03d}.npz')
        supervision = module.build_supervision(observation, target)
        road, edit, protect = observation['road'], observation['road_edit'], observation['protect']
        a6, a37 = observation['visible']
        expected = dict(protected_nonroad=protect & ~a6 & ~a37 & ~road & ~edit, original_road=road)
        assert set(supervision.protection_domains) == set(expected)
        for name, value in expected.items():
            np.testing.assert_array_equal(supervision.protection_domains[name], value)
        np.testing.assert_array_equal(supervision.fit_mask, target['fit_mask'])
        np.testing.assert_array_equal(supervision.fit_rgb, target['edgeblend_rgb'].transpose(2, 0, 1))
        np.testing.assert_array_equal(supervision.protection_rgb, observation['rgb'].transpose(2, 0, 1))
        np.testing.assert_array_equal(supervision.source_protect_mask, target['protect_mask'])
        union = expected['protected_nonroad'] | expected['original_road']
        assert not (target['protect_mask'] & ~union).any()
        assert not (a37 & union).any()
        assert not (target['fit_mask'] & target['protect_mask']).any()
        # Actual old targets overlap original_road. Both source roles remain
        # intact; the registered LP, rather than mask erasure, owns priority.
        assert int((target['fit_mask'] & union).sum()) == int(target['fit_mask'].sum())
        for state in ['keep', 'delete']:
            rec = next(r for r in manifest['records'] if r['raw'] == raw and r['state'] == state)
            for op_key, value in [('protected_nonroad', expected['protected_nonroad']),
                                  ('originalroad', road), ('target_fit_mask', target['fit_mask']),
                                  ('target_protect_mask', target['protect_mask'])]:
                assert rec['mask_pixels'][op_key] == int(value.sum())
            assert all(v == 0 for v in rec['standard_reference_parity_max_abs'].values())
            assert rec['standard_8groups_byte_exact']
        records.append(dict(raw=raw, **supervision.summary(),
                            original_C_fit_domain_preserved_byte_exact=True,
                            protection_domains_original94_masks_byte_exact=True,
                            source_protect_pixels_covered_by_union=int(target['protect_mask'].sum()),
                            source_protect_pixels_outside_union=0,
                            retained_a37_pixels_in_protection_union=0,
                            actual_operator_KEEP_DELETE_mask_counts_match=True))
    result = dict(schema='INDEPENDENT_REAL4_SUPERVISION_API_CPU_V1',
                  status='PASS_REAL4_FIXED_INPUTS_AND_ACTUAL_NATIVE_OPERATOR_API_BINDING',
                  actual_API_source=dict(path=str(API), sha256=source_sha),
                  actual_native_operator_receipt=dict(path=str(OP), sha256=sha(OP),
                      status=manifest['status'], supervision_api_actual_called=True,
                      source_SHA_matches_executed_operator=True,
                      actual_native_forwards=manifest['actual_native_forwards'],
                      actual_backwards=manifest['actual_backwards'],
                      optimizer_updates=manifest['optimizer_updates'],
                      variable_parent_count=manifest['variable_parent_count']),
                  records=records,
                  interpreted_scope='API and real-input semantics only; native parent closure, LP constraints and actual candidate acceptance retain their own receipts.',
                  original94_domains_changed=False, training_role='All four are BUILD, including historical HELD views',
                  material_protection_solved=False, GPU_forwards=0, backwards=0,
                  optimizer_updates=0, models_loaded=False, quality_accepted=False, human_verdict=None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), status=result['status'], api_source_sha256=source_sha,
                          actual_operator_native_forwards=manifest['actual_native_forwards'],
                          variable_parents=manifest['variable_parent_count'])))


if __name__ == '__main__':
    main()
