"""为Git生成本轮小收据，不复制PNG、NPZ、颜色算子或完整结果矩阵。"""
from pathlib import Path
import argparse
import json

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_source_fidelity_20261122_r2')


def compact(value):
    if isinstance(value, list):
        return [compact(x) for x in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ('output', 'source_packet', 'packet', 'bbox_xyxy'):
                continue
            if key == 'non_RGB_fields_exact' and isinstance(item, dict):
                result['all_non_RGB_fields_exact'] = all(item.values())
                result['checked_non_RGB_field_count'] = len(item)
            else:
                result[key] = compact(item)
        return result
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    out = root / 'publication/compact_receipts'
    out.mkdir(parents=True, exist_ok=True)
    expected = json.loads((root / 'recovery/expected_matrix.json').read_text())
    rows = []
    for stage in expected['expected_stages']:
        folder = root / stage['directory']
        contract = json.loads((folder / 'contract.json').read_text())
        receipt = json.loads((folder / 'receipt.json').read_text())
        if not receipt.get('complete'):
            raise RuntimeError('Incomplete branch: ' + stage['directory'])
        value = dict(schema='source_fidelity_compact_receipt.v1', stage=stage['directory'],
                     kind=stage['kind'], contract=compact(contract), receipt=compact(receipt),
                     full_evidence='See curated archive and restore_manifest_compact.json')
        trace_path = folder / 'recovery_trace.json'
        if trace_path.exists():
            value['CPU_recovery_trace'] = compact(json.loads(trace_path.read_text()))
        dest = out / (stage['directory'] + '.json')
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if len(payload.encode()) > 100000:
            raise ValueError('Receipt is not compact: ' + str(dest))
        dest.write_text(payload, encoding='utf-8')
        rows.append(dict(stage=stage['directory'], file=str(dest), bytes=len(payload.encode())))
    print(json.dumps(dict(complete=True, receipts=rows)))


if __name__ == '__main__':
    main()
