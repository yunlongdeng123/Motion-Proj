"""CPU-only V42 preparation inspection. Never loads models or launches work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/experiment_plan_20261026.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    root = Path(plan['weights_root'])
    campaign = Path(plan['campaign'])
    receipt_path = root / 'download_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.is_file() else {}
    entries = receipt.get('files', [])
    valid = []
    for entry in entries:
        path = Path(entry['path'])
        valid.append(entry.get('verified') is True and path.is_file()
                     and path.stat().st_size == entry['size']
                     and not path.with_name(path.name + '.aria2').exists())
    baseline = {name: (Path(plan['baseline']['root']) / name).is_file()
                and (Path(plan['baseline']['root']) / name).stat().st_size > 0
                for name in plan['baseline']['files']}
    checks = {
        'download_receipt_complete': receipt.get('status') == 'complete',
        '36_prior_verified_files_present_with_recorded_sizes': len(entries) == 36 and all(valid),
        'baseline_four_files_present_nonempty': all(baseline.values()),
        'official_source_present': (root / 'code/test.py').is_file() and (root / 'code/network/freeinpaint.py').is_file(),
        'isolated_python_present_not_runtime_validation': Path(plan['isolated_python']).is_file(),
        'input_manifest_present_not_domain_validation': (campaign / plan['input_manifest']).is_file(),
        'execution_contract_present_not_smoke_validation': (campaign / plan['execution_contract']).is_file(),
        'adapter_receipt_present_not_acceptance': (campaign / plan['engineering_receipt']).is_file(),
    }
    disk_path = campaign if campaign.exists() else campaign.parent
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'task_id': plan['task_id'], 'checks': checks, 'baseline_files': baseline,
        'missing_preparation': [key for key, passed in checks.items() if not passed],
        'free_bytes': shutil.disk_usage(disk_path).free,
        'weight_hashes_recomputed_this_check': False,
        'weight_verification_basis': 'prior complete receipt plus current file sizes; no deserialization',
        'launch_allowed_by_this_tool': False,
        'inference_started': False, 'optimizer_updates': 0,
        'next_step': 'Complete missing preparation; user GPU-ready notice and real smoke are required before formal work.',
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
