from pathlib import Path
import json

R = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
paths = [
    'baseline4/receipt.json',
    'contract/first_forward_engineering_failure.json',
    'AA4_retry1/receipt.json',
    'C4_retry1/receipt.json',
    'diagnostic130/status.json',
    'support_qualification4_v2/receipt.json',
    'D4_v2/receipt.json',
    'E4_appearance_qualification/receipt.json',
    'E_zero_identity4/receipt.json',
    'E_smoke_r1/returned.json',
    'E_restore_parity_r1/returned.json',
    'E_formal400_r1/returned.json',
    'F4_bounded_appearance_qualification/receipt.json',
]
for relative in paths:
    value = json.loads((R / relative).read_text())
    selected = {k: v for k, v in value.items() if any(t in k.lower() for t in ['status', 'forward', 'backward', 'update', 'optimizer'])}
    print(json.dumps({'path': relative, 'counters': selected, 'top_keys': list(value)}))
