"""CPU-only finite tensor audit for a specified saved checkpoint; no inference."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
rows = []

def visit(value, name):
    if torch.is_tensor(value):
        checked = value.is_floating_point() or value.is_complex()
        rows.append({'path': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                     'elements': value.numel(), 'finite_checked': checked,
                     'nonfinite_count': int((~torch.isfinite(value)).sum()) if checked else 0})
    elif isinstance(value, dict):
        for key, item in value.items():
            visit(item, name + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            visit(item, name + '/' + str(key))

visit(checkpoint, 'checkpoint')
assert rows
digest = hashlib.sha256()
with args.checkpoint.open('rb') as f:
    for chunk in iter(lambda: f.read(8*1024*1024), b''):
        digest.update(chunk)
result = {'recorded_at': datetime.datetime.now().astimezone().isoformat(),
          'checkpoint': str(args.checkpoint), 'bytes': args.checkpoint.stat().st_size,
          'sha256': digest.hexdigest(), 'device': 'cpu',
          'tensor_count': len(rows), 'checked_float_tensors': sum(r['finite_checked'] for r in rows),
          'checked_float_elements': sum(r['elements'] for r in rows if r['finite_checked']),
          'nonfinite_count': sum(r['nonfinite_count'] for r in rows),
          'tensors': rows,
          'scope': 'Saved tensor finiteness, including recursively stored optimizer tensors; not proof that propagation candidate selection was unaffected by NumPy warnings.'}
args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}))
assert result['nonfinite_count'] == 0
