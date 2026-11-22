"""Record the completed inference in the preserved, failed overlay attempt."""
from pathlib import Path
import hashlib
import json

root = Path('/root/autodl-tmp/drivingeditor_repro/campaign/old_raw_teacher_joint_20261123_r9')
old = root / 'raw_road_semantic_v1'
out = old / 'failure_receipt.json'
assert not out.exists()
pins = {}
for name in ('labels_model_Brown.png', 'road_eroded_model_Brown.png'):
    path = old / '086' / name
    pins[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
assert pins['labels_model_Brown.png']['sha256'] == '146c9547bb8e67cdb9f0388be8e7a5c855635b9f063f3b8bc9a98809f2bb69a2'
out.write_text(json.dumps({
    'schema': 'R9.semantic_partial_execution_receipt.v1', 'complete': False,
    'outcome': 'one semantic inference completed; subsequent diagnostic overlay failed',
    'error': {'type': 'ValueError', 'message': 'assignment destination is read-only',
              'stage': 'diagnostic overlay after saving raw086 labels and eroded road mask'},
    'semantic_image_inference_calls': 1, 'new_diffusion_calls': 0,
    'nativeF': 0, 'nativeB': 0, 'optimizer_updates': 0,
    'preserved_outputs': pins,
    'recovery': 'v2 copied the writable overlay array, reused this exact086 prediction and inferred110 once; two total semantic calls across v1+v2'
}, indent=2) + '\n')
print(out)
