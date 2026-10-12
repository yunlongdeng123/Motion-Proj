import json
import math
from datetime import datetime, timezone
from pathlib import Path

base = Path('/root/autodl-tmp/drivingeditor_repro')
run = base / 'runs/0017085_demo_000_020_dynamic_30000'
metrics_path = run / 'eval/train_30000_render/metrics.json'
metrics = json.loads(metrics_path.read_text())
assert metrics['iteration'] == 30000
assert all(math.isfinite(metrics[k]) for k in ('psnr', 'ssim', 'lpips'))
required = ('chkpnt30000.pth', 'dynamic_chkpnt30000.pth', 'env_light_chkpnt30000.pth')
for name in required:
    assert (run / name).is_file() and (run / name).stat().st_size > 0, name
result = {'status': 'training_completed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
          'iterations': 30000, 'run': str(run), 'metrics': metrics,
          'checkpoints': {name: (run / name).stat().st_size for name in required},
          'protocol': 'Official code defaults, including_dynamic=True, verified video frames0..20',
          'evaluation_scope': 'Training views only; not paper main-experiment or held-out evaluation',
          'editing_render_status': 'Not validated by this completion check'}
(base / 'logs/formal_demo_completion.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
