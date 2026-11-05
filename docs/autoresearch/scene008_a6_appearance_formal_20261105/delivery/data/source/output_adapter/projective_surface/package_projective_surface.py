"""Ledger of prototype deliverables, excluding retained heavy/failed replay data."""
import hashlib,json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105/output_adapter/projective_surface')
excluded={'evaluation_before_pool_fix'}
files=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or excluded.intersection(p.relative_to(ROOT).parts):continue
    if p.name=='preview_manifest.json' or p.name.endswith('raster_hr.npz') or p.name.endswith('native_groups.npz'):continue
    files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'preview_manifest.json').write_text(json.dumps(dict(files=sorted(files,key=lambda f:f['path'])),indent=2)+'\n')
(ROOT/'engineering_overhead.json').write_text(json.dumps(dict(
    PVD_updates=0,colour_optimizer_updates=0,preparation_seconds=3.655536413192749,
    successful_fixed49_evaluation_seconds=39.3144268989563,
    additional_evaluation_replays=1,
    reason='First full49 native rendering completed; pooled metric schema missed max/threshold statistics. Fixed scalar record schema and replayed. Original logs/source/images retained.',
    old_source='source/output_adapter/projective_surface/evaluate_projective_surface.py.before_metric_pool_fix',
    retained_failed_pass='output_adapter/projective_surface/evaluation_before_pool_fix',
    no_parameter_search=True),indent=2)+'\n')
print(json.dumps(dict(files=len(files))))
