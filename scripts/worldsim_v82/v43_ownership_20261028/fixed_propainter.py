"""Identical fixed backend; only swap the authorization scope to this V43 run."""
import sys,hashlib,json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
SOURCE=ROOT.parent/'persistent_world_20261027/code/inpaint_baseline.py'
sys.path.insert(0,str(SOURCE.parent))
original=SOURCE.read_text();assert original.count('from resource_control import require_start')==1
code=original.replace('from resource_control import require_start','from v43_resource_control import require_start')
record=dict(source=str(SOURCE),source_sha256=hashlib.sha256(original.encode()).hexdigest(),only_change='resource guard import: explicit current-user quota override, old queue remains paused',backend_model_and_parameters_unchanged=True)
(ROOT/'fixed_backend_adapter_receipt.json').write_text(json.dumps(record,indent=2))
exec(compile(code,str(SOURCE),'exec'),dict(__name__='__main__',__file__=str(SOURCE)))
