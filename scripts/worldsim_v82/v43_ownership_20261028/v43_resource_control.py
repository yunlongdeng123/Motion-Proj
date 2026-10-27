import json,shutil
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
def require_start():
    assert not (ROOT/'USER_STOP').exists(),'User stop requested'
    auth=json.loads((ROOT/'authorization.json').read_text());assert auth['allow_v43_training'] and not auth['quota_stop_enabled']
    assert shutil.disk_usage(str(ROOT)).free>25*1024**3,'Preserve space reserve and scene recovery states'
