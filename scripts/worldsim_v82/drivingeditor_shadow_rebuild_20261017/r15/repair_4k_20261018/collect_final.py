"""Collect only after all finite GPU queues finished. Does not start GPU work."""
from pathlib import Path
import json,subprocess,tarfile,shutil
W=Path(__file__).parent;I=W.parent/'identity_completion';O=W.parents[1]/'outputs'
Q='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15/repair_4k_20261018'
meta=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011','/root/autodl-tmp/envs/drivingeditor/bin/python '+Q+'/pack_final.py'],text=True))
local=W/Path(meta['path']).name;subprocess.run(['scp','wm-vgpu-1011:'+meta['path'],str(local)],check=True)
assert local.stat().st_size==meta['bytes'];dest=W/'final_download';dest.mkdir(exist_ok=True)
with tarfile.open(local) as tar:
    assert all(m.isfile() and not m.issym() and not m.islnk() and (dest/m.name).resolve().is_relative_to(dest.resolve()) for m in tar.getmembers())
    tar.extractall(dest)
shutil.copytree(dest/'renders_repair4k',O/'identity_comparison_20261018/assets/repair4k',dirs_exist_ok=True)
shutil.copytree(dest/'branches_repair4k',I/'branches/branches_repair4k',dirs_exist_ok=True)
shutil.copytree(dest/'metadata',W/'final_metadata',dirs_exist_ok=True)
shutil.copy2(dest/'metadata/geometry_comparison.json',W/'geometry_comparison.json')
shutil.copy2(dest/'metadata/queue_status.json',W/'queue_status.json')
shutil.copy2(dest/'metadata/train/progress.json',W/'live_progress.json')
print('Repair4K actual evidence copied; build player only after frozen collector has finished.')
