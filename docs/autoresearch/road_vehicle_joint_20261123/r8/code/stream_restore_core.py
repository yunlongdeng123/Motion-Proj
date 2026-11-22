"""Resume the pinned main checkpoint with append-only streaming retries."""
from pathlib import Path
import json,os,runpy,sys,urllib.request

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
BASE=Path('/root/autodl-tmp/dggt/.preparation')


class Redirects(urllib.request.HTTPRedirectHandler):
    http_error_308=urllib.request.HTTPRedirectHandler.http_error_302


urllib.request.install_opener(urllib.request.build_opener(Redirects()))
manifest=json.loads((BASE/'assets_core.json').read_text())
for item in manifest['assets']:
    item['download_url']=item['official_url']+'?download=true'
path=ROOT/'runtime/core_official_stream_manifest.json'
path.write_text(json.dumps(manifest,indent=2)+'\n')
sys.argv=[str(BASE/'download_assets.py'),str(path),'--state',str(ROOT/'runtime/core_stream_restore.json')]
runpy.run_path(str(BASE/'download_assets.py'),run_name='__main__')
