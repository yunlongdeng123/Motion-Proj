"""Bounded ownership repair: global6000->7000, then preview and stop for root review."""
from pathlib import Path
import datetime,json,subprocess,sys,os
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
py='/root/autodl-tmp/envs/drivingeditor/bin/python';state=r/'E2_geometry_controller.json'
def record(status,**extra):
 v=dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
 temp=state.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2));temp.replace(state)
try:
 record('RUNNING',phase='bounded_geometry',parent_global_step=6000,next_gate=7000)
 subprocess.run([py,str(r/'implementation_r6/shadow_rebuild_v3/run.py'),'train','--data',str(r/'data_r3_ownership'),
 '--config',str(r/'E2_ownership_s0/resolved_config.json'),'--output',str(r/'E2_ownership_s0'),
 '--resume',str(r/'E2_ownership_s0/latest.pt'),'--execute','--allow-partial-masks','--max-updates','1000'],check=True,cwd=r)
 record('RUNNING',phase='same7_preview',global_step=7000)
 subprocess.run([py,str(r/'checkpoint_preview_r6.py'),'--step','7000','--run','E2_ownership_s0','--implementation','implementation_r6'],check=True,cwd=r)
 record('WAITING_7000_STEP_OWNERSHIP_VISUAL_REVIEW',phase='geometry',global_step=7000,material_started=False)
except BaseException as exc:
 record('FAILED',error=repr(exc));raise
