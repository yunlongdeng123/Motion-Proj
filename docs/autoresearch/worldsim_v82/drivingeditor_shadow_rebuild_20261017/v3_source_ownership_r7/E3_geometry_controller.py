"""A bounded source-ownership repair, then full49 geometry review and frozen attribution."""
from pathlib import Path
import datetime,json,subprocess
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');py='/root/autodl-tmp/envs/drivingeditor/bin/python';state=r/'E3_geometry_controller.json'
def record(status,**extra):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2));p.replace(state)
try:
 record('RUNNING',phase='bounded_geometry',parent_global_step=7000,next_gate=8000)
 subprocess.run([py,str(r/'implementation_r7/shadow_rebuild_v3/run.py'),'train','--data',str(r/'data_r3_ownership'),
 '--config',str(r/'E3_source_ownership_s0/resolved_config.json'),'--output',str(r/'E3_source_ownership_s0'),
 '--resume',str(r/'E3_source_ownership_s0/latest.pt'),'--execute','--allow-partial-masks','--max-updates','1000'],check=True,cwd=r)
 record('RUNNING',phase='same7_preview',global_step=8000)
 subprocess.run([py,str(r/'checkpoint_preview_r6.py'),'--step','8000','--run','E3_source_ownership_s0','--implementation','implementation_r7'],check=True,cwd=r)
 record('RUNNING',phase='full49_canonical_geometry_review',global_step=8000)
 subprocess.run([py,str(r/'geometry_sequence_review_r6.py'),'--run','E3_source_ownership_s0','--step','8000',
 '--implementation','implementation_r7','--data','data_r3_ownership','--output','E3_geometry49_008000'],check=True,cwd=r)
 record('RUNNING',phase='fullscene_perGaussian_attribution',global_step=8000)
 subprocess.run([py,str(r/'fullscene_ownership_audit_r6.py'),'--run','E3_source_ownership_s0','--step','8000',
 '--implementation','implementation_r7','--output','ownership_008000_r7','--road-evidence','road_evidence_r5','--all-identities'],check=True,cwd=r)
 record('WAITING_8000_STEP_FULL49_SOURCE_OWNERSHIP_REVIEW',phase='geometry',global_step=8000,material_started=False)
except BaseException as exc:
 record('FAILED',error=repr(exc));raise
