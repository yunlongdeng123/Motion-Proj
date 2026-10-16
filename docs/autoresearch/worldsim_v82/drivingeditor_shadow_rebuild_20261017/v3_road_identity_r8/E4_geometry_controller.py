from pathlib import Path
import datetime,json,subprocess
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');py='/root/autodl-tmp/envs/drivingeditor/bin/python';state=r/'E4_geometry_controller.json'
def record(status,**extra):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2));p.replace(state)
try:
 record('RUNNING',phase='bounded_geometry',parent_global_step=8000,next_gate=9000)
 subprocess.run([py,str(r/'implementation_r8/shadow_rebuild_v3/run.py'),'train','--data',str(r/'data_r4_road_identity'),
 '--config',str(r/'E4_road_identity_s0/resolved_config.json'),'--output',str(r/'E4_road_identity_s0'),
 '--resume',str(r/'E4_road_identity_s0/latest.pt'),'--execute','--allow-partial-masks','--max-updates','1000'],check=True,cwd=r)
 record('RUNNING',phase='same7_preview',global_step=9000)
 subprocess.run([py,str(r/'checkpoint_preview_r6.py'),'--step','9000','--run','E4_road_identity_s0','--implementation','implementation_r8'],check=True,cwd=r)
 record('RUNNING',phase='full49_canonical_geometry_review',global_step=9000)
 subprocess.run([py,str(r/'geometry_sequence_review_r6.py'),'--run','E4_road_identity_s0','--step','9000',
 '--implementation','implementation_r8','--data','data_r4_road_identity','--output','E4_geometry49_009000'],check=True,cwd=r)
 record('RUNNING',phase='fullscene_perGaussian_attribution',global_step=9000)
 subprocess.run([py,str(r/'fullscene_ownership_audit_r6.py'),'--run','E4_road_identity_s0','--step','9000',
 '--implementation','implementation_r8','--output','ownership_009000_r8','--road-evidence','road_evidence_r5','--all-identities'],check=True,cwd=r)
 record('WAITING_9000_STEP_ROAD_IDENTITY_FULL49_REVIEW',phase='geometry',global_step=9000,material_started=False)
except BaseException as exc:
 record('FAILED',error=repr(exc));raise
