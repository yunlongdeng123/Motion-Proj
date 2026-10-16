"""One bounded visibility-normalization trial; no material stage without review."""
from pathlib import Path
import datetime,json,subprocess
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');py='/root/autodl-tmp/envs/drivingeditor/bin/python';state=r/'E6_geometry_controller.json'
def record(status,**extra):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2));p.replace(state)
try:
 record('RUNNING',phase='bounded_visible_body_identity',parent_global_step=9000,next_gate=10000,discarded_updates_charged=1000)
 subprocess.run([py,str(r/'implementation_r10/shadow_rebuild_v3/run.py'),'train','--data',str(r/'data_r5_ray_visible_road'),
  '--config',str(r/'E6_visible_identity_s0/resolved_config.json'),'--output',str(r/'E6_visible_identity_s0'),
  '--resume',str(r/'E6_visible_identity_s0/latest.pt'),'--execute','--allow-partial-masks','--max-updates','1000'],check=True,cwd=r)
 record('RUNNING',phase='same7_preview',global_step=10000,charged_updates=11000)
 subprocess.run([py,str(r/'checkpoint_preview_r6.py'),'--step','10000','--run','E6_visible_identity_s0','--implementation','implementation_r10'],check=True,cwd=r)
 record('RUNNING',phase='full49_canonical_geometry_review',global_step=10000)
 subprocess.run([py,str(r/'geometry_sequence_review_r6.py'),'--run','E6_visible_identity_s0','--step','10000',
  '--implementation','implementation_r10','--data','data_r5_ray_visible_road','--output','E6_geometry49_010000'],check=True,cwd=r)
 record('RUNNING',phase='common_R9_fullscene_perGaussian_attribution',global_step=10000)
 subprocess.run([py,str(r/'fullscene_ownership_audit_common_r9.py'),'--run','E6_visible_identity_s0','--step','10000',
  '--implementation','implementation_r10','--data','data_r5_ray_visible_road','--certified-road','--all-identities','--output','commonR9_E6_010000'],check=True,cwd=r)
 baseline=json.loads((r/'commonR9_comparison.json').read_text());s=json.loads((r/'commonR9_E6_010000/summary.json').read_text());pixels=sum(x['road_pixels'] for x in s['frames']);actors=[]
 for a in [5,6,12,18,25,26,37]:
  rows=[next(y for y in x['actors'] if y['actor']==a) for x in s['frames']];body=sum(x['body_pixels'] for x in rows);road=sum(x['measured_road'] for x in rows)
  actors.append(dict(actor=a,road_visible_alpha_mass=road,road_visible_alpha_mean=road/max(1,pixels),known_body_pixels=body,
   known_body_visible_alpha_mean=sum(x['own_body'] for x in rows)/max(1,body),unknown_mask_road_mass=sum(x['measured_road'] for x in rows if not x['mask_known'])))
 if pixels!=245731 or any(x['road_certificate_pixels']!=pixels for x in baseline['models']):raise ValueError('common metric domains differ')
 baseline['models'].append(dict(run='E6_visible_identity_s0',model_step=10000,road_certificate_pixels=pixels,actors=actors));baseline['scope']='same actual R9 road certificate domain; sequential engineering trials, not matched causal ablations; no clean/edit ground truth'
 (r/'commonR10_comparison.json').write_text(json.dumps(baseline,indent=2))
 record('WAITING_10000_STEP_VISIBLE_IDENTITY_FULL49_REVIEW',phase='geometry',global_step=10000,charged_updates=11000,material_started=False,comparison='commonR10_comparison.json')
except BaseException as exc:
 record('FAILED',error=repr(exc));raise
