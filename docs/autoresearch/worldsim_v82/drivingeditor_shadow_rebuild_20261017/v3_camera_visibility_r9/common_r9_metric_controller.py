"""Read-only common-domain attribution: filtering the metric must not masquerade as training gain."""
from pathlib import Path
import subprocess,json,datetime
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');py='/root/autodl-tmp/envs/drivingeditor/bin/python';state=r/'common_r9_metric_controller.json'
runs=[('E3_source_ownership_s0',8000,'commonR9_E3_008000'),('E4_road_identity_s0',9000,'commonR9_E4_009000'),('E5_ray_visible_from8k',9000,'commonR9_E5_009000')]
def write(status,**kw):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));p.replace(state)
try:
 for run,step,out in runs:
  write('RUNNING',run=run,global_step=step,output=out)
  subprocess.run([py,str(r/'fullscene_ownership_audit_common_r9.py'),'--run',run,'--step',str(step),'--implementation','implementation_r8',
    '--data','data_r5_ray_visible_road','--certified-road','--all-identities','--output',out],check=True,cwd=r)
 summaries=[]
 for run,step,out in runs:
  s=json.loads((r/out/'summary.json').read_text());pixels=sum(x['road_pixels'] for x in s['frames'])
  actors=[]
  for a in [5,6,12,18,25,26,37]:
   rows=[next(y for y in x['actors'] if y['actor']==a) for x in s['frames']]
   body=sum(x['body_pixels'] for x in rows);road=sum(x['measured_road'] for x in rows);bm=sum(x['own_body'] for x in rows)
   actors.append(dict(actor=a,road_visible_alpha_mass=road,road_visible_alpha_mean=road/max(1,pixels),known_body_pixels=body,
       known_body_visible_alpha_mean=bm/max(1,body),unknown_mask_road_mass=sum(x['measured_road'] for x in rows if not x['mask_known'])))
  summaries.append(dict(run=run,model_step=step,road_certificate_pixels=pixels,actors=actors))
 if len({x['road_certificate_pixels'] for x in summaries})!=1:raise ValueError('metric domains differ')
 p=r/'commonR9_comparison.json';p.write_text(json.dumps(dict(train_only=True,parameters_updated=False,common_domain='data_r5_ray_visible_road.road_certified',
  old_R5_domain_caveat='includes potential camera-occluded road projections;96.6percent old unknown fraction is not a true visible-road leakage ratio',
  scope='same exact245731 positive measured road pixels for all3 frozen models; not independent clean/editGT',models=summaries),indent=2))
 write('WAITING_COMMON_R9_METRIC_REVIEW',outputs=[x[2] for x in runs],comparison='commonR9_comparison.json',training_updates=0)
except BaseException as exc:
 write('FAILED',error=repr(exc));raise
