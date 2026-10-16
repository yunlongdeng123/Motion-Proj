"""Read-only: separate measured-road leakage under known vs unknown Actor body labels."""
from pathlib import Path
import sys,json
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r7/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import write
d=Dataset(r/'data_r3_ownership','cpu',True)
audit=r/'ownership_008000_r7/summary.json'
if not audit.exists():raise RuntimeError('wait until frozen8k attribution completes')
frames=json.loads(audit.read_text())['frames'];result={a:dict(actor=a,known_road_mass=0.,unknown_road_mass=0.,known_frames=0,unknown_frames=0) for a in [5,6,12,18,25,26,37]}
for row in frames:
 f=d.frame(next(i for i,x in enumerate(d.rows) if x['frame_id']==row['frame_id']))
 for a in row['actors']:
  prefix='known' if bool(f.mask_known[a['actor']]) else 'unknown';result[a['actor']][prefix+'_road_mass']+=a['measured_road'];result[a['actor']][prefix+'_frames']+=1
v=dict(training_updates=0,read_only=True,world_road_evidence='road_evidence_r5 actual measured certified pixels',actors=list(result.values()),
       interpretation='unknown Actor body mask correctly gets no maskloss; certified actual-road pixels can nevertheless supply independent positive road identity, without declaring all unknown pixels background')
write(r/'road_supervision_domains_008000_r7.json',v);print(json.dumps(v))
