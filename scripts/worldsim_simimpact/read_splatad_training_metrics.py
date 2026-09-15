import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
fit=N/'fits/scene-0004/splatad/native-r2'
ea=EventAccumulator(str(fit),size_guidance={'scalars':0});ea.Reload()
out={}
for tag in ea.Tags()['scalars']:
    if 'Eval' in tag:
        out[tag]=[{'step':x.step,'value':x.value} for x in ea.Scalars(tag)]
(N/'training_eval_snapshot.json').write_text(json.dumps({'fit':str(fit),'incomplete_fit':True,'scalars':out},indent=2))
print(json.dumps({k:v[-1] for k,v in out.items() if v},indent=2))
