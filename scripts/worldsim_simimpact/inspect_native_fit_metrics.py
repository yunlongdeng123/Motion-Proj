"""读取既有拟合留出指标作成本和收敛复核，不新增模型推理。"""
import argparse,json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ap=argparse.ArgumentParser();ap.add_argument('--fit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
ea=EventAccumulator(str(a.fit),size_guidance={'scalars':100,'images':0,'histograms':0,'tensors':0}).Reload()
tags=ea.Tags()['scalars'];selected=[t for t in tags if 'Eval Images Metrics' in t or 'Eval Images' in t]
out={'role':'Existing official training-time held-out metrics, no new inference or downstream condition. Squared-distance metrics retain official units.',
     'fit':str(a.fit),'progress':json.loads((a.fit/'progress.json').read_text()),
     'fit_manifest':json.loads((a.fit/'fit_manifest.json').read_text()),'scalar_tags':tags,
     'heldout_metrics':{t:[{'step':e.step,'value':e.value,'wall_time':e.wall_time} for e in ea.Scalars(t)] for t in selected}}
a.out.write_text(json.dumps(out,indent=2))
print(json.dumps({'progress':out['progress'],'metrics':{k:v[-1] for k,v in out['heldout_metrics'].items() if v}},indent=2))
