import gzip,json
from pathlib import Path
from prune_storage import AUDIT,ROOT
from restore_v4_sweeps import restore

rows=[]
with gzip.open(AUDIT/'sweeps-plan.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line);rows.append({'filename':r['restore']['member'],'bytes':r['bytes'],'shard':Path(r['restore']['archive']).name})
scratch=AUDIT/'restore_probe'
result=restore(rows,scratch,True,1)
for name in result['restored']:
 p=Path(name);source=ROOT/'data/worldsim_v4/drivestudio_raw_trainval'/p.relative_to(scratch)
 with p.open('rb') as x,source.open('rb') as y:
  while b:=x.read(1024*1024):assert b==y.read(len(b))
  assert not y.read(1)
 result['probe_original']=str(source);result['bytewise_equal']=True
(AUDIT/'restore-probe-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
