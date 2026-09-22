import json
from pathlib import Path
from prune_storage import ROOT,AUDIT,describe

base=ROOT/'runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1';rows=[];checks=[]
for location in [base/'assets',base/'lidar_policy/assets']:
 for partdir in sorted((location/'range_parts').iterdir()):
  complete=location/partdir.name
  if not complete.is_file():continue
  for p in sorted(partdir.glob('*.part')):
   offset=int(p.stem)
   with complete.open('rb') as whole,p.open('rb') as part:
    whole.seek(offset)
    while b:=part.read(1024*1024):assert whole.read(len(b))==b,p
   rows.append(describe(p,'legacy_download_chunk',{'complete_checkpoint':str(complete),'byte_offset':offset,'operation':'从完整 checkpoint 读取对应字节；实验无需分片'}))
  checks.append({'complete_checkpoint':str(complete),'parts_equal':True})
p=base/'assets/ltf_seed_0.ckpt.partial';complete=p.with_suffix('')
assert complete.is_file() and complete.stat().st_size>p.stat().st_size
rows.append(describe(p,'abandoned_partial_download',{'complete_checkpoint':str(complete),'operation':'保留已完整下载的同名 checkpoint，旧中断副本无须恢复'}))
(AUDIT/'sim-download-plan.json').write_text(json.dumps({'files':rows,'checks':checks},ensure_ascii=False,indent=2))
print(json.dumps({'files':len(rows),'GiB':sum(r['allocated'] for r in rows)/2**30}),flush=True)
