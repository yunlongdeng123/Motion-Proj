import json,zipfile
from pathlib import Path
from prune_storage import describe,AUDIT,ROOT
D=ROOT/'models/worldsim_v74_mainfig'
rows=[]; checks=[]
for model,subdir in [('dvgt2.pt','dvgt2_chunks'),('vggt_omega_1b_512.pt','omega512_chunks')]:
 target=D/model; parts=sorted((D/subdir).glob('*.part'))
 assert len(parts)==8
 offset=target.stat().st_size-sum(p.stat().st_size for p in parts)
 assert offset>=0
 initial_offset=offset
 with target.open('rb') as whole:
  whole.seek(offset)
  for p in parts:
   with p.open('rb') as f:
    while b:=f.read(8*1024*1024):
     assert whole.read(len(b))==b,(p,offset)
   rows.append(describe(p,'download_chunk',{'complete_checkpoint':str(target),'byte_offset':offset,'length':p.stat().st_size,'operation':'从完整 checkpoint 的对应字节范围复制；正常推理无需恢复分片'}))
   offset+=p.stat().st_size
 assert offset==target.stat().st_size
 with zipfile.ZipFile(target) as z: entries=len(z.infolist())
 checks.append({'model':str(target),'parts':8,'initial_offset':initial_offset,'bytewise_equal':True,'zip_entries':entries})
 print(json.dumps(checks[-1]),flush=True)
ck=D/'noksr/Carla_Serial_best.ckpt'
meta=json.loads((D/'noksr/download.json').read_text())
assert ck.stat().st_size==meta['bytes']
with zipfile.ZipFile(ck) as z: assert len(z.infolist())>1
p=D/'noksr_checkpoints.zip39l_2pb8.part'
rows.append(describe(p,'abandoned_partial_download',{'retained_checkpoint':str(ck),'download_metadata':str(D/'noksr/download.json'),'archive_index':str(D/'noksr/archive_index.json'),'operation':'未完成全模型压缩包，不是实验输入；所用 Carla Serial 权重已独立提取并保留'}))
(AUDIT/'download-plan.json').write_text(json.dumps({'checks':checks,'files':rows},indent=2,ensure_ascii=False))
print(json.dumps({'files':len(rows),'GiB':sum(r['bytes'] for r in rows)/2**30}),flush=True)
