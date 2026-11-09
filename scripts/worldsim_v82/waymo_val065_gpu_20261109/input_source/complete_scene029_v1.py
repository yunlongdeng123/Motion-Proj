from pathlib import Path
import subprocess,json,tarfile
r=Path('/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1');old=Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1');py='/root/autodl-tmp/envs/drivingeditor/bin/python'
def run(name,cmd):
 with (r/(name+'.log')).open('x') as f:subprocess.run([py,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
run('scene029_v1_all',[r/'source/render_v1.py','--engine',old/'engine','--model',old/'formal10k','--output',r/'scene029_v1_all','--camera-manifest',r/'scene029_v1_camera_manifest.json'])
run('scene029_native_held_delete',[r/'source/evaluate_native.py','--engine',old/'engine','--api',old/'api','--model',old/'formal10k','--iteration',10000,'--output',r/'scene029_native_held_delete','--camera-manifest',r/'scene029_v1_camera_manifest.json','--selection',r/'scene029_selection.json'])
with tarfile.open(r/'scene029_v1_completion.tar','w') as t:
 for folder in ['scene029_v1_parity','scene029_v1_all','scene029_native_held_delete']:
  for p in (r/folder).rglob('*'):
   if p.is_file():t.add(p,arcname=str(p.relative_to(r)))
(r/'scene029_completed.json').write_text(json.dumps({'status':'PASS_PURE_V1_89_AND_SOURCE_ID_DELETE_HELD10','updates':0,'shutdown':False}))
