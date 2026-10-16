"""Collect finished fixed-camera edits and scope evidence; starts no GPU work."""
from pathlib import Path
import json, subprocess, zipfile, shutil
BASE=Path(__file__).resolve().parents[2]
WORK=BASE/'work/shadow_sequence_v2';OUT=BASE/'outputs/shadow_sequence_v2'
ROOT='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2'
remote_code='''from pathlib import Path
import zipfile,json,subprocess,datetime
root=Path("ROOT_PLACEHOLDER")
items=[]
def add(p,arc):
 if p.is_file():items.append((p,arc))
for p in (root/'anchor_edit_sweep_final').glob('*.json'):add(p,'anchor_edit_sweep_final/'+p.name)
for p in (root/'anchor_edit_sweep_final').glob('*.mp4'):add(p,'raw_videos/anchor_edit_sweep_final/'+p.name)
for kind in ['translation','yaw30']:
 for k in [0,12,24,36,48]:
  p=root/'anchor_edit_sweep_final'/kind/('%03d.png'%k)
  add(p,'anchor_edit_sweep_final/'+kind+'/'+p.name)
for raw in [106,107,108]:
 for kind in ['delete_target','confidence','static']:
  p=root/'final49_unselected_v1'/kind/('0008%03d.png'%raw)
  add(p,'evidence/edge_'+kind+'_raw%d.png'%raw)
jobs=[]
for p in root.glob('*_launch.json'):
 r=json.loads(p.read_text());pid=r['pid'];cmd=Path('/proc',str(pid),'cmdline')
 value=cmd.read_bytes().replace(b'\\0',b' ').decode(errors='replace') if cmd.exists() else ''
 jobs.append({'name':p.stem,'pid':pid,'alive':bool(value),'current_command':value})
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'nvidia_smi':gpu.stdout,'compute_processes':processes.stdout,'no_poweroff_action':True}
p=root/'completion_process_check.json';p.write_text(json.dumps(record,indent=2));add(p,'evidence/completion_process_check.json')
for name in ['final49_unselected_v1_follow.json','anchor_edit_sweep_final_launch.json','unselected_clean_v1_launch.json','training_accounting.json']:
 add(root/name,'evidence/'+name)
dest=root/'final_supplement.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED) as z:
 for p,arc in items:z.write(p,arc)
print(json.dumps({'files':len(items),'bytes':dest.stat().st_size,'jobs_active':[r for r in jobs if r['alive']]}))
'''.replace('ROOT_PLACEHOLDER',ROOT)
p=WORK/'build_supplement_remote.py';p.write_text(remote_code,encoding='utf-8')
def call(args):subprocess.run(args,check=True)
call(['scp',str(p),'wm-vgpu-1011:'+ROOT+'/build_supplement_remote.py'])
call(['ssh','wm-vgpu-1011','/root/autodl-tmp/envs/drivingeditor/bin/python',ROOT+'/build_supplement_remote.py'])
dest=WORK/'final_supplement.zip';call(['scp','wm-vgpu-1011:'+ROOT+'/final_supplement.zip',str(dest)])
with zipfile.ZipFile(dest) as z:
 for entry in z.infolist():
  rel=Path(entry.filename);assert '..' not in rel.parts and not rel.is_absolute()
  target=(WORK if rel.parts[0]=='raw_videos' else OUT)/rel
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(entry))
for src in (WORK/'raw_videos/anchor_edit_sweep_final').glob('*.mp4'):
 target=OUT/'videos/anchor_edit_sweep_final'/src.name;target.parent.mkdir(parents=True,exist_ok=True)
 call(['ffmpeg','-y','-loglevel','error','-i',str(src),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)])
for a,b in [('foreground_old_125.png','rejected_16k_red_fragments_raw125.png'),('foreground_cleared_zero_125.png','foreground_pruning_zero_update_raw125.png')]:
 if (WORK/a).exists():shutil.copy2(WORK/a,OUT/'evidence'/b)
print('Collected final fixed-camera sweep, current scope edges, and process evidence.')
