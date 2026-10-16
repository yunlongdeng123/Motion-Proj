from pathlib import Path
import zipfile,json,subprocess,datetime
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2")
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
 value=cmd.read_bytes().replace(b'\0',b' ').decode(errors='replace') if cmd.exists() else ''
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
