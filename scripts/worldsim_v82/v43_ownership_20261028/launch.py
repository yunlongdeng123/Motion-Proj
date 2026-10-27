"""Detached launch with PID+start identity, never duplicate an existing receipt."""
import subprocess,json,sys
from pathlib import Path
name=sys.argv[1];args=sys.argv[2:]
script='''import subprocess,json,time,os
from pathlib import Path
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');name=NAME;args=ARGS
p=R/(name+'_dispatch_receipt.json');assert not p.exists(), 'No-clobber launch receipt'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-u',str(R/'code'/args[0])]+args[1:]
with (R/(name+'.log')).open('w') as f:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
time.sleep(.1)
d=dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().split()[21],command=cmd,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
p.write_text(json.dumps(d,indent=2));print(json.dumps(d))
'''.replace('name=NAME;args=ARGS','name='+repr(name)+';args='+repr(args))
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace');print(r.stdout.decode())
