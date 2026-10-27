"""Only the three reviewed development cases. No training/legacy queue."""
import subprocess,sys,time,json,os
from pathlib import Path
from segmentation import ROOT,CONTEXTS,write
state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='running',contexts=[])
for ctx in CONTEXTS:
    cmd=[sys.executable,'-u',str(ROOT/'code/attribute_gaussians.py'),'--context',ctx,'--attempt','attribution_v2']
    state.update(context=ctx,command=cmd);write(ROOT/'attribution_dispatch.json',state)
    with (ROOT/('attribute_'+ctx+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state.update(child=p.pid);write(ROOT/'attribution_dispatch.json',state);code=p.wait()
    state['contexts'].append(dict(context=ctx,returncode=code));state.update(child=None)
    write(ROOT/'attribution_dispatch.json',state)
    if code:raise RuntimeError((ctx,code))
state.update(phase='complete');write(ROOT/'attribution_dispatch.json',state)
