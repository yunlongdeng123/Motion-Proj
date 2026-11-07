"""Run the already-authorized fusion400 after the direct400 worker exits."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from shared_fit_common import OUT,write


def main():
    script=Path(__file__).with_name('fit_shared_material.py').resolve()
    lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();pids=[]
    for line in lines:
        words=line.strip().split(None,1)
        if len(words)==2 and str(script) in words[1] and words[1].endswith('--phase formal --variant direct'):
            pids.append(int(words[0]))
    if not pids and not (OUT/'direct/export_receipt.json').exists():raise RuntimeError('Authorized direct worker not found')
    write(OUT/'formal_queue_start.json',dict(waited_direct_processes=pids,controller_PID=os.getpid(),stopped_processes=[]))
    while any((Path('/proc')/str(pid)).exists() for pid in pids):time.sleep(1)
    receipt=json.loads((OUT/'direct/formal/receipt.json').read_text())
    assert receipt['formal_steps']==400 and receipt['forwards']==800 and receipt['backwards']==800
    if (OUT/'fusion/formal').exists():raise FileExistsError('Preserve prior fusion formal worker')
    command=['/root/autodl-tmp/envs/drivingeditor/bin/python','-u',str(script),'--phase','formal','--variant','fusion']
    print(json.dumps(dict(starting_authorized_fusion400=command)),flush=True)
    result=subprocess.run(command)
    write(OUT/'formal_queue_exit.json',dict(returncode=result.returncode,command=command,other_tasks_modified=False))
    sys.exit(result.returncode)


if __name__=='__main__':main()
