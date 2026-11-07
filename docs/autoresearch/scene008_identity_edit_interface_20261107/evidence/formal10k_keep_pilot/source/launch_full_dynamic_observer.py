from pathlib import Path
import argparse,subprocess,sys,json,time
ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['formal5k','formal10k'],required=True);a=ap.parse_args()
assert not (ROOT/(a.phase+'_lifecycle_status.json')).exists()
with (ROOT/(a.phase+'_controller.log')).open('x') as log:
    proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'source/observe_full_dynamic_formal.py'),'--phase',a.phase],
        stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
(ROOT/(a.phase+'_controller.pid')).write_text(str(proc.pid)+'\n')
print(json.dumps(dict(phase=a.phase,controller_pid=proc.pid,unix=time.time(),training_launches_allowed=False)))
