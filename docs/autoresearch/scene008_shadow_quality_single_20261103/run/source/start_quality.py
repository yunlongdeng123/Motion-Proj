"""Validate 3D shadow coverage, smoke the revised entry and execute its queue."""
import os,json,subprocess,sys
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent

def call(args,log):
    with (RUN/log).open('x') as f:
        subprocess.run([sys.executable,str(RUN/'source/local_runtime.py')]+args,cwd=RUN,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    rec=json.loads((RUN/'inputs/receipt.json').read_text());assert rec['observed_shadow_core_coverage']==1.
    call(['--output','smoke','--steps','32'],'smoke.log')
    call(['--output','smoke_restore','--resume','smoke/checkpoint000032.pt','--steps','33'],'smoke_restore.log')
    subprocess.run([sys.executable,str(RUN/'source/run_queue.py')],cwd=RUN,check=True)
    with (RUN/'assess.log').open('x') as f:
        subprocess.run([sys.executable,str(RUN/'source/assess_quality.py')],cwd=RUN,stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
