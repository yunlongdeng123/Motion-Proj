"""有界保存/恢复/下一阶段接线与1000步局部RGB拟合。"""
from pathlib import Path
import subprocess,json,time,os
RUN=Path(__file__).resolve().parent.parent
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'

def write(value):
    p=RUN/'controller_status.json';tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)

def main():
    start=time.monotonic();records=[]
    phases=[('smoke32',['--output','smoke32_r1','--steps','32','--smoke']),
            ('restore2',['--output','restore2_r1','--steps','34','--resume',str(RUN/'smoke32_r1/checkpoint000032.pt'),'--smoke']),
            ('formal1000',['--output','formal_r1','--steps','1000'])]
    for phase,args in phases:
        if (RUN/'STOP_REQUESTED').exists():
            write(dict(stage='stopped',reason='STOP_REQUESTED',records=records));return
        write(dict(stage=phase,records=records,elapsed=time.monotonic()-start))
        with (RUN/'evidence'/f'{phase}.log').open('x') as log:
            result=subprocess.run([PY,str(RUN/'source/fit_vehicle.py'),*args],stdout=log,stderr=subprocess.STDOUT)
        item=dict(phase=phase,returncode=result.returncode,seconds=time.monotonic()-start);records.append(item)
        if result.returncode:
            write(dict(stage='failed',records=records));return
        out=args[args.index('--output')+1]
        receipt=json.loads((RUN/out/'receipt.json').read_text())
        if not receipt['complete']:
            write(dict(stage='stopped',records=records,receipt=receipt));return
    write(dict(stage='complete',records=records,elapsed=time.monotonic()-start))

if __name__=='__main__':main()
