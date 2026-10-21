"""Supplement actual visible raw102 residual locations; CPU reference only."""
import io
import json
import sys
import tarfile
from backend import Adapter, ARMS

def run():
    a=Adapter();t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|');records=[]
    def put(name,value):
        data=value.encode();m=tarfile.TarInfo(name);m.size=len(data);t.addfile(m,io.BytesIO(data))
    for arm in ARMS:
        print('supplement '+arm,file=sys.stderr,flush=True)
        for x,y in ((90,255),(255,240),(365,280)):
            value=a.ray(arm,102,x,y)
            key=f'ray/{arm}/102/{x}/{y}'
            put(f'data/rays/{arm}_102_{x}_{y}.js','window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(value,allow_nan=False,separators=(',',':'))+';')
            records.append(dict(arm=arm,pixel=[x,y],replay_errors=value['replay_errors']))
    put('supplement_receipt.json',json.dumps(dict(status='CPU_RESIDUAL_CASES_EXPORTED',raw=102,cases=records,gpu_used=False,parameter_updates=0,original_assets_modified=False),indent=2))
    t.close()
if __name__=='__main__':run()
