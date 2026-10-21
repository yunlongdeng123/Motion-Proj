"""Serialize frozen native diagnostics over one SSH worker; preserve all failures."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent
ARMS=('full','real_only','prior_only','ownership_only')
CMD='PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python /root/autodl-tmp/gsdiag_gpu_20261021_r1/gpu_backend.py --stdio'

def js(path,key,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(value,separators=(',',':'),allow_nan=False)+';')

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
    started=time.monotonic();queries=[];counterfactuals=[];logs=(out/'ssh_stderr.log').open('x',encoding='utf-8')
    proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','wm-vgpu-1011',CMD],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs,text=True,encoding='utf-8')
    lines=queue.Queue()
    def reader():
        for line in proc.stdout:lines.put(line)
        lines.put(None)
    threading.Thread(target=reader,daemon=True).start()
    def query(q):
        proc.stdin.write(json.dumps(q)+'\n');proc.stdin.flush()
        line=lines.get(timeout=300)
        if line is None:raise RuntimeError('remote GPU worker ended; see preserved stderr')
        value=json.loads(line)
        if value.get('error'):raise RuntimeError(value['error'])
        return value
    try:
        for arm in ARMS:
            requests=[(raw,x,y,'deleted_five') for raw in (89,102,131,137) for x,y in ((120,210),(255,210),(220,280))]
            requests += [(102,x,y,mode) for x,y in ((90,255),(255,240),(365,280)) for mode in ('deleted_five','joint','ordinary','road_only','dynamic')]
            requests += [(102,195,168,mode) for mode in ('joint','dynamic')]
            special={}
            for raw,x,y,mode in requests:
                q=dict(arm=arm,raw=raw,x=x,y=y,mode=mode)
                value=query(q);key=f'gpu/{arm}/{raw}/{x}/{y}/{mode}';path=f'data/gpu/{arm}_{raw}_{x}_{y}_{mode}.js'
                js(out/path,key,value)
                queries.append(dict(key=key,path=path,arm=arm,raw=raw,pixel=[x,y],mode=mode,
                    trace_errors=value['replay_errors'],saved_replay=value['saved_final49_replay_errors'],id_weight_validation=value.get('id_weight_validation'),accepted=value['accepted_count']))
                if raw==102 and mode=='deleted_five' and (x,y) in ((90,255),(255,240),(365,280)):special[x,y]=value
                print(json.dumps(dict(stage='trace',arm=arm,done=len(queries),query=q,errors=value['replay_errors'])),flush=True)
            for (x,y),trace in special.items():
                top=sorted((r for r in trace['top'] if r['branch']=='ordinary'),key=lambda r:sum(v*v for v in r['color_contribution']),reverse=True)[:20]
                ids=[v['id'] for v in top]
                if not ids:raise RuntimeError('preset lacks ordinary contributors; do not fabricate removal')
                value=query(dict(action='counterfactual',arm=arm,raw=102,x=x,y=y,mode='deleted_five',ids=ids))
                temporal=[]
                for raw in (89,131,137):
                    other=query(dict(action='counterfactual',arm=arm,raw=raw,x=x,y=y,mode='deleted_five',ids=ids))
                    temporal.append(dict(raw=raw,metrics=other['metrics'],pixel_scope='same image coordinates; not a tracked world surface'))
                value['temporal_controls']=temporal
                value['selection_rule']='top20 ordinary background by L2 color contribution at this chosen raw102 pixel; diagnostic intervention, not automatic repair'
                key=f'cf/{arm}/102/{x}/{y}';path=f'data/counterfactual/{arm}_102_{x}_{y}.js';js(out/path,key,value)
                counterfactuals.append(dict(key=key,path=path,arm=arm,raw=102,pixel=[x,y],removed=len(ids),metrics=value['metrics'],temporal_controls=temporal))
                print(json.dumps(dict(stage='counterfactual',arm=arm,pixel=[x,y],done=len(counterfactuals))),flush=True)
        proc.stdin.close();proc.wait(timeout=30)
        if proc.returncode:raise RuntimeError('worker failed after output')
        receipt=dict(status='NATIVE_CUDA_TRACE_AND_FULL_RERASTERIZATION_COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            queries=queries,counterfactuals=counterfactuals,query_count=len(queries),counterfactual_count=len(counterfactuals),
            temporal_control_render_count=len(counterfactuals)*3,optimizer_updates=0,checkpoints_modified=False,hidden_texture_GT=False,human_verdict=None,
            elapsed_s=time.monotonic()-started,remote='wm-vgpu-1011',global_gpu_slot_serialized=True)
        (out/'gpu_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
        js(out/'data/gpu_index.js','gpu_index',dict(queries={q['key']:q['path'] for q in queries},counterfactuals={q['key']:q['path'] for q in counterfactuals}))
        print(json.dumps(dict(status=receipt['status'],queries=len(queries),counterfactuals=len(counterfactuals),elapsed_s=receipt['elapsed_s'])),flush=True)
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(e),completed_queries=len(queries),completed_cf=len(counterfactuals),optimizer_updates=0),indent=2),encoding='utf-8')
        raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill()
        logs.close()

if __name__=='__main__':main()
