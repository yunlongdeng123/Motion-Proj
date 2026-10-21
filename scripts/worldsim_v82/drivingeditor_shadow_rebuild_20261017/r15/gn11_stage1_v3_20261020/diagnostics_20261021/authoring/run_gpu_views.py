import datetime
import json
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np
from verify import read_js
from run_gpu_export import js,CMD,ARMS

ROOT=Path(__file__).resolve().parent
out=ROOT/'gpu_views_r1';out.mkdir(exist_ok=False)
log=(out/'stderr.log').open('x',encoding='utf-8')
proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','wm-vgpu-1011',CMD],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8')
scene=read_js(ROOT.parents[1]/'outputs/GS_Reconstruction_Diagnostics_20261021_r1/data/scene.js')
lines=queue.Queue()
def reader():
    for line in proc.stdout:lines.put(line)
    lines.put(None)
threading.Thread(target=reader,daemon=True).start()
def query(q,expect_error=False):
    proc.stdin.write(json.dumps(q)+'\n');proc.stdin.flush();line=lines.get(timeout=300)
    if line is None:raise RuntimeError('worker ended')
    v=json.loads(line)
    if bool(v.get('error'))!=expect_error:raise RuntimeError(v)
    return v
views=[];negative=[]
try:
    for arm in ARMS:
        for mode in ('joint','deleted_five','ordinary','road_only','dynamic'):
            v=query(dict(action='render_view',arm=arm,raw=102,x=90,y=255,mode=mode))
            key=f'view/{arm}/102/{mode}/native';path=f'data/views/{arm}_102_{mode}_native.js'
            js(out/path,key,v);views.append(dict(key=key,path=path,mode=mode,arm=arm,novel=False,replay=v['full_frame_saved_replay_max_error'],profile=v['execution_profile']))
            print(json.dumps(views[-1]),flush=True)
        c=scene['cameras'][13];W=np.asarray(c['W']);K=c['K'];T=W[:3,3].copy();T[0]-=scene['meter_to_model']
        view=dict(R=W[:3,:3].tolist(),T=T.tolist(),intrinsics=[K[0][0],K[1][1],K[0][2],K[1][2]])
        v=query(dict(action='render_view',arm=arm,raw=102,x=90,y=255,mode='joint',view=view))
        key=f'view/{arm}/102/joint/right1m';path=f'data/views/{arm}_102_joint_right1m.js'
        js(out/path,key,v);views.append(dict(key=key,path=path,arm=arm,mode='joint',novel=True,profile=v['execution_profile']))
        print(json.dumps(views[-1]),flush=True)
    cy,sy,cp,sp=np.cos(.5),np.sin(.5),np.cos(1.05),np.sin(1.05)
    R=np.array([[cy,-sy,0],[cp*sy,cp*cy,-sp],[sp*sy,sp*cy,cp]]);T=-R@np.array([2.,0.,0.]);T[2]+=12
    view=dict(R=R.tolist(),T=T.tolist(),intrinsics=[384,384,240,160])
    v=query(dict(action='render_view',arm='full',raw=102,x=90,y=255,mode='deleted_five',view=view))
    key='view/full/102/deleted_five/orbit';path='data/views/full_102_deleted_five_orbit.js';js(out/path,key,v)
    views.append(dict(key=key,path=path,arm='full',novel=True,profile=v['execution_profile']))
    for q in [dict(action='render_view',arm='full',raw=102,x=90,y=255,mode='joint',view=dict(R=[[1,0,0],[0,-1,0],[0,0,1]],T=[0,0,0],intrinsics=[384,384,240,160])),
              dict(action='counterfactual',arm='full',raw=102,x=90,y=255,mode='deleted_five',ids=['full:000000000000:static:123']),
              dict(action='trace',arm='full',raw=138,x=90,y=255,mode='deleted_five')]:
        value=query(q,True);negative.append(dict(query=q,response=value))
    proc.stdin.close();proc.wait(timeout=30)
    receipt=dict(status='NATIVE_FULL_MODEL_VIEW_VALIDATED',views=views,negative_contract_tests=negative,
        optimizer_updates=0,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),novel_views_have_GT=False)
    (out/'view_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
except BaseException as e:
    (out/'failure.json').write_text(json.dumps(dict(error=repr(e),completed_views=len(views))),encoding='utf-8');raise
finally:
    if proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill()
    log.close()
