"""导出可由 file:// 直接读取的完整几何分块与逐帧车辆区域；只读CPU。"""
import argparse
import json
from pathlib import Path
import shutil
import time
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from cloud_sampling import Sampler, ACTORS, V1, hashes, pack
from backend import ARMS, clean


def js(root, path, key, value):
    target=root/path
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.js.tmp')
    temporary.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(clean(value),separators=(',',':'),allow_nan=False)+';',encoding='utf-8')
    temporary.replace(target)


def attributes(b,ids):
    return dict(row=ids.astype(np.int32),xyz=b['xyz'][ids],sigma=b['scale'][ids],q=b['q'][ids],
        velocity=b['vel'][ids],t=b['t'][ids],sigma_t=b['sigma_t'][ids],opacity=b['opacity'][ids],
        color=np.maximum(.5+.28209479177387814*b['dc'][ids,0],0),
        role=np.full(len(ids),0 if b['name']=='static' else 2,np.uint8),road_id=b['road'][ids].astype(np.int32))


def memberships(sampler,b,raw,order,include_box=True):
    cam=sampler.adapter.camera(raw)
    labels=np.asarray(Image.open(sampler.adapter.meta['frames'][f'0008{raw:03d}']['label_path']))
    mask=np.zeros(labels.shape,np.uint8)
    for i in range(len(ACTORS)):
        mask[binary_dilation(labels==i+1,iterations=5)]|=1<<i
    projection=np.zeros(len(order),np.uint8);box_flags=np.zeros(len(order),np.uint8)
    boxes={v['actor']:v for v in sampler.metadata['vehicles'][str(raw)]}
    for start in range(0,len(order),100000):
        ids=order[start:start+100000]
        delta=cam['timestamp']-b['t'][ids]
        active=np.exp(-.5*(delta/b['sigma_t'][ids])**2)>.05
        omega=2*np.pi/b['cycle']
        xyz=b['xyz'][ids]+b['vel'][ids]*(np.sin(delta*omega)/omega)[:,None]
        cp=xyz@cam['W'][:3,:3].T+cam['W'][:3,3]
        uv=cp[:,:2]/np.maximum(cp[:,2,None],1e-8)*[cam['K'][0,0],cam['K'][1,1]]+[cam['K'][0,2],cam['K'][1,2]]
        px=np.rint(uv).astype(np.int64)
        valid=active&(cp[:,2]>.02)&(px[:,0]>=0)&(px[:,0]<480)&(px[:,1]>=0)&(px[:,1]<320)
        local=np.zeros(len(ids),np.uint8)
        local[valid]=mask[px[valid,1],px[valid,0]]
        projection[start:start+len(ids)]=local
        for i,actor in enumerate(ACTORS if include_box else ()):
            box=boxes[actor]
            inside=(np.abs((xyz-box['center'])@np.asarray(box['axes']))<=np.asarray(box['half'])+.5*sampler.adapter.scale).all(1)&active
            box_flags[start:start+len(ids)][inside]|=1<<i
    return projection,box_flags


def export(root,arms,chunk_size=32768,resume=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sampler=Sampler();started=time.time()
    index=dict(version=1,chunk_size=chunk_size,actors=list(ACTORS),arms={},max_rate=100,
        selection='exact ceil counts; nested uint32 rank; roads always retained',gpu_used=False,optimizer_updates=0)
    js(root,'data/sampling_meta.js','samplingMeta',sampler.metadata)
    def completed(path):
        target=root/path
        if not resume or not target.exists() or target.stat().st_size<100:return False
        with target.open('rb') as stream:
            stream.seek(-1,2)
            return stream.read()==b';'
    for arm in arms:
        print('loading '+arm,flush=True)
        model=sampler.get_model(arm);meta=dict(branches={},members={},scopes=['projection'])
        orders={}
        for b in model['branches']:
            ordinary=np.flatnonzero(b['road']<0)
            order=ordinary[np.lexsort((ordinary,hashes(ordinary)))];orders[b['name']]=order
            branch=dict(total=len(b['xyz']),ordinary_total=len(order),road_total=int((b['road']>=0).sum()),
                checkpoint_sha256=b['sha'],chunks=[])
            for i,start in enumerate(range(0,len(order),chunk_size)):
                ids=order[start:start+chunk_size]
                path=f'data/geometry/{arm}/{b["name"]}_{i:03d}.js';key=f'geometry/{arm}/{b["name"]}/{i}'
                if not completed(path):js(root,path,key,pack(attributes(b,ids)))
                branch['chunks'].append(dict(path=path,key=key,start=start,count=len(ids)))
            meta['branches'][b['name']]=branch
        # 默认2%仍是独立的小文件；首次打开不读取完整几何。
        js(root,f'data/{arm}/cloud.js',arm+'/cloud',sampler.query(dict(arm=arm,raw=102)))
        for raw in range(89,138):
            key=f'geometry/{arm}/members/{raw}';path=f'data/geometry/{arm}/members_{raw}.js'
            if not completed(path):
                arrays={}
                for b in model['branches']:
                    projection,_=memberships(sampler,b,raw,orders[b['name']],include_box=False)
                    arrays[b['name']+'_projection']=projection
                js(root,path,key,pack(arrays))
            meta['members'][str(raw)]=dict(key=key,path=path)
            if raw in (89,100,102,113,125,137):print(f'{arm} raw{raw} region masks complete',flush=True)
        index['arms'][arm]=meta
        # 逐组记录断点；失败时未完成的组不发布为可用。
        js(root,'data/geometry_index.js','geometryIndex',index)
        print('complete '+arm,flush=True)
    # 后续标准导出同时携带三层初始化证据及完整逐帧深度标记。
    from export_inspection import export as export_inspection
    export_inspection(root,arms,resume=resume)
    if 'gn11_v1' in arms:
        from export_v1_ground import export as export_v1_ground
        export_v1_ground(root,resume=resume)
    source=Path(__file__).resolve().parent
    # V4分支的一键离线导出同时带上固定初始化验收；不执行训练。
    initialization_audit=source.parents[2]/'gn11_stage1_v4_20261022/initialization_audit/export_initialization_audit.py'
    if initialization_audit.exists():
        subprocess.run([sys.executable,str(initialization_audit),'--output',str(root)],check=True)
    for name in ('app.js','sampling.js','sampling.css','offline_geometry.js','export_offline.ps1','v1_ground.js','inspection.js','inspection_math.js','inspection.css','inspection_help.html'):
        if (source/name).exists():shutil.copy2(source/name,root/name)
    if (root/'index.html').exists():
        p=root/'index.html';html=p.read_text(encoding='utf-8')
        if 'data/geometry_index.js' not in html:
            html=html.replace('<script src="sampling.js">','<script src="data/geometry_index.js"></script><script src="offline_geometry.js"></script><script src="sampling.js">',1)
        if 'v1_ground.js' not in html:html=html.replace('<script src="inspection.js', '<script src="v1_ground.js"></script><script src="inspection.js',1)
        if 'inspection.js' not in html:
            html=html.replace('<script src="app.js','<script src="inspection_math.js"></script><script src="v1_ground.js"></script><script src="inspection.js"></script><script src="app.js',1)
        if 'inspection.css' not in html:html=html.replace('</head>','<link rel="stylesheet" href="inspection.css"></head>',1)
        html=html.replace('app.js?v=sampling2','app.js?v=inspection1')
        p.write_text(html,encoding='utf-8')
    receipt=dict(status='OFFLINE_GEOMETRY_COMPLETE',arms=arms,frames=49,max_rate=100,elapsed_seconds=round(time.time()-started,1),
        gpu_used=False,optimizer_updates=0,entry='index.html',transport='file:// script chunks; no HTTP/SSH required to change sampling')
    (root/'offline_geometry_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)
    return index


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='已有诊断包目录；只更新几何、采样脚本，不覆盖既有原生证据')
    p.add_argument('--arms',nargs='+',help='默认导出backend.ARMS中已登记的实验，以及本机已绑定的历史V1')
    p.add_argument('--chunk-size',type=int,default=32768)
    p.add_argument('--resume',action='store_true',help='继续同一次未完成导出；仅限检查点与参数未变')
    a=p.parse_args()
    if not 1024<=a.chunk_size<=100000:raise ValueError('chunk size outside1024..100000')
    arms=a.arms or list(ARMS)+(['gn11_v1'] if V1.exists() and Path('/root/autodl-tmp/gsdiag_gpu_20261021_r2/v1_binding.json').exists() else [])
    export(a.output,arms,a.chunk_size,a.resume)
