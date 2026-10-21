import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib
import zipfile
import numpy as np
from PIL import Image

ARMS=('full','real_only','prior_only','ownership_only')
def read_js(p):
    text=p.read_text(encoding='utf-8');return json.loads(text.split(']=',1)[1].removesuffix(';'))
def unpack(p):
    raw=zlib.decompress(base64.b64decode(p['data']));arrays={}
    for k,a in p['arrays'].items():
        v=np.frombuffer(raw,dtype=a['dtype'],count=a['bytes']//np.dtype(a['dtype']).itemsize,offset=a['offset']).reshape(a['shape'])
        if a.get('multiplier'): v=v.astype(np.float32)*a['multiplier']
        if a.get('log_depth'): v=np.where(v==0,65535,np.expm1((v.astype(np.float32)-1)/254*np.log(656.34))*100)
        if a.get('nearest_upsample'):v=v.repeat(a['nearest_upsample'],axis=0).repeat(a['nearest_upsample'],axis=1)
        arrays[k]=v
    return arrays
def verify(root):
    scene=read_js(root/'data/scene.js');assert len(scene['cameras'])==49 and scene['gpu_used'] is False
    road=unpack(read_js(root/'data/road.js'));assert len(road['provenance'])==26088
    assert int(road['surface_supported'].sum())==14034
    assert road['gates'].shape==(49,26088) and road['source_uv'].shape==(49,26088,2)
    assert np.isfinite(road['world_local']).all()
    queries=[];totals={}
    native_manifest=json.loads((root/'data/asset_manifest.json').read_text())['files']
    for arm in ARMS:
        p=read_js(root/f'data/{arm}/cloud.js');cloud=unpack(p)
        for b in p['branches']:
            prefix='chkpnt' if b['branch']=='static' else 'dynamic_chkpnt'
            assert b['checkpoint_sha256']==native_manifest[f'arms/{arm}/run30k/{prefix}30000.pth']['sha256']
        ids=cloud['static_road_id'];ids=ids[ids>=0]
        assert len(ids)==14034 and len(np.unique(ids))==14034
        assert set(ids)==set(np.flatnonzero(road['surface_supported']))
        assert len(p['branches'])==2 and p['branches'][0]['displayed']<p['branches'][0]['total']
        opt=unpack(read_js(root/f'data/{arm}/optimization.js'));assert opt['reliability'].shape==(49,26088)
        for raw in range(89,138):
            maps=unpack(read_js(root/f'data/{arm}/{raw}.js'))
            assert all(a.shape==(320,480) for a in maps.values())
            assert maps['static_alpha'].dtype in (np.dtype('uint16'),np.dtype('float32'))
            packed_unit=read_js(root/f'data/{arm}/units_{raw}.js');unit=unpack(packed_unit)
            if packed_unit.get('shared'):unit={**unpack(read_js(root/packed_unit['shared']['path'])),**unit}
            assert len(unit['in_view'])==26088
            for key in ('road_only_alpha','road_alpha_in_background','road_alpha_in_joint'):
                assert np.isfinite(unit[key]).all() and (unit[key]>=-1e-4).all() and (unit[key]<=1.001).all()
            for ch in ('joint','deleted_five','road_raw','road_surface_color','dynamic_raw'):
                with Image.open(root/f'assets/{arm}/{ch}/{raw:03d}.webp') as im: assert im.size==(480,320);im.verify()
        for raw in (89,102,131,137):
            for x,y in ((120,210),(255,210),(220,280)):
                q=read_js(root/f'data/rays/{arm}_{raw}_{x}_{y}.js')
                assert q['arm']==arm and q['raw']==raw and q['pixel']==[x,y]
                assert q['gpu_used'] is False and q['parameter_updates']==0
                assert q['status']=='CPU_REFERENCE_NOT_CUDA_TRACE'
                for item in q['top']:
                    assert len(item['checkpoint_sha256'])==64 and item['weight']>=0
                    assert item['id'].startswith(arm+':')
                queries.append(q)
        for x,y in ((90,255),(255,240),(365,280)):
            path=root/f'data/rays/{arm}_102_{x}_{y}.js'
            if path.exists():
                q=read_js(path);assert q['arm']==arm and q['raw']==102 and q['pixel']==[x,y]
                assert q['status']=='CPU_REFERENCE_NOT_CUDA_TRACE' and q['parameter_updates']==0
                queries.append(q)
        totals[arm]=p['branches']
    for raw in range(89,138):
        with Image.open(root/f'assets/gt/{raw}.webp') as im: assert im.size==(480,320);im.verify()
        assert unpack(read_js(root/f'data/lidar/{raw}.js'))['xyz'].shape[1]==3
    receipt=json.loads((root/'export_receipt.json').read_text())
    assert receipt['protected_inputs_unchanged'] and not receipt['gpu_used'] and receipt['training_updates']==0
    assert 'PENDING_GPU'==receipt['cuda_topk_status']
    result=dict(status='CPU_OFFLINE_SYSTEM_VALIDATED_GPU_DIAGNOSTICS_PENDING',arms=list(ARMS),frames_per_arm=49,
        road_cells=26088,road_gaussians_per_arm=14034,cpu_reference_queries=len(queries),
        cpu_reference_alpha_depth_agree_queries=sum(q['numeric_alpha_depth_agree'] for q in queries),
        maximum_cpu_reference_alpha_abs_error=max(q['replay_errors']['alpha_abs'] for q in queries),
        maximum_cpu_reference_depth_sum_model_abs_error=max(q['replay_errors']['depth_sum_model_abs'] for q in queries),
        gpu_used=False,additional_optimization_steps=0,protected_inputs_unchanged=True,geometry=totals,
        browser_qa='PENDING',native_cuda_topk='PENDING_GPU',full_counterfactual_rerasterization='PENDING_GPU')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--package',action='store_true');a=p.parse_args();root=a.root.resolve()
    result=verify(root)
    old=root/'validation.json'
    if old.exists(): result['browser_qa']=json.loads(old.read_text()).get('browser_qa','PENDING')
    old.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.package:
        files={}
        for f in sorted(root.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.name!='package_manifest.json':
                files[f.relative_to(root).as_posix()]=dict(bytes=f.stat().st_size,sha256=hashlib.file_digest(f.open('rb'),'sha256').hexdigest())
        (root/'package_manifest.json').write_text(json.dumps(dict(version=1,files=files),indent=2),encoding='utf-8')
        archive=root.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for key in files:z.write(root/key,root.name+'/'+key)
            z.write(root/'package_manifest.json',root.name+'/package_manifest.json')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for key,v in files.items(): assert hashlib.sha256(z.read(root.name+'/'+key)).hexdigest()==v['sha256']
        result['zip_bytes']=archive.stat().st_size
        result['zip_sha256']=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
        if (root/'portable_receipt.json').exists(): assert result['zip_bytes']<=100_000_000, 'portable ZIP exceeds user100MB limit'
    print(json.dumps(result,ensure_ascii=False,indent=2))
