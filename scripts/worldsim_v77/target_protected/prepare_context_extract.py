"""提取素材上下文与真实LiDAR参考，公共盘只解包必要成员，不运行模型。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from collections import defaultdict
import ijson

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def stream(p):
    with Path(p).open('rb') as f:yield from ijson.items(f,'item',use_float=True)

def context(root,meta):
    m=read(root/'source_manifest.json')
    wanted={k for c in m['clips'] for k in c['keyframe_tokens']}
    cate={r['token']:r['name'] for r in read(meta/'category.json')}
    inst={r['token']:cate[r['category_token']] for r in read(meta/'instance.json')}
    anns=defaultdict(list)
    for r in stream(meta/'sample_annotation.json'):
        if r['sample_token'] in wanted:
            anns[r['sample_token']].append({k:r[k] for k in ['instance_token','translation','size','rotation','visibility_token','num_lidar_pts']}|{'category':inst[r['instance_token']]})
    channels={r['token']:r['channel'] for r in read(meta/'sensor.json')}
    cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    sd=defaultdict(dict);need_ego=set()
    for r in stream(meta/'sample_data.json'):
        if r['sample_token'] in wanted and r['is_key_frame']:
            ch=channels[cal[r['calibrated_sensor_token']]['sensor_token']]
            sd[r['sample_token']][ch]={k:r[k] for k in ['token','filename','timestamp','ego_pose_token','calibrated_sensor_token']}
            need_ego.add(r['ego_pose_token'])
    ego={r['token']:r for r in stream(meta/'ego_pose.json') if r['token'] in need_ego}
    files={f['filename'] for c in m['clips'] for f in c['frames']}
    result=[]
    for c in m['clips']:
        arr=[]
        for i,k in enumerate(c['keyframe_tokens']):
            rows={}
            for ch,r in sd[k].items():
                row=r|{'calibrated_sensor':cal[r['calibrated_sensor_token']], 'ego_pose':ego[r['ego_pose_token']]}
                # 首/中/末关键时刻额外相机与LiDAR；不冒充连续的跨相机视频。
                if i in [0,3,6]:files.add(r['filename']);row['materialize']=True
                else:row['materialize']=False
                rows[ch]=row
            arr.append({'sample_token':k,'annotations':anns[k],'sensors':rows})
        result.append({'source_id':c['source_id'],'scene':c['scene'],'frames':arr})
    dump(root/'source_context.json',{'clips':result,'purpose':'GT geometry and real reference evidence; never implicit model conditioning'})
    dump(root/'required_files.json',sorted(files))
    print('CONTEXT',len(result),'FILES',len(files),flush=True)

def extract(root,pub,allowed_shards=None,max_workers=1):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    assert isinstance(max_workers,int) and 1<=max_workers<=2
    from prepare_source_pool import sharding
    names=read(root/'required_files.json');maps=sharding()
    groups=defaultdict(list)
    for n in names:
        assert not Path(n).is_absolute() and '..' not in Path(n).parts and n.startswith(('samples/','sweeps/'))
        shards=maps.get(Path(n).name.split('__')[0],set());assert shards,n
        for sh in shards:groups[sh].append(n)
    if allowed_shards is None:
        assert len(groups)<=2, list(groups)
    else:
        allowed=set(allowed_shards)
        assert allowed and allowed <= {f'{i:02}' for i in range(1,11)},sorted(allowed)
        assert set(groups)<=allowed,{'requested':sorted(groups),'frozen_allowed':sorted(allowed)}
    stpath=root/'extract_state.json'
    state=read(stpath) if stpath.exists() else {'state':'running','pid':os.getpid(),'started_unix':time.time(),'shards':[]}
    state.update(pid=os.getpid(),max_archive_readers=max_workers)
    state.pop('current_shard',None);state.pop('requested_count',None)
    (root/'member_lists').mkdir(exist_ok=True)
    (root/'rgb').mkdir(exist_ok=True)
    complete={r['shard'] for r in state['shards']}
    pending=[(sh,files) for sh,files in sorted(groups.items()) if sh not in complete]
    def extract_one(sh,files):
        request=root/'member_lists'/f'{sh}.txt';request.write_text('\n'.join(files)+'\n')
        print('EXTRACT_START',sh,len(files),flush=True)
        start=time.monotonic()
        dest=root/'by_shard'/sh;dest.mkdir(parents=True,exist_ok=True)
        log=root/f'extract_{sh}.stderr.log'
        with log.open('w') as err:
            proc=subprocess.run(['tar','-xzf',str(pub/f'v1.0-trainval{sh}_blobs.tgz'),'-C',str(dest),'--no-same-owner','--skip-old-files','-T',str(request)],stderr=err,stdout=subprocess.DEVNULL)
        row={'shard':sh,'seconds':round(time.monotonic()-start,2),'returncode':proc.returncode,'requested':len(files),'stderr':str(log)}
        print('EXTRACT_END',row,flush=True)
        return row
    if pending:
        state.update(state='extracting',pending_shards=[sh for sh,_ in pending],updated_unix=time.time())
        dump(stpath,state)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures=[pool.submit(extract_one,sh,files) for sh,files in pending]
            for future in as_completed(futures):
                row=future.result();state['shards'].append(row);state['shards'].sort(key=lambda r:r['shard'])
                state['pending_shards'].remove(row['shard']);state.update(updated_unix=time.time());dump(stpath,state)
    from PIL import Image
    found=[];missing=[];bad=[]
    for n in names:
        p=next((root/'by_shard'/sh/n for sh in groups if (root/'by_shard'/sh/n).is_file()),None)
        if p is None:missing.append(n);continue
        if n.endswith('.jpg'):
            try:
                with Image.open(p) as im:im.load();assert im.size==(1600,900)
            except Exception as exc:bad.append({'file':n,'error':repr(exc)});continue
        link=root/'rgb'/n;link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(p)
        found.append({'filename':n,'bytes':p.stat().st_size,'source':str(p)})
    state.update(state='complete' if not missing and not bad else 'missing_or_bad',finished_unix=time.time(),found=len(found),missing=missing,bad=bad,actual_bytes=sum(r['bytes'] for r in found))
    dump(root/'source_file_inventory.json',found);dump(stpath,state);print('EXTRACT_FINAL',state['state'],len(found),len(missing),len(bad),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);p.add_argument('--pub',type=Path,required=True);a=p.parse_args()
    if not (a.root/'required_files.json').exists():context(a.root,a.meta)
    extract(a.root,a.pub)
if __name__=='__main__':main()
