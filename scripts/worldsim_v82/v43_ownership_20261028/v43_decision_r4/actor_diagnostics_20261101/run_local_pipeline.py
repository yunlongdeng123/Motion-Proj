"""远端只传精简原数据，本机单线程worker并行处理；全部输入/日志留存。"""
import concurrent.futures,json,pathlib,subprocess,sys,time
BASE=pathlib.Path(__file__).resolve().parent
REMOTE='/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101/selected'
ORDER=['waymo-train-212','waymo-train-537','waymo-train-564','waymo-train-008']
CFG=json.loads((BASE/'scenes.json').read_text(encoding='utf8'))
ORDER += ['waymo-val-004','waymo-val-020']
SCP=['scp','-q','-X','nrequests=128','-X','buffer=262144']
def transfer(remote,local):
    local=pathlib.Path(local);tmp=local.with_suffix(local.suffix+'.partial');start=time.perf_counter()
    result=subprocess.run(SCP+['wm-vgpu-1011:'+remote,str(tmp)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    tmp.replace(local)
    return dict(local=str(local),bytes=local.stat().st_size,wall_seconds=time.perf_counter()-start)
def scene_manifest(key):
    local=BASE/'raw'/f'{key}_selector_manifest.json'
    if not local.exists():
        for attempt in range(240):
            check=subprocess.run(['ssh','wm-vgpu-1011','test','-f',REMOTE+'/'+key+'/manifest.json'],capture_output=True)
            if check.returncode==0:break
            time.sleep(5)
        else:raise RuntimeError('selector did not finish '+key)
        transfer(REMOTE+'/'+key+'/manifest.json',local)
        transfer(REMOTE+'/'+key+'/source_receipts.json',BASE/'raw'/f'{key}_source_receipts.json')
    return json.loads(local.read_text(encoding='utf8'))
def build(key,front_only):
    context=next(s['context_id'] for s in CFG['scenes'] if s['scene_key']==key)
    out=BASE/('front_generated' if front_only else 'generated')/'scenes'/context
    if (out/'manifest.json').exists():return dict(scene=key,front_only=front_only,status='existing_complete')
    log=BASE/(key+('_front' if front_only else '_full')+'.log')
    cmd=[sys.executable,str(BASE/'build_media.py'),'--scene',key]+(['--front-only'] if front_only else [])
    with log.open('w',encoding='utf8') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('build failed '+key+' '+str(log))
    if not front_only:
        result=subprocess.run([sys.executable,str(BASE/'add_frame_point_views.py'),str(out)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
    return dict(scene=key,front_only=front_only,log=str(log))
def main():
    receipts=[];futures=[];start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for kind in ('front','actor_cache'):
            for key in ORDER:
                manifest=scene_manifest(key)
                if kind=='actor_cache':
                    import tarfile
                    remote='/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101/cropped/'+key+'.tar'
                    path=BASE/'raw'/(key+'_actor_cache.tar')
                    if not path.exists():
                        for attempt in range(720):
                            check=subprocess.run(['ssh','wm-vgpu-1011','test','-f',remote],capture_output=True)
                            if check.returncode==0:break
                            time.sleep(5)
                        else:raise RuntimeError('point cache did not finish '+key)
                        r=transfer(remote,path);r.update(scene=key,kind=kind);receipts.append(r)
                        (BASE/'transfer_receipts.json').write_text(json.dumps(receipts,indent=2));print(json.dumps(r),flush=True)
                    cache=BASE/'cache'/key;cache.mkdir(parents=True,exist_ok=True)
                    with tarfile.open(path) as tar:tar.extractall(cache,filter='data')
                    futures.append(pool.submit(build,key,False))
                    continue
                path=BASE/'raw'/f'{key}_{kind}.tfrecord';expected=manifest[kind+'_bytes']
                if not path.exists():
                    r=transfer(REMOTE+'/'+key+'/'+kind+'.tfrecord',path);r.update(scene=key,kind=kind);receipts.append(r)
                    (BASE/'transfer_receipts.json').write_text(json.dumps(receipts,indent=2))
                    print(json.dumps(r),flush=True)
                if path.stat().st_size!=expected:raise RuntimeError('transfer size mismatch '+key)
                futures.append(pool.submit(build,key,kind=='front'))
                for f in futures:
                    if f.done():f.result()
        for f in futures:print(json.dumps(f.result()),flush=True)
    print(json.dumps(dict(status='complete',wall_seconds=time.perf_counter()-start,scenes=len(ORDER))),flush=True)
if __name__=='__main__':main()
