"""Background file controller; no model calls. Fetch, validate, and update the same report."""
from pathlib import Path, PurePosixPath
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,tarfile,time,traceback
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/road_material_three_rounds_20261121'
DEST=OUT/'common1152'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_common1152_20261121'
DEST.mkdir(parents=True,exist_ok=True)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8');os.replace(tmp,p)
def read(p):return json.loads(p.read_text('utf-8'))
def ssh(command):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011',command],capture_output=True,text=True,encoding='utf-8',timeout=45)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    return r.stdout
def safe_unpack(archive,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive)as t:
        for a in t.getmembers():
            pp=PurePosixPath(a.name)
            if pp.is_absolute()or '..'in pp.parts or ':'in a.name or a.issym()or a.islnk():raise ValueError('unsafe archive member '+a.name)
        t.extractall(dest)
def download_resumable(remote_path,target,expected):
    """Keep a valid prefix across network timeouts instead of restarting SCP."""
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:return
    batch=DEST/'resume_download.sftp'
    batch.write_text('reget "'+remote_path+'" "'+target.as_posix()+'"\n',encoding='utf-8')
    before=target.stat().st_size if target.exists() else 0
    write(DEST/'transfer_status.json',dict(state='resuming',UTC=stamp(),bytes_before=before,source=remote_path))
    transfer=subprocess.run(['sftp','-o','BatchMode=yes','-o','ConnectTimeout=15','-b',str(batch),'wm-vgpu-1011'],capture_output=True,text=True,encoding='utf-8',timeout=1800)
    if transfer.returncode:raise RuntimeError('resumable SFTP: '+transfer.stderr[-2000:])
    actual=hashlib.sha256(target.read_bytes()).hexdigest()
    if actual!=expected:
        bad=target.with_name(target.name+'.badsha.'+str(int(time.time())));target.replace(bad)
        raise ValueError('bundle SHA mismatch; invalid prefix preserved at '+str(bad))
    write(DEST/'transfer_status.json',dict(state='verified',UTC=stamp(),bytes_before=before,bytes_total=target.stat().st_size,SHA256=actual))
def install(bundle):
    release=DEST/'completed_r1'
    safe_unpack(bundle,release)
    candidates=list(release.rglob('latest_manifest.json'))
    if len(candidates)!=1:raise ValueError('need unique latest_manifest.json')
    mp=candidates[0];m=read(mp);metricfiles=list(release.rglob('metrics.json'))
    bp=release/'bench/metrics.json'
    if not bp.exists():
        if len(metricfiles)!=1:raise ValueError('need unique aggregate metrics.json')
        bp=metricfiles[0]
    metrics=read(bp)
    prefix=release.relative_to(OUT).as_posix()+'/'
    def imagepath(p,full=False):
        q=(release/p).resolve()
        if not q.is_relative_to(release.resolve())or not q.is_file():raise ValueError('missing/invalid image '+str(p))
        with Image.open(q)as im:
            if im.width>1152 or im.height>768:raise ValueError('image exceeds teacher '+str(p))
            if full and im.size!=(1152,768):raise ValueError('full frame not 1152x768 '+str(p))
        return prefix+p
    models=m['models']
    for model in models:
        model['round']=str(model['round']).upper().removeprefix('R')
        if isinstance(model.get('budget'),dict):model['budget']=' / '.join(str(k)+' '+str(v)for k,v in model['budget'].items())
        for fr in model['frames'].values():
            for k,v in list(fr.items()):
                if isinstance(v,str)and Path(v).suffix.lower()in ['.png','.webp','.jpg','.jpeg']:fr[k]=imagepath(v,k in ['road','joint'])
    teacher=m['teacher']
    if list(teacher['resolution'])!=[1152,768]:raise ValueError('unexpected teacher resolution')
    for fr in teacher['frames'].values():
        for k,v in list(fr.items()):
            if isinstance(v,str)and Path(v).suffix.lower()in ['.png','.webp','.jpg','.jpeg']:fr[k]=imagepath(v,k in ['A','C','S'])
    reps=m.get('representatives',[])
    if not reps:
        for g,tokens in [('1',['ABC800','LH']),('2',['M11']),('3',['G_NUMERIC','NUMERIC','DETAIL'])]:
            choices=[a for a in models if a['round']==g]
            if not choices:raise ValueError('missing round '+g)
            chosen=next((a for token in tokens for a in choices if token in a['id'].upper()),choices[-1]);reps.append(chosen['id'])
    evidence=[]
    for p in sorted(release.rglob('*')):
        if p.is_file()and p.suffix in ['.json','.md']and len(p.relative_to(release).parts)<=3:evidence.append(dict(path=prefix+p.relative_to(release).as_posix(),name=p.name))
    gaps=metrics.get('gap_rows',metrics.get('blackgap_rows',metrics.get('black_gap_rows',[])))
    keys=['id','raw','source','low_alpha_fraction','no_support_fraction','high_alpha_dark_fraction']
    data=dict(models=models,teacher=teacher,rows=metrics.get('rows',[]),representatives=reps,gaps=gaps,
        gap_keys=keys,gap_heads=['端点','帧','域','低覆盖比例','无支持比例','高覆盖暗色比例'],
        gap_note=metrics.get('gap_note','黑缝诊断保持固定阈值；覆盖空隙与已覆盖暗纹分开记录。联合静态遮挡不能由道路alpha单独判定。'),evidence=evidence)
    history=OUT/'history_before1152.html'
    if not history.exists():shutil.copy2(OUT/'index.html',history)
    template=(Path(__file__).parent/'report_template.html').read_text('utf-8')
    page=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</',r'<\/'))
    write(DEST/'report_data.json',data)
    shutil.copy2(Path(__file__).parent/'architecture_1152.svg',OUT/'architecture_1152.svg')
    # Publish the new page atomically only after all referenced files validate.
    tmp=OUT/'index.common1152.tmp.html';tmp.write_text(page,'utf-8');os.replace(tmp,OUT/'index.html')
    return dict(model_count=len(models),metric_rows=len(data['rows']),gap_rows=len(gaps),manifest=str(mp),index_sha256=hashlib.sha256((OUT/'index.html').read_bytes()).hexdigest())
def pending():
    history=OUT/'history_before1152.html'
    if not history.exists():shutil.copy2(OUT/'index.html',history)
    page=history.read_text('utf-8')
    banner='<section style="border:1px solid #77d6be;background:#17352f;padding:16px;border-radius:10px;margin-bottom:18px"><strong>新版1152×768教师重训已安排</strong><p>三轮正式方法使用新教师重新训练；完成后本页自动切换到统一分辨率结果。下方暂为历史实验。</p><a href="common1152/run_status.json">后台队列与取回记录</a></section>'
    (OUT/'index.html').write_text(page.replace('<main>','<main>'+banner,1),'utf-8')
def main(once=False):
    lock=DEST/'finalizer.lock'
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('another finalizer owns '+str(lock))
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    write(DEST/'finalizer_process.json',dict(pid=os.getpid(),started_utc=stamp(),python=sys.executable,remote=REMOTE))
    errors=0
    try:
        while True:
            try:
                status=json.loads(ssh('cat '+REMOTE+'/status.json'));errors=0
                state=status.get('state',status.get('status'))
                write(DEST/'run_status.json',dict(remote=status,local_state='waiting',checked_utc=stamp()))
                if state in ['failed','error']:
                    write(DEST/'run_status.json',dict(remote=status,local_state='remote_failed',checked_utc=stamp()));return
                if state in ['complete','completed']:
                    path=status.get('bundle',REMOTE+'/final_bundle.tar.gz')
                    if not str(path).startswith(REMOTE+'/')or any(x in str(path)for x in ['..','\n',' ']):raise ValueError('invalid bundle path')
                    target=DEST/'final_bundle.tar.gz'
                    expected=status.get('bundle_sha256',status.get('SHA256'))
                    if not expected:expected=ssh('sha256sum '+path).split()[0]
                    download_resumable(path,target,expected)
                    actual=hashlib.sha256(target.read_bytes()).hexdigest()
                    result=install(target)
                    write(DEST/'run_status.json',dict(remote=status,local_state='report_updated',completed_utc=stamp(),bundle_sha256=actual,report=result));return
            except Exception:
                errors+=1
                write(DEST/'finalizer_error.json',dict(UTC=stamp(),consecutive_errors=errors,traceback=traceback.format_exc()))
                if errors>=12:
                    write(DEST/'run_status.json',dict(local_state='finalizer_failed',UTC=stamp(),error_log='finalizer_error.json'))
                    raise
            if once:return
            time.sleep(60)
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--mark-pending',action='store_true');a=p.parse_args()
    if a.mark_pending:pending()
    else:main(a.once)
