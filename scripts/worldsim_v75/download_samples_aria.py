"""下载已冻结开发样例；凭证/签名地址只进0600临时输入，工具输出仅汇总。"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from huggingface_hub import get_hf_file_metadata,hf_hub_url,get_token
from catalog_samples import RUN,DEST

def main():
    plan=json.loads((RUN/'samples_cohort_manifest.json').read_text())
    items=[f for s in plan['selected'] if s['role']=='development' for f in s['files']]
    state=RUN/'samples_aria';state.mkdir(exist_ok=True)
    def download(item):
        target=DEST/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==item['bytes']:
            return {'path':item['path'],'bytes':item['bytes'],'status':'already_complete'}
        if target.exists():raise ValueError('已有不完整目标需人工核查')
        uid=target.parent.name
        job=state/(uid+'-'+target.name);job.mkdir(exist_ok=True)
        partial=job/'payload.partial'
        if not partial.exists():
            cache=DEST/'.cache/huggingface/download'/Path(item['path']).parent
            candidates=list(cache.glob(target.name+'.*.incomplete'))
            if candidates:
                prefix=max(candidates,key=lambda p:p.stat().st_size)
                if 0<prefix.stat().st_size<item['bytes']:shutil.copyfile(prefix,partial)
        metadata=get_hf_file_metadata(hf_hub_url(plan['repo_id'],item['path'],repo_type='dataset',revision=plan['revision']),token=get_token())
        assert metadata.size==item['bytes']
        # Hub先验证访问权，再给出CDN地址；不能把认证token传给CDN。
        request=job/'request.private'
        fd=os.open(request,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:
            f.write(metadata.location+'\n  dir='+str(job)+'\n  out=payload.partial\n')
        command=['aria2c','--input-file='+str(request),'--continue=true','--file-allocation=none',
                 '--auto-file-renaming=false','--allow-overwrite=false','--max-connection-per-server=8',
                 '--split=8','--min-split-size=1M','--max-tries=3','--retry-wait=5',
                 '--connect-timeout=30','--timeout=60','--auto-save-interval=5','--summary-interval=10',
                 '--download-result=hide','--console-log-level=warn','--all-proxy='+os.environ['HTTPS_PROXY']]
        with (job/'aria2.log').open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        request.unlink()
        if code!=0 or Path(str(partial)+'.aria2').exists():
            return {'path':item['path'],'status':'failed','exitcode':code}
        assert partial.stat().st_size==item['bytes']
        os.replace(partial,target)
        return {'path':item['path'],'bytes':item['bytes'],'status':'complete','exitcode':code}
    began=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,f):f for f in items}
        for future in as_completed(futures):
            try:results.append(future.result())
            except Exception as exc:results.append({'path':futures[future]['path'],'status':'failed','error_type':type(exc).__name__})
            good=[r for r in results if r['status'] in ['complete','already_complete']]
            summary={'status':'downloading','backend':'aria2 segmented HTTP','completed_files':len(good),
                     'total_files':len(items),'completed_bytes':sum(r['bytes'] for r in good),
                     'total_bytes':sum(f['bytes'] for f in items),'elapsed_s':time.monotonic()-began,
                     'failures':[r for r in results if r['status']=='failed'],
                     'completion_basis':'aria2 exit0, control file absent, expected repository bytes; not preallocation'}
            (RUN/'samples_download_result.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(summary),flush=True)
    summary['status']='complete' if len(good)==len(items) else 'failed_stopped'
    summary['files']=results
    (RUN/'samples_download_result.json').write_text(json.dumps(summary,indent=2)+'\n')
    if len(good)!=len(items):raise SystemExit(1)

if __name__=='__main__':main()
