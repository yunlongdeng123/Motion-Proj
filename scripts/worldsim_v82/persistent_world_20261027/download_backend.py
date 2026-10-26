"""补齐已登记的固定revision扩散权重；保留旧.part与历史下载现场。"""
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
TARGET=Path('/root/autodl-tmp/models/persistent_world_diffueraser_20261027')
OLD=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r1/download_manifest.json')


def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(OLD.read_text());entries=manifest['files']
    assert shutil.disk_usage(TARGET).free>sum(x['expected_bytes'] for x in entries)+10*1024**3
    probes=[]
    sample=next(x for x in entries if x['repo']=='lixiaowen/diffuEraser' and x['expected_bytes']>10**9)
    routes={'hf-mirror':sample.get('huggingface_url',sample['url']).replace('https://huggingface.co/','https://hf-mirror.com/'),
            'modelscope':sample['url']}
    def probe(item):
        label,url=item;path=ROOT/('speed_'+label+'.bin')
        result=subprocess.run(['curl','-fLsS','--range','0-1048575','--max-time','15','-o',str(path),'-w','%{speed_download}',url],capture_output=True,text=True)
        return dict(route=label,returncode=result.returncode,speed_bytes_sec=float(result.stdout or 0),bytes=path.stat().st_size if path.exists() else 0,error=result.stderr[-200:])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:probes=list(pool.map(probe,routes.items()))
    (ROOT/'download_route_probe.json').write_text(json.dumps(probes,indent=2))
    viable=[r for r in probes if r['returncode']==0 and r['bytes']>=1048576]
    route=max(viable,key=lambda x:x['speed_bytes_sec'])['route'] if viable else 'hf-mirror'
    def fetch(entry):
        target=TARGET/entry['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        original=Path(entry['local_path'])
        if target.exists() and target.stat().st_size==entry['expected_bytes']:return dict(file=entry['relative_path'],status='complete_existing')
        if original.exists() and original.stat().st_size==entry['expected_bytes']:
            target.symlink_to(original);return dict(file=entry['relative_path'],status='reused',path=str(original))
        upstream=entry.get('huggingface_url',entry['url'])
        urls=[]
        if route=='modelscope' and 'modelscope.cn' in entry['url']:urls.append(entry['url'])
        urls.extend([upstream.replace('https://huggingface.co/','https://hf-mirror.com/'),upstream])
        part=target.with_name(target.name+'.part')
        for url in dict.fromkeys(urls):
            start=time.time()
            result=subprocess.run(['curl','-fLsS','--retry','2','--retry-delay','3','--connect-timeout','15',
                '--speed-limit','10000','--speed-time','45','--max-time','5400','--continue-at','-',
                '--output',str(part),url],capture_output=True,text=True)
            if result.returncode==0 and part.exists() and part.stat().st_size==entry['expected_bytes']:
                # safetensors头检查能发现HTML/截断，不宣称已核对上游SHA。
                if target.suffix=='.safetensors':
                    with part.open('rb') as f:
                        size=int.from_bytes(f.read(8),'little');assert 0<size<20*1024**2;json.loads(f.read(size))
                part.replace(target)
                row=dict(file=entry['relative_path'],status='downloaded',url=url,seconds=time.time()-start,bytes=entry['expected_bytes'])
                print(json.dumps(row),flush=True);return row
            print('DOWNLOAD_ROUTE_FAILED',entry['relative_path'],result.returncode,result.stderr[-160:],flush=True)
        return dict(file=entry['relative_path'],status='incomplete',bytes=part.stat().st_size if part.exists() else 0)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(fetch,entries):
            rows.append(row);(ROOT/'download_status.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2))
    for filename in ['ProPainter.pth','raft-things.pth','recurrent_flow_completion.pth']:
        p=TARGET/'propainter'/filename;p.parent.mkdir(exist_ok=True)
        if not p.exists():p.symlink_to(Path('/root/autodl-tmp/models/worldsim_v77_poc_propainter')/filename)
    status='complete' if all(x['status']!='incomplete' for x in rows) else 'incomplete'
    (ROOT/'download_status.json').write_text(json.dumps(dict(status=status,rows=rows,target=str(TARGET),revisions={x['repo']:x['revision'] for x in entries}),indent=2))


if __name__=='__main__':main()
