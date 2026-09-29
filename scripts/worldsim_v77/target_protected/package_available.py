"""在提取过程中打包已齐备的source；同一文件不反复下载。"""
import argparse
import json
import tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--exclude',default='');a=p.parse_args()
    d=json.loads((a.root/'source_manifest.json').read_text());excluded=set(a.exclude.split(','));ready=[];files={}
    for c in d['clips']:
        if c['source_id'] in excluded:continue
        paths=[]
        for f in c['frames']:
            found=next((a.root/'by_shard'/sh/f['filename'] for sh in ['03','07'] if (a.root/'by_shard'/sh/f['filename']).is_file()),None)
            if found is None:break
            paths.append((f['filename'],found))
        if len(paths)==30:
            ready.append(c['source_id']);files.update(paths)
    if ready:
        with tarfile.open(a.out,'w') as tf:
            for n,p in files.items():tf.add(p,arcname='rgb/'+n,recursive=False)
            for name in ['source_manifest.json','source_geometry_validation.json']:
                tf.add(a.root/name,arcname=name,recursive=False)
    print(json.dumps({'ready':ready,'file_count':len(files),'package':str(a.out) if ready else None,'bytes':a.out.stat().st_size if ready else 0}))
if __name__=='__main__':main()
