"""按需无损恢复一个已压缩条件数组；不会启动模型或覆盖已有文件。"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/root/autodl-tmp/runs').resolve()
INDEX=Path('/root/autodl-tmp/cleanup_manifests/20260922/conditions-compacted.jsonl')
if not INDEX.is_file():
    INDEX=Path(__file__).resolve().parents[2]/'docs/autoresearch/storage_cleanup_20260922/conditions-compacted.jsonl'

def restore(path):
    path=path.resolve()
    assert path.is_relative_to(ROOT), path
    if path.is_file():
        return path
    archive=path.with_name(path.name+'.gz')
    assert archive.is_file(), archive
    rows=[json.loads(line) for line in INDEX.read_text().splitlines()]
    row=next(r for r in rows if r['path']==str(path))
    assert shutil.disk_usage(path.parent).free > row['original_bytes']+256*1024**2
    temp=path.with_name(path.name+'.restoring')
    with gzip.open(archive,'rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src,dst,8*1024**2)
        dst.flush();os.fsync(dst.fileno())
    assert temp.stat().st_size==row['original_bytes']
    assert not path.exists()
    os.replace(temp,path)
    return path

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path',required=True,type=Path)
    a=ap.parse_args()
    print(json.dumps({'restored':str(restore(a.path))}))
