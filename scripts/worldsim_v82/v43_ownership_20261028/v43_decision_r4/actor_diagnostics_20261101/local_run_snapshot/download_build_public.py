"""Fetch the same pinned public raw source; CPU-build the eight disjoint scenes.

All requests keep normal TLS verification. Other scenes belong to the media
worker. Raw data remain in work/, not in the user-facing page or Git.
"""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import re
import subprocess
import sys
import time
import argparse

BASE=Path(__file__).resolve().parents[3]
MEDIA=BASE/"work/actor_diagnostics_20261101/media"
OUT=Path(__file__).resolve().parent
REV="8a6fccd2b022077308c167a4d3c6f646d6e5e65b"
KEYS=["waymo-val-"+number for number in ["029","065","067","079","081","086","113","140"]]
PREFIX="public"
INVENTORY={r["scene_key"]:r for r in json.loads((MEDIA/"raw_inventory.json").read_text())}

def download(key):
    row=INVENTORY[key]
    name=Path(row["path"]).name
    size=row.get("bytes") or row.get("raw_bytes") or row.get("size")
    if not size:raise ValueError(row)
    target=MEDIA/"raw"/name
    if target.exists():
        assert target.stat().st_size==size
        return {"scene_key":key,"status":"already_local","bytes":size,"path":str(target)}
    part=target.with_suffix(target.suffix+".public_partial")
    offset=part.stat().st_size if part.exists() else 0
    start=time.monotonic()
    split=Path(row["path"]).parent.name
    url=f"https://huggingface.co/datasets/AnnaZhang/waymo_open_dataset_v_1_4_3/resolve/{REV}/individual_files/{split}/{name}?download=true"
    with part.open("ab") as stream:
        while offset<size:
            last=min(size-1,offset+32*1024*1024-1)
            for attempt in range(3):
                try:
                    with urlopen(Request(url,headers={"Range":f"bytes={offset}-{last}"}),timeout=40) as response:
                        content_range=response.headers.get("Content-Range","")
                        assert response.status==206 and content_range==f"bytes {offset}-{last}/{size}",(response.status,content_range)
                        data=response.read(last-offset+1)
                    if len(data)!=last-offset+1:raise ValueError("short public range")
                    stream.write(data);stream.flush();offset=last+1
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(1+attempt)
            if offset==size or offset%(128*1024*1024)==0:
                print(json.dumps({"download":key,"bytes":offset,"total":size,"seconds":round(time.monotonic()-start,1)}),flush=True)
    assert part.stat().st_size==size
    part.replace(target)
    return {"scene_key":key,"status":"downloaded","bytes":size,"path":str(target),"url":url,"seconds":round(time.monotonic()-start,3)}

def build(key):
    log=OUT/(key+"_build.log")
    with log.open("w",encoding="utf-8") as stream:
        result=subprocess.run([sys.executable,str(MEDIA/"build_media.py"),"--scene",key],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"build {key} failed: {log}")
    # The independent per-frame point view is generated after full measured NPZ.
    frame_log=OUT/(key+"_frames.log")
    with frame_log.open("w",encoding="utf-8") as stream:
        result=subprocess.run([sys.executable,str(MEDIA/"add_frame_point_views.py"),key],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"frame views {key} failed: {frame_log}")
    print(json.dumps({"built":key,"status":"complete"}),flush=True)
    return {"scene_key":key,"status":"complete","log":str(log),"frame_log":str(frame_log)}

def main():
    downloads=[];build_results=[]
    with ThreadPoolExecutor(max_workers=2) as builders,ThreadPoolExecutor(max_workers=3) as net:
        future_map={net.submit(download,key):key for key in KEYS}
        builds=[]
        for future in as_completed(future_map):
            row=future.result();downloads.append(row)
            (OUT/(PREFIX+"_download_receipts.json")).write_text(json.dumps(downloads,indent=2),encoding="utf-8")
            print(json.dumps(row),flush=True)
            builds.append(builders.submit(build,row["scene_key"]))
        for future in as_completed(builds):
            build_results.append(future.result())
            (OUT/(PREFIX+"_build_receipts.json")).write_text(json.dumps(build_results,indent=2),encoding="utf-8")
    print(json.dumps({"status":"complete","scenes":len(build_results)}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenes",nargs="+")
    parser.add_argument("--prefix",default="public")
    args=parser.parse_args()
    if args.scenes:KEYS=args.scenes
    PREFIX=args.prefix
    main()
