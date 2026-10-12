"""Download/extract selected distro headers/libs privately; NEVER install packages."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT.parent
PREFIX = CANDIDATE / "deps/prefix"
DOWNLOADS = ROOT / "downloads"
MAX_DOWNLOAD = 3 * 1024**3
MAX_GROWTH = 8 * 1024**3
MIN_FREE = 30 * 1024**3
DEFAULT_PACKAGES = ["libeigen3-dev", "libboost1.74-dev", "libpcl-dev", "libpcl-common1.12",
    "libpcl-filters1.12", "libpcl-kdtree1.12", "libpcl-search1.12", "libpcl-octree1.12",
    "libpcl-sample-consensus1.12", "libflann-dev", "libflann1.9", "liblz4-dev"]


def run(argv, cwd=None, timeout=45):
    r = subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError(json.dumps({"command":argv,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}))
    return r.stdout


def blocks(text):
    output=[]
    for block in text.strip().split("\n\n"):
        row={}
        for line in block.splitlines():
            if line and not line[0].isspace() and ": " in line:
                k,v=line.split(": ",1); row[k]=v
        if row: output.append(row)
    return output


def disk_bytes(path):
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() or p.is_symlink()) if path.exists() else 0


def save(receipt):
    (ROOT/"dependency_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


def budget_guard(receipt):
    free=shutil.disk_usage(CANDIDATE).free
    growth=disk_bytes(ROOT)+disk_bytes(PREFIX)
    total=disk_bytes(DOWNLOADS)
    receipt["resource_observation"]={"free_bytes":free,"attempt_plus_prefix_bytes":growth,
                                       "downloaded_deb_bytes":total}
    save(receipt)
    if free<MIN_FREE or growth>MAX_GROWTH or total>MAX_DOWNLOAD:
        raise RuntimeError("scoped resource limit reached; stop without removal or global change")


def main():
    ROOT.mkdir(exist_ok=True); PREFIX.mkdir(parents=True,exist_ok=True); DOWNLOADS.mkdir(exist_ok=True)
    existing=ROOT/"dependency_receipt.json"
    receipt=json.loads(existing.read_text()) if existing.exists() else {
        "scope":"Private .deb extraction, no apt install/update, no maintainer scripts, no global mutation",
        "limits":{"minimum_free_bytes":MIN_FREE,"maximum_download_bytes":MAX_DOWNLOAD,
                  "maximum_attempt_prefix_bytes":MAX_GROWTH,"build_workers":1},
        "prefix":str(PREFIX),"packages":[],"reused_system_packages":[
            {"package":"libtbb-dev/libtbb12","version":"2021.5.0-7ubuntu2"},
            {"package":"libgomp1","version":"12.3.0-1ubuntu1~22.04"},
            {"compiler":"GNU g++","version":"11.3.0"}],
        "detector_initialized":False,"gpu_used":False}
    requested=sys.argv[1:] or DEFAULT_PACKAGES
    complete={p["package"] for p in receipt["packages"] if p.get("extracted")}
    planned=[]
    for name in requested:
        if name in complete: continue
        policy=run(["apt-cache","policy",name])
        candidate=next(line.split(":",1)[1].strip() for line in policy.splitlines() if "Candidate:" in line)
        if candidate=="(none)": raise RuntimeError("no cached distro package candidate: "+name)
        metadata_text=run(["apt-cache","show",name+"="+candidate])
        metadata=next(x for x in blocks(metadata_text) if x.get("Version")==candidate)
        (ROOT/(name+".apt_metadata.txt")).write_text(metadata_text)
        uri_text=run(["apt-get","--print-uris","download",name+"="+candidate])
        planned.append({"package":name,"version":candidate,"source_package":metadata.get("Source",name),
            "architecture":metadata["Architecture"],"origin":metadata.get("Origin"),
            "filename":Path(metadata["Filename"]).name,"archive_path":metadata["Filename"],
            "expected_bytes":int(metadata["Size"]),"expected_sha256":metadata["SHA256"],
            "configured_repository_uri_receipt":uri_text,"metadata_origin":"existing cached apt metadata"})
    if disk_bytes(DOWNLOADS)+sum(p["expected_bytes"] for p in planned)>MAX_DOWNLOAD:
        raise RuntimeError("planned downloads exceed3GiB")
    print(json.dumps({"planned_packages":len(planned),"planned_download_bytes":sum(p["expected_bytes"] for p in planned)}),flush=True)
    budget_guard(receipt)
    for package in planned:
        budget_guard(receipt)
        target=DOWNLOADS/package["filename"]
        if not target.exists():
            output=run(["apt-get","-o","Acquire::Retries=0","-o","Acquire::http::Timeout=20",
                        "-o","Acquire::https::Timeout=20","download",package["package"]+"="+package["version"]],
                       cwd=DOWNLOADS,timeout=60)
            (ROOT/(package["package"]+".download_log.txt")).write_text(output)
        data=target.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        if digest!=package["expected_sha256"] or len(data)!=package["expected_bytes"]:
            raise RuntimeError("archive checksum/size mismatch: "+package["package"])
        package.update({"actual_bytes":len(data),"actual_sha256":digest,"verified":True})
        run(["dpkg-deb","--extract",str(target),str(PREFIX)],timeout=45)
        package["extracted"]=True
        receipt["packages"].append(package)
        budget_guard(receipt)
        print(json.dumps({"package":package["package"],"version":package["version"],
                          "sha256":digest,"extracted_to_private_prefix":True}),flush=True)
    receipt["status"]="selected_private_dependencies_ready"
    budget_guard(receipt)
    print(json.dumps({"status":receipt["status"],"resources":receipt["resource_observation"]}),flush=True)


if __name__=="__main__":
    try: main()
    except Exception as e:
        (ROOT/"dependency_failure.txt").write_text(str(e)+"\n")
        print(str(e),file=sys.stderr); sys.exit(1)
