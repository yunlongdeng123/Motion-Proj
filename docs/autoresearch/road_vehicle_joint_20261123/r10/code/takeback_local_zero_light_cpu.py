"""Create a manifest-bound light report bundle; no float renders are discarded."""
from pathlib import Path
import argparse,hashlib,json,tarfile
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'receipt.json').read_text());assert r['complete']
    paths=['receipt.json','camera_timestamp_audit.json']
    for name in ['local_colour_guard.json','local_colour_guard.npz','fixed_eval_curve.json','losses.json','resources.json']:
        if (a.run/name).is_file():paths.append(name)
    for raw in [86,110,134,179,198]:
        for name in ['shared.png','alpha.png','local_road_alphaT.png','metrics.json']:paths.append(f'{raw:03d}/{name}')
    assert not a.output.exists();a.output.mkdir(parents=True)
    records=[dict(path=x,sha256=sha(a.run/x),bytes=(a.run/x).stat().st_size) for x in paths]
    archive=a.output/'report_light.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as t:
        for name in paths:t.add(a.run/name,arcname=name,recursive=False)
    manifest=dict(complete=True,run=str(a.run),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=records,full_float_readouts_retained_remotely=True,GPU_calls=0)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(dict(files=len(records),bytes=manifest['archive_bytes'],sha256=manifest['archive_sha256']))
if __name__=='__main__':main()
