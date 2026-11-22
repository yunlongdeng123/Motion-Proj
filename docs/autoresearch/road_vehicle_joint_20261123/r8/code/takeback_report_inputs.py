"""Transfer only original report images and completed receipts, with SHA checks."""
from pathlib import Path
import argparse,hashlib,json,tarfile

NAME='road_vehicle_joint_dggt_20261123_r8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','extract']);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--target',type=Path);a=p.parse_args()
    if a.mode=='build':
        root=Path('/root/autodl-tmp/drivingeditor_repro/campaign')/NAME
        names=['baseline_A0_five_views','joint_rgb_formal400_v1','joint_persistent_asset_audit_v1','formal400_summary_v1']
        files=[path for name in names for path in sorted((root/name).rglob('*')) if path.is_file() and path.suffix in ('.json','.png')]
        rows=[dict(name=NAME+'/'+f.relative_to(root).as_posix(),sha256=sha(f),bytes=f.stat().st_size) for f in files]
        assert not a.archive.exists() and not a.manifest.exists();a.archive.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(a.archive,'w:gz',compresslevel=3) as t:
            for f,row in zip(files,rows):t.add(f,arcname=row['name'],recursive=False)
        a.manifest.write_text(json.dumps(dict(complete=True,archive_sha256=sha(a.archive),archive_bytes=a.archive.stat().st_size,members=rows),indent=2)+'\n')
        print(json.dumps(dict(files=len(rows),archive_bytes=a.archive.stat().st_size,sha256=sha(a.archive))));return
    m=json.loads(a.manifest.read_text());assert m['complete'] and sha(a.archive)==m['archive_sha256'] and a.archive.stat().st_size==m['archive_bytes']
    target=a.target.resolve();rows={r['name']:r for r in m['members']};seen=set()
    with tarfile.open(a.archive,'r:gz') as t:
        for item in t:
            dest=(target/item.name).resolve()
            assert item.isfile() and item.name in rows and item.name not in seen and dest.is_relative_to(target)
            with t.extractfile(item) as stream:data=stream.read()
            assert len(data)==rows[item.name]['bytes'] and hashlib.sha256(data).hexdigest()==rows[item.name]['sha256']
            if dest.exists():assert dest.read_bytes()==data,('Existing diagnostic differs',str(dest))
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            seen.add(item.name)
    assert seen==set(rows)
    receipt=dict(complete=True,archive_SHA_verified=True,every_member_SHA_verified=True,files=len(seen),target=str(target))
    a.archive.with_suffix(a.archive.suffix+'.local_verify.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
