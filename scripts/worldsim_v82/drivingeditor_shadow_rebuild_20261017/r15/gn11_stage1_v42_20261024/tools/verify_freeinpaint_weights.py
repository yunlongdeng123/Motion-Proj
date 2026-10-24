"""Read-only weight verification on Windows/Linux; never loads pickle or models."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    plan=json.loads((root/'download_plan.json').read_text(encoding='utf-8'))
    results=[]
    for e in plan['files']:
        p=root/'hub'/('models--'+e['repo'].replace('/','--'))/'snapshots'/e['revision']/e['file']
        h=hashlib.sha256() if e['algorithm']=='sha256' else hashlib.sha1()
        if e['algorithm']=='git_blob_sha1':
            h.update(('blob '+str(e['size'])+'\0').encode())
        if p.exists() and p.stat().st_size==e['size']:
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    h.update(block)
            actual=h.hexdigest()
        else:
            actual=None
        ok=actual==e['expected_hash'] and not p.with_name(p.name+'.aria2').exists()
        results.append(dict(file=e['repo']+'/'+e['file'],verified=ok,actual_hash=actual))
    report=dict(all_verified=all(r['verified'] for r in results),files=len(results),total_bytes=plan['total_bytes'],results=results,inference_run=False)
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if not report['all_verified']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
