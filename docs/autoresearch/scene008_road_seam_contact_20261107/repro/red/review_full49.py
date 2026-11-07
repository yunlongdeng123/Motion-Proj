"""CPU receipt/PNG audit; protection checks remain supporting, not total success."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--control',required=True);p.add_argument('--pilot');args=p.parse_args()
    run=args.run;folder=run/'results'/args.control;receipt=read(folder/'receipt.json');frames=receipt['frames']
    target=run/'red'/f'{args.control}_protection_review.json'
    if target.exists():raise FileExistsError(target)
    assert len(frames)==98 and sorted(set(x['raw'] for x in frames))==list(range(89,138))
    total=defaultdict(lambda:[0.,0]);deltas=defaultdict(list);hashes=[]
    for f in frames:
        variants=f['variants'];hashes.append(variants['parent']['asset_rows_hash']==variants['candidate']['asset_rows_hash'])
        for name in ('parent','candidate'):
            for metric,v in variants[name].items():
                if not isinstance(v,dict) or not all(k in v for k in ('sum','count','mean')) or not v['count']:continue
                t=total[f['state'],name,metric];t[0]+=v['sum'];t[1]+=v['count']
        for metric,v in variants['parent'].items():
            other=variants['candidate'].get(metric)
            if isinstance(v,dict) and other and 'mean' in v and 'mean' in other:
                deltas[f['state'],metric].append(dict(raw=f['raw'],delta=other['mean']-v['mean']))
    stored={(x['state'],x['variant'],x['metric']):x for x in receipt['pooled']}
    pooled=[]
    for key,value in stored.items():
        s,n=total[key];assert n==value['count'] and abs(s-value['sum'])<1e-6 and abs(s/n-value['mean'])<1e-12
        pooled.append(dict(state=key[0],variant=key[1],metric=key[2],count=n,sum=s,mean=s/n))
    tails=[]
    for (state,metric),rows in deltas.items():
        values=np.array([r['delta'] for r in rows]);tails.append(dict(state=state,metric=metric,frames=len(rows),
            worse_frames=int((values>0).sum()),worse_gt0001_frames=int((values>.001).sum()),p95=float(np.percentile(values,95)),
            worst=sorted(rows,key=lambda x:x['delta'],reverse=True)[:5]))
    old=read(run.parent/'scene008_road_r0_v44_20261107_r1/results/full49_corridor_cellfit/receipt.json')
    baseline={(x['state'],x['metric']):x for x in old['pooled'] if x['variant']=='baseline'}
    relative=[]
    for row in pooled:
        key=(row['state'],row['metric'])
        if row['variant']!='candidate' or key not in baseline:continue
        b=baseline[key];assert row['count']==b['pixels']
        relative.append(dict(state=row['state'],metric=row['metric'],count=row['count'],V44=b['mean'],candidate=row['mean'],delta=row['mean']-b['mean']))
    parent_png_checks=[]
    for raw in range(89,138):
        for state in ('keep','delete'):
            current=folder/f'{raw:03d}/parent_{state}.png';base=run/f'results/n2_full49/{raw:03d}/parent_{state}.png'
            parent_png_checks.append(sha(current)==sha(base))
    assert all(hashes) and all(parent_png_checks)
    pilot_checks=[]
    if args.pilot:
        for raw in (105,113,128,130,137):
            for state in ('keep','delete'):
                a=folder/f'{raw:03d}/candidate_{state}.npz';b=run/f'results/{args.pilot}/{raw:03d}/candidate_{state}.npz'
                with np.load(a) as x,np.load(b) as y:
                    checks={k:bool(np.array_equal(x[k],y[k])) for k in x.files if k in y.files};assert all(checks.values()),(raw,state,checks)
                    pilot_checks.append(dict(raw=raw,state=state,common_fields_bitwise_equal=checks))
    coverage=[dict(raw=f['raw'],solo=f['solo_newroad']) for f in frames if f['raw']==137 and f.get('solo_newroad')]
    report=dict(schema='FULL49_INDEPENDENT_SUPPORTING_PROTECTION_REVIEW_V1',control=args.control,receipt_sha256=sha(folder/'receipt.json'),
        rows=98,raws=49,pooled=pooled,pooled_recomputed_mismatches=0,frame_delta_tails_vs_parent=tails,
        candidate_relative_to_old_V44=relative,vehicle_row_hash_equal_records=sum(hashes),parent_png_exact_to_n2_full49_records=sum(parent_png_checks),
        key5_pilot_full49_common_float_fields=pilot_checks,raw137_solo_coverage=coverage,
        interpretation='Supporting observed-domain checks only; no combined pass or scene adoption. Ground geometry reference shares same-scene BUILD LiDAR measurements with R0; no hidden-body GT.',
        depth_note='Gaussian expected centre depth compared to nearest sparse real LiDAR return, not opaque first-hit accuracy.',GPU_used=False,model_updates=0)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(control=args.control,pooled_mismatches=0,hash_equal=sum(hashes),parent_png_exact=sum(parent_png_checks),pilot_full49_pairs=len(pilot_checks),
        supporting_relative_V44=[r for r in relative if r['metric'] in ('observed_road_RGB','protected_RGB')],coverage=coverage)),flush=True)


if __name__=='__main__':main()
