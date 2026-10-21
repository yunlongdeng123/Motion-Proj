"""既有原生trace与完整CF浮点的只读再评价，不重训练/不重渲染。"""
import json
from pathlib import Path
import numpy as np
from evidence import Evidence,POLICY
from gpu_backend import CAMPAIGN,clean,sha

def main():
    root=Path(__file__).resolve().parent
    out=root/'evidence_export_r2';out.mkdir(exist_ok=False)
    config=json.loads((CAMPAIGN/'arms/full/run30k/effective_config.json').read_text())
    meta=json.loads(Path(config['v3_bundle']).read_text());e=Evidence(CAMPAIGN,meta,float(meta['meter_to_model']))
    traces={};counterfactuals={};files=[]
    for path in sorted(Path('/root/autodl-tmp/gsdiag_gpu_20261021_r1/runs').glob('*/query_*.json')):
        v=json.loads(path.read_text())
        if v['status']=='NATIVE_BUFFER_CUDA_TRACE_VERIFIED':
            v=e.enrich_trace(v);key='/'.join(map(str,['gpu',v['arm'],v['raw'],*v['pixel'],v['mode']]))
            traces[key]=dict(geometry_evidence=v['geometry_evidence'],by_id={r['id']:r['geometry_evidence'] for r in v['top']})
        if v['status']=='FULL_NATIVE_RERASTERIZATION_COMPLETE':
            asset=Path(v['float_asset']['path'])
            if sha(asset)!=v['float_asset']['sha256']:raise RuntimeError('CF float changed')
            with np.load(asset) as z:
                b={k:z['before_'+k] for k in ('rgb','depth','alpha','road')};a={k:z['after_'+k] for k in b}
                metrics=e.evaluate_cf(v['raw'],b,a,v['pixel'])
            key='/'.join(map(str,['cf',v['arm'],v['raw'],*v['pixel']]))
            counterfactuals.setdefault(key,[]).append(dict(removed_ids=[r['id'] for r in v['removed']],scientific_evaluation=metrics,
                source_query=str(path),source_float=v['float_asset']))
            files.append(v['float_asset'])
    result=clean(dict(status='FIXED_WORLD_EVIDENCE_EVALUATED',policy=POLICY,traces=traces,counterfactuals=counterfactuals,
        float_sources=files,optimizer_updates=0,human_verdict=None,quality_accepted=False))
    (out/'evidence.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False))
    print(json.dumps(dict(traces=len(traces),CF_renders=sum(map(len,counterfactuals.values())),output=str(out),bytes=(out/'evidence.json').stat().st_size)))

if __name__=='__main__':main()
