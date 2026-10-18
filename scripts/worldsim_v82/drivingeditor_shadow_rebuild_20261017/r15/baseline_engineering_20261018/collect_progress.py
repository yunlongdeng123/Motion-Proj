"""Collect only current receipts/metrics, never checkpoint binaries or workers."""
from pathlib import Path
import subprocess,json,datetime
W=Path(__file__).parent
R='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/'
def get(rel,target,required=True):
    target.parent.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(['scp','-q','wm-vgpu-1011:'+R+rel,str(target)],capture_output=True,text=True)
    if p.returncode and required:raise RuntimeError(rel+': '+p.stderr)
    return p.returncode==0
def main():
    for n in ['queue_status.json','full_queue_status.json','stage_input_audit.json','matched5_input_receipt.json','experiment_contract.json','channel_gate.json','prior_trace_receipt.json']:
        get(n,W/n)
    for n in ['scene_spec.json','CPU_readiness.json','reference_GN11.json']:
        get('full5_pipeline/metadata/scene0008_089_137/'+n,W/'matched_metadata'/n)
    results=[]
    for arm in ['three_replay','five_replay','five_raw_body','five_ID_weight']:
        d=W/'pilots'/arm
        if not get('pilots/'+arm+'/status.json',d/'status.json',False):continue
        get('pilots/'+arm+'/steps.jsonl',d/'steps.jsonl',False)
        get('pilots/'+arm+'/effective_config.json',d/'effective_config.json',False)
        get('pilots/'+arm+'/repair_probes/step_000300/metrics.json',d/'probe300.json',False)
        metrics=[json.loads(x) for x in (d/'steps.jsonl').read_text().splitlines()] if (d/'steps.jsonl').exists() else []
        results.append(dict(arm=arm,status=json.loads((d/'status.json').read_text()),last_metric=metrics[-1] if metrics else None))
    (W/'pilot_results.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=results,
        scope='300-step diagnostics only, no asset quality acceptance; full49 pilot edit renders and restore audit remain pending'),indent=2),encoding='utf-8')
    print(json.dumps(dict(pilot=json.loads((W/'queue_status.json').read_text())['state'],full=json.loads((W/'full_queue_status.json').read_text())['state'],collected_arms=len(results))))
if __name__=='__main__':main()
