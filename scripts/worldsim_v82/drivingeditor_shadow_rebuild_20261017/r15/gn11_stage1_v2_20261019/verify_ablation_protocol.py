"""只读各组真实前100条日志，核对采样与已记录的静态/动态学习率。"""
from pathlib import Path
import argparse,json,subprocess

W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019/ablations'
ARMS=['real_only','prior_only','ownership_only']

def main(arms):
    if len(set(arms))!=len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('explicit distinct authorized arms required')
    source=r'''
from pathlib import Path
import itertools,json,hashlib,datetime
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019')
def first100(path):
    with path.open() as f: rows=[json.loads(x) for x in itertools.islice(f,100)]
    assert len(rows)==100 and [r['iteration'] for r in rows]==list(range(1,101))
    return [{k:r[k] for k in ['iteration','frame_id','time_shift','learning_rates']} for r in rows]
def digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
reference=first100(C/'run30k/paired_steps.jsonl')
out=[]
for arm in ARMS:
    actual=first100(C/'ablations'/arm/'run30k/paired_steps.jsonl')
    assert all(set(r['learning_rates'])=={'static','dynamic'} for r in reference+actual)
    frames=all((a['frame_id'],a['time_shift'])==(b['frame_id'],b['time_shift']) for a,b in zip(actual,reference))
    rates=all(a['learning_rates']==b['learning_rates'] for a,b in zip(actual,reference))
    assert frames and rates
    out.append(dict(status='PASSED_ACTUAL_FIRST100_SAMPLING_AND_LR_EQUAL_TO_CURRENT_FULL',
        variant=arm,first_steps=100,frame_and_time_equal=frames,
        all_optimizer_learning_rates_equal=rates,logged_optimizer_names=['static','dynamic'],
        learning_rate_claim_scope='all static/dynamic parameter groups recorded in paired_steps; environment learning rate is not recorded here',
        reference_first100_protocol_sha256=digest(reference),actual_first100_protocol_sha256=digest(actual),
        reference_log=str(C/'run30k/paired_steps.jsonl'),actual_log=str(C/'ablations'/arm/'run30k/paired_steps.jsonl'),
        parameter_bitwise_initial_snapshot_comparison=False,training_updates=0,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps(out))
'''.replace('for arm in ARMS:', 'for arm in '+repr(arms)+':')
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],
        input=source,capture_output=True,text=True,encoding='utf-8',check=True)
    receipts=json.loads(p.stdout)
    for r in receipts:
        name=r['variant']+'_protocol_receipt.json'
        for dest in [W/'ablations'/name,O/name]:
            dest.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps([dict(variant=r['variant'],first_steps=r['first_steps'],matched=True,
        logged_optimizers=r['logged_optimizer_names'],training_updates=0) for r in receipts]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',required=True,choices=ARMS)
    main(p.parse_args().arms)
