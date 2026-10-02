"""r44工程回归：固定失败轨迹与旧合法Q060，不新增位置/速度搜索。"""
from pathlib import Path
import copy,sys,time,fcntl
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from primary_relation import partition_primary_reveal
from geometry_factory import read,dump
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
P=Path('/root/autodl-tmp/motion_proj_v77')
O=T/'r44'


def main():
    O.mkdir(exist_ok=True)
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fixed=read(T/'r43/prepared.json')['cases']
    cases=[]
    for source in fixed:
        c=copy.deepcopy(source)
        c.update(source_run='r43',requested_family='protected_reveal',primary_instance_token=c['frames'][0]['actors'][0]['instance_token'])
        cases.append(c)
    positive=copy.deepcopy(next(c for c in read(T/'r28/prepared.json')['cases'] if c['case_id']=='Q060'))
    q=next(c for c in read(T/'r28/instance_quality.json')['cases'] if c['case_id']=='Q060')
    assert len(q['quality']['protected_instances'])==1
    positive.update(source_run='r28',requested_family='protected_reveal',primary_instance_token=q['quality']['protected_instances'][0])
    # Q060的实际主B来自完整保留对象列表，不能用来源排序第一辆替换。
    retained=next(a for a in positive['retained_instances'] if a['instance_token']==positive['primary_instance_token'])
    for frame,actor in zip(positive['frames'],retained['annotations']):
        assert actor is not None and not actor.get('interpolation_uncertain',False)
        if not any(a['instance_token']==positive['primary_instance_token'] for a in frame['actors']):
            frame['actors'].append(copy.deepcopy(actor))
    cases.append(positive)
    plan=dict(task_id='WS-V77-TARGET-PROTECTED-20260929',run_id='r44',phase='primary_relation_queue_engineering_regression',
              fixed_cases=[{k:c[k] for k in ['case_id','source_run','primary_instance_token']} for c in cases],
              change='explicit primary B and necessary depth/contact screen before expensive instance labels',
              inputs='GT camera/poses and existing final H; no source RGB, SAM mask or hidden Y',
              positive_control='Q060 existing independently accepted input/condition; actual protected identity comes from its preserved contract',
              frozen_failure_scope='r43 source-designated B; these are replayed failures, not unseen evaluation',
              no_search=True,no_new_gpu=True,resource_budget='one CPU replay of five fixed30-frame cases; no RGB extraction',
              stop='record necessary screen result; no rejection overrides, no threshold or pose changes',
              training_steps=0,training_admission=0,failure_ledger_refs=['V77-F02'],human_verdict=None)
    if (O/'run.json').exists():assert read(O/'run.json')==plan
    else:dump(O/'run.json',plan)
    start=time.time()
    def holes(c):
        return [np.asarray(Image.open(Path(c['observed_folder'])/'proposal_H'/f'{i:05}.png'))>0 for i in range(len(c['frames']))]
    ready,rejected,checks=partition_primary_reveal(cases,holes)
    current=[r for r in checks if r['case_id'].startswith('G')]
    control=next(r for r in checks if r['case_id']=='Q060')
    assert control['necessary_relation_pass'],'旧合法正控制被新前置门误拒，不能发布'
    result=dict(cases=checks,source_cases=len(current),necessary_pass=[r['case_id'] for r in current if r['necessary_relation_pass']],
                early_rejected=[r['case_id'] for r in current if not r['necessary_relation_pass']],positive_control_preserved=True,
                seconds=time.time()-start,training_admission=0,training_steps=0,human_verdict=None,
                full_quality_not_replaced=True,real_DELETE_gain_demonstrated=False,
                positive_control_pose_source='preserved retained_instances GT annotations; source actor ordering not used as substitute')
    dump(O/'replay.json',result)
    queue=dict(cases=[{k:c[k] for k in ['case_id','source_run','primary_instance_token']} for c in ready if c['source_run']=='r43'],
               launch_allowed=False,reason='r43全量实例QA和独立QA已否决；回归仅展示本来可前置的省算判断，不重跑GPU',
               training_admission=0)
    dump(O/'prospective_label_queue.json',queue)
    close=dict(task_id=plan['task_id'],run_id='r44',stage='engineering_control_complete',
               new_knowledge='几何碰撞通过后仍需显式主B关系；前置只判断必要条件，不证明显露或替代独立QA',
               early_rejected=result['early_rejected'],necessary_pass=result['necessary_pass'],positive_control_preserved=True,
               formal_A_B_C_steps=0,new_training_admission=0,real_DELETE_gain_demonstrated=False,shutdown_condition_met=False,
               failure_ledger_refs=['V77-F02'],failure_ledger_delta='updated V77-F02: explicit primary relation before labels',human_verdict=None)
    dump(O/'closeout.json',close)
    E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r44';E.mkdir(exist_ok=True,parents=True)
    for name in ['run.json','replay.json','prospective_label_queue.json','closeout.json']:dump(E/name,read(O/name))
    print({k:v for k,v in result.items() if k!='cases'},flush=True)


if __name__=='__main__':main()
