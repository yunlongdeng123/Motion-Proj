"""r38：冻结新来源的一次车道轨迹检查，完整实例标签先于准入。"""
from pathlib import Path
import sys,os,time,fcntl,shutil,argparse
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import reveal_factory as f
import recover_instance_audit as recover
import evaluate_instance_audit as audit
from prepare_instance_audit import clean
import numpy as np

O=f.T/'r38';SOURCE=f.T/'r37';ROOT=SOURCE/'factory'


def setup():
    f.O=SOURCE;f.ROOT=ROOT;f.legacy.O=SOURCE;f.legacy.ROOT=ROOT
    recover.O=O;recover.p.O=O;audit.O=O
    return f.legacy.geometry()


def register(g):
    O.mkdir(exist_ok=True)
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r38','phase':'fresh_source_fixed_lane_reveal',
        'source_run':'r37','sources':sorted(g.sources),'source_split_unchanged':True,
        'single_change_vs_existing_factory':'new physical-space source cohort, proper location map; r32 role fix retained',
        'trajectory_policy':f.POLICY,'max_proposals_per_source':72,'quality_reveal_policy':'primary_plus_preserved',
        'selection':'all spatially valid trajectories, semantic keys frozen before extra Y-SAM; no first-rejection selector',
        'instance_labels':'all GT envelopes touched by final H; missing/empty/overlapping labels quarantine rather than assume background',
        'Y_role':'offline QA and supervision only; prospective method conditions must reread final-H-erased RGB',
        'review_selection':'at most two per scene, prefer a protected reveal and one background; deterministic lane order; include rejected candidates in counts',
        'quality_gate':'independent sampled-frame input QA2 then condition QA2; human verdict stays null',
        'GT_camera_track_LiDAR':'declared POC auxiliaries','frames':30,'training_steps':0,'surfel':False,
        'stop':'one fixed max24 positions x 0/3/6mps per source; no additional offset/speed grids on this cohort',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert f.read(O/'run.json')==plan
    else:f.dump(O/'run.json',plan)
    E=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r38');E.mkdir(exist_ok=True,parents=True)
    shutil.copy2(O/'run.json',E/'run.json')
    return plan


def propose():
    g=setup();register(g)
    assert f.read(SOURCE/'segmentation_state.json')['stage']=='complete_quarantined_pending_synthetic_QA'
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();maps={};directory=O/'source_proposals';directory.mkdir(exist_ok=True)
    state={'stage':'spatial_proposals','pid':os.getpid(),'training_steps':0};f.dump(O/'controller_state.json',state)
    for sid,c in sorted(g.sources.items()):
        file=directory/(sid+'.json')
        if file.exists():continue
        ground=g.prepare(sid);pm=f.legacy.protections(g,sid);reject=Counter();rows=[];attempts=0;mids=[]
        if ground['pass'] and pm:
            f.legacy.support(g,sid);location=c['location']
            if location not in maps:maps[location]=f.NuScenesMap(dataroot=str(ROOT),map_name=location)
            asset=f.POLICY['split_shape'][c['source_split']];size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
            mids=f.centers(g,maps[location],c,size)
            for q in mids:
                for speed in f.POLICY['speeds_mps']:
                    attempts+=1;tr,why=f.trajectory(g,c,q,speed,asset)
                    if tr is None:reject[why]+=1
                    else:rows.append(clean(tr))
        else:reject['ground_or_stable_primary_labels_unavailable']+=1
        f.dump(file,{'source_id':sid,'scene':c['scene'],'location':c['location'],'attempts':attempts,'midpoints':len(mids),
            'spatial_trajectories':rows,'rejects':dict(reject),'ground':clean(ground),'label_tracks':len(pm)})
        print('NEW_SOURCE_SPACE',sid,len(rows),dict(reject),flush=True)
    stats=[f.read(p) for p in sorted(directory.glob('*.json'))];rows=[]
    for rec in stats:
        for tr in rec['spatial_trajectories']:
            rows.append({'case_id':f'S{len(rows)+1:03}','source_id':rec['source_id'],'trajectory':tr,
                'reuse_saved_pixels':False,'sorted_first_rejection':'not_a_selector_pending_complete_instance_QA'})
    assert len(stats)==len(g.sources)
    roster={'selection_id':'fresh_all_spatial_trajectories_v1','selection_frozen_before_extra_Y_SAM':True,
        'cases':rows,'count':len(rows),'saved_pixel_reuse':0,'sources':sorted(g.sources),
        'attempts':sum(r['attempts'] for r in stats),'training_admission':0,'training_steps':0,
        'space_rejections':dict(sum((Counter(r['rejects']) for r in stats),Counter()))}
    if (O/'candidate_roster.json').exists():assert f.read(O/'candidate_roster.json')==roster
    else:f.dump(O/'candidate_roster.json',roster)
    if not (O/'prepared.json').exists():recover.prepare(g,roster,recovery=False)
    state.update(stage='prepared_pending_full_instance_labels',spatial_candidates=len(rows),seconds=time.time()-start)
    f.dump(O/'controller_state.json',state);print('FRESH_PREPARED',len(rows),flush=True)


def evaluate():
    g=setup();register(g)
    audit.main(output_root=O,reveal_policy='primary_plus_preserved')
    result=f.read(O/'instance_quality.json');f.dump(O/'controller_state.json',{
        'stage':'complete_input_contract_pending_independent_QA','pid':os.getpid(),'counts':result['counts'],
        'training_admission':0,'training_steps':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['propose','evaluate']);a=p.parse_args()
    (propose if a.stage=='propose' else evaluate)()
