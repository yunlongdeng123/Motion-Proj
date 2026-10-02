"""r39：用真实车辆曾占据的位姿提议A，检查地图车道采样的空间缺口。

固定同一新15源，唯一定时差2秒；静止／回放各一次，不增偏移速度网格。
历史GT只决定被删A的合成位置，不是保护车状态条件，更不读取历史RGB。
"""
from pathlib import Path
import sys,time,os,bisect,copy,fcntl,shutil,argparse
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import fresh_reveal_pipeline as base
import reveal_factory as f
import recover_instance_audit as recover
import evaluate_instance_audit as audit
from prepare_instance_audit import clean
from pyquaternion import Quaternion
import numpy as np
O=f.T/'r39';ROOT=f.T/'r37/factory'
META=f.T.parent/'WS-V77-DELETE-AUDIT-20260928/r1/metadata/v1.0-trainval'


def setup():
    g=base.setup();recover.O=O;recover.p.O=O;audit.O=O
    return g


def register(g):
    O.mkdir(exist_ok=True)
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r39','phase':'observed_track_proposal_control',
        'sources':sorted(g.sources),'source_run':'r37','source_split_unchanged':True,'frames':30,
        'question':'can a real annotated prior vehicle footprint avoid the r38 image-level island placement failure?',
        'proposal_A':'same primary B track sampled at t-2s; one moving replay and one fixed world pose at first replay instant',
        'max_proposals_per_source':2,'time_shift_seconds':-2.0,'no_other_shifts_offsets_or_speeds':True,
        'history_role':'GT metadata solely for synthetic deleted A placement; history RGB not read and never actor-state evidence',
        'pose_support':'interpolate adjacent actual keyframes only, max0.65s gap; no extrapolation or held pose completion',
        'shape':'existing split sedan/suv, fixed size; Z fitted from current observed ground, not floating GT bottom',
        'space_checks':'all previous lane-factory ground/clearance/ego/scale gates; in addition full 8px border and velocity-yaw consistency',
        'quality_reveal_policy':'primary_plus_preserved','full_Y_role':'offline labels/supervision only',
        'retained_labels':'reuse exact same r38 source/job masks; label new touched instances before evaluation',
        'stop':'one2s proposal control; no shift or pose grid after results; independent QA before method condition',
        'training_steps':0,'training_admission':0,'surfel':False,'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert f.read(O/'run.json')==plan
    else:f.dump(O/'run.json',plan)
    e=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r39');e.mkdir(exist_ok=True,parents=True);shutil.copy2(O/'run.json',e/'run.json')


def history(g):
    tokens={c['actors'][0]['instance_token'] for c in g.sources.values()}
    samples={s['token']:s for s in f.read(META/'sample.json')};tracks={t:[] for t in tokens}
    for a in f.read(META/'sample_annotation.json'):
        if a['instance_token'] in tokens:tracks[a['instance_token']].append(a|{'timestamp':samples[a['sample_token']]['timestamp'],'scene_token':samples[a['sample_token']]['scene_token']})
    return {t:sorted(a,key=lambda x:x['timestamp']) for t,a in tracks.items()}


def pose_at(track,stamp):
    stamps=[a['timestamp'] for a in track];j=bisect.bisect_left(stamps,stamp)
    if j<len(stamps) and stamps[j]==stamp:a=b=track[j];alpha=0.
    elif j==0 or j==len(stamps):return None
    else:
        a,b=track[j-1],track[j]
        if b['timestamp']-a['timestamp']>650000:return None
        alpha=(stamp-a['timestamp'])/(b['timestamp']-a['timestamp'])
    assert a['scene_token']==b['scene_token']
    return {'translation':((1-alpha)*np.array(a['translation'])+alpha*np.array(b['translation'])).tolist(),
        'rotation':Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),amount=alpha).elements.tolist(),
        'source_annotation_tokens':[a['token'],b['token']],'source_timestamp':stamp}


def propose():
    g=setup();register(g);lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tracks=history(g);rows=[];audit_rows=[];start=time.time()
    for sid,c in sorted(g.sources.items()):
        ground=g.prepare(sid);pm=f.legacy.protections(g,sid);tok=c['actors'][0]['instance_token']
        if not ground['pass'] or tok not in pm:
            audit_rows.append({'source_id':sid,'result':'ground_or_primary_label_unavailable'});continue
        f.legacy.support(g,sid);asset=f.POLICY['split_shape'][c['source_split']];size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        poses=[pose_at(tracks[tok],fr['timestamp']-2000000) for fr in c['frames']]
        for mode in ['stationary_prior_pose','moving_prior_track']:
            selected=[poses[0]]*30 if mode=='stationary_prior_pose' else poses
            rec={'source_id':sid,'scene':c['scene'],'mode':mode,'intended_B':tok}
            if any(a is None for a in selected):rec['result']='missing_real_prior_pose';audit_rows.append(rec);continue
            anchors=[]
            for fr,a in zip(c['frames'],selected):
                R=Quaternion(a['rotation']).rotation_matrix;yaw=np.arctan2(R[1,0],R[0,0]);R=f.ground_orientation(yaw,ground['plane'])
                xyz=np.array(a['translation']);xyz[2]=np.dot(np.r_[xyz[:2],1],ground['plane'])
                actor={'translation':(xyz+R[:,2]*size[2]/2).tolist(),'rotation':Quaternion(matrix=R).elements.tolist(),'size':size}
                anchors.append({'frame':fr['frame'],'timestamp':fr['timestamp'],'actor':actor})
            anchor={'source_id':sid,'scene':c['scene'],'source_split':c['source_split'],'asset':asset,'frames':anchors}
            tr,why=f.legacy.trajectory(g,c,anchor,0.,static=False)
            if tr is not None and any(not f.valid_projection(f.projection(r['actor'],fr)) for r,fr in zip(tr['frames'],c['frames'])):tr=None;why='r23_projection_or_ego_margin'
            if tr is not None:
                tr.update(proposal_policy=mode,trajectory_policy='world_static' if mode=='stationary_prior_pose' else 'replay_annotated_past_track',
                    intended_B=tok,history_time_shift_s=-2.,history=[{k:a[k] for k in ['source_annotation_tokens','source_timestamp']} for a in selected])
                cid=f'T{len(rows)+1:03}';rows.append({'case_id':cid,'source_id':sid,'trajectory':clean(tr),'reuse_saved_pixels':False,
                    'sorted_first_rejection':'not_a_selector_pending_complete_instance_QA'});rec.update(result='spatial_candidate',case_id=cid)
            else:rec['result']=why
            audit_rows.append(rec)
        print('PRIOR_POSE',sid,[r['result'] for r in audit_rows if r['source_id']==sid],flush=True)
    roster={'selection_id':'past_real_vehicle_2s_v1','selection_frozen_before_extra_Y_SAM':True,'cases':rows,'count':len(rows),
        'saved_pixel_reuse':0,'sources':sorted(g.sources),'training_admission':0,'training_steps':0}
    f.dump(O/'candidate_roster.json',roster);f.dump(O/'proposal_audit.json',{'rows':audit_rows,'counts':dict(Counter(r['result'] for r in audit_rows))})
    if not (O/'prepared.json').exists():recover.prepare(g,roster,recovery=False)
    # 同源同实例完整Y标签只复用为评价；不进入任何方法条件目录。
    labelroot=O/'quality_labels';state=f.read(labelroot/'segmentation_state.json');done={r['job_id'] for r in state['completed']};reused=0
    for job in f.read(labelroot/'observed_queue.json')['jobs']:
        jid=job['job_id'];old=f.T/'r38/quality_labels/observed_masks'/jid
        if jid in done or not (old/'result.json').exists():continue
        src=f.read(old/'result.json');assert src['instance_token']==job['instance_token'] and src['case_id']==job['case_id'] and src['frames']==job['frames']
        dest=labelroot/'observed_masks'/jid;shutil.copytree(old,dest,dirs_exist_ok=True);state['completed'].append(src|{'reused_from':str(old)});done.add(jid);reused+=1
    f.dump(labelroot/'segmentation_state.json',state)
    f.dump(O/'controller_state.json',{'stage':'prepared_pending_full_instance_labels','pid':os.getpid(),'spatial_candidates':len(rows),
        'extra_reused_r38_QA_jobs':reused,'total_QA_jobs':len(f.read(labelroot/'observed_queue.json')['jobs']),'seconds':time.time()-start,'training_steps':0})
    print('PRIOR_POSE_PREPARED',len(rows),flush=True)


def evaluate():
    g=setup();register(g);audit.main(output_root=O,reveal_policy='primary_plus_preserved')
    result=f.read(O/'instance_quality.json');f.dump(O/'controller_state.json',{'stage':'complete_pending_independent_QA',
        'counts':result['counts'],'training_steps':0,'training_admission':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['propose','evaluate']);a=p.parse_args();(propose if a.stage=='propose' else evaluate)()
