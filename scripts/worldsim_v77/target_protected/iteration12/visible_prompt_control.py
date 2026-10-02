"""r40固定像素/轨迹，仅检查SAM提示帧选择是否导致身份重复。"""
from pathlib import Path
import sys,copy,os,shutil,time,argparse
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import fresh_reveal_pipeline as base
import evaluate_instance_audit as audit
from actor_state import Observation,cuboid_front_depth
import numpy as np
f=base.f;O=f.T/'r40'


def prepare():
    g=base.setup();O.mkdir(exist_ok=True)
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r40','phase':'SAM_prompt_visibility_control',
        'question':'separate GT-box max-area prompting from ambiguous instance labels before blaming data generation',
        'base_runs':['r38','r39'],'fixed_inputs':'all23 saved trajectories, RGB, final H, frame count and quality gates unchanged',
        'only_change':'SAM2 prompt frame selected by GT-cuboid first-return visible projected pixels, not raw box area or fixed midpoint',
        'ranking_resolution':[256,144],'score':'visible pixel count, then fraction, then nearest middle, then earlier frame',
        'visibility_depth_tolerance_m':.05,'valid_pose_only':True,'max_geometry_gap_s':.65,
        'same_seed_checkpoint_box_only_prompt':True,'no_new_points_or_postprocess':True,
        'GT_proxy_role':'offline Y-quality prompt ranking only, not proof of pixel identity and not a method condition',
        'Y_RGB_role':'isolated quality labels; raw images neither changed nor used to rank frames',
        'no_new_proposals_speeds_offsets_or_quality_thresholds':True,'training_steps':0,'training_admission':0,
        'stop':'one paired prompt rule, no anchor or point grid; independent QA remains necessary',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert f.read(O/'run.json')==plan
    else:f.dump(O/'run.json',plan)
    assert not (O/'prepared.json').exists(),'已有队列不得重复准备'
    E=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r40');E.mkdir(exist_ok=True,parents=True);shutil.copy2(O/'run.json',E/'run.json')
    cases=[];all_jobs={};base_origins={}
    for run in plan['base_runs']:
        cases.extend(f.read(f.T/run/'prepared.json')['cases'])
        for job in f.read(f.T/run/'quality_labels/observed_queue.json')['jobs']:
            if job['job_id'] not in all_jobs:all_jobs[job['job_id']]=job;base_origins[job['job_id']]=f.T/run/'quality_labels/observed_masks'/job['job_id']
    label=O/'quality_labels';label.mkdir(exist_ok=True);completed=[];jobs=[];changes=[];start=time.time()
    corrections=[]
    for jid,job in all_jobs.items():
        rec=f.read(base_origins[jid]/'result.json');sid=job['case_id'];c=g.sources[sid]
        if rec.get('reused_from','').startswith('r37 technically passed Y-SAM'):
            j=next(j for j,a in enumerate(c['actors']) if a['instance_token']==job['instance_token'])
            original=base.ROOT/('segmented' if j==0 else 'segmented_secondary')/(sid if j==0 else sid+'_'+job['instance_token'][:8])
            actual=f.read(original/'mask_manifest.json');pf=actual['prompt_frame'];old_recorded=job['prompt_frame']
            job['prompt_frame']=pf;job['prompt_box']=next(a for a in c['frames'][pf]['actors'] if a['instance_token']==job['instance_token'])['projection']['box_xyxy']
            # 旧标签用quality96 JPEG；保持完全相同的JPEG字节，不能换成quality98而声称只改frame。
            wrapper=label/'original_SAM_inputs'/sid;wrapper.mkdir(parents=True,exist_ok=True)
            for name,target in [('sam_rgb',base.ROOT/'SAM2_rgb'/sid),('H',Path(job['input_folder'])/'H')]:
                if not (wrapper/name).exists():(wrapper/name).symlink_to(target,target_is_directory=True)
            job['input_folder']=str(wrapper)
            corrections.append({'job_id':jid,'old_recorded_frame':old_recorded,'actual_original_frame':pf,
                'actual_recipe':str(original/'mask_manifest.json'),'original_SAM_rgb_folder':str(base.ROOT/'SAM2_rgb'/sid),
                'same_JPEG_bytes_for_control':True})
    for sid in sorted({j['case_id'] for j in all_jobs.values()}):
        g.prepare(sid);c=g.sources[sid];frames=c['frames'];records={}
        for i,(fr,objects) in enumerate(zip(frames,g.obstacles[sid])):
            K=np.asarray(fr['intrinsics_1024']).copy();K[:2]/=4
            ob=Observation(np.zeros((144,256,3),np.uint8),np.zeros((144,256),bool),np.asarray(fr['camera_to_world']),K,fr['timestamp'])
            # 其他物体不确定的框仍保守挡住背后目标；未知不能当作空。
            front=cuboid_front_depth(ob,[a for a in objects if a['instance_token']!='ego_conservative_proxy'])
            for job in [j for j in all_jobs.values() if j['case_id']==sid]:
                tok=job['instance_token'];a=next((x for x in objects if x['instance_token']==tok),None)
                if a is None or a.get('interpolation_uncertain') or a['_projection'] is None:continue
                own=cuboid_front_depth(ob,[a]);valid=np.isfinite(own);visible=valid&(own<=front+.05)
                records.setdefault(tok,[]).append({'frame':i,'visible_pixels':int(visible.sum()),'projected_pixels':int(valid.sum()),
                    'fraction':float(visible.sum()/max(1,valid.sum())),'box':a['_projection']['box'].tolist()})
        for old in [j for j in all_jobs.values() if j['case_id']==sid]:
            job=copy.deepcopy(old);items=records.get(job['instance_token'],[])
            eligible=[r for r in items if r['visible_pixels']>0]
            selected=max(eligible,key=lambda r:(r['visible_pixels'],r['fraction'],-abs(r['frame']-15),-r['frame'])) if eligible else None
            if selected:job.update(prompt_frame=selected['frame'],prompt_box=selected['box'])
            changed=job['prompt_frame']!=old['prompt_frame']
            # 对同帧保留原bbox的精确浮点数，确保只比较提示帧，而不是额外box变化。
            if not changed:job['prompt_box']=old['prompt_box']
            job.update(prompt_policy='cuboid_visible_pixels_v1',original_prompt_frame=old['prompt_frame'])
            jobs.append(job);changes.append({'job_id':job['job_id'],'source_id':sid,'old_frame':old['prompt_frame'],
                'new_frame':job['prompt_frame'],'changed':changed,'valid_visible_frame_found':selected is not None,'scores':items})
            if not changed:
                source=base_origins[job['job_id']];dest=label/'observed_masks'/job['job_id'];assert (source/'result.json').exists()
                shutil.copytree(source,dest,dirs_exist_ok=True)
                record=f.read(source/'result.json')|job|{'reused_from':str(source),'original_metadata_path':str(source/'result.json')}
                f.dump(dest/'result.json',record);completed.append(record)
        print('VISIBLE_PROMPT',sid,len(changes),sum(r['changed'] for r in changes),flush=True)
    f.dump(label/'observed_queue.json',{'jobs':jobs,'role':'Y_evaluation_only'})
    f.dump(label/'segmentation_state.json',{'completed':completed,'stage':'pending_changed_prompt_jobs'})
    f.dump(O/'prepared.json',{'cases':cases,'quality_jobs':len(jobs),'reused_quality_jobs':len(completed),'training_admission':0})
    f.dump(O/'prompt_comparison.json',{'jobs':changes,'changed_jobs':sum(r['changed'] for r in changes),'unchanged_jobs':len(completed),
        'reused_label_provenance_repairs':corrections,'before_GPU_control':'corrected planned-vs-actual frame and JPEG source; old masks unchanged',
        'seconds':time.time()-start})
    f.dump(O/'controller_state.json',{'stage':'ready_for_changed_prompt_SAM','pid':os.getpid(),'changed_jobs':sum(r['changed'] for r in changes),'training_steps':0})


def evaluate():
    base.setup();audit.O=O;audit.main(output_root=O,reveal_policy='primary_plus_preserved')
    before={c['case_id']:c for run in ['r38','r39'] for c in f.read(f.T/run/'instance_quality.json')['cases']}
    after=f.read(O/'instance_quality.json');rows=[]
    for c in after['cases']:
        old=before[c['case_id']];rows.append({'case_id':c['case_id'],'before':old['reason'] or old['quality']['data_family'],
            'after':c['reason'] or c['quality']['data_family'],'old_identity_flags':len(old['identity_flags']),
            'new_identity_flags':len(c['identity_flags']),'technical_candidate':c['technical_candidate']})
    f.dump(O/'paired_quality.json',{'cases':rows,'before_counts':dict(Counter(r['before'] for r in rows)),
        'after_counts':dict(Counter(r['after'] for r in rows)),'training_steps':0,'visual_QA_required':True})
    f.dump(O/'controller_state.json',{'stage':'complete_prompt_control_pending_visual_checks','counts':after['counts'],'training_steps':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);a=p.parse_args();(prepare if a.stage=='prepare' else evaluate)()
