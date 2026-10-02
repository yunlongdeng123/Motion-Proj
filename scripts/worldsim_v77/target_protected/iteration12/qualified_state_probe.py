"""r29：唯一独立QA通过的新reveal候选，验证最终H遮后条件；0训练步。"""
from pathlib import Path
import sys,os,time,subprocess,fcntl,traceback
sys.path.insert(0,str(Path(__file__).parent))
import prepare_instance_audit as p
f=p.f;np=p.np;Image=p.Image
O=f.T/'r29';R=f.T/'r28';S=Path(__file__).parent


def prepare(qa=None,quality_root=R):
    qa=f.read(R/'r28_independent_quality.json') if qa is None else qa
    assert qa['assistant_grade']==2 and qa['usable_for_next_legal_condition_build']
    case=next(c for c in f.read(R/'prepared.json')['cases'] if c['case_id']==qa['case_id'])
    q=next(c for c in f.read(quality_root/'instance_quality.json')['cases'] if c['case_id']==case['case_id']);assert q['technical_candidate']
    assert not (O/'run.json').exists(),'已登记，不能覆盖实验'
    O.mkdir(exist_ok=True);cid=case['case_id'];dest=O/'observed'/cid
    for name in ['rgb','H','sam_rgb']:(dest/name).mkdir(parents=True,exist_ok=True)
    holes=[];origin=Path(case['observed_folder'])
    for i in range(30):
        x=np.asarray(Image.open(origin/'rgb'/f'{i:05}.png').convert('RGB')).copy()
        h=np.asarray(Image.open(origin/'proposal_H'/f'{i:05}.png'))>0;x[h]=127;holes.append(h)
        Image.fromarray(x).save(dest/'rgb'/f'{i:05}.png');Image.fromarray(h.astype('uint8')*255).save(dest/'H'/f'{i:05}.png')
        Image.fromarray(x).save(dest/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
    frames=case['frames'];actors=[];jobs=[]
    for token in q['quality']['protected_instances']:
        actor=next(a for a in case['retained_instances'] if a['instance_token']==token)
        assert all(a is not None and not a.get('interpolation_uncertain',False) for a in actor['annotations'])
        for i,fr in enumerate(frames):
            if not any(a['instance_token']==token for a in fr['actors']):fr['actors'].append(actor['annotations'][i])
        areas=[]
        for i,box in enumerate(actor['boxes']):
            if box is None:areas.append(0);continue
            x0,y0,x1,y1=p.roi(box);areas.append(int((~holes[i][y0:y1,x0:x1]).sum()) if x1>x0 and y1>y0 else 0)
        prompt=int(np.argmax(areas));assert areas[prompt]>=100
        actors.append({'instance_token':token,'category':actor['category']})
        jobs.append({'case_id':cid,'job_id':cid+'_'+token[:8],'instance_token':token,'category':actor['category'],
            'prompt_frame':prompt,'prompt_box':actor['boxes'][prompt],'frames':30,'input_folder':str(dest),
            'role':'final_H_masked_RGB_only','no_Y_or_full_RGB_access':True,'geometry_auxiliary':'GT poses declared POC auxiliary'})
    f.dump(dest/'observations.json',{'case_id':cid,'source_id':case['source_id'],'scene':case['scene'],
        'frames':frames,'actors':actors,'window_frames':30,'window_start':frames[0]['timestamp'],'window_end':frames[-1]['timestamp'],
        'rgb_source':'r28 guard-erased PNG additionally erased by final H BEFORE SAM or appearance extraction',
        'geometry_source':'GT_camera_and_tracks_POC_auxiliary','gt_read_for_state':False})
    f.dump(O/'observed_queue.json',{'jobs':jobs,'input_boundary':'final_masked_fixed_window_only','frames':30,'human_verdict':None})
    f.dump(O/'run.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':O.name,'phase':'new_reveal_candidate_legal_condition',
        'probe_cases':[cid],'source_run':'r28','frozen_H':'r21 sam_full_v2','frames':30,'seed':42,'training_steps':0,'training_admission':0,
        'Y_role':'only prior independent data QA; not opened here','input_quality_root':str(quality_root),
        'surfel':False,'failure_ledger_refs':['V77-F02'],'human_verdict':None})


def main():
    if not (O/'run.json').exists():prepare()
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'stage':'final_H_observed_SAM','pid':os.getpid(),'training_steps':0};f.dump(O/'controller_state.json',state)
    try:
        if not (O/'segmentation_state.json').exists() or f.read(O/'segmentation_state.json')['stage']!='complete_pending_identity_review':
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU忙，不重复'
            with (O/'sam.log').open('a') as log:subprocess.run(['/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python','-u',str(S/'segment_retained.py'),'--root',str(O)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        state['stage']='building_sparse_legal_state';f.dump(O/'controller_state.json',state)
        f.legacy.O=f.O;f.legacy.ROOT=f.ROOT
        import build_state_probe as b
        if not (O/'state_summary.json').exists():b.main(run_root=O,source_root=f.ROOT,geometry_factory=f.legacy.geometry)
        state['stage']='complete_pending_condition_quality';f.dump(O/'controller_state.json',state)
    except Exception as e:
        state.update(stage='engineering_error',error=repr(e),traceback=traceback.format_exc());f.dump(O/'controller_state.json',state);raise

if __name__=='__main__':main()
