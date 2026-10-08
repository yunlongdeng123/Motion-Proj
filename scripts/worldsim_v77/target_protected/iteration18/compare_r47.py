"""r50/r51 固定样本补跑原 r47 权重；复用原条件、源mask与推理实现。"""
from pathlib import Path
import argparse, os, sys, json, subprocess, glob, time
from collections import defaultdict

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO=Path('/root/autodl-tmp/motion_proj_v77')
BASE=REPO/'scripts/worldsim_v77/target_protected/iteration14'
sys.path.insert(0,str(BASE))
import common
T=common.T
O=T/'r51/r47_comparison'
common.O=O
sys.path.insert(0,str(BASE))
from common import read,dump,images


def admitted():
    cases=[]
    for batch in ['r50','r51']:
        m=read(T/batch/'manifest.json')
        gate=read(T/batch/'mask_visual_review.json') if (T/batch/'mask_visual_review.json').exists() else None
        approved=set(gate['approved']) if gate else {c['case_id'] for c in m['cases']}
        for c in m['cases']:
            if c['case_id'] in approved:cases.append(dict(c,source_batch=batch))
    return cases


def metadata():
    if (O/'manifest.json').exists():print('METADATA_EXISTS');return
    import numpy as np
    from prepare import rows,annotation_view,camera_actors
    from geometry_factory import transform
    O.mkdir(parents=True,exist_ok=True)
    source=admitted();meta=common.A/'metadata/v1.0-trainval'
    samples={r['token']:r for r in read(meta/'sample.json')}
    sensors={r['token']:r for r in read(meta/'sensor.json')}
    cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    scene_tokens={r['name']:r['token'] for r in read(meta/'scene.json')}
    scene_samples=defaultdict(list)
    for s in samples.values():scene_samples[s['scene_token']].append(s)
    keyframes={};primary={};needed=set()
    for c in source:
        primary[c['case_id']]={f['sample_data_token']:i for i,f in enumerate(c['frames'])}
        times=[c['frames'][0]['timestamp']-1000000,c['frames'][0]['timestamp'],
               c['frames'][-1]['timestamp'],c['frames'][-1]['timestamp']+1000000]
        pool=scene_samples[scene_tokens[c['scene']]]
        tokens={min(pool,key=lambda s:abs(s['timestamp']-t))['token'] for t in times}
        keyframes[c['case_id']]=tokens;needed|=tokens
    query_tokens=set().union(*(set(x) for x in primary.values()))
    sd={};scans=[]
    for d in rows(meta/'sample_data.json'):
        channel=sensors[cal[d['calibrated_sensor_token']]['sensor_token']]['channel']
        if d['token'] in query_tokens or (d['is_key_frame'] and d['sample_token'] in needed and channel.startswith('CAM_')):
            sd[d['token']]=dict(d,camera=channel)
        if d['is_key_frame'] and d['sample_token'] in needed and channel=='LIDAR_TOP':scans.append(d)
    needed|={d['sample_token'] for d in sd.values()}
    needed|={samples[t]['prev'] for t in list(needed) if samples[t]['prev']}
    cats={r['token']:r['name'] for r in read(meta/'category.json')}
    instances={r['token']:cats[r['category_token']] for r in read(meta/'instance.json')}
    ann=defaultdict(list)
    for a in rows(meta/'sample_annotation.json'):
        if a['sample_token'] in needed:
            ann[a['sample_token']].append(dict(a,category=instances[a['instance_token']],
                category_name=instances[a['instance_token']],timestamp=samples[a['sample_token']]['timestamp']))
    ego_tokens={d['ego_pose_token'] for d in list(sd.values())+scans}
    ego={e['token']:e for e in rows(meta/'ego_pose.json') if e['token'] in ego_tokens}
    sdk=annotation_view(samples,sd,ann)
    frames={}
    for tok,d in sd.items():
        cs=cal[d['calibrated_sensor_token']];ep=ego[d['ego_pose_token']]
        k=np.array(cs['camera_intrinsic']);k[0]*=1024/d['width'];k[1]*=576/d['height']
        frames[tok]={'timestamp':d['timestamp'],'sample_data_token':tok,'sample_token':d['sample_token'],
            'is_key_frame':d['is_key_frame'],'camera_to_world':(transform(ep['translation'],ep['rotation'])@transform(cs['translation'],cs['rotation'])).tolist(),
            'ego_to_world':transform(ep['translation'],ep['rotation']).tolist(),
            'intrinsics_1024':k.tolist(),'actors':camera_actors(sdk,d)}
    inventory=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*.jpg','-g','*.bin',
        '/root/autodl-tmp/data',str(T),str(common.A)],capture_output=True,text=True,check=True).stdout.splitlines()
    byname=defaultdict(list)
    for p in inventory:byname[Path(p).name].append(p)
    shards=defaultdict(set)
    for pattern in ('/root/autodl-tmp/data/worldsim_v67/*member_shards*.json',
                    '/root/autodl-tmp/data/worldsim_v5/manifests/*member_shards*.json',
                    '/root/autodl-tmp/data/dynamic_editing_v2/manifests/*member_shards*.json'):
        for p in glob.glob(pattern):
            for name,shard in read(p).items():
                shard=str(shard);shard=f'{int(shard):02}' if shard.isdigit() else shard.split('trainval',1)[1][:2]
                shards[Path(name).name.split('__',1)[0]].add(shard)
    wanted={};cases=[]
    def locate(d,kind):
        paths=byname.get(Path(d['filename']).name,[])
        if paths:return paths[0]
        wanted[d['filename']]={'filename':d['filename'],'kind':kind,
            'shards':sorted(shards[Path(d['filename']).name.split('__',1)[0]])}
        return None
    for c in source:
        cid=c['case_id'];query=[]
        for old in c['frames']:
            f=frames[old['sample_data_token']]
            assert f['timestamp']==old['timestamp']
            assert np.allclose(f['camera_to_world'],old['camera_to_world'],atol=1e-8)
            assert np.allclose(f['intrinsics_1024'],old['intrinsics_1024'],atol=1e-8)
            assert any(a['instance_token']==c['instance_token'] for a in f['actors'])
            query.append(f)
        refs=[]
        for tok,d in sd.items():
            if tok not in primary[cid] and d['sample_token'] not in keyframes[cid]:continue
            i=primary[cid].get(tok)
            if i is not None:
                path=str(Path(c['folder'])/'rgb'/f'{i:05}.jpg')
                mask=str(Path(c['folder'])/'model_mask'/f'{i:05}.png');kind='r21_full_SAM'
            else:path=locate(d,'RGB');mask=None;kind='GT_target_envelope_source_exclusion'
            refs.append({'source_frame':i,'camera':d['camera'],'frame':frames[tok],
                'path':path,'mask_path':mask,'source_kind':kind,'filename':d['filename'],
                'available':path is not None,'target_excluded_before_encoding':True,
                'auxiliary_mask_requires_GPU_validation':i is None})
        lidar=[]
        for d in scans:
            if d['sample_token'] in keyframes[cid]:
                lidar.append(dict(d,path=locate(d,'LiDAR'),ego_pose=ego[d['ego_pose_token']],
                    calibrated_sensor=cal[d['calibrated_sensor_token']],actors=ann[d['sample_token']]))
        cases.append({'case_id':cid,'scene':c['scene'],'camera':c['camera'],'source_batch':c['source_batch'],
            'kind':'real','split':'real_train_DEV','folder':c['folder'],'frame_indices':list(range(10)),
            'frames':query,'target_token':c['instance_token'],'scans':lidar,'bev_scans':lidar,
            'references':sorted(refs,key=lambda r:(r['frame']['timestamp'],r['camera'])),'GT':None})
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r51','phase':'r47 catch-up and new-scene comparison',
        'cases':cases,'branch_checkpoint':str(T/'r47/training/branch_0320.safetensors'),
        'eval_seed':42,'eval_steps':25,'training_steps':0,'arms':['RGB_and_geometry'],
        'reference_policy':'original r47 fixed top4 visible protected crops +2 contexts, 6 slots; raw bank is 10 existing SAM query frames plus six-camera keyframes at query-start-1s/start/end/end+1s and available same-scene query exposures attached to those sample tokens; original r47 bank had26 SAM frames',
        'query_rgb_mask_alpha_unchanged':True,'main_and_branch_weights_frozen':True,
        'geometry_auxiliary':'official GT pose/tracks + real LiDAR; O is cuboid proxy, N observed points, U unknown',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    dump(O/'manifest.json',plan);dump(O/'extra_input_plan.json',{'files':list(wanted.values())})
    missing_shards=[w['filename'] for w in wanted.values() if not w['shards']]
    dump(O/'metadata_check.json',{'cases':len(cases),'query_geometry_exact':True,'missing_files':len(wanted),
        'missing_shard_hints':missing_shards,'source_batches':dict(__import__('collections').Counter(c['source_batch'] for c in cases))})
    print('METADATA',len(cases),'MISSING',len(wanted),'NO_SHARD',len(missing_shards),flush=True)
    assert not missing_shards,'需补充文件分片索引，不能静默丢失先验'


def extract():
    import extract_inputs
    extract_inputs.main()


def prepare():
    import prepare_inputs
    plan=read(O/'manifest.json');active={c['case_id'] for c in admitted()}
    plan['cases']=[c for c in plan['cases'] if c['case_id'] in active]
    dump(O/'manifest.json',plan)
    prepare_inputs.main()


def source_masks():
    import validate_source_masks
    validate_source_masks.main()


def evaluate():
    import fcntl,signal,numpy as np,torch
    from PIL import Image
    from interface import DeletionRequest,encode_references,check_request
    from multi_prior import MultiPriorBranch
    from engine import MultiPriorDeletionEngine
    from repair_drive import set_seed
    from safetensors.torch import load_file
    assert torch.cuda.is_available()
    def timeout(*_):raise TimeoutError('单车10帧r47推理超过240秒，停止检查，不更换seed')
    signal.signal(signal.SIGALRM,timeout)
    assert (O/'source_mask_validation.json').exists()
    lock=open(O/'GPU.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(O/'manifest.json');torch.set_num_threads(4)
    branch=MultiPriorBranch().cuda()
    branch.load_state_dict(load_file(plan['branch_checkpoint']),strict=True)
    engine=MultiPriorDeletionEngine(branch)
    state={'stage':'r47_running','pid':os.getpid(),'completed':[],'total':len(plan['cases']),'training_steps':0}
    dump(O/'evaluation_state.json',state)
    for c in plan['cases']:
        cid=c['case_id'];out=O/'evaluation'/cid
        if (out/'result.json').exists():state['completed'].append(read(out/'result.json'));continue
        p=O/'inputs'/cid
        priors=dict(np.load(p/'condition.npz'))
        req=DeletionRequest(images(c,'rgb'),images(c,'hole')>0,images(c,'alpha').astype('float32')/255,
            priors,read(p/'references.json'))
        checks=check_request(req);assert not checks['pending_GPU_reference_slots']
        ref=encode_references(engine.model,req)
        engine.get_deletion(req,arm='RGB_and_geometry',reference_latents=ref)
        set_seed(42);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        signal.alarm(240)
        try:
            with torch.inference_mode():engine.predict(1,False,'Deletion')
        finally:signal.alarm(0)
        assert engine.prior_cfg_calls=={'unconditional':25,'conditional':25}
        raw=np.stack(engine.im_result);assert raw.shape==req.target_rgb.shape
        comp=req.compose(raw)
        for folder,array in [('native',raw),('compose',comp)]:
            (out/folder).mkdir(parents=True,exist_ok=True)
            for i,im in enumerate(array):Image.fromarray(im).save(out/folder/f'{i:05}.png')
        result={'case_id':cid,'source_batch':c['source_batch'],'arm':'r47_RGB_and_geometry',
            'branch_checkpoint':plan['branch_checkpoint'],'seconds':time.monotonic()-started,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'seed':42,'steps':25,
            'CFG_prior_calls':engine.prior_cfg_calls.copy(),'request_checks':checks,
            'human_verdict':None,'temporal_verdict':None,'training_steps':0}
        dump(out/'result.json',result);state['completed'].append(result);dump(O/'evaluation_state.json',state)
        print('R47',cid,round(result['seconds'],1),flush=True)
    state['stage']='complete';dump(O/'evaluation_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['metadata','extract','prepare','source-masks','evaluate'])
    a=p.parse_args();{'metadata':metadata,'extract':extract,'prepare':prepare,'source-masks':source_masks,'evaluate':evaluate}[a.phase]()
