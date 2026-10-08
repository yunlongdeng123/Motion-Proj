"""r50两个GPU入口：完整SAM后看身份，再每车一个官方DELETE窗。"""
from common import *
import argparse, fcntl, time
import numpy as np
import cv2
from PIL import Image


def require_gpu():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('当前无GPU；不在CPU加载模型，不后台等卡')
    if not read(O/'cpu_check.json')['CPU_ready']:raise RuntimeError('CPU输入尚未就绪')
    m=read(O/'manifest.json')
    assert all(c.get('input_quality_score')==2 and c.get('structural_audit_eligible') for c in m['cases']), '只允许独立2分输入运行GPU'
    torch.set_num_threads(1);cv2.setNumThreads(1)
    return torch


def hull(polygon):
    result=np.zeros((576,1024),np.uint8)
    cv2.fillConvexPoly(result,np.asarray(polygon,np.int32),1)
    return result.astype(bool)


def masks():
    torch=require_gpu();m=read(O/'manifest.json')
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    from mask_contract import prepare_masks
    torch.manual_seed(42)
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
        '/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda')
    state={'stage':'SAM_running','pid':os.getpid(),'completed':[]}
    dump(O/'mask_state.json',state)
    dump(O/'controller_state.json',{'stage':state['stage'],'pid':os.getpid(),
        'host_alias':os.environ.get('V77_HOST_ALIAS','unknown'),'GPU_jobs':1,'training_steps':0})
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for c in m['cases']:
            if c.get('structural_audit_eligible') is False:continue
            p=Path(c['folder']);record=p/'mask_result.json'
            if record.exists():state['completed'].append(read(record));continue
            t=time.monotonic();track=predictor.init_state(video_path=str(p/'rgb'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            prompt=c['prompt_frame'];box=c['frames'][prompt]['target']['box_xyxy']
            predictor.add_new_points_or_box(track,frame_idx=prompt,obj_id=1,box=np.array(box,dtype='float32'))
            raw={}
            for reverse in [False,True]:
                for i,_,logit in predictor.propagate_in_video(track,start_frame_idx=prompt,reverse=reverse):
                    raw[int(i)]=(logit[0,0]>0).cpu().numpy()
            assert set(raw)==set(range(10));stats=[]
            for f in c['frames']:
                i=f['frame'];neighbors=np.zeros((576,1024),bool)
                for b in f['neighbors']:neighbors|=hull(b['hull'])
                arrays,info=prepare_masks(raw[i],hull(f['target']['hull']),neighbors)
                for name,array in dict(sam=raw[i],**arrays).items():
                    (p/name).mkdir(exist_ok=True)
                    Image.fromarray(np.rint(array*255).astype('uint8')).save(p/name/f'{i:05}.png')
                stats.append(dict(info,frame=i))
            row={'case_id':c['case_id'],'seconds':time.monotonic()-t,'prompt_frame':prompt,
                 'mask_policy':'sam_full_v2','stats':stats,'empty_frames':[r['frame'] for r in stats if r['empty_target']],
                 'identity_review':None,'GT_overlap_is_diagnostic_only':True}
            dump(record,row);state['completed'].append(row);dump(O/'mask_state.json',state)
            print('SAM',c['case_id'],row['empty_frames'],flush=True);del track,raw;torch.cuda.empty_cache()
    state['stage']='SAM_complete_pending_identity_review';dump(O/'mask_state.json',state)
    dump(O/'controller_state.json',{'stage':state['stage'],'GPU_jobs':0,'mask_cases':len(state['completed']),'training_steps':0})


def delete():
    torch=require_gpu();m=read(O/'manifest.json');gate=read(O/'mask_visual_review.json')
    assert set(gate['approved'])|set(gate['rejected'])=={c['case_id'] for c in m['cases']}
    assert not set(gate['approved'])&set(gate['rejected'])
    assert all(c['case_id'] in gate['rejected'] for c in m['cases'] if c.get('structural_audit_eligible') is False)
    from repair_drive import Engine,set_seed
    import signal
    def timeout(*_):raise TimeoutError('单车10帧DELETE超过240秒，停止，不换seed补救')
    signal.signal(signal.SIGALRM,timeout)
    dump(O/'controller_state.json',{'stage':'DELETE_loading','pid':os.getpid(),
        'host_alias':os.environ.get('V77_HOST_ALIAS','unknown'),'GPU_jobs':1,'training_steps':0})
    engine=Engine();state={'stage':'DELETE_running','pid':os.getpid(),'completed':[],'rejected_inputs':gate['rejected']}
    dump(O/'drive_state.json',state)
    dump(O/'controller_state.json',{'stage':state['stage'],'pid':os.getpid(),
        'host_alias':os.environ.get('V77_HOST_ALIAS','unknown'),'GPU_jobs':1,'training_steps':0})
    for c in m['cases']:
        cid=c['case_id'];p=Path(c['folder'])
        if cid not in gate['approved']:continue
        if (p/'result.json').exists():state['completed'].append(read(p/'result.json'));continue
        assert not read(p/'mask_result.json')['empty_frames']
        images=[np.array(Image.open(p/'rgb'/f'{i:05}.jpg').convert('RGB')) for i in range(10)]
        holes=[np.array(Image.open(p/'model_mask'/f'{i:05}.png'))>0 for i in range(10)]
        engine.im=images;engine.masks=holes;engine.previous_segment_last_frame=None
        engine.im_result=[];set_seed(m['seed']);t=time.monotonic();torch.cuda.reset_peak_memory_stats()
        signal.alarm(240)
        try:
            with torch.inference_mode():engine.predict(1,False,'Deletion')
        finally:signal.alarm(0)
        assert len(engine.im_result)==10 and not engine.used_previous_segment_condition
        for name in ['native','compose']:(p/name).mkdir(exist_ok=True)
        checks=[]
        for i,(im,h,raw) in enumerate(zip(images,holes,engine.im_result)):
            alpha=np.array(Image.open(p/'alpha'/f'{i:05}.png')).astype('float32')/255
            core=np.array(Image.open(p/'core'/f'{i:05}.png'))>0
            assert raw.shape==im.shape==(576,1024,3) and raw.dtype==np.uint8
            assert np.all(alpha[core]==1) and np.all(alpha[~h]==0)
            comp=np.rint(alpha[...,None]*raw+(1-alpha[...,None])*im).astype('uint8')
            assert np.array_equal(comp[~h],im[~h]) and np.array_equal(comp[core],raw[core])
            Image.fromarray(raw).save(p/'native'/f'{i:05}.png');Image.fromarray(comp).save(p/'compose'/f'{i:05}.png')
            checks.append({'frame':i,'outside_H_exact':True,'full_SAM_native_exact':True})
        row={'case_id':cid,'seconds':time.monotonic()-t,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
             'seed':42,'steps':25,'checkpoint':m['model_checkpoint'],'adapter':False,'previous_condition':False,
             'checks':checks,'hidden_GT':None,'human_verdict':None,'temporal_verdict':None}
        dump(p/'result.json',row);state['completed'].append(row);dump(O/'drive_state.json',state)
        print('DELETE',cid,row['seconds'],flush=True)
    state['stage']='DELETE_complete_pending_structure_review';dump(O/'drive_state.json',state)
    dump(O/'controller_state.json',{'stage':state['stage'],'GPU_jobs':0,'windows':len(state['completed']),
         'training_steps':0,'human_verdict':None})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['masks','delete']);args=p.parse_args()
    O.mkdir(exist_ok=True);lock=open(O/'GPU.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:(masks if args.phase=='masks' else delete)()
    except Exception as error:
        dump(O/'controller_state.json',{'stage':'GPU_engineering_error','error':repr(error),'GPU_jobs':0,'training_steps':0})
        raise
