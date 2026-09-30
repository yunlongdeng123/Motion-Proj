"""有限真实多视角SAM2预检。默认仅验证CPU输入；GPU明确授权后加--execute。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from donor_gate import load_policy,check_masks

CHECKPOINT='/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt'
CONFIG='configs/sam2.1/sam2.1_hiera_l.yaml'
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(root,execute):
    data=json.loads((root/'multiview/selected_views.json').read_text());policy=load_policy();jobs=[];rejected=[]
    for r in data['views']:
        reasons=[];b=r['box_xyxy'];p=root/'multiview/rgb'/r['filename']
        if not p.is_file():reasons.append('missing_rgb')
        if r['instance_token'] in policy['blocked_instances']:reasons.append('human_source_blocked')
        if b[3]+policy['clearance_px']>=576-policy['source_bottom_exclusion_px']:reasons.append('source_bottom_box_guard')
        if reasons:rejected.append({'view_id':r['view_id'],'reasons':reasons});continue
        with Image.open(p) as im:im.verify()
        jobs.append(r)
    queue={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':root.name,'state':'awaiting_gpu',
        'checkpoint':CHECKPOINT,'config':CONFIG,'jobs':jobs,'pre_gpu_rejected':rejected,
        'purpose':'同一真实instance多角度的实例清洁度预检；单帧mask不是时序视频或合格训练样本。',
        'frozen_selection':'SAM2 image multimask; save all outputs, best official predicted score as primary; do not choose by downstream success',
        'next':'只对新输出一次Sol xhigh非fast检查；通过后以源轨迹固定相机选择扩连续窗口，仍须逐新caseQA及用户全检。',
        'training_ready':0,'new_synthetic_cases':0}
    dump(root/'gpu_queue.json',queue);print('VALIDATED_GPU_QUEUE',len(jobs),'PRE_GATE_REJECTED',len(rejected),flush=True)
    if not execute:return
    # 用户明确开GPU后执行；进程锁防止重复启动。
    import torch,fcntl
    assert torch.cuda.is_available(),'等待用户开GPU，不做CPU模型推理'
    lock=open(root/'multiview/segmentation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(1);torch.manual_seed(42);predictor=SAM2ImagePredictor(build_sam2(CONFIG,CHECKPOINT,device='cuda'))
    out=root/'multiview/segmented';out.mkdir(exist_ok=True);results=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for job in jobs:
            dest=out/job['view_id'];dest.mkdir(exist_ok=True);record=dest/'mask_manifest.json'
            if record.is_file():
                old=json.loads(record.read_text())
                if old['source_filename']==job['filename'] and old['checkpoint']==CHECKPOINT and all((dest/n).is_file() for n in old['mask_files']):
                    for n in old['mask_files']:
                        with Image.open(dest/n) as im:im.load();assert im.size==(1024,576)
                    results.append(old);continue
            arr=np.asarray(Image.open(root/'multiview/rgb'/job['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS));start=time.monotonic();predictor.set_image(arr)
            masks,scores,_=predictor.predict(box=np.array(job['box_xyxy'],np.float32),multimask_output=True)
            files=[]
            for k,m in enumerate(masks):
                name=f'candidate_{k}.png';Image.fromarray(m.astype(np.uint8)*255).save(dest/name);files.append(name)
            best=int(np.argmax(scores));primary=masks[best];arr_out=np.where(primary[...,None],arr,127).astype(np.uint8);Image.fromarray(arr_out).save(dest/'isolated.png')
            row={'view_id':job['view_id'],'source_filename':job['filename'],'instance_token':job['instance_token'],
                'timestamp_us':job['timestamp_us'],'checkpoint':CHECKPOINT,'config':CONFIG,'mask_files':files,
                'scores':scores.tolist(),'primary_candidate':best,'mechanical_gate':check_masks(job['instance_token'],[primary]),
                'seconds':time.monotonic()-start,'quality_status':'pending_new_output_QA','human_verdict':None,'training_ready':False}
            dump(record,row);results.append(row);dump(root/'multiview/segmentation_state.json',{'state':'running','completed':results})
    dump(root/'multiview/segmentation_state.json',{'state':'complete_pending_QA','completed':results})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.root,a.execute)
