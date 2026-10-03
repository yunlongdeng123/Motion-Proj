"""GPU只验证已选额外参考的A剔除；不重选参考、不修改查询帧的r21 SAM。"""
from common import *
import fcntl, shutil
import numpy as np, cv2, torch
from PIL import Image
from actor_state import Observation, cuboid_front_depth
from prepare_inputs import letterbox


def main():
    if not torch.cuda.is_available():raise SystemExit('没有GPU；CPU阶段不执行SAM2')
    lock=open(O/'source_masks.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
        '/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
    plan=read(O/'manifest.json');rows=[]
    for c in plan['cases']:
        folder=O/'inputs'/c['case_id'];manifest=read(folder/'references.json');state=dict(np.load(folder/'condition.npz'))
        pending=[r for r in manifest['references'] if r['GPU_source_mask_validation_pending'] and not r['padding']]
        if not pending:continue
        backup=O/'before_changes/GPU_reference_masks'/c['case_id'];backup.mkdir(parents=True,exist_ok=True)
        for name in ['condition.npz','references.json']:
            if not (backup/name).exists():shutil.copy2(folder/name,backup/name)
        for r in pending:
            frame=r['source_frame_geometry'];rgb=np.asarray(Image.open(r['source_path']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
            obs=Observation(rgb,np.zeros((576,1024),bool),np.array(frame['camera_to_world']),np.array(frame['intrinsics_1024']),frame['timestamp'])
            actors=frame['actors'];target=[a for a in actors if a['instance_token']==c['target_token']]
            envelope=np.isfinite(cuboid_front_depth(obs,target))
            hole=cv2.dilate(envelope.astype('uint8'),np.ones((13,13),'uint8'))>0
            passed=True;reason='annotated target outside this camera';score=None
            if envelope.any():
                yy,xx=np.where(envelope);box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],np.float32)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    predictor.set_image(rgb);masks,scores,_=predictor.predict(box=box,multimask_output=True)
                best=int(np.argmax(scores));sam=masks[best].astype(bool);score=float(scores[best])
                # 对额外参考宁可不采用，不多轮调点/种子。主查询SAM完全不动。
                passed=bool(sam.any() and score>=.6 and (sam&envelope).sum()/max(1,sam.sum())>=.5)
                reason='SAM与A包络一致，联合擦除' if passed else 'SAM与A身份包络一致性不足，停用该参考slot'
                if passed:hole|=cv2.dilate(sam.astype('uint8'),np.ones((5,5),'uint8'))>0
            slot=r['reference_slot']
            if passed:
                rgb=rgb.copy();rgb[hole]=127
                picture,valid,_=letterbox(rgb,hole,r['letterbox']['roi_xyxy'])
                state['references'][slot]=picture;state['reference_valid'][slot]=valid
                Image.fromarray(picture).save(folder/f'reference_{slot:02}.png')
                Image.fromarray(hole.astype('uint8')*255).save(folder/f'reference_{slot:02}_source_mask.png')
            else:
                state['references'][slot]=127;state['reference_valid'][slot]=0
                Image.fromarray(state['references'][slot]).save(folder/f'reference_{slot:02}.png')
                r['disabled']=True
            r['GPU_source_mask_validation_pending']=False;r['GPU_source_mask_check_pass']=passed
            rows.append({'case_id':c['case_id'],'slot':slot,'passed':passed,'reason':reason,'SAM_score':score})
        np.savez_compressed(folder/'condition.npz',**state);dump(folder/'references.json',manifest)
    dump(O/'source_mask_validation.json',{'checks':rows,'query_edit_mask_changed':False,
        'source_reference_checks_are_heuristic':True,'failed_slots_disabled_without_reselection':True,'human_verdict':None})


if __name__=='__main__':main()
