"""核对20景独立2分输入和GPU前置；不创建模型或自动等卡。"""
from common import *
from collections import Counter
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw
from nuscenes.utils.splits import train


def contacts(cases):
    dest=O/'review'/'input_contacts';dest.mkdir(exist_ok=True)
    for start in range(0,len(cases),10):
        part=cases[start:start+10];board=Image.new('RGB',(1440,1630),(18,24,34));draw=ImageDraw.Draw(board)
        for j,c in enumerate(part):
            x=(j%2)*720;y=(j//2)*326;f=c['frames'][5]
            im=Image.open(Path(c['folder'])/'rgb/00005.jpg').convert('RGB')
            boxed=im.copy();d=ImageDraw.Draw(boxed);d.rectangle(f['target']['box_xyxy'],outline=(255,220,20),width=3)
            boxed.thumbnail((384,216));board.paste(boxed,(x,y+24))
            crop=im.crop(tuple(map(int,f['target']['box_xyxy'])));crop.thumbnail((320,216));board.paste(crop,(x+392,y+24))
            draw.text((x+4,y+4),c['case_id']+' '+c['scene']+' '+c['camera'],fill='white')
            draw.text((x+392,y+243),'Target A original crop',fill=(255,220,30))
            ref=c.get('best_protected_reference')
            if ref:
                draw.text((x+4,y+250),f"B frame {ref['frame']}, A/B bbox overlap {ref['target_bbox_overlap_fraction']:.0%}",fill=(50,235,120))
                draw.text((x+4,y+272),'B: projected vehicle, not exact visibility',fill=(160,190,170))
            else:draw.text((x+4,y+250),'No behind-car bbox candidate',fill=(160,190,170))
            draw.text((x+4,y+296),'; '.join(c['difficulty_factors']),fill=(190,195,220))
        board.save(dest/f'inputs_{start//10+1:02}.jpg',quality=94)


def main():
    m=read(O/'manifest.json');cs=m['cases'];counts=Counter(c['scene'] for c in cs)
    assert len(counts)==20 and 40<=len(cs)<=50 and all(2<=n<=3 for n in counts.values())
    assert set(counts).issubset(train) and not set(counts)&set(m['reserve_scenes'])
    assert len(set(m['reserve_scenes']))==5 and set(m['reserve_scenes']).issubset(train)
    assert not set(m['reserve_scenes'])&set(read(O/'sampling.json')['prior_source_scenes'])
    assert len({(c['scene'],c['instance_token']) for c in cs})==len(cs)
    assert all(c.get('input_quality_score')==2 and c.get('structural_audit_eligible') for c in cs)
    independent=read(O/'input_quality_review.json')
    approved={c['case_id'] for c in independent['cases'] if c['score']==2 and c['status']=='pass'}
    assert {c['case_id'] for c in cs}.issubset(approved)
    assert not {c['case_id'] for c in cs}&set(m.get('input_excluded_cases',[])+m.get('quality_uncertain_cases',[]))
    assert m['training_steps']==0 and not m['adapter'] and not m['temporal_module_change']
    ck=Path(m['model_checkpoint']);sk=Path(m['SAM_checkpoint']);assert ck.is_file() and sk.is_file()
    gaps=[];videos=[];keyframes=0
    for c in cs:
        assert len(c['frames'])==10 and len(c['source_RGB_paths'])==10
        assert len({f['sample_data_token'] for f in c['frames']})==10
        stamps=[f['timestamp'] for f in c['frames']];dt=np.diff(stamps)/1e6
        assert np.all((dt>=.04)&(dt<=.17));gaps.extend(dt.tolist())
        assert c['hidden_region_GT'] is None and c['human_verdict'] is None
        for f in c['frames']:
            assert f['target']['instance_token']==c['instance_token'] and f['target']['category']=='vehicle.car'
            if f['is_key_frame']:
                assert f['pose_source']=='SDK associated keyframe';keyframes+=1
            cp=np.array(f['camera_to_world']);assert cp.shape==(4,4) and np.isfinite(cp).all()
            assert np.array(f['intrinsics_1024']).shape==(3,3)
            with Image.open(Path(c['folder'])/'rgb'/f"{f['frame']:05}.jpg") as im:assert im.size==(1024,576)
        p=O/'review'/c['case_id']/'original.mp4'
        v=cv2.VideoCapture(str(p));decoded=0
        while True:
            ok,frame=v.read()
            if not ok:break
            assert frame.shape==(576,1024,3);decoded+=1
        v.release();assert decoded==10
        videos.append({'case_id':c['case_id'],'actual_decoded_frames':10,'resolution':[576,1024]})
    contacts(cs)
    result={'task_id':TASK,'run_id':'r50','CPU_ready':True,'scene_count':20,'case_count':len(cs),
        'scene_targets':dict(counts),'reserve_scene_count':5,'scene_disjoint':True,
        'actual_RGB_decoded':10*len(cs),'original_videos_decoded':videos,'SDK_keyframes_direct':keyframes,
        'independent_quality_score2_only':True,'eligible_cases':len(cs),'eligible_scenes':20,
        'input_candidate_count':m['input_candidate_count'],
        'actual_exposure_gap_min_median_max_s':[min(gaps),float(np.median(gaps)),max(gaps)],
        'official_checkpoint_exists':True,'SAM_checkpoint_exists':True,'GPU_jobs':0,'training_steps':0,
        'mask_identity_review':'after SAM2 before DELETE, not approved by GT overlap',
        'clear_B_evidence':'geometry proxy only; actual visibility reviewed separately',
        'human_verdict':None}
    dump(O/'cpu_check.json',result);dump(O/'review/cpu_check.json',result)
    print('CPU_CHECK',len(cs),len(videos),len(cs)*10,flush=True)


if __name__=='__main__':main()
