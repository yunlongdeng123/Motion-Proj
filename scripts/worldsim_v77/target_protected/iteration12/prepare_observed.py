"""只读取X、H和几何元数据；没有Y读取入口。"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image

T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r22'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    plan=read(O/'run.json');sources={c['source_id']:c for c in read(T/'r16/factory/source_manifest.json')['clips']};jobs=[]
    for cid in plan['probe_cases']:
        original=T/'r16/synthetic'/cid;row=read(original/'pair_manifest.json');c=sources[row['source_id']]
        out=O/'observed'/cid;out.mkdir(parents=True,exist_ok=True)
        for role in ['rgb','H','sam_rgb']:(out/role).mkdir(exist_ok=True)
        holes=[]
        xs=sorted((original/'X').glob('*.png'));hs=sorted((original/'model_hole').glob('*.png'))
        assert len(xs)==len(hs)==len(c['frames'])==30
        for i,(xp,hp) in enumerate(zip(xs,hs)):
            x=np.asarray(Image.open(xp).convert('RGB')).copy();h=np.asarray(Image.open(hp))>0
            x[h]=127;holes.append(h)
            Image.fromarray(x).save(out/'rgb'/f'{i:05}.png');Image.fromarray(h.astype('uint8')*255).save(out/'H'/f'{i:05}.png')
            # 官方SAM目录加载器只接受jpg；在擦除后编码，不存在隐藏RGB回流。
            Image.fromarray(x).save(out/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
        meta={'case_id':cid,'source_id':c['source_id'],'scene':c['scene'],'frames':c['frames'],'actors':c['actors'],'geometry_source':'GT_camera_and_tracks_POC_auxiliary','rgb_source':'X_erased_by_H_before_serialization','gt_read_for_state':False,'window_frames':30,'window_start':c['frames'][0]['timestamp'],'window_end':c['frames'][-1]['timestamp'],'asset_role':'existing_DEV_probe_not_new_training_data'}
        dump(out/'observations.json',meta)
        for actor in c['actors']:
            tok=actor['instance_token'];areas=[];boxes=[]
            for fr,h in zip(c['frames'],holes):
                a=next(a for a in fr['actors'] if a['instance_token']==tok);bb=np.array(a['projection']['box_xyxy']);x0,y0,x1,y1=np.rint(bb).astype(int);x0,x1=np.clip([x0,x1],0,1024);y0,y1=np.clip([y0,y1],0,576)
                fraction=1-float(h[y0:y1,x0:x1].mean()) if x1>x0 and y1>y0 else 0
                areas.append(fraction*(x1-x0)*(y1-y0));boxes.append(bb.tolist())
            prompt=int(np.argmax(areas))
            if areas[prompt]<100:continue
            jobs.append({'case_id':cid,'instance_token':tok,'job_id':cid+'_'+tok[:8],'prompt_frame':prompt,'prompt_box':boxes[prompt],'selection':'maximum_unmasked_GT_bbox_area_in_fixed_window','input_folder':str(out),'frames':30,'no_Y_or_full_RGB_access':True})
    dump(O/'observed_queue.json',{'jobs':jobs,'input_boundary':'masked_fixed_window_only','frames':30,'human_verdict':None})
    print('OBSERVED_QUEUE',len(jobs))

if __name__=='__main__':main()
