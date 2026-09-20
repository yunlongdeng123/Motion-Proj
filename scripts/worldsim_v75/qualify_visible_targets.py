"""只用真实视频完成冻结观测准入，再锁定最多两个模型案例。"""
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from prepare_visible_sources import OUT
from evaluate_localization import model,predict,match
from audit_natural_observation import advance,RAFTFlow

def main():
    path=OUT/'observation_selection.json';assert not path.exists()
    index=json.loads((OUT/'candidate_index.json').read_text());review=json.loads((OUT/'visual_review.json').read_text())
    detector=model();raft=RAFTFlow();rows=[];chosen=[];start=time.monotonic()
    for log in index['logs']:
        selected=None;records=[]
        for candidate in log['candidates']:
            target=candidate['track'];visible=review['candidates'][target]
            record={'target':target,'visible':visible['visible'],'reason':visible['reason']};records.append(record)
            if not visible['visible']:continue
            folder=OUT/'real'/log['log_id'];out=folder/target;out.mkdir()
            frames=[np.array(Image.open(folder/f'reference-{f:03d}.png')) for f in [0,15,30,45,60]]
            detections=[match(predict(detector,image),bounds) for image,bounds in zip(frames,candidate['all_boxes'])]
            record['detections']=detections
            if any(d is None for d in detections):record['measurement_passed']=False;record['reason']='fixed detector not matched at every real time';continue
            box=np.array(detections[0]['box']);c=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*.3
            mask=np.zeros(frames[0].shape[:2],np.uint8);low=np.ceil(c-half).astype(int);high=np.floor(c+half).astype(int)
            mask[low[1]:high[1]+1,low[0]:high[0]+1]=255
            p=cv2.goodFeaturesToTrack(cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY),maxCorners=64,qualityLevel=.01,minDistance=3,mask=mask)
            if p is None or len(p)<8:record.update(measurement_passed=False,reason='insufficient initial texture points');continue
            points=p[:,0,:];cal=[]
            for dx in [-4,4]:
                shifted=np.zeros_like(frames[0])
                if dx>0:shifted[:,dx:]=frames[0][:,:-dx]
                else:shifted[:,:dx]=frames[0][:,-dx:]
                det=match(predict(detector,shifted),(box+[dx,0,dx,0]).tolist())
                derr=None if det is None else det['center'][0]-detections[0]['center'][0]-dx
                q,ok,fb=advance(raft,frames[0],shifted,points,np.ones(len(points),bool),out/f'calibration-{dx}.npz')
                cal.append({'dx':dx,'detector_residual_px':derr,'flow_median_error_px':float(np.median(np.linalg.norm(q-points-[dx,0],axis=1))),
                            'flow_retained':int(ok.sum()),'total':len(points)})
            q=points.copy();valid=np.ones(len(points),bool);positions=[q.copy()];masks=[valid.copy()]
            for i,f in enumerate([15,30,45,60],1):
                q,valid,fb=advance(raft,frames[i-1],frames[i],q,valid,out/f'step-{f:03d}.npz')
                positions.append(q.copy());masks.append(valid.copy())
            np.savez(out/'tracks.npz',points=np.array(positions),valid=np.array(masks))
            support=[int(x.sum()) for x in masks]
            passed=all(r['detector_residual_px'] is not None and abs(r['detector_residual_px'])<=2 and r['flow_median_error_px']<=2
                       and r['flow_retained']/r['total']>=.8 for r in cal) and all(n>=8 and n/len(points)>=.25 for n in support)
            record.update(measurement_passed=passed,calibration=cal,initial_points=points.tolist(),support=support,initial_count=len(points))
            sheet=Image.new('RGB',(1600,250),'#101b2b');draw=ImageDraw.Draw(sheet)
            for i,image in enumerate(frames):
                im=Image.fromarray(image);d=ImageDraw.Draw(im)
                for x,y in positions[i][masks[i]]:d.ellipse((x-2,y-2,x+2,y+2),fill='#00e5b5')
                cx,cy=detections[i]['center'];crop=[round(cx-180),round(cy-108),round(cx+180),round(cy+108)]
                draw.text((i*320+5,7),f'{i*.5:.1f}s | {support[i]}/{len(points)} points',fill='white')
                sheet.paste(im.crop(crop).resize((320,192)),(i*320,30))
            sheet.save(out/'real-tracking.jpg',quality=95)
            if passed:selected=target;break
        rows.append({'log_id':log['log_id'],'selected_target':selected,'candidate_measurements':records})
        if selected and len(chosen)<2:chosen.append({'log_id':log['log_id'],'target':selected})
        print(json.dumps({'log_id':log['log_id'],'selected':selected,'measurements':[{k:v for k,v in r.items() if k in ['target','measurement_passed','support','reason']} for r in records]}),flush=True)
    result={'status':'complete','locked_before_reconstruction_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'logs':rows,'model_cases':chosen,'human_verdict':None,'failure_ledger_delta':'none',
            'wall_s':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reconstruction_calls':0,'generation_calls':0}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
