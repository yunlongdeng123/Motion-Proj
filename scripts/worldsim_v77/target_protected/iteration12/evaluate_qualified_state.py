"""r29后验评价：真实Y/全Y-SAM仅在这里读取，不给方法状态回流。"""
from pathlib import Path
import json,sys
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).parent))
from prepare_instance_audit import f
O=f.T/'r29';R=f.T/'r28'


def main(run_root=O,quality_root=R):
    O=run_root
    assert f.read(O/'controller_state.json')['stage'] in {'complete_pending_condition_quality','complete_pending_paired_evaluation'}
    summary=[]
    for cid in f.read(O/'run.json')['probe_cases']:
        c=next(v for v in f.read(R/'prepared.json')['cases'] if v['case_id']==cid)
        q=next(v for v in f.read(quality_root/'instance_quality.json')['cases'] if v['case_id']==cid)
        tokens=q['quality']['protected_instances'];rows=[];strips=[]
        for i in range(30):
            state=dict(np.load(O/'state'/cid/f'{i:05}.npz'));loo=dict(np.load(O/'state'/cid/f'{i:05}_leave_self.npz'))
            x=np.asarray(Image.open(O/'observed'/cid/'rgb'/f'{i:05}.png'));h=np.asarray(Image.open(O/'observed'/cid/'H'/f'{i:05}.png'))>0
            y=np.asarray(Image.open(Path(c['source_Y_quality_only'])/f'{i:05}.png'));truth=np.zeros_like(h);observed=np.zeros_like(h);boxes=[]
            for tok in tokens:
                actor=next(a for a in c['retained_instances'] if a['instance_token']==tok)
                truth|=np.asarray(Image.open(R/'quality_labels/observed_masks'/actor['Y_quality_job']/f'{i:05}.png'))>0
                observed|=np.asarray(Image.open(O/'observed_masks'/(cid+'_'+tok[:8])/f'{i:05}.png'))>0
                if actor['boxes'][i] is not None:boxes.append(actor['boxes'][i])
            target=truth&h;pred=state['O']&h;correct=pred&truth;visible_truth=truth&~h;visible_correct=observed&visible_truth
            metric={'frame':i,'hole_pixels':int(h.sum()),'hidden_B_pixels':int(target.sum()),'projected_O_H_pixels':int(pred.sum()),
                    'correct_O_H_pixels':int(correct.sum()),'N_H_pixels':int((state['N']&h).sum()),
                    'N_over_Y_SAM_B_pixels':int((state['N']&target).sum()),'U_H_pixels':int((state['U']&h).sum()),
                    'observed_SAM_pixels':int(observed.sum()),'correct_observed_SAM_pixels':int(visible_correct.sum()),
                    'visible_B_pixels':int(visible_truth.sum()),'projected_RGB_MAE_correct_O':float(abs(state['F'].astype('float32')-y)[correct].mean()) if correct.any() else None,
                    'leave_self_O_in_H':int((loo['O']&h).sum()),'observed_SAM_inside_H':int((observed&h).sum())}
            assert not metric['observed_SAM_inside_H'] and np.all(state['U']==~(state['O']|state['N']))
            assert not (state['O']&state['N']).any()
            rows.append(metric)
            bb=np.array(boxes);x0,y0=np.floor(bb[:,:2].min(0)-18).astype(int);x1,y1=np.ceil(bb[:,2:].max(0)+18).astype(int)
            x0,y0=max(0,x0),max(0,y0);x1,y1=min(1024,x1),min(576,y1)
            overlay=x.copy();cv2.drawContours(overlay,cv2.findContours(observed.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(40,255,100),1)
            condition=state['F'].copy();condition[state['U']]=48
            strip=Image.new('RGB',(540,206),(18,23,30));draw=ImageDraw.Draw(strip)
            for j,(name,im) in enumerate([('Y QA only',y),('legal X + visible SAM',overlay),('projected F / U gray',condition)]):
                draw.text((j*180+3,3),f'f{i} '+name,fill='white')
                crop=ImageOps.contain(Image.fromarray(im[y0:y1,x0:x1]),(180,180))
                strip.paste(crop,(j*180+(180-crop.width)//2,24+(180-crop.height)//2))
            strips.append(strip)
        for start in [0,15]:
            sheet=Image.new('RGB',(2700,618),(18,23,30))
            for j,strip in enumerate(strips[start:start+15]):sheet.paste(strip,((j%5)*540,(j//5)*206))
            sheet.save(O/f'{cid}_identity_{start:02}_{start+14:02}.jpg',quality=96)
        sumkey=lambda key:sum(r[key] for r in rows)
        row={'case_id':cid,'scene':c['scene'],'frames':30,'rendered_O_precision_inside_H':sumkey('correct_O_H_pixels')/max(1,sumkey('projected_O_H_pixels')),
             'hidden_B_coverage_by_O':sumkey('correct_O_H_pixels')/max(1,sumkey('hidden_B_pixels')),
             'N_H_fraction':sumkey('N_H_pixels')/sumkey('hole_pixels'),'unknown_H_fraction':sumkey('U_H_pixels')/sumkey('hole_pixels'),
             'N_over_B_pixels':sumkey('N_over_Y_SAM_B_pixels'),
             'observed_SAM_precision':sumkey('correct_observed_SAM_pixels')/max(1,sumkey('observed_SAM_pixels')),
             'observed_SAM_recall':sumkey('correct_observed_SAM_pixels')/max(1,sumkey('visible_B_pixels')),
             'frame_metrics':rows,'reference':'independently sampled full-Y SAM; not pixel-perfect manual GT',
             'training_steps':0,'human_verdict':None,'condition_quality':'pending_all_frame_identity_review'}
        summary.append(row)
    f.dump(O/'condition_evaluation.json',{'cases':summary,'quality_Y_is_separate':True,'surfel':False,'new_training_steps':0})
    print([{k:v for k,v in r.items() if k!='frame_metrics'} for r in summary],flush=True)

if __name__=='__main__':main()
