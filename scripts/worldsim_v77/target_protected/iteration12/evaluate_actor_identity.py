"""多B条件单独核对每个actor_slot，防止车辆联合区域掩盖身份串换。"""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
def read(p):return json.loads(p.read_text())

def main(run,quality_run):
    O=T/run;out=[]
    for cid in read(O/'run.json')['probe_cases']:
        c=next(v for v in read(T/'r28/prepared.json')['cases'] if v['case_id']==cid)
        q=next(v for v in read(T/quality_run/'instance_quality.json')['cases'] if v['case_id']==cid)['quality']
        tokens=sorted(q['protected_instances']);rows={t:[] for t in tokens}
        for i in range(30):
            state=np.load(O/'state'/cid/f'{i:05}.npz');h=np.asarray(Image.open(O/'observed'/cid/'H'/f'{i:05}.png'))>0
            truth={t:np.asarray(Image.open(T/'r28/quality_labels/observed_masks'/next(a for a in c['retained_instances'] if a['instance_token']==t)['Y_quality_job']/f'{i:05}.png'))>0 for t in tokens}
            for slot,t in enumerate(tokens,1):
                pred=(state['actor_slot']==slot)&h;own=truth[t]&h
                other=np.logical_or.reduce([truth[x] for x in tokens if x!=t]) if len(tokens)>1 else np.zeros_like(h)
                obs=np.asarray(Image.open(O/'observed_masks'/(cid+'_'+t[:8])/f'{i:05}.png'))>0
                rows[t].append({'frame':i,'predicted_inside_H':int(pred.sum()),'correct_identity_inside_H':int((pred&own).sum()),
                    'hidden_reference_B':int(own.sum()),'predicted_over_other_B':int((pred&other&~truth[t]).sum()),
                    'visible_observation':int(obs.sum()),'visible_correct_identity':int((obs&truth[t]&~h).sum()),
                    'visible_wrong_B':int((obs&other&~truth[t]).sum())})
        actors=[]
        for slot,t in enumerate(tokens,1):
            s=lambda k:sum(x[k] for x in rows[t])
            actors.append({'instance':t,'slot':slot,'primary_reveal':t in q.get('reveal_instances',tokens),
                'O_H_identity_precision':s('correct_identity_inside_H')/max(1,s('predicted_inside_H')),
                'hidden_B_coverage':s('correct_identity_inside_H')/max(1,s('hidden_reference_B')),
                'projected_onto_other_B_pixels':s('predicted_over_other_B'),
                'visible_identity_precision':s('visible_correct_identity')/max(1,s('visible_observation')),
                'visible_wrong_B_pixels':s('visible_wrong_B'),'frames':rows[t]})
        out.append({'case_id':cid,'actors':actors})
    dest=O/'actor_identity_evaluation.json';assert not dest.exists()
    dest.write_text(json.dumps({'cases':out,'reference':'Y-SAM evaluation only, not manual pixel GT','human_verdict':None},ensure_ascii=False,indent=2)+'\n')
    print([{**r,'actors':[{k:v for k,v in a.items() if k!='frames'} for a in r['actors']]} for r in out],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--quality-run',required=True);a=p.parse_args();main(a.run,a.quality_run)
