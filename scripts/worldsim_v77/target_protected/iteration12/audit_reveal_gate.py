"""r31补充：分清主要显露B与轻微触及的B，不改变任何准入结果。"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parents[1]/'iteration9'))
from temporal_metrics import process
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
def read(p):return json.loads(p.read_text())
def load(folder):return [np.asarray(Image.open(folder/f'{i:05}.png'))>0 for i in range(30)]

def main():
    eligible={'insufficient_actual_occlusion','no_reveal_process','insufficient_other_frame_evidence'}
    results={c['case_id']:c for c in read(T/'r28/instance_quality.json')['cases']}
    data=[]
    for c in read(T/'r28/prepared.json')['cases']:
        result=results[c['case_id']]
        if result['reason'] not in eligible:continue
        h=load(Path(c['observed_folder'])/'proposal_H');pm={};poses={}
        for actor in c['retained_instances']:
            tok=actor['instance_token']
            if not actor['category'].startswith('vehicle.'):continue
            if not all(a is not None and not a.get('interpolation_uncertain',False) for a in actor['annotations']):continue
            path=T/'r28/quality_labels/observed_masks'/actor['Y_quality_job']
            if not (path/'00000.png').exists():continue
            mm=load(path)
            if max((hh&m).sum()/max(1,m.sum()) for hh,m in zip(h,mm))<=.01:continue
            pm[tok]=mm;poses[tok]=actor['annotations']
        proc=process(c['frames'],[p['actor'] for p in c['trajectory']['frames']],h,pm,poses)
        tracks=[]
        for tok,p in proc['protected'].items():
            v=np.array(p['occlusion_fractions']);amount=bool(v.max()>=.30 and sum(v>.05)>=3)
            change=bool(p['visibility_transition'] or p['sweep_over_B']);support=(p['approx_other_frame_support_mean'] or 0)>=.5
            tracks.append({'instance':tok,'max_occlusion':float(v.max()),'amount_pass':amount,'change_pass':change,
                'other_frame_support':p['approx_other_frame_support_mean'],'support_pass':support,
                'all_reveal_gates':amount and change and support})
        data.append({'case_id':c['case_id'],'source_id':c['source_id'],'reason':result['reason'],
            'any_B_passes_reveal':any(v['all_reveal_gates'] for v in tracks),
            'all_affected_B_have_support':all(v['support_pass'] for v in tracks),
            'actors':tracks,'process':proc,'training_admission':False})
        print(c['case_id'],data[-1]['any_B_passes_reveal'],data[-1]['all_affected_B_have_support'],
              [(v['instance'][:8],round(v['max_occlusion'],3),v['all_reveal_gates']) for v in tracks],flush=True)
    path=T/'r31/gate_detail.json';assert not path.exists()
    path.write_text(json.dumps({'cases':data,'boundary':'diagnostic only; unchanged official quality decisions'},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
