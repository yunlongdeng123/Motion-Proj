"""合成按case/scene等权报告；真实DELETE不虚构恢复GT。"""
from pathlib import Path
import sys
from collections import defaultdict,Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read,dump

def mean(v):return float(np.mean(v)) if v else None

def main():
    plan=read(O/'evaluation_plan.json');rows=[];summary={}
    for c in plan['cases']:
        if c['kind']!='synthetic':continue
        r={'eval_id':c['eval_id'],'scene':c['receiver_scene'],'type':c['type'],'arms':{}}
        for arm in plan['arms']:
            scores=read(O/'evaluation'/c['eval_id']/f'{arm}_metrics.json')['scores'];r['arms'][arm]={}
            for key in ['hole','protected_inside_hole','native_outside_hole']:
                vv=[s[key] for s in scores if s[key] is not None]
                r['arms'][arm][key]={'MAE':mean([v['MAE'] for v in vv]),'PSNR':mean([v['PSNR'] for v in vv]),'frames':len(vv),'pixels_per_frame':mean([v['pixels'] for v in vv])}
        rows.append(r)
    for key in ['hole','protected_inside_hole']:
        usable=[r for r in rows if key!='protected_inside_hole' or r['type']!='background']
        result={'cases':len(usable),'scenes':len({r['scene'] for r in usable}),'macro_case':{},'macro_scene':{},'types':{}}
        for arm in plan['arms']:
            vals=[r['arms'][arm][key]['MAE'] for r in usable if r['arms'][arm][key]['MAE'] is not None];by=defaultdict(list)
            for r in usable:
                if r['arms'][arm][key]['MAE'] is not None:by[r['scene']].append(r['arms'][arm][key]['MAE'])
            result['macro_case'][arm]=mean(vals);result['macro_scene'][arm]=mean([mean(v) for v in by.values()])
        for kind in sorted({r['type'] for r in usable}):
            rr=[r for r in usable if r['type']==kind];result['types'][kind]={'cases':len(rr),'scenes':len({r['scene'] for r in rr}),'macro_case':{arm:mean([r['arms'][arm][key]['MAE'] for r in rr if r['arms'][arm][key]['MAE'] is not None]) for arm in plan['arms']}}
        for basis in ['base','r7']:
            valid=[r for r in usable if r['arms'][basis][key]['MAE'] is not None]
            result['new_data_vs_'+basis]={'better_cases':sum(r['arms']['new_data'][key]['MAE']<r['arms'][basis][key]['MAE'] for r in valid),'total_cases':len(valid),'macro_scene_relative_change':result['macro_scene']['new_data']/result['macro_scene'][basis]-1 if result['macro_scene'][basis] else None}
        summary[key]=result
    catalog=read(O/'dataset_catalog.json');state=read(O/'training/state.json')
    result={'task':'WS-V77-TARGET-PROTECTED-20260929','run':'r8','data':catalog['summary'],'training':state,'synthetic_metrics':summary,'synthetic_cases':rows,'real_case_count':sum(c['kind']=='real_development' for c in plan['cases']),'real_actor_free_GT':None,'real_scientific_success':'requires preserved hidden/neighbor identities in independent development worlds; inspect video; no RGB-MAE claim without GT','technical_quality_not_output_quality':True,'same_modules_budget':read(O/'training/same_recipe_as_r7.json'),'human_verdict':None,'final_used':False}
    done={(r['eval_id'],r['arm']) for r in read(O/'evaluation/state.json')['completed']}
    result['all_inference_complete']={(c['eval_id'],a) for c in plan['cases'] for a in plan['arms']}<=done
    result['synthetic_inference_complete']={(c['eval_id'],a) for c in plan['cases'] if c['kind']=='synthetic' for a in plan['arms']}<=done
    assert result['synthetic_inference_complete']
    dump(O/'results_summary.json',result);print(summary,flush=True)
if __name__=='__main__':main()
