"""按最新授权冻结AI2分数据；保留原评分，不把其改成人工分。"""
import argparse,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
def main(task,root):
    if (root/'dataset_catalog.json').exists():
        print('CATALOG_FROZEN: no resplit or rescore');return
    old=read(task/'r2/training_admission.json');rows=[];excluded=[]
    factories=[task/'r1'/n for n in ['', 'expanded_factory','native_window_factory']]
    old_cases={}
    for factory in factories:
        for c in read(factory/'synthetic_review/synthetic_manifest.json')['clips']:old_cases[c['case_id']]=(factory,c)
    for r in old['cases']:
        if r['training_admission']!='train_usable_pending_human':excluded.append({'case_id':r['case_id'],'prior_status':r['training_admission']});continue
        assert r['visual_reassessment']['visual_status']=='usable_candidate' and r['engineering_audit']['engineering_pass']
        factory,c=old_cases[r['case_id']]
        rows.append({'dataset_id':'r2/'+r['case_id'],'case_id':r['case_id'],'generation_run':'r1','quality_run':'r2','folder':str(factory/'synthetic'/r['case_id']),
                     'type':c['type'],'receiver_scene':c['scene'],'donor_scene':c['donor_scene'],'frame_count':c.get('frame_count',len(c['frames'])),
                     'assistant_score':2,'quality_basis':'prior independent five-check re-review + all-frame engineering; latest user permits AI admission',
                     'note':r['visual_reassessment']['reason'],'human_verdict':None})
    reviews={r['case_id']:r for r in read(task/'r3/review/independent_synthetic_reviews.json')['clips']}
    validation={r['case_id']:r for r in read(task/'r3/review/delivery_validation.json')['cases']}
    for c in read(task/'r3/review/synthetic_manifest.json')['clips']:
        r=reviews[c['case_id']]
        if r['synthetic_status']!='pass' or not validation[c['case_id']]['pass']:continue
        rows.append({'dataset_id':'r3/'+c['case_id'],'case_id':c['case_id'],'generation_run':'r3','quality_run':'r3','folder':str(task/'r3/factory/synthetic'/c['case_id']),
                     'type':c['type'],'receiver_scene':c['scene'],'donor_scene':c['donor_scene'],'frame_count':c['frame_count'],
                     'assistant_score':2,'quality_basis':'independent five-check QA + all-frame synthetic validation', 'note':r['note'],'human_verdict':None})
    fresh={r['case_id']:r for r in read(root/'new_synthetic_reviews.json')['cases']}
    val={r['case_id']:r for r in read(root/'review/delivery_validation.json')['cases']}
    for c in read(root/'review/synthetic_manifest.json')['clips']:
        r=fresh[c['case_id']]
        if r['assistant_score']!=2 or r['synthetic_status']!='pass' or not val[c['case_id']]['pass']:continue
        rows.append({'dataset_id':'r6/'+c['case_id'],'case_id':c['case_id'],'generation_run':'r6','quality_run':'r6','folder':str(root/'factory/synthetic'/c['case_id']),
                     'type':c['type'],'receiver_scene':c['scene'],'donor_scene':c['donor_scene'],'frame_count':c['frame_count'],
                     'assistant_score':2,'quality_basis':'new exact mask admission + independent QA + all-frame pixel/geometry validation','note':r['note'],'human_verdict':None})
    # 重新解码无损训练对，RGB泄漏和隐藏条件探针逐帧核查；此处不重新给视觉分。
    for r in rows:
        folder=Path(r['folder']);count=0
        for i in range(r['frame_count']):
            load=lambda name:np.asarray(Image.open(folder/name/f'{i:03}.png')).copy()
            y=load('Y');x=load('X');h=load('model_hole')>0
            assert x.shape==y.shape==(576,1024,3) and h.shape==(576,1024)
            assert not ((x!=y).any(-1)&~h).any(),r['dataset_id']
            cx=x.astype('float32')/127.5-1;cy=y.astype('float32')/127.5-1;cx[h]=0;cy[h]=0
            assert np.array_equal(cx,cy);count+=1
        r['decoded_and_coverage_verified_frames']=count
    # receiver与donor scene都入图，整连通分量分割，避免同一scene跨训练验证。
    graph=defaultdict(set)
    for r in rows:a,b=r['receiver_scene'],r['donor_scene'];graph[a].add(b);graph[b].add(a)
    components=[];seen=set()
    for scene in sorted(graph):
        if scene in seen:continue
        todo=[scene];comp=set()
        while todo:
            s=todo.pop()
            if s in comp:continue
            comp.add(s);seen.add(s);todo+=list(graph[s]-comp)
        components.append(comp)
    sizes=[sum(r['receiver_scene'] in c for r in rows) for c in components]
    # 固定选最接近10%且非主体的一个完整分量；只有一个分量时不假装独立验证。
    candidates=[i for i,n in enumerate(sizes) if 0<n<=max(1,len(rows)//4)]
    holdout=min(candidates,key=lambda i:(abs(sizes[i]-max(1,len(rows)//10)),sorted(components[i]))) if len(components)>1 and candidates else None
    held=set() if holdout is None else components[holdout]
    for r in rows:r['split']='validation' if r['receiver_scene'] in held else 'train'
    assert not {s for r in rows if r['split']=='train' for s in [r['receiver_scene'],r['donor_scene']]} & held
    summary={'case_count':len(rows),'type_counts':dict(Counter(r['type'] for r in rows)),'split_counts':dict(Counter(r['split'] for r in rows)),
             'receiver_scene_count':len({r['receiver_scene'] for r in rows}),'scene_component_case_sizes':sizes,
             'all_retained_assistant_scores':2,'human_verdicts_all_null':True,'GPU_training_steps':0,
             'decoded_and_coverage_verified_frames':sum(r['decoded_and_coverage_verified_frames'] for r in rows),
             'classification':'technical AI2 admission per user 2026-10-01; preserves prior score histories; no claim final unseen evaluation'}
    dump(root/'dataset_catalog.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r6','summary':summary,'cases':rows,'excluded_prior_cases':excluded})
    print('CATALOG',summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.task,a.root)
