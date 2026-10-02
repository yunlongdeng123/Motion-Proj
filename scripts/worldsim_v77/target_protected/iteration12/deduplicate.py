"""按模型可见任务去重目录；逐像素核实且不改原始样本。"""
from pathlib import Path
import json, sys
import numpy as np
from PIL import Image

T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected/iteration7')
from audit import protected_mask


def read(p):return json.loads(p.read_text())
def load(c,role,i):return np.asarray(Image.open(sorted((Path(c['folder'])/role).glob('*.png'))[i])).copy()
def main():
    catalog=read(T/'r14/dataset_catalog.json');by={c['dataset_id']:c for c in catalog['cases']}
    pairs=read(T/'r19/process_and_duplicates.json')['duplicate_tasks'];drop=set();checked=[]
    for p in pairs:
        a,b=by[p['keep']],by[p['duplicate']]
        assert a['split']==b['split']
        assert a['type']==b['type'] and a['frame_count']==b['frame_count']
        for i in range(a['frame_count']):
            ya,yb=load(a,'Y',i),load(b,'Y',i);ha,hb=load(a,'model_hole',i)>0,load(b,'model_hole',i)>0
            xa,xb=load(a,'X',i),load(b,'X',i);xa[ha]=0;xb[hb]=0
            assert np.array_equal(ya,yb) and np.array_equal(ha,hb) and np.array_equal(xa,xb)
            # 保护监督不同则不能自动合并为同一个训练任务。
            pa=protected_mask(Path(a['folder']),i,ya.shape[:2],a['type'])
            pb=protected_mask(Path(b['folder']),i,yb.shape[:2],b['type'])
            assert np.array_equal(pa,pb),'保护监督不同，不能自动合并'
        drop.add(b['dataset_id']);checked.append({'keep':a['dataset_id'],'duplicate':b['dataset_id'],'frames_checked':a['frame_count'],'supervision_compatible':True})
    out=dict(catalog,cases=[c for c in catalog['cases'] if c['dataset_id'] not in drop])
    out.update(deduplicated_from='r14',old_catalog_untouched=True,duplicate_mapping=checked,role='legacy_development_probe_only',new_training_split_approved=False,reason='旧mask资产跨split共享，不作为新方案独立验证')
    for split,n in [('train',46),('validation',10)]:assert sum(c['split']==split for c in out['cases'])==n
    (T/'r21/deduplicated_catalog.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print({'train':46,'validation':10,'duplicate_pairs_checked':len(pairs),'approved_for_new_formal_training':False})


if __name__=='__main__':main()
