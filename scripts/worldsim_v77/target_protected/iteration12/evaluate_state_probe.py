"""独立质量评价可读Y；方法构建程序不可导入本文件或读取Y。"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0]=[str(P),str(P/'iteration11')]
from long_factory import geometry,protections
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r22'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(patches=False):
    g=geometry();rows=[]
    for item in read(O/('state_patches_summary.json' if patches else 'state_summary.json'))['cases']:
        cid=item['case_id'];sid=item['source_id'];dest=O/('state_patches' if patches else 'state')/cid
        gt_masks=protections(g,sid);tokens=sorted(item['actors']);stats=[]
        ys=sorted((T/'r16/synthetic'/cid/'Y').glob('*.png'))
        for i,yp in enumerate(ys):
            y=np.asarray(Image.open(yp).convert('RGB'));h=np.asarray(Image.open(O/'observed'/cid/'H'/f'{i:05}.png'))>0
            r=dict(np.load(dest/f'{i:05}.npz'));lo=dict(np.load(dest/f'{i:05}_leave_self.npz'))
            union=np.zeros_like(h)
            for ms in gt_masks.values():union|=ms[i]
            hidden_b=h&union;op=h&r['O'];npix=h&r['N']
            correct=np.zeros_like(h);visible_correct=np.zeros_like(h)
            for slot,tok in enumerate(tokens,1):
                if tok not in gt_masks:continue
                correct|=(r['actor_slot']==slot)&gt_masks[tok][i]&r['O']
                visible_correct|=(lo['actor_slot']==slot)&gt_masks[tok][i]&lo['O']&~h
            valid=op&correct
            e=abs(r['F'].astype(float)-y).mean(-1)/255
            le=abs(lo['F'].astype(float)-y).mean(-1)/255
            stats.append({'frame':i,'H_pixels':int(h.sum()),'hidden_B_pixels':int(hidden_b.sum()),'O_in_H_pixels':int(op.sum()),'O_correct_identity_pixels':int((correct&h).sum()),'O_identity_precision':float((correct&h).sum()/op.sum()) if op.any() else None,'B_reveal_coverage':float((correct&h).sum()/hidden_b.sum()) if hidden_b.any() else None,'O_RGB_MAE':float(e[valid].mean()) if valid.any() else None,'N_in_H_pixels':int(npix.sum()),'N_over_real_B_pixels':int((npix&union).sum()),'N_over_known_B_rate':float((npix&union).sum()/npix.sum()) if npix.any() else None,'N_RGB_MAE':float(e[npix].mean()) if npix.any() else None,'visible_leave_self_B_pixels':int(visible_correct.sum()),'visible_leave_self_B_RGB_MAE':float(le[visible_correct].mean()) if visible_correct.any() else None})
        sums={k:sum(s[k] for s in stats) for k in ['H_pixels','hidden_B_pixels','O_in_H_pixels','O_correct_identity_pixels','N_in_H_pixels','N_over_real_B_pixels']}
        sums['O_identity_precision']=sums['O_correct_identity_pixels']/max(1,sums['O_in_H_pixels'])
        sums['B_reveal_coverage']=sums['O_correct_identity_pixels']/max(1,sums['hidden_B_pixels'])
        sums['known_background_H_fraction']=sums['N_in_H_pixels']/max(1,sums['H_pixels'])
        row={'case_id':cid,'source_id':sid,'scene':item['scene'],'summary':sums,'per_frame':stats,'evaluation_only':'Y and complete-video SAM2 labels used here ONLY; labels are not segmentation GT','human_verdict':None}
        dump(dest/'quality_metrics.json',row);rows.append(row)
        print('STATE_QA',cid,sums,flush=True)
    dump(O/('condition_patches_quality.json' if patches else 'condition_quality.json'),{'cases':rows,'no_training':True,'condition_method_never_reads_this_file':True,'human_verdict':None})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ground-patches',action='store_true');main(p.parse_args().ground_patches)
