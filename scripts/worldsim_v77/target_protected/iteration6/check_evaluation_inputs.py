"""固定对照窗口必须真正包含删除洞；零洞帧不按成功DELETE计分。"""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
def main(root):
    plan=json.loads((root/'evaluation_plan.json').read_text());rows=[]
    for c in plan['cases']:
        masks=sorted((Path(c['folder'])/('model_hole' if c['kind']=='synthetic' else 'model_mask')).glob('*.png'))
        counts=[int((np.asarray(Image.open(p))>0).sum()) for p in masks]
        candidates=[i for i in range(len(counts)-9) if all(v>0 for v in counts[i:i+10])]
        rows.append({'eval_id':c['eval_id'],'current_frames':c['frames'],'mask_pixels_current':[counts[i] for i in c['frames']],'current_active_frames':sum(counts[i]>0 for i in c['frames']),'review_frame_active':counts[c['frames'][5]]>0,'earliest_fully_active_window_start':candidates[0] if candidates else None,'selector':'earliest 10 consecutive nonempty masks; input-only, no generated-output inspection'})
    (root/'evaluation_input_audit.json').write_text(json.dumps({'cases':rows},ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
