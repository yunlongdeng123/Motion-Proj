"""仅按输入非空mask扩展窗口，保留旧空窗；不得按生成效果选帧。"""
from pathlib import Path
import argparse,json,copy,shutil
def main(root):
    path=root/'evaluation_plan.json';plan=json.loads(path.read_text());audit=json.loads((root/'evaluation_input_audit.json').read_text())
    if any(c.get('input_window_correction') for c in plan['cases']):print('ALREADY_FIXED');return
    shutil.copy2(path,root/'evaluation_plan_pre_input_fix.json');byid={c['eval_id']:c for c in plan['cases']};extra=[]
    for row in audit['cases']:
        if row['current_active_frames']==10:continue
        c=byid[row['eval_id']];c['input_not_exercised']=True
        start=row['earliest_fully_active_window_start']
        if start is None:continue
        new=copy.deepcopy(c);new.update(eval_id=c['eval_id']+f'_w{start:02}',frames=list(range(start,start+10)),input_not_exercised=False,input_window_correction='earliest fully nonempty input-only window; old empty window retained',evaluation_parent=c['eval_id']);extra.append(new)
    plan['cases']+=extra;plan['input_correction']='2026-10-01: original A041 window0 empty; earliest fully active window10, same weights/masks/seed/steps for both arms; no output-based selection'
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print('ADDED',[(c['eval_id'],c['frames']) for c in extra])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
