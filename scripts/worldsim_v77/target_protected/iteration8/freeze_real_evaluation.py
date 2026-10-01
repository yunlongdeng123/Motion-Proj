"""真实DELETE开发评测先冻结ID与合法输入窗，禁止看新输出后选择。"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,T,read,dump
AUDIT=T.parent/'WS-V77-DELETE-AUDIT-20260928/r1'
def main():
    if (O/'real_evaluation_plan.json').exists():print('real eval frozen');return
    clips={c['clip_id']:c for c in read(AUDIT/'selection.json')['clips']};cases=[]
    # 四个人工点名failure；四个已曝光后车/邻车案例辅助检查回归，不作为final。
    for cid in ['A022','A041','A013','A048','A007','A034','A042','A061']:
        c=clips[cid];folder=AUDIT/'clips'/cid;masks=sorted((folder/'model_mask').glob('*.png'));core=sorted((folder/'core').glob('*.png'))
        active=[bool(np.asarray(Image.open(p)).any()) for p in masks]
        starts=[i for i in range(len(active)-9) if all(active[i:i+10])]
        start=starts[0] if starts else None
        row=dict(c,eval_id=cid if start==0 else cid+f'_w{start:02}' if start is not None else cid+'_invalid',kind='real_development',folder=str(folder),frames=[] if start is None else list(range(start,start+10)),GT_available=False,seed=42,steps=25,human_verdict=None,
                 input_window_rule='earliest consecutive10 model masks nonempty; no new outputs used',input_valid=start is not None,original_empty_frames=[i for i,a in enumerate(active[:10]) if not a],role='named_failure' if cid in ['A022','A041','A013','A048'] else 'exposed_protected_neighbor_control')
        row['core_nonempty_in_selected_window']=None if start is None else [bool(np.asarray(Image.open(core[i])).any()) for i in row['frames']]
        cases.append(row)
    dump(O/'real_evaluation_plan.json',{'cases':cases,'arms':['base','r7','new_data'],'provenance':'existing70audit, exposed development only; final untouched; no true actor-free GT so no synthetic-style recovery MAE','scene_count':len({c['scene'] for c in cases}),'seed':42,'steps':25,'architecture_unchanged':True})
    print([(c['eval_id'],c['scene'],c['input_valid']) for c in cases],flush=True)
if __name__=='__main__':main()
