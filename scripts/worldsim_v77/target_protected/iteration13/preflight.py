"""CPU检查实际数据边界与四通道；不把这些检查当模型收益。"""
from common import *
import numpy as np
import torch
from semantic_adapter import checks,pack_onuq
from train_control import resized

def main():
    torch.set_num_threads(1);plan=read(O/'manifest.json');rows=[]
    assert read(O/'condition_state.json')['stage']=='CPU_conditions_complete'
    for c in plan['cases']:
        rgb=images(c,'rgb');h=images(c,'hole')>0
        if c['kind']=='synthetic':
            y=images(c,'target');assert not np.any((rgb!=y).any(-1)&~h)
            # 一例真实数组验证先遮再缩；改洞内X不改变任何网络图像条件。
            if c['case_id']==plan['cases'][0]['case_id']:
                p=resized(y,rgb,h,plan['train_size']);alt=rgb.copy();alt[h]=255-alt[h]
                q=resized(y,alt,h,plan['train_size']);assert torch.equal(p[1],q[1])
                yy=y.copy();yy[h]=255-yy[h];r=resized(yy,rgb,h,plan['train_size'])
                assert torch.equal(p[1],r[1]) and not torch.equal(p[0],r[0])
        known=0
        for i in range(len(h)):
            state=dict(np.load(O/'conditions'/c['case_id']/f'{i:05}.npz'))
            g=pack_onuq({k:torch.from_numpy(state[k][None]) for k in ['O','N','U','Q']})
            assert g.shape==(1,4,576,1024)
            known+=int(((state['O']|state['N'])&h[i]).sum())
        rows.append({'case_id':c['case_id'],'frames':len(h),'known_in_H_pixels':known,
                     'input_shape':list(rgb.shape),'ON_UQ_valid':True})
    result={'stage':'CPU_ready_GPU_not_run','cases':rows,'CPU_adapter_contracts':checks(),
            'synthetic_X_hidden_RGB_no_leak':True,'Y_target_only':True,'geometry_auxiliary_declared':True,
            'training_steps':0,'GPU_forward_backward_verified_for_this_four_channel_version':False,
            'inference_windows':0,'human_verdict':None}
    dump(O/'preflight.json',result);print('CPU_PREFLIGHT_PASS',len(rows),sum(r['frames'] for r in rows))

if __name__=='__main__':main()
