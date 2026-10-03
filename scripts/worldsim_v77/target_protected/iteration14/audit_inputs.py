"""CPU复核实际条件信息，避免把参考数量或GT包络当作有效appearance。"""
from common import *
import numpy as np
import shutil
from datetime import datetime


def main():
    tokens=[]; geometry=[]
    for c in read(O/'manifest.json')['cases']:
        cid=c['case_id'];z=np.load(O/'inputs'/cid/'condition.npz')
        valid=z['reference_valid'].astype('float32');assert valid.shape==(6,256,256)
        # 对应interface的256->32 .999门和分支的32->8 .95门；不是车辆分割面积。
        latent=valid.reshape(6,32,8,32,8).mean((2,4))>.999
        counts=(latent.reshape(6,8,4,8,4).mean((2,4))>.95).sum((1,2))
        refs=read(O/'inputs'/cid/'references.json')['references']
        slots=[{'slot':r['reference_slot'],'role':r['role'],'camera':r['camera'],
                'padding':r['padding'],'valid_attention_tokens':int(counts[r['reference_slot']])} for r in refs]
        tokens.append({'case_id':cid,'kind':c['kind'],'split':c['split'],'slots':slots})
        h=z['hole']>0;g=z['geometry'];frames=[]
        for f in range(10):
            denominator=max(1,h[f].sum())
            frames.append({'frame':f,**{name:float(g[f,index][h[f]].sum()/denominator) for name,index in
                [('O_proxy_H_fraction',0),('N_measured_H_fraction',1),('U_H_fraction',2),('protected_proxy_H_fraction',7)]}})
        geometry.append({'case_id':cid,'kind':c['kind'],'split':c['split'],'frames':frames,
            'mean_H_fraction':{k:float(np.mean([r[k] for r in frames])) for k in list(frames[0])[1:]}})
    results={'reference_token_audit.json':{'latent_size_assumption':[32,32],
        'token_selection_matches_branch':True,'human_verdict':None,'rows':tokens},
        'input_information_audit.json':{'geometry_definition':'O retained GT cuboid envelope, N positive measured LiDAR support, U remaining unknown. O/protected proxy is not silhouette, fractions are control-grid statistics not certified visibility.',
            'rows':geometry,'architecture_limits':{'query_A_RGB_masked_before_encoding':True,'final_writeback_outside_H_exact':True,
                'hard_target_absence_inside_H':False,'hard_protected_preservation_inside_H':False,'training_multiview_references':False}}}
    for name,value in results.items():
        path=O/name
        if path.exists() and read(path)==value:continue
        if path.exists():
            backup=O/'before_changes'/('CPU_information_audit_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
            backup.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup/name)
        dump(path,value)
    print('CPU_INPUT_AUDIT',len(tokens),'cases; no GPU/model inference',flush=True)


if __name__=='__main__':main()
