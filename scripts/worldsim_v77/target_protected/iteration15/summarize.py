"""只汇总已生成的固定对照；不新增推理、训练或人工分。"""
from common import *
import numpy as np
from PIL import Image


def frames(folder,native=False):
    p=folder/'native' if native else folder
    return np.stack([np.asarray(Image.open(p/f'{f:05}.png').convert('RGB')) for f in range(10)])


def hole_error(a,b,h):
    return float(np.abs(a.astype(float)-b).mean(-1)[h].mean()/255)


def main():
    byid=cases();diagnostic=[];evaluation=[]
    if (O/'diagnostics/state.json').exists():
        for cid in DIAGNOSTIC_IDS:
            h=images(byid[cid],'hole')>0;root=O/'diagnostics'/cid
            correct=frames(root/'RGB_and_geometry',True)
            rows=[]
            for label in read(O/'manifest.json')['diagnostics']:
                result=read(root/label/'result.json')
                assert result['C_parameters']==read(root/'RGB_and_geometry/result.json')['C_parameters']
                assert result['UC_parameters']==[0.,0.,0.,0.]
                rows.append({'condition':label,'C_parameters':result['C_parameters'],
                    'native_difference_to_correct_H':hole_error(frames(root/label,True),correct,h),
                    'module_response':result['module_response']})
            diagnostic.append({'case_id':cid,'correct_wrong_none_same_instruction':True,
                'encoding':read(root/'reference_encoding/result.json'),'conditions':rows,
                'scope':'像素差是敏感性，不是成功率；全图RMS受mask面积影响，不据小幅响应断言忽略RGB'})
    for cid in EVAL_128:
        c=byid[cid];req=request(c)
        gt=images(c,'target') if c['kind']=='synthetic' else None
        sources={'r46':PARENT/'evaluation'/cid/'baseline','r47':PARENT/'evaluation'/cid/'RGB_and_geometry'}
        for step in (64,128):
            p=O/'evaluation'/f'step_{step:04}'/cid
            if (p/'result.json').exists():sources[f'r48_{step}']=p
        rows=[];reference=frames(sources['r47'],True)
        for label,folder in sources.items():
            native=frames(folder,True);final=frames(folder)
            assert np.array_equal(final,req.compose(native))
            rows.append({'version':label,'native_difference_to_r47_H':hole_error(native,reference,req.edit_mask),
                'GT_H_MAE':hole_error(final,gt,req.edit_mask) if gt is not None else None,
                'outside_H_exact':True,'native_final_composition_exact':True})
        evaluation.append({'case_id':cid,'kind':c['kind'],'scene':c['scene'],'versions':rows,
            'human_verdict':None,'temporal_verdict':None})
    training=read(O/'training/state.json') if (O/'training/state.json').exists() else None
    result={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r48',
        'diagnostic':diagnostic,'evaluation':evaluation,'training':training,
        'scope':'固定真实DEV；无隐藏GT；两个合成GT对照不代表真实删除通过率',
        'human_verdict':None,'temporal_verdict':None,'no_auto_extra_steps':True}
    dump(O/'summary.json',result)
    print('SUMMARY_READY',len(diagnostic),len(evaluation),flush=True)


if __name__=='__main__':main()
