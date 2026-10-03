"""逐例真实数组检查；不把CPU零等价/梯度当补景收益。"""
from common import *
import numpy as np, torch, cv2
from interface import DeletionRequest, check_request, drop_priors
from multi_prior import checks


def main():
    torch.set_num_threads(1);plan=read(O/'manifest.json');rows=[]
    assert read(O/'prepare_state.json')['stage']=='CPU_inputs_ready_GPU_not_run'
    for c in plan['cases']:
        folder=O/'inputs'/c['case_id'];p=dict(np.load(folder/'condition.npz'))
        x=images(c,'rgb');h=images(c,'hole')>0
        alpha=images(c,'alpha').astype('float32')/255 if c['kind']=='real' else h.astype('float32')
        req=DeletionRequest(x,h,alpha,p,read(folder/'references.json'))
        audit=check_request(req)
        expected=np.stack([cv2.resize(m.astype('float32'),(256,144),interpolation=cv2.INTER_AREA)>0 for m in h])
        assert np.array_equal(expected,p['hole']), '控制mask缩小后漏边界'
        assert p['geometry'].shape==(10,12,144,256) and p['bev'].shape==(10,10,128,128)
        assert p['bev_rays'].shape==(10,4,18,32,2) and p['reference_pose'].shape==(6,13)
        assert np.allclose(p['geometry'][:,:3].sum(1),1)
        # 训练的先验dropout与评测消融共用未知码；不能连编辑范围一起清掉。
        condition=req.branch_condition(torch.ones(6,4,32,32),'cpu')
        unknown=drop_priors(condition,True,True)
        assert torch.equal(unknown['hole'],condition['hole'])
        assert torch.equal(unknown['query_time'],condition['query_time'])
        assert not unknown['reference_valid'].any() and not unknown['reference_latents'].any()
        assert torch.all(unknown['geometry'][:,2]==1) and torch.all(unknown['bev'][:,6]==1)
        assert torch.all(condition['reference_latents']==1), 'dropout原地污染完整条件'
        rows.append({'case_id':c['case_id'],'split':c['split'],'frames':len(x),**audit,
            'control_mask_covers_every_full_resolution_H_block':True,'geometry_and_BEV_shapes_valid':True,
            'prior_dropout_keeps_edit_mask_and_query_time':True,'dropout_does_not_mutate_full_conditions':True})
    contract=checks()
    result={'task_id':plan['task_id'],'run_id':'r47','CPU_ready':True,'cases':rows,'architecture_contract':contract,
        'official_full_model_forward_backward_verified':False,'trained_steps':0,'new_inference_windows':0,
        'next_GPU_work':'validate extra-reference source masks; original-model zero-init and backward/VRAM probe; then fixed320step small branch experiment',
        'outside_mask_final_RGB_pixel_exact':True,'base_inputs_and_results_modified':False,'human_verdict':None}
    dump(O/'preflight.json',result);print('CPU_READY',len(rows),sum(r['frames'] for r in rows),'PARAMETERS',contract['parameters'],flush=True)


if __name__=='__main__':main()
