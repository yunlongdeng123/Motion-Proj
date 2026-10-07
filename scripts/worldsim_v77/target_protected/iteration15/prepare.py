"""r48 CPU准备：固定已有数据、人工记录和输入合同，不加载官方GPU模型。"""
from common import *
import gc, time
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from interface import check_request, drop_priors
from multi_prior import MultiPriorBranch


def main():
    started=time.monotonic();torch.set_num_threads(1)
    parent=read(PARENT/'manifest.json');wanted=TRAIN_IDS+EVAL_128
    selected={c['case_id']:c for c in parent['cases']}
    assert set(wanted)<=set(selected)
    assert all(selected[i]['split']=='train' for i in TRAIN_IDS)
    assert all(selected[i]['split']!='train' for i in EVAL_128)
    assert {selected[i]['scene'] for i in TRAIN_IDS}.isdisjoint(
        {selected[i]['scene'] for i in EVAL_128}), 'BUILD/QUERY scene重叠'
    source_checks=read(PARENT/'source_mask_validation.json')['checks']
    assert len(source_checks)==5 and all(r['passed'] for r in source_checks)
    assert read(PARENT/'training/state.json')['steps']==320
    assert INITIAL.is_file()
    plan={k:v for k,v in parent.items() if k not in ('cases','arms','train_steps','GPU_jobs')}
    plan.update(run_id='r48', parent_run='r47', initial_checkpoint=str(INITIAL),
        cases=[selected[i] for i in wanted], train_ids=TRAIN_IDS, eval_64=EVAL_64, eval_128=EVAL_128,
        diagnostic_ids=DIAGNOSTIC_IDS, max_steps=128, checkpoint_steps=[64,128],
        train_steps='64 → 固定验证 → 最多128', lr=1e-4, weight_decay=.01,
        train_size=[320,576], eval_size=[576,1024], prior_dropout={'RGB':.25,'geometry':.25},
        instruction_policy='所有C组与训练dropout保留同一参数指令；UC另行清空新指令/先验',
        diagnostics=['RGB_and_geometry','mismatched_RGB','geometry_only','null_priors'],
        architecture_change='none; r47 branch unchanged, response scalar logging only',
        scope='四既有train / 两合成DEV / 五真实DEV；两个可靠参考的模块探针；不是最终泛化评测',
        input_source=str(PARENT/'inputs'), data_new_cases=0, GPU_jobs=0,
        role_contract={'BUILD':'M010/M013/M018/M042，Y仅监督，不进入参考',
            'QUERY':'五真实DEV与两合成DEV；真实无隐藏GT；M003/M006的Y仅度量',
            'evidence':'原RGB参考先剔除A；GT pose/tracks与LiDAR辅助几何；不使用生成结果选参考'},
        stop_policy='工程/数值错误即停；128步后收口，不自动增加步数/数据/网格；等待用户开GPU')
    if (O/'manifest.json').exists():
        assert read(O/'manifest.json')==plan, '拒绝改写已登记的实验配置'
    else:dump(O/'manifest.json',plan)
    tokens={r['case_id']:r for r in read(PARENT/'reference_token_audit.json')['rows']}
    rows=[]
    for cid in wanted:
        c=selected[cid];req=request(c);contract=check_request(req)
        assert not contract['pending_GPU_reference_slots'], '不得绕过r47已选源mask验证'
        assert req.target_rgb.shape==(10,576,1024,3)
        assert req.priors['geometry'].shape==(10,12,144,256)
        assert req.priors['bev'].shape==(10,10,128,128)
        assert req.priors['bev_rays'].shape==(10,4,18,32,2)
        expected=F.adaptive_avg_pool2d(torch.from_numpy(req.edit_mask[:,None].astype('float32')),(144,256))[:,0]>0
        assert np.array_equal(expected.numpy(),req.priors['hole']>0), '控制H漏掉原mask触及单元'
        assert np.allclose(req.priors['geometry'][:,:3].sum(1),1), 'O/N/U未划分完整'
        for slot in range(6):
            saved=np.asarray(Image.open(PARENT/'inputs'/cid/f'reference_{slot:02}.png').convert('RGB'))
            assert np.array_equal(saved,req.priors['references'][slot]), '参考审核PNG不是实际输入'
        cond=req.branch_condition(torch.zeros(6,4,32,32),'cpu')
        for rgb,geo in [(False,False),(True,False),(False,True),(True,True)]:
            dropped=drop_priors(cond,rgb,geo)
            assert torch.equal(dropped['parameters'],cond['parameters'])
            assert torch.equal(dropped['hole'],cond['hole']) and torch.equal(dropped['query_time'],cond['query_time'])
        uc=drop_priors(cond,True,True,drop_parameters=True)
        assert not torch.count_nonzero(uc['parameters']) and not torch.count_nonzero(uc['reference_valid'])
        assert torch.equal(cond['parameters'],torch.from_numpy(req.priors['parameters'])), '消融原地修改源条件'
        row={'case_id':cid,'scene':c['scene'],'split':c['split'],'contract':contract,
            'instruction_same_in_all_C_arms':True,'UC_new_parameters_zero':True,
            'source_arrays_unchanged':True,'source_query_indices':c['frame_indices'],
            'query_H_fraction':float(req.edit_mask.mean()), 'reference_slots':tokens[cid]['slots']}
        if c['kind']=='synthetic':
            y=images(c,'target')
            assert np.array_equal(y[~req.edit_mask],req.target_rgb[~req.edit_mask]), 'synthetic X影响越出H'
            row['Y_target_only']=True
        else:
            frame=5; folder=PARENT/'evaluation'/cid/'RGB_and_geometry'
            raw=np.asarray(Image.open(folder/'native'/f'{frame:05}.png').convert('RGB'))
            final=np.asarray(Image.open(folder/f'{frame:05}.png').convert('RGB'))
            assert np.array_equal(req.compose(np.broadcast_to(raw,req.target_rgb.shape))[frame],final)
            interior=req.edit_mask[frame] & (req.alpha[frame]==1)
            assert np.array_equal(raw[interior],final[interior])
            row['f05_native_final']={'alpha_one_pixels':int(interior.sum()),
                'alpha_one_exact':True,'alpha_one_H_fraction':float(interior.sum()/req.edit_mask[frame].sum()),
                'mean_abs_RGB255_difference_H':float(np.abs(raw.astype(float)-final).mean(-1)[req.edit_mask[frame]].mean()),
                'scope':'保存PNG与实际alpha的精确检查；不凭这个数值判定残影来自哪个模块'}
        rows.append(row);print('INPUT_OK',cid,flush=True)
        del req,cond;gc.collect()
    # 真正r47学习过的参数必须严格兼容；这里只用受控激活检查，不冒充官方模型GPU响应。
    from safetensors.torch import load_file
    branch=MultiPriorBranch();branch.load_state_dict(load_file(str(INITIAL)),strict=True)
    assert sum(p.numel() for p in branch.parameters())==389856
    assert all(torch.isfinite(p).all() for p in branch.parameters())
    from iteration14.multi_prior import MultiPriorBranch as R47Branch
    original=R47Branch();original.load_state_dict(branch.state_dict(),strict=True)
    torch.manual_seed(42)
    geometry=torch.zeros(2,12,16,24);geometry[:,2]=1
    probe=dict(geometry=geometry,bev=torch.rand(2,10,24,24),bev_rays=torch.rand(2,4,8,12,2)*2-1,
        reference_latents=torch.rand(6,4,32,32),reference_valid=torch.ones(6,1,32,32),
        reference_pose=torch.rand(6,13),query_time=torch.tensor([[0.],[.9]]),
        parameters=torch.tensor([[1.,1.,1.7,1.]]).repeat(2,1),hole=torch.ones(2,1,16,24))
    branch.set_condition(**probe);original.set_condition(**probe)
    activation=torch.randn(2,320,16,24)
    with torch.no_grad():
        parent_output=original.residual(0,activation)
        assert torch.equal(parent_output,branch.residual(0,activation))
        branch.capture={'CFG':'C','sample_call':1}
        assert torch.equal(parent_output,branch.residual(0,activation)), '响应记录改变模型计算'
    assert len(branch.response_rows)==1 and np.isfinite(branch.response_rows[0]['gated_delta_rms'])
    checkpoint={'strict_load':True,'parameters':389856,'initial_checkpoint':str(INITIAL),
        'state_dict_matches_r47':True,'logging_preserves_r47_output_exact_on_controlled_activation':True,
        'main_frozen_GPU_check_pending':True}
    dump(O/'input_checks.json',{'rows':rows,'checkpoint':checkpoint,'scope':'CPU输入与消融合同，不是GPU模型/训练结果'})
    user=read(O/'user_review/human_review.json')
    assert len(user['sheets'])==5 and len(user['records'])==16
    dump(O/'preflight.json',{'CPU_ready':True,'cases':len(rows),'frames':10*len(rows),
        'train_cases':len(TRAIN_IDS),'QUERY_cases':len(EVAL_128),'C_instruction_invariant':True,
        'UC_parameters_explicitly_zero':True,'checkpoint':checkpoint,'seconds':time.monotonic()-started,
        'GPU_started':False,'failure_ledger_refs':['V77-F02']})
    if not (O/'controller_state.json').exists():
        dump(O/'controller_state.json',{'stage':'CPU_ready_waiting_user_GPU','GPU_jobs':0,
            'training_steps':0,'human_verdict':None,'failure_ledger_delta':'updated V77-F02: human review and ablation confound'})
    print('CPU_READY_WAITING_GPU',flush=True)


if __name__=='__main__':
    main()
