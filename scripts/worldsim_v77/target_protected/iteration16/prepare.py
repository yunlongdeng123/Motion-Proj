"""仅CPU：生成空间对应sidecar、可复核输入与固定GPU实验清单。"""
from common import *
import time, gc
import numpy as np
import torch
from PIL import Image
from routing import build_routing, attention_bias, letterbox_coordinates, mismatch_pairing
from checks import checks
from interface import check_request, drop_priors


def main():
    started = time.monotonic();torch.set_num_threads(1)
    selected = {c['case_id']:c for c in read(PARENT/'manifest.json')['cases']}
    wanted = TRAIN_IDS+EVAL_IDS
    assert {selected[c]['scene'] for c in TRAIN_IDS}.isdisjoint({selected[c]['scene'] for c in EVAL_IDS})
    original_plan = read(CONTROL/'manifest.json')
    plan = {k:original_plan[k] for k in ('task_id','eval_seed','eval_steps','train_seed','train_size','eval_size','lr','weight_decay','prior_dropout')}
    plan.update(run_id='r49',parent_run='r47',cases=[selected[c] for c in wanted],train_ids=TRAIN_IDS,
        eval_ids=EVAL_IDS,diagnostic_ids=DIAGNOSTIC_IDS,train_steps=64,initial_checkpoint=str(INITIAL),
        primary_hypothesis='参考的身份与局部位置软绑定能减少半透明车形，且保住后车',
        architecture_change='仅原RGB cross-attention logits增加无参数soft spatial bias；原state_dict与389856参数范围不变',
        routing={'source_patch_purity':.95,'actor_confidence':.5,'background_confidence':.8,
            'formula':'min(Qquery,Qref) * log(0.1 + 0.9 * similarity)',
            'sigma':'max(one query cell, projected source patch radius)',
            'unknown':'bias=0; original generation remains available',
            'off':'exact original r47 attention path'},
        fixed_controls=['r46 official weights + r21 full SAM','r47 step320','r48 step64 same data/budget'],
        GPU_plan={'zero_shot_windows':9,'after_64_windows':11,'max_total_windows':20,
            'first':'冻结r47权重：5correct+两个真实例各wrong/noRGB',
            'gate':'先直接review零训练输出；若未达标且无输入/工程反证，才一次64步',
            'after':'5correct+两个真实例各wrong/noRGB+两个真实例同权重routing_off',
            'no_auto_training_from_CPU':True,'no_128_or_grid':True},
        role_contract={'BUILD':'只有M010/M013/M018/M042；Y仅训练监督，不进入参考或空间对应',
            'QUERY':'A034/A061_w08/A022/M003/M006；GT合成仅度量；真实DEV不训练',
            'geometry':'合法GT pose/tracks与已有LiDAR辅助；框面proxy不是真实silhouette'},
        scope='4旧train+3真实DEV+2合成DEV；不是未曝光泛化评测',failure_ledger_refs=['V77-F02'],
        GPU_jobs=0,human_verdict=None)
    if (O/'manifest.json').exists():assert read(O/'manifest.json')==plan, '拒绝改写已登记配置'
    else:dump(O/'manifest.json',plan)
    from iteration14.prepare_inputs import background_points
    rows=[]
    for cid in wanted:
        c = selected[cid];p = PARENT/'inputs'/cid;dest = O/'inputs'/cid;dest.mkdir(parents=True,exist_ok=True)
        with np.load(p/'condition.npz') as data:priors = {k:data[k].copy() for k in data.files}
        refs = read(p/'references.json');points = background_points(c)
        route,aux,meta = build_routing(c,priors,refs,points)
        np.savez_compressed(dest/'routing.npz',**route,**aux)
        dump(dest/'routing.json',meta)
        req = request(c);contract = check_request(req)
        assert not contract['pending_GPU_reference_slots']
        cond = req.branch_condition(torch.zeros(6,4,32,32),'cpu')
        for rgb,geo in [(False,False),(True,False),(False,True),(True,True)]:
            arm = drop_priors(cond,rgb,geo)
            assert torch.equal(arm['parameters'],cond['parameters'])
            if rgb or geo:assert not torch.count_nonzero(arm['routing_query_q'])
        uc = drop_priors(cond,True,True,drop_parameters=True)
        assert not torch.count_nonzero(uc['parameters']) and not torch.count_nonzero(uc['routing_projected_q'])
        row = {'case_id':cid,'scene':c['scene'],'split':c['split'],'contract':contract,
            'main_protected_token':meta['main_protected_token'],'slots':meta['source_slots'],
            'query_frames':[],'attention_scales':[],'human_verdict':None}
        main_id = meta['identities'].get(meta['main_protected_token'],0)
        h = req.priors['hole']>0
        for f in range(10):
            owner = route['routing_query_owner'][f,0]
            row['query_frames'].append({'frame':f,'H_cells':int(h[f].sum()),
                'main_B_proxy_cells_H':int(((owner==main_id)&h[f]).sum()) if main_id else 0,
                'known_N_cells_H':int(((owner==-1)&h[f]).sum()),
                'unknown_cells_H':int(((owner==0)&h[f]).sum()),
                'main_B_spatial_patches':int(((route['routing_reference_owner'].ravel()==main_id)&
                    (route['routing_projected_q'][f]>0)).sum()) if main_id else 0})
        for size in ((18,32),(10,18),(5,9)):
            bias,owner,q = attention_bias(cond,size)
            hh = torch.nn.functional.adaptive_max_pool2d(cond['hole'],size).flatten(1)>0
            fractions = []
            for f in range(10):
                fractions.append({'frame':f,'H_queries':int(hh[f].sum()),
                    'main_B_queries_H':int(((owner[f]==main_id)&hh[f]).sum()) if main_id else 0,
                    'known_N_queries_H':int(((owner[f]==-1)&hh[f]).sum()),
                    'unknown_queries_H':int(((owner[f]==0)&hh[f]).sum()),
                    'biased_pairs_H':int((bias[f][hh[f]]<0).sum())})
            row['attention_scales'].append({'size':list(size),'frames':fractions})
        if cid in DIAGNOSTIC_IDS:
            assert main_id and (route['routing_reference_owner']==main_id).any(), '主保护车参考对应为空'
            assert any(x['main_B_spatial_patches'] for x in row['query_frames']), '主保护车没有任何合法位置对应'
        # 原输入PNG、alpha和query参数继续直接从r47读取；不复制或重写。
        for slot in range(6):
            saved = np.asarray(Image.open(p/f'reference_{slot:02}.png').convert('RGB'))
            assert np.array_equal(saved,priors['references'][slot])
            uv,inside = letterbox_coordinates(refs['references'][slot])
            assert np.isfinite(uv[inside]).all()
        dump(dest/'check.json',row);rows.append(row)
        print('ROUTING_READY',cid,main_id,flush=True)
        del req,priors,points,route,aux,cond;gc.collect()
    semantic = checks(INITIAL)
    mismatches=[]
    for cid,donor in [('A034','A061_w08'),('A061_w08','A034')]:
        arrays=[];metadata=[]
        for case_id in (cid,donor):
            with np.load(O/'inputs'/case_id/'routing.npz') as data:arrays.append({k:data[k] for k in data.files})
            metadata.append(read(O/'inputs'/case_id/'routing.json'))
        pairs,stats=mismatch_pairing(arrays[0],metadata[0],arrays[1],metadata[1])
        assert len(set(a for a,b in pairs))==len(pairs)==len(set(b for a,b in pairs))
        mismatches.append({'recipient':cid,'donor':donor,**stats,'pairs':pairs})
    dump(O/'mismatch_plan.json',mismatches)
    dump(O/'input_checks.json',{'cases':rows,'semantic_checks':semantic,
        'mismatch_plan':mismatches,'source_RGB_H_alpha_geometry_unchanged':True,'new_RGB_selection':False,'Y_condition_read':False})
    ready = all(any(f['main_B_queries_H']>0 for f in next(x for x in rows if x['case_id']==cid)['attention_scales'][0]['frames']) for cid in DIAGNOSTIC_IDS)
    dump(O/'preflight.json',{'CPU_ready':ready,'input_cases':len(rows),'input_frames':10*len(rows),
        'semantics':semantic,'seconds':time.monotonic()-started,'GPU_started':False,
        'main_B_coverage_at_18x32':ready,'role':'CPU工程准入，不是视觉效果通过'})
    if not (O/'controller_state.json').exists():
        dump(O/'controller_state.json',{'stage':'CPU_ready_waiting_user_GPU' if ready else 'CPU_stopped_insufficient_correspondence',
            'GPU_jobs':0,'training_steps':0,'new_inference_windows':0,'human_verdict':None,
            'no_background_GPU_waiter':True})
    print('CPU_PREPARED_GPU_NOT_STARTED',ready,flush=True)


if __name__=='__main__':main()
