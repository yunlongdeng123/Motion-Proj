"""r48固定短循环；CPU仅准备，实际模型探针/训练/采样都要求GPU。"""
from common import *
import argparse, fcntl, random, time, gc
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from interface import encode_references, drop_priors
from multi_prior import MultiPriorBranch


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('没有可用GPU；等待用户开GPU，禁止CPU偷偷加载官方模型')
    assert read(O/'preflight.json')['CPU_ready']
    torch.set_num_threads(1)


def load_branch(path):
    branch=MultiPriorBranch().cuda()
    branch.load_state_dict(load_file(str(path)),strict=True)
    assert all(torch.isfinite(p).all() for p in branch.parameters())
    return branch


def lock_folder(path):
    path.mkdir(parents=True,exist_ok=True)
    handle=open(path/'lock','a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle


def snapshot(branch,optimizer,step,order,out):
    # 分支与优化器都保存成功后才推进resume指针；中断时复用最后完整快照。
    stem=f'{step:04}'
    tmp=out/f'branch_{stem}.safetensors.tmp'
    save_file({k:v.detach().cpu().contiguous() for k,v in branch.state_dict().items()},str(tmp))
    tmp.replace(out/f'branch_{stem}.safetensors')
    tmp=out/f'optimizer_{stem}.pt.tmp'
    torch.save({'optimizer':optimizer.state_dict(),'step':step,'order':order,
        'random_state':random.getstate(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'initial_checkpoint':str(INITIAL)},tmp)
    tmp.replace(out/f'optimizer_{stem}.pt')
    dump(out/'resume.json',{'step':step,'branch':str(out/f'branch_{stem}.safetensors'),
        'optimizer':str(out/f'optimizer_{stem}.pt')})


def train_until(until):
    assert until in (64,128)
    out=O/'training';handle=lock_folder(out);plan=read(O/'manifest.json');byid=cases()
    if until==128:
        assert read(O/'evaluation/step_0064/state.json')['stage']=='complete', '必须先检查64步真实结果'
    resume=read(out/'resume.json') if (out/'resume.json').exists() else {'step':0}
    start=resume['step'];assert start<=128
    if start>=until:
        print('TRAIN_ALREADY_COMPLETE',until,flush=True);return
    import train_control as train
    random.seed(plan['train_seed']);torch.manual_seed(plan['train_seed'])
    train.ENCODER=T/'r7/encoder_recovery/official_svd_encoder.safetensors'
    model,_,_=train.model_init();model.requires_grad_(False);model.eval()
    branch=load_branch(Path(resume['branch']) if start else INITIAL)
    model.model.diffusion_model.multi_prior_branch=branch;branch.attach(model.model.diffusion_model)
    optimizer=torch.optim.AdamW(branch.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    cache={};order=list(range(len(TRAIN_IDS)))
    for cid in TRAIN_IDS:
        req=request(byid[cid])
        cache[cid]=(train.resized(images(byid[cid],'target'),req.target_rgb,req.edit_mask,plan['train_size']),
                    req.branch_condition(encode_references(model,req),'cpu'))
    if start:
        state=torch.load(resume['optimizer'],map_location='cpu',weights_only=False)
        assert state['step']==start and state['initial_checkpoint']==str(INITIAL)
        optimizer.load_state_dict(state['optimizer']);order=state['order']
        random.setstate(state['random_state']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    dump(out/'config.json',{'train_ids':TRAIN_IDS,'scene_count':4,'lr':plan['lr'],
        'seed':plan['train_seed'],'resolution':plan['train_size'],'max_steps':128,
        'initial_checkpoint':str(INITIAL),'new_branch_parameters':389856,'main_and_3D_frozen':True,
        'loss':'unchanged official diffusion loss','C_task_instruction':'same even when both priors dropped',
        'prior_dropout':plan['prior_dropout'],'source_encoder_restore':model.encoder_restore_evidence,
        'selection':'fixed 64/128 checkpoints, no real DEV training or best-case selection'})
    started=time.monotonic();attempt=str(time.time_ns());torch.cuda.reset_peak_memory_stats()
    dump(out/'state.json',{'stage':'training','steps':start,'until':until,'resume_step':start,'attempt':attempt})
    for step in range(start,until):
        if step%len(order)==0:random.shuffle(order)
        cid=TRAIN_IDS[order[step%len(order)]];prepared,condition=cache[cid]
        drgb=step>=2 and random.random()<.25;dgeo=step>=2 and random.random()<.25
        branch.set_condition(**{k:v.cuda() for k,v in drop_priors(condition,drgb,dgeo).items()})
        assert torch.equal(branch._condition['parameters'].cpu(),condition['parameters'])
        optimizer.zero_grad(set_to_none=True);value=train.loss(model,prepared,plan['train_seed']+step)
        if not torch.isfinite(value):raise ValueError('非有限loss，停止短循环')
        value.backward()
        assert all(p.grad is None for n,p in model.named_parameters() if '.multi_prior_branch.' not in n)
        if step==0:
            names=('bev_encoder.0.weight','reference_encoder.0.weight')
            probe={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                   for n,p in branch.named_parameters() if n in names}
            assert all(probe.values()), f'RGB/BEV未收到实际梯度: {probe}'
            dump(out/'backward_probe.json',{'branch_gradients':probe,'main_frozen':True})
        norm=float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1,error_if_nonfinite=True))
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in branch.parameters()):
            raise ValueError('分支参数非有限，停止短循环')
        state={'stage':'training','steps':step+1,'case':cid,'loss':float(value.detach()),
            'grad_norm':norm,'drop_RGB':drgb,'drop_geometry':dgeo,'instruction_preserved':True,
            'until':until,'resume_step':start,'attempt':attempt,'elapsed_seconds':time.monotonic()-started,
            'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'main_frozen':True}
        dump(out/'state.json',state)
        with (out/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(state)+'\n')
        if (step+1)%16==0:snapshot(branch,optimizer,step+1,order,out)
        print('TRAIN',step+1,cid,state['loss'],flush=True)
    state['stage']='checkpoint_ready';dump(out/'state.json',state)


def encoding_probe(engine,req,cid,latent):
    """保存真正冻结VAE的重构，让RGB不足与编码损失可分开观察。"""
    embed=next(e for e in engine.model.conditioner.embedders
               if getattr(e,'input_key',None)=='cond_frames' and hasattr(e,'encoder'))
    folder=O/'diagnostics'/cid/'reference_encoding';folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for slot in range(6):
            rec=embed.encoder.decode(latent[slot:slot+1]/embed.scale_factor)
            rgb=((rec[0].float().permute(1,2,0).cpu().numpy()+1)*127.5).round().clip(0,255).astype('uint8')
            assert rgb.shape==(256,256,3) and np.isfinite(rgb).all()
            Image.fromarray(req.priors['references'][slot]).save(folder/f'source_{slot:02}.png')
            Image.fromarray(rgb).save(folder/f'decoded_{slot:02}.png')
            valid=req.priors['reference_valid'][slot]>0
            rows.append({'slot':slot,'valid_pixels':int(valid.sum()),
                'RGB_MAE_on_valid':float(np.abs(rgb.astype(float)-req.priors['references'][slot]).mean(-1)[valid].mean()/255) if valid.any() else None,
                'latent_rms':float(latent[slot].square().mean().sqrt())})
    dump(folder/'result.json',{'case_id':cid,'slots':rows,'encoder':'official frozen cond_frames VAE',
        'scope':'重构误差/图片仅检查编码损失，不代表同一保护车已被生成器正确使用'})


def infer(engine,req,cid,folder,arm,latents,label):
    from repair_drive import set_seed
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    plan=read(O/'manifest.json');engine.get_deletion(req,arm=arm,reference_latents=latents)
    set_seed(plan['eval_seed']);started=time.monotonic()
    with torch.inference_mode():engine.predict(1,False,'Deletion')
    assert engine.prior_cfg_calls=={'unconditional':25,'conditional':25}
    raw=np.stack(engine.im_result);assert raw.shape==req.target_rgb.shape and raw.dtype==np.uint8
    final=req.compose(raw);(folder/'native').mkdir(exist_ok=True)
    for f in range(10):
        Image.fromarray(raw[f]).save(folder/'native'/f'{f:05}.png')
        Image.fromarray(final[f]).save(folder/f'{f:05}.png')
    result={'case_id':cid,'condition':label,'arm':arm,'seed':plan['eval_seed'],'sampling_steps':25,
        'seconds':time.monotonic()-started,'CFG_prior_calls':engine.prior_cfg_calls,
        'C_parameters':engine._conditioned_priors['parameters'][0].tolist(),
        'UC_parameters':engine._unconditional_priors['parameters'][0].tolist(),
        'module_response':engine.branch.response_rows,'outside_H_exact':True,
        'human_verdict':None,'temporal_verdict':None}
    assert result['C_parameters']==req.priors['parameters'][0].tolist()
    assert result['UC_parameters']==[0.,0.,0.,0.]
    dump(folder/'result.json',result);return result


def diagnose():
    out=O/'diagnostics';handle=lock_folder(out)
    from engine import MultiPriorDeletionEngine
    engine=MultiPriorDeletionEngine(load_branch(INITIAL));byid=cases();cache={}
    for cid in DIAGNOSTIC_IDS:
        req=request(byid[cid]);latent=encode_references(engine.model,req)
        assert torch.isfinite(latent).all()
        cache[cid]=(req,latent)
        if not (out/cid/'reference_encoding/result.json').exists():encoding_probe(engine,req,cid,latent)
    results=[]
    for cid in DIAGNOSTIC_IDS:
        req,latent=cache[cid];other=DIAGNOSTIC_IDS[1] if cid==DIAGNOSTIC_IDS[0] else DIAGNOSTIC_IDS[0]
        for label in read(O/'manifest.json')['diagnostics']:
            # 错配仅替换编码RGB内容；query几何/参考pose/valid/时间/指令都固定。
            wrong=label=='mismatched_RGB'
            result=infer(engine,req,cid,out/cid/label,'RGB_and_geometry' if wrong else label,
                         cache[other][1] if wrong else latent,label)
            if wrong:
                result.update(mismatched_donor=other,
                    mismatch_scope='跨case RGB latent替换；保持本case pose/valid/几何/指令，仅作响应探针')
                dump(out/cid/label/'result.json',result)
            results.append(result);print('DIAG',cid,label,flush=True)
    dump(out/'state.json',{'stage':'complete','initial_checkpoint':str(INITIAL),'results':results,
        'scope':'修正消融的r47权重响应；不把像素/响应改变直接当任务改善'})


def evaluate(step):
    assert step in (64,128);ckpt=O/'training'/f'branch_{step:04}.safetensors';assert ckpt.is_file()
    out=O/'evaluation'/f'step_{step:04}';handle=lock_folder(out);byid=cases()
    from engine import MultiPriorDeletionEngine
    engine=MultiPriorDeletionEngine(load_branch(ckpt));results=[]
    for cid in EVAL_64 if step==64 else EVAL_128:
        req=request(byid[cid]);latent=encode_references(engine.model,req)
        result=infer(engine,req,cid,out/cid,'RGB_and_geometry',latent,f'r48_step_{step:04}')
        if byid[cid]['kind']=='synthetic':
            gt=images(byid[cid],'target')
            value=np.stack([np.asarray(Image.open(out/cid/f'{f:05}.png').convert('RGB')) for f in range(10)])
            result['GT_hole_MAE']=float(np.abs(value.astype(float)-gt).mean(-1)[req.edit_mask].mean()/255)
            dump(out/cid/'result.json',result)
        results.append(result);dump(out/'state.json',{'stage':'running','step':step,'completed':results})
        print('EVAL',step,cid,flush=True)
        del req,latent;gc.collect()
    dump(out/'state.json',{'stage':'complete','step':step,'completed':results,'human_verdict':None})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnose','train','evaluate'])
    p.add_argument('--step',type=int,choices=[64,128]);a=p.parse_args();require_gpu()
    if a.action=='diagnose':diagnose()
    elif a.action=='train':train_until(a.step)
    else:evaluate(a.step)
