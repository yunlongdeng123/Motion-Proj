"""r47 GPU入口：先官方模型接入/显存探针，再固定320步；CPU阶段不运行。"""
from common import *
import argparse, random, time, fcntl, shutil
import numpy as np
import torch
from PIL import Image
from interface import DeletionRequest, encode_references, drop_priors
from multi_prior import MultiPriorBranch
import train_control as train


def request(c):
    p=O/'inputs'/c['case_id'];priors=dict(np.load(p/'condition.npz'))
    x=images(c,'rgb');h=images(c,'hole')>0
    alpha=images(c,'alpha').astype('float32')/255 if c['kind']=='real' else h.astype('float32')
    req=DeletionRequest(x,h,alpha,priors,read(p/'references.json'));req.validate()
    return req


def train_branch():
    plan=read(O/'manifest.json');out=O/'training';out.mkdir(exist_ok=True)
    lock=open(out/'lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'state.json').exists():
        if read(out/'state.json')['stage']=='complete':return
        raise RuntimeError('已有未完成训练，保留现场；不可静默覆盖/重复启动')
    torch.set_num_threads(4);random.seed(plan['train_seed']);torch.manual_seed(plan['train_seed'])
    train.ENCODER=T/'r7/encoder_recovery/official_svd_encoder.safetensors'
    model,_,_=train.model_init();model.requires_grad_(False);model.eval()
    branch=MultiPriorBranch().cuda();model.model.diffusion_model.multi_prior_branch=branch
    branch.attach(model.model.diffusion_model)
    optimizer=torch.optim.AdamW(branch.parameters(),lr=1e-4,weight_decay=.01)
    cases=[c for c in plan['cases'] if c['split']=='train'];cache={};order=list(range(len(cases)))
    for c in cases:
        req=request(c)
        if any(r['GPU_source_mask_validation_pending'] for r in req.reference_manifest['references']):
            raise RuntimeError('训练参考仍有待验证目标遮罩')
        cache[c['case_id']]=(train.resized(images(c,'target'),req.target_rgb,req.edit_mask,[320,576]),
                            req.branch_condition(encode_references(model,req),'cpu'))
    started=time.monotonic()
    dump(out/'config.json',{'steps':plan['train_steps'],'lr':1e-4,'seed':plan['train_seed'],
        'parameters':sum(p.numel() for p in branch.parameters()),'main_and_3D_frozen':True,
        'base_checkpoint':'official DriveEditor, no previous finetuned/Adapter checkpoint',
        'frozen_reference_encoder':'official cond_frames RGB VAE',
        'loss':'unchanged official diffusion loss','prior_dropout':{'RGB':.25,'geometry':.25,'first_two_probe_steps':'both present'},'data':[c['case_id'] for c in cases],
        'target_encoder_restore':model.encoder_restore_evidence,'selection':'fixed final step; no real-case tuning'})
    for step in range(plan['train_steps']):
        if step%len(order)==0:random.shuffle(order)
        c=cases[order[step%len(order)]];prepared,condition=cache[c['case_id']]
        drop_rgb=step>=2 and random.random()<.25;drop_geometry=step>=2 and random.random()<.25
        branch.set_condition(**{k:v.cuda() for k,v in drop_priors(condition,drop_rgb,drop_geometry).items()})
        if step==0:
            with torch.no_grad():
                branch.enabled=False;base=train.loss(model,prepared,plan['train_seed'])
                branch.enabled=True;zero=train.loss(model,prepared,plan['train_seed'])
            assert torch.equal(base,zero),'实际官方模型零初始化不等价，停止'
            dump(out/'zero_init.json',{'official_model_loss_exact':True})
        optimizer.zero_grad(set_to_none=True);value=train.loss(model,prepared,plan['train_seed']+step)
        if not torch.isfinite(value):raise ValueError('非有限loss')
        value.backward()
        assert all(p.grad is None for n,p in model.named_parameters() if '.multi_prior_branch.' not in n),'主干被更新'
        if step==1:
            evidence={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                for n,p in branch.named_parameters() if n in ['bev_encoder.0.weight','reference_encoder.0.weight']}
            if not all(evidence.values()):raise RuntimeError(f'实际RGB/BEV梯度未到达: {evidence}')
            dump(out/'backward_probe.json',{'actual_branch_gradients':evidence,'main_frozen':True,
                'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started})
        norm=float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1));optimizer.step()
        state={'stage':'training','steps':step+1,'case':c['case_id'],'loss':float(value.detach()),'grad':norm,'drop_RGB':drop_rgb,'drop_geometry':drop_geometry,
               'elapsed_seconds':time.monotonic()-started,'peak_GiB':torch.cuda.max_memory_allocated()/2**30}
        dump(out/'state.json',state)
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        if (step+1)%40==0:
            from safetensors.torch import save_file
            save_file({k:v.detach().cpu().contiguous() for k,v in branch.state_dict().items()},str(out/f'branch_{step+1:04}.safetensors'))
            torch.save({'optimizer':optimizer.state_dict(),'step':step+1,'order':order,'random_state':random.getstate()},out/f'optimizer_{step+1:04}.pt')
        print('TRAIN',step+1,float(value.detach()),flush=True)
    state['stage']='complete';dump(out/'state.json',state)


def evaluate():
    plan=read(O/'manifest.json');assert read(O/'training/state.json')['stage']=='complete'
    # 额外相机参考的目标剔除需要先独立验证；不能把候选包直接当合法模型输入。
    if not (O/'source_mask_validation.json').exists():raise RuntimeError('先完成额外相机目标遮罩GPU验证')
    dest=O/'evaluation';dest.mkdir(exist_ok=True);lock=open(dest/'lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from repair_drive import set_seed
    from engine import MultiPriorDeletionEngine
    from safetensors.torch import load_file
    branch=MultiPriorBranch().cuda();branch.load_state_dict(load_file(str(O/'training'/f"branch_{plan['train_steps']:04}.safetensors")),strict=True)
    engine=MultiPriorDeletionEngine(branch);completed=[]
    for c in [c for c in plan['cases'] if c['split']!='train']:
        req=request(c)
        if any(r['GPU_source_mask_validation_pending'] and not r['padding'] for r in req.reference_manifest['references']):
            raise RuntimeError(f"{c['case_id']}有未验证源mask")
        cond=req.branch_condition(encode_references(engine.model,req),'cuda')
        for arm in plan['arms']:
            out=dest/c['case_id']/arm;out.mkdir(parents=True,exist_ok=True)
            if (out/'result.json').exists():completed.append(read(out/'result.json'));continue
            if arm=='baseline':
                parent=T/'r46/evaluation'/c['case_id']/'adapter_off'
                assert parent.is_dir()
                # 同一r46目标RGB/H/alpha/seed原样复用；新参考不进入基线。
                for p in parent.glob('*.png'):shutil.copy2(p,out/p.name)
                shutil.copytree(parent/'native',out/'native',dirs_exist_ok=True)
                result=read(parent/'result.json')|{'arm':'baseline','reused_from':str(parent),'GPU_new_window':False}
            else:
                engine.get_deletion(req,arm=arm,reference_latents=cond['reference_latents'])
                set_seed(plan['eval_seed']);started=time.monotonic()
                with torch.inference_mode():engine.predict(1,False,'Deletion')
                raw=np.stack(engine.im_result);assert len(raw)==10;comp=req.compose(raw)
                (out/'native').mkdir(exist_ok=True)
                for i in range(10):
                    Image.fromarray(raw[i]).save(out/'native'/f'{i:05}.png');Image.fromarray(comp[i]).save(out/f'{i:05}.png')
                result={'case_id':c['case_id'],'arm':arm,'seconds':time.monotonic()-started,'GPU_new_window':True,
                        'human_verdict':None,'temporal_verdict':None}
            dump(out/'result.json',result);completed.append(result)
            dump(dest/'state.json',{'stage':'running','completed':completed,'total':60});print('EVAL',c['case_id'],arm,flush=True)
    dump(dest/'state.json',{'stage':'complete_pending_review','completed':completed,'human_verdict':None})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['train','evaluate']);args=parser.parse_args()
    if not torch.cuda.is_available():raise SystemExit('没有GPU，停止；CPU准备不能自动切换或启动训练')
    assert read(O/'preflight.json')['CPU_ready']
    train_branch() if args.action=='train' else evaluate()
