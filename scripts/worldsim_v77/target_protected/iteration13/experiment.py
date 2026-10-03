"""r45只训练小Adapter；固定三臂推理，无条件/全未知/真实条件。"""
from common import *
import argparse, time, random, fcntl
import numpy as np
import torch
from PIL import Image
os.environ['DRIVEEDITOR_SEQUENTIAL_CFG']='1'
from semantic_adapter import SemanticAdapter,pack_onuq,checks
import train_control as train

def condition(c,device='cuda'):
    states=[dict(np.load(O/'conditions'/c['case_id']/f'{i:05}.npz')) for i in range(10)]
    return pack_onuq({k:torch.from_numpy(np.stack([s[k] for s in states])) for k in ['O','N','U','Q']}).to(device)

def train_adapter():
    if not torch.cuda.is_available():raise RuntimeError('没有GPU；不启动训练')
    plan=read(O/'manifest.json');out=O/'training';out.mkdir(exist_ok=True)
    lock=open(out/'lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'state.json').exists():
        if read(out/'state.json')['stage']=='complete':return
        raise RuntimeError('已有未完成训练；保存现场后按实际checkpoint恢复，不静默重置')
    assert read(O/'condition_state.json')['stage']=='CPU_conditions_complete'
    assert read(O/'preflight.json')['stage']=='CPU_ready_GPU_not_run'
    torch.set_num_threads(4);random.seed(plan['seed']);torch.manual_seed(plan['seed'])
    train.ENCODER=T/'r7/encoder_recovery/official_svd_encoder.safetensors'
    model,_,_=train.model_init();model.requires_grad_(False);model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    adapter=SemanticAdapter().cuda();unet=model.model.diffusion_model
    # 注册参数，使保存/梯度审计能明确区分主干与支路。
    unet.onuq_adapter=adapter;adapter.attach(unet)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=plan['lr'],weight_decay=.01)
    cases=[c for c in plan['cases'] if c['split']=='train'];order=list(range(len(cases)));cache={}
    dump(out/'config.json',{'trainable_parameters':sum(p.numel() for p in adapter.parameters()),
        'backbone_frozen':True,'base':'original DriveEditor','channels':['O','N','U','Q'],
        'steps':plan['steps'],'lr':plan['lr'],'loss':'unchanged official diffusion loss',
        'data':[c['case_id'] for c in cases],'input_contract':'erase X before resizing; Y only loss',
        'encoder_restore':model.encoder_restore_evidence})
    started=time.monotonic()
    for step in range(plan['steps']):
        if step%len(cases)==0:random.shuffle(order)
        c=cases[order[step%len(cases)]];cid=c['case_id']
        if cid not in cache:
            h=images(c,'hole')>0
            cache[cid]=(train.resized(images(c,'target'),images(c,'rgb'),h,plan['train_size']),
                        condition(c,'cpu'),torch.from_numpy(h[:,None]))
        prepared,g,h=cache[cid];adapter.set_condition(g.cuda(),h.cuda())
        if step==0:
            with torch.no_grad():
                adapter.enabled=False;a=train.loss(model,prepared,plan['seed'])
                adapter.enabled=True;b=train.loss(model,prepared,plan['seed'])
            assert torch.equal(a,b),'零分支损失不等价'
            dump(out/'zero_init.json',{'actual_model_loss_equal':True,'CPU_contracts':checks()})
        optimizer.zero_grad(set_to_none=True)
        value=train.loss(model,prepared,plan['seed']+step)
        if not torch.isfinite(value):raise ValueError('非有限loss')
        value.backward()
        assert all(p.grad is None for n,p in model.named_parameters() if '.onuq_adapter.' not in n),'主干有梯度'
        if step==0:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0
                       for n,p in adapter.named_parameters() if n.startswith('zero_heads.'))
        grad=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.));optimizer.step()
        state={'stage':'training','steps':step+1,'case':cid,'loss':float(value.detach()),'grad_norm':grad,
               'elapsed_s':time.monotonic()-started,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'human_verdict':None}
        dump(out/'state.json',state)
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        if (step+1)%40==0:
            from safetensors.torch import save_file
            save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(out/f'adapter_{step+1:04}.safetensors'))
            torch.save({'optimizer':optimizer.state_dict(),'step':step+1,'random_state':random.getstate(),'order':order},out/f'optimizer_{step+1:04}.pt')
        print('TRAIN',step+1,cid,float(value.detach()),flush=True)
    state['stage']='complete';dump(out/'state.json',state)

def evaluate():
    if not torch.cuda.is_available():raise RuntimeError('没有GPU；不启动推理')
    plan=read(O/'manifest.json');assert read(O/'training/state.json')['stage']=='complete'
    dest=O/'evaluation';dest.mkdir(exist_ok=True)
    lock=open(dest/'lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from repair_drive import Engine,set_seed
    from safetensors.torch import load_file
    torch.set_num_threads(4);e=Engine();e.model.requires_grad_(False)
    adapter=SemanticAdapter().cuda();adapter.load_state_dict(load_file(str(O/'training'/f"adapter_{plan['steps']:04}.safetensors")),strict=True)
    adapter.attach(e.model.model.diffusion_model);adapter.eval()
    cases=[c for c in plan['cases'] if c['split']!='train'];completed=[]
    for c in cases:
        cid=c['case_id'];x=images(c,'rgb');h=images(c,'hole')>0;g=condition(c)
        alpha=images(c,'alpha').astype('float32')/255 if c['kind']=='real' else h.astype('float32')
        assert not np.any((alpha>0)&~h)
        for arm in plan['arms']:
            out=dest/cid/arm;out.mkdir(parents=True,exist_ok=True)
            if (out/'result.json').exists():completed.append(read(out/'result.json'));continue
            gg=g.clone()
            if arm=='all_unknown':gg.zero_();gg[:,2]=1
            adapter.set_condition(gg,torch.from_numpy(h[:,None]).cuda());adapter.enabled=arm!='adapter_off'
            e.im=list(x);e.masks=list(h);e.masked_condition=None;e.previous_segment_last_frame=None;e.im_result=[];set_seed(plan['eval_seed'])
            started=time.monotonic()
            with torch.inference_mode():e.predict(1,False,'Deletion')
            assert len(e.im_result)==10
            (out/'native').mkdir(exist_ok=True)
            scores=[];y=images(c,'target') if c['kind']=='synthetic' else None
            protected_scores=[]
            for i,raw in enumerate(e.im_result):
                comp=np.rint(raw*alpha[i,...,None]+x[i]*(1-alpha[i,...,None])).clip(0,255).astype('uint8')
                assert np.array_equal(comp[~h[i]],x[i][~h[i]])
                Image.fromarray(raw).save(out/'native'/f'{i:05}.png');Image.fromarray(comp).save(out/f'{i:05}.png')
                if y is not None:
                    error=np.abs(raw.astype(float)-y[i]).mean(-1)/255
                    scores.append(float(error[h[i]].mean()))
                    from audit import protected_mask
                    protected=protected_mask(Path(c['folder']),i,(576,1024),c['type']) & h[i]
                    if protected.any():protected_scores.append(float(error[protected].mean()))
            result={'case_id':cid,'arm':arm,'seconds':time.monotonic()-started,'hole_MAE':float(np.mean(scores)) if scores else None,
                    'protected_hole_MAE':float(np.mean(protected_scores)) if protected_scores else None,
                    'same_input_seed_and_writeback':True,'human_verdict':None,'temporal_verdict':None}
            dump(out/'result.json',result);completed.append(result)
            dump(dest/'state.json',{'stage':'running','completed':completed,'total':len(cases)*len(plan['arms'])})
            print('EVAL',cid,arm,flush=True)
    dump(dest/'state.json',{'stage':'complete_pending_visual_review','completed':completed,'human_verdict':None})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);a=p.parse_args()
    train_adapter() if a.action=='train' else evaluate()
