"""原DriveEditor结构/损失的单卡有界微调；先清零X，再缩放，Y只作监督。"""
from pathlib import Path
import argparse, json, sys, os, time, random, traceback, functools, fcntl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from PIL import Image

REPO=Path('/root/autodl-tmp/motion_proj_v77')
OFFICIAL=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
sys.path.insert(0,str(REPO/'scripts/worldsim_v77/target_protected'))
from iteration2.driveeditor_contract import deletion_batch

def save_json(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def selected(k):
    return k.startswith('model.diffusion_model.') and '_3d' not in k and '.transformer_blocks.' in k and '.attn1.' in k and any(v in k for v in ('.to_q.','.to_k.','.to_v.','.to_out.'))

def arrays(case,start=0):
    folder=Path(case['folder'])
    load=lambda role,rgb: np.stack([np.asarray(Image.open(p).convert('RGB' if rgb else 'L')) for p in sorted((folder/role).glob('*.png'))[start:start+10]])
    return load('Y',True),load('X',True),load('model_hole',False)>0

def resized(y,x,h,size):
    if np.any((x!=y).any(-1)&~h):raise ValueError('X的影响超出H')
    ty=torch.from_numpy(y.copy()).permute(0,3,1,2).float()/127.5-1
    cx=torch.from_numpy(x.copy()).permute(0,3,1,2).float()/127.5-1
    cx.masked_fill_(torch.from_numpy(h[:,None]),0) # 禁止先resize未遮的X
    yy=F.interpolate(ty,size=size,mode='bilinear',align_corners=False,antialias=True)
    cc=F.interpolate(cx,size=size,mode='bilinear',align_corners=False,antialias=True)
    hh=F.interpolate(torch.from_numpy(h[:,None].astype('float32')),size=size,mode='nearest')[:,0].bool()
    return yy,cc,hh

def sample(yy,cc,hh,seed):
    # 此代理只构造官方字段；随后用真实先遮后缩条件覆盖所有主路条件。
    proxy=((yy.permute(0,2,3,1)+1)*127.5).round().clamp(0,255).byte().numpy()
    batch=deletion_batch(proxy,proxy,hh.numpy(),seed=seed)
    gen=torch.Generator().manual_seed(seed+971)
    batch['jpg']=yy
    batch['cond_frames']=cc+batch['cond_aug']*torch.randn(cc.shape,generator=gen)
    batch['cond_frames_eval']=cc+.02*torch.randn(cc.shape,generator=gen)
    batch['cond_frames_without_noise']=[cc[0],torch.ones(3,224,224)]
    return batch

def move(obj):
    if isinstance(obj,torch.Tensor):return obj.cuda()
    if isinstance(obj,list):return [move(v) for v in obj]
    if isinstance(obj,dict):return {k:move(v) for k,v in obj.items()}
    return obj

def model_init():
    os.chdir(OFFICIAL);sys.path.insert(0,str(OFFICIAL))
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    # 原reentrant checkpoint会在冻结输入时丢失参数梯度；只换计算重算API，不改结构/公式。
    import torch.utils.checkpoint as cp
    import sgm.modules.attention as att
    import sgm.modules.video_attention as vatt
    import sgm.modules.diffusionmodules.openaimodel as om
    import sgm.modules.diffusionmodules.video_model as vm
    wrapper=functools.partial(cp.checkpoint,use_reentrant=False)
    for module in (att,vatt,om,vm):
        if hasattr(module,'checkpoint'):module.checkpoint=wrapper
    config=OmegaConf.load(OFFICIAL/'configs/train.yaml')
    config.model.params.ckpt_path=str(OFFICIAL/'checkpoints/model.safetensors')
    config.model.params.init_model=True
    # 和已适配sample.yaml一致：完整checkpoint含CLIP，禁止再次联网取初始权重。
    for key in ('conditioner_config','conditioner_3d_config'):
        config.model.params[key].params.emb_models[0].params.model_config.params.open_clip_embedding_config.params.version=None
    config.model.params.en_and_decode_n_samples_a_time=1
    config.model.params.conditioner_config.params.emb_models[3].params.en_and_decode_n_samples_a_time=1
    model=instantiate_from_config(config.model).to(device='cuda',dtype=torch.bfloat16)
    model.requires_grad_(False)
    params=[];names=[]
    for k,p in model.named_parameters():
        if selected(k):p.data=p.data.float();p.requires_grad_(True);params.append(p);names.append(k)
    assert len(names)==80
    model.eval()
    return model,params,names

def loss(model,prepared,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    batch=move(default_collate([sample(*prepared,seed)]))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        y=model.get_input(batch)
        latent=model.encode_first_stage(y)
        latent3d=model.encode_first_stage(batch['jpg_3d'])
        value,_=model(latent,latent3d,batch)
    return value

def main(a):
    root=a.root;out=root/'training';out.mkdir(exist_ok=True)
    lock=open(out/'train.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else {}
    if previous.get('stage')=='complete':
        print('ALREADY_COMPLETE: no duplicate training',flush=True);return
    resume_steps=previous.get('steps',0)
    resume=(resume_steps//40)*40
    if resume_steps and not (out/f'optimizer_step_{resume:04d}.pt').exists():
        raise RuntimeError('partial run has no optimizer snapshot; preserve this run and register a new bounded run, never silently overwrite')
    torch.set_num_threads(4);random.seed(6201);torch.manual_seed(6201)
    catalog=json.loads((root/'dataset_catalog.json').read_text())
    train=[c for c in catalog['cases'] if c['split']=='train'];val=[c for c in catalog['cases'] if c['split']=='validation']
    prepared={};windows={}
    for c in catalog['cases']:
        starts=list(range(0,c['frame_count']-9,10));windows[c['dataset_id']]=starts
        for start in starts:prepared[(c['dataset_id'],start)]=resized(*arrays(c,start),a.size)
    # 真实数据、真实缩放合同，隐藏X变化不影响条件；Y变化只影响监督。
    y,x,h=arrays(train[0]);p=resized(y,x,h,a.size);alt=x.copy();alt[h]=255-alt[h]
    q=resized(y,alt,h,a.size);assert torch.equal(p[1],q[1])
    y2=y.copy();y2[h]=255-y2[h];q=resized(y2,x,h,a.size)
    assert torch.equal(p[1],q[1]) and not torch.equal(p[0],q[0])
    save_json(out/'input_contract.json',{'hidden_X_change_reaches_condition':False,'hidden_Y_change_reaches_condition':False,'Y_is_target_only':True,'clear_before_resize':True,'size':a.size,'num_frames':10,'architecture_channels':9,'added_channels':0,'actual_case':train[0]['dataset_id']})
    state={'stage':'loading_model','steps':resume,'resume_from_step':resume,'size':a.size,'train_cases':len(train),'validation_cases':len(val),'training_window_count':sum(len(windows[c['dataset_id']]) for c in train),'human_verdict':None}
    save_json(out/'state.json',state);model,params,names=model_init()
    save_json(out/'config.json',dict(state,base_checkpoint=str(OFFICIAL/'checkpoints/model.safetensors'),trainable_parameters=sum(p.numel() for p in params),trainable_tensors=names,lr=1e-5,weight_decay=.01,seed=6201,steps_requested=a.steps,optimizer='AdamW',loss='official StandardDiffusionLoss unchanged',checkpoint='torch nonreentrant recomputation only',dtype='bf16 frozen / fp32 trainables',validation_seeds=[911,912],training_resolution=list(a.size),main_self_attention_only=True))
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
    if resume:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(out/f'attention_step_{resume:04d}.safetensors')),strict=False)
        snapshot=torch.load(out/f'optimizer_step_{resume:04d}.pt',map_location='cpu')
        assert snapshot['size']==a.size and snapshot['steps']==resume
        optimizer.load_state_dict(snapshot['optimizer']);random.setstate(snapshot['random_state'])
        byid={c['dataset_id']:c for c in train};train=[byid[k] for k in snapshot['train_order']]
    torch.cuda.reset_peak_memory_stats();start=time.time()
    def evaluate(tag):
        rows=[]
        with torch.no_grad():
            for c in val:
                for seed in (911,912):
                    value=float(loss(model,prepared[(c['dataset_id'],0)],seed))
                    rows.append({'dataset_id':c['dataset_id'],'seed':seed,'loss':value})
        save_json(out/f'validation_{tag}.json',{'rows':rows,'mean':float(np.mean([v['loss'] for v in rows])),'scope':'4 scene-isolated cases, two fixed noise draws, official latent denoising loss; not rendered DELETE quality'})
    if not resume:evaluate('base')
    for step in range(resume,a.steps):
        if step%len(train)==0:random.shuffle(train)
        c=train[step%len(train)];window=windows[c['dataset_id']][(step//len(train))%len(windows[c['dataset_id']])]
        optimizer.zero_grad(set_to_none=True)
        value=loss(model,prepared[(c['dataset_id'],window)],6201+step)
        if not torch.isfinite(value):raise ValueError('nonfinite loss')
        value.backward()
        if step==0:
            gradients=[k for k,p in zip(names,params) if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0]
            assert len(gradients)==80,(len(gradients),'trainable gradient missing')
            save_json(out/'backward_probe.json',{'forward_backward_pass':True,'nonzero_finite_gradient_tensors':len(gradients),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'seconds_from_load':time.time()-start,'size':a.size})
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0));optimizer.step()
        state.update(stage='training',steps=step+1,last_case=c['dataset_id'],last_window=window,loss=float(value.detach()),grad_norm=norm,elapsed_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        save_json(out/'state.json',state)
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if (step+1)%40==0 or step+1==a.steps:
            from safetensors.torch import save_file
            save_file({k:p.detach().cpu().contiguous() for k,p in zip(names,params)},str(out/f'attention_step_{step+1:04d}.safetensors'))
            torch.save({'optimizer':optimizer.state_dict(),'steps':step+1,'size':a.size,'random_state':random.getstate(),'train_order':[c['dataset_id'] for c in train]},out/f'optimizer_step_{step+1:04d}.pt')
    evaluate('finetuned')
    state.update(stage='complete');save_json(out/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=160);p.add_argument('--size',type=int,nargs=2,default=[320,576]);a=p.parse_args()
    try:main(a)
    except Exception as e:
        save_json(a.root/'training/error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
