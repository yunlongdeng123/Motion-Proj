"""先零训练验证，独立review后至多一次64步；CPU不会加载官方模型。"""
from common import *
import argparse, fcntl, random, time, gc
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from PIL import Image
from interface import encode_references, drop_priors
from multi_prior import MultiPriorBranch
from routing import mismatch_pairing
from iteration15.gpu_experiment import snapshot


def require_gpu():
    if not torch.cuda.is_available():raise RuntimeError('等待用户开GPU；本入口不在CPU加载模型或后台等卡')
    if not read(O/'preflight.json')['CPU_ready']:raise RuntimeError('CPU对应未准入')
    torch.set_num_threads(1)


def lock_folder(path):
    path.mkdir(parents=True,exist_ok=True);handle=open(path/'lock','a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);return handle


def load_branch(path):
    result=MultiPriorBranch().cuda();result.load_state_dict(load_file(str(path)),strict=True)
    assert all(torch.isfinite(p).all() for p in result.parameters());return result


def feature_mismatch(branch,recipient,donor,recipient_latent,donor_latent):
    """只换双方有效且一对一配对的主B patch；query语义、pose、valid不动。"""
    left=read(O/'inputs'/recipient/'routing.json');right=read(O/'inputs'/donor/'routing.json')
    with np.load(O/'inputs'/recipient/'routing.npz') as data:left_arrays={k:data[k] for k in data.files}
    with np.load(O/'inputs'/donor/'routing.npz') as data:right_arrays={k:data[k] for k in data.files}
    pairs,paired=mismatch_pairing(left_arrays,left,right_arrays,right)
    with torch.no_grad():
        lf=F.adaptive_avg_pool2d(branch.reference_encoder(recipient_latent.float()),(8,8)).permute(0,2,3,1)
        rf=F.adaptive_avg_pool2d(branch.reference_encoder(donor_latent.float()),(8,8)).permute(0,2,3,1)
        lf=lf.reshape(384,branch.width);rf=rf.reshape(384,branch.width);delta=torch.zeros_like(lf)
        li=torch.tensor([a for a,b in pairs],device='cuda');ri=torch.tensor([b for a,b in pairs],device='cuda')
        delta[li]=rf[ri]-lf[li]
    evidence={'recipient':recipient,'donor':donor,**paired,'patch_pairs':pairs,
        'query_ID_pose_geometry_valid_instruction_fixed':True,'context_and_other_B_features_fixed':True,
        'swap_stage':'reference_encoder pooled8x8 features before slot pose',
        'scope':'有限覆盖的错配响应，不等于整主体身份替换'}
    return delta,evidence


def infer(engine,req,cid,folder,latent,label,*,routing=True,appearance_delta=None,arm='RGB_and_geometry',mismatch=None):
    from repair_drive import set_seed
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    plan=read(O/'manifest.json')
    engine.get_deletion(req,arm=arm,reference_latents=latent,appearance_delta=appearance_delta,routing=routing)
    set_seed(plan['eval_seed']);started=time.monotonic()
    with torch.inference_mode():engine.predict(1,False,'Deletion')
    assert engine.prior_cfg_calls=={'unconditional':25,'conditional':25}
    raw=np.stack(engine.im_result);assert raw.shape==req.target_rgb.shape and raw.dtype==np.uint8
    final=req.compose(raw);(folder/'native').mkdir(exist_ok=True)
    for f in range(10):
        Image.fromarray(raw[f]).save(folder/'native'/f'{f:05}.png')
        Image.fromarray(final[f]).save(folder/f'{f:05}.png')
    result={'case_id':cid,'condition':label,'arm':arm,'routing_enabled':routing,
        'seed':plan['eval_seed'],'sampling_steps':plan['eval_steps'],'seconds':time.monotonic()-started,
        'CFG_prior_calls':engine.prior_cfg_calls,'C_parameters':engine._conditioned_priors['parameters'][0].tolist(),
        'UC_parameters':engine._unconditional_priors['parameters'][0].tolist(),
        'module_response':engine.branch.response_rows,'routing_response':engine.branch.routing_rows,
        'mismatch':mismatch,'outside_H_exact':True,'human_verdict':None,'temporal_verdict':None}
    assert result['C_parameters']==req.priors['parameters'][0].tolist()
    assert result['UC_parameters']==[0.,0.,0.,0.]
    dump(folder/'result.json',result);return result


def evaluate(stage):
    assert stage in ('zero','step64')
    checkpoint=INITIAL if stage=='zero' else O/'training/branch_0064.safetensors'
    out=O/'evaluation'/stage;handle=lock_folder(out);byid=cases()
    from engine import MultiPriorDeletionEngine
    engine=MultiPriorDeletionEngine(load_branch(checkpoint));cache={};results=[]
    for cid in DIAGNOSTIC_IDS:
        req=request(byid[cid]);cache[cid]=(req,encode_references(engine.model,req))
    for cid in EVAL_IDS:
        req,latent=cache[cid] if cid in cache else (request(byid[cid]),None)
        if latent is None:latent=encode_references(engine.model,req)
        results.append(infer(engine,req,cid,out/cid/'correct',latent,'r49_'+stage+'_correct'))
        if cid in DIAGNOSTIC_IDS:
            donor=DIAGNOSTIC_IDS[1] if cid==DIAGNOSTIC_IDS[0] else DIAGNOSTIC_IDS[0]
            delta,meta=feature_mismatch(engine.branch,cid,donor,latent,cache[donor][1])
            results.append(infer(engine,req,cid,out/cid/'wrong',latent,'r49_'+stage+'_wrong',appearance_delta=delta,mismatch=meta))
            results.append(infer(engine,req,cid,out/cid/'no_RGB',latent,'r49_'+stage+'_no_RGB',arm='geometry_only'))
            if stage=='step64':results.append(infer(engine,req,cid,out/cid/'routing_off',latent,'r49_step64_routing_off',routing=False))
        dump(out/'state.json',{'stage':'running','checkpoint':str(checkpoint),'completed':results})
        print('EVAL',stage,cid,flush=True);gc.collect()
    assert len(results)==(9 if stage=='zero' else 11)
    dump(out/'state.json',{'stage':'complete','checkpoint':str(checkpoint),'completed':results,
        'human_verdict':None,'scope':'3真实DEV+2已知GT合成DEV的受控窗口，不是新scene泛化'})


def train64():
    # GPU零训练已完成、实际图像review过后才允许适配，不根据一个proxy误差自动训练。
    gate=read(O/'zero_shot_gate.json') if (O/'zero_shot_gate.json').exists() else None
    if gate is None or gate.get('decision')!='allow_one_64_step_adaptation':
        raise RuntimeError('先完成零训练图像review并记录唯一64步准入，禁止自动追加训练')
    assert read(O/'evaluation/zero/state.json')['stage']=='complete'
    out=O/'training';handle=lock_folder(out);plan=read(O/'manifest.json');byid=cases()
    resume=read(out/'resume.json') if (out/'resume.json').exists() else {'step':0}
    start=resume['step'];assert start<=64
    if start==64:return
    import train_control as train
    random.seed(plan['train_seed']);torch.manual_seed(plan['train_seed'])
    train.ENCODER=T/'r7/encoder_recovery/official_svd_encoder.safetensors'
    model,_,_=train.model_init();model.requires_grad_(False);model.eval()
    branch=load_branch(Path(resume['branch']) if start else INITIAL)
    model.model.diffusion_model.multi_prior_branch=branch;branch.attach(model.model.diffusion_model)
    optimizer=torch.optim.AdamW(branch.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    cache={};order=list(range(len(TRAIN_IDS)))
    for cid in TRAIN_IDS:
        req=request(byid[cid]);cache[cid]=(train.resized(images(byid[cid],'target'),req.target_rgb,req.edit_mask,plan['train_size']),
            req.branch_condition(encode_references(model,req),'cpu'))
    if start:
        saved=torch.load(resume['optimizer'],map_location='cpu',weights_only=False)
        assert saved['step']==start and saved['initial_checkpoint']==str(INITIAL)
        optimizer.load_state_dict(saved['optimizer']);order=saved['order']
        random.setstate(saved['random_state']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    dump(out/'config.json',{'train_ids':TRAIN_IDS,'scene_count':4,'seed':plan['train_seed'],'lr':plan['lr'],
        'resolution':plan['train_size'],'max_steps':64,'new_branch_parameters':389856,'new_route_parameters':0,
        'main_3D_VAE_frozen':True,'loss':'unchanged official diffusion loss','initial_checkpoint':str(INITIAL),
        'data_order_and_dropout_match_r48_64':True,'prior_dropout':plan['prior_dropout'],
        'encoder_restore':model.encoder_restore_evidence,'true_DEV_not_training':True})
    started=time.monotonic();attempt=str(time.time_ns());torch.cuda.reset_peak_memory_stats()
    for step in range(start,64):
        if step%len(order)==0:random.shuffle(order)
        cid=TRAIN_IDS[order[step%len(order)]];prepared,condition=cache[cid]
        drgb=step>=2 and random.random()<.25;dgeo=step>=2 and random.random()<.25
        branch.set_condition(**{k:v.cuda() for k,v in drop_priors(condition,drgb,dgeo).items()})
        assert torch.equal(branch._condition['parameters'].cpu(),condition['parameters'])
        optimizer.zero_grad(set_to_none=True);value=train.loss(model,prepared,plan['train_seed']+step)
        if not torch.isfinite(value):raise ValueError('非有限loss，停止本轮')
        value.backward()
        assert all(p.grad is None for n,p in model.named_parameters() if '.multi_prior_branch.' not in n)
        norm=float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1,error_if_nonfinite=True));optimizer.step()
        assert all(torch.isfinite(p).all() for p in branch.parameters())
        row={'stage':'training','steps':step+1,'case':cid,'loss':float(value.detach()),'grad_norm':norm,
            'drop_RGB':drgb,'drop_geometry':dgeo,'instruction_preserved':True,'until':64,'resume_step':start,
            'attempt':attempt,'elapsed_seconds':time.monotonic()-started,
            'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'main_frozen':True}
        dump(out/'state.json',row)
        with (out/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if (step+1)%16==0:snapshot(branch,optimizer,step+1,order,out)
        print('TRAIN',step+1,cid,row['loss'],flush=True)
    dump(out/'state.json',row|{'stage':'checkpoint_ready'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['zero','train64','evaluate64'])
    args=p.parse_args();require_gpu()
    if args.action=='zero':evaluate('zero')
    elif args.action=='train64':train64()
    else:evaluate('step64')
