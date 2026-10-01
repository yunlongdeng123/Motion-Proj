"""真实Y→latent→同一预训练decoder：官方encoder对照随机初始化encoder。"""
from pathlib import Path
import json,sys,os,time
import torch,numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r7')
OFFICIAL=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
sys.path.insert(0,str(OFFICIAL));sys.path.insert(0,str(Path(__file__).parent))
from audit import protected_mask,dump
def main():
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
    from evaluate_control import inputs,metrics
    out=ROOT/'encoder_probe';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():print('already complete');return
    torch.set_num_threads(4);torch.manual_seed(6201)
    conf=OmegaConf.load(OFFICIAL/'configs/train.yaml').model.params.first_stage_config
    ae=instantiate_from_config(conf)
    random_state={k:v.clone() for k,v in ae.encoder.state_dict().items()}
    state={}
    with safe_open(str(OFFICIAL/'checkpoints/model.safetensors'),framework='pt') as f:
        for k in f.keys():
            if k.startswith('first_stage_model.'):state[k[len('first_stage_model.'):]]=f.get_tensor(k)
    state.update({k[len('first_stage_model.'):]:v for k,v in load_file(str(ROOT/'encoder_recovery/official_svd_encoder.safetensors')).items()})
    ae.load_state_dict(state,strict=True);del state
    correct_state={k:v.clone() for k,v in ae.encoder.state_dict().items()}
    ae=ae.cuda().eval();ae.requires_grad_(False)
    plan=json.loads((ROOT/'evaluation_plan.json').read_text());case=next(c for c in plan['cases'] if c['eval_id']=='holdout_R014')
    x,y,h,b=inputs(case);rows=[]
    for size in ((320,576),(576,1024)):
        yy=torch.from_numpy(np.stack(y)).permute(0,3,1,2).float().cuda()/127.5-1
        yy=torch.nn.functional.interpolate(yy,size=size,mode='bilinear',align_corners=False,antialias=True)
        target=((yy.permute(0,2,3,1)+1)*127.5).round().clamp(0,255).byte().cpu().numpy()
        holes=torch.nn.functional.interpolate(torch.from_numpy(np.stack(h)[:,None].astype('float32')),size=size,mode='nearest')[:,0].numpy()>0
        protect=torch.nn.functional.interpolate(torch.from_numpy(np.stack(b)[:,None].astype('float32')),size=size,mode='nearest')[:,0].numpy()>0
        for arm,enc in [('random_encoder_control',random_state),('official_SVD_encoder',correct_state)]:
            ae.encoder.load_state_dict(enc,strict=True);dest=out/(f'{size[0]}_'+arm);dest.mkdir(exist_ok=True)
            images=[];zs=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                for i,frame in enumerate(yy):
                    # 同mode移除posterior采样差异，训练仍保留官方sample=True。
                    moments=ae.encoder(frame[None]);z=DiagonalGaussianDistribution(moments).mode();image=ae.decode(z,timesteps=1)
                    images.append(((image[0].permute(1,2,0).float()+1)*127.5).round().clamp(0,255).byte().cpu().numpy());zs.append(float(z.float().std()))
            for i,image in enumerate(images):Image.fromarray(image).save(dest/f'{i:05}.png')
            for i,frame in enumerate(target):
                gt=out/f'{size[0]}_GT';gt.mkdir(exist_ok=True);Image.fromarray(frame).save(gt/f'{i:05}.png')
            scores=[metrics(im,gt,hh,bb) for im,gt,hh,bb in zip(images,target,holes,protect)]
            row={'size':size,'arm':arm,'latent_std_mean':float(np.mean(zs)),'scores':scores,'mean_protected_MAE':float(np.mean([r['protected_inside_hole']['MAE'] for r in scores]))}
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='scores'}),flush=True)
    dump(out/'results.json',{'case':case,'rows':rows,'architecture_unchanged':True,'strict_restore':True,
                             'purpose':'oracle codec diagnostic using clean Y; never query input or generated GT',
                             'random_control':'new seeded random initialization of same Encoder, not byte-verified recovery of unsaved r6 random weights',
                             'decode_timesteps':1,'posterior':'mode for controlled comparison; training sample unchanged'})
if __name__=='__main__':main()
