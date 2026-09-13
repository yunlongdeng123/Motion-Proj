"""无GPU、meta device 严格验证官方模型参数合同，不执行模型前向。"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,resource,time,importlib.metadata as md,struct
from pathlib import Path
import torch
# 仅检查脚本：DINO构造器将小型drop-path日程调用item，需在CPU生成此日程。
# 大型参数仍在meta；正式推理入口不包含这个替换。
native_linspace=torch.linspace
def cpu_schedule(*args,**kwargs):
    kwargs['device']='cpu'
    return native_linspace(*args,**kwargs)
torch.linspace=cpu_schedule
p=argparse.ArgumentParser();p.add_argument('--method',required=True);a=p.parse_args();model=a.method
ROOT=Path('/root/autodl-tmp');run=ROOT/'runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r1';start=time.time()
if model=='dvgt':
    repo=ROOT/'external/worldsim_v81/DVGT';sys.path.insert(0,str(repo));os.chdir(repo)
    from dvgt.models.architectures.dvgt1 import DVGT1
    original=torch.hub.load
    def load(*args,**kwargs):
        if args and 'dinov3' in str(args[0]):kwargs['pretrained']=False;kwargs.pop('weights',None)
        return original(*args,**kwargs)
    torch.hub.load=load
    with torch.device('meta'):net=DVGT1(dino_v3_weight_path=None,frames_chunk_size=1)
    torch.hub.load=original
    state=torch.load(ROOT/'models/worldsim_v81/dvgt1.pt',map_location='meta',weights_only=True,mmap=True)
elif model=='vggt':
    repo=ROOT/'external/worldsim_v72/vggt';sys.path.insert(0,str(repo))
    from vggt.models.vggt import VGGT
    with torch.device('meta'):net=VGGT()
    path=ROOT/'models/eas_vggt/vggt/model.safetensors'
    with path.open('rb') as f:length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
    state={k:torch.empty(v['shape'],device='meta') for k,v in header.items() if k!='__metadata__'}
else:
    repo=ROOT/'external/worldsim_v81/dggt';sys.path.insert(0,str(repo));os.chdir(repo)
    from dggt.models.vggt import VGGT
    with torch.device('meta'):net=VGGT()
    state=torch.load(ROOT/'models/worldsim_v81/model_latest_nuscenes.pt',map_location='meta',weights_only=True,mmap=True)
net.load_state_dict(state,strict=True)
result={'method':model,'status':'STRICT_META_LOAD_PASS','parameter_tensors':len(state),'parameter_elements':sum(p.numel() for p in net.parameters()),'torch':torch.__version__,'cuda_visible':torch.cuda.is_available(),'inference_performed':False,'elapsed_s':time.time()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
(run/('meta_'+model+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
