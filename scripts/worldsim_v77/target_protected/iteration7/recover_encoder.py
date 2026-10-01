"""从官方SVD safetensors只读取目标encoder的byte ranges，禁止下载整模型。"""
from pathlib import Path
import argparse,json,struct,time
import requests,numpy as np,torch
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r7')
URL='https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors'
BASE=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors')
def main(a):
    out=ROOT/'encoder_recovery';out.mkdir(exist_ok=True)
    if (out/'official_svd_encoder.safetensors').exists():print('encoder already downloaded');return
    s=requests.Session();s.proxies={'http':a.proxy,'https':a.proxy}
    def part(lo,hi):
        with s.get(URL+f'?download=true&segment={lo}-{hi}',headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(20,90)) as r:
            assert r.status_code==206,(r.status_code,r.headers.get('content-length'))
            assert r.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'),r.headers.get('Content-Range')
            b=b''.join(r.iter_content(1024*1024));assert len(b)==hi-lo+1
            return b
    n=struct.unpack('<Q',part(0,7))[0];header=json.loads(part(8,7+n))
    (out/'official_header.json').write_text(json.dumps(header,indent=2)+'\n')
    selected={k:v for k,v in header.items() if k.startswith('first_stage_model.encoder.')}
    assert len(selected)==106,(len(selected),'official target encoder keys differ')
    lo=min(v['data_offsets'][0] for v in selected.values());hi=max(v['data_offsets'][1] for v in selected.values())
    print(json.dumps({'header_bytes':n,'tensor_count':len(selected),'encoder_download_bytes':hi-lo}),flush=True)
    raw=part(8+n+lo,8+n+hi-1);weights={}
    dtypes={'F32':np.float32,'F16':np.float16}
    for k,v in selected.items():
        aa,bb=v['data_offsets'];weights[k]=torch.from_numpy(np.frombuffer(raw[aa-lo:bb-lo],dtype=dtypes[v['dtype']]).copy().reshape(v['shape']))
    save_file(weights,str(out/'official_svd_encoder.safetensors'))
    # conditioner encoder与目标encoder不可未经验证就互换；实际比较官方张量。
    comparisons=[]
    with safe_open(str(BASE),framework='pt') as f:
        keys=set(f.keys());missing=[k for k in weights if k not in keys]
        for k,v in weights.items():
            ck=k.replace('first_stage_model.encoder.','conditioner.embedders.3.encoder.encoder.')
            if ck not in keys:continue
            cv=f.get_tensor(ck)
            if cv.shape==v.shape:comparisons.append({'target_key':k,'condition_key':ck,'max_abs_difference':float((v.float()-cv.float()).abs().max())})
    report={'source_url':URL,'official_encoder_tensors':len(weights),'missing_target_keys_in_DriveEditor':len(missing),
            'download_bytes':len(raw),'method':'HTTP206 partial byte ranges, official checkpoint tensor names/shapes unchanged',
            'condition_encoder_comparison':comparisons,'target_encoder_uses_quant_conv':any('quant_conv' in k for k in weights),
            'first_stage_target':'original train.yaml Encoder + DiagonalGaussianRegularizer, restored from original SVD; no extra architecture'}
    (out/'recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='condition_encoder_comparison'},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proxy',required=True);main(p.parse_args())
