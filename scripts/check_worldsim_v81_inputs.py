"""CPU 上用官方预处理读取真实首个窗口；不构造模型或启动 GPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,subprocess,importlib.util,sys
from pathlib import Path
import torch
torch.set_num_threads(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    repo=Path('/root/autodl-tmp/motion_proj');spec=importlib.util.spec_from_file_location('v81_infer',repo/'scripts/run_worldsim_v81_inference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    results=[]
    for pilot in json.loads((run/'gpu_pilot_commands.json').read_text()):
        method=pilot['method'];cmd=pilot['argv'];dry=subprocess.run(cmd,cwd=repo,text=True,capture_output=True,check=True)
        manifest=json.loads(Path(cmd[cmd.index('--manifest')+1]).read_text());views=manifest['views']
        loaderfile=mod.REPOS[method]/('dvgt' if method=='dvgt' else 'vggt' if method=='vggt' else 'dggt')/'utils/load_fn.py'
        spec=importlib.util.spec_from_file_location('native_loader_'+method,loaderfile);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        if method=='dvgt':
            temp=run/'native_preprocess_probe'/'frame_0';temp.mkdir(parents=True,exist_ok=True)
            for i,v in enumerate(views):
                dest=temp/f"{i:02d}_{v['camera']}.jpg"
                if not dest.exists():dest.symlink_to(v['image'])
            data=loader.load_and_preprocess_images(str(temp.parent),mode='crop')
            expected=[1,1,6,3,288,512]
        else:
            data=loader.load_and_preprocess_images([v['image'] for v in views],mode='crop');expected=[6,3,294,518]
        assert list(data.shape)==expected,(method,list(data.shape))
        assert data.isfinite().all() and data.min()>=0 and data.max()<=1
        results.append({'method':method,'dry_plan':json.loads(dry.stdout),'official_preprocess_shape':list(data.shape),'pixel_range':[float(data.min()),float(data.max())],'status':'PASS','model_forward':False,'cuda':False})
        del data
    (run/'cpu_input_contracts.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
if __name__=='__main__':main()
