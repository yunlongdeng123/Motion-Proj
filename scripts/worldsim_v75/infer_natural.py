"""真实观测经官方DVGT-1前向；单卡OOM立即停，保存原生输出。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-01/20260920-r1')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    OUT=parser.parse_args().run_dir
    protocol=json.loads((OUT/'protocol.json').read_text())
    path=OUT/'inference_result.json';assert not path.exists()
    result={'status':'started','human_verdict':None,'model':'official DVGT-1','world_model_generation_calls':0}
    start=time.monotonic()
    try:
        torch.set_num_threads(4);torch.manual_seed(7501)
        os.chdir(protocol['repository']);sys.path.insert(0,protocol['repository'])
        from dvgt.models.architectures.dvgt1 import DVGT1
        from dvgt.utils.load_fn import load_and_preprocess_images
        original=torch.hub.load
        def hub(*args,**kwargs):
            if args and 'dinov3' in str(args[0]):
                kwargs['pretrained']=False;kwargs.pop('weights',None)
            return original(*args,**kwargs)
        torch.hub.load=hub
        try: model=DVGT1(dino_v3_weight_path=None,frames_chunk_size=1)
        finally: torch.hub.load=original
        state=torch.load(protocol['weights'],map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(state,strict=True);del state
        model=model.eval().cuda()
        inputs=load_and_preprocess_images(str(OUT/'native_input'),mode='crop').cuda()
        assert list(inputs.shape)==[1,1,7,3,512,512]
        print(json.dumps({'stage':'forward','shape':list(inputs.shape)}),flush=True)
        torch.cuda.reset_peak_memory_stats();began=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(inputs)
        torch.cuda.synchronize()
        arrays={k:pred[k].float().cpu().numpy() for k in ['points','points_conf','absolute_ego_pose_enc']}
        assert all(np.isfinite(x).all() for x in arrays.values())
        np.savez(OUT/'native_outputs.npz',**arrays)
        np.save(OUT/'network_rgb.npy',(inputs[0,0].permute(0,2,3,1)*255).round().byte().cpu().numpy())
        result.update(status='complete',shape=list(inputs.shape),forward_s=time.monotonic()-began,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      output_shapes={k:list(v.shape) for k,v in arrays.items()},finite=True,
                      strict_weights=True,official_source_changed=False)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        result['wall_s']=time.monotonic()-start
        path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__': main()
