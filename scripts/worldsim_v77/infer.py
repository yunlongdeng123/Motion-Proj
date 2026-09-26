"""冻结官方Ω；RGB-only前向。GT标定与LiDAR只在后续读出控制中读取。"""
import argparse, json, pathlib, sys, time
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
    root=pathlib.Path(args.run_dir);reg=json.loads((root/'registration.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(reg['seed']);assert torch.cuda.is_available()
    sys.path.insert(0,reg['omega_source'])
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    with torch.device('meta'):model=VGGTOmega()
    model.load_state_dict(torch.load(reg['checkpoint'],map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True)
    model.requires_grad_(False);model=model.eval().cuda()
    print('MODEL_READY',flush=True)
    for scene in reg['scenes']:
        out=root/scene['name'];out.mkdir(exist_ok=True)
        assert not (out/'prediction.npz').exists(), '禁止覆盖原始预测'
        images=load_and_preprocess_images([v['image'] for v in scene['views']],image_resolution=512,mode='balanced').cuda()
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.inference_mode():
            pred=model(images);e,k=encoding_to_camera(pred['pose_enc'],images.shape[-2:])
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        arrays={name:pred[name].float().cpu().numpy() for name in ['depth','depth_conf','pose_enc','camera_and_register_tokens']}
        arrays.update(extrinsics=e.float().cpu().numpy(),intrinsics=k.float().cpu().numpy(),images=images.float().cpu().numpy())
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(out/'prediction.npz',**arrays)
        report={'scene':scene['name'],'runtime_s':elapsed,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'frozen':True,'training_steps':0,
                'rgb_views':6,'precision':'official internal bf16 autocast','network_hw':list(images.shape[-2:]),'finite':True,
                'shapes':{k:list(v.shape) for k,v in arrays.items()}}
        (out/'inference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        del arrays,pred,images;torch.cuda.empty_cache()

if __name__=='__main__':main()
