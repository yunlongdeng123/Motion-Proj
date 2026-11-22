"""Frozen road-class proxy on original teacher images; no diffusion or fitting."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
sys.path[:0]=['/root/autodl-tmp/external/dggt_segformer','/root/autodl-tmp/envs/dggt_segformer/overlay']
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-labels',type=Path);args=p.parse_args()
    import numpy as np,torch
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from mmseg.apis import init_segmentor,inference_segmentor
    args.output.mkdir(parents=True,exist_ok=False)
    config=Path('/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py')
    weights=Path('/root/autodl-tmp/drivingeditor_repro/weights/segformer.b5.1024x1024.city.160k.pth')
    assert sha(weights)=='daa9f3500a32ce17d4ce1a89ee91b01a87c2179d0203d9394dc4e5ce526f15b4'
    model=None;torch.set_num_threads(2);rows=[];started=time.time();calls=0
    for source in json.loads(args.sources.read_text())['sources']:
        path=Path(source['raw_model']['path']);assert sha(path)==source['raw_model']['sha256']
        previous=(args.reuse_labels/f"{source['raw']:03d}/labels_model_Brown.png") if args.reuse_labels else None
        reused=previous is not None and previous.is_file()
        if reused:labels=np.asarray(Image.open(previous),np.uint8)
        else:
            if model is None:model=init_segmentor(str(config),str(weights),device='cuda:0')
            with torch.inference_mode():labels=inference_segmentor(model,str(path))[0].astype(np.uint8)
            calls+=1
        assert labels.shape==(768,1152)
        road=binary_erosion(labels==0,iterations=2);folder=args.output/f"{source['raw']:03d}";folder.mkdir()
        Image.fromarray(labels).save(folder/'labels_model_Brown.png')
        Image.fromarray(road.astype(np.uint8)*255).save(folder/'road_eroded_model_Brown.png')
        image=np.asarray(Image.open(path).convert('RGB')).copy();image[~road]=np.rint(.45*image[~road]+.55*np.asarray([240,48,64])).astype(np.uint8)
        Image.fromarray(image).save(folder/'road_proxy_overlay.png')
        rows.append(dict(raw=source['raw'],source_sha256=sha(path),road_pixels=int(road.sum()),car_pixels=int((labels==13).sum()),
            saved_prediction_reused=reused,reused_prediction_sha256=sha(previous) if reused else None,
            labels_sha256=sha(folder/'labels_model_Brown.png'),road_mask_sha256=sha(folder/'road_eroded_model_Brown.png')))
    receipt=dict(complete=True,source_manifest_sha256=sha(args.sources),checkpoint_sha256=sha(weights),config_sha256=sha(config),
        existing_model='SegFormer B5 Cityscapes',coordinate_domain='original model1152 Brown pixel grid',road_class=0,erosion_pixels=2,
        proxy_not_semantic_ground_truth=True,semantic_image_inference_calls=calls,previous_completed_inference_calls_reused=len(rows)-calls,new_diffusion_calls=0,nativeF=0,nativeB=0,optimizer_updates=0,
        rows=rows,seconds=time.time()-started)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
