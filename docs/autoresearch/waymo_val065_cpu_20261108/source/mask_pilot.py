"""Prepare or run a four-frame SAM2 input-mask pilot. Default mode is CPU preflight.

GPU execution requires --run-gpu. It writes per-track mask candidates for source
images only, to be reviewed before use as scene-fitting supervision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sam-root',type=Path,default=Path('/root/autodl-tmp/third_party/worldsim_v32/sam2'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--raws',type=int,nargs='+',default=[130,155,160,190])
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--point-prompts',action='store_true')
    args=parser.parse_args()
    receipt=json.loads((args.bank/'receipt.json').read_text(encoding='utf8'))
    assert receipt['status']=='PASS_CPU_OBSERVATIONS'
    assert set(args.raws)<=set(receipt['build_raws'])
    assert not set(args.raws)&set(receipt['holdout_raws'])
    config='configs/sam2.1/sam2.1_hiera_l.yaml'
    checkpoint=args.checkpoint or args.sam_root/'checkpoints/sam2.1_hiera_large.pt'
    assert checkpoint.is_file() and (args.sam_root/'sam2'/config).is_file()
    by_raw={f['raw']:f for f in receipt['frames']}
    manifest=dict(schema='waymo.sam2.mask_pilot.v1',status='CPU_PREFLIGHT_PASS',
        bank=str(args.bank),raws=args.raws,checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        config=config,model_source=str(args.sam_root),CUDA_executed=False,training_updates=0,
        input_policy='independent image prediction, all projected visible vehicle boxes, same predictor settings for every object',
        mask_policy='single mask per projected 3D box; candidates require silhouette and depth/identity review before use',
        point_prompts=args.point_prompts,
        method_change='source input preparation only; no 3D reconstruction or deletion quality claim',
        source_sha256=digest(Path(__file__)),frames=[])
    for raw in args.raws:
        frame=by_raw[raw]
        path=args.bank/'image/image_0'/f'{frame["id"]}.png'
        assert path.is_file()
        prompts=json.loads((args.bank/'prompts'/f'{raw:03d}.json').read_text(encoding='utf8'))
        manifest['frames'].append(dict(raw=raw,image=str(path),image_sha256=digest(path),prompt_count=len(prompts)))
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'receipt.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.run_gpu:
        print(json.dumps(manifest),flush=True)
        return
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(args.sam_root))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference requested, CUDA is unavailable')
    torch.manual_seed(0)
    start=time.monotonic()
    manifest.update(status='GPU_MASK_PILOT_RUNNING',CUDA_executed=True,checkpoint_sha256=digest(checkpoint))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    predictor=SAM2ImagePredictor(build_sam2(config,str(checkpoint),device='cuda'))
    results=[]
    with torch.inference_mode():
        for frame in manifest['frames']:
            image=np.array(Image.open(frame['image']).convert('RGB'))
            predictor.set_image(image)
            prompts=json.loads((args.bank/'prompts'/f'{frame["raw"]:03d}.json').read_text(encoding='utf8'))
            point_prompts=json.loads((args.bank/'point_prompts'/f'{frame["raw"]:03d}.json').read_text()) if args.point_prompts else {}
            folder=args.output/f'{frame["raw"]:03d}';folder.mkdir()
            for prompt in prompts:
                points=point_prompts.get(prompt['track_id'],{})
                positive=points.get('positive',[]);negative=points.get('negative',[])
                if args.point_prompts and not positive:
                    mask=np.zeros(image.shape[:2],dtype=bool);score=0.
                else:
                    coords=np.array(positive+negative) if positive else None
                    labels=np.array([1]*len(positive)+[0]*len(negative)) if positive else None
                    masks,scores,_=predictor.predict(box=np.array(prompt['bbox_xyxy_native']),point_coords=coords,point_labels=labels,multimask_output=False)
                    mask=np.asarray(masks[0],dtype=bool);score=float(scores[0])
                output=folder/f'{prompt["alias"]}.png'
                Image.fromarray(mask.astype(np.uint8)*255).save(output)
                results.append(dict(raw=frame['raw'],track_id=prompt['track_id'],alias=prompt['alias'],
                    mask=str(output.relative_to(args.output)),pixels=int(mask.sum()),score=score,point_support=points,
                    bbox=prompt['bbox_xyxy_native'],quality_accepted=None))
            print(json.dumps(dict(raw=frame['raw'],masks=len(prompts))),flush=True)
    manifest.update(status='GPU_MASK_PILOT_RENDERED_AWAITING_QA',results=results,
                    seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated())
    path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
