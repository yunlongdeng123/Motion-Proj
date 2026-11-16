"""Prepared SSISv2 association baseline. Default is CPU preflight ONLY.

The --run-gpu switch is never used in the CPU preparation task. Actual execution
still needs a separately started GPU runtime with verified Detectron2/adet ops.
"""
from pathlib import Path
import argparse,json,hashlib,sys,subprocess
from dataclasses import replace
import numpy as np
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',required=True)
    parser.add_argument('--target-mask',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--third-party',default='/root/autodl-tmp/third_party/SSIS_shadow_provider_20261115')
    parser.add_argument('--repository',default='/root/autodl-tmp/motion_proj_v82_gn11_v42')
    parser.add_argument('--out',required=True)
    parser.add_argument('--run-gpu',action='store_true')
    args=parser.parse_args();source=Path(args.third_party)
    out=Path(args.out)
    if args.run_gpu and out.exists() and any(out.iterdir()):
        raise FileExistsError('keep old provider results; use a new output directory')
    config=source/'configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml'
    rgb=np.asarray(Image.open(args.image).convert('RGB')).copy()
    target=np.asarray(Image.open(args.target_mask).convert('L'))>127
    if target.shape!=rgb.shape[:2] or not target.any():raise ValueError('wrong actual target mask camera grid')
    if np.all(rgb[target]==127):raise ValueError('do not use old gray-core pseudo-hole RGB as an object observation')
    if not config.is_file() or not Path(args.checkpoint).is_file():raise FileNotFoundError('provider source/config/weights not ready')
    receipt=dict(stage='CPU_preflight_only',image=args.image,target_mask=args.target_mask,shape=list(target.shape),
                 source=str(source),config=str(config),checkpoint=args.checkpoint,checkpoint_sha256=sha(args.checkpoint),
                 config_sha256=sha(config),
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
                 soft_export_source_sha256=sha(source/'adet/modeling/ssis/condinst.py'),
                 model_inference_executed=False,provider='SSISv2 paired-instance baseline, not MetaShadow',
                 requires_GPU_runtime_and_real_smoke=True)
    mapping_source=source/'adet/data/dataset_mapper.py'
    mapping_sha=sha(mapping_source)
    mapping_verified=mapping_sha=='f729d8d2aa256770e71007e2cedaa7e3e58070d9606fa703e046b8e14078a2f3'
    receipt['class_role_evidence']=dict(source=str(mapping_source),sha256=mapping_sha,
        verified=mapping_verified,object_class_id=0,shadow_class_id=1,
        evidence='official training augmentation code; category_id 0=obj_mask, 1=sha_mask',
        annotation_categories_independently_verified=False)
    if not args.run_gpu:
        print(json.dumps(receipt,ensure_ascii=False));return
    if not mapping_verified:
        raise RuntimeError('official object/shadow class mapping differs; verify its semantics before inference')
    # This boundary requires GPU authorization; no model imports above it.
    sys.path.insert(0,str(source));sys.path.insert(0,str(Path(args.repository)))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU runtime unavailable; CPU preflight does not start or switch AutoDL power mode')
    from adet.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from scripts.worldsim_v82.road_material_field.input_construction.shadow_probability import from_ssis_instance_packet,save_shadow_field
    cfg=get_cfg();cfg.merge_from_file(str(config));cfg.MODEL.WEIGHTS=args.checkpoint;cfg.MODEL.DEVICE='cuda'
    cfg.DATASETS.TRAIN=();cfg.DATASETS.TEST=();cfg.freeze()
    predictor=DefaultPredictor(cfg)
    resized=predictor.aug.get_transform(rgb[:,:,::-1]).apply_image(rgb[:,:,::-1])
    mh,mw=resized.shape[:2];oh,ow=rgb.shape[:2]
    receipt['image_preprocessing']=dict(input_format=cfg.INPUT.FORMAT,model_image_hw=[mh,mw],
        native_from_model_pixel_edges=[[ow/mw,0,0],[0,oh/mh,0],[0,0,1]],
        probability_export='official original-size postprocess soft masks, before binarization')
    with torch.inference_mode():output=predictor(rgb[:,:,::-1].copy())
    # Raw model returns (processed_results, processed_associations), whereas
    # Detectron2 DefaultPredictor may already take the tuple's first element.
    if isinstance(output,tuple) and len(output)==2:results=output[0]
    elif isinstance(output,(list,dict)):results=output
    else:raise RuntimeError('SSIS predictor output contract differs')
    item=results[0] if isinstance(results,(list,tuple)) else results
    inst=item.get('instances')
    if inst is None:
        soft=np.empty((0,*target.shape),np.float32)
        classes=np.empty(0,np.int64);associations=np.empty(0,np.int64);scores=np.empty(0,np.float32)
    else:
        inst=inst.to('cpu')
        if not inst.has('pred_soft_masks'):raise RuntimeError('soft-map export hook is missing; binary masks cannot stand in for probabilities')
        soft=inst.pred_soft_masks.numpy();classes=inst.pred_classes.numpy()
        associations=np.asarray(inst.pred_associations,dtype=np.int64);scores=inst.scores.numpy()
    # Official SSIS dataset_mapper encodes object=0, associated shadow=1.
    # A shadow overlapping a broad target mask must never reverse these roles.
    field=from_ssis_instance_packet(rgb,target,soft,classes,associations,scores,
        model_id='official_SSISv2_9a1cac2_soft_export',
        checkpoint_sha256=receipt['checkpoint_sha256'],object_class_id=0,shadow_class_id=1)
    field=replace(field,diagnostics=dict(field.diagnostics,config_sha256=receipt['config_sha256'],
        source_commit=receipt['source_commit'],soft_export_source_sha256=receipt['soft_export_source_sha256']))
    out.mkdir(parents=True,exist_ok=True);save_shadow_field(field,out/'shadow_field.npz')
    receipt.update(stage='model_prediction',model_inference_executed=True,status=field.status,field=field.metadata())
    (out/'provider_result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
