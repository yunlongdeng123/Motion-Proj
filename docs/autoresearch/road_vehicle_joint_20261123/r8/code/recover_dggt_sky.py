"""Exact saved-prediction recovery of DGGT sky only. Root schedules --execute.

--validate-cpu verifies files/shapes/finiteness without torch or GPU imports.
The original failed directory is read-only. Never calls the main DGGT forward.
"""
from pathlib import Path,PurePosixPath
import argparse,importlib,json,os,shutil,sys,time,traceback
from minimal_forward import sha,write,save_npz,REVISION,CORE_SHA,CORE_BYTES,INPUT_SHA

ESSENTIAL=['pose_enc','world_points','world_points_conf','gs_map','gs_conf','dynamic_conf','depth','depth_conf','predicted_world_to_camera','predicted_camera_K_pixel_index','means_from_depth_pose']

def validate_cpu(args):
    import numpy as np
    source=args.source_failed_dir.resolve();receipt_path=source/'receipt.json';source_receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    assert not source_receipt['complete'] and source_receipt['completed_model_forward_calls']==1
    assert source_receipt['revision']==REVISION and source_receipt['strict_checkpoint_loading']
    assert source_receipt['checkpoint_sha256']==CORE_SHA and source_receipt['checkpoint_bytes']==CORE_BYTES
    assert source_receipt['input_bundle_sha256']==INPUT_SHA
    assert source_receipt['DGGT_sky_raster_api_attempts']==1 and source_receipt['DGGT_sky_raster_api_completed']==0 and source_receipt['DGGT_sky_raster_F']==0
    repo=(args.repo or Path(source_receipt['repository'])).resolve();checkpoint=(args.checkpoint or Path(source_receipt['checkpoint'])).resolve();bundle_path=(args.input_bundle or Path(source_receipt['input_bundle'])).resolve()
    assert checkpoint.is_file() and checkpoint.stat().st_size==CORE_BYTES and sha(checkpoint)==CORE_SHA
    assert bundle_path.is_file() and sha(bundle_path)==INPUT_SHA
    source_pins=source_receipt['source_pins'];assert source_pins
    for name,digest in source_pins.items():
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
        path=(repo/Path(*rel.parts)).resolve();assert path.is_relative_to(repo) and sha(path)==digest,name
    sky_path=repo/'dggt/models/sky.py';assert sky_path.is_file()
    prediction_path=source/'predictions.npz';gauge_path=source/'gauge_audit.json'
    prediction_sha=sha(prediction_path)
    if source_receipt.get('predictions_sha256'):assert prediction_sha==source_receipt['predictions_sha256']
    gauge=json.loads(gauge_path.read_text(encoding='utf-8'));assert gauge['accepted_as_metric_world_transform'] is False
    with np.load(bundle_path,allow_pickle=False) as z:bundle={key:z[key] for key in z.files}
    image=bundle['images_srgb'];N,C,H,W=image.shape;assert N==5 and C==3 and H%14==W%14==0 and np.isfinite(image).all() and image.min()>=0 and image.max()<=1
    with np.load(prediction_path,allow_pickle=False) as z:pred={key:z[key] for key in z.files}
    assert all(key in pred for key in ESSENTIAL)
    statistics={}
    for key,array in pred.items():
        assert np.issubdtype(array.dtype,np.number) and np.isfinite(array).all(),key
        statistics[key]=dict(shape=list(array.shape),dtype=str(array.dtype),finite=True)
    for key,last in [('world_points',3),('gs_map',11),('depth',1),('means_from_depth_pose',3)]:assert pred[key].shape==(1,N,H,W,last),(key,pred[key].shape)
    assert pred['pose_enc'].shape==(1,N,9)
    for key in ['world_points_conf','gs_conf','dynamic_conf','depth_conf']:assert pred[key].shape in [(1,N,H,W),(1,N,H,W,1)],(key,pred[key].shape)
    assert pred['predicted_world_to_camera'].shape==(1,N,3,4) and pred['predicted_camera_K_pixel_index'].shape==(1,N,3,3)
    for key in ['raw_frames','timestamp_micros','elapsed_seconds','official_normalized_time','known_world_to_camera','known_camera_K_pixel_index']:assert np.array_equal(pred[key],bundle[key]),key
    assert pred['raw_frames'].tolist()==source_receipt['raw_frames']
    assert args.output.resolve()!=source and not args.output.exists(),'Use a fresh recovery directory; preserve all prior outputs'
    audit=dict(schema='R8.DGGT_sky_recovery_CPU_preflight.v1',complete=True,source_failed_dir=source.as_posix(),source_failure_receipt_sha256=sha(receipt_path),source_complete=False,
        inherited_completed_model_forward_calls=1,new_model_forward_calls=0,checkpoint_sha256=CORE_SHA,input_bundle_sha256=INPUT_SHA,predictions_sha256=prediction_sha,gauge_audit_sha256=sha(gauge_path),source_pins=source_pins,
        recovery_sky_source_sha256=sha(sky_path),shape=[N,C,H,W],raw_frames=pred['raw_frames'].tolist(),output_statistics=statistics,
        original_receipt_had_final_prediction_digest=bool(source_receipt.get('predictions_sha256')),torch_imported=False,GPU_calls=0,model_constructed=False,quality_accepted=False)
    return audit,source_receipt,bundle,pred,repo,checkpoint

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-failed-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--input-bundle',type=Path);parser.add_argument('--preflight-output',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--validate-cpu',action='store_true');mode.add_argument('--execute',action='store_true');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    audit,original,bundle,pred,repo,checkpoint=validate_cpu(args)
    if args.preflight_output:
        assert not args.preflight_output.resolve().is_relative_to(args.source_failed_dir.resolve())
        args.preflight_output.parent.mkdir(parents=True,exist_ok=True);write(args.preflight_output,audit)
    if args.validate_cpu:print(json.dumps(audit,indent=2));return
    import numpy as np
    source=args.source_failed_dir.resolve();args.output.mkdir(parents=True,exist_ok=False)
    for filename,key in [('predictions.npz','predictions_sha256'),('gauge_audit.json','gauge_audit_sha256')]:
        shutil.copy2(source/filename,args.output/filename);assert sha(args.output/filename)==audit[key]
    if (source/'output_semantics.json').is_file():shutil.copy2(source/'output_semantics.json',args.output/'output_semantics.json')
    metadata=dict(original);metadata.pop('error',None);metadata.pop('traceback',None)
    metadata.update(schema='R8.DGGT_saved_main_sky_recovery.v2',complete=False,phase='recovery_verified_before_GPU',source_failure_receipt_sha256=audit['source_failure_receipt_sha256'],source_failed_dir=source.as_posix(),
        predictions_sha256=audit['predictions_sha256'],gauge_audit_sha256=audit['gauge_audit_sha256'],output_statistics=audit['output_statistics'],
        completed_model_forward_calls=1,inherited_completed_model_forward_calls=1,new_model_forward_calls=0,new_model_forward_attempts=0,sky_model_forward_attempts=0,sky_model_forward_completed=0,
        source_failed_sky_raster_api_attempts=original['DGGT_sky_raster_api_attempts'],source_failed_sky_raster_api_completed=0,source_failed_sky_camera_raster_attempts=original['DGGT_sky_camera_raster_attempts'],source_failed_sky_raster_F=0,
        DGGT_sky_raster_api_attempts=0,DGGT_sky_raster_api_completed=0,DGGT_sky_camera_raster_attempts=0,DGGT_sky_raster_F=0,
        native_Gaussian_F=0,native_Gaussian_B=0,Gaussian_renderer_calls=0,training_updates=0,diffusion_calls=0,quality_accepted=False,whole_scene_restoration_accepted=False,
        source_failure_preserved=True,source_prediction_bytes_reused=True,checkpoint_strict_CPU_loading=True,main_forward_reexecution_forbidden=True)
    write(args.output/'receipt.json',metadata);start=time.time();torch=None;sky_module=None;original_raster=None
    try:
        sys.path.insert(0,str(repo));import torch
        from dggt.models.vggt import VGGT
        torch.set_num_threads(2);torch.manual_seed(20261123)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);model=VGGT();incompatible=model.load_state_dict(state,strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        metadata['state_tensor_keys']=len(state);del state
        sky_model=model.sky_model;sky_model.requires_grad_(False).eval();del model
        metadata['main_model_deleted_before_GPU']=True
        assert torch.cuda.is_available(),'Root-scheduled GPU unavailable'
        sky_model=sky_model.to('cuda');images=torch.from_numpy(bundle['images_srgb']).float().to('cuda');N,_,H,W=images.shape
        E=torch.from_numpy(pred['predicted_world_to_camera'][0]).float().to('cuda');K=torch.from_numpy(pred['predicted_camera_K_pixel_index'][0]).float().to('cuda')
        E4=torch.eye(4,device='cuda',dtype=torch.float32)[None].repeat(N,1,1);E4[:,:3,:]=E
        sky_module=importlib.import_module('dggt.models.sky');original_raster=sky_module.rasterization;trace=[]
        def counted(*positional,**kwargs):
            count=int(kwargs['viewmats'].shape[-3]);metadata['DGGT_sky_raster_api_attempts']+=1;metadata['DGGT_sky_camera_raster_attempts']+=count
            write(args.output/'receipt.json',metadata)
            result=original_raster(*positional,**kwargs)
            metadata['DGGT_sky_raster_api_completed']+=1;metadata['DGGT_sky_raster_F']+=count
            trace.append(dict(camera_views=count,width=int(kwargs['width']),height=int(kwargs['height']),gaussians=int(kwargs['means'].shape[0])))
            return result
        sky_module.rasterization=counted;torch.cuda.reset_peak_memory_stats();metadata['sky_model_forward_attempts']=1;write(args.output/'receipt.json',metadata)
        with torch.inference_mode():sky_raw=sky_model(images[None],E4,K)
        torch.cuda.synchronize();metadata['sky_model_forward_completed']=1
        assert tuple(sky_raw.shape)==(N,H,W,3) and torch.isfinite(sky_raw).all()
        minimum=sky_raw.min();maximum=sky_raw.max();normalized=(sky_raw-minimum)/(maximum-minimum+1e-8)
        save_npz(args.output/'sky_bundle.npz',dict(sky_raw=sky_raw.float().cpu().numpy(),sky_official_minmax=normalized.float().cpu().numpy(),raw_frames=pred['raw_frames']),np)
        assert sha(source/'receipt.json')==audit['source_failure_receipt_sha256'] and sha(source/'predictions.npz')==audit['predictions_sha256'] and sha(source/'gauge_audit.json')==audit['gauge_audit_sha256']
        metadata.update(complete=True,phase='recovered_sky_complete_not_quality_acceptance',sky_bundle_sha256=sha(args.output/'sky_bundle.npz'),sky_global_min=float(minimum),sky_global_max=float(maximum),sky_raster_trace=trace,
            aggregate_DGGT_sky_raster_api_attempts=original['DGGT_sky_raster_api_attempts']+metadata['DGGT_sky_raster_api_attempts'],aggregate_DGGT_sky_raster_api_completed=metadata['DGGT_sky_raster_api_completed'],
            aggregate_DGGT_sky_camera_raster_attempts=original['DGGT_sky_camera_raster_attempts']+metadata['DGGT_sky_camera_raster_attempts'],aggregate_DGGT_sky_raster_F=metadata['DGGT_sky_raster_F'],
            recovery_seconds=time.time()-start,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CPU_preflight=audit,existing_A20_or_RoadRows_assets_modified=False)
        write(args.output/'receipt.json',metadata);print(json.dumps(metadata,indent=2))
    except BaseException as error:
        metadata.update(complete=False,phase='recovery_failed_preserved',error=repr(error),traceback=traceback.format_exc(),recovery_seconds=time.time()-start)
        write(args.output/'receipt.json',metadata);raise
    finally:
        if sky_module is not None and original_raster is not None:sky_module.rasterization=original_raster

if __name__=='__main__':main()
