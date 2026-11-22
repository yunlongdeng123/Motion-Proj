"""Pinned DGGT main-only RGB forward; explicit --execute required.

No official inference.py import, dataset/GT loader, LPIPS, Difix, TAPIP, renderer,
optimizer or native-asset write. Known cameras are post-forward gauge evidence.
"""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,subprocess,sys,time,traceback

REVISION='a3276d2bbe4cbb03bcc117830b1836110a27adeb'
CORE_SHA='352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def'
CORE_BYTES=5411266466
INPUT_SHA='34754200f13a5d8f68bb969689f3ae325847f7e09fbb485400b9d994e2eae7a5'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    data=(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode()
    temporary=path.with_name(path.name+'.pending');temporary.write_bytes(data);temporary.replace(path)

def save_npz(path,arrays,np):
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)

def gauge_candidate(predicted_extrinsic,known_extrinsic,np):
    """Closed-form pose-gauge candidate, never a model input or accepted fit."""
    pe=np.asarray(predicted_extrinsic,np.float64);ke=np.asarray(known_extrinsic,np.float64)
    Rp=pe[:,:3,:3];Rk=ke[:,:3,:3]
    Cp=-np.einsum('nij,nj->ni',Rp.transpose(0,2,1),pe[:,:3,3]);Ck=-np.einsum('nij,nj->ni',Rk.transpose(0,2,1),ke[:,:3,3])
    R=Rk[0].T@Rp[0];dp=(Cp-Cp[0])@R.T;dk=Ck-Ck[0];den=float((dp*dp).sum())
    singular=np.linalg.svd(dk,compute_uv=False);orientation=[]
    for a,b in zip(Rp,Rk):
        delta=(b.T@a)@R.T;orientation.append(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))))
    data=dict(status='UNRESOLVED',accepted_as_metric_world_transform=False,
        method='first-camera orientation, one positive least-squares camera-baseline scale, first-camera translation anchor',
        known_camera_centers_m=Ck.tolist(),predicted_camera_centers=Cp.tolist(),known_translation_singular_values=singular.tolist(),
        known_translation_rank=int(np.linalg.matrix_rank(dk,tol=1e-5)),camera_orientation_residual_degrees=orientation,
        cameras_used_only_after_network_forward=True)
    if den<=1e-12:
        data['reason']='predicted camera translation baseline is insufficient';return data,None
    scale=float((dp*dk).sum()/den)
    if not np.isfinite(scale) or scale<=0:
        data['reason']='camera-baseline least-squares scale is nonpositive or nonfinite';return data,None
    translation=Ck[0]-scale*R@Cp[0];mapped=Cp@(scale*R).T+translation
    transform=np.eye(4);transform[:3,:3]=scale*R;transform[:3,3]=translation
    data.update(status='CANDIDATE_NOT_ACCEPTED',scale=scale,world_from_dggt_candidate=transform.tolist(),
        camera_center_residual_m=np.linalg.norm(mapped-Ck,axis=-1).tolist(),
        camera_center_RMSE_m=float(np.sqrt(np.mean((mapped-Ck)**2))),
        application='raw DGGT outputs remain unchanged; native import requires separate gauge and reconstruction checks')
    return data,transform

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/root/autodl-tmp/dggt'))
    parser.add_argument('--checkpoint',type=Path,default=Path('/root/autodl-tmp/dggt/pretrained/model_latest_waymo.pth'))
    parser.add_argument('--checkpoint-sha256',default=CORE_SHA);parser.add_argument('--checkpoint-bytes',type=int,default=CORE_BYTES)
    parser.add_argument('--input-bundle',type=Path,required=True);parser.add_argument('--input-sha256',default=INPUT_SHA)
    parser.add_argument('--input-contract',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='float32')
    parser.add_argument('--max-frames',type=int,default=8);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--validate-input-only',action='store_true');parser.add_argument('--input-preflight-output',type=Path)
    parser.add_argument('--export-sky',action='store_true',help='Reuse resident model.sky_model; count its actual gsplat API and camera-view rasters separately')
    args=parser.parse_args()
    if args.execute and args.validate_input_only:parser.error('Choose execution or CPU input preflight, not both')
    if not args.execute and not args.validate_input_only:parser.error('No execution performed: --execute is required for model loading and the scheduled GPU forward')
    # No imports that can initialize CUDA occur before the explicit guard above.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
    import numpy as np
    assert args.input_bundle.is_file() and sha(args.input_bundle)==args.input_sha256,'Input bundle pin mismatch'
    with np.load(args.input_bundle,allow_pickle=False) as z:bundle={k:z[k] for k in z.files}
    required=['images_srgb','raw_frames','timestamp_micros','elapsed_seconds','official_normalized_time','known_world_to_camera','known_camera_K_pixel_index']
    assert all(k in bundle for k in required),list(bundle)
    images_array=bundle['images_srgb'];N=images_array.shape[0]
    assert images_array.ndim==4 and images_array.shape[1]==3 and 1<N<=args.max_frames
    assert np.isfinite(images_array).all() and images_array.min()>=0 and images_array.max()<=1
    H,W=images_array.shape[-2:];assert H%14==0 and W%14==0
    raw=bundle['raw_frames'];stamp=bundle['timestamp_micros'].astype(np.int64);elapsed=(stamp-stamp[0]).astype(np.float64)/1e6
    assert len(raw)==len(stamp)==N and len(set(raw.tolist()))==N and (np.diff(stamp)>0).all()
    assert np.allclose(bundle['elapsed_seconds'],elapsed,rtol=0,atol=1e-8)
    normalized=elapsed/elapsed[-1]*(N/4.)
    assert np.allclose(bundle['official_normalized_time'],normalized,rtol=1e-6,atol=1e-7)
    known_e=bundle['known_world_to_camera'];known_k=bundle['known_camera_K_pixel_index']
    assert known_e.shape in [(N,4,4),(N,3,4)] and known_k.shape==(N,3,3)
    contract=args.input_contract or args.input_bundle.parent/'contract.json'
    if contract.exists():
        input_contract=json.loads(contract.read_text(encoding='utf-8-sig'))
        assert input_contract.get('input_bundle_sha256',args.input_sha256)==args.input_sha256
    if args.validate_input_only:
        preflight=dict(schema='R8.DGGT_input_bundle_preflight.v1',complete=True,phase='CPU_input_only',input_bundle_sha256=args.input_sha256,
            shape=list(images_array.shape),raw_frames=raw.tolist(),timestamp_micros=stamp.tolist(),elapsed_seconds=elapsed.tolist(),official_normalized_time=normalized.tolist(),
            known_camera_shapes=[list(known_e.shape),list(known_k.shape)],known_cameras_are_model_inputs=False,
            model_constructed=False,checkpoint_loaded=False,model_forward_calls=0,Gaussian_renderer_calls=0,training_updates=0,diffusion_calls=0,
            GPU_output_directory_created=False)
        if args.input_preflight_output:args.input_preflight_output.parent.mkdir(parents=True,exist_ok=True);write(args.input_preflight_output,preflight)
        print(json.dumps(preflight,indent=2));return
    assert args.checkpoint.is_file() and args.checkpoint.stat().st_size==args.checkpoint_bytes,'Main checkpoint is not fully restored'
    print('VERIFYING_PINNED_MAIN',flush=True);checkpoint_digest=sha(args.checkpoint)
    assert checkpoint_digest==args.checkpoint_sha256,'Main checkpoint SHA mismatch'
    revision=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip();assert revision==REVISION
    assert not args.output.exists(),'Preserve every failed/successful output; use a fresh output directory'
    args.output.mkdir(parents=True)
    source_pins={str(p.relative_to(args.repo)):sha(p) for p in [args.repo/'dggt/models/vggt.py',args.repo/'dggt/heads/dpt_head.py',args.repo/'dggt/heads/head_act.py',args.repo/'dggt/utils/pose_enc.py',args.repo/'dggt/utils/geometry.py']}
    metadata=dict(schema='R8.DGGT_main_only_forward.v1',complete=False,phase='verified_inputs_before_model_import',
        repository=str(args.repo),revision=revision,source_pins=source_pins,checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_digest,checkpoint_bytes=args.checkpoint_bytes,
        input_bundle=str(args.input_bundle),input_bundle_sha256=args.input_sha256,input_contract_sha256=sha(contract) if contract.exists() else None,
        input_shape=list(images_array.shape),raw_frames=raw.tolist(),timestamp_micros=stamp.tolist(),elapsed_seconds=elapsed.tolist(),official_normalized_time=normalized.tolist(),
        model_input_fields=['images_srgb'],known_cameras_are_model_inputs=False,network_input_resized_again=False,
        time_semantics='one front image per actual frame; true Unix microseconds, elapsed seconds and official N/4 normalized time are separate; no camera-count grouping',
        dtype=args.dtype,strict_checkpoint_loading=True,model_forward_attempts=0,completed_model_forward_calls=0,
        native_Gaussian_F=0,native_Gaussian_B=0,Gaussian_renderer_calls=0,training_updates=0,diffusion_calls=0,
        Difix_loaded=False,LPIPS_loaded=False,GT_depth_loaded=False,TAPIP_3D_motion_loaded=False)
    metadata.update(DGGT_sky_raster_api_attempts=0,DGGT_sky_raster_api_completed=0,DGGT_sky_camera_raster_attempts=0,DGGT_sky_raster_F=0)
    write(args.output/'receipt.json',metadata);start=time.time();torch=None
    try:
        sys.path.insert(0,str(args.repo))
        import torch
        from dggt.models.vggt import VGGT
        from dggt.utils.pose_enc import pose_encoding_to_extri_intri
        from dggt.utils.geometry import unproject_depth_map_to_point_map
        torch.set_num_threads(2);torch.manual_seed(20261123)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        print('LOADING_MAIN_STRICT_CPU',flush=True)
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
        assert isinstance(state,dict) and state and all(isinstance(k,str) for k in state)
        model=VGGT();incompatible=model.load_state_dict(state,strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        metadata['state_tensor_keys']=len(state);del state
        model.requires_grad_(False).eval()
        metadata.update(phase='main_strict_loaded_cpu',torch=torch.__version__);write(args.output/'receipt.json',metadata)
        assert torch.cuda.is_available(),'The root-scheduled GPU is unavailable'
        model=model.to('cuda');images=torch.from_numpy(images_array).float().to('cuda')
        torch.cuda.reset_peak_memory_stats();metadata.update(phase='forward_running',model_forward_attempts=1,GPU_device=torch.cuda.get_device_name())
        write(args.output/'receipt.json',metadata);print('FORWARD_RGB_ONLY',flush=True);forward_start=time.time()
        dtype={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}[args.dtype]
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=dtype,enabled=args.dtype!='float32'):
            predictions=model(images)
        torch.cuda.synchronize();metadata.update(completed_model_forward_calls=1,model_forward_seconds=time.time()-forward_start)
        essential=['pose_enc','world_points','world_points_conf','gs_map','gs_conf','dynamic_conf','depth','depth_conf']
        assert all(k in predictions for k in essential)
        arrays={};statistics={}
        for key in essential+(['semantic_logits'] if 'semantic_logits' in predictions else []):
            array=predictions[key].detach().float().cpu().numpy();assert np.isfinite(array).all(),key
            arrays[key]=array;statistics[key]=dict(shape=list(array.shape),dtype=str(array.dtype),min=float(array.min()),max=float(array.max()),finite=True)
        assert arrays['gs_map'].shape==(1,N,H,W,11) and arrays['depth'].shape==(1,N,H,W,1)
        extrinsic,intrinsic=pose_encoding_to_extri_intri(predictions['pose_enc'],(H,W))
        arrays['predicted_world_to_camera']=extrinsic.detach().float().cpu().numpy()
        arrays['predicted_camera_K_pixel_index']=intrinsic.detach().float().cpu().numpy()
        arrays['means_from_depth_pose']=unproject_depth_map_to_point_map(arrays['depth'][0],arrays['predicted_world_to_camera'][0],arrays['predicted_camera_K_pixel_index'][0])[None]
        assert np.isfinite(arrays['means_from_depth_pose']).all()
        # Keep raw outputs and both clocks. No lifespan fading, scene merge or render.
        arrays.update(raw_frames=raw,timestamp_micros=stamp,elapsed_seconds=elapsed,official_normalized_time=bundle['official_normalized_time'],
            known_world_to_camera=known_e,known_camera_K_pixel_index=known_k)
        gauge,transform=gauge_candidate(arrays['predicted_world_to_camera'][0],known_e,np)
        if transform is not None:arrays['known_world_from_dggt_candidate']=transform
        write(args.output/'gauge_audit.json',gauge)
        save_npz(args.output/'predictions.npz',arrays,np)
        if args.export_sky:
            import importlib
            sky_module=importlib.import_module('dggt.models.sky');original_raster=sky_module.rasterization
            sky_trace=[]
            def counted_sky(*positional,**kwargs):
                count=int(kwargs['viewmats'].shape[-3]);metadata['DGGT_sky_raster_api_attempts']+=1;metadata['DGGT_sky_camera_raster_attempts']+=count
                result=original_raster(*positional,**kwargs)
                metadata['DGGT_sky_raster_api_completed']+=1;metadata['DGGT_sky_raster_F']+=count
                sky_trace.append(dict(camera_views=count,width=int(kwargs['width']),height=int(kwargs['height']),gaussians=int(kwargs['means'].shape[0])))
                return result
            sky_module.rasterization=counted_sky;sky_start=time.time()
            try:
                E4=torch.eye(4,device='cuda',dtype=torch.float32)[None].repeat(N,1,1);E4[:,:3,:]=extrinsic[0].float()
                with torch.inference_mode():sky_raw=model.sky_model(images[None],E4,intrinsic[0].float())
                torch.cuda.synchronize();sky_min=sky_raw.min();sky_max=sky_raw.max()
                sky_normalized=(sky_raw-sky_min)/(sky_max-sky_min+1e-8)
                save_npz(args.output/'sky_bundle.npz',dict(sky_raw=sky_raw.float().cpu().numpy(),sky_official_minmax=sky_normalized.float().cpu().numpy(),raw_frames=raw),np)
                metadata.update(sky_bundle_sha256=sha(args.output/'sky_bundle.npz'),sky_seconds=time.time()-sky_start,
                    sky_global_min=float(sky_min),sky_global_max=float(sky_max),sky_raster_trace=sky_trace)
            finally:sky_module.rasterization=original_raster
        semantics=dict(gs_map_channels={'RGB':[0,3],'opacity':[3,4],'scale':[4,7],'quaternion_wxyz':[7,11]},
            gs_map_activated_by_native_head=True,scale_units='DGGT reconstruction gauge; not yet accepted native model metres',
            means_from_depth_pose='official mode2 choice: predicted depth unprojected by predicted cameras at integer pixel indices; raw world_points head also retained',
            dynamic_conf='raw dynamic logits; sigmoid is a derived probability, not ground truth',
            gs_conf='native lifespan gamma: 1/(ReLU(raw)+1e-6), not calibrated confidence or Unix seconds',
            lifespan_code_equation='alpha(t)=alpha0*exp(log(0.1)*(t-t0)^2/(gamma^2+1e-6)), using official normalized time',
            motion='no query_points or TAPIP interpolation executed; no full 3D trajectory claim',
            semantic_logits='unvalidated native head output; public train.py freezes this head and comments out its semantic loss',
            gauge='post-forward camera-baseline candidate only; no direct A20/RoadRows import or parameter update',quality_accepted=False)
        write(args.output/'output_semantics.json',semantics)
        metadata.update(complete=True,phase='core_forward_complete_not_quality_acceptance',output_statistics=statistics,
            predictions_sha256=sha(args.output/'predictions.npz'),gauge_status=gauge['status'],
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds_total=time.time()-start,quality_accepted=False,existing_A20_or_RoadRows_assets_modified=False)
        write(args.output/'receipt.json',metadata);print(json.dumps(metadata,indent=2),flush=True)
    except BaseException as error:
        metadata.update(complete=False,phase='failed_preserved',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds_total=time.time()-start)
        if torch is not None:metadata['CUDA_initialized_on_failure']=bool(torch.cuda.is_initialized())
        write(args.output/'receipt.json',metadata);raise

if __name__=='__main__':main()
