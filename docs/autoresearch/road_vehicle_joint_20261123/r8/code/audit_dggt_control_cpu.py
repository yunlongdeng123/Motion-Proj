"""Independent CPU audit of saved DGGT predictions and actual mode2 rasters.

No torch import, model call, raster, official-repository mutation or asset edit.
Metrics concern the same five original input photographs; A9 is not removed.
"""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
from PIL import Image
from joint_rgb_common import read,load,write,sha


def pin(path):
    path=Path(path);return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def stats(value,mask=None):
    value=np.asarray(value);value=value[mask] if mask is not None else value.reshape(-1)
    return dict(values=int(value.size),min=float(value.min()) if value.size else None,
        median=float(np.median(value)) if value.size else None,p99=float(np.quantile(value,.99)) if value.size else None,
        max=float(value.max()) if value.size else None)


def quality(output,target,domain):
    if not domain.any():return dict(pixels=0,MAE=None,MSE=None,PSNR_dB=None)
    residual=(output.astype(np.float64)-target.astype(np.float64))[domain]
    mse=float(np.mean(residual**2));return dict(pixels=int(domain.sum()),MAE=float(abs(residual).mean()),MSE=mse,
        PSNR_dB=float(-10*np.log10(mse)) if mse>0 else None,PSNR_infinite=mse==0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.root;o=a.output;o.mkdir(parents=True,exist_ok=False);started=time.time()
    old_dir=r/'forward_smoke_five_views_v1';rec_dir=r/'forward_smoke_five_views_recovered_v2';render_dir=r/'dggt_no_diff_control_v1'
    old=read(old_dir/'receipt.json');rec=read(rec_dir/'receipt.json');render=read(render_dir/'receipt.json')
    assert old['complete'] is False and old['completed_model_forward_calls']==1 and old['DGGT_sky_raster_api_completed']==0
    assert rec['complete'] and rec['inherited_completed_model_forward_calls']==1 and rec['new_model_forward_calls']==0
    assert render['complete'] and render['model_forward_calls']==0 and render['DGGT_raster_F']==5
    refs={name:pin(path) for name,path in [('failed_forward_receipt',old_dir/'receipt.json'),('recovery_receipt',rec_dir/'receipt.json'),('render_receipt',render_dir/'receipt.json')]}
    print('VERIFY_INPUT_AND_OFFICIAL_SOURCE_PINS',flush=True)
    input_path=Path(old['input_bundle']);bundle=load(input_path);contract_path=input_path.parent/'contract.json';contract=read(contract_path)
    assert sha(input_path)==old['input_bundle_sha256']==rec['input_bundle_sha256']==render['input_bundle_sha256']
    assert sha(contract_path)==old['input_contract_sha256'];frames=bundle['raw_frames'].tolist()
    assert frames==[86,110,134,179,198] and old['model_input_fields']==['images_srgb'] and old['known_cameras_are_model_inputs'] is False
    assert bundle['images_srgb'].shape==(5,3,350,518) and np.isfinite(bundle['images_srgb']).all()
    elapsed=(bundle['timestamp_micros']-bundle['timestamp_micros'][0])/1e6
    assert np.array_equal(elapsed,bundle['elapsed_seconds'])
    assert np.allclose(bundle['official_normalized_time'],elapsed/elapsed[-1]*(len(frames)/4),rtol=0,atol=1e-12)
    input_rows=[]
    for index,item in enumerate(contract['rows']):
        raw=item['raw'];cache_path=Path(item['cache_path']);cache=load(cache_path)
        checked={}
        for role,pathkey,hashkey in [('cache','cache_path','cache_sha256'),('observation','observation_path','observation_sha256'),('original_photo','original_photo','original_photo_sha256')]:
            path=Path(item[pathkey]);checked[role]=pin(path);assert checked[role]['sha256']==item[hashkey]
        for suffix,key in [('real_pinhole','pinhole_image_sha256'),('dggt_input','model_input_sha256')]:
            path=input_path.parent/f'{raw:03d}_{suffix}.png';checked[suffix]=pin(path);assert checked[suffix]['sha256']==item[key]
        image=np.asarray(Image.open(input_path.parent/f'{raw:03d}_dggt_input.png'),np.float32).transpose(2,0,1)/255
        assert image.tobytes()==bundle['images_srgb'][index].tobytes()
        quant=np.rint(np.clip(cache['photo_pinhole_srgb'],0,1)*255).astype(np.uint8)
        assert np.array_equal(np.asarray(Image.open(input_path.parent/f'{raw:03d}_real_pinhole.png')),quant)
        resized=np.asarray(Image.fromarray(quant).resize((518,350),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1)/255
        assert resized.tobytes()==image.tobytes()
        known=np.diag([518/1152,350/768,1.])@cache['camera_K'];known[:2,2]-=.5
        assert np.array_equal(known,bundle['known_camera_K_pixel_index'][index])
        assert np.array_equal(cache['world_to_camera'],bundle['known_world_to_camera'][index])
        assert int(read(item['observation_path'])['timestamp_micros'])==int(bundle['timestamp_micros'][index])
        input_rows.append(dict(raw=raw,pins=checked,RGB_preprocessing_exact=True,known_K_pixel_index_shift_exact=True,timestamp_exact=True))
    repo=Path(old['repository']);revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();assert revision==old['revision']
    source_pins={key:pin(repo/key) for key in old['source_pins']}
    assert all(source_pins[key]['sha256']==digest for key,digest in old['source_pins'].items())
    print('VERIFY_MAIN_CHECKPOINT_SHA_CPU_ONLY',flush=True)
    checkpoint=pin(old['checkpoint']);assert checkpoint['sha256']==old['checkpoint_sha256'] and checkpoint['bytes']==old['checkpoint_bytes']
    pred_path=old_dir/'predictions.npz';pred_pin=pin(pred_path);pred=load(pred_path)
    assert pred_pin['sha256']==rec['predictions_sha256']==render['predictions_sha256']==sha(rec_dir/'predictions.npz')
    fields={}
    for key,value in pred.items():
        assert value.dtype.kind in 'biuf' and np.isfinite(value).all(),key
        fields[key]=dict(shape=list(value.shape),dtype=str(value.dtype),all_finite=True,min=float(value.min()),max=float(value.max()))
    N,_,H,W=bundle['images_srgb'].shape
    assert pred['gs_map'].shape==(1,N,H,W,11) and pred['depth'].shape==(1,N,H,W,1)
    assert pred['means_from_depth_pose'].shape==(1,N,H,W,3) and pred['world_points'].shape==(1,N,H,W,3)
    assert np.array_equal(pred['raw_frames'],bundle['raw_frames'])
    gs=pred['gs_map'];assert ((gs[...,3]>=0)&(gs[...,3]<=1)).all() and (gs[...,4:7]>0).all()
    qnorm=np.linalg.norm(gs[...,7:11].astype(np.float64),axis=-1)
    # Independent SIM(3) camera-baseline candidate recomputation, never applied.
    pe=pred['predicted_world_to_camera'][0].astype(np.float64);ke=bundle['known_world_to_camera'].astype(np.float64)
    Rp,Rk=pe[:,:3,:3],ke[:,:3,:3]
    Cp=-np.einsum('nij,nj->ni',Rp.transpose(0,2,1),pe[:,:3,3]);Ck=-np.einsum('nij,nj->ni',Rk.transpose(0,2,1),ke[:,:3,3])
    R=Rk[0].T@Rp[0];dp=(Cp-Cp[0])@R.T;dk=Ck-Ck[0];scale=float((dp*dk).sum()/(dp*dp).sum());translation=Ck[0]-scale*R@Cp[0]
    mapped=Cp@(scale*R).T+translation;error=mapped-Ck;singular=np.linalg.svd(dk,compute_uv=False)
    angles=[float(np.degrees(np.arccos(np.clip((np.trace((b.T@x)@R.T)-1)/2,-1,1)))) for x,b in zip(Rp,Rk)]
    gauge=read(old_dir/'gauge_audit.json');assert gauge['accepted_as_metric_world_transform'] is False
    assert abs(gauge['scale']-scale)<1e-12 and abs(gauge['camera_center_RMSE_m']-float(np.sqrt(np.mean(error**2))))<1e-12
    candidate=dict(status='CANDIDATE_NOT_ACCEPTED',scale_m_per_DGGT_unit=scale,
        coordinate_RMSE_m=float(np.sqrt(np.mean(error**2))),camera_center_3D_RMSE_m=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        camera_center_residual_m=np.linalg.norm(error,axis=1).tolist(),orientation_residual_degrees=angles,
        translation_singular_values=singular.tolist(),s2_over_s1=float(singular[1]/singular[0]),s3_over_s1=float(singular[2]/singular[0]),
        raw_world_points_and_scales_not_transformed=True,no_metric_or_native_import=True,
        interpretation='near-line camera path; small camera residual is not independent metric surface/depth/identity validation')
    audit=dict(schema='R8.independent_DGGT_forward_prediction_CPU.v1',complete=True,CPU_only=True,modelF=0,rasterF=0,updates=0,
        input_bundle=pin(input_path),input_contract=pin(contract_path),inputs=input_rows,official_revision=revision,source_pins=source_pins,
        checkpoint=checkpoint,predictions=pred_pin,prediction_fields=fields,quaternion_norm=stats(qnorm),
        gs_scale_gauge_units=stats(gs[...,4:7]),opacity=stats(gs[...,3]),gauge_candidate=candidate,
        v1_failure_preserved=True,completed_main_forward_unique=1,recovery_new_main_forward=0,original_A9_is_not_deleted=True)
    write(o/'forward_prediction_audit.json',audit)
    print('VERIFY_RENDER_FLOAT_LAYERS_AND_SELF_RECONSTRUCTION',flush=True)
    sky_path=rec_dir/'sky_bundle.npz';assert sha(sky_path)==rec['sky_bundle_sha256']==render['sky_bundle_sha256'];sky=load(sky_path)['sky_official_minmax']
    rows=[];mse_all={'official_literal_RGB':[],'diagnostic_single_alpha_RGB':[]};proxy=r/'inputs/forward_smoke_five_views/sky_proxy.npz'
    if not proxy.exists():
        options=[x for x in input_path.parent.glob('*.npz') if sha(x)==render['sky_proxy_sha256']];assert len(options)==1;proxy=options[0]
    assert sha(proxy)==render['sky_proxy_sha256'];proxy_data=load(proxy);labels=proxy_data['semantic_labels']
    for index,raw in enumerate(frames):
        path=render_dir/f'{raw:03d}/layers_and_depth.npz';z=load(path);assert all(np.isfinite(v).all() for v in z.values())
        alpha=z['alpha'];image=z['joint_RGB_premult'];literal=alpha*image+(1-alpha)*sky[index];single=image+(1-alpha)*sky[index]
        assert np.array_equal(literal,z['official_literal_RGB']) and np.array_equal(single,z['diagnostic_single_alpha_RGB'])
        RGBsum=float(abs(image-z['static_RGB_premult']-z['dynamic_RGB_premult']).max());Asum=float(abs(alpha-z['static_alpha']-z['dynamic_alpha']).max())
        assert RGBsum<2e-5 and Asum<2e-5
        original=bundle['images_srgb'][index].transpose(1,2,0);valid=bundle['valid_photo_support'][index]
        domains=dict(all_input_pixels=np.ones((H,W),bool),valid_original_photo=valid,road_proxy=labels[index]==0,
            vehicle_proxy=np.isin(labels[index],[13,14,15,16,17,18]),sky_proxy=labels[index]==10)
        record=dict(raw=raw,float_layers=pin(path),all_layers_finite=True,RGB_sum_error=RGBsum,alpha_sum_error=Asum,domains={},
            depth_units='DGGT gauge, not metric ground truth',dynamic_is_model_partition_not_tracked_actor_identity=True)
        for name,domain in domains.items():
            data=dict(pixels=int(domain.sum()),alpha=stats(alpha[...,0],domain),
                static_alpha=stats(z['static_alpha'][...,0],domain),dynamic_alpha=stats(z['dynamic_alpha'][...,0],domain),
                official_literal=quality(literal,original,domain),diagnostic_single_alpha=quality(single,original,domain),
                visible_expected_depth=stats(z['expected_camera_depth'][...,0],domain&(alpha[...,0]>1e-5)))
            record['domains'][name]=data
        for key in mse_all:mse_all[key].append(quality(z[key],original,np.ones((H,W),bool))['MSE'])
        row0=next(x for x in render['rows'] if x['raw']==raw)
        assert abs(record['domains']['all_input_pixels']['official_literal']['MAE']-row0['input_view_literal_MAE'])<1e-7
        rows.append(record)
    metrics=dict(schema='R8.DGGT_original_input_self_reconstruction_CPU.v1',complete=True,rows=rows,
        equal_size_five_view_global_PSNR_dB={key:float(-10*np.log10(np.mean(values))) for key,values in mse_all.items()},
        metric_space='encoded input RGB, data range1, saved unrounded float renders; no PNG-derived PSNR',
        official_literal_preserved=True,single_alpha_diagnostic_not_substituted=True,all_views_were_network_inputs=True,
        original_A9_is_retained=True,road_deletion_or_heldout_quality_claim=False,semantic_domains_are_proxy=True)
    write(o/'self_reconstruction_metrics.json',metrics)
    native_files=[('baseline86_110','baseline_A0_first2/receipt.json'),('baseline5','baseline_A0_five_views/receipt.json'),
        ('owner_measurement','owner_measurement_A0_v1/receipt.json'),('colour_K_guard','joint_colour_guard_v1/receipt.json'),
        ('failed_strict_smoke','joint_rgb_smoke6_v1/resources.json'),('discarded_smoke_v2','joint_rgb_smoke6_v2/resources.json'),
        ('smoke_fresh','joint_rgb_smoke6_v2/fresh_reload/receipt.json'),('formal400','joint_rgb_formal400_v1/resources.json'),
        ('formal_fresh','joint_rgb_formal400_v1/fresh_reload/receipt.json'),('supervision_readout','joint_rgb_formal400_v1/supervision_layers/receipt.json')]
    stages=[]
    for name,path in native_files:
        path=r/path;q=read(path);stages.append(dict(stage=name,receipt=pin(path),complete=q['complete'],
            nativeF=q.get('nativeF',q.get('actual_nativeF',0)),nativeB=q.get('nativeB',q.get('actual_nativeB',0)),
            optimizer_updates=q.get('actual_optimizer_updates',q.get('updates',q.get('optimizer_updates',0)))))
    totals={key:sum(x[key] for x in stages) for key in ['nativeF','nativeB','optimizer_updates']}
    assert totals==dict(nativeF=968,nativeB=833,optimizer_updates=413),totals
    ledger=dict(schema='R8_R9_explicit_actual_compute_ledger.v1',complete=True,native_branch=dict(stages=stages,totals=totals,formal_updates=400,discarded_smoke_updates=13),
        R9_semantic_forward_calls=2,new_diffusion_calls=0,
        DGGT_main=dict(unique_completed_forward=1,original_attempts=1,inherited_in_recovery=1,new_recovery_forward=0,mode2_new_forward=0,seconds=old['model_forward_seconds']),
        DGGT_sky=dict(original_failed_API_attempts=old['DGGT_sky_raster_api_attempts'],original_failed_completed=old['DGGT_sky_raster_api_completed'],
            original_failed_camera_attempts=old['DGGT_sky_camera_raster_attempts'],original_failed_camera_F=old['DGGT_sky_raster_F'],
            recovery_API_attempts=rec['DGGT_sky_raster_api_attempts'],recovery_API_completed=rec['DGGT_sky_raster_api_completed'],
            recovery_camera_F=rec['DGGT_sky_raster_F'],aggregate_API_attempts=rec['aggregate_DGGT_sky_raster_api_attempts'],
            aggregate_API_completed=rec['aggregate_DGGT_sky_raster_api_completed'],aggregate_camera_attempts=rec['aggregate_DGGT_sky_camera_raster_attempts'],
            aggregate_completed_camera_F=rec['aggregate_DGGT_sky_raster_F']),
        DGGT_mode2=dict(API_attempts=render['DGGT_raster_api_attempts'],API_completed=render['DGGT_raster_api_completed'],camera_F=render['DGGT_raster_F'],B=0,updates=0,seconds=render['seconds']),
        backend_probe=read(r/'runtime/backend_probe.json'),CPU_audit_new_model_or_raster_calls=0,
        model_and_Gaussian_raster_F_are_not_added_as_one_kind=True,failed_sky_attempt_is_not_erased=True)
    write(o/'explicit_resource_ledger.json',ledger)
    summary=dict(schema='R8.DGGT_native_separate_control_summary.v1',complete=True,refs=refs,
        forward_audit=pin(o/'forward_prediction_audit.json'),self_reconstruction_metrics=pin(o/'self_reconstruction_metrics.json'),resource_ledger=pin(o/'explicit_resource_ledger.json'),
        previews=pin(r/'dggt_control_preview_pngs/manifest.json'),
        branches=dict(DGGT='same five original input photographs reconstructed in predicted DGGT gauge; original A9 retained; no diffusion',
            native='separate fixed-geometry road/selected-A20 colour fit with road-removal source target; not the same task/target and not ranked by the DGGT self-PSNR'),
        native_summary=pin(r/'formal400_summary_v1/compact_result.json'),
        root_quality_acceptance=False,metric_native_import=False,seconds=time.time()-started)
    write(o/'control_summary.json',summary)
    print(json.dumps(dict(complete=True,output=str(o),summary_sha256=sha(o/'control_summary.json'),PSNR=metrics['equal_size_five_view_global_PSNR_dB'],native_totals=totals)))


if __name__=='__main__':main()
