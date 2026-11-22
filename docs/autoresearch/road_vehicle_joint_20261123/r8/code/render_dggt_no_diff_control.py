"""Scheduled DGGT mode2 no-diffusion raster control; --execute required.

One extended-feature joint gsplat raster per camera. Literal official double-
alpha composition is preserved beside a separately named diagnostic correction.
No network forward, native DrivingEditor raster, optimizer or diffusion occurs.
"""
from pathlib import Path
import argparse,json,os,time,traceback
import numpy as np
from minimal_forward import sha,write,save_npz

def png(path,array):
    from PIL import Image
    a=np.rint(np.clip(array,0,1)*255).astype(np.uint8)
    if a.ndim==3 and a.shape[-1]==1:a=a[...,0]
    Image.fromarray(a).save(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--forward-dir',type=Path,required=True);p.add_argument('--input-bundle',type=Path,required=True)
    p.add_argument('--sky-proxy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('GPU raster execution requires the root-scheduled --execute flag')
    forward=json.loads((a.forward_dir/'receipt.json').read_text());assert forward['complete'] and forward['completed_model_forward_calls']==1
    assert sha(a.input_bundle)==forward['input_bundle_sha256'];assert sha(a.forward_dir/'predictions.npz')==forward['predictions_sha256']
    assert (a.forward_dir/'sky_bundle.npz').is_file() and sha(a.forward_dir/'sky_bundle.npz')==forward['sky_bundle_sha256'],'Run main once with --export-sky; never silently substitute another sky'
    proxy_receipt=json.loads(a.sky_proxy.with_name('sky_proxy_receipt.json').read_text());assert proxy_receipt['sky_proxy_sha256']==sha(a.sky_proxy)
    assert proxy_receipt['input_bundle_sha256']==forward['input_bundle_sha256'] and not proxy_receipt['semantic_ground_truth']
    with np.load(a.forward_dir/'predictions.npz',allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    with np.load(a.forward_dir/'sky_bundle.npz',allow_pickle=False) as z:sky=z['sky_official_minmax'].copy();sky_frames=z['raw_frames'].copy()
    with np.load(a.sky_proxy,allow_pickle=False) as z:bg=z['background_geometry_mask'].copy();proxy_frames=z['raw_frames'].copy();labels=z['semantic_labels'].copy()
    with np.load(a.input_bundle,allow_pickle=False) as z:input_rgb=z['images_srgb'].transpose(0,2,3,1).copy()
    frames=pred['raw_frames'];N=len(frames);H,W=bg.shape[-2:]
    assert np.array_equal(frames,proxy_frames) and np.array_equal(frames,sky_frames) and sky.shape==(N,H,W,3)
    assert not a.output.exists();a.output.mkdir(parents=True)
    receipt=dict(schema='R8.DGGT_no_diff_extended_feature_mode2.v1',complete=False,phase='verified_before_GPU',forward_receipt_sha256=sha(a.forward_dir/'receipt.json'),
        predictions_sha256=sha(a.forward_dir/'predictions.npz'),sky_bundle_sha256=sha(a.forward_dir/'sky_bundle.npz'),sky_proxy_sha256=sha(a.sky_proxy),
        input_bundle_sha256=forward['input_bundle_sha256'],frames=frames.tolist(),model_forward_calls=0,
        DGGT_raster_api_attempts=0,DGGT_raster_api_completed=0,DGGT_raster_F=0,native_DrivingEditor_F=0,native_DrivingEditor_B=0,training_updates=0,diffusion_calls=0,
        literal_original_preserved=True,diagnostic_single_alpha_is_not_original=True,sky_mask_is_proxy_not_GT=True,
        static_policy='literal official raw dynamic logit < .5; alpha times (1-sigmoid(logit)); static union across input frames',
        dynamic_policy='literal official all non-sky current-frame points; alpha times sigmoid(logit)',
        time_policy='exact provided official_normalized_time; real seconds remain separate; no camera-count grouping')
    write(a.output/'receipt.json',receipt);start=time.time()
    try:
        import torch
        from gsplat.rendering import rasterization
        assert torch.cuda.is_available();device='cuda'
        points=torch.as_tensor(pred['means_from_depth_pose'],device=device).float();gs=torch.as_tensor(pred['gs_map'],device=device).float()
        dynamic=torch.as_tensor(pred['dynamic_conf'],device=device).float().squeeze(-1);gamma=torch.as_tensor(pred['gs_conf'],device=device).float()
        if gamma.ndim==5:gamma=gamma.squeeze(-1)
        background=torch.as_tensor(bg[None],device=device);clock=torch.as_tensor(pred['official_normalized_time'],device=device,dtype=torch.float32)
        E=torch.as_tensor(pred['predicted_world_to_camera'][0],device=device).float();K=torch.as_tensor(pred['predicted_camera_K_pixel_index'][0],device=device).float()
        assert torch.isfinite(points).all() and torch.isfinite(gs).all() and torch.isfinite(dynamic).all() and torch.isfinite(gamma).all()
        assert (gs[...,4:7]>0).all() and (gs[...,3]>=0).all() and (gs[...,3]<=1).all()
        static=(dynamic<.5)&background;sp=points[static];sg=gs[static];so=sg[:,3]*(1-dynamic[static].sigmoid());life=gamma[static]
        source_times=clock[torch.where(static)[1]];rows=[];saved=[];torch.cuda.reset_peak_memory_stats()
        for i,raw in enumerate(frames):
            folder=a.output/f'{int(raw):03d}';folder.mkdir();mask=background[:,i]
            dp=points[:,i][mask];dg=gs[:,i][mask];do=dg[:,3]*dynamic[:,i][mask].sigmoid()
            timed=so*torch.exp(torch.log(torch.tensor(.1,device=device))/(life.square()+1e-6)*(source_times-clock[i]).square())
            means=torch.cat([sp,dp]);attrs=torch.cat([sg,dg]);opacity=torch.cat([timed,do]);count_static=len(sp)
            ownership=torch.zeros((len(means),2),device=device);ownership[:count_static,0]=1;ownership[count_static:,1]=1
            z=means@E[i,2,:3]+E[i,2,3];rgb=attrs[:,:3]
            features=torch.cat([rgb,rgb*ownership[:,0:1],rgb*ownership[:,1:2],ownership,z[:,None]*ownership],dim=1)
            assert features.shape[-1]==13
            view=torch.eye(4,device=device)[None];view[0,:3,:]=E[i]
            receipt['DGGT_raster_api_attempts']+=1;write(a.output/'receipt.json',receipt)
            with torch.inference_mode():image,alpha,_=rasterization(means=means,quats=attrs[:,7:11],scales=attrs[:,4:7],opacities=opacity,colors=features,
                viewmats=view,Ks=K[i:i+1],width=W,height=H,render_mode='RGB+ED')
            torch.cuda.synchronize();receipt['DGGT_raster_api_completed']+=1;receipt['DGGT_raster_F']+=int(view.shape[0])
            image=image[0].float().cpu().numpy();alpha=alpha[0].float().cpu().numpy();assert image.shape==(H,W,14)
            rgb=image[...,:3];s_rgb=image[...,3:6];d_rgb=image[...,6:9];s_alpha=image[...,9:10];d_alpha=image[...,10:11]
            expected=image[...,13:14];s_depth=np.divide(image[...,11:12],s_alpha,out=np.zeros_like(s_alpha),where=s_alpha>1e-8)
            d_depth=np.divide(image[...,12:13],d_alpha,out=np.zeros_like(d_alpha),where=d_alpha>1e-8)
            literal=alpha*rgb+(1-alpha)*sky[i];diagnostic=rgb+(1-alpha)*sky[i]
            data=dict(joint_RGB_premult=rgb,alpha=alpha,expected_camera_depth=expected,static_RGB_premult=s_rgb,dynamic_RGB_premult=d_rgb,
                static_alpha=s_alpha,dynamic_alpha=d_alpha,static_expected_depth=s_depth,dynamic_expected_depth=d_depth,
                static_depth_valid=s_alpha>1e-8,dynamic_depth_valid=d_alpha>1e-8,official_literal_RGB=literal,diagnostic_single_alpha_RGB=diagnostic)
            save_npz(folder/'layers_and_depth.npz',data,np)
            rgb_error=float(abs(rgb-s_rgb-d_rgb).max());alpha_error=float(abs(alpha-s_alpha-d_alpha).max())
            assert np.isfinite(image).all() and np.isfinite(alpha).all() and rgb_error<2e-5 and alpha_error<2e-5
            for name in ['joint_RGB_premult','alpha','static_RGB_premult','dynamic_RGB_premult','static_alpha','dynamic_alpha','official_literal_RGB','diagnostic_single_alpha_RGB']:png(folder/(name+'.png'),data[name])
            road=labels[i]==0;vehicle=np.isin(labels[i],[13,14,15,16,17,18])
            row=dict(raw=int(raw),gaussians=len(means),static_union_gaussians=count_static,current_dynamic_gaussians=len(dp),
                source_RGB_sum_max_error=rgb_error,source_alpha_sum_max_error=alpha_error,alpha_mean=float(alpha.mean()),
                literal_vs_single_alpha_RGB_MAE=float(abs(literal-diagnostic).mean()),
                input_view_literal_MAE=float(abs(literal-input_rgb[i]).mean()),input_view_single_alpha_MAE=float(abs(diagnostic-input_rgb[i]).mean()),
                proxy_road_pixels=int(road.sum()),proxy_vehicle_pixels=int(vehicle.sum()),
                proxy_road_single_alpha_MAE=float(abs(diagnostic-input_rgb[i])[road].mean()) if road.any() else None,
                proxy_vehicle_single_alpha_MAE=float(abs(diagnostic-input_rgb[i])[vehicle].mean()) if vehicle.any() else None,
                metrics_scope='input-view reconstruction only, masks are proxy; no held-out/ground-truth road claim')
            rows.append(row);saved.append(data);write(folder/'receipt.json',row)
        depths=np.concatenate([d['expected_camera_depth'][d['alpha']>1e-5] for d in saved]);depth_max=float(np.quantile(depths[depths>0],.99))
        for raw,data in zip(frames,saved):
            gray=np.clip(data['expected_camera_depth']/max(depth_max,1e-8),0,1);png(a.output/f'{int(raw):03d}/expected_depth_display.png',np.repeat(gray,3,-1))
        receipt.update(complete=True,phase='render_complete_not_quality_acceptance',rows=rows,
            feature_schema=['RGB3','static_premult_RGB3','dynamic_premult_RGB3','static_alpha','dynamic_alpha','static_depth_numerator','dynamic_depth_numerator','ED'],
            one_joint_raster_per_camera=True,layers_share_actual_joint_depth_order=True,source_depths_are_joint_visibility_weighted_expected_depth=True,
            depth_display_vmax=depth_max,depth_units='DGGT prediction gauge, not accepted metric/native depth',
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start)
        write(a.output/'receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    except BaseException as error:
        receipt.update(complete=False,phase='failed_preserved',error=repr(error),traceback=traceback.format_exc(),seconds=time.time()-start)
        write(a.output/'receipt.json',receipt);raise

if __name__=='__main__':main()
