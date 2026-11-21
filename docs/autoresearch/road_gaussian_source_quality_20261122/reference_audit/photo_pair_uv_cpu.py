"""Approximate photo-to-photo consistency on a common metric chart; CPU only."""
import json
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.interpolate import LinearNDInterpolator
from audit_source_quality import BASE, OUT, load, cosine, sha


def bands(a,b,mask):
    rows=[]
    for world_sigma in [.005,.01,.02]:
        sigma=world_sigma/.005
        ah=(a-gaussian_filter(a,(sigma,sigma,0)))[mask]
        bh=(b-gaussian_filter(b,(sigma,sigma,0)))[mask]
        ap,bp=np.mean(ah*ah),np.mean(bh*bh)
        rows.append(dict(highpass_sigma_m=world_sigma,cosine=cosine(ah,bh),
            first_RMS=float(np.sqrt(ap)),second_RMS=float(np.sqrt(bp)),
            second_to_first_RMS=float(np.sqrt(bp/ap)),residual_RMS=float(np.sqrt(np.mean((ah-bh)**2)))))
    return rows


def sample(image,uv,order=1):
    if image.ndim==2:return map_coordinates(image,[uv[:,1],uv[:,0]],order=order,mode='constant',cval=0)
    return np.stack([map_coordinates(image[:,:,k],[uv[:,1],uv[:,0]],order=order,mode='constant',cval=0)
        for k in range(image.shape[2])],axis=1)


def main():
    plan_path=OUT.parent/'source_view_ranking/source00_probe_plan.json'
    plan=json.loads(plan_path.read_text()); records={row['raw']:row for row in plan['records']}
    source=load(BASE/'source_selection/candidates/patch_00.npz')
    B=source['source_tangent_frame_world'][:3,:3];origin=source['source_center_world']
    local=(source['world_xyz'][source['core_mask']]-origin)@B
    grid=(np.arange(160)+.5)*.005-.4
    u,v=np.meshgrid(grid,grid);flat=np.c_[u.ravel(),v.ravel()]
    old_geometry=LinearNDInterpolator(local[:,:2],local[:,2])(flat)
    finite=np.isfinite(old_geometry);old_geometry=np.nan_to_num(old_geometry)
    physical_inner=(abs(u)<=.32)&(abs(v)<=.32)
    refit_receipt=json.loads((OUT.parent/'source_rgb_refit/receipt.json').read_text())
    before={row['raw']:row for row in refit_receipt['original']}
    results=[]
    for mode,height in [('source_chart_plane_approximation',np.zeros(len(flat))),('original_GS_center_piecewise_height_approximation',old_geometry)]:
        world=origin+np.c_[flat,height]@B.T
        views={}
        for raw in [98,110,137,173]:
            row=records[raw];T=np.array(row['world_to_camera']);K=np.array(row['camera_K_native'])
            camera=world@T[:3,:3].T+T[:3,3];h=camera@K.T;uv=h[:,:2]/camera[:,2:]-.5
            photo=np.asarray(Image.open(row['original_RGB']).convert('RGB'),np.float64)/255
            legal=np.asarray(Image.open(row['core_photo_valid_mask']))>0
            valid=sample(legal.astype(float),uv,order=0)>.5
            valid&=camera[:,2]>0
            if mode!='source_chart_plane_approximation':valid&=finite
            data=load(OUT.parent/f'source_rgb_refit/original/{raw:03d}/core_float.npz')
            box=before[raw]['bbox_xyxy'];gs_uv=uv-np.array(box[:2])
            views[raw]=dict(photo=sample(photo,uv).reshape(160,160,3),
                GS=sample(data['isolated_rgb'].astype(float),gs_uv).reshape(160,160,3),
                valid=valid.reshape(160,160))
        for first,second in [(137,173),(98,110)]:
            a,b=views[first],views[second];mask=a['valid']&b['valid']&physical_inner
            assert mask.sum()>100
            mean_a=a['photo'][mask].mean(0);mean_b=b['photo'][mask].mean(0)
            entry=dict(mapping=mode,first_raw=first,second_raw=second,common_UV_grid_cells=int(mask.sum()),
                grid_cell_m=.005,independent_native_pixels=False,
                photo_pair_RGB_MAE=float(np.abs(a['photo'][mask]-b['photo'][mask]).mean()),
                photo_pair_mean_centered_RGB_MSE=float(np.mean(((a['photo'][mask]-mean_a)-(b['photo'][mask]-mean_b))**2)),
                photo_pair_mean_RGB_delta=(mean_b-mean_a).tolist(),
                photo_pair_HF=bands(a['photo'],b['photo'],mask),
                first_original_GS_to_photo_HF=bands(a['photo'],a['GS'],mask),
                second_original_GS_to_photo_HF=bands(b['photo'],b['GS'],mask))
            results.append(entry)
            np.savez_compressed(OUT/f'photo_pair_{first}_{second}_{mode}.npz',UV_u_m=u,UV_v_m=v,
                first_photo=a['photo'],second_photo=b['photo'],common_mask=mask)
    result=dict(schema='source_photo_pair_common_UV_CPU_diagnostic.v1',GPU_calls=0,optimizer_updates=0,
        new_diffusion_calls=0,geometry_or_asset_modified=False,
        contract='Existing source chart plane or piecewise-linear height through original GS centers; inherited pinhole cameras; bilinear RGB and nearest photo-mask sampling; common valid core, inset0.08m for filter support.',
        interpretation='Approximate registered-photo consistency only. It is neither reconstructed geometry nor ground truth. 5mm UV cells oversample the grazing-angle original photos and are not independent observations. No shift search, alignment optimization or camera refit.',
        fine_frequency_limit='5mm highpass is below the coarsest source pixel spacing in the grazing direction; sensor/JPEG/interpolation and approximate geometry can dominate. Compare scales jointly.',
        baseline_137_173_m=plan['repeated_query_137_to173_camera_baseline_m'],
        records=results,inputs=[dict(path=str(plan_path),sha256=sha(plan_path))])
    path=OUT/'photo_pair_uv_consistency.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),records=results),indent=2))


if __name__=='__main__':main()
