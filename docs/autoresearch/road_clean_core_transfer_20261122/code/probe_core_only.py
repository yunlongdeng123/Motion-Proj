"""真正的3D核心子集渲染；不以2D评分mask遮图，不改变高斯参数。"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_clean_core_transfer_20261122_r4'
sys.path.insert(0,str(C/'road_source_fidelity_20261122_r2/code'))
import probe_source_chain as p
import camera_corrected_source as cam
h=p.h;torch=p.torch
h.ROOT=ROOT;h.native.ROOT=ROOT;h.native.m.ROOT=ROOT;h.native.m.OWNER=ROOT.name

def component(source,road,rgb,world=None,normal=None,quaternion=None):
    core=source['core_mask'];ids=source['world_row_ids'][core];T=road['model_from_world']
    moved=world is not None
    result=dict(active=np.ones(len(ids),bool),model_from_world=T.copy(),meter_to_model=road['meter_to_model'].copy(),
        model_xyz=world@T[:3,:3].T+T[:3,3] if moved else road['model_xyz'][ids].copy(),
        normal_world=source['normal_world'][core].copy() if normal is None else normal,
        gaussian_sigma_m=source['gaussian_sigma_m'][core].copy(),road_opacity=source['road_opacity'][core].copy(),
        reference_srgb=rgb[core].copy(),source_world_row_ids=ids.copy())
    if quaternion is not None:result.update(gaussian_rotation_world_wxyz=quaternion,gaussian_rotation_world_override=np.ones(len(ids),bool))
    return result

def main():
    out=ROOT/'source_core_probe'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    variants=[('source00_original','patch_00',None,[110,137]),
        ('source00_camera137','patch_00',C/'road_source_fidelity_20261122_r2/camera_corrected_00_137/source_camera_corrected_RGB.npz',[110,137]),
        ('source02_original','patch_02',None,[100,110,115,117]),
        ('source02_eight400','patch_02',C/'road_source_multiframe_20261122_r3/paired400_seed0/eight_frames/source_patch_02.npz',[100,110,115,117])]
    frames=sorted({raw for _,_,_,rows in variants for raw in rows});ctx=h.context(frames);torch.set_num_threads(2)
    from drivingeditor_dynamic_edit.v44_factory import RoadRows
    entries={e['candidate_id']:e for e in h.read(h.PRIOR/'source_selection/manifest.json')['candidates']}
    F=0;results=[]
    h.write(out/'contract.json',dict(task_id='ROAD-CLEAN-CORE-TRANSFER-20261122',stage='source_core_probe',
        definition='Only source world_row_ids where packet core_mask is true; halo rows not exported',
        render='Actual native RGBA on black and neutral(.32) backgrounds; no 2D mask applied to display',
        pixel_metrics='Diagnostic physical core eroded3px AND alpha>=.95; masks only used for metrics',
        prior_boundary='Earlier r2 transfer already copied core rows only; support halo was a source-gallery context',
        optimizer_updates=0,new_diffusion_calls=0,target_fusion=False,source_geometry_unchanged=True))
    def raster(raw,provider,background):
        nonlocal F
        packet=provider.rows(raw,ctx.camera(raw),ctx.scene.pipe)
        ctx.require_lease();ctx.native_calls+=1;F+=1
        return ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,
            torch.tensor(background,device='cuda',dtype=torch.float32))
    try:
      with torch.no_grad():
        for label,patch,material,raws in variants:
            entry=entries[patch];source=h.load(entry['packet']);rgb=source['reference_srgb'] if material is None else h.load(material)['reference_srgb']
            folder=out/label;folder.mkdir();asset=component(source,ctx.road,rgb)
            np.savez_compressed(folder/'road.npz',**asset)
            provider=RoadRows(folder/'road.npz','cuda');reloaded=RoadRows(folder/'road.npz','cuda')
            rows=[]
            for raw in raws:
                photo,project,path,K,T=p.projection(raw);grid,inverse,roundtrip=cam.brown_grid(raw);photo_pin=cam.warp(photo,grid)
                uv,depth=project(entry['world_roi_corners']);canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255)
                physical=np.asarray(canvas)>0
                lo=np.maximum(0,np.floor(uv.min(0)-20).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+21).astype(int))
                x0,y0=lo;x1,y1=hi;crop=np.s_[y0:y1,x0:x1]
                black=raster(raw,provider,[0,0,0]);grey=raster(raw,provider,[.32,.32,.32]);again=raster(raw,reloaded,[.32,.32,.32])
                alpha=black['alpha'].squeeze().cpu().numpy();premult=black['rgb'].cpu().numpy().transpose(1,2,0);composite=grey['rgb'].cpu().numpy().transpose(1,2,0)
                assert torch.equal(grey['rgb'],again['rgb']) and torch.equal(grey['alpha'],again['alpha'])
                score=binary_erosion(physical,iterations=3)&(alpha>=.95)
                effective=premult/np.maximum(alpha[:,:,None],1e-8)
                photo_chroma=np.ptp(photo_pin[score],axis=1);render_chroma=np.ptp(effective[score],axis=1)
                threshold=float(np.quantile(photo_chroma,.999)+.10) if score.any() else None
                anomaly=(render_chroma>threshold) if score.any() else np.array([],bool)
                view=folder/f'{raw:03d}';view.mkdir()
                for key,image in [('photo_pinhole',photo_pin),('core_black',premult),('core_grey',composite),('alpha',alpha)]:h.image(view/(key+'.png'),image[crop])
                Image.fromarray(np.uint8(score[crop])*255).save(view/'metric_mask.png')
                np.savez_compressed(view/'core_float.npz',photo_pinhole=photo_pin[crop],core_black=premult[crop],core_grey=composite[crop],
                    alpha=alpha[crop],physical_core=physical[crop],score_mask=score[crop],bbox_xyxy=np.r_[lo,hi])
                baseline_difference=None
                if material is None:
                    baseline=h.native.render(ctx,raw,subset_world=source['world_row_ids'][source['core_mask']]);F+=1
                    baseline_difference=float(abs(baseline['rgb'].cpu().numpy().transpose(1,2,0)-premult).max())
                rows.append(dict(raw=raw,core_n=int(physical.sum()),score_n=int(score.sum()),
                    high_chroma_threshold=threshold,high_chroma_pixels=int(anomaly.sum()),high_chroma_fraction=float(anomaly.mean()) if len(anomaly) else None,
                    reload_RGB_max_abs=0.,component_vs_parent_core_RGB_max_abs=baseline_difference,
                    output=str(view),bbox_xyxy=np.r_[lo,hi].tolist()))
            results.append(dict(label=label,patch=patch,core_rows=int(source['core_mask'].sum()),halo_rows_exported=0,
                material_input=str(material or entry['packet']),component=str(folder/'road.npz'),views=rows))
            h.write(out/'progress.json',dict(variants=results,nativeF=F,nativeB=0))
        h.write(out/'receipt.json',dict(complete=True,variants=results,nativeF=F,nativeB=0,
            optimizer_updates=0,new_diffusion_calls=0,original_asset_modified=False))
    finally:h.native.close()

if __name__=='__main__':main()
