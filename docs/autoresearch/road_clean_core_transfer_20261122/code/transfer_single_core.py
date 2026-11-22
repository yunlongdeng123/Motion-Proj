"""单核心刚体迁移，保存独立三维组件；不叠加、不替换、不做接缝拟合。"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import probe_core_only as h4
h=h4.h;torch=h4.torch;ROOT=h4.ROOT;p=h4.p
sys.path.insert(0,str(h4.C/'road_gaussian_transfer_20261122_r2_isometric/code'))
from rigid_patch import transport_rigid
from gaussian_patch_transport import covariance_from_sigma_quaternion

def main():
    out=ROOT/'single_stamp_zero'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    target_record=h.read(ROOT/'target_audit/target_candidates.json')['recommended']
    target={k:np.asarray(v) if k in ['origin','frame','normal'] else v for k,v in target_record['target_frame'].items()}
    frames=[86,110,134,179,198];ctx=h.context(frames);torch.set_num_threads(2)
    from drivingeditor_dynamic_edit.v44_factory import RoadRows,_legacy_road_world_quaternions
    entries={e['candidate_id']:e for e in h.read(h.PRIOR/'source_selection/manifest.json')['candidates']}
    sources=h.read(ROOT/'source_core_probe/receipt.json')['variants'];F=0;rows=[]
    h.write(out/'contract.json',dict(task_id='ROAD-CLEAN-CORE-TRANSFER-20261122',stage='single_stamp_zero',
        target_selection=str(ROOT/'target_audit/target_candidates.json'),frames=frames,
        placement='One common target origin/frame for source00 .8m and source02 .6m; full existing core, no tiling or row clipping',
        transport='Existing locally-rigid world mean and covariance mapping; source RGB/sigma/opacity/internal3D layout preserved',
        output='Independent RoadRows-compatible road.npz component with source row IDs and transfer transform',
        display='Actual native RGBA on neutral grey or black; no 2D pixel mask',
        parent_asset_modified=False,parent_retirement=False,parent_stamp_compositing=False,
        target_material_is_same_position_observation=False,material_source='similar_road_inference',
        optimizer_updates=0,new_diffusion_calls=0,human_verdict=None,problem_solved=None))
    def raster(raw,provider,background):
        nonlocal F
        ctx.require_lease();ctx.native_calls+=1;F+=1
        return ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,provider.rows(raw,ctx.camera(raw),ctx.scene.pipe),
            torch.tensor(background,device='cuda',dtype=torch.float32))
    contexts={}
    try:
      with torch.no_grad():
        for raw in frames:
            parent=h.native.render(ctx,raw);F+=1
            contexts[raw]=parent['rgb'].cpu().numpy().transpose(1,2,0)
        for record in sources:
            label=record['label'];entry=entries[record['patch']];source=h.load(entry['packet']);core=source['core_mask']
            source_component=h.load(record['component']);rgb=source['reference_srgb'].copy();rgb[core]=source_component['reference_srgb']
            q=_legacy_road_world_quaternions(source['normal_world'][core],source['model_from_world'],float(ctx.road['meter_to_model']))
            moved=transport_rigid(source['world_xyz'][core],source['gaussian_sigma_m'][core],q,rgb[core],source['road_opacity'][core],
                source['world_row_ids'][core],source_origin=source['source_center_world'],source_frame=source['source_tangent_frame_world'][:3,:3],
                target=target,patch_id=0,source_asset_id=record['component'])
            asset=h4.component(source,ctx.road,rgb,world=moved['world_xyz'],normal=moved['normal_world'],quaternion=moved['gaussian_rotation_world_wxyz'])
            assert np.array_equal(asset['reference_srgb'],source_component['reference_srgb'])
            assert np.array_equal(asset['road_opacity'],source_component['road_opacity']) and np.array_equal(asset['gaussian_sigma_m'],source_component['gaussian_sigma_m'])
            R=target['frame']@source['source_tangent_frame_world'][:3,:3].T
            corners=(np.asarray(entry['world_roi_corners'])-source['source_center_world'])@R.T+target['origin']
            folder=out/label;folder.mkdir();np.savez_compressed(folder/'road.npz',**asset)
            np.savez_compressed(folder/'transfer_trace.npz',source_world_row_ids=source['world_row_ids'][core],
                source_world_xyz=source['world_xyz'][core],target_world_xyz=moved['world_xyz'],rotation_world=R,
                source_origin=source['source_center_world'],target_origin=target['origin'],source_frame=source['source_tangent_frame_world'][:3,:3],
                target_frame=target['frame'],target_core_corners=corners,source_RGB=source_component['reference_srgb'],
                covariance_world=moved['covariance_world'],source_kind=np.asarray('similar_road_inference'))
            h.write(folder/'provenance.json',dict(source_component=record['component'],source_patch=record['patch'],
                material_input=record['material_input'],target_origin=target['origin'].tolist(),rotation_world=R.tolist(),
                source_core_rows=len(asset['active']),source_halo_rows=0,scale=1.,optimizer_updates=0,material_source='similar_road_inference'))
            provider=RoadRows(folder/'road.npz','cuda');again=RoadRows(folder/'road.npz','cuda');views=[]
            for raw in frames:
                photo,project,path,K,T=p.projection(raw);uv,depth=project(corners)
                if (depth<=0).any():raise ValueError('Target behind camera')
                canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255);physical=np.asarray(canvas)>0
                lo=np.maximum(0,np.floor(uv.min(0)-20).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+21).astype(int));x0,y0=lo;x1,y1=hi
                if x1<=x0 or y1<=y0:raise ValueError('Target outside view')
                crop=np.s_[y0:y1,x0:x1];view=folder/f'{raw:03d}';view.mkdir()
                black=raster(raw,provider,[0,0,0]);grey=raster(raw,provider,[.32,.32,.32]);reload=raster(raw,again,[.32,.32,.32])
                assert torch.equal(grey['rgb'],reload['rgb']) and torch.equal(grey['alpha'],reload['alpha'])
                alpha=black['alpha'].squeeze().cpu().numpy();brgb=black['rgb'].cpu().numpy().transpose(1,2,0);grgb=grey['rgb'].cpu().numpy().transpose(1,2,0)
                score=binary_erosion(physical,iterations=3)&(alpha>=.95);effective=brgb/np.maximum(alpha[:,:,None],1e-8)
                chroma=np.ptp(effective[score],axis=1);threshold=.15
                for key,img in [('target_black',brgb),('target_grey',grgb),('alpha',alpha),('parent_road_context',contexts[raw])]:h.image(view/(key+'.png'),img[crop])
                h.image(view/'target_full_grey.png',grgb)
                context_image=Image.fromarray(np.rint(np.clip(contexts[raw],0,1)*255).astype(np.uint8));draw=ImageDraw.Draw(context_image)
                draw.line([tuple(v) for v in np.vstack((uv,uv[:1]))],fill=(80,230,210),width=2);context_image.save(view/'parent_footprint_locator.png')
                Image.fromarray(np.uint8(score[crop])*255).save(view/'metric_mask.png')
                np.savez_compressed(view/'target_float.npz',target_black=brgb[crop],target_grey=grgb[crop],alpha=alpha[crop],
                    parent_road_context=contexts[raw][crop],physical_core=physical[crop],score_mask=score[crop],bbox_xyxy=np.r_[lo,hi])
                views.append(dict(raw=raw,physical_core_pixels=int(physical.sum()),opaque_core_pixels=int(score.sum()),
                    alpha95_fraction=float((physical&(alpha>=.95)).sum()/max(1,physical.sum())),
                    high_chroma_threshold=threshold,high_chroma_pixels=int((chroma>threshold).sum()),
                    high_chroma_fraction=float((chroma>threshold).mean()) if len(chroma) else None,
                    reload_RGB_max_abs=0.,output=str(view),bbox_xyxy=np.r_[lo,hi].tolist()))
            rows.append(dict(label=label,patch=record['patch'],core_rows=len(asset['active']),halo_rows=0,source_RGB_sigma_opacity_exact=True,
                internal_pairwise_distances_rigid=True,covariance_reconstruction_max_abs=float(moved['trace']['covariance_reconstruction_max_abs'].max()),
                component=str(folder/'road.npz'),views=views))
            h.write(out/'progress.json',dict(variants=rows,nativeF=F,nativeB=0))
        h.write(out/'receipt.json',dict(complete=True,variants=rows,target_origin=target['origin'].tolist(),nativeF=F,nativeB=0,
            optimizer_updates=0,new_diffusion_calls=0,parent_asset_modified=False,parent_retirement=False,parent_stamp_compositing=False))
    finally:h.native.close()

if __name__=='__main__':main()
