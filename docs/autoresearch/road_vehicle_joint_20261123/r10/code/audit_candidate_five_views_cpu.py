"""Five-view CPU geometric projection of the explicit local candidate, with fixed denominators."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design


def png(path,rgb):
    Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10'
    geo_path=R/'audit/local_surface_candidate_v2/candidate_geometry.npz';geo=load(geo_path)
    spec=importlib.util.spec_from_file_location('candidate_local_projector',R/'code/local_surface_projection.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    contact=read(R/'audit/ground_contact_prior_raw_GT_v2/ground_contact_prior_audit.json')
    rows=[];previews=[];Wcell=geo['cell_asset_mask'].shape[1];Ncell=geo['cell_asset_mask'].size
    for raw in [86,110,134,179,198]:
        folder=out/f'{raw:03d}';folder.mkdir();cp=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz';v=load(cp)
        fixedpath=R/f'fixed_eval_v1_replay1/{raw:03d}/fixed_eval.npz';f=load(fixedpath)
        pr=mod.raster_surface(geo['world_xyz'],geo['uv_global'],geo['candidate_mask'],v['world_to_camera'],v['camera_K'],v['projection_valid'].shape[::-1])
        ids=np.maximum(pr['triangle_index'],0)%Ncell
        asset=pr['projection_valid']&geo['cell_asset_mask'].ravel()[ids]
        reconstruction=pr['projection_valid']&geo['cell_reconstruction_mask'].ravel()[ids]
        ring=asset&~reconstruction;body=f['body_protected'];K=f['K_protected']
        visible=asset&~body&~K
        photo=np.clip(v['photo_pinhole_srgb'],0,1);H,W=asset.shape
        yy,xx=np.mgrid[:H,:W];background=np.where((((xx//8)+(yy//8))%2)[...,None],.11,.17)*np.ones((H,W,3))
        surface=background.copy();surface[reconstruction]=[.55,.61,.64];surface[ring]=[.25,.46,.51]
        uv=pr['atlas_uv_global'];gridline=(abs(uv[...,0]-np.rint(uv[...,0]/10)*10)<.55)|(abs(uv[...,1]-np.rint(uv[...,1]/10)*10)<.55)
        surface[asset&gridline]=[.75,.83,.85]
        overlay=photo.copy();overlay[visible]=.65*photo[visible]+.35*surface[visible]
        overlay[visible&gridline]=[.12,.80,.73]
        edge=asset&~binary_erosion(asset);overlay[edge&~body&~K]=[1.,.65,.1]
        overlay_image=Image.fromarray(np.rint(overlay*255).astype(np.uint8));draw=ImageDraw.Draw(overlay_image)
        contacts=[]
        for row in contact['GT_bottom_rows']:
            if row['raw']!=raw or not row.get('present'):continue
            points=np.asarray(row['bottom_world_corners_and_centre']);E,camK=v['world_to_camera'],v['camera_K'];cam=points@E[:3,:3].T+E[:3,3];q=cam@camK.T;screen=q[:,:2]/cam[:,2,None]-.5
            colour=(60,235,70) if row['alias']=='a20' else (235,80,215)
            if (cam[:,2]>.1).all():
                order=[0,1,3,2,0];draw.line([tuple(screen[j]) for j in order],fill=colour,width=2)
            roadZ=design(points[:,:2],geo['fit_XY_centre'])@geo['fit_coefficients']
            contacts.append(dict(alias=row['alias'],GT_bottom_world_points=points.tolist(),candidate_road_minus_GT_bottom_Z_m=(roadZ-points[:,2]).tolist(),
                GT_bottom_is_indirect_crosscheck_not_dense_road_truth=True))
        png(folder/'original_pinhole_photo.png',photo);png(folder/'candidate_geometry_surface_only.png',surface)
        overlay_image.save(folder/'candidate_geometry_photo_context.png')
        depth=np.zeros((H,W,3));d=pr['camera_depth_m'];t=np.clip((d-10)/25,0,1);depth[asset]=np.stack([t,1-abs(2*t-1),1-t],-1)[asset]
        png(folder/'candidate_geometry_depth.png',depth)
        np.savez_compressed(folder/'projection_and_scope.npz',**pr,asset_scope_projected=asset,reconstruction_scope_projected=reconstruction,
            blend_scope_projected=ring,protected_body=body,protected_K=K,display_context_visible=visible)
        regions={}
        for name in ['full_original_band','ground_band_before_support','ground_eval','red_diagnostic','blue_cyan_diagnostic','dark_context_diagnostic']:
            mask=f[name];partition=dict(no_candidate_context=int((mask&~pr['projection_valid']).sum()),
                candidate_context_outside_asset_scope=int((mask&pr['projection_valid']&~asset).sum()),
                asset_behind_body_or_K=int((mask&asset&(body|K)).sum()),asset_display_visible=int((mask&visible).sum()))
            assert sum(partition.values())==int(mask.sum())
            regions[name]=dict(fixed_pixels=int(mask.sum()),candidate_context_supported=int((mask&pr['projection_valid']).sum()),
                explicit_asset_scope_supported=int((mask&asset).sum()),partition=partition)
        row=dict(raw=raw,camera=pin(cp),fixed_evaluation=pin(fixedpath),regions=regions,GT_contact=contacts,
            default_context_photo_is_display_only=True,geometric_grid_is_not_RGB_material=True,
            PNGs={name:pin(folder/name) for name in ['original_pinhole_photo.png','candidate_geometry_surface_only.png','candidate_geometry_photo_context.png','candidate_geometry_depth.png']})
        write(folder/'geometry_projection_receipt.json',row);rows.append(row)
        pair=Image.new('RGB',(1152,410),(20,23,30));pair.paste(Image.open(folder/'original_pinhole_photo.png').resize((576,384)),(0,26));pair.paste(overlay_image.resize((576,384)),(576,26));ImageDraw.Draw(pair).text((8,7),f'raw{raw:03d}  original pinhole photo | inferred geometry grid + GT bottom outlines',fill=(235,235,235));previews.append(pair)
        print(json.dumps(dict(stage='five_view_projection',raw=raw,regions=regions)),flush=True)
    contact_sheet=Image.new('RGB',(1152,410*len(previews)),(20,23,30))
    for j,im in enumerate(previews):contact_sheet.paste(im,(0,410*j))
    contact_sheet.save(out/'five_view_geometry_contactsheet.png')
    result=dict(schema='R10.local_geometry_candidate_five_view_CPU_projection.v1',complete=True,decision='INDEPENDENT_GEOMETRY_VISUAL_REVIEW_PENDING',
        candidate=pin(geo_path),rows=rows,contactsheet=pin(out/'five_view_geometry_contactsheet.png'),
        shader='neutral geometric grid only, no teacher/photo RGB used as new material',
        overlay='original pinhole photo as context; body/K only excluded in display, actual projected geometry scope and all fixed denominators preserved',
        root_geometry_acceptance='PENDING',new_Gaussian_asset_created=False,new_GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'five_view_geometry_receipt.json',result);print(json.dumps(dict(complete=True,receipt=pin(out/'five_view_geometry_receipt.json'))),flush=True)


if __name__=='__main__':main()
