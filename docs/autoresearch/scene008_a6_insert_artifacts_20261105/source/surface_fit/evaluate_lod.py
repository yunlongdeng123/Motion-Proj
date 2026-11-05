"""Fixed49 actual SurfaceLOD preservation; no fitting or parameter selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from fit_surface import CAM,ART,OLD,HELD,sha,write
from evaluate_combined import preflight

def main():
    dest=ART/'scale/surface_lod_v1/evaluation49';dest.mkdir(parents=True,exist_ok=False)
    root=ART/'surface_fit/combined_v1';checks=preflight(root)
    sys.path.insert(0,str(CAM/'scene008_a6_asset_controls_20261105/source'));sys.path.insert(0,str(ART/'source/scale'))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime,raster,tagged
    from relative_area_asset import RelativeAreaOpacityAsset
    from surface_lod_asset import SurfaceLODAsset
    from metrics import evaluate_frame,evaluate_pair,expected_depth,fixed_masks,metric,observation
    from evaluate_control import aggregate
    finalpath=root/'observed_fit/formal/complete_a6.npz';oldpath=OLD/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
    posterior=ART/'surface_fit/masked_depth_v1/posterior.npz'
    base=SHAsset.from_geometry(finalpath).cuda();old=SHAsset.from_geometry(oldpath).cuda();lod=SurfaceLODAsset(base,posterior)
    variants=dict(old_native=old,final_native=base,final_area=RelativeAreaOpacityAsset(base),surface_LOD=lod)
    comparisons=dict(LOD_vs_old_native=('surface_LOD','old_native'),LOD_vs_final_native=('surface_LOD','final_native'),LOD_vs_final_area=('surface_LOD','final_area'))
    records={k:[] for k in comparisons};lodrecords=[];rt=Runtime();start=time.time();conservation=0.
    hashes={str(p):sha(p) for p in [finalpath,oldpath,posterior,ART/'source/scale/surface_lod_asset.py']}
    def cpu(v):return v.detach().cpu().numpy()
    def scalar(v):
        a=cpu(v);return a[0] if a.ndim==3 else a
    def png(path,image):
        if torch.is_tensor(image):image=cpu(image)
        if image.ndim==3 and image.shape[0]==3:image=image.transpose(1,2,0)
        Image.fromarray(np.uint8(np.clip(image,0,1)*255+.5)).save(path)
    def strip(path,values,labels,crop,height=280):
        tiles=[]
        for val,label in zip(values,labels):
            im=Image.fromarray(np.uint8(np.clip(cpu(val).transpose(1,2,0),0,1)*255+.5)).crop(crop)
            im=im.resize((round(im.width*height/im.height),height),Image.Resampling.NEAREST)
            tile=Image.new('RGB',(im.width,height+32),(242,246,249));tile.paste(im,(0,32));ImageDraw.Draw(tile).text((6,9),label,fill=(20,35,45));tiles.append(tile)
        out=Image.new('RGB',(sum(t.width for t in tiles),height+32));x=0
        for tile in tiles:out.paste(tile,(x,0));x+=tile.width
        out.save(path)
    def solo(data,a):
        rows=tagged(a.rows(data['pose'],data['cam'],rt.pipe,1,precomputed=True),'asset')
        return raster(data['cam'],rt.pipe,0,rows,rt.n.bg.new_zeros(3))
    def extras(data,current,before,state):
        source=observation(data);m=fixed_masks(data);z=expected_depth(current,rt.scale);z0=expected_depth(before,rt.scale)
        a=scalar(current['alpha']);a0=scalar(before['alpha']);result={}
        for key,mask in dict(a37=m['a37_visible'],road=m['road_visible'],outside=m['outside'],observed_non_a6=m['observed_non_a6']).items():
            row=dict(alpha_parent_change=metric(abs(a-a0),mask),expected_depth_parent_change_m=metric(abs(z-z0),mask))
            if state=='keep' or key!='a37':
                measured=source['scene_depth_m'];valid=mask&np.isfinite(measured)&(measured>0)
                row.update(measured_expected_depth_error_m=metric(abs(z-measured),valid),baseline_measured_expected_depth_error_m=metric(abs(z0-measured),valid))
            for i,name in enumerate(('asset','a37','road','legacy_static')):
                row[name+'_alphaT_parent_change']=metric(abs(scalar(current['component_alpha_T'][i:i+1])-scalar(before['component_alpha_T'][i:i+1])),mask)
            result[key]=row
        if state=='keep':measured=source['depth_m'][1];valid=m['a37_depth'];key='a37_associated_LiDAR'
        else:measured=source['road_edit_depth_m'];valid=m['road_reobserved']&np.isfinite(measured)&(measured>0);key='road_reobserved_LiDAR'
        result[key]=dict(measured_depth_error_m=metric(abs(z-measured),valid),baseline_measured_depth_error_m=metric(abs(z0-measured),valid),parent_depth_change_m=metric(abs(z-z0),valid))
        return result
    with torch.no_grad():
        for raw in range(89,138):
            data=rt.raw(raw);packages={}
            for label,a in variants.items():
                s=solo(data,a);k=rt.render_scene(raw,a);d=rt.render_scene(raw,a,delete=True)
                assert k['meta']['native_a6_rows_input']==d['meta']['native_a6_rows_input']==0
                assert d['meta']['native_a37_packed_rows']==0
                packages[label]=(s,k,d)
                for out in (k,d):conservation=max(conservation,float((out['component_alpha_T'].sum(0)-out['alpha'][0]).abs().max()))
                if label=='surface_LOD':lodrecords.append(dict(raw=raw,**lod.last_lod_record))
            for label,(new,before) in comparisons.items():
                s,k,d=packages[new];s0,k0,d0=packages[before]
                records[label].append(dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',
                    solo=evaluate_frame(data,s,s0,rt.scale,'solo'),scene=evaluate_pair(data,k,d,k0,d0,rt.scale),
                    keep_fixed_depth_preservation=extras(data,k,k0,'keep'),delete_fixed_depth_preservation=extras(data,d,d0,'delete')))
            if raw in (105,106,113,130):
                folder=dest/f'raw{raw}';folder.mkdir()
                support=data['a6_visible'].bool()
                for s,_,_ in packages.values():support=support|(s['alpha'][0]>.01)
                y,x=torch.where(support);crop=[max(0,int(x.min())-8),max(0,int(y.min())-8),min(480,int(x.max())+9),min(320,int(y.max())+9)]
                labels=list(variants)
                for state,index in [('black',0),('keep',1),('delete',2)]:
                    strip(folder/(state+'_crop_comparison.png'),[packages[n][index]['rgb'] for n in labels],labels,crop)
                    # The street image/large crop retains raw113's old lower dark ROI.
                    if state!='black':
                        strip(folder/(state+'_street_comparison.png'),[packages[n][index]['rgb'] for n in labels],labels,(0,0,480,320),320)
                        strip(folder/(state+'_largecrop_comparison.png'),[packages[n][index]['rgb'] for n in labels],labels,(180,120,360,320),400)
                    for n in labels:png(folder/(n+'_'+state+'480.png'),packages[n][index]['rgb'])
                np.savez_compressed(folder/'actual_crop_replay.npz',crop_xyxy=crop,
                    **{n+'_'+state+'_'+key:cpu(packages[n][index][key][:,crop[1]:crop[3],crop[0]:crop[2]]) for n in labels for state,index in [('black',0),('keep',1),('delete',2)] for key in ('rgb','alpha','depth_sum')})
            print(json.dumps(dict(stage='fixed49_surface_LOD',raw=raw,complete=raw-88,du=lodrecords[-1]['du'])),flush=True)
            del packages,data;rt._data.clear();torch.cuda.empty_cache()
    assert conservation<2e-5 and rt.modelunchanged() and lod.assert_persistent_unchanged() and old.assert_geometry_unchanged()
    assert all(sha(p)==h for p,h in hashes.items())
    write(dest/'metrics.json',records);write(dest/'lod_records.json',lodrecords)
    summary={label:{split:aggregate(rows if split=='all49' else [r for r in rows if r['split']==split]) for split in ('all49','BUILD','HELD')} for label,rows in records.items()}
    write(dest/'summary.json',summary)
    write(dest/'receipt.json',dict(status='PASS_FIXED49_SURFACE_LOD_PRESERVATION',frames=49,checks=checks,
        source_sha256=sha(__file__),source_files_sha256=hashes,canonical_asset=str(finalpath),posterior=str(posterior),
        persistent_SF_hash=lod.persistent_SF_hash,surface_or_texture_or_camera_or_neural_parameters_fitted=False,
        canonical_asset_geometry_opacity_colours_unchanged=True,render_primitive_count_and_positions_camera_sampled=True,
        fixed_original_masks_denominators=True,native_neg_fov_unchanged=bool(rt.pipe.neg_fov),native_SO_unchanged=True,
        same_frozen_background_parameters=True,actual_RGB_alpha_depth_changes_measured=True,
        geometry_equality_between_render_samples_not_claimed=True,
        deleted_a37_original_RGB_depth_not_hidden_road_truth=True,
        HELD_excluded_from_recorded_colour_objective_but_previously_used_for_development=True,
        independent_untouched_generalization_claim=False,adoption_decision=False,training_or_colour_fit_steps=0,
        group_conservation_maxabs=conservation,seconds=time.time()-start,runtime=rt.meta))
    print(json.dumps(dict(status='PASS',output=str(dest),seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
