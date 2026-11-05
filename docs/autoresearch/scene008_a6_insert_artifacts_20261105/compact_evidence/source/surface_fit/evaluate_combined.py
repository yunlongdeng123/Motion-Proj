"""One fixed49 geometry/RGBA audit, with separate native and area viewport controls.

Targets/masks are original measured observations; no alpha-based denominator
shrinking. Frozen scene parameters do not imply unchanged composited images.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from fit_surface import bank,sha,same,write,APPEARANCE,CAM,OLD,ART,HELD

CONTROLS=CAM/'scene008_a6_asset_controls_20261105'
RAW=CAM/'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
PREVIEW=(105,106,113,130)

def preflight(root):
    init=bank(root/'combined_initial.npz');final=bank(root/'observed_fit/formal/complete_a6.npz')
    mapping=bank(root/'row_mapping.npz');post=bank(ART/'surface_fit/masked_depth_v1/posterior.npz')
    prototype=bank(OLD/'output_adapter/projective_surface/projective_surface_asset.npz')
    chart=bank(OLD/'output_adapter/projective_surface/reference_chart.npz')
    m=bank(ART/'vehicle/matte_contract/matte_texture_samples_filled_v2.npz')
    inputs=bank(root/'observed_fit/inputs.npz');unknown=inputs['unknown']
    for k in set(init)-APPEARANCE:assert same(init[k],final[k]),'Frozen colour-fit field changed:'+k
    for k in ('canonical_sh','rgb','colour_inferred'):assert same(init[k][unknown],final[k][unknown]),'Unknown colour changed:'+k
    assert same(init['canonical_sh'][:,1:],final['canonical_sh'][:,1:])
    original=mapping['original_projective_asset_rows'];added=init['representation_kind']==1;retained=~added
    assert np.array_equal(original,post['query_original_asset_rows'][mapping['source_depth_rows_keep']])
    assert np.array_equal(init['resample_reference_pixel_xy'][added],prototype['resample_reference_pixel_xy'][original[added]])
    assert np.array_equal(init['resample_source_row'],prototype['resample_source_row'][original])
    assert np.array_equal(init['resample_source_parent'],prototype['resample_source_parent'][original])
    for k in ('xyz','normal','sigma','opacity','thickness','source_index','sampling_parent_index','sampling_child_index'):
        assert same(init[k][retained],prototype[k][original[retained]]),'Retained source changed:'+k
    cut=int((prototype['representation_kind']==0).sum());q=original[added]-cut
    assert np.array_equal(m['original_asset_row'][q],original[added])
    real=init['resample_rgb_real'][added]
    assert np.array_equal(real,m['real106_alwayskeep'][q])
    assert np.all(init['opacity'][added][real]==np.float32(.99))
    assert np.array_equal(init['opacity'][added][~real],np.float32(.99)*m['sampled_alpha'][q][~real])
    assert (m['effective_allowed_real_priority'][q]).all()
    assert (init['opacity'][added]>0).all()
    # Exported float32 camera projection, independent of the native rasterizer.
    local=init['xyz'][added].astype(float);c=post['reference_camera_from_local_m'];K=post['reference_K']
    xyz=local@c[:3,:3].T+c[:3,3];xy=(xyz@K.T)[:,:2]/xyz[:,2,None]-.5
    reprojection=float(abs(xy-init['resample_reference_pixel_xy'][added]).max())
    assert reprojection<1e-3
    return dict(final_geometry_opacity_provenance_exact_to_combined_initial=True,
        unknown_colours_exact_to_matte_initialization=True,unknown_columns=int(unknown.sum()),
        higher_SH_coefficients_exact=True,retained_source_geometry_exact=True,
        source_parent_IDs_original_reference_pixels_verified=True,matte_alpha_contract_verified=True,
        reference_float32_projection_maxabs_HRpx=reprojection,rows=len(init['xyz']),
        geometry_changed_vs_old_projective=True,opacity_changed_vs_old_projective=True,
        PVD_model_weights_changed=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ART/'surface_fit/combined_v1'))
    p.add_argument('--output');args=p.parse_args();root=Path(args.root)
    dest=Path(args.output) if args.output else root/'observed_fit/evaluation'
    dest.mkdir(parents=True,exist_ok=False);checks=preflight(root)
    sys.path.insert(0,str(CONTROLS/'source'));sys.path.insert(0,str(ART/'source/scale'))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime,raster,tagged
    from metrics import evaluate_frame,evaluate_pair,expected_depth,fixed_masks,metric,observation
    from evaluate_control import aggregate
    from relative_area_asset import RelativeAreaOpacityAsset
    finalpath=root/'observed_fit/formal/complete_a6.npz'
    oldpath=OLD/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
    thinpath=CONTROLS/'m2/complete_a6.npz'
    paths=dict(final=finalpath,old=oldpath,thin=thinpath,initial=root/'combined_initial.npz')
    hashes={k:sha(v) for k,v in paths.items()}
    bases={k:SHAsset.from_geometry(v).cuda() for k,v in paths.items()}
    variants=dict(final_native=bases['final'],final_area=RelativeAreaOpacityAsset(bases['final']),
        old_native=bases['old'],old_area=RelativeAreaOpacityAsset(bases['old']),thin_m2=bases['thin'])
    comparisons=dict(native_vs_old=('final_native','old_native'),area_vs_old_area=('final_area','old_area'),
        native_vs_thin=('final_native','thin_m2'),area_vs_thin=('final_area','thin_m2'),
        viewport_area_vs_native=('final_area','final_native'))
    records={k:[] for k in comparisons};hr_records=[];area_records=[];rt=Runtime();start=time.time();conservation=0.
    def cpu(v):return v.detach().cpu().numpy()
    def scalar(v):
        a=cpu(v);return a[0] if a.ndim==3 else a
    def png(path,value):
        a=cpu(value) if torch.is_tensor(value) else value
        if a.ndim==3 and a.shape[0]==3:a=a.transpose(1,2,0)
        Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).save(path)
    def strip(path,values,labels,crop=None,height=250):
        tiles=[]
        for a,label in zip(values,labels):
            if torch.is_tensor(a):a=cpu(a)
            if a.shape[0]==3:a=a.transpose(1,2,0)
            im=Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))
            if crop is not None:im=im.crop(crop)
            im=im.resize((round(im.width*height/im.height),height),Image.Resampling.NEAREST)
            tile=Image.new('RGB',(im.width,height+32),(242,246,249));tile.paste(im,(0,32));ImageDraw.Draw(tile).text((6,9),label,fill=(20,35,45));tiles.append(tile)
        out=Image.new('RGB',(sum(t.width for t in tiles),height+32));x=0
        for tile in tiles:out.paste(tile,(x,0));x+=tile.width
        out.save(path)
    def solo(data,a,camera):
        rows=tagged(a.rows(data['pose'],camera,rt.pipe,1,precomputed=True),'asset')
        return raster(camera,rt.pipe,0,rows,rt.n.bg.new_zeros(3))
    def extras(data,current,before,state):
        source=observation(data);m=fixed_masks(data);z=expected_depth(current,rt.scale);z0=expected_depth(before,rt.scale)
        a=scalar(current['alpha']);a0=scalar(before['alpha']);result={}
        for key,mask in dict(a37=m['a37_visible'],road=m['road_visible'],outside=m['outside'],observed_non_a6=m['observed_non_a6']).items():
            row=dict(alpha_parent_change=metric(abs(a-a0),mask),expected_depth_parent_change_m=metric(abs(z-z0),mask))
            # Deleted a37 pixels have no factual same-frame hidden-road truth.
            if state=='keep' or key!='a37':
                measured=source['scene_depth_m'];valid=mask&np.isfinite(measured)&(measured>0)
                row.update(measured_expected_depth_error_m=metric(abs(z-measured),valid),baseline_measured_expected_depth_error_m=metric(abs(z0-measured),valid))
            for i,name in enumerate(('asset','a37','road','legacy_static')):
                row[name+'_alphaT_parent_change']=metric(abs(scalar(current['component_alpha_T'][i:i+1])-scalar(before['component_alpha_T'][i:i+1])),mask)
            result[key]=row
        if state=='keep':
            measured=source['depth_m'][1];valid=m['a37_depth'];key='a37_associated_LiDAR'
        else:
            measured=source['road_edit_depth_m'];valid=m['road_reobserved']&np.isfinite(measured)&(measured>0);key='road_reobserved_LiDAR'
        result[key]=dict(measured_depth_error_m=metric(abs(z-measured),valid),baseline_measured_depth_error_m=metric(abs(z0-measured),valid),parent_depth_change_m=metric(abs(z-z0),valid))
        return result
    with torch.no_grad():
        for raw in range(89,138):
            data=rt.raw(raw);packages={}
            for label,a in variants.items():
                s=solo(data,a,data['cam']);keep=rt.render_scene(raw,a);deleted=rt.render_scene(raw,a,delete=True)
                assert keep['meta']['native_a6_rows_input']==deleted['meta']['native_a6_rows_input']==0
                assert deleted['meta']['native_a37_packed_rows']==0
                packages[label]=(s,keep,deleted)
                for out in (keep,deleted):conservation=max(conservation,float((out['component_alpha_T'].sum(0)-out['alpha'][0]).abs().max()))
                if label.endswith('_area'):area_records.append(dict(raw=raw,label=label,**a.last_area_record))
            for label,(new,old) in comparisons.items():
                s,k,d=packages[new];s0,k0,d0=packages[old]
                records[label].append(dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',
                    solo=evaluate_frame(data,s,s0,rt.scale,'solo'),scene=evaluate_pair(data,k,d,k0,d0,rt.scale),
                    keep_fixed_depth_preservation=extras(data,k,k0,'keep'),delete_fixed_depth_preservation=extras(data,d,d0,'delete')))
            if raw in PREVIEW:
                folder=dest/f'raw{raw}';folder.mkdir()
                support=data['a6_visible'].bool()
                for s,_,_ in packages.values():support=support|(s['alpha'][0]>.01)
                y,x=torch.where(support);crop=[max(0,int(x.min())-8),max(0,int(y.min())-8),min(480,int(x.max())+9),min(320,int(y.max())+9)]
                labels=list(variants)
                strip(folder/'black_native480_comparison.png',[packages[n][0]['rgb'] for n in labels],labels,crop)
                for state,index in [('keep',1),('delete',2)]:
                    strip(folder/(state+'_street480_comparison.png'),[packages[n][index]['rgb'] for n in labels],labels,None,320)
                    strip(folder/(state+'_crop480_comparison.png'),[packages[n][index]['rgb'] for n in labels],labels,crop)
                for label,(s,k,d) in packages.items():
                    png(folder/(label+'_black480.png'),s['rgb']);png(folder/(label+'_keep480.png'),k['rgb']);png(folder/(label+'_delete480.png'),d['rgb'])
                np.savez_compressed(folder/'native480_replay.npz',crop_xyxy=crop,
                    **{label+'_'+state+'_'+key:cpu(packages[label][index][key]) for label in labels for state,index in [('solo',0),('keep',1),('delete',2)] for key in ('rgb','alpha','depth_sum')})
            # HR black-only observations/details. Area wrapper is exactly identity.
            c=copy.copy(data['cam']);c.image_width=1920;c.image_height=1280
            for key in ('fx','fy','cx','cy'):setattr(c,key,getattr(c,key)*4)
            hr={k:solo(data,bases[k],c) for k in ('final','old','thin')}
            if raw in PREVIEW:hr['initial']=solo(data,bases['initial'],c)
            hrwrap=solo(data,variants['final_area'],c)
            assert all(torch.equal(hrwrap[k],hr['final'][k]) for k in ('rgb','alpha','depth_sum'))
            source=np.asarray(Image.open(RAW/'image/image_0'/f'0008{raw:03d}.png').convert('RGB'),dtype=np.float32)/255
            low=binary_erosion(data['a6_visible'].cpu().numpy(),iterations=1);fixed=np.repeat(np.repeat(low,4,0),4,1)
            # Old M2 gate is a fixed parent domain, never a new-chart coverage test.
            fixed=fixed&(scalar(hr['thin']['alpha'])>.9)
            hrow=dict(raw=raw,split='HELD' if raw in HELD else 'BUILD')
            for label,out in hr.items():
                color=cpu(out['rgb']).transpose(1,2,0);alpha=scalar(out['alpha'])
                hrow[label]=dict(premultiplied_RGB_error=metric(abs(color-source).mean(-1),fixed),
                    straight_RGB_error=metric(abs(color/np.maximum(alpha[:,:,None],1e-6)-source).mean(-1),fixed),
                    alpha=metric(alpha,fixed),coverage_below09=metric((alpha<=.9).astype(float),fixed),
                    zero_coverage=metric((alpha<=1e-6).astype(float),fixed))
            hr_records.append(hrow)
            if raw in PREVIEW:
                mask=(hr['final']['alpha'][0]>.01)|(hr['old']['alpha'][0]>.01)
                y,x=torch.where(mask);crop=[max(0,int(x.min())-16),max(0,int(y.min())-16),min(1920,int(x.max())+17),min(1280,int(y.max())+17)]
                labels=['thin','old','initial','final']
                strip(folder/'black_HR_comparison.png',[hr[n]['rgb'] for n in labels],['thin M2','old rough-depth formal','new C2 matte initial','new C2 matte formal'],crop,300)
                for label in labels:
                    png(folder/(label+'_black1920.png'),hr[label]['rgb'])
                    x0,y0,x1,y1=crop;png(folder/(label+'_blackHR_crop.png'),hr[label]['rgb'][:,y0:y1,x0:x1])
                np.savez_compressed(folder/'HR_black_replay.npz',crop_xyxy=crop,
                    **{label+'_'+key:cpu(hr[label][key][:,crop[1]:crop[3],crop[0]:crop[2]]) for label in labels for key in ('rgb','alpha','depth_sum')})
            print(json.dumps(dict(stage='fixed49_combined_RGBA',raw=raw,complete=raw-88)),flush=True)
            del packages,data,hr,hrwrap;rt._data.clear();torch.cuda.empty_cache()
    assert conservation<2e-5 and rt.modelunchanged()
    assert all(a.assert_geometry_unchanged() for a in bases.values())
    assert all(sha(paths[k])==h for k,h in hashes.items())
    write(dest/'metrics.json',records);write(dest/'HR_metrics.json',hr_records);write(dest/'area_records.json',area_records)
    def summarized(rows):return {split:aggregate(rows if split=='all49' else [r for r in rows if r['split']==split]) for split in ('all49','BUILD','HELD')}
    write(dest/'summary.json',{k:summarized(v) for k,v in records.items()});write(dest/'HR_summary.json',summarized(hr_records))
    write(dest/'receipt.json',dict(status='PASS_FIXED49_COMBINED_RGBA_EVALUATION',frames=49,checks=checks,source_sha256=sha(__file__),
        assets={k:dict(path=str(v),sha256=hashes[k],rows=len(bases[k].xyz),geometry_hash=bases[k].geometry_hash) for k,v in paths.items()},
        area_wrapper_path=str(ART/'source/scale/relative_area_asset.py'),area_wrapper_sha256=sha(ART/'source/scale/relative_area_asset.py'),
        native_neg_fov_unchanged=bool(rt.pipe.neg_fov),native_SO_unchanged=True,
        main_scene_viewport=[480,320],detail_black_viewport=[1920,1280],HR_area_wrapper_bitwise_identity_all49=True,
        fixed_original_observation_masks_no_new_alpha_denominator_shrink=True,
        HR_domain='Original extra-eroded visible4x and thinM2 alpha>.9; same BUILD394206 training pixels',
        same_frozen_background_parameters=True,actual_RGB_depth_alpha_changes_measured=True,
        geometry_or_opacity_equality_across_old_and_new_claimed=False,
        deleted_a37_original_RGB_or_depth_not_hidden_road_truth=True,
        deleted_road_measurement_reference='Original road_edit_rgb and road_edit_depth_m fixed mask',
        HELD_only_final_evaluation=True,adoption_decision_or_parameter_selection=False,
        colour_optimization_or_model_training_steps=0,group_conservation_maxabs=conservation,
        seconds=time.time()-start,runtime=rt.meta))
    print(json.dumps(dict(status='PASS',output=str(dest),seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
