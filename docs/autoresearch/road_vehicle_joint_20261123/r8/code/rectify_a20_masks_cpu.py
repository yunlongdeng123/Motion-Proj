"""Explicit-domain categorical mask readout; CPU only, originals unchanged."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
WS=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-22\065-file-c-users-dengyunlong-documents-2')
OLD=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian')
FRAMES=(86,110,134,179,198)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def raster(path):
    with Image.open(path) as im:value=np.asarray(im).copy()
    if value.ndim!=2:raise ValueError('Expected categorical/grayscale mask, not RGB colours: '+str(path))
    return value

def rectify_categorical_to_pinhole(source,cache,*,source_domain,source_native_wh=(1920,1280),fill=0):
    """One nearest-neighbour lookup from declared source pixel-centre domain.

    Brown UV is the cached original-photo pixel INDEX, with raw pixel centres
    at integer indices. For a lower-resolution Brown mask, nearest index is
    floor((uv_raw+0.5)*mask_size/native_size). Native-pinhole masks never pass
    through Brown mapping. The returned validity also requires valid RGB.
    """
    source=np.asarray(source);valid=np.asarray(cache['photo_pinhole_valid'],bool)
    h,w=valid.shape
    if source_domain=='native_pinhole':
        result=source.copy() if source.shape==(h,w) else np.asarray(Image.fromarray(source).resize((w,h),Image.Resampling.NEAREST)).copy()
        result[~valid]=fill;return result,valid.copy()
    if source_domain!='brown':raise ValueError('Explicit source_domain must be brown or native_pinhole')
    uv=np.asarray(cache['photo_source_brown_uv']);nw,nh=source_native_wh;sh,sw=source.shape
    finite=np.isfinite(uv).all(-1);safe=np.where(finite[...,None],uv,0.)
    x=np.floor((safe[...,0]+.5)*sw/nw).astype(np.int64);y=np.floor((safe[...,1]+.5)*sh/nh).astype(np.int64)
    good=valid&finite&(x>=0)&(x<sw)&(y>=0)&(y<sh)
    result=np.full((h,w),fill,dtype=source.dtype);result[good]=source[y[good],x[good]]
    return result,good

def full_brown_crop(folder,field,mapping,*,outside=0):
    width,height=mapping['image_native_size'];x0,y0,x1,y1=mapping['crop_native_xyxy_half_open']
    crop=raster(folder/field)>0
    if crop.shape!=(y1-y0,x1-x0):raise ValueError('Brown crop dimensions disagree with source mapping')
    full=np.full((height,width),outside,np.uint8);full[y0:y1,x0:x1]=crop.astype(np.uint8)
    return full

def export_masks(source_mask_root,conditioning_root,cache_root,output_root):
    source_mask_root=Path(source_mask_root);conditioning_root=Path(conditioning_root);cache_root=Path(cache_root);output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for raw in FRAMES:
        stem=f'{raw:03d}';sources=source_mask_root/stem;folder=output_root/stem;folder.mkdir(exist_ok=True)
        cache_path=cache_root/stem/'view_cache.npz'
        with np.load(cache_path,allow_pickle=False) as z:cache={k:z[k].copy() for k in ['photo_source_brown_uv','photo_pinhole_valid','camera_K']}
        labels_path=sources/'labels.png';labels=raster(labels_path)
        pinhole_labels,valid=rectify_categorical_to_pinhole(labels,cache,source_domain='brown',fill=255)
        protected=valid&(pinhole_labels>0)&(pinhole_labels!=21)&(pinhole_labels!=10)&(pinhole_labels!=255)
        target=raster(sources/'a20.png');target_rectified,_=rectify_categorical_to_pinhole((target>0).astype(np.uint8),cache,source_domain='brown')
        actor_proxy=(target_rectified>0)&(pinhole_labels==21)&valid
        legacy_labels=np.asarray(Image.fromarray(labels).resize((1152,768),Image.Resampling.NEAREST));legacy_K=(legacy_labels>0)&(legacy_labels!=21)&(legacy_labels!=10)&(legacy_labels!=255)
        values=dict(labels=pinhole_labels,protected=protected,actor_visible_proxy=actor_proxy,RGB_valid=valid)
        record=dict(raw=raw,source_labels=dict(path=str(labels_path),sha256=sha(labels_path),size_wh=[labels.shape[1],labels.shape[0]],coordinate_domain='Brown raw-photo domain, original approximate repaired instance annotation'),
            source_target=dict(path=str(sources/'a20.png'),sha256=sha(sources/'a20.png'),coordinate_domain='Brown raw-photo domain'),
            cache=dict(path=str(cache_path),sha256=sha(cache_path)),output_size_wh=[1152,768],camera_K=cache['camera_K'].tolist(),
            target_proxy_pixels=int(actor_proxy.sum()),protected_pixels=int(protected.sum()),legacy_K_pixels=int(legacy_K.sum()),
            K_changed_pixels_in_valid_RGB_domain=int(((protected^legacy_K)&valid).sum()),fixed_conservative_V_available=False,known_background_available=False,
            actor_proxy_is_conservative_legacy_V=False,original_masks_modified=False)
        source=conditioning_root/stem
        if raw in (86,110):
            mapping=read(source/'mapping.json')
            if 'Brown' not in mapping['mask_contract']['potential']:raise ValueError('Expected explicit Brown crop provenance')
            for name,file,outside in [('actor_visible','visible_mask.png',0),('unknown_potential','unknown_potential_mask.png',0),('known_background','known_background_mask.png',1)]:
                brown=full_brown_crop(source,file,mapping,outside=outside)
                corrected,_=rectify_categorical_to_pinhole(brown,cache,source_domain='brown')
                values[name]=(corrected>0)&valid
                legacy=np.asarray(Image.fromarray(brown).resize((1152,768),Image.Resampling.NEAREST))>0
                record[name+'_source']=dict(path=str(source/file),sha256=sha(source/file),coordinate_domain='Brown 1920x1280 crop restored at crop_native_xyxy_half_open',crop_xyxy=mapping['crop_native_xyxy_half_open'])
                record[name+'_legacy_pixels']=int(legacy.sum());record[name+'_corrected_pixels_before_exclusion']=int(values[name].sum());record[name+'_changed_in_valid_RGB']=int(((values[name]^legacy)&valid).sum())
            if (values['actor_visible']&protected).any() or (values['actor_visible']&(pinhole_labels==10)).any():raise ValueError('Rectified fixed V overlaps protected or deleted-occluder labels')
            values['known_background']&=~protected&~values['actor_visible']&~values['unknown_potential']
            record.update(fixed_conservative_V_available=True,known_background_available=True,actor_visible_pixels=int(values['actor_visible'].sum()),known_background_pixels=int(values['known_background'].sum()),V_K_intersection=0,V_a9_intersection=0)
        np.savez_compressed(folder/'rectified_masks.npz',**values)
        files={}
        for key,array in values.items():
            path=folder/(key+'.png');Image.fromarray(array.astype(np.uint8)*(255 if array.dtype==np.bool_ else 1)).save(path);files[key]=dict(path=str(path),sha256=sha(path),pixels=int(array.sum()) if array.dtype==np.bool_ else None)
        record['outputs']=files;record['NPZ']=dict(path=str(folder/'rectified_masks.npz'),sha256=sha(folder/'rectified_masks.npz'))
        (folder/'receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');rows.append(record)
    receipt=dict(schema='R8.A20_mask_coordinate_rectification.v1',complete=True,CPU_only=True,GPU_calls=0,parameter_updates=0,frames=list(FRAMES),rows=rows,
        rules={'Brown_labels':'Use photo_source_brown_uv once, nearest categorical lookup with explicit half-pixel size mapping; preserve 255 invalid separately',
          'Brown_V_U_B':'Restore exact native Brown crop first, then same nearest Brown inverse map as RGB; known B excludes new K/U/V',
          'native_pinhole_raster_alpha':'Already in native pinhole domain; no Brown warp',
          'native_3D_K_or_a9_row_sidecars':'Global row identities, no image resampling',
          'R7_world_xyz_geometry_valid_camera_K':'Already native pinhole geometry coordinates; no Brown warp'},
        fixed_V_frames=[86,110],other_actor_visible_outputs='Approximate repaired-instance proxy only; not the sealed conservative conditioning V',
        old_metrics_rewritten=False,old_files_modified=False)
    (output_root/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-mask-root',type=Path,default=HERE/'brown_sources');p.add_argument('--conditioning-root',type=Path,default=OLD/'work/a20_sd15_sameview_20261122/audit/reference/conditioning/train');p.add_argument('--cache-root',type=Path,default=WS/'work/road_shared_appearance_20261122_r7/projection_cache_v1');p.add_argument('--output-root',type=Path,default=HERE/'rectified_masks')
    a=p.parse_args();result=export_masks(a.source_mask_root,a.conditioning_root,a.cache_root,a.output_root)
    print(json.dumps(dict(complete=True,CPU_only=True,receipt=str(a.output_root/'receipt.json'),rows=[{k:r.get(k) for k in ['raw','protected_pixels','target_proxy_pixels','actor_visible_pixels','K_changed_pixels_in_valid_RGB_domain']} for r in result['rows']]),ensure_ascii=False))
