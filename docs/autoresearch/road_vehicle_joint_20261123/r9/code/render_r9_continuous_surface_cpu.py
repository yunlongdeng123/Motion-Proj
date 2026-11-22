"""Same-domain CPU surface previews for continuous v2; original pixels retained."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from audit_r9_shared_field_cpu import load,read,write,pin,gather,all4,any4,srgb,png

WORK=Path(__file__).resolve().parents[2];R9=WORK/'old_raw_teacher_joint_20261123_r9';R7=WORK/'road_shared_appearance_20261122_r7'
OUT=R9/'cpu_review/joint_shared_continuous_v2';OUT.mkdir(parents=True,exist_ok=True)
field_path=R9/'joint_shared_continuous_v2/final_atlas.npz';f=load(field_path);seed=load(R7/'quilt_seed/atlas_seed.npz')
rgb=f['atlas_linear'];base=seed['atlas_seed_linear'];kind=f['source_kind'];permission=f['permission_mask'];material=permission&(kind>0)
gains={int(x['raw']):float(x['gain']) for x in read(R7/'exposure_gain.json')['records']}
colours=np.array([[.12,.15,.18],[.12,.85,.85],[.77,.50,.18],[.68,.35,.95],[.90,.38,.62]])
display_kind=kind.copy();display_kind[(kind==3)&f['teacher_inferred_field']]=4
panels=[];v1panels=[];rows=[];rim=None
for frame in [86,110,134,179,198]:
    cp=R7/f'projection_cache_v1/{frame:03d}/view_cache.npz';c=load(cp);uv=c['atlas_uv_map'];geom=c['geometry_valid']&c['projection_valid']
    projected,bounds,idx=gather(rgb,uv);initial,_,_=gather(base,uv)
    permitted=all4(permission,idx);anyperm=any4(permission,idx);available=all4((kind>0)&np.isfinite(rgb).all(-1),idx)
    evaluated=geom&bounds&permitted&available
    v1=load(R9/f'cpu_review/joint_shared_v1/{frame:03d}_domain_masks.npz')
    assert np.array_equal(evaluated,v1['evaluated']),f'{frame}: material denominator changed'
    categories={'unsupported_geometry':~geom,'UV_unsupported':geom&~bounds,
        'outside_permission':geom&bounds&~anyperm,'permission_border':geom&bounds&anyperm&~permitted,
        'authorized_unknown':geom&bounds&permitted&~available}
    masks={'geometry':geom,'evaluated':evaluated,**categories};np.savez_compressed(OUT/f'{frame:03d}_domain_masks.npz',**masks)
    yy,xx=np.indices(geom.shape);hatch=((xx+yy)%8)<3
    final_image=np.full((*geom.shape,3),[.0627,.0902,.1333]);seed_image=final_image.copy()
    final_image[evaluated]=srgb(projected[evaluated]/gains[frame]);seed_image[evaluated]=srgb(initial[evaluated]/gains[frame])
    png(OUT/f'{frame:03d}_material_only.png',final_image);png(OUT/f'{frame:03d}_seed_material_only.png',seed_image)
    final_hatch=final_image.copy();seed_hatch=seed_image.copy()
    for name,col in [('unsupported_geometry',[.28,.30,.33]),('UV_unsupported',[.8,.1,.7]),('outside_permission',[.36,.34,.24]),('permission_border',[.95,.6,.05]),('authorized_unknown',[1,.1,.15])]:
        final_hatch[categories[name]&hatch]=col;seed_hatch[categories[name]&hatch]=col
    png(OUT/f'{frame:03d}_material_hatched.png',final_hatch);png(OUT/f'{frame:03d}_seed_material_hatched.png',seed_hatch)
    source,_,_=gather(colours[display_kind],uv);source_image=np.full_like(final_image,[.0627,.0902,.1333]);source_image[evaluated]=source[evaluated]
    png(OUT/f'{frame:03d}_source_kind.png',source_image)
    context=c['photo_pinhole_srgb'].copy();context[evaluated]=final_image[evaluated];png(OUT/f'{frame:03d}_optional_photo_context.png',context)
    row={'raw':frame,'projection_cache':pin(cp),'geometry_fixed_denominator':int(geom.sum()),'evaluated_material':int(evaluated.sum()),
        'same_evaluated_domain_bitexact_v1':True,'counts':{k:int(v.sum()) for k,v in categories.items()},
        'old_CS_W_used':False,'photo_RGB_used_as_material':False,'exposure_gain':gains[frame]}
    write(OUT/f'{frame:03d}_scope.json',row);rows.append(row)
    crop=(0,400,1152,768)
    final_crop=Image.open(OUT/f'{frame:03d}_material_hatched.png').crop(crop).resize((864,276))
    for kind_label,left_path,container in [('SEED',OUT/f'{frame:03d}_seed_material_hatched.png',panels),('V1 FAILED CONTROL',R9/f'cpu_review/joint_shared_v1/{frame:03d}_material_hatched.png',v1panels)]:
        panel=Image.new('RGB',(1728,306),'#111827');draw=ImageDraw.Draw(panel)
        draw.text((8,8),f'{frame:03d}  {kind_label}  |  same actual-surface material domain',fill='white')
        draw.text((872,8),f'{frame:03d}  CONTINUOUS V2  |  fixed display crop only',fill='white')
        panel.paste(Image.open(left_path).crop(crop).resize((864,276)),(0,30));panel.paste(final_crop,(864,30));container.append(panel)
    if frame==110:
        mp=R7/'cpu_review/black_rim_domain_diagnostic_v1/fixed_primary1792_masks.npz';fixed=load(mp)['rim1792'];assert fixed.sum()==1792
        parts={'covered_new_field_RGB':fixed&evaluated,**{k:fixed&v for k,v in categories.items()}}
        union=np.zeros_like(fixed)
        for v in parts.values():assert not (v&union).any();union|=v
        assert np.array_equal(union,fixed)
        np.savez_compressed(OUT/'fixed1792_classification.npz',fixed1792=fixed,**parts)
        rim={'fixed_mask':pin(mp),'fixed_denominator':1792,'counts':{k:int(v.sum()) for k,v in parts.items()},'partition_complete':True,
            'covered_dark_encoded_max_below40_255':int((parts['covered_new_field_RGB']&(final_image.max(-1)<40/255)).sum()),
            'geometry_unsupported_not_teacher_black':True,'photo_RGB_not_used_to_fill_new_material':True}
for filename,collection in [('five_view_seed_final_material_contactsheet.png',panels),('five_view_v1_v2_material_contactsheet.png',v1panels)]:
    sheet=Image.new('RGB',(1728,sum(p.height for p in collection)));top=0
    for p in collection:sheet.paste(p,(0,top));top+=p.height
    sheet.save(OUT/filename)
atlas=[]
for name,x in [('SEED',base),('CONTINUOUS V2',rgb)]:
    display=srgb(x);display[~material]=[.06,.09,.13];im=Image.fromarray(np.rint(display*255).astype(np.uint8))
    panel=Image.new('RGB',(im.width,im.height+24),'#111827');ImageDraw.Draw(panel).text((8,5),name,fill='white');panel.paste(im,(0,24));atlas.append(panel)
sheet=Image.new('RGB',(atlas[0].width,sum(p.height for p in atlas)));top=0
for p in atlas:sheet.paste(p,(0,top));top+=p.height
sheet.save(OUT/'world_atlas_seed_final.png');png(OUT/'world_source_kind_observed_inferred.png',colours[display_kind]);png(OUT/'world_teacher_influence.png',f['effective_teacher_blend'])
receipt={'schema':'R9.continuous_v2_independent_surface_views.v1','complete':True,'final_atlas':pin(field_path),'frames':[86,110,134,179,198],
    'views':rows,'fixed1792':rim,'contactsheets':[pin(OUT/n) for n in ['five_view_seed_final_material_contactsheet.png','five_view_v1_v2_material_contactsheet.png']],
    'source_display_legend':{'1':'observed real','2':'similar real-photo inferred','3':'teacher direct observations + seed detail','4':'teacher inferred continuous influence + seed detail'},
    'acceptance':'PENDING independent full QA and visual inspection','CPU_only':True,'GPU_forward':0,'diffusion_calls':0,'Gaussian_updates':0}
write(OUT/'surface_views_receipt.json',receipt);print(receipt)
